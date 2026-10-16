// Copyright 2026 The Thud-on-OpenSpiel authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests playout cap randomisation in our copy of the trainer (thud/az/alpha_zero.h:
// playout_cap_full_share, playout_cap_fast_simulations, FastBots; KataGo, arXiv
// 1902.10565, section 3.1), with games played by PlayGame as the actors play them and a
// stand-in network (a uniform prior, every value 0):
//
//   1. Off changes nothing: PlayGame with fast bots but a full share of 1, and without
//      them, play the game of the old call (the same moves, the same recorded positions
//      and targets) — no random number drawn.
//   2. On (a quarter in full, the fast searches at 4 simulations, the full at 48): over
//      20 games, the positions recorded are about a quarter of the network's moves
//      (within 4 standard deviations of the binomial), and every one comes from a full
//      search: at dwarf positions with 48+ legal moves its target has more than 4 moves
//      (a search of 4 simulations visits at most 4). The control: off, every move is
//      recorded.
//   3. FastBots: none when the cap is off; when on, two bots that search 4 simulations
//      and add no root noise (the same position searched twice gives the same visits).
//
//   thud/run_tests.sh playout_cap_test   # builds and runs it
//   build-tests/bin/playout_cap_test

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/logger.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/mcts.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::ChildSelectionPolicy;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::torch_az::AlphaZeroConfig;
using open_spiel::thud_az::torch_az::FastBots;
using open_spiel::thud_az::torch_az::PlayGame;
using open_spiel::thud_az::torch_az::Trajectory;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

class StandIn : public open_spiel::thud_az::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override { return {0, 0}; }
  open_spiel::ActionsAndProbs Prior(const State& state) override {
    const std::vector<Action> legal = state.LegalActions();
    open_spiel::ActionsAndProbs prior;
    for (Action a : legal) prior.push_back({a, 1.0 / legal.size()});
    return prior;
  }
};

class Capture : public open_spiel::Logger {
 public:
  using open_spiel::Logger::Print;
  void Print(const std::string& str) override { lines.push_back(str); }
  std::vector<std::string> lines;
};

struct Played {
  Trajectory trajectory;
  std::vector<std::string> moves;
};

std::vector<std::unique_ptr<MCTSBot>> FullBots(const open_spiel::Game& game,
                                               std::shared_ptr<StandIn> eval, int seed) {
  std::vector<std::unique_ptr<MCTSBot>> bots;
  for (int p = 0; p < 2; ++p) {
    bots.push_back(std::make_unique<MCTSBot>(
        game, eval, /*uct_c=*/2, /*max_simulations=*/48, /*max_memory_mb=*/10,
        /*solve=*/false, seed, /*verbose=*/false, ChildSelectionPolicy::PUCT, 0.1, 0.25));
  }
  return bots;
}

// One game; `mode`: 0 the old call, 1 with fast bots at a full share of 1, 2 the cap on.
Played Play(const open_spiel::Game& game, const AlphaZeroConfig& on, int mode, int seed) {
  auto eval = std::make_shared<StandIn>();
  auto bots = FullBots(game, eval, seed);
  std::vector<std::unique_ptr<MCTSBot>> fast = FastBots(on, game, eval);
  std::mt19937 rng(seed);
  Capture logger;
  Played out;
  if (mode == 0) {
    out.trajectory = PlayGame(&logger, 1, game, &bots, &rng, 1, 10, 2, false, nullptr, 0, 2);
  } else {
    out.trajectory = PlayGame(&logger, 1, game, &bots, &rng, 1, 10, 2, false, nullptr, 0, 2,
                              nullptr, open_spiel::kInvalidPlayer, nullptr, 0, 1, {}, &fast,
                              mode == 1 ? 1.0 : on.playout_cap_full_share);
  }
  const std::string line = logger.lines.back();
  out.moves = absl::StrSplit(line.substr(line.find("Actions: ") + 9), ' ', absl::SkipEmpty());
  return out;
}

bool SameRecord(const Trajectory& a, const Trajectory& b) {
  if (a.states.size() != b.states.size()) return false;
  for (size_t i = 0; i < a.states.size(); ++i) {
    if (a.states[i].action != b.states[i].action || a.states[i].policy != b.states[i].policy ||
        a.states[i].value != b.states[i].value) {
      return false;
    }
  }
  return a.returns == b.returns;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  AlphaZeroConfig on;
  on.uct_c = 2;
  on.max_simulations = 48;
  on.policy_alpha = 0.1;
  on.policy_epsilon = 0.25;
  on.untried_move_value = "sibling_mean_minus_reduction";
  on.untried_move_reduction = 0.2;
  on.policy_target_pruning = true;  // Off in fast searches all the same.
  on.playout_cap_full_share = 0.25;
  on.playout_cap_fast_simulations = 4;

  // 1. Off changes nothing.
  bool same = true;
  for (int seed : {1, 2, 3}) {
    const Played old_call = Play(*game, on, 0, seed), share_one = Play(*game, on, 1, seed);
    same = same && old_call.moves == share_one.moves &&
           SameRecord(old_call.trajectory, share_one.trajectory);
  }
  Check(same, "1. off (a full share of 1): the old call's games, moves and records, 3 seeds");

  // 2. On.
  int64_t ours = 0, recorded = 0, wide = 0, wide_full = 0;
  for (int seed = 10; seed < 30; ++seed) {
    const Played p = Play(*game, on, 2, seed);
    ours += p.moves.size();  // Self-play: every move is the network's.
    recorded += p.trajectory.states.size();
    for (const auto& s : p.trajectory.states) {
      if (s.current_player == 0 && s.legal_actions.size() >= 48) {
        ++wide;
        int visited = 0;
        for (const auto& [a, q] : s.policy) visited += q > 0;
        wide_full += visited > 4;
      }
    }
  }
  const double share = static_cast<double>(recorded) / ours;
  const double sd = std::sqrt(0.25 * 0.75 / ours);
  Check(std::abs(share - 0.25) <= 4 * sd,
        absl::StrFormat("2. on, a quarter in full: %d of %d moves recorded (%.3f; 0.25 ± %.3f)",
                        recorded, ours, share, 4 * sd));
  Check(wide > 0 && wide_full == wide,
        absl::StrFormat("2. every recorded position a full search: %d of %d wide dwarf "
                        "positions' targets cover more than 4 moves",
                        wide_full, wide));
  {
    const Played off = Play(*game, on, 1, 10);
    Check(off.trajectory.states.size() == off.moves.size(),
          absl::StrFormat("2. control: off, every move recorded (%d of %d)",
                          off.trajectory.states.size(), off.moves.size()));
  }

  // 3. FastBots.
  {
    AlphaZeroConfig off = on;
    off.playout_cap_full_share = 1;
    auto eval = std::make_shared<StandIn>();
    Check(FastBots(off, *game, eval).empty(), "3. FastBots: none when the cap is off");
    auto fast = FastBots(on, *game, eval);
    std::unique_ptr<State> state = game->NewInitialState();
    auto r1 = fast[0]->MCTSearch(*state), r2 = fast[0]->MCTSearch(*state);
    int visited = 0;
    bool same_visits = r1->children.size() == r2->children.size();
    for (size_t i = 0; i < r1->children.size(); ++i) {
      visited += r1->children[i].explore_count > 0;
      same_visits = same_visits && r1->children[i].explore_count == r2->children[i].explore_count;
    }
    Check(fast.size() == 2 && r1->explore_count <= 5 && visited <= 4 && same_visits,
          absl::StrFormat("3. FastBots: 2 bots, a search of 4 simulations (root %d visits, %d "
                          "children visited), no noise (the same visits twice: %s)",
                          r1->explore_count, visited, same_visits ? "yes" : "no"));
  }

  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
