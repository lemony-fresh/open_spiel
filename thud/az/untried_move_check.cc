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

// Checks the value our copy of OpenSpiel's MCTS gives moves it has not tried yet
// (UntriedMoveValue in thud/az/mcts.h; thud/PLAN.md Phase 6, roadmap stage 3a).
//
//   1. The formula, on a hand-built node: each rule gives the value it should, and an
//      unvisited child's PUCT value uses it; the old PUCTValue equals the new one with
//      0, upstream's rule.
//   2. The behaviour, on self-play positions: games played by a trained network (upstream's
//      rule, 100 simulations, after a few random opening moves), every 7th position
//      searched again under each rule; reports per side how many moves a search visits
//      and the most visited move's share. Upstream's rule should spread the dwarfs'
//      searches over nearly every move (thud/PLAN.md: 99 of 100); the other rules should
//      not.
//
//   thud/az/build.sh thud/az/untried_move_check.cc
//   OMP_NUM_THREADS=1 build-shared/untried_move_check RUN_DIR STEP [games=6]

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az;
using open_spiel::Action;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

az::MCTSBot MakeBot(const open_spiel::Game& game,
                    std::shared_ptr<az::Evaluator> evaluator, int sims,
                    az::UntriedMoveValue rule) {
  return az::MCTSBot(game, evaluator, /*uct_c=*/2, sims, /*max_memory_mb=*/100,
                     /*solve=*/false, /*seed=*/0, /*verbose=*/false,
                     az::ChildSelectionPolicy::PUCT, 0, 0,
                     /*dont_return_chance_node=*/true, /*max_wall_clock_time=*/-1, rule,
                     /*untried_move_reduction=*/0.2);
}

void CheckFormula(const open_spiel::Game& game) {
  auto evaluator = std::make_shared<az::RandomRolloutEvaluator>(1, 0);
  // Three children: two visited (4 visits, total -2.0, prior 0.5; 1 visit, total 0.5,
  // prior 0.2), one not (prior 0.3). Siblings' mean (-2.0 + 0.5) / 5 = -0.3, their
  // prior mass 0.7.
  az::SearchNode node(open_spiel::kInvalidAction, 0, 1);
  node.explore_count = 5;
  node.children.emplace_back(1, 0, 0.5);
  node.children.emplace_back(2, 0, 0.2);
  node.children.emplace_back(3, 0, 0.3);
  node.children[0].explore_count = 4;
  node.children[0].total_reward = -2.0;
  node.children[1].explore_count = 1;
  node.children[1].total_reward = 0.5;
  const double expected = -0.3 - 0.2 * std::sqrt(0.7);
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-12; };
  auto value = [&](az::UntriedMoveValue rule) {
    return MakeBot(game, evaluator, 1, rule).UntriedValue(node);
  };
  Expect(value(az::UntriedMoveValue::kUpstream) == 0, "upstream: an untried move is 0");
  Expect(near(value(az::UntriedMoveValue::kSiblingMeanMinusReduction), expected),
         absl::StrFormat("siblings' mean minus reduction: %.6f (expected %.6f)",
                         value(az::UntriedMoveValue::kSiblingMeanMinusReduction),
                         expected));
  Expect(value(az::UntriedMoveValue::kLoss) == game.MinUtility(),
         absl::StrFormat("loss: %.1f", value(az::UntriedMoveValue::kLoss)));
  az::SearchNode fresh(open_spiel::kInvalidAction, 0, 1);
  fresh.children.emplace_back(1, 0, 0.5);
  fresh.children.emplace_back(2, 0, 0.5);
  Expect(MakeBot(game, evaluator, 1, az::UntriedMoveValue::kSiblingMeanMinusReduction)
                 .UntriedValue(fresh) == 0,
         "siblings' mean minus reduction with no sibling visited: 0");
  const az::SearchNode& untried = node.children[2];
  const double bonus = 2 * 0.3 * std::sqrt(5.0);
  Expect(near(untried.PUCTValue(5, 2, expected), expected + bonus),
         "an unvisited child's PUCT value uses the rule's value");
  Expect(untried.PUCTValue(5, 2) == untried.PUCTValue(5, 2, 0) &&
             node.children[0].PUCTValue(5, 2) == node.children[0].PUCTValue(5, 2, 0.7),
         "upstream's PUCTValue equals the new one with 0; visited children ignore it");
}

struct Breadth {
  std::vector<double> visited, top_share;
};

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

void CheckBehaviour(const open_spiel::Game& game, const std::string& dir, int step,
                    int games) {
  az::torch_az::DeviceManager devices;
  az::torch_az::VPNetModel model(game, dir, "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(step);
  devices.AddDevice(std::move(model));
  auto evaluator =
      std::make_shared<az::torch_az::VPNetEvaluator>(&devices, 1, 0, 1 << 18, 1);

  // Self-play positions: a few random opening moves, then the network with upstream's
  // rule; every 7th position.
  std::vector<std::unique_ptr<open_spiel::State>> positions;
  az::MCTSBot player = MakeBot(game, evaluator, 100, az::UntriedMoveValue::kUpstream);
  std::mt19937 rng(11);
  for (int g = 0; g < games; ++g) {
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    for (int move = 0; !state->IsTerminal(); ++move) {
      // Every 7th, an odd step, so both sides to move appear (every 8th gave only
      // dwarf positions: the dwarfs move on even moves).
      if (move % 7 == 0 && move >= 4) positions.push_back(state->Clone());
      if (move < 4) {
        std::vector<Action> legal = state->LegalActions();
        state->ApplyAction(
            legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
      } else {
        state->ApplyAction(player.MCTSearch(*state)->BestChild().action);
      }
    }
  }
  std::cout << "behaviour: " << positions.size() << " self-play positions from " << games
            << " games, 100 simulations a search" << std::endl;

  const std::vector<std::pair<std::string, az::UntriedMoveValue>> rules = {
      {"upstream", az::UntriedMoveValue::kUpstream},
      {"sibling_mean_minus_reduction", az::UntriedMoveValue::kSiblingMeanMinusReduction},
      {"loss", az::UntriedMoveValue::kLoss}};
  std::map<std::string, Breadth> dwarfs, trolls;
  for (const auto& [name, rule] : rules) {
    az::MCTSBot bot = MakeBot(game, evaluator, 100, rule);
    Breadth side[2];
    for (const auto& state : positions) {
      std::unique_ptr<az::SearchNode> root = bot.MCTSearch(*state);
      int visited = 0, top = 0;
      for (const az::SearchNode& child : root->children) {
        visited += child.explore_count > 0;
        top = std::max(top, child.explore_count);
      }
      Breadth& b = side[state->CurrentPlayer()];
      b.visited.push_back(visited);
      b.top_share.push_back(static_cast<double>(top) / root->explore_count);
    }
    std::cout << absl::StrFormat(
                     "  %-29s dwarfs: median %3.0f moves visited, most visited %4.2f; "
                     "trolls: %3.0f, %4.2f",
                     name, Median(side[0].visited), Median(side[0].top_share),
                     Median(side[1].visited), Median(side[1].top_share))
              << std::endl;
    dwarfs[name] = side[0];
    trolls[name] = side[1];
  }
  Expect(!dwarfs["upstream"].visited.empty() && !trolls["upstream"].visited.empty(),
         absl::StrFormat("control: both sides sampled (%d dwarf, %d troll positions)",
                         dwarfs["upstream"].visited.size(), trolls["upstream"].visited.size()));
  Expect(Median(dwarfs["upstream"].visited) >= 90,
         "control: upstream's rule spreads the dwarfs' searches (90+ of 100 moves)");
  Expect(Median(dwarfs["sibling_mean_minus_reduction"].visited) <
             Median(dwarfs["upstream"].visited) / 2,
         "siblings' mean minus reduction at least halves the dwarfs' moves visited");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " RUN_DIR STEP [games=6]" << std::endl;
    return 1;
  }
  const int games = argc > 3 ? std::stoi(std::string(argv[3]).substr(6)) : 6;
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  CheckFormula(*game);
  CheckBehaviour(*game, argv[1], std::stoi(argv[2]), games);
  std::cout << (failures ? "UNTRIED MOVE CHECK FAILED" : "untried move check passed")
            << std::endl;
  return failures ? 1 : 0;
}
