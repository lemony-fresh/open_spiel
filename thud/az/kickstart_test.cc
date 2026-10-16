// Copyright 2026 The Thud-on-OpenSpiel authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests kickstarting in our copy of the trainer (thud/PLAN.md, kickstarting): the
// teacher's scores (AlphaBetaPlayer::Scores), its policy (TeacherPolicy) and the mixed
// policy targets PlayGame records — with a stand-in network (a uniform prior, every
// value 0), so that no model is needed:
//
//   1. Scores: a troll next to a lone dwarf, trolls to move — at depth 1 every capture
//      scores one point above every other move; a lone dwarf that could step next to
//      a troll, dwarfs to move — at depth 2 those steps score one point below the
//      rest, and (the control that the position needs the reply) at depth 1 all moves
//      score alike.
//   2. TeacherPolicy sums to 1, follows the scores' order, and a move one point worse
//      gets exp(-1 / temperature) of the weight.
//   3. PlayGame: weight 0 records exactly the targets without a teacher; with weight w
//      the same moves are played (the teacher does not choose) and every recorded
//      target is (1 - w) x the target without + w x TeacherPolicy of its position —
//      the control: they differ. Per player (approach D1 of
//      thud/DESIGN_dwarf_signal.md): with the weights 0.7 for the dwarfs and the
//      single weight 0.3 left for the trolls, the same moves again, each dwarf
//      position's target 0.3 x the visits + 0.7 x the teacher's and each troll
//      position's as with the single weight; an empty list changes nothing.
//   4. The configuration: the settings round-trip; an older config.json loads with
//      weight 0 and no per-player weights; KickstartWeights fills missing entries
//      from the single weight.
//   5. Cost (printed): Scores at depths 2 and 3 on self-play-like positions.
//
//   thud/az/build.sh thud/az/kickstart_test.cc
//   build-shared/kickstart_test

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/logger.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/mcts.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::torch_az::AlphaZeroConfig;
using open_spiel::thud_az::torch_az::PlayGame;
using open_spiel::thud_az::torch_az::TeacherPolicy;
using open_spiel::thud_az::torch_az::Trajectory;

int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  failures += !ok;
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

Trajectory Play(const open_spiel::Game& game, const AlphaBetaPlayer* teacher, double weight,
                const std::vector<double>& weights = {}) {
  auto evaluator = std::make_shared<StandIn>();
  std::vector<std::unique_ptr<MCTSBot>> bots;
  for (int p = 0; p < 2; ++p) {
    bots.push_back(std::make_unique<MCTSBot>(
        game, evaluator, 2, 8, 10, false, 3, false,
        open_spiel::thud_az::ChildSelectionPolicy::PUCT));
  }
  std::mt19937 rng(11);
  open_spiel::NoopLogger logger;
  return PlayGame(&logger, 1, game, &bots, &rng, /*temperature=*/1, /*temperature_drop=*/10,
                  /*cutoff_value=*/2, false, nullptr, 0, 2, nullptr,
                  open_spiel::kInvalidPlayer, teacher, weight, /*teacher_temperature=*/1,
                  weights);
}

const char* kRows =
    "-----.....-----\n----.......----\n---.........---\n--...........--\n"
    "-.............-\n...............\n%s\n.......OT......\n"
    "...............\n...............\n-.............-\n--...........--\n"
    "---.........---\n----.......----\n-----.....-----\n";

std::string Position(const std::string& row6, const std::string& to_move) {
  char text[512];
  std::snprintf(text, sizeof(text), kRows, row6.c_str());
  return std::string(text) + "to_move=" + to_move;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  AlphaBetaPlayer depth1(*game, 1), depth2(*game, 2), depth3(*game, 3);

  // 1. Scores.
  {
    // A lone dwarf at (6,7), next to the troll at (7,8) (and a second dwarf far away,
    // so the battle goes on after the capture), trolls to move.
    std::unique_ptr<State> s =
        game->NewInitialState(Position(".......d......d", "trolls"));
    const std::vector<Action> legal = s->LegalActions();
    const std::vector<double> scores = depth1.Scores(*s);
    double capture = -1e9, other = 1e9, other_max = -1e9;
    for (int i = 0; i < legal.size(); ++i) {
      if (absl::EndsWith(s->ActionToString(legal[i]), "x")) {
        capture = std::max(capture, scores[i]);
      } else {
        other = std::min(other, scores[i]);
        other_max = std::max(other_max, scores[i]);
      }
    }
    Check(std::abs(capture - other_max - 1) < 1e-9 && std::abs(other_max - other) < 1e-9,
          "1. depth 1, trolls: a capture scores one point above every other move");
    // A lone dwarf at (6,1), dwarfs to move (its capture ends the battle: the score is
    // then the result, still one point below).
    std::unique_ptr<State> d = game->NewInitialState(Position(".d.............", "dwarfs"));
    const std::vector<Action> moves = d->LegalActions();
    const std::vector<double> s1 = depth1.Scores(*d), s2 = depth2.Scores(*d);
    int next_to_troll = 0;
    bool depth1_flat = true;
    for (int i = 0; i < moves.size(); ++i) {
      const std::string m = d->ActionToString(moves[i]);
      const bool unsafe = m == "(6,1)-(6,7)" || m == "(6,1)-(6,8)" || m == "(6,1)-(6,9)";
      next_to_troll += unsafe;
      depth1_flat = depth1_flat && std::abs(s1[i] - s1[0]) < 1e-9;
    }
    // Depth 2: the unsafe steps one point below the best of the rest.
    double safe_best = -1e9, unsafe_best = -1e9;
    for (int i = 0; i < moves.size(); ++i) {
      const std::string m = d->ActionToString(moves[i]);
      const bool unsafe = m == "(6,1)-(6,7)" || m == "(6,1)-(6,8)" || m == "(6,1)-(6,9)";
      (unsafe ? unsafe_best : safe_best) = std::max(unsafe ? unsafe_best : safe_best, s2[i]);
    }
    Check(next_to_troll == 3 && std::abs(safe_best - unsafe_best - 1) < 1e-9,
          "1. depth 2, dwarfs: the three steps next to the troll score one point below the "
          "best of the rest");
    Check(depth1_flat, "1. control: at depth 1 every dwarf move scores alike (it needs the reply)");
  }

  // 2. TeacherPolicy.
  {
    std::unique_ptr<State> d = game->NewInitialState(Position(".d.............", "dwarfs"));
    const open_spiel::ActionsAndProbs t = TeacherPolicy(*d, depth2, /*temperature=*/0.5);
    const std::vector<double> s = depth2.Scores(*d);
    double sum = 0, safe = 0, unsafe = 0;
    for (int i = 0; i < t.size(); ++i) {
      sum += t[i].second;
      const std::string m = d->ActionToString(t[i].first);
      if (m == "(6,1)-(6,7)") unsafe = t[i].second;
      if (m == "(6,1)-(6,2)") safe = t[i].second;
    }
    Check(std::abs(sum - 1) < 1e-9 && std::abs(unsafe / safe - std::exp(-1 / 0.5)) < 1e-9,
          "2. TeacherPolicy sums to 1; a move one point worse gets exp(-1/T) of the weight");
  }

  // 3. PlayGame.
  {
    const Trajectory none = Play(*game, nullptr, 0);
    const Trajectory zero = Play(*game, &depth2, 0);
    const double w = 0.3;
    const Trajectory mixed = Play(*game, &depth2, w);
    bool zero_same = none.states.size() == zero.states.size();
    for (int i = 0; zero_same && i < none.states.size(); ++i) {
      zero_same = none.states[i].policy == zero.states[i].policy;
    }
    bool same_moves = none.states.size() == mixed.states.size();
    for (int i = 0; same_moves && i < none.states.size(); ++i) {
      same_moves = none.states[i].action == mixed.states[i].action;
    }
    bool mix_exact = same_moves, differ = false;
    std::unique_ptr<State> state = game->NewInitialState();
    for (int i = 0; mix_exact && i < none.states.size(); ++i) {
      std::map<Action, double> taught;
      for (const auto& [a, p] : TeacherPolicy(*state, depth2, 1)) taught[a] = p;
      for (int j = 0; j < none.states[i].policy.size(); ++j) {
        const auto& [a, v] = none.states[i].policy[j];
        const double expected = (1 - w) * v + w * taught[a];
        mix_exact = mix_exact && mixed.states[i].policy[j].first == a &&
                    std::abs(mixed.states[i].policy[j].second - expected) < 1e-12;
        differ = differ || std::abs(mixed.states[i].policy[j].second - v) > 1e-9;
      }
      state->ApplyAction(none.states[i].action);
    }
    std::cout << "  a game of " << none.states.size() << " moves" << std::endl;
    Check(zero_same, "3. weight 0 records exactly the targets without a teacher");
    Check(same_moves, "3. with weight 0.3 the same moves are played");
    Check(mix_exact && differ,
          "3. every target is 0.7 x the visits + 0.3 x TeacherPolicy (control: they differ)");
    // Per player: the dwarfs 0.7, the trolls the single weight.
    const Trajectory empty_list = Play(*game, &depth2, w, {});
    const Trajectory per_player = Play(*game, &depth2, w, {0.7});
    bool empty_same = mixed.states.size() == empty_list.states.size();
    for (int i = 0; empty_same && i < mixed.states.size(); ++i) {
      empty_same = mixed.states[i].policy == empty_list.states[i].policy;
    }
    bool per_moves = none.states.size() == per_player.states.size();
    for (int i = 0; per_moves && i < none.states.size(); ++i) {
      per_moves = none.states[i].action == per_player.states[i].action;
    }
    bool dwarfs_exact = per_moves, trolls_as_single = per_moves, dwarfs_differ = false;
    int dwarf_positions = 0, troll_positions = 0;
    state = game->NewInitialState();
    for (int i = 0; per_moves && i < none.states.size(); ++i) {
      const bool dwarfs = state->CurrentPlayer() == open_spiel::thud::kDwarfPlayer;
      ++(dwarfs ? dwarf_positions : troll_positions);
      std::map<Action, double> taught;
      for (const auto& [a, p] : TeacherPolicy(*state, depth2, 1)) taught[a] = p;
      for (int j = 0; j < none.states[i].policy.size(); ++j) {
        const auto& [a, v] = none.states[i].policy[j];
        const double got = per_player.states[i].policy[j].second;
        if (dwarfs) {
          dwarfs_exact = dwarfs_exact && std::abs(got - (0.3 * v + 0.7 * taught[a])) < 1e-12;
          dwarfs_differ = dwarfs_differ ||
                          std::abs(got - mixed.states[i].policy[j].second) > 1e-9;
        } else {
          trolls_as_single = trolls_as_single && got == mixed.states[i].policy[j].second;
        }
      }
      state->ApplyAction(none.states[i].action);
    }
    Check(empty_same, "3. an empty list of weights per player changes nothing");
    Check(per_moves && dwarf_positions > 0 && troll_positions > 0,
          "3. with the dwarfs' weight 0.7 the same moves are played");
    Check(dwarfs_exact && dwarfs_differ,
          "3. every dwarf position's target is 0.3 x the visits + 0.7 x TeacherPolicy "
          "(control: it differs from the single weight's)");
    Check(trolls_as_single, "3. every troll position's target is the single weight's");
  }

  // 4. The configuration.
  {
    AlphaZeroConfig c;
    open_spiel::json::Object j = {
        {"game", std::string("thud")}, {"path", std::string()},
        {"graph_def", std::string()}, {"nn_model", std::string("resnet")},
        {"nn_width", 1}, {"nn_depth", 1}, {"devices", std::string()},
        {"explicit_learning", false},
        {"learning_rate", 0.0}, {"weight_decay", 0.0}, {"train_batch_size", 1},
        {"inference_batch_size", 1}, {"inference_threads", 1}, {"inference_cache", 1},
        {"replay_buffer_size", 1}, {"replay_buffer_reuse", 1}, {"checkpoint_freq", 1},
        {"evaluation_window", 1}, {"uct_c", 2.0}, {"max_simulations", 1},
        {"policy_alpha", 0.0}, {"policy_epsilon", 0.0}, {"temperature", 1.0},
        {"temperature_drop", 10.0}, {"cutoff_probability", 0.0}, {"cutoff_value", 0.0},
        {"actors", 1}, {"evaluators", 1}, {"eval_levels", 1}, {"max_steps", 0}};
    c.FromJson(j);
    const bool old_ok = c.kickstart_weight == 0 && c.kickstart_depth == 2 &&
                        c.kickstart_temperature == 1 && c.player_kickstart_weight.empty();
    c.kickstart_weight = 0.3;
    c.kickstart_depth = 3;
    c.kickstart_temperature = 0.5;
    c.player_kickstart_weight = "0.7";
    AlphaZeroConfig back;
    back.FromJson(c.ToJson());
    Check(old_ok && back.kickstart_weight == 0.3 && back.kickstart_depth == 3 &&
              back.kickstart_temperature == 0.5 && back.player_kickstart_weight == "0.7",
          "4. the settings round-trip; an older config.json loads with weight 0, depth 2, "
          "temperature 1, no weights per player");
    const std::vector<double> filled = open_spiel::thud_az::torch_az::KickstartWeights(back);
    back.player_kickstart_weight = "";
    const std::vector<double> single = open_spiel::thud_az::torch_az::KickstartWeights(back);
    back.player_kickstart_weight = "0.3,0.6";
    const std::vector<double> both = open_spiel::thud_az::torch_az::KickstartWeights(back);
    Check(filled == std::vector<double>({0.7, 0.3}) &&
              single == std::vector<double>({0.3, 0.3}) &&
              both == std::vector<double>({0.3, 0.6}),
          "4. KickstartWeights: \"0.7\" gives the dwarfs 0.7 and the trolls the single "
          "weight; empty gives both the single weight; \"0.3,0.6\" sets both");
  }

  // 5. Cost.
  {
    std::mt19937 rng(5);
    std::vector<std::unique_ptr<State>> positions;
    for (int g = 0; positions.size() < 40; ++g) {
      std::unique_ptr<State> s = game->NewInitialState();
      for (int ply = 0; !s->IsTerminal() && ply < 120; ++ply) {
        if (ply % 12 == 6 || ply % 12 == 7) positions.push_back(s->Clone());  // Both sides.
        const std::vector<Action> legal = s->LegalActions();
        s->ApplyAction(legal[rng() % legal.size()]);
      }
    }
    for (const AlphaBetaPlayer* t : {&depth2, &depth3}) {
      double seconds[2] = {0, 0};
      int count[2] = {0, 0};
      for (const auto& s : positions) {
        const auto start = std::chrono::steady_clock::now();
        t->Scores(*s);
        seconds[s->CurrentPlayer()] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ++count[s->CurrentPlayer()];
      }
      std::cout << "  cost at depth " << t->depth() << ": dwarfs "
                << 1000 * seconds[0] / count[0] << " ms a position (" << count[0]
                << "), trolls " << 1000 * seconds[1] / count[1] << " ms (" << count[1] << ")"
                << std::endl;
    }
  }

  std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
  return failures ? 1 : 0;
}
