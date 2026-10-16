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

// Tests the games against alpha-beta in our copy of the trainer (thud/PLAN.md, How we
// evaluate networks: training against alpha-beta) — games played by PlayGame as the
// actors play them, with a stand-in network (a uniform prior, every value 0) so that
// no model is needed:
//
//   1. With alpha-beta as the trolls, the game records only the dwarfs' positions —
//      one per dwarf move, the moves the dwarfs played — and marks the trolls as the
//      opponent; the returns are the final position's.
//   2. Every troll move is the one AlphaBetaPlayer chooses there with the game's seed
//      (the game replayed from its logged moves). Control: with another seed some
//      troll move differs, so the test can fail.
//   3. With alpha-beta as the dwarfs, only the trolls' positions are recorded.
//   4. Control: self-play records both sides' positions and no opponent.
//   5. The actor's log line names the opponent and its side (which
//      thud/experiments/analyse_games.py reads); self-play's has none.
//   6. AlphaBetaPlayer takes an available capture at depth 1 (a troll next to a lone
//      dwarf) and, at depth 2, does not leave a dwarf next to a troll that can take
//      it — the two habits our networks lack; controls: the capture is not the only
//      legal move, and the losing dwarf move is legal.
//   7. The configuration: alphabeta_share and alphabeta_depth round-trip through
//      JSON, a config.json from before them loads with 0 and "2", and one holding
//      the depth as a number (before 2026-10-08) as that number.
//   8. The depth per side (AlphaBetaDepth): "2,1" is 2 as the dwarfs, 1 as the
//      trolls; one value is both.
//
//   thud/run_tests.sh alphabeta_training_test   # builds and runs it
//   build-tests/bin/alphabeta_training_test

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/mcts.h"

namespace {

using open_spiel::Action;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
using open_spiel::thud_az::ChildSelectionPolicy;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::torch_az::AlphaZeroConfig;
using open_spiel::thud_az::torch_az::PlayGame;
using open_spiel::thud_az::torch_az::Trajectory;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// A network that knows nothing: a uniform prior, every position even.
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

// Keeps what PlayGame logs.
class Capture : public open_spiel::Logger {
 public:
  using open_spiel::Logger::Print;
  void Print(const std::string& str) override { lines.push_back(str); }
  std::vector<std::string> lines;
};

struct Game {
  Trajectory trajectory;
  std::string line;                // The logged game line.
  std::vector<std::string> moves;  // Every move, as logged.
};

Game Play(const open_spiel::Game& game, const AlphaBetaPlayer* opponent, Player side,
          int seed) {
  auto evaluator = std::make_shared<StandIn>();
  std::vector<std::unique_ptr<MCTSBot>> bots;
  for (int p = 0; p < 2; ++p) {
    bots.push_back(std::make_unique<MCTSBot>(
        game, evaluator, /*uct_c=*/2, /*max_simulations=*/8, /*max_memory_mb=*/10,
        /*solve=*/false, /*seed=*/seed, /*verbose=*/false, ChildSelectionPolicy::PUCT));
  }
  std::mt19937 rng(seed);
  Capture logger;
  Game result;
  result.trajectory = PlayGame(&logger, 1, game, &bots, &rng, /*temperature=*/1,
                               /*temperature_drop=*/10, /*cutoff_value=*/2,
                               /*verbose=*/false, /*prior_source=*/nullptr,
                               /*pruning_k=*/0, /*uct_c=*/2, opponent, side);
  result.line = logger.lines.back();
  const std::string actions = result.line.substr(result.line.find("Actions: ") + 9);
  result.moves = absl::StrSplit(actions, ' ', absl::SkipEmpty());
  return result;
}

// Whether every move of `side` in `moves` is the one `player` chooses there.
bool OpponentMovesMatch(const open_spiel::Game& game, const std::vector<std::string>& moves,
                        Player side, const AlphaBetaPlayer& player) {
  std::unique_ptr<State> state = game.NewInitialState();
  for (const std::string& move : moves) {
    Action played = open_spiel::kInvalidAction;
    for (Action a : state->LegalActions()) {
      if (state->ActionToString(a) == move) played = a;
    }
    if (played == open_spiel::kInvalidAction) return false;
    if (state->CurrentPlayer() == side && player.Choose(*state) != played) return false;
    state->ApplyAction(played);
  }
  return true;
}

// The game's moves of `player`.
int MovesOf(const open_spiel::Game& game, const std::vector<std::string>& moves,
            Player player, std::vector<Action>* actions) {
  std::unique_ptr<State> state = game.NewInitialState();
  int count = 0;
  for (const std::string& move : moves) {
    for (Action a : state->LegalActions()) {
      if (state->ActionToString(a) != move) continue;
      if (state->CurrentPlayer() == player) {
        ++count;
        actions->push_back(a);
      }
      state->ApplyAction(a);
      break;
    }
  }
  return count;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  constexpr int kSeed = 7;

  // 1-2. Alpha-beta (depth 1, fast) as the trolls.
  AlphaBetaPlayer trolls(*game, /*depth=*/1);
  trolls.NewGame(kSeed);
  const Game vs_trolls = Play(*game, &trolls, /*side=*/1, kSeed);
  const Trajectory& t = vs_trolls.trajectory;
  bool only_dwarfs = !t.states.empty();
  for (const auto& s : t.states) only_dwarfs = only_dwarfs && s.current_player == 0;
  std::vector<Action> dwarf_actions;
  const int dwarf_moves = MovesOf(*game, vs_trolls.moves, 0, &dwarf_actions);
  bool same_actions = dwarf_actions.size() == t.states.size();
  for (int i = 0; same_actions && i < t.states.size(); ++i) {
    same_actions = t.states[i].action == dwarf_actions[i];
  }
  std::cout << "  a game of " << vs_trolls.moves.size() << " moves, " << t.states.size()
            << " positions recorded, ending " << t.ending << std::endl;
  Check(only_dwarfs && t.opponent == 1,
        "1. alpha-beta as the trolls: only the dwarfs' positions, the trolls the opponent");
  Check(dwarf_moves == t.states.size() && same_actions,
        "1. one position per dwarf move, with the move the dwarfs played");
  {
    std::unique_ptr<State> end = game->NewInitialState();
    for (const std::string& move : vs_trolls.moves) {
      for (Action a : end->LegalActions()) {
        if (end->ActionToString(a) == move) {
          end->ApplyAction(a);
          break;
        }
      }
    }
    Check(end->IsTerminal() && end->Returns() == t.returns,
          "1. the returns are the final position's");
  }
  Check(OpponentMovesMatch(*game, vs_trolls.moves, 1, trolls),
        "2. every troll move is AlphaBetaPlayer's choice with the game's seed");
  AlphaBetaPlayer other_seed(*game, 1);
  other_seed.NewGame(kSeed + 1);
  Check(!OpponentMovesMatch(*game, vs_trolls.moves, 1, other_seed),
        "2. control: with another seed some troll move differs");

  // 3. Alpha-beta as the dwarfs.
  AlphaBetaPlayer dwarfs(*game, 1);
  dwarfs.NewGame(kSeed);
  const Game vs_dwarfs = Play(*game, &dwarfs, /*side=*/0, kSeed);
  bool only_trolls = !vs_dwarfs.trajectory.states.empty();
  for (const auto& s : vs_dwarfs.trajectory.states) {
    only_trolls = only_trolls && s.current_player == 1;
  }
  std::vector<Action> troll_actions;
  Check(only_trolls && vs_dwarfs.trajectory.opponent == 0 &&
            MovesOf(*game, vs_dwarfs.moves, 1, &troll_actions) ==
                vs_dwarfs.trajectory.states.size() &&
            OpponentMovesMatch(*game, vs_dwarfs.moves, 0, dwarfs),
        "3. alpha-beta as the dwarfs: only the trolls' positions, its moves its own");

  // 4. Control: self-play.
  const Game self_play = Play(*game, nullptr, open_spiel::kInvalidPlayer, kSeed);
  int sides[2] = {0, 0};
  for (const auto& s : self_play.trajectory.states) ++sides[s.current_player];
  Check(sides[0] > 0 && sides[1] > 0 &&
            self_play.trajectory.states.size() == self_play.moves.size() &&
            self_play.trajectory.opponent == open_spiel::kInvalidPlayer,
        "4. control: self-play records every position, both sides, no opponent");

  // 5. The log lines.
  Check(absl::StrContains(vs_trolls.line, "; Opponent: alphabeta:1 as trolls; Actions: ") &&
            absl::StrContains(vs_dwarfs.line, "; Opponent: alphabeta:1 as dwarfs; Actions: ") &&
            !absl::StrContains(self_play.line, "Opponent"),
        "5. the log line names the opponent and its side; self-play's none");

  // 6. AlphaBetaPlayer's two habits. A troll next to a lone dwarf, trolls to move: depth
  // 1 captures. The same troll, dwarfs to move with a dwarf that can step next to it:
  // depth 2 does not.
  const std::string capture_text =
      "-----d....-----\n----.......----\n---.........---\n--...........--\n"
      "-.............-\n...............\n.......d.......\n.......OT......\n"
      "...............\n...............\n-.............-\n--...........--\n"
      "---.........---\n----.......----\n-----.....-----\nto_move=trolls";
  std::unique_ptr<State> capture = game->NewInitialState(capture_text);
  int captures = 0;
  for (Action a : capture->LegalActions()) captures += absl::EndsWith(capture->ActionToString(a), "x");
  AlphaBetaPlayer depth1(*game, 1), depth2(*game, 2);
  depth1.NewGame(kSeed);
  depth2.NewGame(kSeed);
  Check(captures >= 1 && capture->LegalActions().size() > captures &&
            absl::EndsWith(capture->ActionToString(depth1.Choose(*capture)), "x"),
        "6. depth 1 takes an available capture (control: other moves were legal)");
  // A lone dwarf at (6,1) and the troll at (7,8), the dwarfs to move: no capture for
  // them, and moving the dwarf to (6,7), (6,8) or (6,9) puts it next to the troll,
  // which then takes it.
  const std::string blunder_text =
      "-----.....-----\n----.......----\n---.........---\n--...........--\n"
      "-.............-\n...............\n.d.............\n.......OT......\n"
      "...............\n...............\n-.............-\n--...........--\n"
      "---.........---\n----.......----\n-----.....-----\nto_move=dwarfs";
  std::unique_ptr<State> blunder = game->NewInitialState(blunder_text);
  auto next_to_troll = [](const State& s, const std::string& move) {
    // The destination of "(r,c)-(r,c)".
    const size_t dash = move.find(")-(");
    const std::string to = move.substr(dash + 3, move.find(')', dash + 3) - dash - 3);
    const std::vector<std::string> rc = absl::StrSplit(to, ',');
    const int r = std::stoi(rc[0]), c = std::stoi(rc[1]);
    const auto& thud_state = open_spiel::down_cast<const open_spiel::thud::ThudState&>(s);
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        if ((dr || dc) && open_spiel::thud::IsOnBoard(r + dr, c + dc) &&
            thud_state.CellAt(r + dr, c + dc) == open_spiel::thud::Cell::kTroll) {
          return true;
        }
      }
    }
    return false;
  };
  int losing = 0, captures_for_dwarfs = 0;
  for (Action a : blunder->LegalActions()) {
    const std::string move = blunder->ActionToString(a);
    losing += next_to_troll(*blunder, move);
    captures_for_dwarfs += absl::EndsWith(move, "x");
  }
  // Depth 1 sees no difference between the moves (no capture either way) — the control
  // that the position needs the troll's reply to be seen; depth 2 sees it on every seed.
  bool depth2_safe = true;
  int depth1_next_to_troll = 0;
  for (int seed = 0; seed < 40; ++seed) {
    depth1.NewGame(seed);
    depth2.NewGame(seed);
    depth1_next_to_troll += next_to_troll(*blunder, blunder->ActionToString(depth1.Choose(*blunder)));
    depth2_safe = depth2_safe &&
                  !next_to_troll(*blunder, blunder->ActionToString(depth2.Choose(*blunder)));
  }
  std::cout << "  " << losing << " of " << blunder->LegalActions().size()
            << " dwarf moves stand next to the troll; depth 1 chose one on "
            << depth1_next_to_troll << " of 40 seeds" << std::endl;
  Check(losing > 0 && captures_for_dwarfs == 0 && depth2_safe,
        "6. depth 2 never steps the lone dwarf next to the troll (40 seeds; control: "
        "such moves exist)");

  // 7. The configuration.
  AlphaZeroConfig config;
  config.game = "thud";
  config.path = config.graph_def = config.nn_model = config.devices = "";
  config.nn_width = config.nn_depth = 1;
  config.explicit_learning = false;
  config.learning_rate = config.weight_decay = 0;
  config.train_batch_size = config.inference_batch_size = config.inference_threads = 1;
  config.inference_cache = config.replay_buffer_size = config.replay_buffer_reuse = 1;
  config.learner_batches = 0;
  config.symmetry_augmentation = config.policy_target_pruning = false;
  config.replay_buffer_start_size = 0;
  config.checkpoint_freq = config.evaluation_window = 1;
  config.uct_c = 2;
  config.max_simulations = 100;
  config.policy_alpha = config.policy_epsilon = 0;
  config.untried_move_value = "upstream";
  config.untried_move_reduction = 0.2;
  config.player_simulations = config.player_policy_temperature = "";
  config.alphabeta_share = 0.3;
  config.alphabeta_depth = "2,1";
  config.temperature = 1;
  config.temperature_drop = 10;
  config.cutoff_probability = config.cutoff_value = 0;
  config.actors = config.evaluators = config.eval_levels = 1;
  config.max_steps = 0;
  AlphaZeroConfig round_trip;
  round_trip.FromJson(config.ToJson());
  open_spiel::json::Object old = config.ToJson();
  old.erase("alphabeta_share");
  old.erase("alphabeta_depth");
  AlphaZeroConfig from_old;
  from_old.FromJson(old);
  // Run I's config.json (2026-10-07) holds the depth as a number.
  open_spiel::json::Object as_number = config.ToJson();
  as_number["alphabeta_depth"] = 3;
  AlphaZeroConfig from_number;
  from_number.FromJson(as_number);
  Check(round_trip.alphabeta_share == 0.3 && round_trip.alphabeta_depth == "2,1" &&
            from_old.alphabeta_share == 0 && from_old.alphabeta_depth == "2" &&
            from_number.alphabeta_depth == "3",
        "7. alphabeta_share and _depth round-trip; an older config.json loads with 0 and "
        "\"2\", one with a number as that number");
  using open_spiel::thud_az::torch_az::AlphaBetaDepth;
  AlphaZeroConfig one = config;
  one.alphabeta_depth = "3";
  Check(AlphaBetaDepth(round_trip, 0) == 2 && AlphaBetaDepth(round_trip, 1) == 1 &&
            AlphaBetaDepth(one, 0) == 3 && AlphaBetaDepth(one, 1) == 3,
        "8. the depth per side: \"2,1\" is 2 as the dwarfs and 1 as the trolls; one value "
        "is both");

  std::cout << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
