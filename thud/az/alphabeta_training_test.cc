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
//   9. (2026-10-26) The teacher as the opponent (alphabeta_player "teacher",
//      AlphaBetaPlayer::Mode::kTeacher), as the trolls at depth 2 and as the dwarfs at depth
//      1: every one of its moves is one the teacher's Scores rate best, and the one Choose
//      picks there with the game's seed (reproducible); the control: another seed draws
//      other moves somewhere (its choice among tied moves is the seed's). alphabeta_player
//      round-trips; an older config.json loads with "search"; per side (AlphaBetaMode),
//      "search,teacher" is the search as the dwarfs and the teacher as the trolls.
//  10. (2026-11-07) The opponent's finer evaluation (alphabeta_finer, MakeOpponent): with it
//      off, MakeOpponent's opponent chooses as one built by hand without it, though the
//      teacher's finer evaluation is on; with it on, as one given the threat term and the
//      positional terms by hand — the same moves and the same Choice on the positions of a
//      game, as the dwarfs and as the trolls. Choice: the plain opponent's draw is even
//      (effective = tied), the finer one's at most its tied count and below it somewhere, its
//      move among those it scores best. Controls: the finer opponent chooses otherwise than
//      the plain one somewhere; its draw is uneven somewhere. alphabeta_finer round-trips, an
//      older config.json loads without it. PlayGame logs the teacher opponent's choices on a
//      line of their own ("Game N: the opponent's choices: ..."), its moves counted and its
//      means those of Choice over them; the game line stays as before (its parsers read it);
//      control: the search opponent's game logs no such line.
//
//   thud/run_tests.sh alphabeta_training_test   # builds and runs it
//   build-tests/bin/alphabeta_training_test

#include <algorithm>
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
  std::string choices;             // The opponent's choices' line, if logged (2026-11-07).
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
  for (const std::string& line : logger.lines) {
    if (absl::StrContains(line, "Actions: ")) result.line = line;
    if (absl::StrContains(line, "the opponent's choices: ")) result.choices = line;
  }
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

  // 9. The teacher as the opponent.
  for (const auto& [side, depth] : std::vector<std::pair<Player, int>>{{1, 2}, {0, 1}}) {
    AlphaBetaPlayer teacher(*game, depth);
    teacher.SetMode(AlphaBetaPlayer::Mode::kTeacher);
    teacher.NewGame(kSeed);
    const Game g = Play(*game, &teacher, side, kSeed);
    // Each of its moves among the best-scored.
    int moves = 0, best = 0;
    std::unique_ptr<State> state = game->NewInitialState();
    for (const std::string& move : g.moves) {
      const std::vector<Action> legal = state->LegalActions();
      Action played = open_spiel::kInvalidAction;
      for (Action a : legal) {
        if (state->ActionToString(a) == move) played = a;
      }
      if (played == open_spiel::kInvalidAction) break;
      if (state->CurrentPlayer() == side) {
        const std::vector<double> scores = teacher.Scores(*state);
        const double top = *std::max_element(scores.begin(), scores.end());
        for (size_t i = 0; i < legal.size(); ++i) best += legal[i] == played && scores[i] == top;
        ++moves;
      }
      state->ApplyAction(played);
    }
    AlphaBetaPlayer other(*game, depth);
    other.SetMode(AlphaBetaPlayer::Mode::kTeacher);
    other.NewGame(kSeed + 1);
    const std::string as = side == 1 ? "trolls" : "dwarfs";
    std::cout << "  the teacher as the " << as << ": a game of " << g.moves.size() << " moves, "
              << moves << " of them its own" << std::endl;
    Check(moves > 10 && best == moves && g.trajectory.opponent == side,
          "9. the teacher as the " + as + " (depth " + std::to_string(depth) +
              "): every one of its moves is one its scores rate best");
    Check(OpponentMovesMatch(*game, g.moves, side, teacher),
          "9. ... and the one it chooses there with the game's seed");
    Check(!OpponentMovesMatch(*game, g.moves, side, other),
          "9. control: with another seed it draws another move somewhere");
  }
  {
    AlphaZeroConfig teacher_config = config;
    teacher_config.alphabeta_player = "teacher";
    AlphaZeroConfig back;
    back.FromJson(teacher_config.ToJson());
    Check(back.alphabeta_player == "teacher" && from_old.alphabeta_player == "search",
          "9. alphabeta_player round-trips; an older config.json loads with \"search\"");
    // Per side (AlphaBetaMode): "search,teacher" the search as the dwarfs, the teacher as the
    // trolls; one value both; the default the search.
    using open_spiel::thud_az::torch_az::AlphaBetaMode;
    using Mode = AlphaBetaPlayer::Mode;
    AlphaZeroConfig split = config;
    split.alphabeta_player = "search,teacher";
    Check(AlphaBetaMode(split, 0) == Mode::kSearch && AlphaBetaMode(split, 1) == Mode::kTeacher &&
              AlphaBetaMode(back, 0) == Mode::kTeacher && AlphaBetaMode(back, 1) == Mode::kTeacher &&
              AlphaBetaMode(from_old, 0) == Mode::kSearch &&
              AlphaBetaMode(from_old, 1) == Mode::kSearch,
          "9. the mode per side: \"search,teacher\" the search as the dwarfs and the teacher as "
          "the trolls; \"teacher\" both; the default the search");
  }

  // 10. The opponent's finer evaluation.
  {
    using open_spiel::thud_az::Threat;
    using open_spiel::thud_az::torch_az::MakeOpponent;
    using Choice = AlphaBetaPlayer::Choice;
    AlphaZeroConfig finer = config;
    finer.alphabeta_player = "teacher";
    finer.alphabeta_depth = "2";
    finer.kickstart_ties = "positional";
    finer.kickstart_temperature = 0.25;
    finer.teacher_threat = "captures";
    finer.teacher_threat_weights = "0.5,0.25";
    finer.teacher_terms_weights = "0.1,-0.2";
    AlphaZeroConfig off = finer;  // The teacher finer, the opponent plain.
    finer.alphabeta_finer = true;
    AlphaBetaPlayer plain(*game, 2), by_hand(*game, 2);
    by_hand.SetThreat(Threat::kCaptures, {0.5, 0.25});
    by_hand.SetPositionalTerms({0.1, -0.2}, 0.25);
    std::unique_ptr<AlphaBetaPlayer> made_off[2], made_on[2];
    for (int p = 0; p < 2; ++p) {
      made_off[p] = MakeOpponent(off, *game, p);
      made_on[p] = MakeOpponent(finer, *game, p);
      made_off[p]->NewGame(kSeed);
      made_on[p]->NewGame(kSeed);
    }
    for (AlphaBetaPlayer* a : {&plain, &by_hand}) {
      a->SetMode(AlphaBetaPlayer::Mode::kTeacher);
      a->NewGame(kSeed);
    }
    const auto same = [](const Choice& a, const Choice& b) {
      return a.tied == b.tied && a.effective == b.effective;
    };
    bool off_same = true, on_same = true, plain_even = true, finer_within = true,
         finer_best = true, differs = false, uneven = false;
    int positions[2] = {0, 0};
    std::mt19937 rng(kSeed);
    std::unique_ptr<State> s = game->NewInitialState();
    for (int ply = 0; ply < 160 && !s->IsTerminal(); ++ply) {
      const std::vector<Action> legal = s->LegalActions();
      if (ply % 7 == 3) {
        const Player p = s->CurrentPlayer();
        ++positions[p];
        Choice c_plain, c_hand, c_off, c_on;
        const Action a_plain = plain.Choose(*s, &c_plain);
        const Action a_hand = by_hand.Choose(*s, &c_hand);
        off_same = off_same && made_off[p]->Choose(*s, &c_off) == a_plain && same(c_off, c_plain);
        on_same = on_same && made_on[p]->Choose(*s, &c_on) == a_hand && same(c_on, c_hand);
        plain_even = plain_even && c_plain.tied >= 1 && c_plain.effective == c_plain.tied;
        finer_within = finer_within && c_hand.effective >= 1 - 1e-9 &&
                       c_hand.effective <= c_hand.tied + 1e-9;
        uneven = uneven || c_hand.effective < c_hand.tied - 1e-6;
        differs = differs || a_hand != a_plain || c_hand.tied != c_plain.tied;
        const std::vector<double> scores = by_hand.Scores(*s);  // Every score exact.
        const double top = *std::max_element(scores.begin(), scores.end());
        const size_t i = std::find(legal.begin(), legal.end(), a_hand) - legal.begin();
        finer_best = finer_best && i < legal.size() && scores[i] == top;
      }
      s->ApplyAction(legal[rng() % legal.size()]);
    }
    std::cout << "  " << positions[0] << " dwarf and " << positions[1] << " troll positions"
              << std::endl;
    Check(positions[0] > 5 && positions[1] > 5 && off_same && on_same,
          "10. MakeOpponent: with alphabeta_finer off it chooses as a plain opponent built by "
          "hand (the teacher's finer evaluation on), with it on as one given the threat term "
          "and the positional terms by hand — the same moves and Choice, both sides");
    Check(plain_even && finer_within && finer_best,
          "10. Choice: the plain draw even (effective = tied), the finer one's at most its tied "
          "count, its move among those it scores best");
    Check(differs && uneven,
          "10. controls: the finer opponent chooses otherwise than the plain one somewhere, and "
          "its draw is uneven somewhere");
    AlphaZeroConfig back;
    back.FromJson(finer.ToJson());
    open_spiel::json::Object older = finer.ToJson();
    older.erase("alphabeta_finer");
    AlphaZeroConfig from_older;
    from_older.FromJson(older);
    Check(back.alphabeta_finer && !from_older.alphabeta_finer && !config.alphabeta_finer,
          "10. alphabeta_finer round-trips; an older config.json loads with false, the default");

    // PlayGame's log of the teacher opponent's choices (as the trolls, depth 1: fast).
    AlphaBetaPlayer teacher(*game, 1);
    teacher.SetMode(AlphaBetaPlayer::Mode::kTeacher);
    teacher.NewGame(kSeed);
    const Game g = Play(*game, &teacher, /*side=*/1, kSeed);
    std::unique_ptr<State> replay = game->NewInitialState();
    int moves = 0, single = 0;
    double tied = 0, effective = 0;
    for (const std::string& move : g.moves) {
      Action played = open_spiel::kInvalidAction;
      for (Action a : replay->LegalActions()) {
        if (replay->ActionToString(a) == move) played = a;
      }
      if (played == open_spiel::kInvalidAction) break;
      if (replay->CurrentPlayer() == 1) {
        Choice c;
        teacher.Choose(*replay, &c);
        ++moves;
        single += c.tied == 1;
        tied += c.tied;
        effective += c.effective;
      }
      replay->ApplyAction(played);
    }
    char expected[256];
    std::snprintf(expected, sizeof(expected),
                  "Game 1: the opponent's choices: %d moves, tied for its best %.2f a move, its "
                  "draw %.2f effective, a single move in %.1f%%",
                  moves, tied / moves, effective / moves, 100.0 * single / moves);
    std::cout << "  " << g.choices << std::endl;
    Check(moves > 10 && g.choices == expected &&
              absl::StrContains(g.line, "; Opponent: alphabeta:1 as trolls; Actions: ") &&
              !absl::StrContains(g.choices, "Actions: "),
          "10. PlayGame logs the teacher opponent's choices on a line of their own, its moves "
          "and its means those of Choice; the game line as before");
    Check(vs_trolls.choices.empty(),
          "10. control: the search opponent's game logs no choices");
  }

  std::cout << (failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
