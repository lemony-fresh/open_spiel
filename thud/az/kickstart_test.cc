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
//      weight 0 and no per-player weights or depths; KickstartWeights fills missing
//      entries from the single weight; KickstartDepths takes the list, one entry for both,
//      or kickstart_depth.
//   5. Cost, with `cost=1` only (a measurement, not a test, and the slowest part):
//      Scores at depths 2 and 3 on self-play-like positions, printed.
//   7. (2026-10-24) A teacher with a depth per side to move (player_kickstart_depth):
//      {2, 3} scores the dwarfs' positions as depth 2 and the trolls' as depth 3, {3, 2}
//      the other way round (the control: depths 2 and 3 differ on some of each side's).
//   6. The teacher's ties shared by the search (kickstart_ties=search, 2026-10-23):
//      TeacherTiesBySearch by hand — a tied group's total shared by the visits within
//      it, evenly where none, single moves and the groups' totals unchanged, the result
//      summing to 1 (the control: it differs from the teacher's) — and in PlayGame: the
//      same moves played, every target (1 - w) x the visits + w x TeacherTiesBySearch of
//      TeacherPolicy and the visits, exactly (the control: some target differs from the
//      even ties'); the setting round-trips, an older config.json loads "even" (part 4).
//   8. (2026-11-04) The teacher's finer evaluation (thud/PLAN.md): TermsScores equal each
//      child's PositionalScore from the mover's view (computed apart, by Child); with
//      kickstart_ties=positional TeacherTiesByTerms keeps every group's total, shares it
//      inside by exp(term / temperature), sums to 1, gives the teacher's own policy at zero
//      weights (the control: with weights it differs); PlayGame plays the same moves and
//      records (1 - w) x the visits + w x TeacherTiesByTerms(TeacherPolicy); the teacher as a
//      player draws inside its best group, by the terms' shares over 4,000 seeds (each within
//      0.03), evenly without terms; a threat setting reaches the player's Scores
//      (MaterialSearch with SetThreat, the same scores; the control: kOff's differ); the three
//      settings round-trip, an older config.json loads them off (part 4).
//
//   thud/run_tests.sh kickstart_test   # builds and runs it
//   build-tests/bin/kickstart_test

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
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
                const std::vector<double>& weights = {}, bool ties_by_search = false) {
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
                  weights, /*fast_bots=*/nullptr, /*full_share=*/1, ties_by_search);
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

int main(int argc, char** argv) {
  const bool cost = argc > 1 && std::string(argv[1]) == "cost=1";
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

  // 6. The teacher's ties shared by the search.
  {
    using open_spiel::ActionsAndProbs;
    using open_spiel::thud_az::torch_az::TeacherTiesBySearch;
    auto near = [](const ActionsAndProbs& got, const ActionsAndProbs& want) {
      bool ok = got.size() == want.size();
      double sum = 0;
      for (size_t i = 0; ok && i < got.size(); ++i) {
        ok = got[i].first == want[i].first && std::abs(got[i].second - want[i].second) < 1e-12;
        sum += got[i].second;
      }
      return ok && std::abs(sum - 1) < 1e-12;
    };
    // A group of three (0.3 each) and a single move: the group's 0.9 by its visits 0.5,
    // 0, 0.25; the single move keeps 0.1.
    const ActionsAndProbs teacher1 = {{1, 0.3}, {2, 0.3}, {3, 0.3}, {4, 0.1}};
    const ActionsAndProbs visits1 = {{1, 0.5}, {2, 0}, {3, 0.25}, {4, 0.25}};
    const ActionsAndProbs shared1 = TeacherTiesBySearch(teacher1, visits1);
    // A group the search left unvisited stays even; another is shared by its visits.
    const ActionsAndProbs teacher2 = {{1, 0.2}, {2, 0.2}, {3, 0.3}, {4, 0.3}};
    const ActionsAndProbs visits2 = {{1, 0}, {2, 0}, {3, 0.9}, {4, 0.1}};
    const ActionsAndProbs shared2 = TeacherTiesBySearch(teacher2, visits2);
    Check(near(shared1, {{1, 0.6}, {2, 0}, {3, 0.3}, {4, 0.1}}) &&
              near(shared2, {{1, 0.2}, {2, 0.2}, {3, 0.54}, {4, 0.06}}) &&
              !near(shared1, teacher1),
          "6. TeacherTiesBySearch: a group's total shared by its visits, evenly where none, "
          "single moves kept, summing to 1 (control: it differs from the teacher)");
    // In PlayGame.
    const double w = 0.3;
    const Trajectory none = Play(*game, nullptr, 0);
    const Trajectory even = Play(*game, &depth2, w);
    const Trajectory searched = Play(*game, &depth2, w, {}, /*ties_by_search=*/true);
    bool same_moves = none.states.size() == searched.states.size();
    for (int i = 0; same_moves && i < none.states.size(); ++i) {
      same_moves = none.states[i].action == searched.states[i].action;
    }
    bool exact = same_moves, differ = false;
    std::unique_ptr<State> state = game->NewInitialState();
    for (int i = 0; exact && i < none.states.size(); ++i) {
      std::map<Action, double> taught;
      for (const auto& [a, p] :
           TeacherTiesBySearch(TeacherPolicy(*state, depth2, 1), none.states[i].policy)) {
        taught[a] = p;
      }
      for (int j = 0; j < none.states[i].policy.size(); ++j) {
        const auto& [a, v] = none.states[i].policy[j];
        const double got = searched.states[i].policy[j].second;
        exact = exact && searched.states[i].policy[j].first == a &&
                std::abs(got - ((1 - w) * v + w * taught[a])) < 1e-12;
        differ = differ || std::abs(got - even.states[i].policy[j].second) > 1e-9;
      }
      state->ApplyAction(none.states[i].action);
    }
    Check(same_moves, "6. with the ties shared by the search the same moves are played");
    Check(exact && differ,
          "6. every target is 0.7 x the visits + 0.3 x TeacherTiesBySearch(TeacherPolicy, "
          "the visits) (control: some differ from the even ties')");
  }

  // 8. The teacher's finer evaluation.
  {
    using open_spiel::ActionsAndProbs;
    using open_spiel::thud_az::ComputePositionalTerms;
    using open_spiel::thud_az::PositionalScore;
    using open_spiel::thud_az::Threat;
    using open_spiel::thud_az::torch_az::TeacherTiesByTerms;
    const std::array<double, 2> weights = {0.1, -0.2};
    const double temperature = 0.25;
    AlphaBetaPlayer termed(*game, 2), zero(*game, 2);
    termed.SetPositionalTerms(weights, temperature);
    zero.SetPositionalTerms({0, 0}, temperature);
    // Positions of a game played on, every 9th move, both sides to move.
    std::vector<std::unique_ptr<State>> positions;
    {
      std::mt19937 rng(5);
      std::unique_ptr<State> s = game->NewInitialState();
      for (int ply = 0; ply < 120 && !s->IsTerminal(); ++ply) {
        if (ply % 9 == 4) positions.push_back(s->Clone());
        const std::vector<Action> legal = s->LegalActions();
        s->ApplyAction(legal[rng() % legal.size()]);
      }
    }
    bool terms_ok = true, totals = true, inside = true, sums = true, zero_same = true,
         differs = false;
    for (const auto& s : positions) {
      const std::vector<double> terms = termed.TermsScores(*s);
      const std::vector<Action> legal = s->LegalActions();
      const double sign = s->CurrentPlayer() == open_spiel::thud::kDwarfPlayer ? 1 : -1;
      for (size_t i = 0; i < legal.size(); ++i) {
        const auto child = s->Child(legal[i]);
        const double want = sign * PositionalScore(ComputePositionalTerms(
            open_spiel::down_cast<const open_spiel::thud::ThudState&>(*child)), weights);
        terms_ok = terms_ok && std::abs(terms[i] - want) < 1e-12;
      }
      const ActionsAndProbs teacher = TeacherPolicy(*s, termed, temperature);
      const ActionsAndProbs shared = TeacherTiesByTerms(teacher, *s, termed, temperature);
      const ActionsAndProbs unshared = TeacherTiesByTerms(teacher, *s, zero, temperature);
      std::map<double, double> before, after;
      double sum = 0;
      for (size_t i = 0; i < teacher.size(); ++i) {
        before[teacher[i].second] += teacher[i].second;
        after[teacher[i].second] += shared[i].second;
        sum += shared[i].second;
        zero_same = zero_same && unshared[i].first == teacher[i].first &&
                    std::abs(unshared[i].second - teacher[i].second) < 1e-12;
        differs = differs || std::abs(shared[i].second - teacher[i].second) > 1e-6;
        for (size_t j = 0; j < teacher.size(); ++j) {
          if (teacher[j].second != teacher[i].second || shared[j].second < 1e-200) continue;
          const double ratio = shared[i].second / shared[j].second;
          const double want = std::exp((terms[i] - terms[j]) / temperature);
          inside = inside && std::abs(ratio / want - 1) < 1e-9;
        }
      }
      for (const auto& [p, total] : before) totals = totals && std::abs(after[p] - total) < 1e-12;
      sums = sums && std::abs(sum - 1) < 1e-12;
    }
    Check(terms_ok && !positions.empty(),
          "8. TermsScores: each child's PositionalScore from the mover's view");
    Check(totals && inside && sums && zero_same && differs,
          "8. TeacherTiesByTerms: groups' totals kept, shared by exp(term / temperature), "
          "summing to 1, the teacher's own at zero weights (control: with weights it differs)");
    // In PlayGame.
    const double w = 0.3;
    const Trajectory none = Play(*game, nullptr, 0);
    const Trajectory termed_play = Play(*game, &termed, w);
    bool same_moves = none.states.size() == termed_play.states.size();
    for (int i = 0; same_moves && i < none.states.size(); ++i) {
      same_moves = none.states[i].action == termed_play.states[i].action;
    }
    bool exact = same_moves;
    std::unique_ptr<State> state = game->NewInitialState();
    for (int i = 0; exact && i < none.states.size(); ++i) {
      std::map<Action, double> taught;
      // PlayGame's teacher policy at Play's teacher temperature (1), its ties split at the
      // terms' (0.25); the trainer sets both to kickstart_temperature.
      for (const auto& [a, p] : TeacherTiesByTerms(TeacherPolicy(*state, termed, 1), *state,
                                                   termed, termed.terms_temperature())) {
        taught[a] = p;
      }
      for (int j = 0; j < none.states[i].policy.size(); ++j) {
        const auto& [a, v] = none.states[i].policy[j];
        exact = exact && termed_play.states[i].policy[j].first == a &&
                std::abs(termed_play.states[i].policy[j].second -
                         ((1 - w) * v + w * taught[a])) < 1e-12;
      }
      state->ApplyAction(none.states[i].action);
    }
    Check(same_moves && exact,
          "8. PlayGame with positional ties: the same moves, every target 0.7 x the visits + "
          "0.3 x TeacherTiesByTerms(TeacherPolicy)");
    // The teacher as a player: a position whose best group has moves of different terms.
    AlphaBetaPlayer player(*game, 2), even_player(*game, 2);
    player.SetMode(AlphaBetaPlayer::Mode::kTeacher);
    even_player.SetMode(AlphaBetaPlayer::Mode::kTeacher);
    player.SetPositionalTerms(weights, temperature);
    const State* chosen = nullptr;
    std::vector<double> shares;
    std::vector<Action> group;
    // More positions than above: one whose best group has a move standing out is needed.
    std::vector<std::unique_ptr<State>> candidates;
    {
      std::mt19937 rng(7);
      for (int g = 0; g < 6; ++g) {
        std::unique_ptr<State> s = game->NewInitialState();
        for (int ply = 0; ply < 160 && !s->IsTerminal(); ++ply) {
          if (ply % 3 == 2) candidates.push_back(s->Clone());
          const std::vector<Action> legal = s->LegalActions();
          s->ApplyAction(legal[rng() % legal.size()]);
        }
      }
    }
    for (const auto& s : candidates) {
      const std::vector<double> scores = player.Scores(*s);
      const std::vector<double> terms = player.TermsScores(*s);
      const std::vector<Action> legal = s->LegalActions();
      const double best = *std::max_element(scores.begin(), scores.end());
      group.clear();
      shares.clear();
      double sum = 0;
      for (size_t i = 0; i < legal.size(); ++i) {
        if (scores[i] != best) continue;
        group.push_back(legal[i]);
        shares.push_back(std::exp(terms[i] / temperature));
        sum += shares.back();
      }
      for (double& x : shares) x /= sum;
      const double most = *std::max_element(shares.begin(), shares.end());
      if (group.size() >= 3 && group.size() <= 60 && most >= std::max(0.2, 3.0 / group.size())) {
        chosen = s.get();
        break;
      }
    }
    Check(chosen != nullptr,
          "8. a position with a best group of 3-60 moves, one with a share of 0.2 and 3x even");
    if (chosen != nullptr) {
      std::map<Action, int> drawn, drawn_even;
      const int n = 4000;
      for (int seed = 0; seed < n; ++seed) {
        player.NewGame(seed);
        even_player.NewGame(seed);
        ++drawn[player.Choose(*chosen)];
        ++drawn_even[even_player.Choose(*chosen)];
      }
      bool within = true, in_group = true, even_ok = true;
      const size_t top = std::max_element(shares.begin(), shares.end()) - shares.begin();
      // The control: the top move's frequency away from the even share, so that an even draw
      // would fail the comparison with the terms' shares.
      const bool not_even =
          std::abs(static_cast<double>(drawn[group[top]]) / n - 1.0 / group.size()) > 0.1;
      for (const auto& [a, k] : drawn) {
        in_group = in_group && std::find(group.begin(), group.end(), a) != group.end();
      }
      for (size_t i = 0; i < group.size(); ++i) {
        within = within && std::abs(static_cast<double>(drawn[group[i]]) / n - shares[i]) < 0.03;
        even_ok = even_ok &&
                  std::abs(static_cast<double>(drawn_even[group[i]]) / n - 1.0 / group.size()) <
                      0.03;
      }
      Check(in_group && within && even_ok && not_even,
            absl::StrCat("8. the teacher as a player: inside its best group of ", group.size(),
                         ", drawn by the terms' shares (each within 0.03 over 4,000 seeds), "
                         "evenly without terms (control: the top move's frequency 0.1 or more "
                         "from even)"));
    }
    // The threat setting reaches the player's Scores.
    AlphaBetaPlayer threatened(*game, 3);
    threatened.SetThreat(Threat::kCaptures, {0.5, 0.5});
    bool same = true, other = false;
    for (const auto& s : positions) {
      open_spiel::thud_az::MaterialSearch search;
      search.SetThreat(Threat::kCaptures, {0.5, 0.5});
      const std::vector<double> got = threatened.Scores(*s);
      same = same && got == search.Scores(*s, 3);
      other = other || got != depth3.Scores(*s);
    }
    Check(same && other, "8. SetThreat reaches the player's Scores (control: kOff's differ)");
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
                        c.kickstart_temperature == 1 && c.player_kickstart_weight.empty() &&
                        c.kickstart_ties == "even" && c.teacher_threat == "off" &&
                        c.teacher_threat_weights.empty() && c.teacher_terms_weights.empty();
    c.kickstart_weight = 0.3;
    c.kickstart_depth = 3;
    c.kickstart_temperature = 0.5;
    c.player_kickstart_weight = "0.7";
    c.kickstart_ties = "search";
    c.player_kickstart_depth = "4,5";
    c.teacher_threat = "captures";
    c.teacher_threat_weights = "0.5,0.25";
    c.teacher_terms_weights = "0.1,-0.2";
    AlphaZeroConfig back;
    back.FromJson(c.ToJson());
    Check(back.teacher_threat == "captures" && back.teacher_threat_weights == "0.5,0.25" &&
              back.teacher_terms_weights == "0.1,-0.2",
          "8. the finer evaluation's settings round-trip");
    Check(old_ok && c.player_kickstart_depth.size() == 3 && back.kickstart_weight == 0.3 &&
              back.kickstart_depth == 3 && back.kickstart_temperature == 0.5 &&
              back.player_kickstart_weight == "0.7" && back.kickstart_ties == "search" &&
              back.player_kickstart_depth == "4,5",
          "4. the settings round-trip; an older config.json loads with weight 0, depth 2, "
          "temperature 1, no weights or depths per player, even ties");
    // KickstartDepths: the list's entries, one entry for both, empty: kickstart_depth.
    const std::array<int, 2> listed = open_spiel::thud_az::torch_az::KickstartDepths(back);
    back.player_kickstart_depth = "5";
    const std::array<int, 2> one = open_spiel::thud_az::torch_az::KickstartDepths(back);
    back.player_kickstart_depth = "";
    const std::array<int, 2> none = open_spiel::thud_az::torch_az::KickstartDepths(back);
    Check(listed == std::array<int, 2>{4, 5} && one == std::array<int, 2>{5, 5} &&
              none == std::array<int, 2>{3, 3},
          "4. KickstartDepths: \"4,5\" gives the dwarfs 4 and the trolls 5; \"5\" both 5; "
          "empty both kickstart_depth (3)");
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

  // 7. The teacher's depth per side to move (player_kickstart_depth, 2026-10-24): a teacher
  // with the depths {2, 3} scores every dwarf position as one of depth 2 and every troll
  // position as one of depth 3, and {3, 2} the other way round, on positions of random,
  // capture-greedy games. The control: depths 2 and 3 score some positions of each side
  // differently (else a teacher ignoring the side would pass).
  {
    std::vector<std::unique_ptr<State>> positions;
    std::mt19937 rng(7);
    for (int g = 0; positions.size() < 40; ++g) {
      std::unique_ptr<State> s = game->NewInitialState();
      for (int ply = 0; !s->IsTerminal() && ply < 300; ++ply) {
        if (ply % 11 == 5) positions.push_back(s->Clone());
        std::vector<Action> legal = s->LegalActions(), captures;
        for (Action a : legal) {
          if (absl::EndsWith(s->ActionToString(a), "x")) captures.push_back(a);
        }
        const std::vector<Action>& pool = captures.empty() ? legal : captures;
        s->ApplyAction(pool[rng() % pool.size()]);
      }
    }
    const AlphaBetaPlayer two_three(*game, std::array<int, 2>{2, 3});
    const AlphaBetaPlayer three_two(*game, std::array<int, 2>{3, 2});
    int wrong = 0, sides[2] = {0, 0}, differ[2] = {0, 0};
    for (const auto& s : positions) {
      const open_spiel::Player p = s->CurrentPlayer();
      if (p < 0) continue;
      const std::vector<double> d2 = depth2.Scores(*s), d3 = depth3.Scores(*s);
      wrong += two_three.Scores(*s) != (p == 0 ? d2 : d3);
      wrong += three_two.Scores(*s) != (p == 0 ? d3 : d2);
      ++sides[p];
      differ[p] += d2 != d3;
    }
    Check(wrong == 0 && differ[0] > 0 && differ[1] > 0 && two_three.depth(0) == 2 &&
              two_three.depth(1) == 3,
          absl::StrCat("7. a teacher with a depth per side: ", wrong, " of ",
                       2 * (sides[0] + sides[1]), " positions' scores differ from those of "
                       "the side's depth (0 required); the control, depths 2 and 3 differ on ",
                       differ[0], " of ", sides[0], " dwarf positions and ", differ[1], " of ",
                       sides[1], " troll positions (more than 0 each required)"));
  }

  // 5. Cost, if asked for.
  if (cost) {
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
      std::cout << "  cost at depth " << t->depth(0) << ": dwarfs "
                << 1000 * seconds[0] / count[0] << " ms a position (" << count[0]
                << "), trolls " << 1000 * seconds[1] / count[1] << " ms (" << count[1] << ")"
                << std::endl;
    }
  }

  std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
  return failures ? 1 : 0;
}
