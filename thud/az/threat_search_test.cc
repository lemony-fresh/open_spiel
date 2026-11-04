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

// Tests MaterialSearch's threat term (material_search.h, Threat; thud/PLAN.md, *The teacher's
// finer evaluation*) against a reference written here: plain negamax, every move searched, no
// table, its frontier valued by brute force — the threats from the same board with the other
// side to move, the captures and the quiet moves from LegalActions() and Captures(), never
// from BestCaptureFor, CaptureActions or HasQuietMove. On positions of random games, uniform
// and capture-greedy, in the training's game (no-capture limit 100) and the default:
//
//   1. kOff is today's search: SetThreat(kOff) gives a fresh search's scores and nodes, and
//      after SetThreat(kScan) and back to kOff too (the table cleared in between).
//   2. kScan and kCaptures equal the reference exactly at depth 2 (and 3 on every 12th
//      position: the reference takes ~1 s a position there), for three pairs of
//      weights, with and without a table, and on copies of positions 1-3 turns short of the
//      no-capture limit and 1-2 short of the turn limit (where a move ends the battle and no
//      threat counts).
//   3. The root window with the threat term: at margin 1 the best moves' scores exact, every
//      other move at most best - 1 and at least its true score.
//   4. The controls, required: frontier nodes with threats, threats where a quiet move ends
//      the battle, capture lines with a threat after the capture, a frontier side without a
//      quiet move; kScan with weights differing from kOff somewhere, and kCaptures at weight
//      0 differing from kOff somewhere (a capture walking into a recapture).
//   5. With `deep`: depth 4 on positions with the trolls to move (the reference's cost).
//
//   threat_search_test [deep|quick]   (quick: depth 2 alone, for the sanitized build)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/material_search.h"

namespace {

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::thud_az::MaterialSearch;
using open_spiel::thud_az::Threat;
namespace thud = open_spiel::thud;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

constexpr int kUnit = 100;  // The search's unit with a threat term.

const thud::ThudState& AsThud(const State& state) {
  return open_spiel::down_cast<const thud::ThudState&>(state);
}

// The coverage of the reference's frontier nodes.
struct Tally {
  int64_t frontier = 0, threatened = 0, threatened_ending = 0, capture_lines = 0,
          recaptured = 0, no_quiet = 0;
};

// The reference: values in units for the side to move, every move searched.
class Reference {
 public:
  Reference(Threat threat, std::array<double, 2> weights, Tally* tally)
      : threat_(threat), weights_(weights), tally_(tally) {}

  std::vector<double> Scores(const State& state, int depth) {
    const Player me = state.CurrentPlayer();
    std::vector<double> scores;
    for (Action action : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(action);
      const int value = child->IsTerminal() || depth == 1
                            ? Material(*child, me)
                            : -Search(*child, depth - 1, 1 - me);
      scores.push_back(static_cast<double>(value) / kUnit);
    }
    return scores;
  }

 private:
  static int Material(const State& state, Player me) {
    const auto& s = AsThud(state);
    const int margin = s.NumDwarfs() * thud::kDwarfValue - s.NumTrolls() * thud::kTrollValue;
    return (me == thud::kDwarfPlayer ? margin : -margin) * kUnit;
  }

  // The most pieces one legal move of the side to move takes.
  static int MostTaken(const thud::ThudState& s) {
    int most = 0;
    for (Action a : s.LegalActions()) most = std::max(most, s.Captures(a));
    return most;
  }

  // The most pieces `side` could take on this board: the same board with it to move.
  static int MostTakenBy(const thud::ThudState& s, Player side) {
    if (s.CurrentPlayer() == side) return MostTaken(s);
    std::string text = s.ToString();
    const std::string from = side == thud::kDwarfPlayer ? "to_move=trolls" : "to_move=dwarfs";
    text.replace(text.find(from), from.size(),
                 side == thud::kDwarfPlayer ? "to_move=dwarfs" : "to_move=trolls");
    std::unique_ptr<State> other = s.GetGame()->NewInitialState(text);
    return other->IsTerminal() ? 0 : MostTaken(AsThud(*other));
  }

  int Frontier(const State& state, Player me) {
    const auto& s = AsThud(state);
    const auto& game = open_spiel::down_cast<const thud::ThudGame&>(*s.GetGame());
    const int material = Material(s, me);
    const int mine = (me == thud::kDwarfPlayer ? thud::kDwarfValue : thud::kTrollValue) * kUnit;
    const int theirs = (me == thud::kDwarfPlayer ? thud::kTrollValue : thud::kDwarfValue) * kUnit;
    const bool any_ends = s.TurnsPlayed() + 1 >= game.max_turns();
    const bool quiet_ends =
        any_ends || s.TurnsWithoutCapture() + 1 >= game.max_turns_without_capture();
    const int threat = MostTakenBy(s, 1 - me) * mine;
    const int penalty = static_cast<int>(std::lround(weights_[me] * threat));
    ++tally_->frontier;
    tally_->threatened += threat > 0;
    tally_->threatened_ending += threat > 0 && quiet_ends;
    int best = std::numeric_limits<int>::min();
    bool quiet = false;
    for (Action a : s.LegalActions()) {
      const int taken = s.Captures(a);
      if (taken == 0) {
        quiet = true;
        continue;
      }
      int value;
      if (threat_ == Threat::kScan) {
        value = material + taken * theirs - (any_ends ? 0 : penalty);
      } else {
        std::unique_ptr<State> child = s.Child(a);
        if (child->IsTerminal()) {
          value = static_cast<int>(std::lround(child->PlayerReturn(me) * thud::kMaxMargin)) * kUnit;
        } else {
          const int back = MostTaken(AsThud(*child));
          ++tally_->capture_lines;
          tally_->recaptured += back > 0;
          value = Material(*child, me) - back * mine;
        }
      }
      best = std::max(best, value);
    }
    if (quiet) best = std::max(best, material - (quiet_ends ? 0 : penalty));
    tally_->no_quiet += !quiet;
    return best;
  }

  int Search(const State& state, int depth, Player me) {
    if (state.IsTerminal() || depth == 0) return Material(state, me);
    if (depth == 1) return Frontier(state, me);
    int best = std::numeric_limits<int>::min();
    for (Action action : state.LegalActions()) {
      best = std::max(best, -Search(*state.Child(action), depth - 1, 1 - me));
    }
    return best;
  }

  const Threat threat_;
  const std::array<double, 2> weights_;
  Tally* tally_;
};

// Positions of random games, every `every`th, uniform or capture-greedy.
std::vector<std::unique_ptr<State>> RandomPositions(const Game& game, int games, int every,
                                                    bool greedy, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<std::unique_ptr<State>> positions;
  for (int g = 0; g < games; ++g) {
    std::unique_ptr<State> state = game.NewInitialState();
    for (int ply = 0; !state->IsTerminal() && ply < 400; ++ply) {
      if (ply % every == every - 1) positions.push_back(state->Clone());
      std::vector<Action> legal = state->LegalActions(), captures;
      for (Action a : legal) {
        if (greedy && AsThud(*state).Captures(a) > 0) captures.push_back(a);
      }
      const std::vector<Action>& pool = captures.empty() ? legal : captures;
      state->ApplyAction(pool[rng() % pool.size()]);
    }
  }
  return positions;
}

// `state` with its counters set `short_of_capture` turns short of the no-capture limit and
// `short_of_turns` short of the turn limit (0: unchanged).
std::unique_ptr<State> NearLimits(const State& state, int short_of_capture, int short_of_turns) {
  const auto& s = AsThud(state);
  const auto& game = open_spiel::down_cast<const thud::ThudGame&>(*s.GetGame());
  std::string text = s.ToString();
  while (!text.empty() && text.back() == '\n') text.pop_back();
  text = text.substr(0, text.rfind('\n'));  // The board alone.
  int turns = short_of_turns > 0 ? game.max_turns() - short_of_turns : s.TurnsPlayed();
  int quiet = s.TurnsWithoutCapture();
  if (short_of_capture > 0) {
    quiet = game.max_turns_without_capture() - short_of_capture;
    turns = std::max(turns, quiet);
  }
  return s.GetGame()->NewInitialState(absl::StrCat(
      text, "\nto_move=", s.CurrentPlayer() == thud::kDwarfPlayer ? "dwarfs" : "trolls",
      " turns=", turns, " turns_without_capture=", std::min(quiet, turns)));
}

}  // namespace

int main(int argc, char** argv) {
  const bool deep = argc > 1 && std::string(argv[1]) == "deep";
  const bool quick = argc > 1 && std::string(argv[1]) == "quick";
  std::shared_ptr<const Game> training =
      open_spiel::LoadGame("thud(max_turns_without_capture=100)");
  std::shared_ptr<const Game> standard = open_spiel::LoadGame("thud");
  std::vector<std::unique_ptr<State>> positions;
  for (auto& p : RandomPositions(*training, 4, 37, false, 1)) positions.push_back(std::move(p));
  for (auto& p : RandomPositions(*training, 6, 23, true, 2)) positions.push_back(std::move(p));
  for (auto& p : RandomPositions(*standard, 3, 41, true, 3)) positions.push_back(std::move(p));
  // A dwarf boxed in on the cut corner by four trolls, a fifth troll free, the trolls to
  // move: after the free troll's moves the dwarfs reach the frontier with captures only.
  {
    std::string board;
    for (int r = 0; r < thud::kBoardSize; ++r) {
      for (int c = 0; c < thud::kBoardSize; ++c) {
        const bool troll = (r == 0 && c == 6) || (r == 1 && c >= 4 && c <= 6) ||
                           (r == 10 && c == 10);
        board += !thud::IsOnBoard(r, c) ? '-'
                 : r == 7 && c == 7     ? 'O'
                 : r == 0 && c == 5     ? 'd'
                 : troll                ? 'T'
                                        : '.';
      }
      board += '\n';
    }
    positions.push_back(training->NewInitialState(board + "to_move=trolls"));
  }
  const size_t far = positions.size();
  for (size_t i = 0; i < far; i += 4) {
    for (int c = 1; c <= 3; ++c) positions.push_back(NearLimits(*positions[i], c, 0));
    for (int t = 1; t <= 2; ++t) positions.push_back(NearLimits(*positions[i], 0, t));
  }
  int dwarfs_to_move = 0;
  for (const auto& p : positions) dwarfs_to_move += p->CurrentPlayer() == thud::kDwarfPlayer;
  std::cout << positions.size() << " positions (" << far << " from the games, "
            << dwarfs_to_move << " with the dwarfs to move)" << std::endl;

  // 1. kOff is today's search.
  {
    bool same = true;
    for (int depth : {3, 4}) {
      for (size_t i = 0; i < far; i += 3) {
        MaterialSearch fresh, off, toggled(open_spiel::thud_az::Evaluation::kMaterial, true, true,
                                           16);
        MaterialSearch fresh_table(open_spiel::thud_az::Evaluation::kMaterial, true, true, 16);
        off.SetThreat(Threat::kOff);
        const std::vector<double> a = fresh.Scores(*positions[i], depth);
        const std::vector<double> b = off.Scores(*positions[i], depth);
        toggled.SetThreat(Threat::kScan, {0.5, 0.5});
        toggled.Scores(*positions[i], depth);
        toggled.SetThreat(Threat::kOff);
        toggled.ResetNodes();
        const std::vector<double> c = toggled.Scores(*positions[i], depth);
        const std::vector<double> d = fresh_table.Scores(*positions[i], depth);
        same = same && a == b && fresh.nodes() == off.nodes() && c == d &&
               toggled.nodes() == fresh_table.nodes();
      }
    }
    Check(same, "1. kOff equals a fresh search, scores and nodes, also after kScan and back");
  }

  // 2. and 4. The threat term against the reference.
  Tally tally;
  bool scan_differs = false, captures_differs = false;
  const std::vector<std::array<double, 2>> weights = {{0, 0}, {0.5, 0.25}, {1, 1}};
  for (Threat threat : {Threat::kScan, Threat::kCaptures}) {
    const char* name = threat == Threat::kScan ? "kScan" : "kCaptures";
    for (int depth : {2, 3}) {
      if (quick && depth == 3) continue;
      int moves = 0, wrong = 0;
      for (const auto& w : weights) {
        Reference reference(threat, w, &tally);
        MaterialSearch plain, tabled(open_spiel::thud_az::Evaluation::kMaterial, true, true, 16);
        plain.SetThreat(threat, w);
        tabled.SetThreat(threat, w);
        for (size_t i = 0; i < positions.size(); ++i) {
          // Depth 3 is slow for the reference (~1 s a position): every 12th.
          if (depth == 3 && i % 12 != 0) continue;
          const std::vector<double> want = reference.Scores(*positions[i], depth);
          const std::vector<double> got = plain.Scores(*positions[i], depth);
          const std::vector<double> got_tabled = tabled.Scores(*positions[i], depth);
          MaterialSearch off;
          const std::vector<double> material = off.Scores(*positions[i], depth);
          for (size_t m = 0; m < want.size(); ++m) {
            ++moves;
            if (got[m] != want[m] || got_tabled[m] != want[m]) {
              if (++wrong <= 3) {
                std::cout << absl::StrFormat("  %s depth %d weights %.2f,%.2f move %d: got %.2f "
                                             "(table %.2f), want %.2f in\n%s\n",
                                             name, depth, w[0], w[1], m, got[m], got_tabled[m],
                                             want[m], positions[i]->ToString());
              }
            }
            if (threat == Threat::kScan && w[0] > 0 && got[m] != material[m]) scan_differs = true;
            if (threat == Threat::kCaptures && w[0] == 0 && got[m] != material[m]) {
              captures_differs = true;
            }
          }
        }
      }
      Check(wrong == 0, absl::StrFormat("2. %s at depth %d: %d of %d moves' scores equal the "
                                        "reference (3 weights, with and without a table)",
                                        name, depth, moves - wrong, moves));
    }
  }
  Check(tally.threatened > 100 && tally.threatened_ending > 10 && tally.capture_lines > 100 &&
            tally.recaptured > 10 && tally.no_quiet > 0,
        absl::StrFormat("4. coverage: %d frontier nodes, %d threatened, %d threatened where a "
                        "quiet move ends the battle, %d capture lines, %d recaptured, %d "
                        "without a quiet move",
                        tally.frontier, tally.threatened, tally.threatened_ending,
                        tally.capture_lines, tally.recaptured, tally.no_quiet));
  Check(scan_differs, "4. the control: kScan with weights changes scores somewhere");
  Check(captures_differs, "4. the control: kCaptures at weight 0 changes scores somewhere");

  // 3. The root window with the threat term.
  {
    int exact = 0, bounded = 0, wrong = 0;
    for (Threat threat : {Threat::kScan, Threat::kCaptures}) {
      Tally unused;
      Reference reference(threat, {0.5, 0.5}, &unused);
      MaterialSearch search;
      search.SetThreat(threat, {0.5, 0.5});
      for (size_t i = 0; i < far; i += 2) {
        const std::vector<double> want = reference.Scores(*positions[i], 2);
        const std::vector<double> got = search.Scores(*positions[i], 2, /*margin=*/1);
        const double best = *std::max_element(want.begin(), want.end());
        for (size_t m = 0; m < want.size(); ++m) {
          if (want[m] > best - 1) {
            ++exact;
            wrong += got[m] != want[m];
          } else {
            ++bounded;
            wrong += !(got[m] <= best - 1 && got[m] >= want[m]);
          }
        }
      }
    }
    Check(wrong == 0 && bounded > 0,
          absl::StrFormat("3. the root window at margin 1: %d moves within a point of the best "
                          "exact, %d below it bounded, %d wrong", exact, bounded, wrong));
  }

  // 5. Depth 4 with the trolls to move.
  if (deep) {
    int moves = 0, wrong = 0;
    for (Threat threat : {Threat::kScan, Threat::kCaptures}) {
      Tally unused;
      Reference reference(threat, {0.5, 0.25}, &unused);
      MaterialSearch search(open_spiel::thud_az::Evaluation::kMaterial, true, true, 16);
      search.SetThreat(threat, {0.5, 0.25});
      int done = 0;
      for (size_t i = 0; i < far && done < 4; ++i) {
        if (positions[i]->CurrentPlayer() != thud::kTrollPlayer) continue;
        ++done;
        const std::vector<double> want = reference.Scores(*positions[i], 4);
        const std::vector<double> got = search.Scores(*positions[i], 4);
        for (size_t m = 0; m < want.size(); ++m) {
          ++moves;
          wrong += got[m] != want[m];
        }
      }
    }
    Check(wrong == 0, absl::StrFormat("5. depth 4, the trolls to move: %d of %d moves' scores "
                                      "equal the reference", moves - wrong, moves));
  }

  std::cout << (failures == 0 ? "threat_search_test: all passed"
                              : absl::StrCat("threat_search_test: ", failures, " FAILED"))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
