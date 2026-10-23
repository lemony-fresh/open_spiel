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

// Tests that AlphaBetaPlayer (thud/az/alphabeta_player.h: kickstarting's teacher and the
// alpha-beta opponent) computes exactly what it did before it was made faster
// (2026-10-23: material from the state's piece counts instead of a scan of the board, the
// tie-break's hash carried on from the root instead of over each leaf's whole history),
// against the code before, kept here verbatim but for its names. On positions of random
// games, uniform and capture-greedy (captures, fewer pieces, late positions):
//
//   1. the positions cover what the change touches: captures available, pieces taken off
//      on both sides (else the test proves little);
//   2. Scores at depths 1 and 2 equal the old ones exactly, every legal move; the control:
//      the old code with a troll worth 3 differs somewhere;
//   3. Choose at depths 1 and 2, for two seeds, picks the old move exactly; the control:
//      the two seeds pick differently somewhere (the tie-break decides moves, so the
//      comparison could fail);
//   4. reported, not tested: the time of each, old against new.
//
// Prints PASS or FAIL per part, then ALL PASSED or the count of failures; exits non-zero on
// a failure. `teacher_identity_test profile [DEPTH] [ROUNDS]` instead runs only the new
// Scores at DEPTH (default 2) on the positions, ROUNDS times over (default 5), for a
// profiler (perf record) or a timing, per side to move.

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/alphabeta_player.h"

namespace {

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
namespace thud = open_spiel::thud;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// The code before 2026-10-23 (alphabeta_player.h), verbatim but for its names and the
// troll's worth, a parameter for the control.
double OldMaterial(const State& state, Player me, int troll_value = thud::kTrollValue) {
  const auto& thud_state = open_spiel::down_cast<const thud::ThudState&>(state);
  int margin = 0;  // The dwarfs'.
  for (int row = 0; row < thud::kBoardSize; ++row) {
    for (int col = 0; col < thud::kBoardSize; ++col) {
      if (!thud::IsOnBoard(row, col)) continue;
      const thud::Cell cell = thud_state.CellAt(row, col);
      if (cell == thud::Cell::kDwarf) margin += thud::kDwarfValue;
      if (cell == thud::Cell::kTroll) margin -= troll_value;
    }
  }
  return static_cast<double>(me == thud::kDwarfPlayer ? margin : -margin) / thud::kMaxMargin;
}

double OldTieBreak(uint64_t seed, const State& leaf) {
  auto mix = [](uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  };
  uint64_t h = mix(seed);
  for (Action action : leaf.History()) h = mix(h ^ static_cast<uint64_t>(action));
  const double uniform = (h >> 11) * 0x1.0p-53;  // [0, 1)
  return (uniform - 0.5) * 0.5 / thud::kMaxMargin;  // [-1/4, 1/4) of a point
}

Action OldChoose(const Game& game, const State& state, int depth, uint64_t seed) {
  const Player me = state.CurrentPlayer();
  auto value = [me, seed](const State& leaf) {
    return OldMaterial(leaf, me) + OldTieBreak(seed, leaf);
  };
  return open_spiel::algorithms::AlphaBetaSearch(game, &state, value, depth, me,
                                                 /*use_undo=*/false)
      .second;
}

std::vector<double> OldScores(const Game& game, const State& state, int depth,
                              int troll_value = thud::kTrollValue) {
  const Player me = state.CurrentPlayer();
  std::vector<double> scores;
  for (Action action : state.LegalActions()) {
    std::unique_ptr<State> child = state.Child(action);
    double value;  // In returns, for `me`.
    if (child->IsTerminal() || depth <= 1) {
      value = child->IsTerminal() ? child->PlayerReturn(me) : OldMaterial(*child, me, troll_value);
    } else {
      const Player opponent = child->CurrentPlayer();
      auto material = [opponent, troll_value](const State& leaf) {
        return OldMaterial(leaf, opponent, troll_value);
      };
      value = -open_spiel::algorithms::AlphaBetaSearch(game, child.get(), material, depth - 1,
                                                       opponent, /*use_undo=*/false)
                   .first;
    }
    scores.push_back(value * thud::kMaxMargin);
  }
  return scores;
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("thud");
  const bool profile = argc > 1 && std::string(argv[1]) == "profile";

  // Positions: every 7th ply of random games, half of them capture-greedy (a capture,
  // written with an "x", whenever there is one).
  std::vector<std::unique_ptr<State>> positions;
  std::mt19937 rng(1);
  for (int g = 0; positions.size() < 120; ++g) {
    const bool greedy = g % 2 == 1;
    std::unique_ptr<State> s = game->NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 600; ++ply) {
      if (ply % 7 == 3) positions.push_back(s->Clone());
      const std::vector<Action> legal = s->LegalActions();
      std::vector<Action> captures;
      if (greedy) {
        for (Action a : legal) {
          if (absl::StrContains(s->ActionToString(s->CurrentPlayer(), a), "x")) {
            captures.push_back(a);
          }
        }
      }
      const std::vector<Action>& pool = captures.empty() ? legal : captures;
      s->ApplyAction(pool[rng() % pool.size()]);
    }
  }

  if (profile) {
    const int depth = argc > 2 ? std::stoi(argv[2]) : 2;
    const int rounds = argc > 3 ? std::stoi(argv[3]) : 5;
    const AlphaBetaPlayer teacher(*game, depth);
    double seconds[2] = {0, 0};
    int count[2] = {0, 0};
    for (int round = 0; round < rounds; ++round) {
      for (const auto& s : positions) {
        if (s->IsTerminal()) continue;
        const auto t0 = std::chrono::steady_clock::now();
        teacher.Scores(*s);
        seconds[s->CurrentPlayer()] += Seconds(t0);
        ++count[s->CurrentPlayer()];
      }
    }
    std::cout << absl::StrFormat("profile: depth %d, %d rounds: dwarfs %.2f ms a position (%d), "
                                 "trolls %.2f ms (%d)",
                                 depth, rounds, 1000 * seconds[0] / std::max(count[0], 1),
                                 count[0], 1000 * seconds[1] / std::max(count[1], 1), count[1])
              << std::endl;
    return 0;
  }

  // 1. Coverage.
  int with_capture = 0, dwarfs_taken = 0, trolls_taken = 0;
  for (const auto& s : positions) {
    const auto& t = open_spiel::down_cast<const thud::ThudState&>(*s);
    dwarfs_taken += t.NumDwarfs() < thud::kNumDwarfs;
    trolls_taken += t.NumTrolls() < thud::kNumTrolls;
    for (Action a : s->LegalActions()) {
      if (absl::StrContains(s->ActionToString(s->CurrentPlayer(), a), "x")) {
        ++with_capture;
        break;
      }
    }
  }
  Check(with_capture > 10 && dwarfs_taken > 10 && trolls_taken > 10,
        absl::StrFormat("1. %d positions: %d with a capture available, %d with dwarfs taken, "
                        "%d with trolls taken (more than 10 each required)",
                        positions.size(), with_capture, dwarfs_taken, trolls_taken));

  // 2. Scores.
  double old_seconds = 0, new_seconds = 0;
  for (int depth : {1, 2}) {
    const AlphaBetaPlayer teacher(*game, depth);
    int moves = 0, differ = 0, control_differ = 0;
    for (const auto& s : positions) {
      auto t0 = std::chrono::steady_clock::now();
      const std::vector<double> old_scores = OldScores(*game, *s, depth);
      if (depth == 2) old_seconds += Seconds(t0);
      t0 = std::chrono::steady_clock::now();
      const std::vector<double> new_scores = teacher.Scores(*s);
      if (depth == 2) new_seconds += Seconds(t0);
      if (new_scores.size() != old_scores.size()) {
        ++differ;
        continue;
      }
      for (size_t i = 0; i < new_scores.size(); ++i) differ += new_scores[i] != old_scores[i];
      moves += new_scores.size();
      if (depth == 1) {
        const std::vector<double> control = OldScores(*game, *s, depth, /*troll_value=*/3);
        for (size_t i = 0; i < control.size(); ++i) control_differ += control[i] != new_scores[i];
      }
    }
    Check(differ == 0 && (depth != 1 || control_differ > 0),
          absl::StrFormat("2. Scores at depth %d: %d of %d moves' scores differ from the old "
                          "code's (0 required)%s",
                          depth, differ, moves,
                          depth == 1 ? absl::StrFormat("; the control, a troll worth 3: %d "
                                                       "differ (more than 0 required)",
                                                       control_differ)
                                     : std::string()));
  }
  std::cout << absl::StrFormat("   Scores at depth 2: %.1f ms a position before, %.1f now "
                               "(%.1fx)",
                               1000 * old_seconds / positions.size(),
                               1000 * new_seconds / positions.size(), old_seconds / new_seconds)
            << std::endl;

  // 3. Choose.
  double old_choose = 0, new_choose = 0;
  for (int depth : {1, 2}) {
    int differ = 0, seeds_differ = 0;
    std::vector<Action> first_seed;
    for (uint64_t seed : {11, 12}) {
      AlphaBetaPlayer player(*game, depth);
      player.NewGame(seed);
      for (size_t i = 0; i < positions.size(); ++i) {
        auto t0 = std::chrono::steady_clock::now();
        const Action old_move = OldChoose(*game, *positions[i], depth, seed);
        if (depth == 2) old_choose += Seconds(t0);
        t0 = std::chrono::steady_clock::now();
        const Action new_move = player.Choose(*positions[i]);
        if (depth == 2) new_choose += Seconds(t0);
        differ += new_move != old_move;
        if (seed == 11) {
          first_seed.push_back(new_move);
        } else {
          seeds_differ += new_move != first_seed[i];
        }
      }
    }
    Check(differ == 0 && seeds_differ > 0,
          absl::StrFormat("3. Choose at depth %d, two seeds: %d of %d moves differ from the old "
                          "code's (0 required); the control, the two seeds' moves differ in "
                          "%d positions (more than 0 required)",
                          depth, differ, 2 * positions.size(), seeds_differ));
  }
  std::cout << absl::StrFormat("   Choose at depth 2: %.1f ms a position before, %.1f now (%.1fx)",
                               1000 * old_choose / (2 * positions.size()),
                               1000 * new_choose / (2 * positions.size()),
                               old_choose / new_choose)
            << std::endl;

  if (failures == 0) {
    std::cout << "ALL PASSED" << std::endl;
    return 0;
  }
  std::cout << failures << " FAILED" << std::endl;
  return 1;
}
