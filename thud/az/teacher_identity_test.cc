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
//   4. reported, not tested: the time of each, old against new;
//   5. (2026-10-23) Scores by MaterialSearch: at depth 3 on the first 30 positions equal to
//      OpenSpiel's alpha-beta's exactly (on the piece counts, by undo: ReferenceScores), and every combination of its move ordering (captures first,
//      killer moves, both, neither) giving the same scores on them — the order changes only
//      what is pruned; with `deep`, also depth 4 on the first 8 positions (slow: the old
//      code takes seconds a position there); and the battle ending inside the search: 12
//      positions copied with the no-capture counter 1, 2 and 3 turns short of its limit and
//      the turn counter 1 and 2 short of its, at depth 3 (and 4 with `deep`) — every one of
//      those searches reaching the limits checked to end a battle (the control that they do).
//   6. (2026-10-23) MaterialSearch's transposition table: (a) at depth 3 on the first 30
//      positions (and 4 on 8 with `deep`) equal to OpenSpiel's alpha-beta with a fresh table a
//      position and with one table shared by all, of 2^16 entries and of 16 (slots shared and
//      overwritten all the time), and the table ending some nodes (the control that it is
//      used); (b) depths 4, 3, 2, 3, 4 in turn on one table, 6 positions, equal to the search
//      without one (the control: depth 4's scores differ from depth 3's somewhere, so a deeper
//      entry used in a shallower search would show); (c) each of 8 positions searched with
//      its own counters, then copied 1-3 turns short of the no-capture limit and 1-2 short of
//      the turn limit, on one table, in the training's game (no-capture limit 100; with
//      `deep` the default game too), equal to OpenSpiel's alpha-beta (the control: near the
//      limits the scores differ from the far copy's somewhere, so a key without the counters
//      would reuse wrong entries); (d) with `deep`, depth 5 on 6 positions, where
//      transpositions meet inside a search, the table against none (the control: the table
//      ends nodes there). The sides alternate, so
//      a transposition needs two moves by one side, 3 plies down: inside one search the table
//      meets them only from depth 5 on; at depths 3 and 4 its values end nodes only when a
//      position comes again (a: each searched twice). (e) The table's bounds: 20 positions
//      each searched with 40 random windows on one table, every result checked by the
//      fail-soft rule against the exact value (the control: windows below, above and around
//      it).
//   7. The golden fingerprint: the scores at depths 3 and 4, recorded with the code before the
//      game's speed-ups (2026-10-27).
//   8. (2026-10-27) The root window (MaterialSearch::Scores' margin): at margins 1-3 every move
//      less than the margin below the best exact and the others bounded as promised, the best
//      moves the same; the controls: some bounds differ from the scores, and each margin breaks
//      the next one's promise.
//   The parts above call AlphaBetaPlayer::Scores, so its table (one a thread, kept from call
//   to call) is under test there too. In these builds (assertions on) MaterialSearch also
//   checks every 16th frontier node against a full search of it (material_search.h).
//
// Prints PASS or FAIL per part, then ALL PASSED or the count of failures; exits non-zero on
// a failure. `teacher_identity_test profile [DEPTH] [ROUNDS]` instead runs only the new
// Scores at DEPTH (default 2) on the positions, ROUNDS times over (default 5), for a
// profiler (perf record) or a timing, per side to move, and then each of MaterialSearch's
// orderings once, without and with the table: its time and nodes a position.

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/material_search.h"

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

// The reference for depths 3 and 4: OpenSpiel's alpha-beta as the teacher used it before
// MaterialSearch — the material from the piece counts, the search by undo (both shown equal
// to the code before at depths 1 and 2, part 2), the moves in the game's order.
std::vector<double> ReferenceScores(const Game& game, const State& state, int depth) {
  const Player me = state.CurrentPlayer();
  auto material = [](const State& s, Player p) {
    const auto& t = open_spiel::down_cast<const thud::ThudState&>(s);
    const int margin = t.NumDwarfs() * thud::kDwarfValue - t.NumTrolls() * thud::kTrollValue;
    return static_cast<double>(p == thud::kDwarfPlayer ? margin : -margin) / thud::kMaxMargin;
  };
  std::vector<double> scores;
  for (Action action : state.LegalActions()) {
    std::unique_ptr<State> child = state.Child(action);
    double value;
    if (child->IsTerminal() || depth <= 1) {
      value = child->IsTerminal() ? child->PlayerReturn(me) : material(*child, me);
    } else {
      const Player opponent = child->CurrentPlayer();
      auto leaf = [opponent, &material](const State& s) { return material(s, opponent); };
      value = -open_spiel::algorithms::AlphaBetaSearch(game, child.get(), leaf, depth - 1,
                                                       opponent, /*use_undo=*/true)
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
  const bool deep = argc > 1 && std::string(argv[1]) == "deep";

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
    double seconds[2] = {0, 0};
    int count[2] = {0, 0};
    for (int round = 0; round < rounds; ++round) {
      // A fresh table each round: the rounds repeat the positions.
      open_spiel::thud_az::MaterialSearch teacher(
          open_spiel::thud_az::Evaluation::kMaterial, true, true,
          open_spiel::thud_az::MaterialSearch::kTeacherTableBits);
      for (const auto& s : positions) {
        if (s->IsTerminal()) continue;
        const auto t0 = std::chrono::steady_clock::now();
        teacher.Scores(*s, depth);
        seconds[s->CurrentPlayer()] += Seconds(t0);
        ++count[s->CurrentPlayer()];
      }
    }
    std::cout << absl::StrFormat("profile: depth %d, %d rounds: dwarfs %.2f ms a position (%d), "
                                 "trolls %.2f ms (%d)",
                                 depth, rounds, 1000 * seconds[0] / std::max(count[0], 1),
                                 count[0], 1000 * seconds[1] / std::max(count[1], 1), count[1])
              << std::endl;
    for (const auto& [captures, killers, bits] :
         std::vector<std::tuple<bool, bool, int>>{
             {false, false, 0}, {true, false, 0}, {false, true, 0}, {true, true, 0},
             {true, true, open_spiel::thud_az::MaterialSearch::kTeacherTableBits}}) {
      open_spiel::thud_az::MaterialSearch search(open_spiel::thud_az::Evaluation::kMaterial,
                                                 captures, killers, bits);
      double t[2] = {0, 0};
      int64_t nodes[2] = {0, 0};
      int n[2] = {0, 0};
      for (const auto& s : positions) {
        if (s->IsTerminal()) continue;
        const Player p = s->CurrentPlayer();
        search.ResetNodes();
        const auto t0 = std::chrono::steady_clock::now();
        search.Scores(*s, depth);
        t[p] += Seconds(t0);
        nodes[p] += search.nodes();
        ++n[p];
      }
      std::cout << absl::StrFormat("  ordering: captures first %d, killers %d, table 2^%d: "
                                   "dwarfs %.2f ms, %d nodes a position; trolls %.2f ms, %d nodes",
                                   captures, killers, bits, 1000 * t[0] / std::max(n[0], 1),
                                   nodes[0] / std::max(n[0], 1), 1000 * t[1] / std::max(n[1], 1),
                                   nodes[1] / std::max(n[1], 1))
                << std::endl;
    }
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

  // 5. MaterialSearch at depth 3 (and 4 with `deep`), and its orderings.
  std::vector<std::vector<double>> references[5];  // By depth, for part 6a.
  for (int depth : deep ? std::vector<int>{3, 4} : std::vector<int>{3}) {
    const size_t n = std::min<size_t>(positions.size(), depth == 3 ? 30 : 8);
    int moves = 0, differ = 0, ordering_differ = 0;
    for (size_t i = 0; i < n; ++i) {
      const std::vector<double> old_scores = ReferenceScores(*game, *positions[i], depth);
      references[depth].push_back(old_scores);
      const std::vector<double> new_scores = AlphaBetaPlayer(*game, depth).Scores(*positions[i]);
      if (old_scores.size() != new_scores.size()) {
        ++differ;
        continue;
      }
      for (size_t j = 0; j < new_scores.size(); ++j) differ += new_scores[j] != old_scores[j];
      moves += new_scores.size();
      for (const auto& [captures, killers] :
           std::vector<std::pair<bool, bool>>{{false, false}, {true, false}, {false, true}}) {
        const std::vector<double> other =
            open_spiel::thud_az::MaterialSearch(open_spiel::thud_az::Evaluation::kMaterial,
                                                captures, killers)
                .Scores(*positions[i], depth);
        ordering_differ += other != new_scores;
      }
    }
    Check(differ == 0 && ordering_differ == 0 && moves > 0,
          absl::StrFormat("5. MaterialSearch at depth %d on %d positions: %d of %d moves' scores "
                          "differ from OpenSpiel's alpha-beta's (0 required); %d orderings' scores "
                          "differ from the default's (0 required)",
                          depth, n, differ, moves, ordering_differ));
  }

  // 5. The battle ending inside the search: the no-capture and turn limits.
  {
    const int limit = 200, max_turns = 800;  // thud's defaults (THUD_RULES.md section 6).
    std::vector<std::unique_ptr<State>> near;
    for (size_t i = 0; near.size() < 60 && i < positions.size(); ++i) {
      const auto& t = open_spiel::down_cast<const thud::ThudState&>(*positions[i]);
      if (positions[i]->IsTerminal()) continue;
      const std::string text = positions[i]->ToString();
      const std::string rows = text.substr(0, text.rfind('\n'));
      const std::string side = positions[i]->CurrentPlayer() == thud::kDwarfPlayer ? "dwarfs" : "trolls";
      for (const auto& [turns, quiet] : std::vector<std::pair<int, int>>{
               {300, limit - 1}, {300, limit - 2}, {300, limit - 3},
               {max_turns - 1, 50}, {max_turns - 2, 50}}) {
        if (quiet < t.TurnsWithoutCapture()) continue;
        near.push_back(game->NewInitialState(absl::StrFormat(
            "%s\nto_move=%s turns=%d turns_without_capture=%d", rows, side, turns, quiet)));
      }
    }
    for (int depth : deep ? std::vector<int>{3, 4} : std::vector<int>{3}) {
      const size_t n = std::min<size_t>(near.size(), depth == 3 ? 60 : 10);
      int moves = 0, differ = 0, ended = 0;
      for (size_t i = 0; i < n; ++i) {
        const std::vector<double> want = ReferenceScores(*game, *near[i], depth);
        const std::vector<double> got = AlphaBetaPlayer(*game, depth).Scores(*near[i]);
        differ += want != got;
        moves += got.size();
        // The control: a quiet line of `depth` moves from here ends the battle.
        std::unique_ptr<State> s = near[i]->Clone();
        for (int k = 0; k < depth && !s->IsTerminal(); ++k) {
          for (Action a : s->LegalActions()) {
            if (open_spiel::down_cast<const thud::ThudState&>(*s).Captures(a) == 0) {
              s->ApplyAction(a);
              break;
            }
          }
        }
        ended += s->IsTerminal();
      }
      Check(differ == 0 && ended == static_cast<int>(n) && n >= 10,
            absl::StrFormat("5. at the limits, depth %d: %d of %d positions' scores (%d moves) "
                            "differ from OpenSpiel's alpha-beta's (0 required); in %d of them a "
                            "quiet line ends the battle inside the search (all required)",
                            depth, differ, n, moves, ended));
    }
  }

  // 6. The transposition table (2026-10-23).
  {
    using open_spiel::thud_az::Evaluation;
    using open_spiel::thud_az::MaterialSearch;
    auto plain = [](const State& s, int depth) {  // No table: equal to the reference (part 5).
      return MaterialSearch(Evaluation::kMaterial).Scores(s, depth);
    };
    auto table = [](int bits) { return MaterialSearch(Evaluation::kMaterial, true, true, bits); };
    // 6a. Against the reference: a fresh table a position, and one shared by all of them in
    // turn, each searched twice, of 2^16 entries and of 2^4 (16: slots shared all the time,
    // entries overwritten). The sides alternate, so a transposition needs two moves by one
    // side — 3 plies down — and the table, above the frontier, meets one inside a search only
    // from depth 5 on (6d); the repeat is where its values end nodes here (the control: they
    // do, with 2^16 entries).
    for (int depth : deep ? std::vector<int>{3, 4} : std::vector<int>{3}) {
      const std::vector<std::vector<double>>& want = references[depth];  // Part 5's.
      const size_t n = want.size();
      for (int bits : {16, 4}) {
        MaterialSearch shared = table(bits);
        int differ_fresh = 0, differ_shared = 0;
        for (size_t i = 0; i < n; ++i) {
          MaterialSearch fresh = table(bits);
          differ_fresh += fresh.Scores(*positions[i], depth) != want[i];
          for (int repeat = 0; repeat < 2; ++repeat) {
            differ_shared += shared.Scores(*positions[i], depth) != want[i];
          }
        }
        Check(differ_fresh == 0 && differ_shared == 0 && (bits < 16 || shared.table_cutoffs() > 0),
              absl::StrFormat("6a. the table (2^%d entries) at depth %d on %d positions: %d "
                              "positions' scores differ from OpenSpiel's alpha-beta's with a "
                              "fresh table each, %d of %d with one table for all, each searched "
                              "twice (0 required); %d nodes ended by the shared table (more than "
                              "0 required with 2^16 entries)",
                              bits, depth, n, differ_fresh, differ_shared, 2 * n,
                              shared.table_cutoffs()));
      }
    }
    // 6b. Depths mixed on one table: 4, 3, 2, 3, 4 on each position in turn, each equal to the
    // search without a table — only an entry of the node's own depth may end it. The control:
    // the depths' scores differ somewhere, so a deeper entry used in a shallower search shows.
    {
      MaterialSearch shared = table(16);
      int differ = 0, depths_differ = 0;
      const size_t n = std::min<size_t>(positions.size(), 6);
      for (size_t i = 0; i < n; ++i) {
        std::vector<double> by_depth[5];
        for (int depth : {2, 3, 4}) by_depth[depth] = plain(*positions[i], depth);
        for (int depth : {4, 3, 2, 3, 4}) differ += shared.Scores(*positions[i], depth) != by_depth[depth];
        depths_differ += by_depth[4] != by_depth[3];
      }
      Check(differ == 0 && depths_differ > 0,
            absl::StrFormat("6b. depths 4, 3, 2, 3, 4 on one table, %d positions: %d searches' "
                            "scores differ from the search's without a table (0 required); the "
                            "control, depth 4's scores differ from depth 3's in %d positions "
                            "(more than 0 required)",
                            n, differ, depths_differ));
    }
    // 6c. The limits: each position first searched with its own counters (far from the limits),
    // then copied near them — as part 5 — on the same table, in the training's game (the
    // no-capture limit 100; with `deep` the default game too — part 5's limits run on
    // AlphaBetaPlayer's table after the same boards far from them); every search equal to the
    // reference. The control: near a limit the scores differ from the far copy's in some (so a
    // key without the counters would reuse wrong entries).
    for (const char* game_string :
         deep ? std::vector<const char*>{"thud(max_turns_without_capture=100)", "thud"}
              : std::vector<const char*>{"thud(max_turns_without_capture=100)"}) {
      std::shared_ptr<const Game> g = open_spiel::LoadGame(game_string);
      const int limit = open_spiel::down_cast<const thud::ThudGame&>(*g).max_turns_without_capture();
      const int max_turns = open_spiel::down_cast<const thud::ThudGame&>(*g).max_turns();
      MaterialSearch shared = table(16);
      int searches = 0, differ = 0, far_differ = 0;
      for (size_t i = 0, used = 0; used < 8 && i < positions.size(); ++i) {
        const auto& t = open_spiel::down_cast<const thud::ThudState&>(*positions[i]);
        if (positions[i]->IsTerminal() || t.TurnsWithoutCapture() > limit - 4) continue;
        ++used;
        const std::string text = positions[i]->ToString();
        const std::string rows = text.substr(0, text.rfind('\n'));
        const std::string side =
            positions[i]->CurrentPlayer() == thud::kDwarfPlayer ? "dwarfs" : "trolls";
        auto at = [&](int turns, int quiet) {
          return g->NewInitialState(absl::StrFormat(
              "%s\nto_move=%s turns=%d turns_without_capture=%d", rows, side, turns, quiet));
        };
        std::unique_ptr<State> far = at(t.TurnsPlayed(), t.TurnsWithoutCapture());
        const std::vector<double> far_scores = ReferenceScores(*g, *far, 3);
        differ += shared.Scores(*far, 3) != far_scores;
        ++searches;
        for (const auto& [turns, quiet] : std::vector<std::pair<int, int>>{
                 {300, limit - 1}, {300, limit - 2}, {300, limit - 3},
                 {max_turns - 1, 50}, {max_turns - 2, 50}}) {
          if (quiet < t.TurnsWithoutCapture()) continue;
          std::unique_ptr<State> near_state = at(turns, quiet);
          const std::vector<double> want = ReferenceScores(*g, *near_state, 3);
          differ += shared.Scores(*near_state, 3) != want;
          far_differ += want != far_scores;
          ++searches;
        }
      }
      Check(differ == 0 && far_differ > 0 && searches >= 30,
            absl::StrFormat("6c. the limits in %s, depth 3, one table: %d of %d searches' scores "
                            "differ from OpenSpiel's alpha-beta's (0 required); the control, near "
                            "the limits %d differ from their far copy's (more than 0 required)",
                            game_string, differ, searches, far_differ));
    }
    // 6d. With `deep`: depth 5, where the table meets transpositions inside a search (their
    // bounds under other windows), on 6 positions, the table against none; the control: the
    // table ends nodes there.
    if (deep) {
      MaterialSearch shared = table(MaterialSearch::kTeacherTableBits);
      int differ = 0;
      const size_t n = 6;
      for (size_t i = 0; i < n; ++i) differ += shared.Scores(*positions[i], 5) != plain(*positions[i], 5);
      Check(differ == 0 && shared.table_cutoffs() > 0,
            absl::StrFormat("6d. depth 5 on %d positions: %d positions' scores differ with the "
                            "table from without it (0 required); %d nodes ended by the table "
                            "(more than 0 required)",
                            n, differ, shared.table_cutoffs()));
    }
  }

  // 7. The golden fingerprint of the teacher (2026-10-27, before the speed-ups of the game's
  // rules): the scores of every legal move of the first 40 positions at depth 3 and of the
  // first 6 at depth 4, folded into one number, recorded with the code before them — so that
  // a change to the rules' results or to the order of the moves fails here even where the
  // reference (OpenSpiel's alpha-beta on the same game) would change alike.
  {
    constexpr uint64_t kGoldenScores = 18313312276673694004ULL;
    uint64_t h = 0;
    auto fold = [&h](uint64_t x) {
      uint64_t z = (h ^ x) + 0x9e3779b97f4a7c15ULL;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      h = z ^ (z >> 31);
    };
    int scored = 0;
    for (int depth : {3, 4}) {
      const size_t n = std::min<size_t>(positions.size(), depth == 3 ? 40 : 6);
      for (size_t i = 0; i < n; ++i) {
        for (Action a : positions[i]->LegalActions()) fold(a);
        for (double v : open_spiel::thud_az::MaterialSearch().Scores(*positions[i], depth)) {
          fold(static_cast<uint64_t>(static_cast<int64_t>(std::llround(v))));
          ++scored;
        }
      }
    }
    Check(h == kGoldenScores && scored > 5000,
          absl::StrFormat("7. the teacher's golden fingerprint (depths 3 and 4, %d scores): %d "
                          "(%d required)",
                          scored, h, kGoldenScores));
  }

  // 6e. The table's bounds: 20 positions, each searched at depth 3 with 40 random windows
  // around its value on one table (the position itself and its children then stored as
  // exact values or bounds under one window, probed under the next), each result checked
  // against the exact value by the fail-soft rule — at most alpha: the true value at most
  // it; at least beta: at least it; in between: it. The control: the windows fall on both
  // sides of the value and around it.
  {
    using open_spiel::thud_az::Evaluation;
    using open_spiel::thud_az::MaterialSearch;
    MaterialSearch shared(Evaluation::kMaterial, true, true, 16);
    std::mt19937 windows(5);
    int searches = 0, wrong = 0, below = 0, above = 0, inside = 0;
    for (size_t i = 0, used = 0; used < 20 && i < positions.size(); ++i) {
      if (positions[i]->IsTerminal()) continue;
      ++used;
      const int exact = MaterialSearch(Evaluation::kMaterial).Value(*positions[i], 3, -1000, 1000);
      for (int w = 0; w < 40; ++w) {
        const int a = exact - 6 + static_cast<int>(windows() % 13);
        const int b = a + 1 + static_cast<int>(windows() % 6);
        const int v = shared.Value(*positions[i], 3, a, b);
        const bool ok = v <= a ? exact <= v : v >= b ? exact >= v : exact == v;
        wrong += !ok;
        below += exact <= a;
        above += exact >= b;
        inside += a < exact && exact < b;
        ++searches;
      }
    }
    Check(wrong == 0 && below > 0 && above > 0 && inside > 0 && shared.table_cutoffs() > 0,
          absl::StrFormat("6e. the table's bounds: %d of %d searches with random windows break the "
                          "fail-soft rule against the exact value (0 required); the control, the "
                          "value below the window %d times, above it %d, inside it %d, and %d nodes "
                          "ended by the table (more than 0 each required)",
                          wrong, searches, below, above, inside, shared.table_cutoffs()));
  }

  // 8. The root window (2026-10-27; material_search.h): margins 1, 2 and 3 at depths 2 and 3
  // on the first 30 positions and at depth 4 on the first 8, on one shared table as the
  // teacher's, against the exact scores — every move less than the margin below the best
  // exact, every other a bound between its score and best - margin, so the moves tied for the
  // best the same (the teacher's choice as a player, ChooseByScores at margin 1, depends on
  // nothing else). The controls: some moves get a bound that is not their score (the window
  // does something), and each margin's scores break the promise of the next margin up (the
  // check can fail).
  {
    using open_spiel::thud_az::Evaluation;
    using open_spiel::thud_az::MaterialSearch;
    MaterialSearch windowed(Evaluation::kMaterial, true, true, 16);
    // The promise of `margin`, for `got` against the exact scores `want`: the breaks.
    auto breaks = [](const std::vector<double>& got, const std::vector<double>& want,
                     int margin) {
      const double best = *std::max_element(want.begin(), want.end());
      int n = 0;
      for (size_t k = 0; k < want.size(); ++k) {
        n += want[k] > best - margin ? got[k] != want[k]
                                     : got[k] < want[k] || got[k] > best - margin;
      }
      return n;
    };
    int moves = 0, wrong = 0, bounded = 0, best_differs = 0, control = 0;
    for (int depth : {2, 3, 4}) {
      const size_t n = std::min<size_t>(positions.size(), depth == 4 ? 8 : 30);
      for (size_t i = 0; i < n; ++i) {
        if (positions[i]->IsTerminal()) continue;
        const std::vector<double> want = MaterialSearch(Evaluation::kMaterial).Scores(*positions[i], depth);
        const double best = *std::max_element(want.begin(), want.end());
        for (int margin : {1, 2, 3}) {
          const std::vector<double> got = windowed.Scores(*positions[i], depth, margin);
          moves += got.size();
          wrong += breaks(got, want, margin);
          control += breaks(got, want, margin + 1);
          const double got_best = *std::max_element(got.begin(), got.end());
          for (size_t k = 0; k < got.size(); ++k) {
            bounded += got[k] != want[k];
            best_differs += (got[k] == got_best) != (want[k] == best);
          }
        }
      }
    }
    Check(wrong == 0 && best_differs == 0 && bounded > 0 && control > 0,
          absl::StrFormat("8. the root window: %d of %d scores break its promise and %d differ in "
                          "being among the best (0 each required); the controls, %d scores a "
                          "bound and %d breaks of the next margin's promise (more than 0 each "
                          "required)",
                          wrong, moves, best_differs, bounded, control));
  }

  if (failures == 0) {
    std::cout << "ALL PASSED" << std::endl;
    return 0;
  }
  std::cout << failures << " FAILED" << std::endl;
  return 1;
}
