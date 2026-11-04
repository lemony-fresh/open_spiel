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

// Tests thud/az/teacher_terms.h, the teacher's positional terms, on the hand-built positions
// of thud/experiments/teacher_terms_test.py (their values worked out by hand from the
// definitions): two dwarfs side by side (reach 1), a line of three (2), a line of four at the
// Thudstone (1), a troll on a hurl square (1), a line of three trolls (2 pairs) and a dwarf
// in its shove path (still 2), a diagonal pair on the cut corner (1), two lines reaching one
// square (1), the starting position (reach 0, 12 troll pairs) — each with the dwarfs and with
// the trolls to move (the terms read the board alone) — and PositionalScore's weighting.
//
//   teacher_terms_test               the tests
//   teacher_terms_test dump N SEED   N positions of random games, one a line: the board's
//                                    text and both terms, for teacher_terms_cross_test.py

#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/teacher_terms.h"

namespace {

namespace thud = open_spiel::thud;
using open_spiel::thud_az::ComputePositionalTerms;
using open_spiel::thud_az::PositionalScore;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// A board with `pieces` and every other square empty, as ThudState's text.
std::string Board(const std::map<std::pair<int, int>, char>& pieces) {
  std::string text;
  for (int r = 0; r < thud::kBoardSize; ++r) {
    for (int c = 0; c < thud::kBoardSize; ++c) {
      auto it = pieces.find({r, c});
      text += !thud::IsOnBoard(r, c)   ? '-'
              : r == 7 && c == 7       ? 'O'
              : it != pieces.end()     ? it->second
                                       : '.';
    }
    text += '\n';
  }
  return text;
}

void Expect(const std::string& name, const std::string& board, int reach, int pairs) {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  for (const char* side : {"dwarfs", "trolls"}) {
    std::unique_ptr<open_spiel::State> state =
        game->NewInitialState(absl::StrCat(board, "to_move=", side));
    const auto terms =
        ComputePositionalTerms(open_spiel::down_cast<const thud::ThudState&>(*state));
    Check(terms.dwarf_hurl_reach == reach && terms.troll_cohesion == pairs,
          absl::StrCat(name, " (", side, " to move): reach ", terms.dwarf_hurl_reach, " (want ",
                       reach, "), troll pairs ", terms.troll_cohesion, " (want ", pairs, ")"));
  }
}

void Dump(int n, int seed) {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::mt19937 rng(seed);
  int written = 0;
  while (written < n) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    const bool greedy = rng() % 2;
    for (int ply = 0; !state->IsTerminal() && ply < 400 && written < n; ++ply) {
      const auto& s = open_spiel::down_cast<const thud::ThudState&>(*state);
      if (ply % 7 == 3) {
        const auto terms = ComputePositionalTerms(s);
        std::cout << absl::StrReplaceAll(s.ToString(), {{"\n", "|"}}) << "\t"
                  << terms.dwarf_hurl_reach << "\t" << terms.troll_cohesion << "\n";
        ++written;
      }
      std::vector<open_spiel::Action> legal = s.LegalActions(), captures;
      for (auto a : legal) {
        if (greedy && s.Captures(a) > 0) captures.push_back(a);
      }
      const auto& pool = captures.empty() ? legal : captures;
      state->ApplyAction(pool[rng() % pool.size()]);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "dump") {
    Dump(std::stoi(argv[2]), std::stoi(argv[3]));
    return 0;
  }
  const char d = 'd', t = 'T';
  Expect("1. two dwarfs side by side", Board({{{5, 1}, d}, {{5, 2}, d}}), 1, 0);
  Expect("2. a line of three", Board({{{7, 1}, d}, {{7, 2}, d}, {{7, 3}, d}}), 2, 0);
  Expect("3. a line of four at the Thudstone",
         Board({{{7, 1}, d}, {{7, 2}, d}, {{7, 3}, d}, {{7, 4}, d}}), 1, 0);
  Expect("4. a troll on a hurl square",
         Board({{{5, 1}, d}, {{5, 2}, d}, {{5, 3}, d}, {{5, 5}, t}}), 1, 0);
  Expect("5. a line of three trolls", Board({{{10, 5}, t}, {{10, 6}, t}, {{10, 7}, t}}), 0, 2);
  Expect("5. and a dwarf in its shove path",
         Board({{{10, 5}, t}, {{10, 6}, t}, {{10, 7}, t}, {{10, 9}, d}}), 0, 2);
  Expect("6. a diagonal pair on the cut corner", Board({{{1, 4}, d}, {{2, 3}, d}}), 1, 0);
  Expect("7. two lines reaching one square",
         Board({{{5, 1}, d}, {{5, 2}, d}, {{2, 4}, d}, {{3, 4}, d}}), 1, 0);
  {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
    std::unique_ptr<open_spiel::State> start = game->NewInitialState();
    const auto terms =
        ComputePositionalTerms(open_spiel::down_cast<const thud::ThudState&>(*start));
    Check(terms.dwarf_hurl_reach == 0 && terms.troll_cohesion == 12,
          absl::StrCat("8. the starting position: reach ", terms.dwarf_hurl_reach,
                       " (want 0), troll pairs ", terms.troll_cohesion, " (want 12)"));
    Check(PositionalScore(terms, {0.5, -0.25}) == -3.0 &&
              PositionalScore({7, 2}, {0.5, -0.25}) == 3.0,
          "9. PositionalScore: the weights, reach first");
  }
  std::cout << (failures == 0 ? "teacher_terms_test: all passed"
                              : absl::StrCat("teacher_terms_test: ", failures, " FAILED"))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
