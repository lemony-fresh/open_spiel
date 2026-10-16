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

// Tests the value probe's position edits (thud/experiments/position_edits.h, used by
// az_dwarf_value_probe for measurement 4 of thud/DESIGN_dwarf_signal.md), on positions of
// random games (every 5th ply, both sides) and hand-built ones, each against a
// computation of its own here (the board read from the text format):
//
//   1. FarPieces: exactly the pieces of a kind at least `far` king's moves from every
//      piece of the other (for far 2, 4 and 6, both kinds); none when the kind has one
//      piece left.
//   2. Without: the board differs in exactly that square, now empty; the status line
//      (who moves, the counters) unchanged.
//   3. Shifted: the dwarf moved to a neighbouring empty square at least `far` from every
//      troll — the board differs in exactly those two squares, the status line unchanged;
//      and null exactly when no neighbouring square qualifies (confirmed here square by
//      square); by hand, the opening's dwarf at (0,5) moves at far 1 and not at far 13
//      (every square of the board is nearer a troll).
//   Control: an edit applied to the wrong square (Without at another piece) must differ,
//   so 2 can fail.
//
//   thud/run_tests.sh position_edits_test   # builds and runs it
//   build-tests/bin/position_edits_test

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/experiments/position_edits.h"

namespace {

namespace thud = open_spiel::thud;
namespace edits = open_spiel::thud_az::position_edits;
using Square = std::pair<int, int>;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// The board's 15 rows and the status line, from the text format.
struct Board {
  std::vector<std::string> rows;
  std::string status;
};

Board Read(const open_spiel::State& state) {
  Board b;
  const std::string text = state.ToString();
  size_t start = 0;
  for (int r = 0; r < thud::kBoardSize; ++r) {
    b.rows.push_back(text.substr(start, thud::kBoardSize));
    start += thud::kBoardSize + 1;
  }
  b.status = text.substr(start);
  return b;
}

std::vector<Square> Differences(const Board& a, const Board& b) {
  std::vector<Square> out;
  for (int r = 0; r < thud::kBoardSize; ++r) {
    for (int c = 0; c < thud::kBoardSize; ++c) {
      if (a.rows[r][c] != b.rows[r][c]) out.push_back({r, c});
    }
  }
  return out;
}

std::vector<Square> Pieces(const Board& b, char kind) {
  std::vector<Square> out;
  for (int r = 0; r < thud::kBoardSize; ++r) {
    for (int c = 0; c < thud::kBoardSize; ++c) {
      if (b.rows[r][c] == kind) out.push_back({r, c});
    }
  }
  return out;
}

int Distance(Square a, Square b) {
  return std::max(std::abs(a.first - b.first), std::abs(a.second - b.second));
}

std::set<Square> FarByHand(const Board& b, char kind, char other, int far) {
  std::set<Square> out;
  const auto mine = Pieces(b, kind), theirs = Pieces(b, other);
  if (mine.size() < 2) return out;
  for (Square m : mine) {
    bool ok = true;
    for (Square t : theirs) ok = ok && Distance(m, t) >= far;
    if (ok) out.insert(m);
  }
  return out;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::mt19937 rng(3);
  std::vector<std::unique_ptr<open_spiel::State>> positions;
  for (int g = 0; g < 12; ++g) {
    std::unique_ptr<open_spiel::State> s = game->NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 300; ++ply) {
      if (ply % 5 == 2) positions.push_back(s->Clone());
      const auto legal = s->LegalActions();
      s->ApplyAction(legal[rng() % legal.size()]);
    }
  }

  // 1. FarPieces.
  bool far_ok = true;
  int far_cases = 0, far_found = 0;
  for (const auto& p : positions) {
    const auto& t = open_spiel::down_cast<const thud::ThudState&>(*p);
    const Board b = Read(*p);
    for (int far : {2, 4, 6}) {
      for (bool dwarfs : {true, false}) {
        const auto got = edits::FarPieces(t, dwarfs ? thud::Cell::kDwarf : thud::Cell::kTroll,
                                          dwarfs ? thud::Cell::kTroll : thud::Cell::kDwarf, far);
        const std::set<Square> want = FarByHand(b, dwarfs ? 'd' : 'T', dwarfs ? 'T' : 'd', far);
        far_ok = far_ok && std::set<Square>(got.begin(), got.end()) == want && got.size() == want.size();
        ++far_cases;
        far_found += !got.empty();
      }
    }
  }
  Check(far_ok && far_found > 0 && far_found < far_cases,
        absl::StrFormat("1. FarPieces equals the pieces worked out here, in %d cases of %d "
                        "positions (%d with some far piece, %d without)",
                        far_cases, positions.size(), far_found, far_cases - far_found));

  // 2. Without, and its control; 3. Shifted.
  int removed = 0, shifted = 0, none = 0;
  bool without_ok = true, control_ok = true, shifted_ok = true;
  for (const auto& p : positions) {
    const auto& t = open_spiel::down_cast<const thud::ThudState&>(*p);
    const Board b = Read(*p);
    for (char kind : {'d', 'T'}) {
      const auto pieces = Pieces(b, kind);
      if (pieces.size() < 2) continue;
      const Square at = pieces[rng() % pieces.size()];
      const Board after = Read(*edits::Without(*game, *p, at.first, at.second));
      const auto diff = Differences(b, after);
      without_ok = without_ok && diff.size() == 1 && diff[0] == at &&
                   after.rows[at.first][at.second] == '.' && after.status == b.status;
      ++removed;
      const Square other = pieces[(std::find(pieces.begin(), pieces.end(), at) - pieces.begin() + 1) %
                                  pieces.size()];
      const auto wrong = Differences(b, Read(*edits::Without(*game, *p, other.first, other.second)));
      control_ok = control_ok && !(wrong.size() == 1 && wrong[0] == at);
    }
    for (const Square d : Pieces(b, 'd')) {
      const auto moved = edits::Shifted(*game, t, d.first, d.second, 4);
      if (moved == nullptr) {
        // No neighbouring empty square at least 4 from every troll: confirm by hand.
        bool any = false;
        for (int dr = -1; dr <= 1; ++dr) {
          for (int dc = -1; dc <= 1; ++dc) {
            const int r = d.first + dr, c = d.second + dc;
            if ((dr || dc) && thud::IsOnBoard(r, c) && b.rows[r][c] == '.') {
              bool ok = true;
              for (Square tr : Pieces(b, 'T')) ok = ok && Distance({r, c}, tr) >= 4;
              any = any || ok;
            }
          }
        }
        shifted_ok = shifted_ok && !any;
        ++none;
        continue;
      }
      const Board after = Read(*moved);
      const auto diff = Differences(b, after);
      bool ok = diff.size() == 2 && after.status == b.status;
      for (Square s : diff) {
        if (s == d) {
          ok = ok && after.rows[s.first][s.second] == '.';
        } else {
          ok = ok && Distance(s, d) == 1 && b.rows[s.first][s.second] == '.' &&
               after.rows[s.first][s.second] == 'd';
          for (Square tr : Pieces(b, 'T')) ok = ok && Distance(s, tr) >= 4;
        }
      }
      shifted_ok = shifted_ok && ok;
      ++shifted;
    }
  }
  Check(without_ok && removed > 0,
        absl::StrFormat("2. Without: exactly that square emptied, the status line unchanged "
                        "(%d edits)",
                        removed));
  Check(control_ok, "2. control: removing another piece gives another board");
  Check(shifted_ok && shifted > 0 && none > 0,
        absl::StrFormat("3. Shifted: the dwarf moved one square, to a square at least 4 from "
                        "every troll (%d), or none when no such square exists (%d, confirmed "
                        "here)",
                        shifted, none));

  // 3. Hand-built: a dwarf boxed in by dwarfs, and one whose free neighbours are near a troll.
  {
    std::unique_ptr<open_spiel::State> start = game->NewInitialState();
    const Board b = Read(*start);
    // The initial position's corner dwarfs at (0,5) and (0,6): (0,5)'s neighbours on the
    // board are (0,6) (a dwarf), (1,4) (a dwarf), (1,5) and (1,6) (empty).
    const auto& t = open_spiel::down_cast<const thud::ThudState&>(*start);
    // At far 1 (no troll nearby matters) it moves; at far 13 no square is far enough.
    Check(edits::Shifted(*game, t, 0, 5, 1) != nullptr && edits::Shifted(*game, t, 0, 5, 13) == nullptr,
          "3. hand-built: the opening's dwarf at (0,5) moves at far 1, and at far 13 (every "
          "square near a troll) it does not");
    (void)b;
  }

  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
