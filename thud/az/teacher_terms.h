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

#ifndef THUD_AZ_TEACHER_TERMS_H_
#define THUD_AZ_TEACHER_TERMS_H_

#include <array>
#include <bitset>

#include "open_spiel/games/thud/thud.h"

namespace open_spiel {
namespace thud_az {

// The positional terms of the teacher's tie-break (2026-11-04; thud/PLAN.md, *The teacher's
// finer evaluation*): of the six candidates, the two that the fit to our own games kept — in
// self-play and in the matches alike, holding 90-98% of the six's gain over material alone
// (the other four's weights differed in sign or size between the two). Defined as
// thud/experiments/teacher_terms.py defines them (teacher_terms_test compares the two):
//   dwarf_hurl_reach  the distinct squares, empty or holding a troll, that a dwarf could reach
//                     by a hurl of 2 or more squares: its unbroken line behind it (it counting)
//                     at least that long, the squares between empty;
//   troll_cohesion    pairs of trolls on neighbouring squares, each pair once.
// They read the board alone, not the side to move.
struct PositionalTerms {
  int dwarf_hurl_reach = 0;
  int troll_cohesion = 0;
};

inline PositionalTerms ComputePositionalTerms(const thud::ThudState& s) {
  using thud::Cell;
  using thud::kBoardSize;
  auto at = [&s](int r, int c) {
    return thud::IsOnBoard(r, c) ? s.CellAt(r, c) : Cell::kOffBoard;
  };
  PositionalTerms terms;
  std::bitset<kBoardSize * kBoardSize> reached;
  for (int r = 0; r < kBoardSize; ++r) {
    for (int c = 0; c < kBoardSize; ++c) {
      const Cell piece = at(r, c);
      if (piece == Cell::kTroll) {
        // The four directions E, SE, S, SW (thud.h's 2-5) visit each neighbouring pair once.
        for (int d = 2; d <= 5; ++d) {
          terms.troll_cohesion += at(r + thud::kRowStep[d], c + thud::kColStep[d]) == Cell::kTroll;
        }
      }
      if (piece != Cell::kDwarf) continue;
      for (int d = 0; d < thud::kNumDirections; ++d) {
        const int dr = thud::kRowStep[d], dc = thud::kColStep[d];
        int line = 0;
        while (at(r - line * dr, c - line * dc) == Cell::kDwarf) ++line;
        for (int k = 1; k <= line; ++k) {
          const Cell square = at(r + k * dr, c + k * dc);
          if (square != Cell::kEmpty && square != Cell::kTroll) break;
          if (k >= 2) reached.set((r + k * dr) * kBoardSize + c + k * dc);
          if (square == Cell::kTroll) break;
        }
      }
    }
  }
  terms.dwarf_hurl_reach = static_cast<int>(reached.count());
  return terms;
}

// The terms' score in points of the final margin, from the dwarfs' view: `weights` for
// dwarf_hurl_reach and troll_cohesion, in that order.
inline double PositionalScore(const PositionalTerms& terms, const std::array<double, 2>& weights) {
  return weights[0] * terms.dwarf_hurl_reach + weights[1] * terms.troll_cohesion;
}

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_TEACHER_TERMS_H_
