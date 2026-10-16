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

#ifndef THUD_EXPERIMENTS_POSITION_EDITS_H_
#define THUD_EXPERIMENTS_POSITION_EDITS_H_

// The position edits of az_dwarf_value_probe (measurement 4 of
// thud/DESIGN_dwarf_signal.md), in a header so that position_edits_test.cc can test them:
// the pieces far from the other side, a position with one piece removed, and one with a
// dwarf moved to a neighbouring square.

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace thud_az {
namespace position_edits {

using ::open_spiel::State;
namespace thud = ::open_spiel::thud;

// The squares of `kind` at least `far` king's moves from every piece of `other`.
inline std::vector<std::pair<int, int>> FarPieces(const thud::ThudState& state, thud::Cell kind,
                                           thud::Cell other, int far) {
  std::vector<std::pair<int, int>> mine, theirs, out;
  for (int r = 0; r < thud::kBoardSize; ++r) {
    for (int c = 0; c < thud::kBoardSize; ++c) {
      if (!thud::IsOnBoard(r, c)) continue;
      if (state.CellAt(r, c) == kind) mine.push_back({r, c});
      if (state.CellAt(r, c) == other) theirs.push_back({r, c});
    }
  }
  if (mine.size() < 2) return out;  // Never remove a side's last piece.
  for (const auto& [r, c] : mine) {
    bool ok = true;
    for (const auto& [r2, c2] : theirs) {
      if (std::max(std::abs(r - r2), std::abs(c - c2)) < far) ok = false;
    }
    if (ok) out.push_back({r, c});
  }
  return out;
}

// `state` with the piece at (row, col) removed, through the text format.
inline std::unique_ptr<State> Without(const open_spiel::Game& game, const State& state, int row,
                               int col) {
  std::string text = state.ToString();
  const size_t at = row * (thud::kBoardSize + 1) + col;  // Rows of 15 and a newline.
  SPIEL_CHECK_TRUE(text[at] == 'd' || text[at] == 'T');
  text[at] = '.';
  return game.NewInitialState(text);
}

// `state` with the dwarf at (row, col) moved to an empty neighbouring square at least
// `far` king's moves from every troll, or null if there is none.
inline std::unique_ptr<State> Shifted(const open_spiel::Game& game, const thud::ThudState& state,
                               int row, int col, int far) {
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const int r = row + dr, c = col + dc;
      if ((dr == 0 && dc == 0) || !thud::IsOnBoard(r, c) ||
          state.CellAt(r, c) != thud::Cell::kEmpty) {
        continue;
      }
      bool ok = true;
      for (int r2 = 0; r2 < thud::kBoardSize && ok; ++r2) {
        for (int c2 = 0; c2 < thud::kBoardSize && ok; ++c2) {
          if (thud::IsOnBoard(r2, c2) && state.CellAt(r2, c2) == thud::Cell::kTroll &&
              std::max(std::abs(r - r2), std::abs(c - c2)) < far) {
            ok = false;
          }
        }
      }
      if (!ok) continue;
      std::string text = state.ToString();
      text[row * (thud::kBoardSize + 1) + col] = '.';
      text[r * (thud::kBoardSize + 1) + c] = 'd';
      return game.NewInitialState(text);
    }
  }
  return nullptr;
}

}  // namespace position_edits
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_EXPERIMENTS_POSITION_EDITS_H_
