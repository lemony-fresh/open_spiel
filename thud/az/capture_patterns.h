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

#ifndef THUD_AZ_CAPTURE_PATTERNS_H_
#define THUD_AZ_CAPTURE_PATTERNS_H_

// Positions with one capture of every kind, direction and distance the board allows
// (thud/PLAN.md, the input planes: the tests must light every input plane that can be
// lit, and real games hold almost no long captures). For a hurl of k squares in
// direction d: k dwarfs in line (the hurled one in front), k - 1 empty squares, a troll.
// For a troll's move of k squares (k = 1 a capturing step, k >= 2 a shove): k trolls in
// line, k - 1 empty squares, the landing square, and a dwarf next to it — beyond it if
// there is room, else on its first open neighbour off the line. On an otherwise empty
// board, at the first place (in row-major order of the moving piece's square) where
// every square is on the board and not the Thudstone.

#include <string>
#include <vector>

#include "open_spiel/games/thud/thud.h"

namespace open_spiel {
namespace thud_az {

struct CapturePattern {
  bool troll;    // A troll's move, or a hurl.
  int d, k;      // Direction and distance.
  int row, col;  // The moving piece's square.
  std::string board;  // Fifteen rows of fifteen characters, each ending in '\n'.
};

// The pattern of a hurl (troll = false) or a troll's move of `k` squares in direction
// `d`, if the board has room for it.
inline bool FindCapturePattern(bool troll, int d, int k, CapturePattern* out) {
  constexpr int n = thud::kBoardSize;
  auto open = [](int r, int c) {
    return thud::IsOnBoard(r, c) && !(r == thud::kThudstoneRow && c == thud::kThudstoneCol);
  };
  const int dr = thud::kRowStep[d], dc = thud::kColStep[d];
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      // The squares from the line's back to the target (a hurl) or the landing square
      // (a troll's move); for a troll, a dwarf's square next to the landing square.
      const int first = -(k - 1);
      bool fits = true;
      for (int i = first; i <= k && fits; ++i) fits = open(r + i * dr, c + i * dc);
      if (!fits) continue;
      int dwarf_r = -1, dwarf_c = -1;
      if (troll) {
        const int tr = r + k * dr, tc = c + k * dc;
        const int behind = (d + thud::kNumDirections / 2) % thud::kNumDirections;
        for (int e : {d, (d + 1) % 8, (d + 7) % 8, (d + 2) % 8, (d + 6) % 8, (d + 3) % 8,
                      (d + 5) % 8}) {
          if (e == behind) continue;
          if (open(tr + thud::kRowStep[e], tc + thud::kColStep[e])) {
            dwarf_r = tr + thud::kRowStep[e];
            dwarf_c = tc + thud::kColStep[e];
            break;
          }
        }
        if (dwarf_r < 0) continue;
      }
      std::string board;
      for (int rr = 0; rr < n; ++rr) {
        for (int cc = 0; cc < n; ++cc) {
          board += !thud::IsOnBoard(rr, cc) ? '-' : open(rr, cc) ? '.' : 'O';
        }
        board += '\n';
      }
      auto set = [&](int i, char piece) { board[(r + i * dr) * (n + 1) + c + i * dc] = piece; };
      for (int i = first; i <= 0; ++i) set(i, troll ? 'T' : 'd');
      if (troll) {
        board[dwarf_r * (n + 1) + dwarf_c] = 'd';
      } else {
        set(k, 'T');
      }
      *out = {troll, d, k, r, c, board};
      return true;
    }
  }
  return false;
}

// Every pattern the board has room for, distances 1 to `max_k`.
inline std::vector<CapturePattern> AllCapturePatterns(int max_k) {
  std::vector<CapturePattern> out;
  for (bool troll : {false, true}) {
    for (int d = 0; d < thud::kNumDirections; ++d) {
      for (int k = 1; k <= max_k; ++k) {
        CapturePattern p;
        if (FindCapturePattern(troll, d, k, &p)) out.push_back(p);
      }
    }
  }
  return out;
}

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_CAPTURE_PATTERNS_H_
