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

#ifndef THUD_AZ_THUD_FEATURES_H_
#define THUD_AZ_THUD_FEATURES_H_

// Input planes from the rules (thud/PLAN.md, the input planes, as decided — with the user,
// 2026-10-11): facts the rules decide at once, given to the network beside the board so
// that it need not compute them (as AlphaGo's "capture size" and "liberties" planes do).
// They are computed from Thud's 6-plane observation alone (thud.h: the dwarf, troll and
// empty planes describe the whole board; the Thudstone and the cut-off corners are in
// none) — so that they can be built for any stored position, the replay buffer keeping
// only the observation, and for a turned board after symmetry augmentation. Every plane
// is kBoardSize x kBoardSize and states what would happen "if that side moved now",
// for both sides at once, whoever is to move: the network sees the mover's captures
// and the opponent's threats together, and the observation's trolls-to-move plane says
// whose turn it is. All values are 0 or positive. 50 planes:
//
// 48 MOVE PLANES, MovePlane(d, k): direction d (thud.h's order N, NE, E, SE, S, SW, W,
// NW) and distance k, 1..7 for the straight directions and 1..5 for the diagonals
// (MaxCaptureDistance(d)). Each is set at the MOVING PIECE'S square — where the
// convolutional policy head scores the move (square, d, k) — and describes that move:
//   - on a dwarf: 1 if hurling it k squares in direction d captures a troll — a line of
//     at least k dwarfs behind it (it counting), k - 1 empty squares, and a troll on
//     the k-th. A hurl lands on the first piece in its path, so a dwarf has at most
//     one nonzero distance a direction;
//   - on a troll: the number of dwarfs its move of k squares in direction d takes, over
//     8 — k = 1 a capturing step, k >= 2 a shove (a line of at least k trolls behind it,
//     an empty path, a dwarf next to the landing square). A troll may have several
//     nonzero distances a direction.
// The planes are shared by both sides: a square holds one piece, and the observation's
// dwarf and troll planes, at the same square, say which (AlphaZero separates the
// players' pieces because both players own the same kinds; in Thud the kind says
// whose). Elsewhere they are 0.
//
// WHY 7, AND 5 ON THE DIAGONALS: a capture of k squares needs 2k squares in a straight
// line — k pieces, k - 1 empty squares, and the target (a hurl) or the landing square (a
// troll's move). The board's longest straight lines (rows and columns 5, 6, 8, 9) hold 15
// squares, so k <= 7; its longest diagonals hold 10 (such as (5,0) to (14,9); the cut-off
// corners shorten every diagonal, and the Thudstone cuts the central ones), so k <= 5.
// Other pieces only take squares away. So a diagonal capture of 6 or 7 squares is
// impossible in every position, and its 8 planes (4 diagonals x 2 distances), which would
// always be zero, are left out (the user, 2026-10-11; features_test, part 5, finds room
// on the board for every other kind, direction and distance, and for none of these).
//
// 2 MAPS, at the AFFECTED squares:
//   kTrollLandsNear  1 on a dwarf or an empty square next to a square some troll could
//                    land on — a step, or a shove along an empty path with a line of
//                    trolls behind it as long as the shove — whether or not that move
//                    captures now. On a dwarf: a troll move would take it now. On an
//                    empty square: a dwarf moving there would be taken (judged on the
//                    board as it is, the moving dwarf blocking no path).
//   kHurlLands       1 on a troll or an empty square a hurl could land on (along a dwarf
//                    line's direction, within its length, up to the first piece). On a
//                    troll: a hurl would take it now. On an empty square: a troll moving
//                    there could be hurled onto.
// (Each map is one fact read on two kinds of square: "takeable" is the map on the
// pieces, "danger" the map on the empty squares; the network separates them with the
// observation's piece planes.)

#include <algorithm>
#include <array>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/thud/thud.h"

namespace open_spiel {
namespace thud_az {

// The longest capture in direction d (see WHY 7, AND 5 ON THE DIAGONALS above): 7 for N,
// E, S, W (thud.h's even directions), 5 for the diagonals.
inline constexpr int kMaxCaptureDistance = 7;
inline constexpr int kMaxDiagonalCaptureDistance = 5;
constexpr int MaxCaptureDistance(int d) {
  return d % 2 == 0 ? kMaxCaptureDistance : kMaxDiagonalCaptureDistance;
}
inline constexpr int kNumMovePlanes = 4 * kMaxCaptureDistance +
                                      4 * kMaxDiagonalCaptureDistance;  // 48
// The move plane of direction d and distance k (1..MaxCaptureDistance(d)): the planes of
// N, NE, E, SE, S, SW, W, NW in turn, each direction's distances in order.
constexpr int MovePlane(int d, int k) {
  return (d / 2) * (kMaxCaptureDistance + kMaxDiagonalCaptureDistance) +
         (d % 2) * kMaxCaptureDistance + (k - 1);
}
// The direction and distance of move plane p (0..kNumMovePlanes - 1).
struct PlaneMove {
  int d, k;
};
constexpr PlaneMove MoveOfPlane(int p) {
  const int pair = p / (kMaxCaptureDistance + kMaxDiagonalCaptureDistance);
  const int rest = p % (kMaxCaptureDistance + kMaxDiagonalCaptureDistance);
  return rest < kMaxCaptureDistance ? PlaneMove{2 * pair, rest + 1}
                                    : PlaneMove{2 * pair + 1, rest - kMaxCaptureDistance + 1};
}
inline constexpr int kTrollLandsNear = kNumMovePlanes;        // 48
inline constexpr int kHurlLands = kNumMovePlanes + 1;         // 49
inline constexpr int kNumFeaturePlanes = kNumMovePlanes + 2;  // 50

namespace features_internal {

enum class Square { kBlocked, kEmpty, kDwarf, kTroll };

class Board {
 public:
  explicit Board(absl::Span<const float> observation) {
    constexpr int n = thud::kBoardSize;
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        auto at = [&](int plane) { return observation[(plane * n + r) * n + c] > 0.5f; };
        squares_[r * n + c] = at(thud::kDwarfPlane)   ? Square::kDwarf
                              : at(thud::kTrollPlane) ? Square::kTroll
                              : at(thud::kEmptyPlane) ? Square::kEmpty
                                                      : Square::kBlocked;
      }
    }
  }
  Square At(int r, int c) const {
    if (r < 0 || c < 0 || r >= thud::kBoardSize || c >= thud::kBoardSize) {
      return Square::kBlocked;
    }
    return squares_[r * thud::kBoardSize + c];
  }
  // The unbroken line of pieces like the one at (r, c), from it in direction d,
  // counting it.
  int Line(int r, int c, int d) const {
    const Square kind = At(r, c);
    int n = 0;
    while (At(r, c) == kind) {
      ++n;
      r += thud::kRowStep[d];
      c += thud::kColStep[d];
    }
    return n;
  }

 private:
  std::array<Square, thud::kBoardSize * thud::kBoardSize> squares_;
};

}  // namespace features_internal

// The kNumFeaturePlanes planes of `observation` (thud::kNumObservationPlanes planes of
// kBoardSize x kBoardSize), each kBoardSize x kBoardSize, row by row, appended to
// `out` (plane-major, as the observation).
inline void AppendFeaturePlanes(absl::Span<const float> observation,
                                std::vector<float>* out) {
  using features_internal::Board;
  using features_internal::Square;
  constexpr int n = thud::kBoardSize;
  const Board board(observation);
  const size_t base = out->size();
  out->resize(base + kNumFeaturePlanes * n * n, 0.0f);
  auto at = [&](int plane, int r, int c) -> float& {
    return (*out)[base + (plane * n + r) * n + c];
  };
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      const Square piece = board.At(r, c);
      if (piece != Square::kDwarf && piece != Square::kTroll) continue;
      for (int d = 0; d < thud::kNumDirections; ++d) {
        const int dr = thud::kRowStep[d], dc = thud::kColStep[d];
        const int behind = (d + thud::kNumDirections / 2) % thud::kNumDirections;
        const int line = board.Line(r, c, behind);
        // The board allows nothing longer (see WHY 7, AND 5 ON THE DIAGONALS).
        const int longest = MaxCaptureDistance(d);
        if (piece == Square::kDwarf) {
          // A hurl lands on the first piece in its path, within the line's length.
          for (int k = 1; k <= std::min(line, longest); ++k) {
            const int tr = r + k * dr, tc = c + k * dc;
            const Square target = board.At(tr, tc);
            if (target == Square::kEmpty) {
              at(kHurlLands, tr, tc) = 1;
              continue;
            }
            if (target == Square::kTroll) {
              at(MovePlane(d, k), r, c) = 1;
              at(kHurlLands, tr, tc) = 1;
            }
            break;
          }
        } else {
          // A step, then shoves of 2 up to the line's length, along an empty path.
          for (int k = 1; k <= std::min(std::max(1, line), longest); ++k) {
            const int tr = r + k * dr, tc = c + k * dc;
            if (board.At(tr, tc) != Square::kEmpty) break;
            int taken = 0;
            for (int e = 0; e < thud::kNumDirections; ++e) {
              const int nr = tr + thud::kRowStep[e], nc = tc + thud::kColStep[e];
              const Square next = board.At(nr, nc);
              if (next == Square::kDwarf) ++taken;
              if (next == Square::kDwarf || next == Square::kEmpty) {
                at(kTrollLandsNear, nr, nc) = 1;
              }
            }
            if (taken > 0) at(MovePlane(d, k), r, c) = static_cast<float>(taken) / 8;
          }
        }
      }
    }
  }
}

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_THUD_FEATURES_H_
