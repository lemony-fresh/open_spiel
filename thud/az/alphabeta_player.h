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

#ifndef THUD_AZ_ALPHABETA_PLAYER_H_
#define THUD_AZ_ALPHABETA_PLAYER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace thud_az {

// A Thud player without a network: OpenSpiel's alpha-beta search
// (algorithms/minimax.h) to `depth` plies. At its depth limit it scores a position as
// the battle is scored — by material, the dwarfs left less 4 per troll left, over 32
// (a battle's return, which the search uses at terminal positions) — plus a tie-break
// of under a quarter of a point, random but fixed by the game's seed and the moves
// played, so that among equal scores it neither always plays the first legal action
// nor changes its mind within a search. Random leaf values also favour moves that keep
// more options open (Beal and Smith, "Random evaluations in chess", ICCA Journal 17(1),
// 1994). Thud's state has no undo, so the search copies it at each node.
//
// The opponent of az_match's b=alphabeta:DEPTH and, in training, of the games against
// alpha-beta (thud/PLAN.md, How we evaluate networks: training against alpha-beta).
class AlphaBetaPlayer {
 public:
  AlphaBetaPlayer(const Game& game, int depth) : game_(game), depth_(depth) {}

  // Fixes the tie-break for the next game.
  void NewGame(uint64_t seed) { seed_ = seed; }

  Action Choose(const State& state) const {
    const Player me = state.CurrentPlayer();
    auto value = [this, me](const State& leaf) {
      return Material(leaf, me) + TieBreak(leaf);
    };
    return algorithms::AlphaBetaSearch(game_, &state, value, depth_, me,
                                       /*use_undo=*/false)
        .second;
  }

  int depth() const { return depth_; }

  // Our change (thud/PLAN.md, kickstarting): each legal action's score, in points of
  // material from the side to move's view — the material after it, the opponent
  // replying as OpenSpiel's alpha-beta search to depth() - 1 plies would (depth 1: the
  // material right after the move; depth 2: after the opponent's best reply). No
  // tie-break: a teacher's scores are the same every time. In LegalActions() order.
  std::vector<double> Scores(const State& state) const {
    const Player me = state.CurrentPlayer();
    std::vector<double> scores;
    for (Action action : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(action);
      double value;  // In returns, for `me`.
      if (child->IsTerminal() || depth_ <= 1) {
        value = child->IsTerminal() ? child->PlayerReturn(me) : Material(*child, me);
      } else {
        const Player opponent = child->CurrentPlayer();
        auto material = [opponent](const State& leaf) { return Material(leaf, opponent); };
        value = -algorithms::AlphaBetaSearch(game_, child.get(), material, depth_ - 1,
                                              opponent, /*use_undo=*/false)
                     .first;
      }
      scores.push_back(value * thud::kMaxMargin);
    }
    return scores;
  }

 private:
  static double Material(const State& state, Player me) {
    const auto& thud_state = down_cast<const thud::ThudState&>(state);
    int margin = 0;  // The dwarfs'.
    for (int row = 0; row < thud::kBoardSize; ++row) {
      for (int col = 0; col < thud::kBoardSize; ++col) {
        if (!thud::IsOnBoard(row, col)) continue;
        const thud::Cell cell = thud_state.CellAt(row, col);
        if (cell == thud::Cell::kDwarf) margin += thud::kDwarfValue;
        if (cell == thud::Cell::kTroll) margin -= thud::kTrollValue;
      }
    }
    return static_cast<double>(me == thud::kDwarfPlayer ? margin : -margin) /
           thud::kMaxMargin;
  }

  double TieBreak(const State& leaf) const {
    // SplitMix64's finaliser over the seed and the moves.
    auto mix = [](uint64_t x) {
      x += 0x9e3779b97f4a7c15ULL;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      return x ^ (x >> 31);
    };
    uint64_t h = mix(seed_);
    for (Action action : leaf.History()) h = mix(h ^ static_cast<uint64_t>(action));
    const double uniform = (h >> 11) * 0x1.0p-53;  // [0, 1)
    return (uniform - 0.5) * 0.5 / thud::kMaxMargin;  // [-1/4, 1/4) of a point
  }

  const Game& game_;
  const int depth_;
  uint64_t seed_ = 0;
};

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_ALPHABETA_PLAYER_H_
