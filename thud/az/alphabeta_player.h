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
#include "thud/az/material_search.h"
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
// 1994). The search makes and undoes its moves in place (ThudState::UndoAction), material
// is read from the state's piece counts, and the tie-break's hash carries on from the
// root's (2026-10-23: the search had copied the state at every node, scanning the board
// took ~73% of the teacher's time, and hashing a leaf's whole history ran over every move
// of the game; the same values either way — teacher_identity_test).
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
    const uint64_t root_hash = HistoryHash(mix(seed_), state, 0);
    const size_t root_length = state.FullHistory().size();
    auto value = [this, me, root_hash, root_length](const State& leaf) {
      return Material(leaf, me) + TieBreak(HistoryHash(root_hash, leaf, root_length));
    };
    return algorithms::AlphaBetaSearch(game_, &state, value, depth_, me,
                                       /*use_undo=*/true)
        .second;
  }

  int depth() const { return depth_; }

  // Our change (thud/PLAN.md, kickstarting): each legal action's score, in points of
  // material from the side to move's view — the material after it, the opponent
  // replying as OpenSpiel's alpha-beta search to depth() - 1 plies would (depth 1: the
  // material right after the move; depth 2: after the opponent's best reply). No
  // tie-break: a teacher's scores are the same every time. In LegalActions() order.
  std::vector<double> Scores(const State& state) const {
    // Since 2026-10-23 MaterialSearch (material_search.h: the same values, its moves
    // ordered and its frontier counted without applying moves).
    return MaterialSearch().Scores(state, depth_);
  }

 private:
  static double Material(const State& state, Player me) {
    const auto& thud_state = down_cast<const thud::ThudState&>(state);
    const int margin = thud_state.NumDwarfs() * thud::kDwarfValue -
                       thud_state.NumTrolls() * thud::kTrollValue;  // The dwarfs'.
    return static_cast<double>(me == thud::kDwarfPlayer ? margin : -margin) /
           thud::kMaxMargin;
  }

  // SplitMix64's finaliser.
  static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  // The tie-break's hash, mix(seed) then each move mixed in, carried on from `hash` (the
  // hash of the first `from` moves) over the moves after them.
  static uint64_t HistoryHash(uint64_t hash, const State& state, size_t from) {
    const std::vector<State::PlayerAction>& history = state.FullHistory();
    for (size_t i = from; i < history.size(); ++i) {
      hash = mix(hash ^ static_cast<uint64_t>(history[i].action));
    }
    return hash;
  }

  static double TieBreak(uint64_t hash) {
    const double uniform = (hash >> 11) * 0x1.0p-53;  // [0, 1)
    return (uniform - 0.5) * 0.5 / thud::kMaxMargin;  // [-1/4, 1/4) of a point
  }

  const Game& game_;
  const int depth_;
  uint64_t seed_ = 0;
};

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_ALPHABETA_PLAYER_H_
