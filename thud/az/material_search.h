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

#ifndef THUD_AZ_MATERIAL_SEARCH_H_
#define THUD_AZ_MATERIAL_SEARCH_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace thud_az {

// The evaluation a MaterialSearch scores its leaves with. Only kMaterial today: the
// battle's own score, the dwarfs left less 4 per troll left, from the side to move's view —
// which is also what a finished battle returns (thud.cc, Returns). A positional
// evaluation, if one is added, gets its own entry, and the search then runs without the
// frontier rule below, which is exact only for kMaterial.
enum class Evaluation { kMaterial };

// Alpha-beta search of Thud to a fixed depth for exact values (2026-10-23; thud/PLAN.md,
// the GPU checklist, item 3: kickstarting's teacher, AlphaBetaPlayer::Scores), on the
// game's own state, made and unmade in place (ThudState::UndoAction). Its values are the
// minimax values OpenSpiel's alpha-beta finds (algorithms/minimax.h) — only the order the
// moves are tried in, and so how much is pruned, differs (teacher_identity_test).
//
// The frontier rule, for kMaterial only: one ply above the leaves only captures change the
// score — a quiet move keeps the material, and a battle that ends keeps it too — so a node
// there is worth its material plus its best capture's gain: its material is a lower bound
// (a cut-off as soon as it reaches beta, with no move generated), and otherwise the best
// capture is found without generating moves (ThudState::BestCapture), none applied. The
// rule depends on the evaluation, so it is tied to it: Evaluate() is the only place the
// leaves are scored, and the rule runs only for Evaluation::kMaterial. In builds with
// assertions on (the tests', without NDEBUG), every 16th frontier node is also searched in
// full — every move applied, each child scored by Evaluate(), or by the game's own return if
// the battle ends — and must agree (exactly, or as the bound it returned): an evaluation
// that stops being material alone, or a capture BestCapture misses, fails every test that
// searches.
//
// Above the frontier, moves are tried captures first (the largest gain first) if
// `captures_first`, then the killer moves of that depth (the last two quiet moves that cut
// off there) if `killers`, then in the game's order.
class MaterialSearch {
 public:
  explicit MaterialSearch(Evaluation evaluation = Evaluation::kMaterial,
                          bool captures_first = true, bool killers = true)
      : evaluation_(evaluation), captures_first_(captures_first), killers_on_(killers) {}

  // Every legal action's value, in points for the side to move at `state`, searched
  // `depth` plies (the move itself the first): the material after the move at depth 1, after
  // the best reply at depth 2, ... In LegalActions() order.
  std::vector<double> Scores(const State& state, int depth) {
    SPIEL_CHECK_GE(depth, 1);
    SPIEL_CHECK_LT(depth, kMaxDepth);
    std::unique_ptr<State> copy = state.Clone();
    auto& s = down_cast<thud::ThudState&>(*copy);
    const Player me = s.CurrentPlayer();
    for (auto& k : killer_moves_) k = {kInvalidAction, kInvalidAction};
    std::vector<double> scores;
    for (Action action : s.LegalActions()) {
      s.ApplyAction(action);
      const int value = s.IsTerminal() || depth == 1
                            ? Evaluate(s, me)
                            : -Search(&s, depth - 1, -kInfinity, kInfinity, 1 - me);
      s.UndoAction(me, action);
      scores.push_back(value);
    }
    return scores;
  }

  // Nodes visited since the last ResetNodes(), for measurements.
  int64_t nodes() const { return nodes_; }
  void ResetNodes() { nodes_ = 0; }

 private:
  static constexpr int kInfinity = 1 << 20;
  static constexpr int kMaxDepth = 16;

  // The leaves' score in points from `me`'s view: the only place an evaluation is chosen.
  int Evaluate(const thud::ThudState& s, Player me) const {
    switch (evaluation_) {
      case Evaluation::kMaterial: {
        const int margin =
            s.NumDwarfs() * thud::kDwarfValue - s.NumTrolls() * thud::kTrollValue;
        return me == thud::kDwarfPlayer ? margin : -margin;
      }
    }
    SpielFatalError("MaterialSearch: unknown evaluation");
  }

  // Negamax with alpha-beta, fail-soft: the value for `me`, the side to move at `s`.
  int Search(thud::ThudState* s, int depth, int alpha, int beta, Player me) {
    ++nodes_;
    if (s->IsTerminal() || depth == 0) return Evaluate(*s, me);
    if (depth == 1 && evaluation_ == Evaluation::kMaterial) {  // The frontier rule.
      const int material = Evaluate(*s, me);
      const int worth = me == thud::kDwarfPlayer ? thud::kTrollValue : thud::kDwarfValue;
      const int value = material >= beta ? material : material + s->BestCapture() * worth;
#ifndef NDEBUG
      if ((++frontier_checks_ & 15) == 0) {
        const int full = FullFrontier(s, me);
        if (material >= beta) {
          SPIEL_CHECK_GE(full, material);  // A lower bound.
        } else {
          SPIEL_CHECK_EQ(full, value);
        }
      }
#endif
      return value;
    }
    std::vector<Action> moves = s->LegalActions();
    SPIEL_DCHECK_FALSE(moves.empty());  // A battle not over always has a move.
    Order(*s, depth, me, &moves);
    int best = -kInfinity;
    for (Action action : moves) {
      s->ApplyAction(action);
      const int value = -Search(s, depth - 1, -beta, -alpha, 1 - me);
      s->UndoAction(me, action);
      if (value > best) best = value;
      if (best > alpha) alpha = best;
      if (alpha >= beta) {
        if (killers_on_ && s->Captures(action) == 0) {
          auto& k = killer_moves_[depth];
          if (k[0] != action) k = {action, k[0]};
        }
        break;
      }
    }
    return best;
  }

#ifndef NDEBUG
  // The frontier node searched in full, for the rule's check: every move applied, each
  // child scored by Evaluate(), or by the game's return (in points) if the battle ends.
  int FullFrontier(thud::ThudState* s, Player me) {
    int best = -kInfinity;
    for (Action action : s->LegalActions()) {
      s->ApplyAction(action);
      const int value =
          s->IsTerminal()
              ? static_cast<int>(std::lround(s->PlayerReturn(me) * thud::kMaxMargin))
              : Evaluate(*s, me);
      s->UndoAction(me, action);
      best = std::max(best, value);
    }
    return best;
  }
  int64_t frontier_checks_ = 0;
#endif

  // Captures first, the largest gain first; then the killers; then the rest, as listed.
  void Order(const thud::ThudState& s, int depth, Player me, std::vector<Action>* moves) {
    if (!captures_first_ && !killers_on_) return;
    const int worth = me == thud::kDwarfPlayer ? thud::kTrollValue : thud::kDwarfValue;
    std::vector<std::pair<int, Action>> keyed;  // (priority, action), larger first.
    keyed.reserve(moves->size());
    const auto& k = killer_moves_[depth];
    for (Action action : *moves) {
      int priority = 0;
      if (captures_first_) priority = 1000 * s.Captures(action) * worth;
      if (priority == 0 && killers_on_) priority = action == k[0] ? 2 : action == k[1] ? 1 : 0;
      keyed.push_back({priority, action});
    }
    std::stable_sort(keyed.begin(), keyed.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < moves->size(); ++i) (*moves)[i] = keyed[i].second;
  }

  const Evaluation evaluation_;
  const bool captures_first_, killers_on_;
  std::array<std::array<Action, 2>, kMaxDepth> killer_moves_{};
  int64_t nodes_ = 0;
};

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_MATERIAL_SEARCH_H_
