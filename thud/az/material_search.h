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
#include <cstdlib>
#include <memory>
#include <numeric>
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

// The threat term at the frontier (2026-11-04; thud/PLAN.md, *The teacher's finer
// evaluation*). The frontier rule stands in for the last ply with a scan of the best capture;
// what the opponent could take after it stays unseen. With a threat term the frontier also
// scans T, the most points the opponent could capture against the side to move there if it
// were the opponent's move (ThudState::BestCaptureFor), and a node there is worth the better
// of its best quiet move, estimated as its material less w x T (w the side's weight; no
// threat if the move ends the battle), and its captures:
//   kScan      the best capture's gain, less w x T (none if the move ends the battle);
//   kCaptures  each capture applied and the opponent's best capture after it scanned and
//              counted in full: depth 6's exact value for those lines (as chess's quiescence
//              search looks at captures alone at the horizon).
// kOff: the material frontier rule, today's search, bit for bit.
enum class Threat { kOff, kScan, kCaptures };

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
// off there) if `killers`, then in the game's order — after the transposition table's move.
// Tried and dropped (2026-10-27): principal-variation search — every move after a node's
// first searched with a null window first — changed the nodes by +2.6% (dwarfs) and -6.6%
// (trolls) at depth 5 and +14% for the dwarfs at depth 4 (60 positions a side of run R6).
//
// The root window (2026-10-27; Scores' `margin`): with a margin m > 0, each root move after
// the first is searched only far enough to show it is m or more points below the best so
// far. Every move within m - 1 points of the best gets its exact score; the others get an
// upper bound at most best - m (fail-soft: at least their true score). At m = 1 only the
// best moves are exact and nothing else ranks with them: the teacher's choice as a player
// (AlphaBetaPlayer::ChooseByScores) is that of exact scores, from 0.61 of the nodes at
// depth 5 for the trolls, 0.57 for the dwarfs (60 positions a side of run R6). For
// kickstarting's policy, a softmax over the scores at T = 0.25 that gives a bounded move at
// most e^(-4m) of a best move's weight, m = 3 (kTeacherRootMargin) would change the policy
// by at most 0.16% of its mass (the worst of 60 dwarf positions of run R7 at depth 5; 0.004%
// on average) but saves little: 1.4% of the dwarfs' nodes at depth 5, 12% of the trolls'
// at depth 4 — so the trainer's --kickstart_root_margin stays 0. At depth 6 the gain is no
// larger (24 positions a side: the dwarfs' nodes 0.984 at m = 3, 0.855 at m = 1; the
// trolls' 0.77 and 0.45).
class MaterialSearch {
 public:
  // The teacher's table (AlphaBetaPlayer::Scores, one a thread): 4 MB.
  static constexpr int kTeacherTableBits = 18;
  // The smallest root window kickstarting's policy could take (above; az_teacher_stats' gap
  // line, 2026-10-27: at 2 the bound reached 10% of the policy in one of 60 dwarf positions,
  // at 3 0.16%).
  static constexpr int kTeacherRootMargin = 3;

  explicit MaterialSearch(Evaluation evaluation = Evaluation::kMaterial,
                          bool captures_first = true, bool killers = true, int table_bits = 0)
      : evaluation_(evaluation), captures_first_(captures_first), killers_on_(killers) {
    SPIEL_CHECK_GE(table_bits, 0);
    SPIEL_CHECK_LE(table_bits, 28);
    if (table_bits > 0) {
      table_.resize(size_t{1} << table_bits);
      table_mask_ = table_.size() - 1;
    }
  }

  // The threat term (Threat, above) with the weight w for each side at the frontier, in
  // points per point threatened ({dwarfs, trolls}); kOff, the default, with no weights.
  // Values inside the search are then in hundredths of a point; with kOff in points, as
  // before. Clears the table, whose values depend on it.
  void SetThreat(Threat threat, std::array<double, 2> weights = {0, 0}) {
    SPIEL_CHECK_TRUE(threat != Threat::kOff || (weights[0] == 0 && weights[1] == 0));
    for (double w : weights) SPIEL_CHECK_TRUE(w >= 0 && w <= 1);
    threat_ = threat;
    threat_weights_ = weights;
    unit_ = threat == Threat::kOff ? 1 : kThreatUnit;
    std::fill(table_.begin(), table_.end(), Entry{});
  }
  Threat threat() const { return threat_; }

  // Every legal action's value, in points for the side to move at `state`, searched
  // `depth` plies (the move itself the first): the material after the move at depth 1, after
  // the best reply at depth 2, ... In LegalActions() order. With `margin` > 0 (the root
  // window, above) the moves m or more points below the best get an upper bound instead,
  // at most best - m; 0: every score exact.
  std::vector<double> Scores(const State& state, int depth, int margin = 0) {
    SPIEL_CHECK_GE(depth, 1);
    SPIEL_CHECK_LT(depth, kMaxDepth);
    SPIEL_CHECK_GE(margin, 0);
    std::unique_ptr<State> copy = state.Clone();
    auto& s = down_cast<thud::ThudState&>(*copy);
    const Player me = s.CurrentPlayer();
    const auto& game = down_cast<const thud::ThudGame&>(*s.GetGame());
    max_turns_ = game.max_turns();
    max_turns_without_capture_ = game.max_turns_without_capture();
    for (auto& k : killer_moves_) k = {kInvalidAction, kInvalidAction};
    const std::vector<Action> legal = s.LegalActions();
    std::vector<double> scores(legal.size());
    // With a margin, the likely best moves first (the table's move for this position if it
    // has one, then captures), so that the window rises early; the scores stay in legal order.
    std::vector<Action> moves = legal;
    if (margin > 0) {
      Action table_move = kInvalidAction;
      if (!table_.empty()) {
        const uint64_t key = Key(s, depth);
        const Entry& entry = table_[key & table_mask_];
        if (entry.depth > 0 && entry.key == key) table_move = entry.move;
      }
      Order(s, depth, me, table_move, &moves);
    }
    if (root_index_.empty()) root_index_.resize(thud::kNumDistinctActions);
    for (size_t i = 0; i < legal.size(); ++i) root_index_[legal[i]] = i;
    int best = -kInfinity;
    for (Action action : moves) {
      s.ApplyAction(action);
      int value;
      if (s.IsTerminal() || depth == 1) {
        value = Evaluate(s, me);
      } else {
        const int alpha = margin > 0 && best > -kInfinity ? best - margin * unit_ : -kInfinity;
        value = -Search(&s, depth - 1, -kInfinity, -alpha, 1 - me);
      }
      s.UndoAction(me, action);
      scores[root_index_[action]] = static_cast<double>(value) / unit_;
      best = std::max(best, value);
    }
    return scores;
  }

  // The value of `state` for the side to move, searched `depth` plies (>= 1) with the window
  // (alpha, beta), fail-soft: a value at most alpha says the true one is at most it, one at
  // least beta that it is at least it, one in between is exact. For tests of the table's
  // bounds (teacher_identity_test, 6e).
  int Value(const State& state, int depth, int alpha, int beta) {
    SPIEL_CHECK_GE(depth, 1);
    SPIEL_CHECK_LT(depth, kMaxDepth);
    SPIEL_CHECK_LT(alpha, beta);
    SPIEL_CHECK_TRUE(threat_ == Threat::kOff);  // Values in points: the material search alone.
    std::unique_ptr<State> copy = state.Clone();
    auto& s = down_cast<thud::ThudState&>(*copy);
    const auto& game = down_cast<const thud::ThudGame&>(*s.GetGame());
    max_turns_ = game.max_turns();
    max_turns_without_capture_ = game.max_turns_without_capture();
    for (auto& k : killer_moves_) k = {kInvalidAction, kInvalidAction};
    return Search(&s, depth, alpha, beta, s.CurrentPlayer());
  }

  // Nodes visited, and nodes the table's value ended, since the last ResetNodes(), for
  // measurements and tests.
  int64_t nodes() const { return nodes_; }
  int64_t table_cutoffs() const { return table_cutoffs_; }
  void ResetNodes() { nodes_ = table_cutoffs_ = 0; }

 private:
  static constexpr int kInfinity = 1 << 20;
  // The search's unit with a threat term: a hundredth of a point.
  static constexpr int kThreatUnit = 100;
  static constexpr int kMaxDepth = 16;

  // A table entry. `depth` 0: empty. Values are points (|value| <= 32 + 4 x 8; with a threat
  // term hundredths, |value| <= 6,400 + the threat's 700), actions below 19,800: 16 bits each.
  enum Bound : uint8_t { kExact, kLower, kUpper };
  struct Entry {
    uint64_t key = 0;
    int16_t value = 0;
    int16_t move = -1;
    int8_t depth = 0;
    Bound bound = kExact;
  };
  static_assert(sizeof(Entry) == 16, "an entry is 16 bytes");

  // The node's key: the board's hash, and near a battle limit the counters and the limits.
  uint64_t Key(const thud::ThudState& s, int depth) const {
    const uint64_t hash = s.BoardHash();
    if (s.TurnsPlayed() + depth < max_turns_ &&
        s.TurnsWithoutCapture() + depth < max_turns_without_capture_) {
      return hash;
    }
    uint64_t near = Mix(s.TurnsPlayed());
    near = Mix(near ^ s.TurnsWithoutCapture());
    near = Mix(near ^ max_turns_);
    near = Mix(near ^ max_turns_without_capture_);
    return hash ^ near;
  }
  // splitmix64's finaliser: every bit of `x` moves about half the bits of the result.
  static uint64_t Mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  // The leaves' score in points from `me`'s view: the only place an evaluation is chosen.
  int Evaluate(const thud::ThudState& s, Player me) const {
    switch (evaluation_) {
      case Evaluation::kMaterial: {
        const int margin =
            s.NumDwarfs() * thud::kDwarfValue - s.NumTrolls() * thud::kTrollValue;
        return (me == thud::kDwarfPlayer ? margin : -margin) * unit_;
      }
    }
    SpielFatalError("MaterialSearch: unknown evaluation");
  }

  // Negamax with alpha-beta, fail-soft: the value for `me`, the side to move at `s`.
  int Search(thud::ThudState* s, int depth, int alpha, int beta, Player me) {
    ++nodes_;
    if (s->IsTerminal() || depth == 0) return Evaluate(*s, me);
    if (depth == 1 && threat_ != Threat::kOff) return ThreatFrontier(s, beta, me);
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
    Entry* entry = nullptr;
    uint64_t key = 0;
    Action table_move = kInvalidAction;
    if (!table_.empty()) {
      key = Key(*s, depth);
      entry = &table_[key & table_mask_];
      if (entry->depth > 0 && entry->key == key) {
        table_move = entry->move;
        if (entry->depth == depth &&
            (entry->bound == kExact || (entry->bound == kLower && entry->value >= beta) ||
             (entry->bound == kUpper && entry->value <= alpha))) {
          ++table_cutoffs_;
          return entry->value;
        }
      }
    }
    const int alpha_before = alpha;
    std::vector<Action> moves = s->LegalActions();
    SPIEL_DCHECK_FALSE(moves.empty());  // A battle not over always has a move.
    Order(*s, depth, me, table_move, &moves);
    int best = -kInfinity;
    Action best_move = kInvalidAction;
    for (Action action : moves) {
      s->ApplyAction(action);
      const int value = -Search(s, depth - 1, -beta, -alpha, 1 - me);
      s->UndoAction(me, action);
      if (value > best) {
        best = value;
        best_move = action;
      }
      if (best > alpha) alpha = best;
      if (alpha >= beta) {
        if (killers_on_ && s->Captures(action) == 0) {
          auto& k = killer_moves_[depth];
          if (k[0] != action) k = {action, k[0]};
        }
        break;
      }
    }
    if (entry != nullptr) {  // Fail-soft: below the window an upper bound, above it a lower.
      SPIEL_DCHECK_LT(best_move, 1 << 15);
      SPIEL_DCHECK_LE(std::abs(best), 1 << 14);
      *entry = {key, static_cast<int16_t>(best), static_cast<int16_t>(best_move),
                static_cast<int8_t>(depth),
                best <= alpha_before ? kUpper : best >= beta ? kLower : kExact};
    }
    return best;
  }

  // A frontier node's value with the threat term (Threat, above), from `me`'s view, fail-soft
  // (a value at least beta is a lower bound). In units.
  int ThreatFrontier(thud::ThudState* s, int beta, Player me) {
    const int material = Evaluate(*s, me);
    const int mine = (me == thud::kDwarfPlayer ? thud::kDwarfValue : thud::kTrollValue) * unit_;
    const int theirs = (me == thud::kDwarfPlayer ? thud::kTrollValue : thud::kDwarfValue) * unit_;
    // A move ends the battle if it reaches the turn limit, a quiet one also the no-capture
    // limit; then the opponent never gets to carry out a threat.
    const bool any_move_ends = s->TurnsPlayed() + 1 >= max_turns_;
    const bool quiet_move_ends =
        any_move_ends || s->TurnsWithoutCapture() + 1 >= max_turns_without_capture_;
    const int threat = s->BestCaptureFor(1 - me) * mine;
    const int penalty = static_cast<int>(std::lround(threat_weights_[me] * threat));
    int best = -kInfinity;
    if (s->HasQuietMove()) best = material - (quiet_move_ends ? 0 : penalty);
    if (best >= beta) return best;
    if (threat_ == Threat::kScan) {
      const int capture = s->BestCapture();
      if (capture > 0) {
        best = std::max(best, material + capture * theirs - (any_move_ends ? 0 : penalty));
      }
      return best;
    }
    for (Action action : s->CaptureActions()) {
      s->ApplyAction(action);
      const int value =
          s->IsTerminal()
              ? static_cast<int>(std::lround(s->PlayerReturn(me) * thud::kMaxMargin)) * unit_
              : Evaluate(*s, me) - s->BestCapture() * mine;
      s->UndoAction(me, action);
      best = std::max(best, value);
      if (best >= beta) break;
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

  // The table's move first (`first`, if it is among the moves); then captures, the
  // largest gain first; then the killers; then the rest, as listed — the order a stable sort
  // of all the moves by priority gives, but only the few moves above priority 0 are sorted
  // (a profile at depth 5, 2026-10-23: the full sort was a quarter of the time).
  void Order(const thud::ThudState& s, int depth, Player me, Action first,
             std::vector<Action>* moves) {
    if (!captures_first_ && !killers_on_ && first == kInvalidAction) return;
    const int worth = me == thud::kDwarfPlayer ? thud::kTrollValue : thud::kDwarfValue;
    ranked_.clear();  // (priority, index) of the moves above priority 0.
    const auto& k = killer_moves_[depth];
    for (size_t i = 0; i < moves->size(); ++i) {
      const Action action = (*moves)[i];
      int priority = 0;
      if (action == first) {
        priority = 1 << 24;  // Above any capture's 1000 x 7 x 4.
      } else {
        if (captures_first_) priority = 1000 * s.Captures(action) * worth;
        if (priority == 0 && killers_on_) {
          priority = action == k[0] ? 2 : action == k[1] ? 1 : 0;
        }
      }
      if (priority > 0) ranked_.push_back({priority, static_cast<int>(i)});
    }
    if (ranked_.empty()) return;
    std::stable_sort(ranked_.begin(), ranked_.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    unordered_.assign(moves->begin(), moves->end());
    size_t out = 0;
    for (const auto& [priority, i] : ranked_) {
      (*moves)[out++] = unordered_[i];
      unordered_[i] = kInvalidAction;  // Placed.
    }
    for (Action action : unordered_) {
      if (action != kInvalidAction) (*moves)[out++] = action;
    }
  }

  const Evaluation evaluation_;
  const bool captures_first_, killers_on_;
  Threat threat_ = Threat::kOff;  // SetThreat's.
  std::array<double, 2> threat_weights_ = {0, 0};
  int unit_ = 1;  // The search's unit: a point, or with a threat term kThreatUnit to a point.
  std::array<std::array<Action, 2>, kMaxDepth> killer_moves_{};
  std::vector<std::pair<int, int>> ranked_;  // Order()'s, reused (no recursion inside it).
  std::vector<Action> unordered_;
  std::vector<int> root_index_;  // Scores()': an action's index among the legal ones.
  std::vector<Entry> table_;
  uint64_t table_mask_ = 0;
  int max_turns_ = 0, max_turns_without_capture_ = 0;  // The game's, set by Scores().
  int64_t nodes_ = 0, table_cutoffs_ = 0;
};

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_MATERIAL_SEARCH_H_
