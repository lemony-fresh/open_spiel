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

// Tests the taper's condition (c) (thud/experiments/unsafe_targets.h, used by
// az_unsafe_targets):
//
//   1. IsUnsafe by hand: in a built position, a dwarf step next to a troll is unsafe, a
//      step away from it is not, a capture is not even beside a troll; and on positions of
//      random games it equals a computation of its own (the board read from the text).
//   2. Ratio: the mean visit share over the mean prior mass, worked out by hand.
//   3. RatioUpper: at least R; with 4x the positions (the same values repeated) the
//      interval narrows (the upper end falls towards R); identical positions give R itself.
//   4. TaperAllowed: the design's table — R at most 2.0 for 0.7 -> 0.6, 1.6 for 0.6 ->
//      0.5, 1.33 for 0.5 -> 0.4, 1.0 for 0.3 -> 0.2, 0.89 for 0.2 -> 0.1: just below each
//      bound allowed, just above refused.
//
//   thud/run_tests.sh unsafe_targets_test   # builds and runs it
//   build-tests/bin/unsafe_targets_test

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/experiments/unsafe_targets.h"

namespace {

namespace unsafe = open_spiel::thud_az::unsafe_targets;
int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

open_spiel::Action Find(const open_spiel::State& s, const std::string& move) {
  for (open_spiel::Action a : s.LegalActions()) {
    if (s.ActionToString(a) == move) return a;
  }
  return open_spiel::kInvalidAction;
}

// Unsafe by hand: the board after the move read from the text format.
bool UnsafeByHand(const open_spiel::State& s, open_spiel::Action a) {
  const std::string move = s.ActionToString(a);
  if (move.back() == 'x') return false;
  const auto [r, c] = unsafe::Destination(move);
  const std::string text = s.Child(a)->ToString();
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const int rr = r + dr, cc = c + dc;
      if ((dr || dc) && rr >= 0 && rr < 15 && cc >= 0 && cc < 15 && text[rr * 16 + cc] == 'T') {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // 1. By hand: a lone dwarf at (5,2), a troll at (5,5) (and the Thudstone), dwarfs to move.
  {
    const std::string rows[15] = {
        "-----.....-----", "----.......----", "---.........---", "--...........--",
        "-.............-", "..d..T.........", "...............", ".......O.......",
        "...............", "...............", "-.............-", "--...........--",
        "---.........---", "----.......----", "-----.....-----"};
    std::string text;
    for (const auto& r : rows) text += r + "\n";
    std::unique_ptr<open_spiel::State> s = game->NewInitialState(text + "to_move=dwarfs");
    const open_spiel::Action toward = Find(*s, "(5,2)-(5,4)"), away = Find(*s, "(5,2)-(5,1)");
    Check(toward != open_spiel::kInvalidAction && away != open_spiel::kInvalidAction &&
              unsafe::IsUnsafe(*s, toward) && !unsafe::IsUnsafe(*s, away),
          "1. by hand: the dwarf's step to (5,4), next to the troll, unsafe; to (5,1) not");
  }
  {
    // A capture next to a troll is not unsafe: two dwarfs in a line hurl at a troll.
    const std::string rows[15] = {
        "-----.....-----", "----.......----", "---.........---", "--...........--",
        "-.............-", ".dd.T..........", "....T..........", ".......O.......",
        "...............", "...............", "-.............-", "--...........--",
        "---.........---", "----.......----", "-----.....-----"};
    std::string text;
    for (const auto& r : rows) text += r + "\n";
    std::unique_ptr<open_spiel::State> s = game->NewInitialState(text + "to_move=dwarfs");
    const open_spiel::Action hurl = Find(*s, "(5,2)-(5,4)x");
    Check(hurl != open_spiel::kInvalidAction && !unsafe::IsUnsafe(*s, hurl),
          "1. by hand: a hurl that captures, ending next to another troll, not unsafe");
  }
  {
    std::mt19937 rng(9);
    int compared = 0, unsafe_moves = 0;
    bool same = true;
    for (int g = 0; g < 6; ++g) {
      std::unique_ptr<open_spiel::State> s = game->NewInitialState();
      for (int ply = 0; !s->IsTerminal() && ply < 200; ++ply) {
        const auto legal = s->LegalActions();
        if (ply % 6 == 4 && s->CurrentPlayer() == 0) {
          for (open_spiel::Action a : legal) {
            const bool u = unsafe::IsUnsafe(*s, a);
            same = same && u == UnsafeByHand(*s, a);
            ++compared;
            unsafe_moves += u;
          }
        }
        s->ApplyAction(legal[rng() % legal.size()]);
      }
    }
    Check(same && unsafe_moves > 0 && unsafe_moves < compared,
          absl::StrFormat("1. on random games IsUnsafe equals the board read here: %d dwarf "
                          "moves, %d unsafe",
                          compared, unsafe_moves));
  }

  // 2. Ratio.
  const std::vector<double> prior = {0.2, 0.1, 0.3, 0.0}, visits = {0.3, 0.1, 0.5, 0.1};
  Check(std::abs(unsafe::Ratio(prior, visits) - 1.0 / 0.6) < 1e-12,
        absl::StrFormat("2. Ratio: (0.3+0.1+0.5+0.1) / (0.2+0.1+0.3+0.0) = %.4f",
                        unsafe::Ratio(prior, visits)));

  // 3. RatioUpper.
  {
    const double r = unsafe::Ratio(prior, visits), u = unsafe::RatioUpper(prior, visits);
    std::vector<double> p4, v4;
    for (int k = 0; k < 4; ++k) {
      p4.insert(p4.end(), prior.begin(), prior.end());
      v4.insert(v4.end(), visits.begin(), visits.end());
    }
    const double u4 = unsafe::RatioUpper(p4, v4);
    const double same = unsafe::RatioUpper({0.2, 0.2, 0.2}, {0.3, 0.3, 0.3});
    Check(u >= r && u4 >= r && u4 < u && std::abs(same - 1.5) < 1e-12,
          absl::StrFormat("3. RatioUpper: %.3f >= R %.3f; with 4x the positions %.3f, "
                          "narrower; identical positions give R (%.3f)",
                          u, r, u4, same));
  }

  // 4. TaperAllowed: the design's table.
  {
    struct Row { double w, bound; };
    bool ok = true;
    std::string got;
    for (const Row& row : {Row{0.7, 2.0}, Row{0.6, 1.6}, Row{0.5, 0.8 / 0.6}, Row{0.3, 1.0},
                           Row{0.2, 0.8 / 0.9}}) {
      ok = ok && unsafe::TaperAllowed(row.w, row.bound - 1e-6) &&
           !unsafe::TaperAllowed(row.w, row.bound + 1e-6);
      got += absl::StrFormat(" %.1f->%.1f: R <= %.3f;", row.w, row.w - 0.1, row.bound);
    }
    Check(ok, "4. TaperAllowed, the design's table:" + got);
  }

  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
