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

// Checks how the trainer of our copy of OpenSpiel's C++ AlphaZero classifies the
// end of a game (GameEnding in thud/az/alpha_zero.h; thud/PLAN.md, Instrumentation):
// one hand-built Thud position per ending, played to its end, must give that
// ending — every one of the five, so each branch is exercised (the trainer's
// smoke runs end almost only with the dwarfs gone) — and a cut-off game and
// another game's end give "cutoff" and "terminal". Control: a position one move
// before an ending is not terminal.
//
//   thud/az/build.sh thud/az/instrumentation_check.cc
//   build-shared/instrumentation_check
//
// Exits non-zero on any failure.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "thud/az/alpha_zero.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::torch_az::GameEnding;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

// An empty board with `pieces` ({row, col, 'd' or 'T'}) and the status line.
std::string Board(const std::vector<std::tuple<int, int, char>>& pieces,
                  const std::string& status) {
  std::vector<std::string> rows = {
      "-----.....-----", "----.......----", "---.........---", "--...........--",
      "-.............-", "...............", "...............", ".......O.......",
      "...............", "...............", "-.............-", "--...........--",
      "---.........---", "----.......----", "-----.....-----"};
  for (const auto& [row, col, piece] : pieces) rows[row][col] = piece;
  return absl::StrCat(absl::StrJoin(rows, "\n"), "\n", status);
}

// Plays the first legal move whose text ends in "x" (a capture) if `capture`,
// else the first legal move.
void Play(State* state, bool capture) {
  for (Action action : state->LegalActions()) {
    const std::string text = state->ActionToString(state->CurrentPlayer(), action);
    if (!capture || text.back() == 'x') {
      state->ApplyAction(action);
      return;
    }
  }
  std::cout << "no such move" << std::endl;
  ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  std::shared_ptr<const open_spiel::Game> thud = open_spiel::LoadGame("thud");
  const std::string initial_board =
      thud->NewInitialState()->ToString().substr(0, 16 * 15);

  struct Case {
    std::string name, text;
    bool capture;
  };
  const std::vector<Case> cases = {
      {"turn_limit",
       initial_board + "to_move=dwarfs turns=799 turns_without_capture=0", false},
      {"no_capture_limit",
       initial_board + "to_move=dwarfs turns=10 turns_without_capture=199", false},
      // A lone dwarf hurls itself one square onto the last troll.
      {"trolls_gone", Board({{7, 3, 'd'}, {7, 4, 'T'}}, "to_move=dwarfs"), true},
      // The last troll steps next to the last dwarf and captures it.
      {"dwarfs_gone", Board({{7, 2, 'T'}, {7, 4, 'd'}}, "to_move=trolls"), true},
  };
  for (const Case& c : cases) {
    std::unique_ptr<State> state = thud->NewInitialState(c.text);
    Expect(!state->IsTerminal(),
           absl::StrCat("control: not over one move before ", c.name));
    Play(state.get(), c.capture);
    const std::string ending = state->IsTerminal() ? GameEnding(*state, false) : "";
    Expect(ending == c.name, absl::StrCat(c.name, ": ", ending));
  }
  // A troll walled in by dwarfs on its edge square, the trolls to move.
  std::unique_ptr<State> walled = thud->NewInitialState(Board(
      {{7, 0, 'T'}, {6, 0, 'd'}, {8, 0, 'd'}, {6, 1, 'd'}, {7, 1, 'd'}, {8, 1, 'd'}},
      "to_move=trolls"));
  Expect(walled->IsTerminal() && GameEnding(*walled, false) == "no_legal_move",
         absl::StrCat("no_legal_move: ",
                      walled->IsTerminal() ? GameEnding(*walled, false) : "not over"));
  std::unique_ptr<State> free_troll = thud->NewInitialState(Board(
      {{7, 0, 'T'}, {6, 0, 'd'}, {8, 0, 'd'}, {6, 1, 'd'}, {8, 1, 'd'}},
      "to_move=trolls"));
  Expect(!free_troll->IsTerminal(), "control: a troll with one empty neighbour can move");
  Expect(GameEnding(*thud->NewInitialState(), true) == "cutoff", "cut off: cutoff");

  std::unique_ptr<State> tic = open_spiel::LoadGame("tic_tac_toe")->NewInitialState();
  for (Action a : {0, 3, 1, 4, 2}) tic->ApplyAction(a);
  Expect(tic->IsTerminal() && GameEnding(*tic, false) == "terminal",
         "another game: terminal");

  std::cout << (failures ? "INSTRUMENTATION CHECK FAILED"
                         : "instrumentation check passed")
            << std::endl;
  return failures ? 1 : 0;
}
