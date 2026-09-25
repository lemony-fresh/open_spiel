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

// Unit tests for Thud, written from thud/THUD_RULES.md before the code they
// test (thud/PLAN.md, Phase 3). Section numbers below refer to THUD_RULES.md.
//
// Positions are board diagrams in the format documented in thud.h. Moves are
// written in ActionToString's notation, with squares as (row,col) and a
// trailing "x" marking a capture:
//   "(r,c)-(r,c)"   dwarf move, or troll step that captures nothing
//   "(r,c)-(r,c)x"  dwarf hurl, troll step capturing all adjacent dwarfs, or
//                   troll shove
//
// So that these tests check the implementation instead of repeating it, the
// square numbering, the direction order and the action formulas are re-derived
// below from sections 1 and 8, and are compared with the implementation's own
// versions only in the tests of those.

#include "open_spiel/games/thud/thud.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace thud {
namespace {

// ---------------------------------------------------------------------------
// The board and the action encoding, re-derived from sections 1 and 8.

// Playable columns of each row (section 1).
constexpr std::array<std::array<int, 2>, 15> kColumns = {{{5, 9},
                                                          {4, 10},
                                                          {3, 11},
                                                          {2, 12},
                                                          {1, 13},
                                                          {0, 14},
                                                          {0, 14},
                                                          {0, 14},
                                                          {0, 14},
                                                          {0, 14},
                                                          {1, 13},
                                                          {2, 12},
                                                          {3, 11},
                                                          {4, 10},
                                                          {5, 9}}};

bool OnBoard(int r, int c) {
  return r >= 0 && r < 15 && c >= kColumns[r][0] && c <= kColumns[r][1];
}

// Squares are numbered 0..164 in row-major order (section 8).
int Square(int r, int c) {
  SPIEL_CHECK_TRUE(OnBoard(r, c));
  int square = 0;
  for (int row = 0; row < r; ++row) {
    square += kColumns[row][1] - kColumns[row][0] + 1;
  }
  return square + c - kColumns[r][0];
}

std::array<int, 2> RowCol(int square) {
  for (int r = 0; r < 15; ++r) {
    for (int c = kColumns[r][0]; c <= kColumns[r][1]; ++c) {
      if (Square(r, c) == square) return {r, c};
    }
  }
  SpielFatalError(absl::StrCat("No square ", square));
}

// Directions N, NE, E, SE, S, SW, W, NW (section 8).
constexpr std::array<int, 8> kDr = {-1, -1, 0, 1, 1, 1, 0, -1};
constexpr std::array<int, 8> kDc = {0, 1, 1, 1, 0, -1, -1, -1};

// The two action ranges of section 8.
Action LineAction(int r, int c, int direction, int distance) {
  return (Square(r, c) * 8 + direction) * 14 + (distance - 1);
}
Action CaptureStepAction(int r, int c, int direction) {
  return 18480 + Square(r, c) * 8 + direction;
}

// An action split into its parts by the same formulas.
struct Parts {
  int square, direction, distance;
  bool capture_step;
};

Parts Decode(Action action) {
  if (action >= 18480) {
    return {static_cast<int>(action - 18480) / 8,
            static_cast<int>(action - 18480) % 8, 1, true};
  }
  return {static_cast<int>(action) / 112, static_cast<int>(action / 14) % 8,
          static_cast<int>(action % 14) + 1, false};
}

int Sign(int x) { return (x > 0) - (x < 0); }

// A move written in the notation above, split into its parts.
struct WrittenMove {
  int r1, c1, direction, distance;
  bool capture;  // Ends in "x".
};

WrittenMove ReadMove(const std::string& move) {
  int r1, c1, r2, c2, length = 0;
  SPIEL_CHECK_EQ(std::sscanf(move.c_str(), "(%d,%d)-(%d,%d)%n", &r1, &c1, &r2,
                             &c2, &length),
                 4);
  const std::string rest = move.substr(length);
  SPIEL_CHECK_TRUE(rest.empty() || rest == "x");
  const int dr = r2 - r1, dc = c2 - c1;
  const int distance = std::max(std::abs(dr), std::abs(dc));
  SPIEL_CHECK_TRUE(dr == 0 || dc == 0 || std::abs(dr) == std::abs(dc));
  SPIEL_CHECK_GT(distance, 0);
  int direction = 0;
  while (kDr[direction] != Sign(dr) || kDc[direction] != Sign(dc)) ++direction;
  return {r1, c1, direction, distance, rest == "x"};
}

// Parses a move written in the notation above. A one-square capture is a line
// action when a dwarf makes it (a hurl) but a capture step when a troll does,
// so only the position can tell: this version refuses one.
Action Parse(const std::string& move) {
  const WrittenMove written = ReadMove(move);
  SPIEL_CHECK_FALSE(written.capture && written.distance == 1);
  return LineAction(written.r1, written.c1, written.direction,
                    written.distance);
}

// Parses any move, looking up the piece that makes a one-square capture.
Action Parse(const ThudState& state, const std::string& move) {
  const WrittenMove written = ReadMove(move);
  if (written.capture && written.distance == 1) {
    const Cell mover = state.CellAt(written.r1, written.c1);
    SPIEL_CHECK_TRUE(mover == Cell::kDwarf || mover == Cell::kTroll);
    if (mover == Cell::kTroll) {
      return CaptureStepAction(written.r1, written.c1, written.direction);
    }
  }
  return LineAction(written.r1, written.c1, written.direction,
                    written.distance);
}

// ---------------------------------------------------------------------------
// Reading positions and classifying moves.

std::shared_ptr<const Game> Thud() { return LoadGame("thud"); }

std::unique_ptr<ThudState> FromDiagram(std::shared_ptr<const Game> game,
                                       const std::string& diagram,
                                       const std::string& status = "") {
  std::unique_ptr<State> state =
      game->NewInitialState(absl::StrCat(diagram, "\n", status));
  return std::unique_ptr<ThudState>(static_cast<ThudState*>(state.release()));
}

std::unique_ptr<ThudState> FromDiagram(const std::string& diagram,
                                       const std::string& status = "") {
  return FromDiagram(Thud(), diagram, status);
}

// A diagram's fifteen rows as ToString() writes them: no indentation.
std::string Rows(absl::string_view diagram) {
  std::vector<std::string> rows;
  for (absl::string_view line : absl::StrSplit(diagram, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    if (!line.empty()) rows.push_back(std::string(line));
  }
  SPIEL_CHECK_EQ(static_cast<int>(rows.size()), 15);
  return absl::StrJoin(rows, "\n");
}

void Require(bool condition, const std::string& what, const ThudState& state) {
  if (!condition) {
    SpielFatalError(absl::StrCat(what, " in\n", state.ToString()));
  }
}

enum class Kind { kDwarfMove, kHurl, kTrollStep, kCaptureStep, kShove };

struct Move {
  Kind kind;
  int r1, c1, r2, c2;
  int direction, distance;
};

// Decodes an action with section 8's formulas and names its kind from the
// pieces on the board. Fails if the move could not be legal under sections
// 4-5 regardless of any line lengths, so every legal action that passes
// through here is also checked for those rules.
Move Classify(const ThudState& state, Action action) {
  const auto [square, direction, distance, capture_step] = Decode(action);
  const auto [r1, c1] = RowCol(square);
  const int r2 = r1 + kDr[direction] * distance;
  const int c2 = c1 + kDc[direction] * distance;
  const std::string what = absl::StrCat("Legal action ", action, " (", r1, ",",
                                        c1, ") to (", r2, ",", c2, ")");
  Require(OnBoard(r2, c2), absl::StrCat(what, " leaves the board"), state);
  for (int k = 1; k < distance; ++k) {
    Require(state.CellAt(r1 + kDr[direction] * k, c1 + kDc[direction] * k) ==
                Cell::kEmpty,
            absl::StrCat(what, " passes over a piece or the Thudstone"), state);
  }
  const Cell mover = state.CellAt(r1, c1);
  const Cell landing = state.CellAt(r2, c2);
  Kind kind;
  if (mover == Cell::kDwarf) {
    Require(!capture_step, absl::StrCat(what, " is a dwarf capture step"),
            state);
    kind = landing == Cell::kTroll ? Kind::kHurl : Kind::kDwarfMove;
    Require(kind == Kind::kHurl || landing == Cell::kEmpty,
            absl::StrCat(what, " lands on a dwarf or the Thudstone"), state);
  } else {
    Require(mover == Cell::kTroll, absl::StrCat(what, " moves no piece"),
            state);
    Require(landing == Cell::kEmpty,
            absl::StrCat(what, " is a troll landing on something"), state);
    kind = capture_step    ? Kind::kCaptureStep
           : distance == 1 ? Kind::kTrollStep
                           : Kind::kShove;
    if (kind != Kind::kTrollStep) {
      // A troll that captures must land next to at least one dwarf.
      bool next_to_dwarf = false;
      for (int d = 0; d < 8; ++d) {
        const int r = r2 + kDr[d], c = c2 + kDc[d];
        if (OnBoard(r, c) && state.CellAt(r, c) == Cell::kDwarf) {
          next_to_dwarf = true;
        }
      }
      Require(next_to_dwarf,
              absl::StrCat(what, " captures but lands next to no dwarf"),
              state);
    }
  }
  return {kind, r1, c1, r2, c2, direction, distance};
}

std::string Notation(const Move& move) {
  const std::string from = absl::StrCat("(", move.r1, ",", move.c1, ")");
  const std::string to = absl::StrCat("(", move.r2, ",", move.c2, ")");
  switch (move.kind) {
    case Kind::kHurl:
    case Kind::kCaptureStep:
    case Kind::kShove:
      return absl::StrCat(from, "-", to, "x");
    default:
      return absl::StrCat(from, "-", to);
  }
}

// Every legal action, classified; also checks that LegalActions() is sorted
// and free of duplicates, as OpenSpiel requires.
std::vector<Move> LegalMoves(const ThudState& state) {
  const std::vector<Action> actions = state.LegalActions();
  for (size_t i = 1; i < actions.size(); ++i) {
    SPIEL_CHECK_LT(actions[i - 1], actions[i]);
  }
  std::vector<Move> moves;
  for (Action action : actions) moves.push_back(Classify(state, action));
  return moves;
}

bool IsLegal(const ThudState& state, Action action) {
  const std::vector<Action> actions = state.LegalActions();
  return std::find(actions.begin(), actions.end(), action) != actions.end();
}

// ---------------------------------------------------------------------------
// Checks.

// The legal moves of one kind are exactly `expected`, in any order.
void CheckMoves(const ThudState& state, Kind kind,
                std::vector<std::string> expected) {
  std::vector<std::string> actual;
  for (const Move& move : LegalMoves(state)) {
    if (move.kind == kind) actual.push_back(Notation(move));
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  SPIEL_CHECK_EQ(absl::StrJoin(actual, " "), absl::StrJoin(expected, " "));
}

void CheckLegal(const ThudState& state, const std::string& move) {
  if (!IsLegal(state, Parse(state, move))) {
    SpielFatalError(absl::StrCat("Expected ", move, " to be legal in\n",
                                 state.ToString()));
  }
}

void CheckNotLegal(const ThudState& state, Action action) {
  if (IsLegal(state, action)) {
    SpielFatalError(absl::StrCat("Expected action ", action,
                                 " not to be legal in\n", state.ToString()));
  }
}

void CheckNotLegal(const ThudState& state, const std::string& move) {
  if (IsLegal(state, Parse(state, move))) {
    SpielFatalError(absl::StrCat("Expected ", move, " not to be legal in\n",
                                 state.ToString()));
  }
}

// For the piece on (r, c), the legal moves or steps (captures excluded) in
// directions N, NE, E, SE, S, SW, W, NW are exactly distances 1..reach[d].
void CheckReach(const ThudState& state, int r, int c,
                const std::array<int, 8>& reach) {
  std::array<std::vector<int>, 8> distances;
  for (const Move& move : LegalMoves(state)) {
    if (move.r1 == r && move.c1 == c &&
        (move.kind == Kind::kDwarfMove || move.kind == Kind::kTrollStep)) {
      distances[move.direction].push_back(move.distance);
    }
  }
  for (int d = 0; d < 8; ++d) {
    std::vector<int> expected;
    for (int k = 1; k <= reach[d]; ++k) expected.push_back(k);
    std::sort(distances[d].begin(), distances[d].end());
    if (distances[d] != expected) {
      SpielFatalError(absl::StrCat("From (", r, ",", c, ") in direction ", d,
                                   ": moves of distances ",
                                   absl::StrJoin(distances[d], ","),
                                   ", expected 1..", reach[d], " in\n",
                                   state.ToString()));
    }
  }
}

// One piece's moves or steps (captures excluded): distances 1..distances[d]
// in directions N, NE, E, SE, S, SW, W, NW.
struct Reach {
  int r, c;
  std::array<int, 8> distances;
};

// The complete set of legal actions, of every kind: exactly the moves and
// steps given by `reaches` plus the captures listed in `captures`.
void CheckAllLegalMoves(const ThudState& state,
                        const std::vector<Reach>& reaches,
                        std::vector<std::string> captures) {
  std::vector<std::string> expected = std::move(captures);
  for (const Reach& reach : reaches) {
    for (int d = 0; d < 8; ++d) {
      for (int k = 1; k <= reach.distances[d]; ++k) {
        expected.push_back(Notation({Kind::kDwarfMove, reach.r, reach.c,
                                     reach.r + kDr[d] * k,
                                     reach.c + kDc[d] * k, d, k}));
      }
    }
  }
  std::vector<std::string> actual;
  for (const Move& move : LegalMoves(state)) actual.push_back(Notation(move));
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  SPIEL_CHECK_EQ(absl::StrJoin(actual, " "), absl::StrJoin(expected, " "));
}

// Applying `move` gives the position `diagram` with status line `status`.
void CheckAfter(const ThudState& state, const std::string& move,
                const std::string& diagram, const std::string& status) {
  CheckLegal(state, move);
  std::unique_ptr<State> next = state.Clone();
  next->ApplyAction(Parse(state, move));
  SPIEL_CHECK_EQ(next->ToString(), absl::StrCat(Rows(diagram), "\n", status));
}

// Section 7: the returns for a final margin of dwarfs minus 4 x trolls.
std::vector<double> ReturnsForMargin(double dwarf_margin) {
  return {dwarf_margin / 32, -dwarf_margin / 32};
}

// The battle is over, however it ended: nobody is to move, no action is legal,
// and the returns are those of `dwarf_margin`.
void CheckOver(const State& state, double dwarf_margin) {
  SPIEL_CHECK_TRUE(state.IsTerminal());
  SPIEL_CHECK_EQ(state.CurrentPlayer(), kTerminalPlayerId);
  SPIEL_CHECK_TRUE(state.LegalActions().empty());
  SPIEL_CHECK_EQ(state.Returns(), ReturnsForMargin(dwarf_margin));
}

// ---------------------------------------------------------------------------
// Positions.

constexpr char kInitial[] = R"(
      -----dd.dd-----
      ----d.....d----
      ---d.......d---
      --d.........d--
      -d...........d-
      d.............d
      d.....TTT.....d
      ......TOT......
      d.....TTT.....d
      d.............d
      -d...........d-
      --d.........d--
      ---d.......d---
      ----d.....d----
      -----dd.dd-----
)";

// ---------------------------------------------------------------------------
// Board, encoding and positions as text.

void TestBoardGeometry() {
  // Section 1: 165 squares; section 8: numbered 0..164 in row-major order.
  int count = 0;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      SPIEL_CHECK_EQ(IsOnBoard(r, c), OnBoard(r, c));
      if (!OnBoard(r, c)) continue;
      SPIEL_CHECK_EQ(SquareIndex(r, c), count);
      SPIEL_CHECK_EQ(SquareCoord(count).row, r);
      SPIEL_CHECK_EQ(SquareCoord(count).col, c);
      ++count;
    }
  }
  SPIEL_CHECK_EQ(count, 165);
  SPIEL_CHECK_EQ(kNumSquares, 165);
  SPIEL_CHECK_EQ(SquareIndex(0, 5), 0);
  SPIEL_CHECK_EQ(SquareIndex(1, 4), 5);
  SPIEL_CHECK_EQ(SquareIndex(7, 7), 82);  // The Thudstone.
  SPIEL_CHECK_EQ(SquareIndex(14, 9), 164);
  // Outside the 15x15 grid.
  SPIEL_CHECK_FALSE(IsOnBoard(-1, 7));
  SPIEL_CHECK_FALSE(IsOnBoard(15, 7));
  SPIEL_CHECK_FALSE(IsOnBoard(7, -1));
  SPIEL_CHECK_FALSE(IsOnBoard(7, 15));
}

void TestDirectionOrder() {
  // Section 8: N, NE, E, SE, S, SW, W, NW.
  SPIEL_CHECK_EQ(kNorth, 0);
  SPIEL_CHECK_EQ(kNorthEast, 1);
  SPIEL_CHECK_EQ(kEast, 2);
  SPIEL_CHECK_EQ(kSouthEast, 3);
  SPIEL_CHECK_EQ(kSouth, 4);
  SPIEL_CHECK_EQ(kSouthWest, 5);
  SPIEL_CHECK_EQ(kWest, 6);
  SPIEL_CHECK_EQ(kNorthWest, 7);
  for (int d = 0; d < 8; ++d) {
    SPIEL_CHECK_EQ(kRowStep[d], kDr[d]);
    SPIEL_CHECK_EQ(kColStep[d], kDc[d]);
  }
}

void TestActionEncoding() {
  // Section 8's two ranges.
  SPIEL_CHECK_EQ(kNumDistinctActions, 19800);
  SPIEL_CHECK_EQ(Thud()->NumDistinctActions(), 19800);
  for (int s = 0; s < 165; ++s) {
    for (int d = 0; d < 8; ++d) {
      for (int k = 1; k <= 14; ++k) {
        const Action action = EncodeLineAction(s, d, k);
        SPIEL_CHECK_EQ(action, (s * 8 + d) * 14 + (k - 1));
        const DecodedAction decoded = DecodeAction(action);
        SPIEL_CHECK_EQ(decoded.square, s);
        SPIEL_CHECK_EQ(decoded.direction, d);
        SPIEL_CHECK_EQ(decoded.distance, k);
        SPIEL_CHECK_FALSE(decoded.capture_step);
      }
      const Action action = EncodeCaptureStepAction(s, d);
      SPIEL_CHECK_EQ(action, 18480 + s * 8 + d);
      const DecodedAction decoded = DecodeAction(action);
      SPIEL_CHECK_EQ(decoded.square, s);
      SPIEL_CHECK_EQ(decoded.direction, d);
      SPIEL_CHECK_EQ(decoded.distance, 1);
      SPIEL_CHECK_TRUE(decoded.capture_step);
    }
  }
  // Every action ID decodes and encodes back to itself.
  for (Action action = 0; action < 19800; ++action) {
    const DecodedAction parts = DecodeAction(action);
    SPIEL_CHECK_EQ(
        parts.capture_step
            ? EncodeCaptureStepAction(parts.square, parts.direction)
            : EncodeLineAction(parts.square, parts.direction, parts.distance),
        action);
  }
}

void TestInitialPosition() {
  // Section 2: 32 dwarfs on the edge except the four squares in line with the
  // Thudstone, 8 trolls around it; section 3: dwarfs move first.
  std::unique_ptr<State> state = Thud()->NewInitialState();
  SPIEL_CHECK_EQ(
      state->ToString(),
      absl::StrCat(Rows(kInitial),
                   "\nto_move=dwarfs turns=0 turns_without_capture=0"));
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kDwarfPlayer);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  const auto& thud = static_cast<const ThudState&>(*state);
  SPIEL_CHECK_TRUE(thud.CellAt(0, 5) == Cell::kDwarf);
  SPIEL_CHECK_TRUE(thud.CellAt(0, 7) == Cell::kEmpty);
  SPIEL_CHECK_TRUE(thud.CellAt(6, 6) == Cell::kTroll);
  SPIEL_CHECK_TRUE(thud.CellAt(7, 7) == Cell::kThudstone);
  SPIEL_CHECK_EQ(thud.TurnsPlayed(), 0);
  SPIEL_CHECK_EQ(thud.TurnsWithoutCapture(), 0);
}

void TestDiagramRoundTrip() {
  // Written by ToString() and read back unchanged.
  std::shared_ptr<const Game> game = Thud();
  const std::string initial = game->NewInitialState()->ToString();
  SPIEL_CHECK_EQ(game->NewInitialState(initial)->ToString(), initial);
  // Every status field.
  constexpr char kPosition[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      .......T.......
      .......T.d.....
      ....TTTO.......
      .d.....T.......
      .......T.......
      -.............-
      --...........--
      ---.....d...---
      ----.......----
      -----.....-----
  )";
  const std::string status = "to_move=trolls turns=37 turns_without_capture=5";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition, status);
  SPIEL_CHECK_EQ(state->ToString(),
                 absl::StrCat(Rows(kPosition), "\n", status));
  SPIEL_CHECK_EQ(state->TurnsPlayed(), 37);
  SPIEL_CHECK_EQ(state->TurnsWithoutCapture(), 5);
  // Missing fields default to dwarfs to move and zero counters, and the fields
  // may come in any order.
  SPIEL_CHECK_EQ(
      FromDiagram(kPosition)->ToString(),
      absl::StrCat(Rows(kPosition),
                   "\nto_move=dwarfs turns=0 turns_without_capture=0"));
  SPIEL_CHECK_EQ(
      FromDiagram(kPosition, "turns_without_capture=2 to_move=trolls")
          ->ToString(),
      absl::StrCat(Rows(kPosition),
                   "\nto_move=trolls turns=0 turns_without_capture=2"));
}

// Reading `text` fails.
void CheckRejected(const std::string& text) {
  if (PositionFromText(text).has_value()) {
    SpielFatalError(
        absl::StrCat("Read a malformed or impossible position:\n", text));
  }
}

void TestRejectedPositions() {
  // Reading fails on malformed text and on impossible positions (thud.h).
  // Each case below is one small edit of text that reads fine, so a reader
  // that rejected everything would not pass.
  const std::vector<std::string> rows = absl::StrSplit(Rows(kInitial), '\n');
  const std::string opening = absl::StrJoin(rows, "\n");
  SPIEL_CHECK_TRUE(PositionFromText(opening).has_value());
  const absl::optional<Position> position = PositionFromText(absl::StrCat(
      opening, "\nto_move=trolls turns=5 turns_without_capture=2"));
  SPIEL_CHECK_TRUE(position.has_value());
  SPIEL_CHECK_TRUE(position->board[Square(0, 5)] == Cell::kDwarf);
  SPIEL_CHECK_EQ(position->to_move, kTrollPlayer);
  SPIEL_CHECK_EQ(position->turns_played, 5);
  SPIEL_CHECK_EQ(position->turns_without_capture, 2);
  // The opening with the square (r, c) written as `cell`.
  auto with = [&rows](int r, int c, char cell) {
    std::vector<std::string> edited = rows;
    edited[r][c] = cell;
    return absl::StrJoin(edited, "\n");
  };
  // Impossible positions.
  CheckRejected(with(7, 0, 'd'));  // 33 dwarfs.
  CheckRejected(with(5, 7, 'T'));  // 9 trolls.
  CheckRejected(with(0, 0, 'd'));  // A dwarf on a cut-off corner.
  CheckRejected(with(0, 7, '-'));  // A board square written as cut off.
  CheckRejected(with(7, 7, 'd'));  // A dwarf in the Thudstone's place.
  CheckRejected(with(5, 7, 'O'));  // A second Thudstone.
  // Malformed text.
  CheckRejected(with(7, 0, 'x'));  // An unknown character.
  // Not even '#', which starts a comment line in OpenSpiel's saved-game files.
  CheckRejected(with(0, 0, '#'));
  std::vector<std::string> short_row = rows;
  short_row[7].pop_back();
  CheckRejected(absl::StrJoin(short_row, "\n"));  // A row of 14 squares.
  CheckRejected(absl::StrJoin(rows.begin(), rows.end() - 1, "\n"));  // 14 rows.
  CheckRejected(absl::StrCat(opening, "\nmoves=3"));  // An unknown field.
  CheckRejected(absl::StrCat(opening, "\nto_move=elves"));  // An unknown value.
  CheckRejected(absl::StrCat(opening, "\nturns=-1"));  // A negative count.
}

void TestGameParameters() {
  std::shared_ptr<const Game> game = Thud();
  const GameType& type = game->GetType();
  SPIEL_CHECK_EQ(type.short_name, "thud");
  SPIEL_CHECK_TRUE(type.dynamics == GameType::Dynamics::kSequential);
  SPIEL_CHECK_TRUE(type.chance_mode == GameType::ChanceMode::kDeterministic);
  SPIEL_CHECK_TRUE(type.information ==
                   GameType::Information::kPerfectInformation);
  SPIEL_CHECK_TRUE(type.utility == GameType::Utility::kZeroSum);
  SPIEL_CHECK_TRUE(type.reward_model == GameType::RewardModel::kTerminal);
  SPIEL_CHECK_EQ(game->NumPlayers(), 2);
  // Section 6's defaults; every turn is one action (section 8).
  const auto& thud = static_cast<const ThudGame&>(*game);
  SPIEL_CHECK_EQ(thud.max_turns_without_capture(), 200);
  SPIEL_CHECK_EQ(thud.max_turns(), 800);
  SPIEL_CHECK_EQ(game->MaxGameLength(), 800);
  std::shared_ptr<const Game> custom =
      LoadGame("thud(max_turns=10,max_turns_without_capture=4)");
  SPIEL_CHECK_EQ(custom->MaxGameLength(), 10);
  SPIEL_CHECK_EQ(
      static_cast<const ThudGame&>(*custom).max_turns_without_capture(), 4);
}

// ---------------------------------------------------------------------------
// Hurls (section 4b).

constexpr char kHurlLine[] = R"(
      -----.....-----
      ----.dT....----
      ---.........---
      --..d.T......--
      -.............-
      T.ddd..T.......
      ...............
      .......O.......
      ...............
      .......T..dd...
      -.............-
      --...........--
      ---.dd.d.T..---
      ----.......----
      -----.....-----
)";

void TestHurlLineLength() {
  // The front dwarf of a line of N dwarfs travels 1..N squares onto a troll; N
  // counts the hurled dwarf and every dwarf directly behind it.
  std::unique_ptr<ThudState> state = FromDiagram(kHurlLine);
  CheckMoves(*state, Kind::kHurl,
             {
                 "(1,5)-(1,6)x",  // N = 1: a lone dwarf onto an adjacent troll.
                 "(5,4)-(5,7)x",  // Line (5,2)-(5,4), N = 3: 3 squares.
                 "(5,2)-(5,0)x",  // From the line's other end: 2 squares.
             });
  CheckNotLegal(*state, "(3,4)-(3,6)x");    // A lone dwarf: 2 squares > N = 1.
  CheckNotLegal(*state, "(9,10)-(9,7)x");   // A line of 2: 3 squares > N = 2.
  CheckNotLegal(*state, "(12,7)-(12,9)x");  // (12,6) is empty, so N = 1.
  CheckNotLegal(*state, "(5,3)-(5,7)x");    // Only the front dwarf is hurled.
}

void TestHurlBlocked() {
  constexpr char kPosition[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ...............
      ............T..
      .......O....T..
      ...............
      dddd.d.T....d..
      -...........d.-
      --..........d--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  CheckMoves(*state, Kind::kHurl,
             {
                 "(9,12)-(7,12)x",  // N = 3: onto the nearer troll.
             });
  CheckNotLegal(*state, "(9,3)-(9,7)x");    // 4 = N squares, but (9,5) blocks.
  CheckNotLegal(*state, "(9,12)-(6,12)x");  // Cannot pass over the troll.
}

void TestHurlThudstone() {
  constexpr char kPosition[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      .........T.....
      ...............
      .T..dddOdd..T..
      ......d........
      .....d.........
      -...d.........-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  // The Thudstone ends the line (7,4)-(7,6): N = 3.
  CheckMoves(*state, Kind::kHurl, {"(7,4)-(7,1)x"});
  // The Thudstone ends the line (7,8)-(7,9): N = 2, not 5.
  CheckNotLegal(*state, "(7,9)-(7,12)x");
  // The line (10,4)-(8,6) has N = 3, but the Thudstone is in the path.
  CheckNotLegal(*state, "(8,6)-(5,9)x");
  // Nothing lands on the Thudstone.
  CheckNotLegal(*state, "(7,6)-(7,7)");
  CheckNotLegal(*state, "(7,8)-(7,7)");
}

void TestHurlDirectionsAndMaxDistance() {
  constexpr char kPosition[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ddddddd......T.
      dddddd......T..
      .......O.......
      ..........T....
      ..........T....
      -.............-
      --.....d..d..--
      ---...d...d.---
      ----.d.....----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  CheckMoves(
      *state, Kind::kHurl,
      {
          "(11,10)-(9,10)x",  // Vertical: N = 2, 2 squares.
          "(11,7)-(8,10)x",   // Diagonal: N = 3, 3 squares.
          "(5,6)-(5,13)x",    // N = 7, 7 squares: the longest hurl in a row.
      });
  CheckNotLegal(*state, "(6,5)-(6,12)x");  // N = 6: 7 squares is too far.
}

void TestHurlSeveralLines() {
  // One dwarf, (4,9), is the front of several lines at once and hurls along
  // each of them. N is counted separately for each direction: 3 to the east
  // (backed by (4,8)-(4,7)) and north-east (by (5,8)-(6,7)), but 2 to the north
  // (by (5,9)) and north-west (by (5,10)).
  constexpr char kPosition[] = R"(
      -----.....-----
      ----..T....----
      ---......T.T---
      --...........--
      -......ddd..T.-
      ........ddd....
      .......d.......
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  CheckMoves(*state, Kind::kHurl,
             {
                 "(4,9)-(4,12)x",  // East: N = 3, 3 squares.
                 "(4,9)-(2,11)x",  // North-east: N = 3, 2 squares.
                 "(4,9)-(2,9)x",   // North: N = 2, 2 squares.
             });
  // North-west, N = 2: 3 squares is too far, although (4,9) heads lines of 3
  // in other directions.
  CheckNotLegal(*state, "(4,9)-(1,6)x");
}

void TestHurlBorders() {
  constexpr char kPosition[] = R"(
      -----T....-----
      ----.......----
      ---d........---
      --d..........--
      -d............-
      ddd..T.........
      ...............
      .......O.......
      ...........dd.T
      ...............
      -T.dd.........-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  CheckMoves(*state, Kind::kHurl,
             {
                 "(8,12)-(8,14)x",  // Onto a troll on the east edge.
                 "(10,3)-(10,1)x",  // Onto a troll on the west edge of row 10.
                 "(5,2)-(5,5)x",    // The line (5,0)-(5,2) ends at the edge.
                 "(2,3)-(0,5)x",    // Along the north-west edge: N = 4.
             });
  CheckNotLegal(*state, LineAction(8, 12, kEast, 3));      // Off the board.
  CheckNotLegal(*state, LineAction(2, 3, kNorthWest, 1));  // Off the board.
}

void TestHurlNoWrapAround() {
  // In rows 5-9 both ends of a row are on the board, so a board stored as a
  // flat array must not let a path or a line run from one row into the next.
  // Each position has one legal hurl as a control.
  {
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ............ddd
        TTT............
        .......O.......
        ...............
        ...............
        -.............-
        --...........--
        ---..dd.T...---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition);
    CheckMoves(*state, Kind::kHurl, {"(12,6)-(12,8)x"});
    // East of (5,14) is off the board; a flat array continues on (6,0)-(6,2).
    CheckNotLegal(*state, LineAction(5, 14, kEast, 1));
    CheckNotLegal(*state, LineAction(5, 14, kEast, 2));
    CheckNotLegal(*state, LineAction(5, 14, kEast, 3));
  }
  {
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ...............
        ..............d
        dd..T..O.......
        ..............T
        ddd............
        -.............-
        --...........--
        ---..dd.T...---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition);
    CheckMoves(*state, Kind::kHurl, {"(12,6)-(12,8)x"});
    // West of (9,0) is off the board; a flat array continues on (8,14).
    CheckNotLegal(*state, LineAction(9, 0, kWest, 1));
    // The line behind (7,1) is (7,0) only, N = 2; a flat array would count
    // (6,14) too and allow 3 squares.
    CheckNotLegal(*state, "(7,1)-(7,4)x");
  }
  {
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -...........d.-
        .............d.
        T.............d
        .......O.....d.
        T...........d..
        ...............
        -.............-
        --...........--
        ---..dd.T...---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition);
    CheckMoves(*state, Kind::kHurl, {"(12,6)-(12,8)x"});
    // Diagonally off the board from (6,14); a flat array lands on (6,0)
    // (north-east) and (8,0) (south-east).
    CheckNotLegal(*state, LineAction(6, 14, kNorthEast, 1));
    CheckNotLegal(*state, LineAction(6, 14, kSouthEast, 1));
  }
  {
    // A column-major array would wrap between columns instead.
    constexpr char kPosition[] = R"(
        -----.T..d-----
        ----.....d.----
        ---......d..---
        --...........--
        -.............-
        ...............
        ...............
        .......O.......
        ...............
        ...dd.T........
        -.............-
        --...........--
        ---..d......---
        ----.d.....----
        -----d..T.-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition);
    CheckMoves(*state, Kind::kHurl, {"(9,4)-(9,6)x"});
    CheckNotLegal(*state, LineAction(14, 5, kSouth, 1));  // Wraps to (0,6).
    CheckNotLegal(*state, LineAction(0, 9, kNorth, 1));   // Wraps to (14,8).
  }
}

void TestApplyHurl() {
  // The hurled dwarf takes the troll's square; the troll is captured.
  constexpr char kAfter[] = R"(
      -----.....-----
      ----.dT....----
      ---.........---
      --..d.T......--
      -.............-
      T.dd...d.......
      ...............
      .......O.......
      ...............
      .......T..dd...
      -.............-
      --...........--
      ---.dd.d.T..---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state =
      FromDiagram(kHurlLine, "to_move=dwarfs turns=10 turns_without_capture=5");
  CheckAfter(*state, "(5,4)-(5,7)x", kAfter,
             "to_move=trolls turns=11 turns_without_capture=0");
}

// ---------------------------------------------------------------------------
// Dwarf moves (section 4a).

void TestDwarfMoveOpenBoard() {
  // Any number of squares in each of the 8 directions, up to the edge.
  constexpr char kPosition[] = R"(
      -----.....-----
      ----....T..----
      ---.........---
      --...........--
      -.............-
      ...............
      ...............
      .......O.......
      ...............
      ....d..........
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  //                        N NE   E SE  S SW  W NW
  CheckReach(*state, 9, 4, {8, 7, 10, 5, 4, 2, 4, 4});
}

constexpr char kMoveBlocked[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -...T.........-
      ...............
      ...............
      ....d..O.......
      ...............
      ...............
      -...d.........-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

void TestDwarfMoveBlocked() {
  std::unique_ptr<ThudState> state = FromDiagram(kMoveBlocked);
  // N stops before the troll, E before the Thudstone, S before the dwarf.
  //                         N NE  E SE  S SW  W NW
  CheckReach(*state, 7, 4, {2, 6, 2, 6, 2, 3, 4, 3});
  // NE stops before the Thudstone.
  CheckReach(*state, 10, 4, {2, 2, 9, 4, 3, 1, 3, 4});
  CheckNotLegal(*state, "(7,4)-(4,4)");   // Onto a troll: a hurl, but N = 1.
  CheckNotLegal(*state, "(7,4)-(3,4)");   // Over a troll.
  CheckNotLegal(*state, "(7,4)-(10,4)");  // Onto a dwarf.
  CheckNotLegal(*state, "(7,4)-(11,4)");  // Over a dwarf.
  CheckNotLegal(*state, "(7,4)-(7,7)");   // Onto the Thudstone.
  CheckNotLegal(*state, "(7,4)-(7,8)");   // Over the Thudstone.
}

void TestDwarfMoveMaxDistanceAndEdges() {
  constexpr char kPosition[] = R"(
      -----d....-----
      ----.......----
      ---d........---
      --...........--
      -.............-
      d..............
      ...............
      .......O.......
      ...............
      d..............
      -.............-
      --...........--
      ---.......T.---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  CheckLegal(*state, "(5,0)-(5,14)");  // 14 squares: a whole row.
  CheckLegal(*state, "(0,5)-(14,5)");  // 14 squares: a whole column.
  CheckLegal(*state, "(9,0)-(0,9)");   // 9 squares: the longest diagonal.
  //                         N NE   E SE   S SW  W NW
  CheckReach(*state, 0, 5, {0, 0, 4, 9, 14, 1, 0, 0});   // North edge.
  CheckReach(*state, 2, 3, {0, 1, 8, 9, 10, 2, 0, 0});   // Diagonal edge.
  CheckReach(*state, 5, 0, {0, 2, 14, 9, 3, 0, 0, 0});   // West edge.
  CheckReach(*state, 9, 0, {3, 9, 14, 5, 0, 0, 0, 0});   // Next to a corner.
}

void TestDwarfMoveNoWrapAround() {
  // Dwarfs at the ends of rows and columns. A flat row-major array would let
  // their moves run on into the neighbouring row (east of (5,14) onto (6,0),
  // north-east of (7,14) onto (7,0), west of (9,0) onto (8,14)); a
  // column-major array would do the same between columns (south of (14,5) onto
  // (0,6), north of (0,9) onto (14,8)). The tables pin every direction of
  // every dwarf, so any such move fails the test.
  constexpr char kPosition[] = R"(
      -----....d-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ..............d
      ...............
      .......O......d
      ...............
      d..............
      -.............-
      --...........--
      ---.......T.---
      ----.......----
      -----d....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition);
  //                           N NE   E SE   S SW   W NW
  CheckReach(*state, 5, 14, {0, 0, 0, 0, 1, 8, 14, 4});
  CheckReach(*state, 7, 14, {1, 0, 0, 0, 2, 7, 6, 7});
  CheckReach(*state, 9, 0, {4, 8, 14, 4, 0, 0, 0, 0});
  CheckReach(*state, 14, 5, {14, 8, 4, 0, 0, 0, 0, 4});
  CheckReach(*state, 0, 9, {0, 0, 0, 4, 14, 8, 4, 0});
}

void TestInitialDwarfMoves() {
  // Every legal action of the opening position: each dwarf's moves in the 8
  // directions, and no hurls. The position has the board's symmetry, so the
  // 32 dwarfs are 4 kinds of 8: the rows below come in groups of 8 whose
  // distances are the same up to rotation and mirroring.
  std::unique_ptr<ThudState> state = FromDiagram(kInitial);
  //                     N  NE   E  SE   S  SW   W  NW
  CheckAllLegalMoves(*state,
                     {
                         // Next to a cut corner, on a straight edge.
                         {0, 5, {0, 0, 0, 8, 13, 0, 0, 0}},
                         {0, 9, {0, 0, 0, 0, 13, 8, 0, 0}},
                         {5, 0, {0, 0, 13, 8, 0, 0, 0, 0}},
                         {9, 0, {0, 8, 13, 0, 0, 0, 0, 0}},
                         {5, 14, {0, 0, 0, 0, 0, 8, 13, 0}},
                         {9, 14, {0, 0, 0, 0, 0, 0, 13, 8}},
                         {14, 5, {13, 8, 0, 0, 0, 0, 0, 0}},
                         {14, 9, {13, 0, 0, 0, 0, 0, 0, 8}},
                         // Next to the empty square in line with the stone.
                         {0, 6, {0, 0, 1, 7, 5, 5, 0, 0}},
                         {0, 8, {0, 0, 0, 5, 5, 7, 1, 0}},
                         {6, 0, {0, 5, 5, 7, 1, 0, 0, 0}},
                         {8, 0, {1, 7, 5, 5, 0, 0, 0, 0}},
                         {6, 14, {0, 0, 0, 0, 1, 7, 5, 5}},
                         {8, 14, {1, 0, 0, 0, 0, 5, 5, 7}},
                         {14, 6, {5, 7, 1, 0, 0, 0, 0, 5}},
                         {14, 8, {5, 5, 0, 0, 0, 0, 1, 7}},
                         // On a diagonal edge, next to a straight edge.
                         {1, 4, {0, 0, 5, 8, 11, 0, 0, 0}},
                         {1, 10, {0, 0, 0, 0, 11, 8, 5, 0}},
                         {4, 1, {0, 0, 11, 8, 5, 0, 0, 0}},
                         {4, 13, {0, 0, 0, 0, 5, 8, 11, 0}},
                         {10, 1, {5, 8, 11, 0, 0, 0, 0, 0}},
                         {10, 13, {5, 0, 0, 0, 0, 0, 11, 8}},
                         {13, 4, {11, 8, 5, 0, 0, 0, 0, 0}},
                         {13, 10, {11, 0, 0, 0, 0, 0, 5, 8}},
                         // In the middle of a diagonal edge.
                         {2, 3, {0, 0, 7, 3, 9, 0, 0, 0}},
                         {2, 11, {0, 0, 0, 0, 9, 3, 7, 0}},
                         {3, 2, {0, 0, 9, 3, 7, 0, 0, 0}},
                         {3, 12, {0, 0, 0, 0, 7, 3, 9, 0}},
                         {11, 2, {7, 3, 9, 0, 0, 0, 0, 0}},
                         {11, 12, {7, 0, 0, 0, 0, 0, 9, 3}},
                         {12, 3, {9, 3, 7, 0, 0, 0, 0, 0}},
                         {12, 11, {9, 0, 0, 0, 0, 0, 7, 3}},
                     },
                     /*captures=*/{});
  // 8 x (21 + 18 + 24 + 19) = 656, as hexparrot/thudgame's engine also counts.
  SPIEL_CHECK_EQ(static_cast<int>(state->LegalActions().size()), 656);
}

void TestApplyDwarfMove() {
  constexpr char kAfter[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -...T.........-
      ...............
      ...............
      ......dO.......
      ...............
      ...............
      -...d.........-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(
      kMoveBlocked, "to_move=dwarfs turns=3 turns_without_capture=2");
  CheckAfter(*state, "(7,4)-(7,6)", kAfter,
             "to_move=trolls turns=4 turns_without_capture=3");
  // A dwarf may end its move next to a troll: nothing is captured, since
  // pieces capture only as part of their own move (section 3), and the troll
  // on (4,4) stays.
  constexpr char kNextToTroll[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -...T.........-
      ....d..........
      ...............
      .......O.......
      ...............
      ...............
      -...d.........-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  CheckAfter(*state, "(7,4)-(5,4)", kNextToTroll,
             "to_move=trolls turns=4 turns_without_capture=3");
}

// ---------------------------------------------------------------------------
// Shoves (section 5b).

constexpr char kShoveLine[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -......d......-
      ..TTT..........
      d.......d......
      .......O.......
      ...............
      ...T..d........
      -........T....-
      --.......T...--
      ---.........---
      ----.......----
      -----.....-----
)";

void TestShoveLineLength() {
  // The end troll of a line of N trolls travels 2..N squares to an empty
  // square next to at least one dwarf.
  std::unique_ptr<ThudState> state = FromDiagram(kShoveLine, "to_move=trolls");
  CheckMoves(*state, Kind::kShove,
             {
                 "(5,4)-(5,6)x",  // Line (5,2)-(5,4), N = 3: 2 squares...
                 "(5,4)-(5,7)x",  // ...and 3, both next to a dwarf.
                 "(5,2)-(5,0)x",  // The other end: next to the dwarf at (6,0).
             });
  CheckNotLegal(*state, "(5,4)-(5,8)x");   // 4 squares > N = 3.
  CheckNotLegal(*state, "(10,9)-(8,9)x");  // No dwarf next to (8,9).
  CheckNotLegal(*state, "(9,3)-(9,5)x");   // A lone troll never shoves.
}

// A line of seven trolls along a full row, and a dwarf next to the square
// seven squares ahead of it.
constexpr char kLongestShove[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      TTTTTTT........
      ..............d
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

void TestShoveMaxDistance() {
  // Seven squares, the longest shove possible: the line (N trolls) and the
  // path (at most N squares) must fit in a row or column of 15. The perft
  // reference engine stops at 6 (see TestPerft), so only this test covers it.
  std::unique_ptr<ThudState> state =
      FromDiagram(kLongestShove, "to_move=trolls");
  CheckMoves(*state, Kind::kShove, {"(5,6)-(5,13)x"});  // N = 7, 7 squares.
  CheckNotLegal(*state, "(5,6)-(5,14)x");  // 8 squares > N = 7.
}

void TestShoveSeveralLines() {
  // One troll, (4,9), ends two lines at once and is shoved along each. N is
  // counted separately for each direction: 3 to the east (backed by
  // (4,8)-(4,7)), 2 to the north (backed by (5,9)).
  constexpr char kPosition[] = R"(
      -----.....-----
      ----....d..----
      ---.........---
      --..........d--
      -......TTT....-
      .........T.....
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition, "to_move=trolls");
  CheckMoves(*state, Kind::kShove,
             {
                 "(4,9)-(4,11)x",  // East: N = 3, 2 squares, next to (3,12).
                 "(4,9)-(4,12)x",  // East: 3 squares, next to (3,12).
                 "(4,9)-(2,9)x",   // North: N = 2, 2 squares, next to (1,8).
             });
  // North, N = 2: 3 squares is too far, although (1,9) is next to a dwarf and
  // (4,9) ends a line of 3 in another direction.
  CheckNotLegal(*state, "(4,9)-(1,9)x");
  // The line's other end: 2 squares would be allowed, but no dwarf is next to
  // (4,5).
  CheckNotLegal(*state, "(4,7)-(4,5)x");
}

void TestShoveBlockedAndThudstone() {
  // Lines that the Thudstone ends or breaks. The dwarf at (6,9) is next to
  // (7,8), so passing over the Thudstone is the only thing wrong with a shove
  // from (7,6) to (7,8).
  constexpr char kThudstone[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      .......T.......
      .......T.d.....
      ....TTTO.......
      .d.....T.......
      .......T.......
      -.............-
      --...........--
      ---.....d...---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state =
      FromDiagram(kThudstone, "to_move=trolls");
  CheckMoves(*state, Kind::kShove,
             {
                 "(7,4)-(7,2)x",   // Line (7,4)-(7,6) ends at the Thudstone...
                 "(7,4)-(7,1)x",   // ...N = 3.
                 "(9,7)-(11,7)x",  // Line (8,7)-(9,7), N = 2.
             });
  CheckNotLegal(*state, "(7,6)-(7,8)x");  // Would pass over the Thudstone.
  // 3 squares > N = 2: the trolls at (5,7)-(6,7) are beyond the Thudstone.
  CheckNotLegal(*state, "(9,7)-(12,7)x");
  // Lines whose path a dwarf blocks.
  constexpr char kBlocked[] = R"(
      -----.....-----
      ----TTTd...----
      ---.........---
      --...........--
      -.............-
      ...............
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---..TT.d...---
      ----.......----
      -----.....-----
  )";
  state = FromDiagram(kBlocked, "to_move=trolls");
  CheckMoves(*state, Kind::kShove, {});
  CheckNotLegal(*state, "(1,6)-(1,8)x");    // The dwarf at (1,7) blocks.
  CheckNotLegal(*state, "(12,6)-(12,8)x");  // Would land on a dwarf.
}

constexpr char kShoveBorders[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ....d..........
      TT.............
      .......O.......
      d.............d
      ..........TTT..
      -............d-
      --T..........--
      ---T........---
      ----T......----
      -----.....-----
)";

void TestShoveBorders() {
  std::unique_ptr<ThudState> state =
      FromDiagram(kShoveBorders, "to_move=trolls");
  CheckMoves(*state, Kind::kShove,
             {
                 "(9,12)-(9,14)x",  // Onto the east edge.
                 "(6,1)-(6,3)x",    // Line (6,0)-(6,1) ends at the edge: N = 2.
                 "(11,2)-(9,0)x",   // Along the diagonal edge to the west edge.
             });
  CheckNotLegal(*state, "(6,1)-(6,4)x");                // 3 squares > N = 2.
  CheckNotLegal(*state, LineAction(9, 12, kEast, 3));  // Off the board.
}

void TestShoveNoWrapAround() {
  {
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        .............TT
        ...............
        .d.....O....d..
        TT.........TT..
        d..............
        -.............-
        --...........--
        ---.TT......---
        ----...d...----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition, "to_move=trolls");
    CheckMoves(*state, Kind::kShove, {"(12,5)-(12,7)x"});  // The control.
    // A flat array would land on (6,1), next to the dwarf at (7,1)...
    CheckNotLegal(*state, LineAction(5, 14, kEast, 2));
    // ...and on (7,13), next to the dwarf at (7,12).
    CheckNotLegal(*state, LineAction(8, 0, kWest, 2));
    // No dwarf is next to (8,14); a flat array would count (9,0) as its
    // eastern neighbour.
    CheckNotLegal(*state, "(8,12)-(8,14)x");
  }
  {
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ...............
        ....d.........T
        TT.....O.......
        ...............
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition, "to_move=trolls");
    CheckMoves(*state, Kind::kShove, {"(7,1)-(7,3)x"});
    // N = 2; a flat array would count (6,14) as part of the line.
    CheckNotLegal(*state, "(7,1)-(7,4)x");
  }
  {
    constexpr char kPosition[] = R"(
        -----..d..-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ...............
        ...............
        .......O.......
        ...............
        ...TT..........
        -......d......-
        --...........--
        ---.........---
        ----.T.....----
        -----T....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(kPosition, "to_move=trolls");
    CheckMoves(*state, Kind::kShove, {"(9,4)-(9,6)x"});
    // A column-major array would land on (1,6), next to the dwarf at (0,7).
    CheckNotLegal(*state, LineAction(14, 5, kSouth, 2));
  }
}

void TestApplyShove() {
  {
    // Both dwarfs next to (5,7) are captured; the trolls behind stay put.
    constexpr char kAfter[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ..TT...T.......
        d..............
        .......O.......
        ...............
        ...T..d........
        -........T....-
        --.......T...--
        ---.........---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(
        kShoveLine, "to_move=trolls turns=20 turns_without_capture=9");
    CheckAfter(*state, "(5,4)-(5,7)x", kAfter,
               "to_move=dwarfs turns=21 turns_without_capture=0");
  }
  {
    // (9,14) has only four neighbouring squares on the board; both dwarfs on
    // them are captured.
    constexpr char kAfter[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ....d..........
        TT.............
        .......O.......
        d..............
        ..........TT..T
        -.............-
        --T..........--
        ---T........---
        ----T......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(
        kShoveBorders, "to_move=trolls turns=30 turns_without_capture=4");
    CheckAfter(*state, "(9,12)-(9,14)x", kAfter,
               "to_move=dwarfs turns=31 turns_without_capture=0");
  }
}

// ---------------------------------------------------------------------------
// Troll steps (section 5a).

constexpr char kStepOpen[] = R"(
      -----.....-----
      ----.d.....----
      ---.....d...---
      --....T......--
      -.............-
      ...............
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

void TestTrollStepOpen() {
  std::unique_ptr<ThudState> state = FromDiagram(kStepOpen, "to_move=trolls");
  CheckMoves(*state, Kind::kTrollStep,
             {"(3,6)-(2,5)", "(3,6)-(2,6)", "(3,6)-(2,7)", "(3,6)-(3,5)",
              "(3,6)-(3,7)", "(3,6)-(4,5)", "(3,6)-(4,6)", "(3,6)-(4,7)"});
  // Capturing is optional: a step that lands next to a dwarf exists both with
  // and without the capture.
  CheckMoves(*state, Kind::kCaptureStep,
             {"(3,6)-(2,5)x", "(3,6)-(2,6)x", "(3,6)-(2,7)x", "(3,6)-(3,7)x"});
  CheckNotLegal(*state, "(3,6)-(4,6)x");  // No dwarf next to (4,6).
  CheckNotLegal(*state, "(3,6)-(5,6)");   // A troll steps only one square.
  CheckMoves(*state, Kind::kShove, {});   // A lone troll never shoves.
}

void TestTrollStepBlocked() {
  constexpr char kPosition[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      .....d.........
      ......TT.......
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition, "to_move=trolls");
  CheckMoves(*state, Kind::kTrollStep,
             {"(6,6)-(5,6)", "(6,6)-(5,7)", "(6,6)-(6,5)", "(6,6)-(7,5)",
              "(6,6)-(7,6)", "(6,7)-(5,6)", "(6,7)-(5,7)", "(6,7)-(5,8)",
              "(6,7)-(6,8)", "(6,7)-(7,6)", "(6,7)-(7,8)"});
  CheckMoves(*state, Kind::kCaptureStep,
             {"(6,6)-(5,6)x", "(6,6)-(6,5)x", "(6,7)-(5,6)x"});
  CheckNotLegal(*state, "(6,6)-(5,5)");  // Onto a dwarf.
  CheckNotLegal(*state, "(6,6)-(6,7)");  // Onto a troll.
  CheckNotLegal(*state, "(6,6)-(7,7)");  // Onto the Thudstone.
  CheckNotLegal(*state, "(6,7)-(7,7)");
}

constexpr char kStepEdges[] = R"(
      -----.d...-----
      ----dT.....----
      ---.........---
      --T..........--
      -.............-
      .d.............
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

void TestTrollStepEdges() {
  std::unique_ptr<ThudState> state = FromDiagram(kStepEdges, "to_move=trolls");
  CheckMoves(*state, Kind::kTrollStep,
             {"(1,5)-(0,5)", "(1,5)-(1,6)", "(1,5)-(2,4)", "(1,5)-(2,5)",
              "(1,5)-(2,6)", "(3,2)-(2,3)", "(3,2)-(3,3)", "(3,2)-(4,1)",
              "(3,2)-(4,2)", "(3,2)-(4,3)"});
  CheckMoves(*state, Kind::kCaptureStep,
             {"(1,5)-(0,5)x", "(1,5)-(1,6)x", "(1,5)-(2,4)x", "(1,5)-(2,5)x",
              "(3,2)-(2,3)x", "(3,2)-(4,1)x", "(3,2)-(4,2)x"});
  CheckNotLegal(*state, LineAction(1, 5, kNorthWest, 1));  // Off the board.
  CheckNotLegal(*state, LineAction(3, 2, kWest, 1));       // Off the board.
}

void TestTrollStepNoWrapAround() {
  constexpr char kPosition[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -............T-
      d..............
      d..............
      d......O.......
      ..............T
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> state = FromDiagram(kPosition, "to_move=trolls");
  CheckMoves(*state, Kind::kTrollStep,
             {"(4,13)-(3,12)", "(4,13)-(4,12)", "(4,13)-(5,12)",
              "(4,13)-(5,13)", "(4,13)-(5,14)", "(8,14)-(7,13)",
              "(8,14)-(7,14)", "(8,14)-(8,13)", "(8,14)-(9,13)",
              "(8,14)-(9,14)"});
  // No dwarf is next to (5,14). A flat array would count (5,0), (6,0) and
  // (7,0) as its eastern neighbours, and those hold dwarfs.
  CheckMoves(*state, Kind::kCaptureStep, {});
  CheckNotLegal(*state, LineAction(8, 14, kEast, 1));       // Wraps to (9,0).
  CheckNotLegal(*state, LineAction(8, 14, kNorthEast, 1));  // Wraps to (8,0).
}

void TestInitialTrollMoves() {
  // Every legal action with the trolls to move in the opening position: 32
  // steps onto the empty squares around the trolls, no captures, no shoves.
  std::unique_ptr<ThudState> state = FromDiagram(kInitial, "to_move=trolls");
  CheckAllLegalMoves(
      *state, /*reaches=*/{},
      {"(6,6)-(5,5)", "(6,6)-(5,6)", "(6,6)-(5,7)", "(6,6)-(6,5)",
       "(6,6)-(7,5)", "(6,7)-(5,6)", "(6,7)-(5,7)", "(6,7)-(5,8)",
       "(6,8)-(5,7)", "(6,8)-(5,8)", "(6,8)-(5,9)", "(6,8)-(6,9)",
       "(6,8)-(7,9)", "(7,6)-(6,5)", "(7,6)-(7,5)", "(7,6)-(8,5)",
       "(7,8)-(6,9)", "(7,8)-(7,9)", "(7,8)-(8,9)", "(8,6)-(7,5)",
       "(8,6)-(8,5)", "(8,6)-(9,5)", "(8,6)-(9,6)", "(8,6)-(9,7)",
       "(8,7)-(9,6)", "(8,7)-(9,7)", "(8,7)-(9,8)", "(8,8)-(7,9)",
       "(8,8)-(8,9)", "(8,8)-(9,7)", "(8,8)-(9,8)", "(8,8)-(9,9)"});
}

void TestApplyTrollStep() {
  {
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---..d......---
        --..d.d......--
        -...T.d.......-
        ....dT.........
        ......d........
        .......O.......
        ...............
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    // All four dwarfs next to (4,5) are captured, diagonal and orthogonal
    // alike. The troll on (4,4) is next to the landing square too, but only
    // dwarfs are captured; (6,6), next to where the troll came from, and
    // (2,5), two squares away, stay as well.
    constexpr char kCaptureAll[] = R"(
        -----.....-----
        ----.......----
        ---..d......---
        --...........--
        -...TT........-
        ...............
        ......d........
        .......O.......
        ...............
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    // The same step without capturing leaves every piece in place.
    constexpr char kCaptureNone[] = R"(
        -----.....-----
        ----.......----
        ---..d......---
        --..d.d......--
        -...TTd.......-
        ....d..........
        ......d........
        .......O.......
        ...............
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(
        kPosition, "to_move=trolls turns=7 turns_without_capture=6");
    CheckAfter(*state, "(5,5)-(4,5)x", kCaptureAll,
               "to_move=dwarfs turns=8 turns_without_capture=0");
    CheckAfter(*state, "(5,5)-(4,5)", kCaptureNone,
               "to_move=dwarfs turns=8 turns_without_capture=7");
  }
  {
    // Next to the Thudstone: the three dwarfs are captured, the stone stays.
    constexpr char kPosition[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ...............
        .....d.d.......
        .......O.......
        .....T.d.......
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    constexpr char kAfter[] = R"(
        -----.....-----
        ----.......----
        ---.........---
        --...........--
        -.............-
        ...............
        ...............
        ......TO.......
        ...............
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(
        kPosition, "to_move=trolls turns=12 turns_without_capture=11");
    CheckAfter(*state, "(8,5)-(7,6)x", kAfter,
               "to_move=dwarfs turns=13 turns_without_capture=0");
  }
  {
    // On the north edge: both dwarfs next to (0,5) are captured.
    constexpr char kAfter[] = R"(
        -----T....-----
        ----.......----
        ---.........---
        --T..........--
        -.............-
        .d.............
        ...............
        .......O.......
        ...............
        ...............
        -.............-
        --...........--
        ---.........---
        ----.......----
        -----.....-----
    )";
    std::unique_ptr<ThudState> state = FromDiagram(
        kStepEdges, "to_move=trolls turns=5 turns_without_capture=3");
    CheckAfter(*state, "(1,5)-(0,5)x", kAfter,
               "to_move=dwarfs turns=6 turns_without_capture=0");
  }
}

// ---------------------------------------------------------------------------
// Actions as text.

void TestActionToString() {
  // Each kind of move, in thud.h's notation.
  std::unique_ptr<ThudState> dwarfs = FromDiagram(kHurlLine);
  SPIEL_CHECK_EQ(dwarfs->ActionToString(kDwarfPlayer, Parse("(5,4)-(5,7)x")),
                 "(5,4)-(5,7)x");
  SPIEL_CHECK_EQ(dwarfs->ActionToString(kDwarfPlayer, Parse("(5,4)-(4,4)")),
                 "(5,4)-(4,4)");
  std::unique_ptr<ThudState> trolls = FromDiagram(kShoveLine, "to_move=trolls");
  SPIEL_CHECK_EQ(trolls->ActionToString(kTrollPlayer, Parse("(5,4)-(5,5)")),
                 "(5,4)-(5,5)");
  SPIEL_CHECK_EQ(
      trolls->ActionToString(kTrollPlayer, Parse(*trolls, "(5,2)-(5,1)x")),
      "(5,2)-(5,1)x");
  // A one-square capture reads the same for both sides, although a dwarf's is
  // a line action (a hurl) and a troll's a capture step.
  SPIEL_CHECK_EQ(
      dwarfs->ActionToString(kDwarfPlayer, Parse(*dwarfs, "(1,5)-(1,6)x")),
      "(1,5)-(1,6)x");
  SPIEL_CHECK_EQ(trolls->ActionToString(kTrollPlayer, Parse("(5,4)-(5,7)x")),
                 "(5,4)-(5,7)x");
  // For every legal action in several positions: the notation, and
  // StringToAction() inverts it.
  std::vector<std::unique_ptr<ThudState>> states;
  states.push_back(FromDiagram(kInitial));
  states.push_back(FromDiagram(kInitial, "to_move=trolls"));
  states.push_back(FromDiagram(kHurlLine));
  states.push_back(FromDiagram(kShoveLine, "to_move=trolls"));
  states.push_back(FromDiagram(kStepOpen, "to_move=trolls"));
  for (const auto& state : states) {
    for (Action action : state->LegalActions()) {
      const std::string text =
          state->ActionToString(state->CurrentPlayer(), action);
      SPIEL_CHECK_EQ(text, Notation(Classify(*state, action)));
      SPIEL_CHECK_EQ(state->StringToAction(text), action);
    }
  }
}

// ---------------------------------------------------------------------------
// End of the battle (section 6) and scoring (section 7).

constexpr char kBoxedTroll[] = R"(
      -----Td...-----
      ----ddd....----
      ---.........---
      --...........--
      -.............-
      ...............
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

constexpr char kNoTrolls[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      .....ddd.......
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

constexpr char kNoDwarfs[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ...............
      ......T.T......
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

// kNoTrolls plus the last troll, within hurling range of the dwarfs' line.
constexpr char kLastTroll[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      .....ddd..T....
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

// kNoDwarfs plus the last two dwarfs, both next to (5,6).
constexpr char kLastDwarfs[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.....dd......-
      ...............
      ......T.T......
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
)";

// Four non-capturing turns from the opening position that return to it.
const std::vector<std::string>& QuietCycle() {
  static const auto* cycle = new std::vector<std::string>{
      "(0,5)-(1,5)", "(6,6)-(5,5)", "(1,5)-(0,5)", "(5,5)-(6,6)"};
  return *cycle;
}

// Only the trolls can run out of moves while they have pieces left, so there
// is no stuck-dwarf position to test. A dwarf next to an empty square can move
// onto it, and one next to a troll can hurl onto it (section 4b). If every
// dwarf were stuck, every square next to a dwarf would hold another dwarf or
// the Thudstone, so dwarfs would fill all 164 other squares; there are 32.
void TestEndNoLegalMove() {
  // The troll on (0,5) has no empty square next to it (the board edge and four
  // dwarfs), and a lone troll cannot shove. Four dwarfs against one troll draw.
  CheckOver(*FromDiagram(kBoxedTroll, "to_move=trolls"), 4 - 1 * 4);
  {
    // The same position with the dwarfs to move goes on.
    std::unique_ptr<ThudState> state =
        FromDiagram(kBoxedTroll, "to_move=dwarfs");
    SPIEL_CHECK_FALSE(state->IsTerminal());
    SPIEL_CHECK_EQ(state->CurrentPlayer(), kDwarfPlayer);
  }
  // A side without pieces has no move.
  CheckOver(*FromDiagram(kNoTrolls, "to_move=trolls"), 3);
  CheckOver(*FromDiagram(kNoDwarfs, "to_move=dwarfs"), -2 * 4);
  // So capturing the last opposing piece ends the battle at once.
  {
    std::unique_ptr<ThudState> state =
        FromDiagram(kLastTroll, "to_move=dwarfs");
    SPIEL_CHECK_FALSE(state->IsTerminal());
    state->ApplyAction(Parse("(5,7)-(5,10)x"));
    CheckOver(*state, 3);
  }
  {
    std::unique_ptr<ThudState> state =
        FromDiagram(kLastDwarfs, "to_move=trolls");
    SPIEL_CHECK_FALSE(state->IsTerminal());
    state->ApplyAction(Parse(*state, "(6,6)-(5,6)x"));  // Takes both dwarfs.
    CheckOver(*state, -2 * 4);
  }
}

void TestEndNoCaptureLimit() {
  std::shared_ptr<const Game> game =
      LoadGame("thud(max_turns_without_capture=4)");
  std::unique_ptr<State> state = game->NewInitialState();
  for (int turn = 1; turn <= 4; ++turn) {
    state->ApplyAction(Parse(QuietCycle()[turn - 1]));
    SPIEL_CHECK_EQ(state->IsTerminal(), turn == 4);
  }
  CheckOver(*state, 32 - 8 * 4);
  // A capture resets the count.
  std::unique_ptr<ThudState> hurl = FromDiagram(
      game, kHurlLine, "to_move=dwarfs turns=3 turns_without_capture=3");
  SPIEL_CHECK_FALSE(hurl->IsTerminal());
  hurl->ApplyAction(Parse("(5,4)-(5,7)x"));
  SPIEL_CHECK_EQ(hurl->TurnsWithoutCapture(), 0);
  SPIEL_CHECK_FALSE(hurl->IsTerminal());
}

void TestEndTurnLimit() {
  std::shared_ptr<const Game> game =
      LoadGame("thud(max_turns=6,max_turns_without_capture=100)");
  std::unique_ptr<State> state = game->NewInitialState();
  for (int turn = 1; turn <= 6; ++turn) {
    state->ApplyAction(Parse(QuietCycle()[(turn - 1) % 4]));
    SPIEL_CHECK_EQ(state->IsTerminal(), turn == 6);
  }
  CheckOver(*state, 32 - 8 * 4);
  // A position read with turns=800 is already over under the default limit.
  CheckOver(*FromDiagram(kInitial, "turns=800"), 32 - 8 * 4);
  SPIEL_CHECK_FALSE(FromDiagram(kInitial, "turns=799")->IsTerminal());
}

void TestEndDefaultNoCaptureLimit() {
  // With the default limits, exactly 200 turns without a capture end the
  // battle, long before the 800-turn limit.
  std::unique_ptr<State> state = Thud()->NewInitialState();
  for (int turn = 1; turn <= 200; ++turn) {
    state->ApplyAction(Parse(QuietCycle()[(turn - 1) % 4]));
    SPIEL_CHECK_EQ(state->IsTerminal(), turn == 200);
  }
  CheckOver(*state, 32 - 8 * 4);
  // A position read with 200 turns without a capture is already over.
  CheckOver(*FromDiagram(kInitial, "turns=250 turns_without_capture=200"),
            32 - 8 * 4);
  SPIEL_CHECK_FALSE(
      FromDiagram(kInitial, "turns=250 turns_without_capture=199")
          ->IsTerminal());
}

void TestReturns() {
  std::shared_ptr<const Game> game = Thud();
  SPIEL_CHECK_EQ(game->MinUtility(), -1);
  SPIEL_CHECK_EQ(game->MaxUtility(), 1);
  SPIEL_CHECK_EQ(*game->UtilitySum(), 0);
  SPIEL_CHECK_EQ(game->NewInitialState()->Returns(), ReturnsForMargin(0));
  // The official example: five dwarfs and two trolls score 5 against 8, so the
  // trolls win by 3.
  constexpr char kExample[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ddddd..........
      ...............
      .......O.......
      ...............
      .....TT........
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  std::unique_ptr<ThudState> example = FromDiagram(kExample, "turns=800");
  SPIEL_CHECK_TRUE(example->IsTerminal());
  SPIEL_CHECK_EQ(example->Returns(), ReturnsForMargin(5 - 2 * 4));
  SPIEL_CHECK_EQ(FromDiagram(kNoTrolls, "to_move=trolls")->Returns(),
                 ReturnsForMargin(3));
  SPIEL_CHECK_EQ(FromDiagram(kNoDwarfs, "to_move=dwarfs")->Returns(),
                 ReturnsForMargin(-2 * 4));
  // The largest margins are exactly +1 and -1.
  constexpr char kAllDwarfs[] = R"(
      -----dd.dd-----
      ----d.....d----
      ---d.......d---
      --d.........d--
      -d...........d-
      d.............d
      d.............d
      .......O.......
      d.............d
      d.............d
      -d...........d-
      --d.........d--
      ---d.......d---
      ----d.....d----
      -----dd.dd-----
  )";
  constexpr char kAllTrolls[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ...............
      ......TTT......
      ......TOT......
      ......TTT......
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  SPIEL_CHECK_EQ(FromDiagram(kAllDwarfs, "to_move=trolls")->Returns(),
                 std::vector<double>({1.0, -1.0}));
  SPIEL_CHECK_EQ(FromDiagram(kAllTrolls, "to_move=dwarfs")->Returns(),
                 std::vector<double>({-1.0, 1.0}));
}

// ---------------------------------------------------------------------------
// Observations and information states (thud.h; thud/PLAN.md Phase 3: the
// observation must include the side to move and both counters).

void TestObservation() {
  std::shared_ptr<const Game> game =
      LoadGame("thud(max_turns=10,max_turns_without_capture=4)");
  SPIEL_CHECK_EQ(game->ObservationTensorShape(), std::vector<int>({6, 15, 15}));
  std::unique_ptr<State> state = game->NewInitialState();
  std::vector<float> values(game->ObservationTensorSize());
  auto at = [&values](int plane, int r, int c) {
    return values[(plane * 15 + r) * 15 + c];
  };
  state->ObservationTensor(kDwarfPlayer, absl::MakeSpan(values));
  SPIEL_CHECK_EQ(at(kDwarfPlane, 0, 5), 1);
  SPIEL_CHECK_EQ(at(kTrollPlane, 6, 6), 1);
  SPIEL_CHECK_EQ(at(kEmptyPlane, 0, 7), 1);
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      // A dwarf, a troll or an empty square is 1 in exactly one of planes 0-2.
      // A hole, a cut-off corner or the Thudstone's square, is 0 in all three:
      // only the empty plane tells it from a square a piece could move to.
      const bool hole = !OnBoard(r, c) || (r == 7 && c == 7);
      SPIEL_CHECK_EQ(at(kDwarfPlane, r, c) + at(kTrollPlane, r, c) +
                         at(kEmptyPlane, r, c),
                     hole ? 0 : 1);
      // Dwarfs to move, nothing played yet.
      SPIEL_CHECK_EQ(at(kTrollsToMovePlane, r, c), 0);
      SPIEL_CHECK_EQ(at(kNoCaptureCountPlane, r, c), 0);
      SPIEL_CHECK_EQ(at(kTurnCountPlane, r, c), 0);
    }
  }
  // After one turn without a capture: trolls to move, 1 of 4 turns without a
  // capture, 1 of 10 turns. Both players observe the same.
  state->ApplyAction(Parse("(0,5)-(1,5)"));
  for (Player player : {kDwarfPlayer, kTrollPlayer}) {
    state->ObservationTensor(player, absl::MakeSpan(values));
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 15; ++c) {
        SPIEL_CHECK_EQ(at(kTrollsToMovePlane, r, c), 1);
        SPIEL_CHECK_EQ(at(kNoCaptureCountPlane, r, c), 1.0f / 4);
        SPIEL_CHECK_EQ(at(kTurnCountPlane, r, c), 1.0f / 10);
      }
    }
    // The observation string is the whole position, counters included.
    SPIEL_CHECK_EQ(state->ObservationString(player), state->ToString());
  }
}

void TestInformationState() {
  // Both players see every move, so each one's information state is the
  // whole history: the actions played so far, as in tic-tac-toe, chess and Go.
  std::unique_ptr<State> state = Thud()->NewInitialState();
  const Action first = Parse("(0,5)-(1,5)");
  const Action second = Parse("(6,6)-(5,5)");
  for (Player player : {kDwarfPlayer, kTrollPlayer}) {
    SPIEL_CHECK_EQ(state->InformationStateString(player), "");
  }
  state->ApplyAction(first);
  state->ApplyAction(second);
  for (Player player : {kDwarfPlayer, kTrollPlayer}) {
    SPIEL_CHECK_EQ(state->InformationStateString(player),
                   absl::StrCat(first, ", ", second));
  }
  // A game read from a diagram starts its history there.
  SPIEL_CHECK_EQ(FromDiagram(kHurlLine)->InformationStateString(kDwarfPlayer),
                 "");
}

// Restoring a saved game, directly and from an OpenSpiel saved-game file, gives
// back the same position and the same history.
void CheckRestored(const Game& game, const State& state) {
  std::unique_ptr<State> restored = game.DeserializeState(state.Serialize());
  SPIEL_CHECK_EQ(restored->ToString(), state.ToString());
  SPIEL_CHECK_EQ(restored->History(), state.History());
  const auto [file_game, file_state] =
      DeserializeGameAndState(SerializeGameAndState(game, state));
  SPIEL_CHECK_EQ(file_game->ToString(), game.ToString());
  SPIEL_CHECK_EQ(file_state->ToString(), state.ToString());
  SPIEL_CHECK_EQ(file_state->History(), state.History());
}

void TestSerialize() {
  // A game from the opening saves just its actions, as OpenSpiel does by
  // default.
  std::shared_ptr<const Game> game = Thud();
  std::unique_ptr<State> opening = game->NewInitialState();
  opening->ApplyAction(Parse("(0,5)-(1,5)"));
  opening->ApplyAction(Parse("(6,6)-(5,5)"));
  SPIEL_CHECK_EQ(opening->Serialize(),
                 absl::StrCat(Parse("(0,5)-(1,5)"), "\n",
                              Parse("(6,6)-(5,5)"), "\n"));
  CheckRestored(*game, *opening);
  // A game read from text saves that position first, then its actions. The
  // saved-game file would drop any line starting with '#' as a comment, which
  // is why the text writes the cut-off corners as '-'.
  std::unique_ptr<ThudState> read =
      FromDiagram(kHurlLine, "to_move=dwarfs turns=7 turns_without_capture=3");
  const std::string start = read->ToString();
  read->ApplyAction(Parse("(5,4)-(5,7)x"));
  SPIEL_CHECK_EQ(read->Serialize(),
                 absl::StrCat(start, "\n", Parse("(5,4)-(5,7)x"), "\n"));
  CheckRestored(*game, *read);
}

// ---------------------------------------------------------------------------
// Whole positions: perft, symmetry and random play.

// Many interactions at once: a line of six trolls whose front troll can shove
// 2-6 squares, a line of five dwarfs along a diagonal edge, dwarfs and a troll
// around the Thudstone, and pieces at the ends of rows.
constexpr char kTangled[] = R"(
      -----d....-----
      ----d......----
      ---d........---
      --d..........--
      -d............-
      .....d.......Td
      d......T.......
      .......Od....d.
      .......ddd...d.
      .TTTTTT......d.
      -.........d..d-
      --...........--
      ---.........---
      ----.......----
      -----.ddd.-----
)";

// From a game of hexparrot/thudgame's AI against itself (seed 0, lookahead 3),
// after 17 turns, with the trolls to move.
constexpr char kMidgame[] = R"(
      -----dd.d.-----
      ----.......----
      ---........d---
      --..........d--
      -..T..........-
      dd.............
      d......TT.....d
      .d.....OT......
      dd....TT......d
      dd......d......
      -d...........d-
      --d.........d--
      ---d.......d---
      ----d.....d----
      -----dd..d-----
)";

// The number of move sequences of exactly `depth` turns ("perft", as in
// games/chess/chess_test.cc).
uint64_t Perft(const State& state, int depth) {
  const std::vector<Action> actions = state.LegalActions();
  if (depth == 1) return actions.size();
  uint64_t count = 0;
  for (Action action : actions) count += Perft(*state.Child(action), depth - 1);
  return count;
}

// The number of legal actions of each kind.
void CheckKindCounts(const ThudState& state, int dwarf_moves, int hurls,
                     int troll_steps, int capture_steps, int shoves) {
  std::array<int, 5> counts = {};
  for (const Move& move : LegalMoves(state)) {
    ++counts[static_cast<int>(move.kind)];
  }
  SPIEL_CHECK_EQ(counts[static_cast<int>(Kind::kDwarfMove)], dwarf_moves);
  SPIEL_CHECK_EQ(counts[static_cast<int>(Kind::kHurl)], hurls);
  SPIEL_CHECK_EQ(counts[static_cast<int>(Kind::kTrollStep)], troll_steps);
  SPIEL_CHECK_EQ(counts[static_cast<int>(Kind::kCaptureStep)], capture_steps);
  SPIEL_CHECK_EQ(counts[static_cast<int>(Kind::kShove)], shoves);
}

void TestPerft() {
  // Reference counts from the independent Thud engine of hexparrot/thudgame
  // (commit 7b171108), computed on 2026-09-23 by
  // thud/experiments/perft_reference.py. Its moves correspond one-to-one to our
  // actions, with one known difference: it tries hurls and shoves of at most 6
  // squares, where our rules allow 7 along a full row or column. No position
  // reached in these counts has a line of more than 6 pieces, so that
  // difference does not affect them.
  std::unique_ptr<ThudState> opening = FromDiagram(kInitial);
  SPIEL_CHECK_EQ(Perft(*opening, 1), 656);
  SPIEL_CHECK_EQ(Perft(*opening, 2), 22624);
  SPIEL_CHECK_EQ(Perft(*opening, 3), 14142624);
  std::unique_ptr<ThudState> opening_trolls =
      FromDiagram(kInitial, "to_move=trolls");
  SPIEL_CHECK_EQ(Perft(*opening_trolls, 1), 32);
  SPIEL_CHECK_EQ(Perft(*opening_trolls, 2), 20360);
  std::unique_ptr<ThudState> tangled = FromDiagram(kTangled);
  CheckKindCounts(*tangled, /*dwarf_moves=*/376, /*hurls=*/4,
                  /*troll_steps=*/0, /*capture_steps=*/0, /*shoves=*/0);
  SPIEL_CHECK_EQ(Perft(*tangled, 2), 24359);
  std::unique_ptr<ThudState> tangled_trolls =
      FromDiagram(kTangled, "to_move=trolls");
  CheckKindCounts(*tangled_trolls, /*dwarf_moves=*/0, /*hurls=*/0,
                  /*troll_steps=*/48, /*capture_steps=*/11, /*shoves=*/5);
  SPIEL_CHECK_EQ(Perft(*tangled_trolls, 2), 24155);
  std::unique_ptr<ThudState> midgame = FromDiagram(kMidgame, "to_move=trolls");
  CheckKindCounts(*midgame, /*dwarf_moves=*/0, /*hurls=*/0,
                  /*troll_steps=*/32, /*capture_steps=*/7, /*shoves=*/1);
  SPIEL_CHECK_EQ(Perft(*midgame, 2), 20141);
  SPIEL_CHECK_EQ(Perft(*midgame, 3), 849507);
}

// The board's 8 symmetries: quarter turns and mirror images leave the octagon
// and the Thudstone unchanged. Each is a matrix acting on (row, col) offsets
// from the Thudstone.
constexpr std::array<std::array<int, 4>, 8> kSymmetries = {{{1, 0, 0, 1},
                                                            {0, 1, -1, 0},
                                                            {-1, 0, 0, -1},
                                                            {0, -1, 1, 0},
                                                            {1, 0, 0, -1},
                                                            {-1, 0, 0, 1},
                                                            {0, 1, 1, 0},
                                                            {0, -1, -1, 0}}};

std::array<int, 2> Map(const std::array<int, 4>& m, int x, int y) {
  return {m[0] * x + m[1] * y, m[2] * x + m[3] * y};
}

// The position as it looks after the symmetry `m`.
std::unique_ptr<ThudState> Transform(const ThudState& state,
                                     const std::array<int, 4>& m) {
  std::vector<std::string> rows(15, std::string(15, '-'));
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      if (!OnBoard(r, c)) continue;
      const auto [x, y] = Map(m, r - 7, c - 7);
      SPIEL_CHECK_TRUE(OnBoard(x + 7, y + 7));
      rows[x + 7][y + 7] = ".dTO"[static_cast<int>(state.CellAt(r, c))];
    }
  }
  const std::string text = state.ToString();
  return FromDiagram(state.GetGame(), absl::StrJoin(rows, "\n"),
                     text.substr(text.rfind('\n') + 1));
}

// The action as it looks after the symmetry `m`.
Action TransformAction(Action action, const std::array<int, 4>& m) {
  const Parts parts = Decode(action);
  const auto [r, c] = RowCol(parts.square);
  const auto [x, y] = Map(m, r - 7, c - 7);
  const auto [dr, dc] =
      Map(m, kDr[parts.direction], kDc[parts.direction]);
  int direction = 0;
  while (kDr[direction] != dr || kDc[direction] != dc) ++direction;
  return parts.capture_step
             ? CaptureStepAction(x + 7, y + 7, direction)
             : LineAction(x + 7, y + 7, direction, parts.distance);
}

// The legal actions of every mirror image of `state` are the mirror images of
// its legal actions.
void CheckSymmetric(const ThudState& state) {
  const std::vector<Action> actions = state.LegalActions();
  for (const auto& m : kSymmetries) {
    std::vector<Action> expected;
    for (Action action : actions) {
      expected.push_back(TransformAction(action, m));
    }
    std::sort(expected.begin(), expected.end());
    std::unique_ptr<ThudState> image = Transform(state, m);
    if (image->LegalActions() != expected) {
      SpielFatalError(absl::StrCat(
          "The legal actions of a mirror image differ. Position:\n",
          state.ToString(), "\nMirror image:\n", image->ToString()));
    }
  }
}

// Playing `action` and then mirroring gives the same position as mirroring
// and then playing the mirrored action.
void CheckSymmetricApply(const ThudState& state, Action action) {
  std::unique_ptr<State> child = state.Child(action);
  for (const auto& m : kSymmetries) {
    SPIEL_CHECK_EQ(
        Transform(static_cast<const ThudState&>(*child), m)->ToString(),
        Transform(state, m)->Child(TransformAction(action, m))->ToString());
  }
}

// A uniformly random legal action. The random games use fixed seeds, as
// OpenSpiel's own tests do, so that a failure can be replayed.
Action RandomAction(const State& state, std::mt19937& rng) {
  const std::vector<Action> actions = state.LegalActions();
  // A battle that is not over always has a legal action (section 6).
  SPIEL_CHECK_FALSE(actions.empty());
  return actions[std::uniform_int_distribution<int>(
      0, static_cast<int>(actions.size()) - 1)(rng)];
}

void TestSymmetry() {
  // The mirroring code above computes every expected value in this test, so
  // check it first: if it emptied the board, or left positions as they are,
  // the comparisons below could pass without checking anything. Each
  // symmetry moves the 165 squares one-to-one onto the board, and no two are
  // the same: they take (0,5) to 8 different squares.
  auto image = [](const std::array<int, 4>& m, int r, int c) {
    const auto [x, y] = Map(m, r - 7, c - 7);
    return Square(x + 7, y + 7);  // Checks that it is on the board.
  };
  std::array<bool, 165> reached_from_0_5 = {};
  for (const auto& m : kSymmetries) {
    std::array<bool, 165> taken = {};
    for (int square = 0; square < 165; ++square) {
      const auto [r, c] = RowCol(square);
      SPIEL_CHECK_FALSE(taken[image(m, r, c)]);
      taken[image(m, r, c)] = true;
    }
    SPIEL_CHECK_FALSE(reached_from_0_5[image(m, 0, 5)]);
    reached_from_0_5[image(m, 0, 5)] = true;
  }
  // And one symmetry worked out by hand: the quarter turn kSymmetries[1] takes
  // (r, c) to (c, 14 - r), for positions and moves alike.
  constexpr char kBefore[] = R"(
      -----d....-----
      ----.......----
      ---...T.....---
      --...........--
      -.............-
      ...............
      ...............
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  constexpr char kAfter[] = R"(
      -----.....-----
      ----.......----
      ---.........---
      --...........--
      -.............-
      ..............d
      ............T..
      .......O.......
      ...............
      ...............
      -.............-
      --...........--
      ---.........---
      ----.......----
      -----.....-----
  )";
  SPIEL_CHECK_EQ(Transform(*FromDiagram(kBefore), kSymmetries[1])->ToString(),
                 FromDiagram(kAfter)->ToString());
  SPIEL_CHECK_EQ(TransformAction(Parse("(0,5)-(0,9)"), kSymmetries[1]),
                 Parse("(5,14)-(9,14)"));
  SPIEL_CHECK_EQ(
      TransformAction(CaptureStepAction(2, 6, kNorth), kSymmetries[1]),
      CaptureStepAction(6, 12, kEast));
  // The positions above, with each side to move.
  for (const char* diagram : {kInitial, kHurlLine, kMoveBlocked, kShoveLine,
                              kLongestShove, kShoveBorders, kStepOpen,
                              kStepEdges, kTangled, kMidgame}) {
    for (const char* status : {"to_move=dwarfs", "to_move=trolls"}) {
      CheckSymmetric(*FromDiagram(diagram, status));
    }
  }
  // Every position of ten random games, and every move played in them.
  std::mt19937 rng(/*seed=*/20260923);
  for (int game = 0; game < 10; ++game) {
    std::unique_ptr<State> state = Thud()->NewInitialState();
    while (!state->IsTerminal()) {
      const auto& thud = static_cast<const ThudState&>(*state);
      CheckSymmetric(thud);
      const Action action = RandomAction(thud, rng);
      CheckSymmetricApply(thud, action);
      state->ApplyAction(action);
    }
  }
}

int CountPieces(const ThudState& state, Cell cell) {
  int count = 0;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      if (OnBoard(r, c) && state.CellAt(r, c) == cell) ++count;
    }
  }
  return count;
}

// The number of squares next to (r, c) that hold `cell`.
int CountNextTo(const ThudState& state, int r, int c, Cell cell) {
  int count = 0;
  for (int d = 0; d < 8; ++d) {
    if (OnBoard(r + kDr[d], c + kDc[d]) &&
        state.CellAt(r + kDr[d], c + kDc[d]) == cell) {
      ++count;
    }
  }
  return count;
}

// Whether the side to move has a legal move, decided from the pieces alone.
// The dwarfs have one while a dwarf is left (see TestEndNoLegalMove). The
// trolls have one while a troll has an empty square next to it, since every
// troll move starts onto such a square.
bool CanMove(const ThudState& state, bool dwarfs_to_move) {
  if (dwarfs_to_move) return CountPieces(state, Cell::kDwarf) > 0;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      if (OnBoard(r, c) && state.CellAt(r, c) == Cell::kTroll &&
          CountNextTo(state, r, c, Cell::kEmpty) > 0) {
        return true;
      }
    }
  }
  return false;
}

// What each player sees (thud.h): planes 0-2 of the observation match the
// board, planes 3-5 the side to move and the counters, and the strings are the
// position and the move history. `trolls_to_move` is the side to move, or
// after the battle has ended the side that would have moved.
void CheckWhatPlayersSee(const ThudState& state, bool trolls_to_move) {
  const auto& game = static_cast<const ThudGame&>(*state.GetGame());
  std::vector<float> values(game.ObservationTensorSize());
  auto at = [&values](int plane, int r, int c) {
    return values[(plane * 15 + r) * 15 + c];
  };
  for (Player player : {kDwarfPlayer, kTrollPlayer}) {
    state.ObservationTensor(player, absl::MakeSpan(values));
    for (int r = 0; r < 15; ++r) {
      for (int c = 0; c < 15; ++c) {
        // Holes, the Thudstone's square and the cut-off corners, are in none
        // of planes 0-2.
        const Cell cell =
            OnBoard(r, c) ? state.CellAt(r, c) : Cell::kThudstone;
        SPIEL_CHECK_EQ(at(kDwarfPlane, r, c), cell == Cell::kDwarf ? 1 : 0);
        SPIEL_CHECK_EQ(at(kTrollPlane, r, c), cell == Cell::kTroll ? 1 : 0);
        SPIEL_CHECK_EQ(at(kEmptyPlane, r, c), cell == Cell::kEmpty ? 1 : 0);
        SPIEL_CHECK_EQ(at(kTrollsToMovePlane, r, c), trolls_to_move ? 1 : 0);
        SPIEL_CHECK_FLOAT_EQ(at(kNoCaptureCountPlane, r, c),
                             static_cast<double>(state.TurnsWithoutCapture()) /
                                 game.max_turns_without_capture());
        SPIEL_CHECK_FLOAT_EQ(
            at(kTurnCountPlane, r, c),
            static_cast<double>(state.TurnsPlayed()) / game.max_turns());
      }
    }
    SPIEL_CHECK_EQ(state.ObservationString(player), state.ToString());
    SPIEL_CHECK_EQ(state.InformationStateString(player), state.HistoryString());
  }
}

void TestRandomPlay() {
  // Ten random games (sections 3-7). In every position:
  //   - the battle is over exactly when one of the endings of section 6
  //     holds, and then the returns are those of the pieces left;
  //   - otherwise the players take turns, the dwarfs first;
  //   - both players see the position as they should, also after the battle
  //     has ended (CheckWhatPlayersSee).
  // After every move:
  //   - the turn count goes up by one and the piece arrives on its landing
  //     square, leaving its own square empty;
  //   - a dwarf move or a troll step without capture removes nothing, and
  //     counts one more turn without a capture;
  //   - a hurl removes exactly one troll, a capturing step or a shove exactly
  //     the dwarfs next to its landing square, and each resets that count;
  //   - so the moving side never loses a piece.
  std::mt19937 rng(/*seed=*/20260924);
  std::array<int, 5> played = {};  // The number of moves played, by kind.
  for (int game = 0; game < 10; ++game) {
    std::unique_ptr<State> state = Thud()->NewInitialState();
    const auto& thud = static_cast<const ThudState&>(*state);
    while (true) {
      const bool dwarfs_to_move = thud.TurnsPlayed() % 2 == 0;
      CheckWhatPlayersSee(thud, !dwarfs_to_move);
      // The endings of section 6, with the default limits.
      const bool over = thud.TurnsPlayed() >= 800 ||
                        thud.TurnsWithoutCapture() >= 200 ||
                        !CanMove(thud, dwarfs_to_move);
      SPIEL_CHECK_EQ(thud.IsTerminal(), over);
      if (over) break;
      SPIEL_CHECK_EQ(thud.CurrentPlayer(),
                     dwarfs_to_move ? kDwarfPlayer : kTrollPlayer);
      const Action action = RandomAction(thud, rng);
      const Move move = Classify(thud, action);
      ++played[static_cast<int>(move.kind)];
      const Cell mover = thud.CellAt(move.r1, move.c1);
      const int dwarfs = CountPieces(thud, Cell::kDwarf);
      const int trolls = CountPieces(thud, Cell::kTroll);
      const int next_to_landing =
          CountNextTo(thud, move.r2, move.c2, Cell::kDwarf);
      const int turns = thud.TurnsPlayed();
      const int without_capture = thud.TurnsWithoutCapture();
      int dwarfs_captured = 0;
      int trolls_captured = 0;
      if (move.kind == Kind::kHurl) trolls_captured = 1;
      if (move.kind == Kind::kCaptureStep || move.kind == Kind::kShove) {
        dwarfs_captured = next_to_landing;
      }
      state->ApplyAction(action);
      SPIEL_CHECK_EQ(thud.TurnsPlayed(), turns + 1);
      SPIEL_CHECK_TRUE(thud.CellAt(move.r1, move.c1) == Cell::kEmpty);
      SPIEL_CHECK_TRUE(thud.CellAt(move.r2, move.c2) == mover);
      SPIEL_CHECK_EQ(CountPieces(thud, Cell::kDwarf), dwarfs - dwarfs_captured);
      SPIEL_CHECK_EQ(CountPieces(thud, Cell::kTroll), trolls - trolls_captured);
      SPIEL_CHECK_EQ(thud.TurnsWithoutCapture(),
                     dwarfs_captured + trolls_captured > 0
                         ? 0
                         : without_capture + 1);
    }
    CheckOver(thud, CountPieces(thud, Cell::kDwarf) -
                        4 * CountPieces(thud, Cell::kTroll));
  }
  // Every kind of move was played, so every check above was exercised. Ten
  // games are plenty: in 1,000 random games with hexparrot/thudgame's engine
  // (thud/experiments/move_kinds_sim.py), the rarest kinds, hurls and shoves,
  // occurred in 78% and 89% of the games, so ten games miss one with a chance
  // below one in a million. If a kind is ever missing here, play more games.
  SPIEL_CHECK_GT(played[static_cast<int>(Kind::kDwarfMove)], 0);
  SPIEL_CHECK_GT(played[static_cast<int>(Kind::kHurl)], 0);
  SPIEL_CHECK_GT(played[static_cast<int>(Kind::kTrollStep)], 0);
  SPIEL_CHECK_GT(played[static_cast<int>(Kind::kCaptureStep)], 0);
  SPIEL_CHECK_GT(played[static_cast<int>(Kind::kShove)], 0);
}

}  // namespace
}  // namespace thud
}  // namespace open_spiel

int main(int argc, char** argv) {
  namespace thud = open_spiel::thud;
  // In the order of thud/PLAN.md Phase 3, so that while the implementation is
  // being written the first failure is always in the part being worked on.
  thud::TestBoardGeometry();
  thud::TestDirectionOrder();
  thud::TestActionEncoding();
  thud::TestInitialPosition();
  thud::TestDiagramRoundTrip();
  thud::TestRejectedPositions();
  thud::TestGameParameters();
  thud::TestHurlLineLength();
  thud::TestHurlBlocked();
  thud::TestHurlThudstone();
  thud::TestHurlDirectionsAndMaxDistance();
  thud::TestHurlSeveralLines();
  thud::TestHurlBorders();
  thud::TestHurlNoWrapAround();
  thud::TestApplyHurl();
  thud::TestDwarfMoveOpenBoard();
  thud::TestDwarfMoveBlocked();
  thud::TestDwarfMoveMaxDistanceAndEdges();
  thud::TestDwarfMoveNoWrapAround();
  thud::TestInitialDwarfMoves();
  thud::TestApplyDwarfMove();
  thud::TestShoveLineLength();
  thud::TestShoveMaxDistance();
  thud::TestShoveSeveralLines();
  thud::TestShoveBlockedAndThudstone();
  thud::TestShoveBorders();
  thud::TestShoveNoWrapAround();
  thud::TestApplyShove();
  thud::TestTrollStepOpen();
  thud::TestTrollStepBlocked();
  thud::TestTrollStepEdges();
  thud::TestTrollStepNoWrapAround();
  thud::TestInitialTrollMoves();
  thud::TestApplyTrollStep();
  thud::TestActionToString();
  thud::TestEndNoLegalMove();
  thud::TestEndNoCaptureLimit();
  thud::TestEndTurnLimit();
  thud::TestEndDefaultNoCaptureLimit();
  thud::TestReturns();
  thud::TestObservation();
  thud::TestInformationState();
  thud::TestSerialize();
  thud::TestPerft();
  thud::TestSymmetry();
  thud::TestRandomPlay();
}
