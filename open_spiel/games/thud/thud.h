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

#ifndef OPEN_SPIEL_GAMES_THUD_THUD_H_
#define OPEN_SPIEL_GAMES_THUD_THUD_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Thud: an asymmetric two-player battle between 32 dwarfs and 8 trolls on an
// octagonal board, with an immovable stone in the centre.
//
// Summary (the precise ruleset this implements, including the readings chosen
// where published rules are ambiguous, is thud/THUD_RULES.md in this fork):
//   - Dwarfs (player 0) move first. A dwarf slides any distance in a straight
//     line, or, as the front piece of a straight line of dwarfs, is thrown up
//     to that many squares to land on, and capture, a troll ("hurl").
//   - A troll steps one square, or, as the end piece of a straight line of
//     trolls, is pushed two up to that many squares ("shove"). A troll that
//     lands next to dwarfs captures all of them; after a one-square step it
//     may instead capture none. A shove must capture.
//   - The battle ends when the side to move has no move, or after a number of
//     turns without a capture, or after a total number of turns. Each dwarf is
//     worth 1 and each troll 4; the return is the margin divided by 32.
//
// Parameters:
//   "max_turns_without_capture"  int  consecutive turns without any capture
//                                     that end the battle (default 200)
//   "max_turns"                  int  turns that end the battle (default 800)
//
// Positions as text, written by ToString() and read by PositionFromText() and
// ThudGame::NewInitialState(const std::string&): fifteen rows of fifteen
// characters, row 0 (north) first, then a status line:
//
//   '-' not part of the board (a cut-off corner)   '.' empty square
//   'd' dwarf   'T' troll   'O' the Thudstone (always at row 7, column 7)
//
//   to_move=dwarfs turns=0 turns_without_capture=0
//
// The cut-off corners are '-' rather than '#' because OpenSpiel's saved-game
// files treat a line starting with '#' as a comment (see Serialize() below).
//
// When reading, the status line and each of its fields are optional (the
// defaults are those shown), the fields may come in any order, and blank lines
// and spaces around a row are ignored. Reading fails on malformed text (a row
// of the wrong length, the wrong number of rows, an unknown character, an
// unknown status field or value, a negative count) and on impossible
// positions: '-' anywhere but exactly the cut-off corners, anything but the
// Thudstone at row 7, column 7, a second Thudstone, or more than 32 dwarfs or
// 8 trolls. PositionFromText() then returns nullopt, and NewInitialState()
// stops with a fatal error, as chess does for an invalid FEN.
//
// Saved games (Serialize()): the actions played, one per line, as OpenSpiel
// saves by default. A game that started from a position other than the opening
// first saves that position, in the text format above, and then its actions,
// much as chess saves its starting FEN; DeserializeState() tells the two
// apart by the first character.
//
// Actions as text (ActionToString), squares written (row,col), with a trailing
// "x" for a move that captures:
//   "(r,c)-(r,c)"   a dwarf move, or a troll step that captures nothing
//   "(r,c)-(r,c)x"  a hurl, which captures the troll on the landing square; a
//                   troll step that captures all adjacent dwarfs; or a shove

namespace open_spiel {
namespace thud {

// Players.
inline constexpr int kNumPlayers = 2;
inline constexpr Player kDwarfPlayer = 0;
inline constexpr Player kTrollPlayer = 1;

// Board geometry: a 15x15 grid minus a 15-square triangle at each corner.
inline constexpr int kBoardSize = 15;
inline constexpr int kNumSquares = 165;  // Includes the Thudstone's square.
inline constexpr int kThudstoneRow = 7;
inline constexpr int kThudstoneCol = 7;

// The eight directions, in the order used by the action encoding.
enum Direction : int {
  kNorth = 0,
  kNorthEast,
  kEast,
  kSouthEast,
  kSouth,
  kSouthWest,
  kWest,
  kNorthWest,
};
inline constexpr int kNumDirections = 8;
inline constexpr std::array<int, kNumDirections> kRowStep = {-1, -1, 0, 1,
                                                             1,  1,  0, -1};
inline constexpr std::array<int, kNumDirections> kColStep = {0, 1,  1,  1,
                                                             0, -1, -1, -1};

// Action encoding (THUD_RULES.md section 8).
inline constexpr int kMaxDistance = 14;
inline constexpr int kNumLineActions =
    kNumSquares * kNumDirections * kMaxDistance;  // 18,480
inline constexpr int kNumCaptureStepActions =
    kNumSquares * kNumDirections;  // 1,320
inline constexpr int kNumDistinctActions =
    kNumLineActions + kNumCaptureStepActions;  // 19,800

// Material and scoring.
inline constexpr int kNumDwarfs = 32;
inline constexpr int kNumTrolls = 8;
inline constexpr int kDwarfValue = 1;
inline constexpr int kTrollValue = 4;
inline constexpr int kMaxMargin = 32;

// Default end-of-battle limits (THUD_RULES.md section 6).
inline constexpr int kDefaultMaxTurnsWithoutCapture = 200;
inline constexpr int kDefaultMaxTurns = 800;

// Observation tensor: kNumObservationPlanes planes of kBoardSize x kBoardSize,
// the same for both players. Planes 0-2 describe each square:
//
//   dwarf  troll  empty
//     1      0      0    a dwarf
//     0      1      0    a troll
//     0      0      1    an empty square, which a piece could move to
//     0      0      0    a hole, which no piece can ever enter: one of the
//                        cut-off corners, or the Thudstone's square
//
// So the empty plane is what tells a square a piece could move to from a hole;
// without it, the two would look the same. (By the rules the Thudstone is just
// a hole, so it needs no plane of its own.) Planes 3-5 hold one value repeated
// over the whole plane.
inline constexpr int kDwarfPlane = 0;
inline constexpr int kTrollPlane = 1;
inline constexpr int kEmptyPlane = 2;
// 1 if the trolls are to move; after the battle has ended, if they would have
// moved next.
inline constexpr int kTrollsToMovePlane = 3;
inline constexpr int kNoCaptureCountPlane =
    4;  // turns_without_capture / max_turns_without_capture.
inline constexpr int kTurnCountPlane = 5;  // turns / max_turns.
inline constexpr int kNumObservationPlanes = 6;

// What occupies a square of the board. kOffBoard marks the cells of the
// implementation's padded grid that are not squares (its border and the
// cut-off corners), so that walking a line stops there; CellAt() never returns
// it, and a Position never contains it.
enum class Cell : std::int8_t { kEmpty, kDwarf, kTroll, kThudstone, kOffBoard };

// A square as (row, column) on the 15x15 grid; row 0 is north.
struct Coord {
  int row;
  int col;
};

// True for the 165 squares of the octagonal board.
bool IsOnBoard(int row, int col);

// The squares of the board are numbered 0..164 in row-major order.
int SquareIndex(int row, int col);  // Requires IsOnBoard(row, col).
Coord SquareCoord(int square);      // Requires 0 <= square < kNumSquares.

// An action split into its parts (THUD_RULES.md section 8).
struct DecodedAction {
  int square;          // The moving piece's square, 0..164.
  int direction;       // A Direction.
  int distance;        // 1..kMaxDistance; always 1 for a capture step.
  bool capture_step;   // True for a troll step that captures all.
};

Action EncodeLineAction(int square, int direction, int distance);
Action EncodeCaptureStepAction(int square, int direction);
DecodedAction DecodeAction(Action action);

// A position: what the text format at the top of this file describes.
struct Position {
  std::array<Cell, kNumSquares> board;  // Indexed by SquareIndex().
  Player to_move = kDwarfPlayer;
  int turns_played = 0;
  int turns_without_capture = 0;
};

// Reads a position written in the text format at the top of this file.
// Returns nullopt, after printing the reason to std::cerr, if the text is
// malformed or the position impossible.
absl::optional<Position> PositionFromText(const std::string& text);

// State of a battle.
class ThudState : public State {
 public:
  // The starting position.
  explicit ThudState(std::shared_ptr<const Game> game);
  // Any position, such as one read by PositionFromText().
  ThudState(std::shared_ptr<const Game> game, const Position& position);

  ThudState(const ThudState&) = default;
  ThudState& operator=(const ThudState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  // The actions played so far (HistoryString()): both players see every move.
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;
  // See "Saved games" at the top of this file.
  std::string Serialize() const override;

  // What occupies (row, col). Requires IsOnBoard(row, col).
  Cell CellAt(int row, int col) const;
  // Turns played so far, and consecutive turns since the last capture.
  int TurnsPlayed() const { return turns_played_; }
  int TurnsWithoutCapture() const { return turns_without_capture_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  // The 15x15 grid inside a one-cell border: kGridSize x kGridSize cells, row
  // by row. The border and the cut-off corners hold Cell::kOffBoard, so a line
  // walked by a fixed step per direction stops at the first cell that is not
  // empty, whatever is in the way.
  static constexpr int kGridSize = kBoardSize + 2;
  std::array<Cell, kGridSize * kGridSize> grid_;
  Player to_move_ = kDwarfPlayer;
  int turns_played_ = 0;
  int turns_without_capture_ = 0;
  int max_turns_without_capture_;
  int max_turns_;
  // Where the game started, if not from the opening; Serialize() saves it.
  absl::optional<Position> start_;
};

// Game object.
class ThudGame : public Game {
 public:
  explicit ThudGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  using Game::NewInitialState;
  std::unique_ptr<State> NewInitialState() const override;
  // A position read from text by PositionFromText(); stops with a fatal error
  // if the text is malformed or the position impossible, as chess does for an
  // invalid FEN.
  std::unique_ptr<State> NewInitialState(
      const std::string& diagram) const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {kNumObservationPlanes, kBoardSize, kBoardSize};
  }
  // Every turn is exactly one action, so the turn limit bounds the game.
  int MaxGameLength() const override { return max_turns_; }
  // See "Saved games" at the top of this file.
  std::unique_ptr<State> DeserializeState(
      const std::string& str) const override;

  int max_turns_without_capture() const { return max_turns_without_capture_; }
  int max_turns() const { return max_turns_; }

 private:
  const int max_turns_without_capture_;
  const int max_turns_;
};

}  // namespace thud
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_THUD_THUD_H_
