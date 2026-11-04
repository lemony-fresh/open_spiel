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

#include "open_spiel/games/thud/thud.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace thud {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"thud",
    /*long_name=*/"Thud",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"max_turns_without_capture",
      GameParameter(kDefaultMaxTurnsWithoutCapture)},
     {"max_turns", GameParameter(kDefaultMaxTurns)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ThudGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// The opening position (THUD_RULES.md section 2).
constexpr char kOpening[] =
    "-----dd.dd-----\n"
    "----d.....d----\n"
    "---d.......d---\n"
    "--d.........d--\n"
    "-d...........d-\n"
    "d.............d\n"
    "d.....TTT.....d\n"
    "......TOT......\n"
    "d.....TTT.....d\n"
    "d.............d\n"
    "-d...........d-\n"
    "--d.........d--\n"
    "---d.......d---\n"
    "----d.....d----\n"
    "-----dd.dd-----\n"
    "to_move=dwarfs turns=0 turns_without_capture=0";

// The padded grid: the 15x15 board inside a one-cell border, row by row.
constexpr int kGridSize = kBoardSize + 2;
constexpr int kGridCells = kGridSize * kGridSize;

int GridCell(int row, int col) { return (row + 1) * kGridSize + (col + 1); }

// How far one step in each direction moves on the grid.
constexpr std::array<int, kNumDirections> GridSteps() {
  std::array<int, kNumDirections> steps{};
  for (int d = 0; d < kNumDirections; ++d) {
    steps[d] = kRowStep[d] * kGridSize + kColStep[d];
  }
  return steps;
}
constexpr std::array<int, kNumDirections> kGridStep = GridSteps();

// The first column of each row; the last is 14 minus it (section 1).
int FirstCol(int row) { return std::max(0, std::abs(row - 7) - 2); }

// Square numbers and grid cells, computed once from (row, col).
struct Geometry {
  std::array<int, kBoardSize> row_start;  // Square number of each row's first.
  std::array<int, kNumSquares> cell;      // Grid cell of each square.
  std::array<Coord, kNumSquares> coord;   // (row, col) of each square.
  std::array<int, kGridCells> square;     // Square of each grid cell, -1 off the board.
};

// Zobrist keys (2026-10-23, for the teacher's transposition table): a random 64-bit number
// for each grid cell and piece, and one for the trolls to move. A position's hash is the
// XOR of the keys of its pieces, and of the trolls' key when they are to move. The keys
// are fixed (splitmix64 from a constant seed), so a hash means the same everywhere.
struct ZobristKeys {
  std::array<uint64_t, kGridCells> dwarf;
  std::array<uint64_t, kGridCells> troll;
  uint64_t trolls_to_move;
};

const ZobristKeys& Zobrist() {
  static const ZobristKeys* keys = [] {
    uint64_t seed = 0x7468756420323032ULL;  // "thud 202".
    auto next = [&seed] {  // splitmix64 (Steele, Lea and Flood, 2014).
      uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      return z ^ (z >> 31);
    };
    auto* k = new ZobristKeys();
    for (uint64_t& key : k->dwarf) key = next();
    for (uint64_t& key : k->troll) key = next();
    k->trolls_to_move = next();
    return k;
  }();
  return *keys;
}

// The key of `piece` on grid cell `cell`: 0 for anything but a dwarf or a troll.
uint64_t PieceKey(Cell piece, int cell) {
  if (piece == Cell::kDwarf) return Zobrist().dwarf[cell];
  if (piece == Cell::kTroll) return Zobrist().troll[cell];
  return 0;
}

const Geometry& Geo() {
  static const Geometry* geometry = [] {
    auto* g = new Geometry();
    g->square.fill(-1);
    int square = 0;
    for (int row = 0; row < kBoardSize; ++row) {
      g->row_start[row] = square;
      for (int col = FirstCol(row); col < kBoardSize - FirstCol(row); ++col) {
        g->cell[square] = GridCell(row, col);
        g->coord[square] = {row, col};
        g->square[g->cell[square]] = square;
        ++square;
      }
    }
    SPIEL_CHECK_EQ(square, kNumSquares);
    return g;
  }();
  return *geometry;
}

// Sets of squares, a bit each in three words (ThudState's dwarf_squares_ and troll_squares_).
using SquareSet = std::array<uint64_t, 3>;
void AddSquare(SquareSet* set, int square) {
  (*set)[square >> 6] |= uint64_t{1} << (square & 63);
}
void RemoveSquare(SquareSet* set, int square) {
  (*set)[square >> 6] &= ~(uint64_t{1} << (square & 63));
}
// The set's squares in increasing order: for (int w = 0; w < 3; ++w) for (uint64_t b = set[w];
// b != 0; b &= b - 1) the square w * 64 + LowestBit(b).
int LowestBit(uint64_t bits) { return __builtin_ctzll(bits); }

char CellChar(Cell cell) {
  switch (cell) {
    case Cell::kEmpty:
      return '.';
    case Cell::kDwarf:
      return 'd';
    case Cell::kTroll:
      return 'T';
    case Cell::kThudstone:
      return 'O';
    default:
      return '-';
  }
}

// A position in the text format of thud.h.
std::string PositionText(const Position& position) {
  std::string text;
  for (int row = 0; row < kBoardSize; ++row) {
    for (int col = 0; col < kBoardSize; ++col) {
      text += IsOnBoard(row, col)
                  ? CellChar(position.board[SquareIndex(row, col)])
                  : '-';
    }
    text += '\n';
  }
  absl::StrAppend(
      &text, "to_move=",
      position.to_move == kDwarfPlayer ? "dwarfs" : "trolls",
      " turns=", position.turns_played,
      " turns_without_capture=", position.turns_without_capture);
  return text;
}

const Position& OpeningPosition() {
  static const Position* opening = [] {
    absl::optional<Position> position = PositionFromText(kOpening);
    SPIEL_CHECK_TRUE(position.has_value());
    return new Position(*position);
  }();
  return *opening;
}

}  // namespace

// ---------------------------------------------------------------------------
// Board geometry and the action encoding (sections 1 and 8).

bool IsOnBoard(int row, int col) {
  return row >= 0 && row < kBoardSize && col >= FirstCol(row) &&
         col < kBoardSize - FirstCol(row);
}

int SquareIndex(int row, int col) {
  SPIEL_CHECK_TRUE(IsOnBoard(row, col));
  return Geo().row_start[row] + col - FirstCol(row);
}

Coord SquareCoord(int square) {
  SPIEL_CHECK_GE(square, 0);
  SPIEL_CHECK_LT(square, kNumSquares);
  return Geo().coord[square];
}

Action EncodeLineAction(int square, int direction, int distance) {
  SPIEL_CHECK_GE(distance, 1);
  SPIEL_CHECK_LE(distance, kMaxDistance);
  return (square * kNumDirections + direction) * kMaxDistance + (distance - 1);
}

Action EncodeCaptureStepAction(int square, int direction) {
  return kNumLineActions + square * kNumDirections + direction;
}

namespace {

// Every action's parts, worked out once (2026-10-27, a speed-up: the profile of the teacher's
// search put ~8% of its time in DecodeAction, called through the library's symbol table from
// the moves made and undone and the captures counted; Decode, inline here, reads this table).
struct PackedAction {
  uint8_t square, direction, distance;
  bool capture_step;
};

constexpr std::array<PackedAction, kNumDistinctActions> MakeActionTable() {
  std::array<PackedAction, kNumDistinctActions> table{};
  for (int action = 0; action < kNumDistinctActions; ++action) {
    PackedAction& p = table[action];
    if (action >= kNumLineActions) {
      const int index = action - kNumLineActions;
      p = {static_cast<uint8_t>(index / kNumDirections),
           static_cast<uint8_t>(index % kNumDirections), 1, true};
    } else {
      p = {static_cast<uint8_t>(action / (kNumDirections * kMaxDistance)),
           static_cast<uint8_t>(action / kMaxDistance % kNumDirections),
           static_cast<uint8_t>(action % kMaxDistance + 1), false};
    }
  }
  return table;
}
static_assert(kNumSquares <= 255 && kMaxDistance <= 255, "an action's parts fit in a byte");
constexpr std::array<PackedAction, kNumDistinctActions> kActionTable = MakeActionTable();

// DecodeAction for the state's own use: the action is legal there, so in range.
inline DecodedAction Decode(Action action) {
  SPIEL_DCHECK_GE(action, 0);
  SPIEL_DCHECK_LT(action, kNumDistinctActions);
  const PackedAction& p = kActionTable[action];
  return {p.square, p.direction, p.distance, p.capture_step};
}

}  // namespace

DecodedAction DecodeAction(Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  return Decode(action);
}

int PolicyPlaneIndex(Action action) {
  const DecodedAction parts = DecodeAction(action);
  const Coord coord = SquareCoord(parts.square);
  const int plane = parts.capture_step
                        ? kNumDirections * kMaxDistance + parts.direction
                        : parts.direction * kMaxDistance + parts.distance - 1;
  return (plane * kBoardSize + coord.row) * kBoardSize + coord.col;
}

// ---------------------------------------------------------------------------
// The board's symmetries.

namespace {

// Each symmetry as a matrix {a, b, c, d} taking an offset (row, col) from the
// Thudstone to (a * row + b * col, c * row + d * col): the identity, the three
// quarter turns, and the four mirror images.
constexpr std::array<std::array<int, 4>, kNumSymmetries> kSymmetryMatrices = {
    {{1, 0, 0, 1},
     {0, 1, -1, 0},
     {-1, 0, 0, -1},
     {0, -1, 1, 0},
     {1, 0, 0, -1},
     {-1, 0, 0, 1},
     {0, 1, 1, 0},
     {0, -1, -1, 0}}};

}  // namespace

Coord SymmetricCoord(Coord coord, int symmetry) {
  SPIEL_CHECK_GE(symmetry, 0);
  SPIEL_CHECK_LT(symmetry, kNumSymmetries);
  const std::array<int, 4>& m = kSymmetryMatrices[symmetry];
  const int row = coord.row - kThudstoneRow, col = coord.col - kThudstoneCol;
  return {kThudstoneRow + m[0] * row + m[1] * col,
          kThudstoneCol + m[2] * row + m[3] * col};
}

int SymmetricDirection(int direction, int symmetry) {
  SPIEL_CHECK_GE(direction, 0);
  SPIEL_CHECK_LT(direction, kNumDirections);
  SPIEL_CHECK_GE(symmetry, 0);
  SPIEL_CHECK_LT(symmetry, kNumSymmetries);
  const std::array<int, 4>& m = kSymmetryMatrices[symmetry];
  const int row = m[0] * kRowStep[direction] + m[1] * kColStep[direction];
  const int col = m[2] * kRowStep[direction] + m[3] * kColStep[direction];
  for (int d = 0; d < kNumDirections; ++d) {
    if (kRowStep[d] == row && kColStep[d] == col) return d;
  }
  SpielFatalError("A symmetry took a direction to no direction.");
}

Action SymmetricAction(Action action, int symmetry) {
  const DecodedAction parts = DecodeAction(action);
  const Coord coord = SymmetricCoord(SquareCoord(parts.square), symmetry);
  const int square = SquareIndex(coord.row, coord.col);
  const int direction = SymmetricDirection(parts.direction, symmetry);
  return parts.capture_step
             ? EncodeCaptureStepAction(square, direction)
             : EncodeLineAction(square, direction, parts.distance);
}

std::vector<float> SymmetricObservation(absl::Span<const float> observation,
                                        int symmetry) {
  SPIEL_CHECK_EQ(observation.size(),
                 kNumObservationPlanes * kBoardSize * kBoardSize);
  std::vector<float> transformed(observation.size());
  for (int plane = 0; plane < kNumObservationPlanes; ++plane) {
    for (int row = 0; row < kBoardSize; ++row) {
      for (int col = 0; col < kBoardSize; ++col) {
        const Coord to = SymmetricCoord({row, col}, symmetry);
        transformed[(plane * kBoardSize + to.row) * kBoardSize + to.col] =
            observation[(plane * kBoardSize + row) * kBoardSize + col];
      }
    }
  }
  return transformed;
}

// ---------------------------------------------------------------------------
// Positions as text.

absl::optional<Position> PositionFromText(const std::string& text) {
  auto invalid = [&text](absl::string_view why) -> absl::optional<Position> {
    std::cerr << "Invalid Thud position: " << why << "\n" << text << std::endl;
    return absl::nullopt;
  };
  std::vector<std::string> lines;
  for (absl::string_view line : absl::StrSplit(text, '\n')) {
    line = absl::StripAsciiWhitespace(line);
    if (!line.empty()) lines.emplace_back(line);
  }
  if (lines.size() < kBoardSize) return invalid("fewer than 15 rows");
  if (lines.size() > kBoardSize + 1) {
    return invalid("more than 15 rows and a status line");
  }

  Position position;
  int dwarfs = 0;
  int trolls = 0;
  for (int row = 0; row < kBoardSize; ++row) {
    const std::string& line = lines[row];
    if (line.size() != kBoardSize) {
      return invalid(absl::StrCat("row ", row, " is not 15 characters"));
    }
    for (int col = 0; col < kBoardSize; ++col) {
      const char ch = line[col];
      if (!IsOnBoard(row, col)) {
        if (ch != '-') {
          return invalid(absl::StrCat("(", row, ",", col,
                                      ") is a cut-off corner, written '-'"));
        }
        continue;
      }
      Cell cell;
      switch (ch) {
        case '.':
          cell = Cell::kEmpty;
          break;
        case 'd':
          cell = Cell::kDwarf;
          ++dwarfs;
          break;
        case 'T':
          cell = Cell::kTroll;
          ++trolls;
          break;
        case 'O':
          cell = Cell::kThudstone;
          break;
        default:
          return invalid(absl::StrCat("'", std::string(1, ch), "' at (", row,
                                      ",", col, ") is none of . d T O"));
      }
      const bool thudstone_square =
          row == kThudstoneRow && col == kThudstoneCol;
      if ((cell == Cell::kThudstone) != thudstone_square) {
        return invalid("the Thudstone must be at (7,7), and only there");
      }
      position.board[SquareIndex(row, col)] = cell;
    }
  }
  if (dwarfs > kNumDwarfs) return invalid("more than 32 dwarfs");
  if (trolls > kNumTrolls) return invalid("more than 8 trolls");

  if (lines.size() == kBoardSize + 1) {
    std::vector<std::string> seen;
    for (absl::string_view field :
         absl::StrSplit(lines[kBoardSize], ' ', absl::SkipEmpty())) {
      const std::vector<std::string> key_value =
          absl::StrSplit(field, absl::MaxSplits('=', 1));
      if (key_value.size() != 2) {
        return invalid(absl::StrCat("status field '", field, "' has no '='"));
      }
      const std::string& key = key_value[0];
      const std::string& value = key_value[1];
      if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
        return invalid(absl::StrCat("status field '", key, "' given twice"));
      }
      seen.push_back(key);
      if (key == "to_move") {
        if (value == "dwarfs") {
          position.to_move = kDwarfPlayer;
        } else if (value == "trolls") {
          position.to_move = kTrollPlayer;
        } else {
          return invalid(absl::StrCat("to_move=", value));
        }
      } else if (key == "turns" || key == "turns_without_capture") {
        int count;
        if (!absl::SimpleAtoi(value, &count) || count < 0) {
          return invalid(absl::StrCat(key, "=", value));
        }
        if (key == "turns") {
          position.turns_played = count;
        } else {
          position.turns_without_capture = count;
        }
      } else {
        return invalid(absl::StrCat("unknown status field '", key, "'"));
      }
    }
  }
  return position;
}

// ---------------------------------------------------------------------------
// The state.

ThudState::ThudState(std::shared_ptr<const Game> game) : State(game) {
  const auto& thud_game = static_cast<const ThudGame&>(*game);
  max_turns_without_capture_ = thud_game.max_turns_without_capture();
  max_turns_ = thud_game.max_turns();
  SetPosition(OpeningPosition());
}

ThudState::ThudState(std::shared_ptr<const Game> game,
                     const Position& position)
    : State(game), start_(position) {
  const auto& thud_game = static_cast<const ThudGame&>(*game);
  max_turns_without_capture_ = thud_game.max_turns_without_capture();
  max_turns_ = thud_game.max_turns();
  SetPosition(position);
}

void ThudState::SetPosition(const Position& position) {
  // PositionFromText() rejects impossible positions; a Position built another
  // way must be possible too, or the returns could leave [-1, 1].
  const int thudstone = SquareIndex(kThudstoneRow, kThudstoneCol);
  grid_.fill(Cell::kOffBoard);
  num_dwarfs_ = 0;
  num_trolls_ = 0;
  dwarf_squares_ = {};
  troll_squares_ = {};
  for (int square = 0; square < kNumSquares; ++square) {
    const Cell cell = position.board[square];
    SPIEL_CHECK_TRUE(cell != Cell::kOffBoard);
    SPIEL_CHECK_EQ(cell == Cell::kThudstone, square == thudstone);
    grid_[Geo().cell[square]] = cell;
    num_dwarfs_ += cell == Cell::kDwarf;
    num_trolls_ += cell == Cell::kTroll;
    if (cell == Cell::kDwarf) AddSquare(&dwarf_squares_, square);
    if (cell == Cell::kTroll) AddSquare(&troll_squares_, square);
  }
  SPIEL_CHECK_LE(num_dwarfs_, kNumDwarfs);
  SPIEL_CHECK_LE(num_trolls_, kNumTrolls);
  SPIEL_CHECK_TRUE(position.to_move == kDwarfPlayer ||
                   position.to_move == kTrollPlayer);
  SPIEL_CHECK_GE(position.turns_played, 0);
  SPIEL_CHECK_GE(position.turns_without_capture, 0);
  to_move_ = position.to_move;
  turns_played_ = position.turns_played;
  turns_without_capture_ = position.turns_without_capture;
  terminal_ = BattleOver();
  hash_ = BoardHashFromScratch();
}

bool ThudState::PieceSetsMatchBoard() const {
  SquareSet dwarfs{}, trolls{};
  for (int square = 0; square < kNumSquares; ++square) {
    const Cell cell = grid_[Geo().cell[square]];
    if (cell == Cell::kDwarf) AddSquare(&dwarfs, square);
    if (cell == Cell::kTroll) AddSquare(&trolls, square);
  }
  return dwarfs == dwarf_squares_ && trolls == troll_squares_;
}

uint64_t ThudState::BoardHashFromScratch() const {
  uint64_t hash = to_move_ == kTrollPlayer ? Zobrist().trolls_to_move : 0;
  for (int square = 0; square < kNumSquares; ++square) {
    const int cell = Geo().cell[square];
    hash ^= PieceKey(grid_[cell], cell);
  }
  return hash;
}

Position ThudState::GetPosition() const {
  Position position;
  for (int square = 0; square < kNumSquares; ++square) {
    position.board[square] = grid_[Geo().cell[square]];
  }
  position.to_move = to_move_;
  position.turns_played = turns_played_;
  position.turns_without_capture = turns_without_capture_;
  return position;
}

Cell ThudState::CellAt(int row, int col) const {
  SPIEL_CHECK_TRUE(IsOnBoard(row, col));
  return grid_[GridCell(row, col)];
}

Player ThudState::CurrentPlayer() const {
  return terminal_ ? kTerminalPlayerId : to_move_;
}

// ---------------------------------------------------------------------------
// Moves (sections 4 and 5).

int ThudState::LineLength(int cell, int step) const {
  const Cell piece = grid_[cell];
  int length = 0;
  for (int c = cell; grid_[c] == piece; c += step) ++length;
  return length;
}

int ThudState::DwarfsNextTo(int cell) const {
  int dwarfs = 0;
  for (int step : kGridStep) dwarfs += grid_[cell + step] == Cell::kDwarf;
  return dwarfs;
}

// Section 4: a dwarf moves over empty squares (4a), or is hurled onto a troll
// at most as many squares away as its line behind it is long (4b).
void ThudState::AddDwarfMoves(int square, int cell,
                              std::vector<Action>* actions) const {
  for (int d = 0; d < kNumDirections; ++d) {
    const int step = kGridStep[d];
    int distance = 1;
    int target = cell + step;
    for (; grid_[target] == Cell::kEmpty; ++distance, target += step) {
      actions->push_back(EncodeLineAction(square, d, distance));
    }
    // The path ends at the first square that is not empty: a hurl lands there
    // if it holds a troll within reach of the line behind the dwarf.
    if (grid_[target] == Cell::kTroll && distance <= LineLength(cell, -step)) {
      actions->push_back(EncodeLineAction(square, d, distance));
    }
  }
}

// Section 5: a troll steps to an empty square next to it, capturing none of
// the dwarfs next to its landing square or all of them (5a), or is shoved 2 up
// to as many squares as its line behind it is long, over and onto empty
// squares, to a square next to a dwarf, capturing them all (5b).
void ThudState::AddTrollMoves(int square, int cell,
                              std::vector<Action>* actions,
                              std::vector<Action>* capture_steps) const {
  for (int d = 0; d < kNumDirections; ++d) {
    const int step = kGridStep[d];
    int target = cell + step;
    if (grid_[target] != Cell::kEmpty) continue;
    actions->push_back(EncodeLineAction(square, d, 1));
    if (DwarfsNextTo(target) > 0) {
      capture_steps->push_back(EncodeCaptureStepAction(square, d));
    }
    const int line = LineLength(cell, -step);
    for (int distance = 2; distance <= line; ++distance) {
      target += step;
      if (grid_[target] != Cell::kEmpty) break;
      if (DwarfsNextTo(target) > 0) {
        actions->push_back(EncodeLineAction(square, d, distance));
      }
    }
  }
}

std::vector<Action> ThudState::LegalActions() const {
  if (terminal_) return {};
  const Cell own = to_move_ == kDwarfPlayer ? Cell::kDwarf : Cell::kTroll;
  // Squares and directions are visited in increasing order, and every capture
  // step's ID is above every line action's, so the result is sorted. The own pieces' set
  // gives their squares in increasing order (2026-10-27; before, every square was looked at).
  std::vector<Action> actions;
  std::vector<Action> capture_steps;
  actions.reserve(256);
  const Geometry& geo = Geo();
  const SquareSet& pieces = own == Cell::kDwarf ? dwarf_squares_ : troll_squares_;
  for (int w = 0; w < 3; ++w) {
    for (uint64_t bits = pieces[w]; bits != 0; bits &= bits - 1) {
      const int square = w * 64 + LowestBit(bits);
      const int cell = geo.cell[square];
      if (own == Cell::kDwarf) {
        AddDwarfMoves(square, cell, &actions);
      } else {
        AddTrollMoves(square, cell, &actions, &capture_steps);
      }
    }
  }
  actions.insert(actions.end(), capture_steps.begin(), capture_steps.end());
  return actions;
}

void ThudState::DoApplyAction(Action action) {
  const DecodedAction move = Decode(action);
  const int from = Geo().cell[move.square];
  const int to = from + move.distance * kGridStep[move.direction];
  const Cell mover = grid_[from];
  SPIEL_CHECK_TRUE(mover ==
                   (to_move_ == kDwarfPlayer ? Cell::kDwarf : Cell::kTroll));
  Undo undo{0, false, terminal_, turns_without_capture_, hash_};
  const Geometry& geo = Geo();
  SquareSet& movers = mover == Cell::kDwarf ? dwarf_squares_ : troll_squares_;
  bool captured = false;
  if (mover == Cell::kDwarf && grid_[to] == Cell::kTroll) {
    --num_trolls_;  // A hurl captures the troll it lands on.
    captured = true;
    undo.troll_taken = true;
    hash_ ^= PieceKey(Cell::kTroll, to);
    RemoveSquare(&troll_squares_, geo.square[to]);
  }
  grid_[to] = mover;
  grid_[from] = Cell::kEmpty;
  hash_ ^= PieceKey(mover, from) ^ PieceKey(mover, to);
  RemoveSquare(&movers, move.square);
  AddSquare(&movers, geo.square[to]);
  if (mover == Cell::kTroll && (move.capture_step || move.distance >= 2)) {
    // A capturing step or a shove captures every dwarf next to the troll.
    for (int d = 0; d < kNumDirections; ++d) {
      const int cell = to + kGridStep[d];
      if (grid_[cell] == Cell::kDwarf) {
        grid_[cell] = Cell::kEmpty;
        --num_dwarfs_;
        captured = true;
        undo.dwarfs_taken |= 1 << d;
        hash_ ^= PieceKey(Cell::kDwarf, cell);
        RemoveSquare(&dwarf_squares_, geo.square[cell]);
      }
    }
  }
  undo_.push_back(undo);
  ++turns_played_;
  turns_without_capture_ = captured ? 0 : turns_without_capture_ + 1;
  to_move_ = 1 - to_move_;
  hash_ ^= Zobrist().trolls_to_move;
  terminal_ = BattleOver();
}

int ThudState::Captures(Action action) const {
  const DecodedAction move = Decode(action);
  const int from = Geo().cell[move.square];
  const int to = from + move.distance * kGridStep[move.direction];
  if (grid_[from] == Cell::kDwarf) return grid_[to] == Cell::kTroll ? 1 : 0;
  if (!move.capture_step && move.distance < 2) return 0;
  // As DoApplyAction counts them after the move: `from`, next to `to` for a capture step,
  // holds the troll itself, never a dwarf.
  int dwarfs = 0;
  for (int step : kGridStep) dwarfs += grid_[to + step] == Cell::kDwarf;
  return dwarfs;
}

int ThudState::BestCapture() const { return BestCaptureFor(to_move_); }

int ThudState::BestCaptureFor(Player player) const {
  SPIEL_DCHECK_TRUE(player == kDwarfPlayer || player == kTrollPlayer);
  if (terminal_) return 0;
  const Geometry& geo = Geo();
  if (player == kDwarfPlayer) {
    // A hurl takes one troll, the most a dwarf move can, so the question is whether there is
    // one — looked for from the trolls' side (2026-10-27, a speed-up: at most 8 trolls against
    // up to 32 dwarfs): from each troll along each line, the empty path to the first piece; a
    // dwarf there at `distance` hurls onto the troll if its line behind it, away from the
    // troll, is at least that long (as AddDwarfMoves: a hurl goes at most the line's length).
    for (int w = 0; w < 3; ++w) {
      for (uint64_t bits = troll_squares_[w]; bits != 0; bits &= bits - 1) {
        const int troll = geo.cell[w * 64 + LowestBit(bits)];
        for (int step : kGridStep) {
          int cell = troll + step;
          int distance = 1;
          for (; grid_[cell] == Cell::kEmpty; ++distance) cell += step;
          if (grid_[cell] == Cell::kDwarf && distance <= LineLength(cell, step)) return 1;
        }
      }
    }
    return 0;
  }
  int best = 0;
  for (int w = 0; w < 3; ++w) {
    for (uint64_t bits = troll_squares_[w]; bits != 0; bits &= bits - 1) {
      const int cell = geo.cell[w * 64 + LowestBit(bits)];
      for (int d = 0; d < kNumDirections; ++d) {
        // As AddTrollMoves: the capture step, and the shoves up to the line's length. At most
        // 7 dwarfs: of the landing square's 8 neighbours one is on the troll's path.
        const int step = kGridStep[d];
        int target = cell + step;
        if (grid_[target] != Cell::kEmpty) continue;
        best = std::max(best, DwarfsNextTo(target));
        const int line = LineLength(cell, -step);
        for (int distance = 2; distance <= line; ++distance) {
          target += step;
          if (grid_[target] != Cell::kEmpty) break;
          best = std::max(best, DwarfsNextTo(target));
        }
        if (best == 7) return best;
      }
    }
  }
  return best;
}

std::vector<Action> ThudState::CaptureActions() const {
  std::vector<Action> actions;
  if (terminal_) return actions;
  std::vector<Action> capture_steps;
  const Geometry& geo = Geo();
  const bool dwarfs = to_move_ == kDwarfPlayer;
  const SquareSet& pieces = dwarfs ? dwarf_squares_ : troll_squares_;
  // As AddDwarfMoves and AddTrollMoves, keeping only the moves that capture, in their order.
  for (int w = 0; w < 3; ++w) {
    for (uint64_t bits = pieces[w]; bits != 0; bits &= bits - 1) {
      const int square = w * 64 + LowestBit(bits);
      const int cell = geo.cell[square];
      for (int d = 0; d < kNumDirections; ++d) {
        const int step = kGridStep[d];
        int target = cell + step;
        if (dwarfs) {
          int distance = 1;
          for (; grid_[target] == Cell::kEmpty; ++distance) target += step;
          if (grid_[target] == Cell::kTroll && distance <= LineLength(cell, -step)) {
            actions.push_back(EncodeLineAction(square, d, distance));
          }
          continue;
        }
        if (grid_[target] != Cell::kEmpty) continue;
        if (DwarfsNextTo(target) > 0) {
          capture_steps.push_back(EncodeCaptureStepAction(square, d));
        }
        const int line = LineLength(cell, -step);
        for (int distance = 2; distance <= line; ++distance) {
          target += step;
          if (grid_[target] != Cell::kEmpty) break;
          if (DwarfsNextTo(target) > 0) {
            actions.push_back(EncodeLineAction(square, d, distance));
          }
        }
      }
    }
  }
  actions.insert(actions.end(), capture_steps.begin(), capture_steps.end());
  return actions;
}

bool ThudState::HasQuietMove() const {
  if (terminal_) return false;
  // A dwarf's one-square move or a troll's step that captures nothing goes to any empty
  // square next to the piece, and every quiet move passes over one first.
  const Geometry& geo = Geo();
  const SquareSet& pieces = to_move_ == kDwarfPlayer ? dwarf_squares_ : troll_squares_;
  for (int w = 0; w < 3; ++w) {
    for (uint64_t bits = pieces[w]; bits != 0; bits &= bits - 1) {
      const int cell = geo.cell[w * 64 + LowestBit(bits)];
      for (int step : kGridStep) {
        if (grid_[cell + step] == Cell::kEmpty) return true;
      }
    }
  }
  return false;
}

void ThudState::UndoAction(Player player, Action action) {
  SPIEL_CHECK_FALSE(history_.empty());
  SPIEL_CHECK_EQ(history_.back().player, player);
  SPIEL_CHECK_EQ(history_.back().action, action);
  SPIEL_CHECK_EQ(undo_.size(), history_.size());  // A record for every move made.
  const Undo undo = undo_.back();
  const DecodedAction move = Decode(action);
  const int from = Geo().cell[move.square];
  const int to = from + move.distance * kGridStep[move.direction];
  const Geometry& geo = Geo();
  const Cell mover = grid_[to];
  SquareSet& movers = mover == Cell::kDwarf ? dwarf_squares_ : troll_squares_;
  grid_[from] = mover;
  grid_[to] = undo.troll_taken ? Cell::kTroll : Cell::kEmpty;
  RemoveSquare(&movers, geo.square[to]);
  AddSquare(&movers, move.square);
  num_trolls_ += undo.troll_taken;
  if (undo.troll_taken) AddSquare(&troll_squares_, geo.square[to]);
  for (int d = 0; d < kNumDirections; ++d) {
    if (undo.dwarfs_taken & (1 << d)) {
      grid_[to + kGridStep[d]] = Cell::kDwarf;
      AddSquare(&dwarf_squares_, geo.square[to + kGridStep[d]]);
      ++num_dwarfs_;
    }
  }
  --turns_played_;
  turns_without_capture_ = undo.turns_without_capture;
  to_move_ = player;
  terminal_ = undo.terminal;
  hash_ = undo.hash;
  undo_.pop_back();
  history_.pop_back();
  --move_number_;
}

std::string ThudState::ActionToString(Player player, Action action_id) const {
  const DecodedAction move = DecodeAction(action_id);
  const Coord from = SquareCoord(move.square);
  const int row = from.row + kRowStep[move.direction] * move.distance;
  const int col = from.col + kColStep[move.direction] * move.distance;
  bool captures;
  if (grid_[Geo().cell[move.square]] == Cell::kDwarf) {
    captures = IsOnBoard(row, col) && CellAt(row, col) == Cell::kTroll;
  } else {
    captures = move.capture_step || move.distance >= 2;
  }
  return absl::StrCat("(", from.row, ",", from.col, ")-(", row, ",", col, ")",
                      captures ? "x" : "");
}

// ---------------------------------------------------------------------------
// End of the battle and scoring (sections 6 and 7).

bool ThudState::SideToMoveCanMove() const {
  const Cell own = to_move_ == kDwarfPlayer ? Cell::kDwarf : Cell::kTroll;
  const Geometry& geo = Geo();
  const SquareSet& pieces = own == Cell::kDwarf ? dwarf_squares_ : troll_squares_;
  for (int w = 0; w < 3; ++w) {
    for (uint64_t bits = pieces[w]; bits != 0; bits &= bits - 1) {
      const int cell = geo.cell[w * 64 + LowestBit(bits)];
      for (int step : kGridStep) {
        const Cell next = grid_[cell + step];
        if (next == Cell::kEmpty) return true;
        if (own == Cell::kDwarf && next == Cell::kTroll) return true;
      }
    }
  }
  return false;
}

bool ThudState::BattleOver() const {
  return turns_played_ >= max_turns_ ||
         turns_without_capture_ >= max_turns_without_capture_ ||
         !SideToMoveCanMove();
}

bool ThudState::IsTerminal() const { return terminal_; }

std::vector<double> ThudState::Returns() const {
  if (!terminal_) return {0, 0};
  const double margin = num_dwarfs_ * kDwarfValue - num_trolls_ * kTrollValue;
  return {margin / kMaxMargin, -margin / kMaxMargin};
}

// ---------------------------------------------------------------------------
// What the players see.

std::string ThudState::ToString() const { return PositionText(GetPosition()); }

std::string ThudState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ThudState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ThudState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(),
                 kNumObservationPlanes * kBoardSize * kBoardSize);
  auto at = [&values](int plane, int row, int col) -> float& {
    return values[(plane * kBoardSize + row) * kBoardSize + col];
  };
  const float trolls_to_move = to_move_ == kTrollPlayer ? 1 : 0;
  const float no_capture_count =
      static_cast<float>(turns_without_capture_) / max_turns_without_capture_;
  const float turn_count = static_cast<float>(turns_played_) / max_turns_;
  for (int row = 0; row < kBoardSize; ++row) {
    for (int col = 0; col < kBoardSize; ++col) {
      // Holes, the Thudstone's square and the cut-off corners, are in none of
      // the board planes.
      const Cell cell = grid_[GridCell(row, col)];
      at(kDwarfPlane, row, col) = cell == Cell::kDwarf;
      at(kTrollPlane, row, col) = cell == Cell::kTroll;
      at(kEmptyPlane, row, col) = cell == Cell::kEmpty;
      at(kTrollsToMovePlane, row, col) = trolls_to_move;
      at(kNoCaptureCountPlane, row, col) = no_capture_count;
      at(kTurnCountPlane, row, col) = turn_count;
    }
  }
}

std::unique_ptr<State> ThudState::Clone() const {
  return std::unique_ptr<State>(new ThudState(*this));
}

std::string ThudState::Serialize() const {
  if (!start_.has_value()) return State::Serialize();
  return absl::StrCat(PositionText(*start_), "\n",
                      absl::StrJoin(History(), "\n"), "\n");
}

// ---------------------------------------------------------------------------
// The game.

ThudGame::ThudGame(const GameParameters& params)
    : Game(kGameType, params),
      max_turns_without_capture_(
          ParameterValue<int>("max_turns_without_capture")),
      max_turns_(ParameterValue<int>("max_turns")) {
  SPIEL_CHECK_GT(max_turns_without_capture_, 0);
  SPIEL_CHECK_GT(max_turns_, 0);
}

std::unique_ptr<State> ThudGame::NewInitialState() const {
  return std::make_unique<ThudState>(shared_from_this());
}

std::unique_ptr<State> ThudGame::NewInitialState(
    const std::string& diagram) const {
  const absl::optional<Position> position = PositionFromText(diagram);
  if (!position.has_value()) {
    SpielFatalError(absl::StrCat("thud: invalid position:\n", diagram));
  }
  return std::make_unique<ThudState>(shared_from_this(), *position);
}

std::unique_ptr<State> ThudGame::DeserializeState(
    const std::string& str) const {
  // A game from the opening is saved as OpenSpiel saves by default; any other
  // starts with its position text, whose first row begins with '-'.
  if (str.empty() || str[0] != '-') return Game::DeserializeState(str);
  const std::vector<std::string> lines = absl::StrSplit(str, '\n');
  const size_t position_lines = kBoardSize + 1;  // The rows and status line.
  SPIEL_CHECK_GE(lines.size(), position_lines);
  std::unique_ptr<State> state = NewInitialState(absl::StrJoin(
      lines.begin(), lines.begin() + position_lines, "\n"));
  for (size_t i = position_lines; i < lines.size(); ++i) {
    if (lines[i].empty()) continue;
    Action action;
    SPIEL_CHECK_TRUE(absl::SimpleAtoi(lines[i], &action));
    state->ApplyAction(action);
  }
  return state;
}

}  // namespace thud
}  // namespace open_spiel
