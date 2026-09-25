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
};

const Geometry& Geo() {
  static const Geometry* geometry = [] {
    auto* g = new Geometry();
    int square = 0;
    for (int row = 0; row < kBoardSize; ++row) {
      g->row_start[row] = square;
      for (int col = FirstCol(row); col < kBoardSize - FirstCol(row); ++col) {
        g->cell[square] = GridCell(row, col);
        g->coord[square] = {row, col};
        ++square;
      }
    }
    SPIEL_CHECK_EQ(square, kNumSquares);
    return g;
  }();
  return *geometry;
}

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

DecodedAction DecodeAction(Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  if (action >= kNumLineActions) {
    const int index = action - kNumLineActions;
    return {index / kNumDirections, index % kNumDirections, 1, true};
  }
  return {static_cast<int>(action / (kNumDirections * kMaxDistance)),
          static_cast<int>(action / kMaxDistance % kNumDirections),
          static_cast<int>(action % kMaxDistance) + 1, false};
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
  for (int square = 0; square < kNumSquares; ++square) {
    const Cell cell = position.board[square];
    SPIEL_CHECK_TRUE(cell != Cell::kOffBoard);
    SPIEL_CHECK_EQ(cell == Cell::kThudstone, square == thudstone);
    grid_[Geo().cell[square]] = cell;
    num_dwarfs_ += cell == Cell::kDwarf;
    num_trolls_ += cell == Cell::kTroll;
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
  // step's ID is above every line action's, so the result is sorted.
  std::vector<Action> actions;
  std::vector<Action> capture_steps;
  actions.reserve(256);
  for (int square = 0; square < kNumSquares; ++square) {
    const int cell = Geo().cell[square];
    if (grid_[cell] != own) continue;
    if (own == Cell::kDwarf) {
      AddDwarfMoves(square, cell, &actions);
    } else {
      AddTrollMoves(square, cell, &actions, &capture_steps);
    }
  }
  actions.insert(actions.end(), capture_steps.begin(), capture_steps.end());
  return actions;
}

void ThudState::DoApplyAction(Action action) {
  const DecodedAction move = DecodeAction(action);
  const int from = Geo().cell[move.square];
  const int to = from + move.distance * kGridStep[move.direction];
  const Cell mover = grid_[from];
  SPIEL_CHECK_TRUE(mover ==
                   (to_move_ == kDwarfPlayer ? Cell::kDwarf : Cell::kTroll));
  bool captured = false;
  if (mover == Cell::kDwarf && grid_[to] == Cell::kTroll) {
    --num_trolls_;  // A hurl captures the troll it lands on.
    captured = true;
  }
  grid_[to] = mover;
  grid_[from] = Cell::kEmpty;
  if (mover == Cell::kTroll && (move.capture_step || move.distance >= 2)) {
    // A capturing step or a shove captures every dwarf next to the troll.
    for (int step : kGridStep) {
      if (grid_[to + step] == Cell::kDwarf) {
        grid_[to + step] = Cell::kEmpty;
        --num_dwarfs_;
        captured = true;
      }
    }
  }
  ++turns_played_;
  turns_without_capture_ = captured ? 0 : turns_without_capture_ + 1;
  to_move_ = 1 - to_move_;
  terminal_ = BattleOver();
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
  for (int square = 0; square < kNumSquares; ++square) {
    const int cell = Geo().cell[square];
    if (grid_[cell] != own) continue;
    for (int step : kGridStep) {
      const Cell next = grid_[cell + step];
      if (next == Cell::kEmpty) return true;
      if (own == Cell::kDwarf && next == Cell::kTroll) return true;
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
