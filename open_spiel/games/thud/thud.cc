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

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
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

// The tests in thud_test.cc were written first (thud/PLAN.md, Phase 3); the
// implementation replaces these stubs.
[[noreturn]] void NotImplemented(const char* what) {
  SpielFatalError(absl::StrCat("thud: ", what, " is not implemented yet"));
}

}  // namespace

bool IsOnBoard(int row, int col) { NotImplemented("IsOnBoard"); }
int SquareIndex(int row, int col) { NotImplemented("SquareIndex"); }
Coord SquareCoord(int square) { NotImplemented("SquareCoord"); }

Action EncodeLineAction(int square, int direction, int distance) {
  NotImplemented("EncodeLineAction");
}
Action EncodeCaptureStepAction(int square, int direction) {
  NotImplemented("EncodeCaptureStepAction");
}
DecodedAction DecodeAction(Action action) { NotImplemented("DecodeAction"); }

absl::optional<Position> PositionFromText(const std::string& text) {
  NotImplemented("PositionFromText");
}

ThudState::ThudState(std::shared_ptr<const Game> game) : State(game) {
  NotImplemented("the starting position");
}

ThudState::ThudState(std::shared_ptr<const Game> game,
                     const Position& position)
    : State(game) {
  NotImplemented("a state from a Position");
}

Player ThudState::CurrentPlayer() const { NotImplemented("CurrentPlayer"); }

std::string ThudState::ActionToString(Player player, Action action_id) const {
  NotImplemented("ActionToString");
}

std::string ThudState::ToString() const { NotImplemented("ToString"); }

bool ThudState::IsTerminal() const { NotImplemented("IsTerminal"); }

std::vector<double> ThudState::Returns() const { NotImplemented("Returns"); }

std::string ThudState::InformationStateString(Player player) const {
  NotImplemented("InformationStateString");
}

std::string ThudState::ObservationString(Player player) const {
  NotImplemented("ObservationString");
}

void ThudState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  NotImplemented("ObservationTensor");
}

std::unique_ptr<State> ThudState::Clone() const { NotImplemented("Clone"); }

std::vector<Action> ThudState::LegalActions() const {
  NotImplemented("LegalActions");
}

std::string ThudState::Serialize() const { NotImplemented("Serialize"); }

Cell ThudState::CellAt(int row, int col) const { NotImplemented("CellAt"); }

void ThudState::DoApplyAction(Action action) {
  NotImplemented("DoApplyAction");
}

ThudGame::ThudGame(const GameParameters& params)
    : Game(kGameType, params),
      max_turns_without_capture_(
          ParameterValue<int>("max_turns_without_capture")),
      max_turns_(ParameterValue<int>("max_turns")) {}

std::unique_ptr<State> ThudGame::NewInitialState() const {
  return std::make_unique<ThudState>(shared_from_this());
}

std::unique_ptr<State> ThudGame::NewInitialState(
    const std::string& diagram) const {
  NotImplemented("NewInitialState from text");
}

std::unique_ptr<State> ThudGame::DeserializeState(
    const std::string& str) const {
  NotImplemented("DeserializeState");
}

}  // namespace thud
}  // namespace open_spiel
