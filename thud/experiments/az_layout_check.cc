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

// The C++ counterpart of az_layout_check.py: does OpenSpiel's C++ AlphaZero model
// (LibTorch, planes first) learn Thud's board as well as the Python model does once its
// input is transposed to planes last, and better than the Python model as it is?
//
// It trains upstream's VPNetModel, unmodified, on the Python runs' supervised task, with
// their data and settings: az_layout_check.py --export writes the random games, the
// checksums of the positions taken from them, and each seed's batches. This program
// replays the games with Thud's C++ code, stops unless its positions reproduce the
// checksums, and then trains on the same batches in the same order: resnet 32 x 2,
// learning rate 1e-3, weight decay 1e-4, evaluated on the same per-side measures.
//
//   thud/experiments/build_az_program.sh az_layout_check   # next to libopen_spiel.so
//   python3 thud/experiments/az_layout_check.py --export DATA
//   OMP_NUM_THREADS=3 build-shared/az_layout_check DATA SEED
//
// Prints one JSON line per evaluation, as the Python script does.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"

namespace {

using open_spiel::Action;
using open_spiel::algorithms::torch_az::VPNetModel;

constexpr int kEvery = 7;  // As az_layout_check.py samples.
constexpr int kPlanes = 6;
constexpr int kTrollsToMovePlane = 3;
constexpr int kNnWidth = 32;
constexpr int kNnDepth = 2;
constexpr double kLearningRate = 1e-3;
constexpr double kWeightDecay = 1e-4;
constexpr int kEvalEvery = 250;
constexpr int kEvalBatch = 256;

struct Position {
  std::vector<float> observation;
  std::vector<Action> legal;
  std::vector<Action> captures;
  bool trolls_to_move;
};

// Every kEvery-th position of each game, replayed from its actions.
std::vector<Position> Replay(const open_spiel::Game& game, const std::string& path) {
  std::ifstream file(path);
  if (!file) open_spiel::SpielFatalError(absl::StrCat("Cannot read ", path));
  std::vector<Position> positions;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream actions(line);
    std::unique_ptr<open_spiel::State> state = game.NewInitialState();
    Action action;
    for (int turn = 0; actions >> action; ++turn) {
      SPIEL_CHECK_FALSE(state->IsTerminal());
      if (turn % kEvery == 0) {
        Position p{state->ObservationTensor(), state->LegalActions(), {}, false};
        for (Action a : p.legal) {
          if (absl::EndsWith(state->ActionToString(a), "x")) p.captures.push_back(a);
        }
        p.trolls_to_move =
            p.observation[kTrollsToMovePlane * p.observation.size() / kPlanes] > 0;
        positions.push_back(std::move(p));
      }
      state->ApplyAction(action);
    }
    SPIEL_CHECK_TRUE(state->IsTerminal());
  }
  return positions;
}

// Stops unless the positions reproduce checksums() of az_layout_check.py, as exported:
// the counts and sums of actions exactly, the plane sums (two planes hold fractions,
// summed in another order) to within 0.01.
void CheckAgainstExport(const std::vector<Position>& positions,
                        const open_spiel::json::Object& expected,
                        const std::string& name) {
  int64_t legal = 0, legal_sum = 0, captures = 0, captures_sum = 0;
  std::vector<double> plane_sums(kPlanes, 0);
  for (const Position& p : positions) {
    legal += p.legal.size();
    for (Action a : p.legal) legal_sum += a;
    captures += p.captures.size();
    for (Action a : p.captures) captures_sum += a;
    const int plane_size = p.observation.size() / kPlanes;
    for (int i = 0; i < p.observation.size(); ++i) {
      plane_sums[i / plane_size] += p.observation[i];
    }
  }
  const std::vector<std::pair<std::string, int64_t>> counts = {
      {"positions", static_cast<int64_t>(positions.size())}, {"legal", legal},       {"legal_sum", legal_sum},
      {"captures", captures},          {"captures_sum", captures_sum}};
  for (const auto& [key, mine] : counts) {
    if (expected.at(key).GetInt() != mine) {
      open_spiel::SpielFatalError(absl::StrCat("The ", name, " positions differ from the "
                                               "export: ", key, " ", mine, " instead of ",
                                               expected.at(key).GetInt()));
    }
  }
  const open_spiel::json::Array& sums = expected.at("plane_sums").GetArray();
  SPIEL_CHECK_EQ(sums.size(), kPlanes);
  for (int plane = 0; plane < kPlanes; ++plane) {
    const double theirs =
        sums[plane].IsInt() ? sums[plane].GetInt() : sums[plane].GetDouble();
    if (std::abs(plane_sums[plane] - theirs) > 0.01) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "The %s positions differ from the export: plane %d sums to %.3f instead of %.3f",
          name, plane, plane_sums[plane], theirs));
    }
  }
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) open_spiel::SpielFatalError(absl::StrCat("Cannot read ", path));
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

// The value target: does the side to move have a capture? The policy target: uniform
// over the captures if there are any, else over all legal moves.
VPNetModel::TrainInputs Target(const Position& p) {
  const std::vector<Action>& target = p.captures.empty() ? p.legal : p.captures;
  open_spiel::ActionsAndProbs policy;
  for (Action a : target) policy.push_back({a, 1.0 / target.size()});
  return {p.legal, p.observation, policy, p.captures.empty() ? -1.0 : 1.0};
}

// Value sign accuracy and policy mass on the captures, per side to move, as JSON.
std::string Evaluate(VPNetModel& model, const std::vector<Position>& test) {
  struct Stats {
    int n = 0, right = 0, with_capture = 0;
    double mass = 0;
  };
  std::map<std::string, Stats> stats;
  for (int start = 0; start < test.size(); start += kEvalBatch) {
    std::vector<VPNetModel::InferenceInputs> inputs;
    for (int i = start; i < std::min<int>(start + kEvalBatch, test.size()); ++i) {
      inputs.push_back({test[i].legal, test[i].observation});
    }
    std::vector<VPNetModel::InferenceOutputs> outputs = model.Inference(inputs);
    for (int row = 0; row < outputs.size(); ++row) {
      const Position& p = test[start + row];
      Stats& s = stats[p.trolls_to_move ? "trolls" : "dwarfs"];
      s.n += 1;
      s.right += (outputs[row].value > 0) == !p.captures.empty();
      if (!p.captures.empty()) {
        s.with_capture += 1;
        for (const auto& [action, probability] : outputs[row].policy) {
          for (Action c : p.captures) s.mass += c == action ? probability : 0;
        }
      }
    }
  }
  std::string out;
  for (const auto& [side, s] : stats) {  // std::map: dwarfs, then trolls.
    absl::StrAppend(&out, out.empty() ? "" : ", ",
                    absl::StrFormat("\"%s\": {\"value_accuracy\": %.4f, "
                                    "\"capture_mass\": %.4f}",
                                    side, static_cast<double>(s.right) / s.n,
                                    s.mass / std::max(1, s.with_capture)));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " DATA_DIR SEED" << std::endl;
    return 1;
  }
  const std::string data = argv[1];
  const int seed = std::stoi(argv[2]);
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  const std::vector<Position> train = Replay(*game, data + "/train_games.txt");
  const std::vector<Position> test = Replay(*game, data + "/test_games.txt");
  const open_spiel::json::Object summary =
      open_spiel::json::FromString(ReadFile(data + "/summary.json"))->GetObject();
  CheckAgainstExport(train, summary.at("train").GetObject(), "training");
  CheckAgainstExport(test, summary.at("test").GetObject(), "test");
  const int batch_size = summary.at("batch_size").GetInt();

  // The batches az_layout_check.py draws for this seed, in order.
  std::ifstream batches_file(absl::StrCat(data, "/batches_seed", seed, ".bin"),
                             std::ios::binary);
  if (!batches_file) open_spiel::SpielFatalError("No batches for this seed");
  std::vector<int32_t> indices;
  for (int32_t i; batches_file.read(reinterpret_cast<char*>(&i), sizeof(i));) {
    indices.push_back(i);  // Little-endian, as this machine.
  }
  const int steps = summary.at("steps").GetInt();
  SPIEL_CHECK_EQ(indices.size(), steps * batch_size);

  const std::string model_dir = absl::StrCat(data, "/model_seed", seed);
  std::filesystem::create_directories(model_dir);
  SPIEL_CHECK_TRUE(open_spiel::algorithms::torch_az::CreateGraphDef(
      *game, kLearningRate, kWeightDecay, model_dir, "vpnet.pb", "resnet", kNnWidth,
      kNnDepth));
  torch::manual_seed(seed);  // The weights are initialised in the constructor.
  VPNetModel model(*game, model_dir, "vpnet.pb", "/cpu:0");

  std::cout << absl::StrFormat(
                   "{\"variant\": \"cpp\", \"seed\": %d, \"train\": %d, \"test\": %d, "
                   "\"steps\": %d, \"checksums\": \"match the Python export\"}",
                   seed, train.size(), test.size(), steps)
            << std::endl;
  const auto start = std::chrono::steady_clock::now();
  for (int step = 1; step <= steps; ++step) {
    std::vector<VPNetModel::TrainInputs> inputs;
    for (int j = 0; j < batch_size; ++j) {
      inputs.push_back(Target(train[indices[(step - 1) * batch_size + j]]));
    }
    VPNetModel::LossInfo losses = model.Learn(inputs);
    if (step % kEvalEvery == 0) {
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << absl::StrFormat(
                       "{\"variant\": \"cpp\", \"seed\": %d, \"step\": %d, "
                       "\"seconds\": %.0f, \"loss_policy\": %.4f, \"loss_value\": "
                       "%.4f, %s}",
                       seed, step, seconds, losses.Policy(), losses.Value(),
                       Evaluate(model, test))
                << std::endl;
    }
  }
  return 0;
}
