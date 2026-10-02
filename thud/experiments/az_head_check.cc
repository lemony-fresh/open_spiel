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

// Does the convolutional policy head (nn_model "resnet_conv_policy", thud/az/model.h)
// learn Thud's board better than upstream's linear one, and does either overfit?
// (thud/PLAN.md Phase 6.)
//
// az_layout_check.cc's supervised task, on our copy of the C++ AlphaZero (thud/az/):
// the value head learns whether the side to move has a capture, the policy target is
// uniform over the capturing moves if there are any, else over all legal moves.
// Positions of random games, exported by az_layout_check.py --export (training and
// test positions from different games), trained on the same batches in the same order
// as az_layout_check.cc. With nn_model=resnet width=32 depth=2 it must reproduce that
// program's results (thud/PLAN.md, "Thud layout control"), as our copy computes what
// upstream's does (identity_check).
//
// It evaluates on the test positions and, for overfitting, on as many training
// positions spread over all training games: value sign accuracy and policy mass on the
// captures per side to move, and both losses (policy cross-entropy with the target,
// value squared error) on each set. A test loss rising while the training loss falls
// is overfitting. On the test positions it also measures how differently the network
// answers a position and its mirror image (SymmetryGap).
//
// With augment=1 each training position is turned or mirrored by a random one of the
// board's 8 symmetries (SymmetricTrainInputs, thud/az/vpnet.h), as the trainer's
// --symmetry_augmentation does.
//
//   thud/az/build.sh thud/experiments/az_head_check.cc
//   python3 thud/experiments/az_layout_check.py --export DATA
//   OMP_NUM_THREADS=3 build-shared/az_head_check DATA SEED [nn_model=resnet]
//     [width=32] [depth=2] [steps=all] [augment=0]
//
// Prints one JSON line per evaluation, every 250 steps. With timing=1 it trains nothing and prints the
// network's time per inference of 1 and of 32 positions and per learning step of 128,
// on the test positions.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/json.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::thud_az::torch_az::SymmetricTrainInputs;
using open_spiel::thud_az::torch_az::VPNetModel;

constexpr int kEvery = 7;  // As az_layout_check.py samples.
constexpr int kPlanes = 6;
constexpr int kTrollsToMovePlane = 3;
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
      {"positions", static_cast<int64_t>(positions.size())},
      {"legal", legal},
      {"legal_sum", legal_sum},
      {"captures", captures},
      {"captures_sum", captures_sum}};
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

// The policy target's moves: the captures if there are any, else all legal moves.
const std::vector<Action>& TargetMoves(const Position& p) {
  return p.captures.empty() ? p.legal : p.captures;
}

// The value target: does the side to move have a capture? The policy target: uniform
// over TargetMoves.
VPNetModel::TrainInputs Target(const Position& p) {
  const std::vector<Action>& target = TargetMoves(p);
  open_spiel::ActionsAndProbs policy;
  for (Action a : target) policy.push_back({a, 1.0 / target.size()});
  return {p.legal, p.observation, policy, p.captures.empty() ? -1.0 : 1.0};
}

// Value sign accuracy and policy mass on the captures per side to move, and both
// losses over all positions, as JSON.
std::string Evaluate(VPNetModel& model, const std::vector<Position>& positions) {
  struct Stats {
    int n = 0, right = 0, with_capture = 0;
    double mass = 0;
  };
  std::map<std::string, Stats> stats;
  double policy_loss = 0, value_loss = 0;
  for (int start = 0; start < positions.size(); start += kEvalBatch) {
    std::vector<VPNetModel::InferenceInputs> inputs;
    for (int i = start; i < std::min<int>(start + kEvalBatch, positions.size()); ++i) {
      inputs.push_back({positions[i].legal, positions[i].observation});
    }
    std::vector<VPNetModel::InferenceOutputs> outputs = model.Inference(inputs);
    for (int row = 0; row < outputs.size(); ++row) {
      const Position& p = positions[start + row];
      Stats& s = stats[p.trolls_to_move ? "trolls" : "dwarfs"];
      s.n += 1;
      s.right += (outputs[row].value > 0) == !p.captures.empty();
      const double value_target = p.captures.empty() ? -1 : 1;
      value_loss += std::pow(outputs[row].value - value_target, 2);
      // The policy holds the legal moves in order, and the target moves are some of
      // them, in order too.
      const std::vector<Action>& target = TargetMoves(p);
      int t = 0;
      for (const auto& [action, probability] : outputs[row].policy) {
        if (t < target.size() && target[t] == action) {
          policy_loss -= std::log(std::max(probability, 1e-30)) / target.size();
          ++t;
          if (!p.captures.empty()) s.mass += probability;
        }
      }
      SPIEL_CHECK_EQ(t, target.size());
      if (!p.captures.empty()) s.with_capture += 1;
    }
  }
  std::string out = absl::StrFormat("\"policy_loss\": %.4f, \"value_loss\": %.4f",
                                    policy_loss / positions.size(),
                                    value_loss / positions.size());
  for (const auto& [side, s] : stats) {  // std::map: dwarfs, then trolls.
    absl::StrAppend(&out,
                    absl::StrFormat(", \"%s\": {\"value_accuracy\": %.4f, "
                                    "\"capture_mass\": %.4f}",
                                    side, static_cast<double>(s.right) / s.n,
                                    s.mass / std::max(1, s.with_capture)));
  }
  return out;
}

// How differently the network answers a position and its mirror image: the mean
// difference of the values and the mean total variation distance of the policies
// (each move against its mirror image), position i turned or mirrored by symmetry
// 1 + i % 7. An exactly symmetric network scores 0 on both.
std::string SymmetryGap(VPNetModel& model, const std::vector<Position>& positions) {
  double value_gap = 0, policy_gap = 0;
  for (int start = 0; start < positions.size(); start += kEvalBatch) {
    std::vector<VPNetModel::InferenceInputs> inputs, images;
    for (int i = start; i < std::min<int>(start + kEvalBatch, positions.size()); ++i) {
      const Position& p = positions[i];
      inputs.push_back({p.legal, p.observation});
      const VPNetModel::TrainInputs image =
          SymmetricTrainInputs({p.legal, p.observation, {}, 0}, 1 + i % 7);
      images.push_back({image.legal_actions, image.observations});
    }
    const std::vector<VPNetModel::InferenceOutputs> outputs = model.Inference(inputs);
    const std::vector<VPNetModel::InferenceOutputs> image_outputs =
        model.Inference(images);
    for (int row = 0; row < outputs.size(); ++row) {
      value_gap += std::abs(outputs[row].value - image_outputs[row].value);
      std::map<Action, double> image_policy(image_outputs[row].policy.begin(),
                                            image_outputs[row].policy.end());
      double distance = 0;
      for (const auto& [action, probability] : outputs[row].policy) {
        distance += std::abs(probability -
                             image_policy.at(open_spiel::thud::SymmetricAction(
                                 action, 1 + (start + row) % 7)));
      }
      policy_gap += distance / 2;
    }
  }
  return absl::StrFormat("\"symmetry_value_gap\": %.4f, \"symmetry_policy_tv\": %.4f",
                         value_gap / positions.size(), policy_gap / positions.size());
}

// Milliseconds per call, the median of `calls` after 3 to warm up.
template <typename F>
double Milliseconds(int calls, F f) {
  for (int i = 0; i < 3; ++i) f();
  std::vector<double> times;
  for (int i = 0; i < calls; ++i) {
    const auto start = std::chrono::steady_clock::now();
    f();
    times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

std::string Timing(VPNetModel& model, const std::vector<Position>& test) {
  auto inputs = [&test](int n, int offset) {
    std::vector<VPNetModel::InferenceInputs> batch;
    for (int i = 0; i < n; ++i) {
      const Position& p = test[(offset + i) % test.size()];
      batch.push_back({p.legal, p.observation});
    }
    return batch;
  };
  int offset = 0;
  const double one = Milliseconds(200, [&] { model.Inference(inputs(1, offset++)); });
  const double batch = Milliseconds(100, [&] {
    model.Inference(inputs(32, offset));
    offset += 32;
  });
  const double learn = Milliseconds(20, [&] {
    std::vector<VPNetModel::TrainInputs> train;
    for (int i = 0; i < 128; ++i) train.push_back(Target(test[(offset + i) % test.size()]));
    offset += 128;
    model.Learn(train);
  });
  return absl::StrFormat("\"inference_1_ms\": %.3f, \"inference_32_ms\": %.3f, "
                         "\"learn_128_ms\": %.1f",
                         one, batch, learn);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " DATA_DIR SEED [nn_model=resnet] [width=32] [depth=2] [timing=0] "
                 "[steps=all] [augment=0]"
              << std::endl;
    return 1;
  }
  const std::string data = argv[1];
  const int seed = std::stoi(argv[2]);
  std::map<std::string, std::string> options = {
      {"nn_model", "resnet"}, {"width", "32"}, {"depth", "2"}, {"timing", "0"},
      {"steps", "0"}, {"augment", "0"}};
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos || !options.count(arg.substr(0, eq))) {
      open_spiel::SpielFatalError(absl::StrCat("Unknown option: ", arg));
    }
    options[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  const std::string nn_model = options["nn_model"];
  const int width = std::stoi(options["width"]);
  const int depth = std::stoi(options["depth"]);
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  const std::vector<Position> train = Replay(*game, data + "/train_games.txt");
  const std::vector<Position> test = Replay(*game, data + "/test_games.txt");
  const open_spiel::json::Object summary =
      open_spiel::json::FromString(ReadFile(data + "/summary.json"))->GetObject();
  CheckAgainstExport(train, summary.at("train").GetObject(), "training");
  CheckAgainstExport(test, summary.at("test").GetObject(), "test");
  const int batch_size = summary.at("batch_size").GetInt();
  // As many training positions as test positions, spread over all training games.
  std::vector<Position> train_sample;
  for (int i = 0; i < test.size(); ++i) {
    train_sample.push_back(train[static_cast<int64_t>(i) * train.size() / test.size()]);
  }

  // The batches az_layout_check.py draws for this seed, in order.
  std::ifstream batches_file(absl::StrCat(data, "/batches_seed", seed, ".bin"),
                             std::ios::binary);
  if (!batches_file) open_spiel::SpielFatalError("No batches for this seed");
  std::vector<int32_t> indices;
  for (int32_t i; batches_file.read(reinterpret_cast<char*>(&i), sizeof(i));) {
    indices.push_back(i);  // Little-endian, as this machine.
  }
  SPIEL_CHECK_EQ(indices.size(), summary.at("steps").GetInt() * batch_size);
  // Fewer steps than exported, for a quick try.
  const int steps = options["steps"] != "0" ? std::stoi(options["steps"])
                                            : summary.at("steps").GetInt();
  SPIEL_CHECK_LE(steps, summary.at("steps").GetInt());

  const bool augment = options["augment"] == "1";
  const std::string variant = absl::StrFormat("%s_%dx%d%s", nn_model, width, depth,
                                              augment ? "_augmented" : "");
  const std::string model_dir = absl::StrCat(data, "/model_", variant, "_seed", seed);
  std::filesystem::create_directories(model_dir);
  SPIEL_CHECK_TRUE(open_spiel::thud_az::torch_az::CreateGraphDef(
      *game, kLearningRate, kWeightDecay, model_dir, "vpnet.pb", nn_model, width,
      depth));
  torch::manual_seed(seed);  // The weights are initialised in the constructor.
  VPNetModel model(*game, model_dir, "vpnet.pb", "/cpu:0");

  if (options["timing"] == "1") {
    std::cout << absl::StrFormat("{\"variant\": \"%s\", \"timing\": {%s}}", variant,
                                 Timing(model, test))
              << std::endl;
    return 0;
  }
  std::cout << absl::StrFormat(
                   "{\"variant\": \"%s\", \"seed\": %d, \"train\": %d, \"test\": %d, "
                   "\"train_sample\": %d, \"steps\": %d, \"checksums\": \"match the "
                   "Python export\"}",
                   variant, seed, train.size(), test.size(), train_sample.size(), steps)
            << std::endl;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> symmetry(0, open_spiel::thud::kNumSymmetries - 1);
  const auto start = std::chrono::steady_clock::now();
  for (int step = 1; step <= steps; ++step) {
    std::vector<VPNetModel::TrainInputs> inputs;
    for (int j = 0; j < batch_size; ++j) {
      inputs.push_back(Target(train[indices[(step - 1) * batch_size + j]]));
      if (augment) inputs.back() = SymmetricTrainInputs(inputs.back(), symmetry(rng));
    }
    VPNetModel::LossInfo losses = model.Learn(inputs);
    if (step % kEvalEvery == 0) {
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << absl::StrFormat(
                       "{\"variant\": \"%s\", \"seed\": %d, \"step\": %d, "
                       "\"seconds\": %.0f, \"loss_policy\": %.4f, \"loss_value\": "
                       "%.4f, \"test\": {%s, %s}, \"train\": {%s}}",
                       variant, seed, step, seconds, losses.Policy(), losses.Value(),
                       Evaluate(model, test), SymmetryGap(model, test),
                       Evaluate(model, train_sample))
                << std::endl;
    }
  }
  return 0;
}
