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

// Checks the convolutional policy head of our copy of OpenSpiel's C++ AlphaZero
// (nn_model "resnet_conv_policy", thud/az/model.h; thud/PLAN.md Phase 6):
//
//   1. Layout: on random inputs and weights, each action's logit is the entry of its
//      own plane at the moving piece's square, the plane worked out here from the
//      decoded action rather than by thud::PolicyPlaneIndex. Control: the same
//      comparison with rows and columns swapped must fail.
//   2. Masking: on positions of random games, a network's policy covers exactly the
//      legal actions and sums to 1. Without the mask the legal actions, a few hundred
//      of 19,800, would get a few percent.
//   3. Size: the weights of both heads at our runs' width 64 and depth 4.
//   4. Learning: one step gives every weight of the new model a gradient, and both
//      heads fit a fixed batch of positions with random targets: the policy loss falls
//      below half its start.
//   5. Checkpoints: a reloaded checkpoint gives the saved network's outputs. Control:
//      a fresh network's differ.
//
//   thud/az/build.sh thud/az/conv_policy_check.cc
//   OMP_NUM_THREADS=4 build-shared/conv_policy_check
//
// Exits non-zero on any failure. Writes its networks to a temporary folder (TMPDIR)
// and removes it.

#include <stdlib.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/model.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az::torch_az;
namespace thud = open_spiel::thud;
using open_spiel::Action;
using open_spiel::Game;
using open_spiel::State;

constexpr int kWidth = 64;  // Our runs' network (thud/PLAN.md Phase 6).
constexpr int kDepth = 4;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

// Positions of random games, every 7th turn: 7 is odd, so both sides are to move.
std::vector<std::unique_ptr<State>> Positions(const Game& game, int count,
                                              int seed) {
  std::mt19937 rng(seed);
  std::vector<std::unique_ptr<State>> positions;
  while (positions.size() < count) {
    std::unique_ptr<State> state = game.NewInitialState();
    for (int turn = 0; !state->IsTerminal() && positions.size() < count;
         ++turn) {
      if (turn % 7 == 0) positions.push_back(state->Clone());
      const std::vector<Action> actions = state->LegalActions();
      state->ApplyAction(actions[std::uniform_int_distribution<int>(
          0, actions.size() - 1)(rng)]);
    }
  }
  return positions;
}

az::ModelConfig Config(const Game& game, const std::string& nn_model) {
  return az::WithPolicyMap(
      game, {game.ObservationTensorShape(), game.NumDistinctActions(), kDepth,
             kWidth, /*learning_rate=*/1e-3, /*weight_decay=*/1e-4, nn_model});
}

void CheckLayout(const Game& game) {
  const az::ModelConfig config = Config(game, "resnet_conv_policy");
  Expect(config.policy_planes == 120 && config.policy_map.size() == 19800,
         "the map has 120 planes and an entry for each of the 19,800 actions");
  az::ResOutputBlockConfig output_config = {
      /*input_channels=*/kWidth, /*value_filters=*/1, /*policy_filters=*/2,
      /*kernel_size=*/1, /*padding=*/0, /*value_linear_in_features=*/225,
      /*value_linear_out_features=*/kWidth,
      /*policy_linear_in_features=*/450,
      /*policy_linear_out_features=*/19800, /*value_observation_size=*/225,
      /*policy_observation_size=*/450};
  az::ResConvPolicyOutputBlock block(output_config, config.policy_planes,
                                     config.policy_map);
  block->eval();
  torch::NoGradGuard no_grad;
  torch::manual_seed(1);
  const torch::Tensor x = torch::randn({2, kWidth, 15, 15});
  const torch::Tensor mask = torch::ones({2, 19800}, torch::kByte);
  const torch::Tensor logits = block->forward(x, mask)[1].contiguous();
  const torch::Tensor planes = block->PolicyPlanes(x).contiguous();
  const auto logit = logits.accessor<float, 2>();
  const auto entry = planes.accessor<float, 4>();
  int matches = 0, swapped_matches = 0;
  for (int b = 0; b < 2; ++b) {
    for (Action action = 0; action < 19800; ++action) {
      const thud::DecodedAction parts = thud::DecodeAction(action);
      const thud::Coord coord = thud::SquareCoord(parts.square);
      const int plane = parts.capture_step
                            ? 8 * 14 + parts.direction
                            : parts.direction * 14 + parts.distance - 1;
      const float value = logit[b][action];
      matches +=
          std::abs(value - entry[b][plane][coord.row][coord.col]) < 1e-6;
      swapped_matches +=
          std::abs(value - entry[b][plane][coord.col][coord.row]) < 1e-6;
    }
  }
  Expect(matches == 2 * 19800,
         absl::StrFormat("each action's logit is its plane's entry at its "
                         "square: %d of %d",
                         matches, 2 * 19800));
  Expect(swapped_matches < matches / 10,
         absl::StrFormat("control, rows and columns swapped: %d of %d match "
                         "(only squares on the diagonal should)",
                         swapped_matches, 2 * 19800));
}

std::vector<az::VPNetModel::InferenceInputs> Inputs(
    const std::vector<std::unique_ptr<State>>& positions) {
  std::vector<az::VPNetModel::InferenceInputs> inputs;
  for (const auto& state : positions) {
    inputs.push_back({state->LegalActions(), state->ObservationTensor()});
  }
  return inputs;
}

void CheckMasking(const Game& game, const std::string& dir) {
  SPIEL_CHECK_TRUE(az::CreateGraphDef(game, 1e-3, 1e-4, dir, "conv.pb",
                                      "resnet_conv_policy", kWidth, kDepth));
  az::VPNetModel model(game, dir, "conv.pb", "/cpu:0");
  const std::vector<az::VPNetModel::InferenceInputs> inputs =
      Inputs(Positions(game, 64, /*seed=*/2));
  const std::vector<az::VPNetModel::InferenceOutputs> outputs =
      model.Inference(inputs);
  int good = 0;
  double worst = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    bool same_actions =
        outputs[i].policy.size() == inputs[i].legal_actions.size();
    double sum = 0;
    for (int j = 0; same_actions && j < outputs[i].policy.size(); ++j) {
      same_actions = outputs[i].policy[j].first == inputs[i].legal_actions[j];
      sum += outputs[i].policy[j].second;
    }
    worst = std::max(worst, std::abs(sum - 1));
    good += same_actions && std::abs(sum - 1) < 1e-4 &&
            std::abs(outputs[i].value) <= 1;
  }
  Expect(good == inputs.size(),
         absl::StrFormat("policies cover the legal actions and sum to 1 "
                         "(largest error %.2g), values in [-1, 1]: %d of %d",
                         worst, good, inputs.size()));
}

int64_t Weights(az::Model& model) {
  int64_t weights = 0;
  for (const torch::Tensor& parameter : model->parameters()) {
    weights += parameter.numel();
  }
  return weights;
}

void CheckSize(const Game& game) {
  az::Model flat(Config(game, "resnet"), "cpu");
  az::Model conv(Config(game, "resnet_conv_policy"), "cpu");
  const int64_t flat_weights = Weights(flat), conv_weights = Weights(conv);
  std::cout << "        weights at width " << kWidth << ", depth " << kDepth
            << ": resnet " << flat_weights << ", resnet_conv_policy "
            << conv_weights << std::endl;
  Expect(conv_weights < flat_weights / 10,
         "the convolutional head needs less than a tenth of the weights");
}

// Training tensors as VPNetModel::Learn builds them: one-hot targets on a random
// legal action and random values of -1 or 1.
std::vector<az::VPNetModel::TrainInputs> Targets(
    const std::vector<std::unique_ptr<State>>& positions, int seed) {
  std::mt19937 rng(seed);
  std::vector<az::VPNetModel::TrainInputs> batch;
  for (const auto& state : positions) {
    const std::vector<Action> actions = state->LegalActions();
    const Action target = actions[std::uniform_int_distribution<int>(
        0, actions.size() - 1)(rng)];
    batch.push_back({actions, state->ObservationTensor(), {{target, 1.0}},
                     std::bernoulli_distribution(0.5)(rng) ? 1.0 : -1.0});
  }
  return batch;
}

void CheckGradients(const Game& game) {
  az::Model model(Config(game, "resnet_conv_policy"), "cpu");
  const std::vector<az::VPNetModel::TrainInputs> batch =
      Targets(Positions(game, 32, /*seed=*/3), /*seed=*/4);
  const int n = batch.size(), size = game.ObservationTensorSize();
  torch::Tensor inputs = torch::zeros({n, size});
  torch::Tensor mask = torch::zeros({n, 19800}, torch::kByte);
  torch::Tensor policy = torch::zeros({n, 19800});
  torch::Tensor value = torch::zeros({n, 1});
  for (int i = 0; i < n; ++i) {
    inputs[i] = torch::tensor(batch[i].observations);
    for (Action action : batch[i].legal_actions) mask[i][action] = 1;
    for (const auto& [action, p] : batch[i].policy) policy[i][action] = p;
    value[i][0] = batch[i].value;
  }
  model->train();
  model->zero_grad();
  const std::vector<torch::Tensor> losses =
      model->losses(inputs, mask, policy, value);
  (losses[0] + losses[1] + losses[2]).backward();
  int with_gradient = 0, total = 0;
  for (const auto& parameter : model->named_parameters()) {
    ++total;
    const torch::Tensor& grad = parameter.value().grad();
    if (grad.defined() && grad.abs().sum().item<double>() > 0) {
      ++with_gradient;
    } else {
      std::cout << "        no gradient: " << parameter.key() << std::endl;
    }
  }
  Expect(with_gradient == total,
         absl::StrFormat("every weight tensor gets a gradient: %d of %d",
                         with_gradient, total));
}

void CheckLearning(const Game& game, const std::string& dir,
                   const std::string& nn_model, int steps) {
  SPIEL_CHECK_TRUE(az::CreateGraphDef(game, 1e-3, 1e-4, dir, nn_model + ".pb",
                                      nn_model, kWidth, kDepth));
  az::VPNetModel model(game, dir, nn_model + ".pb", "/cpu:0");
  const std::vector<az::VPNetModel::TrainInputs> batch =
      Targets(Positions(game, 64, /*seed=*/5), /*seed=*/6);
  const double start = model.Learn(batch).Policy();
  double end = start;
  for (int step = 1; step < steps; ++step) end = model.Learn(batch).Policy();
  Expect(end < start / 2,
         absl::StrFormat("%s fits a fixed batch of 64: policy loss %.3f -> "
                         "%.3f in %d steps",
                         nn_model, start, end, steps));
}

double LargestDifference(const std::vector<az::VPNetModel::InferenceOutputs>& a,
                         const std::vector<az::VPNetModel::InferenceOutputs>& b) {
  double largest = 0;
  for (int i = 0; i < a.size(); ++i) {
    largest = std::max(largest, std::abs(a[i].value - b[i].value));
    for (int j = 0; j < a[i].policy.size(); ++j) {
      largest = std::max(largest, std::abs(a[i].policy[j].second -
                                           b[i].policy[j].second));
    }
  }
  return largest;
}

void CheckCheckpoint(const Game& game, const std::string& dir) {
  SPIEL_CHECK_TRUE(az::CreateGraphDef(game, 1e-3, 1e-4, dir, "saved.pb",
                                      "resnet_conv_policy", kWidth, kDepth));
  const std::vector<az::VPNetModel::InferenceInputs> inputs =
      Inputs(Positions(game, 16, /*seed=*/7));
  az::VPNetModel saved(game, dir, "saved.pb", "/cpu:0");
  saved.Learn(Targets(Positions(game, 16, /*seed=*/8), /*seed=*/9));
  const std::string path = saved.SaveCheckpoint(0);
  az::VPNetModel loaded(game, dir, "saved.pb", "/cpu:0");
  const double fresh = LargestDifference(saved.Inference(inputs),
                                         loaded.Inference(inputs));
  loaded.LoadCheckpoint(path);
  const double reloaded = LargestDifference(saved.Inference(inputs),
                                            loaded.Inference(inputs));
  Expect(reloaded < 1e-6,
         absl::StrFormat("a reloaded checkpoint gives the same outputs "
                         "(largest difference %.2g)",
                         reloaded));
  Expect(fresh > 1e-3, absl::StrFormat("control, a fresh network's outputs "
                                       "differ (largest difference %.2g)",
                                       fresh));
}

}  // namespace

int main(int argc, char** argv) {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "conv_policy_check_XXXXXX")
          .string();
  if (mkdtemp(pattern.data()) == nullptr) {
    std::cerr << "Cannot create a temporary folder." << std::endl;
    return 1;
  }
  const std::string dir = pattern;
  std::shared_ptr<const Game> game = open_spiel::LoadGame("thud");
  torch::manual_seed(0);
  CheckLayout(*game);
  CheckMasking(*game, dir);
  CheckSize(*game);
  CheckGradients(*game);
  CheckLearning(*game, dir, "resnet", /*steps=*/150);
  CheckLearning(*game, dir, "resnet_conv_policy", /*steps=*/150);
  CheckCheckpoint(*game, dir);
  std::filesystem::remove_all(dir);
  std::cout << (failures ? "CONV POLICY CHECK FAILED"
                         : "conv policy check passed")
            << std::endl;
  return failures ? 1 : 0;
}
