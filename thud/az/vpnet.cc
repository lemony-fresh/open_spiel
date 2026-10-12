// Copyright 2021 DeepMind Technologies Limited
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

// Changed by the Thud-on-OpenSpiel authors, 2026: copied from
// open_spiel/algorithms/alpha_zero_torch/vpnet.cc at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#include "thud/az/vpnet.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <fstream>  // For ifstream/ofstream.
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/thud/thud.h"
#include "thud/az/model.h"
#include "thud/az/thud_features.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace thud_az {
namespace torch_az {

// Saves a struct that holds initialization data for the model to a file.
//
// The TensorFlow version creates a TensorFlow graph definition when
// CreateGraphDef is called. To avoid having to change this, allow calls to
// CreateGraphDef, however now it simply saves a struct to a file which can
// then be loaded and used to initialize a model.
bool SaveModelConfig(const std::string& path, const std::string& filename,
                     const ModelConfig& net_config) {
  std::ofstream file;
  file.open(absl::StrCat(path, "/", filename));

  if (!file) {
    return false;
  } else {
    file << net_config;
  }
  file.close();

  return true;
}

// Loads a struct that holds initialization data for the model from a file.
//
// The TensorFlow version creates a TensorFlow graph definition when
// CreateGraphDef is called. To avoid having to change this, allow calls to
// CreateGraphDef, however now it simply saves a struct to a file which can
// then be loaded and used to initialize a model.
ModelConfig LoadModelConfig(const std::string& path,
                            const std::string& filename) {
  std::ifstream file;
  file.open(absl::StrCat(path, "/", filename));
  ModelConfig net_config;

  file >> net_config;
  file.close();

  return net_config;
}

// Modifies a given device string to one that can be accepted by the
// Torch library.
//
// The Torch library accepts 'cpu', 'cpu:0', 'cuda:0', 'cuda:1',
// 'cuda:2', 'cuda:3'..., but complains when there's a slash in front
// of the device name.
//
// Currently, this function only disregards a slash if it exists at the
// beginning of the device string, more functionality can be added if
// needed.
std::string TorchDeviceName(const std::string& device) {
  if (device[0] == '/') {
    return device.substr(1);
  }
  return device;
}

// Our change: the convolutional policy head's layout of the actions over the
// board (ModelConfig::policy_map). Only Thud defines one
// (open_spiel/games/thud/thud.h).
ModelConfig WithPolicyMap(const Game& game, ModelConfig config) {
  if (config.nn_model != "resnet_conv_policy") return config;
  SPIEL_CHECK_EQ(game.GetType().short_name, "thud");
  config.policy_planes = thud::kNumPolicyPlanes;
  for (Action action = 0; action < game.NumDistinctActions(); ++action) {
    config.policy_map.push_back(thud::PolicyPlaneIndex(action));
  }
  return config;
}

bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& filename,
                    std::string nn_model, int nn_width, int nn_depth,
                    bool verbose, int extra_input_planes) {
  std::vector<int> input_shape = game.ObservationTensorShape();
  input_shape[0] += extra_input_planes;  // Our change: the input planes.
  ModelConfig net_config = {
      /*observation_tensor_shape=*/input_shape,
      /*number_of_actions=*/game.NumDistinctActions(),
      /*nn_depth=*/nn_depth,
      /*nn_width=*/nn_width,
      /*learning_rate=*/learning_rate,
      /*weight_decay=*/weight_decay,
      /*nn_model=*/nn_model};

  return SaveModelConfig(path, filename, net_config);
}

VPNetModel::VPNetModel(const Game& game, const std::string& path,
                       const std::string& file_name, const std::string& device)
    : device_(device),
      path_(path),
      flat_input_size_(game.ObservationTensorSize()),
      observation_size_(game.ObservationTensorSize()),
      num_actions_(game.NumDistinctActions()),
      model_config_(WithPolicyMap(game, LoadModelConfig(path, file_name))),
      model_(model_config_, TorchDeviceName(device)),
      model_optimizer_(
          model_->parameters(),
          torch::optim::AdamOptions(  // NOLINT(misc-include-cleaner)
              model_config_.learning_rate)),
      torch_device_(TorchDeviceName(device)),
      side_to_move_index_(game.GetType().short_name == "thud"
                              ? thud::kTrollsToMovePlane * thud::kBoardSize *
                                    thud::kBoardSize
                              : -1) {
  // Some assumptions that we can remove eventually. The value net returns
  // a single value in terms of player 0 and the game is assumed to be zero-sum,
  // so player 1 can just be -value.
  SPIEL_CHECK_EQ(game.NumPlayers(), 2);
  SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);

  // Put this model on the specified device.
  model_->to(torch_device_);

  // Our change (thud/PLAN.md, the input planes): a model whose input has the input
  // planes from the rules besides the observation's.
  int input_size = 1;
  for (int d : model_config_.observation_tensor_shape) input_size *= d;
  if (input_size != observation_size_) {
    SPIEL_CHECK_EQ(game.GetType().short_name, "thud");
    SPIEL_CHECK_EQ(input_size, observation_size_ + kNumFeaturePlanes * thud::kBoardSize *
                                                       thud::kBoardSize);
    input_planes_ = true;
    flat_input_size_ = input_size;
  }
}

void VPNetModel::ModelInput(const std::vector<float>& observation, float* out) const {
  SPIEL_CHECK_EQ(observation.size(), observation_size_);
  std::copy(observation.begin(), observation.end(), out);
  if (input_planes_) {
    std::vector<float> planes;
    planes.reserve(flat_input_size_ - observation_size_);
    AppendFeaturePlanes(observation, &planes);
    std::copy(planes.begin(), planes.end(), out + observation_size_);
  }
}

torch::Tensor VPNetModel::Parameter(const std::string& name) {
  const torch::Tensor* p = model_->named_parameters().find(name);
  if (p == nullptr) SpielFatalError("No parameter " + name);
  return *p;
}

std::vector<std::string> VPNetModel::CopyParametersFrom(
    VPNetModel& other, const std::function<bool(const std::string&)>& include,
    bool widen_inputs) {
  std::vector<std::string> copied;
  torch::NoGradGuard no_grad;
  // More input channels here, otherwise alike.
  auto wider = [](const torch::Tensor& to, const torch::Tensor& from) {
    if (to.dim() < 2 || to.dim() != from.dim() || to.size(1) <= from.size(1)) return false;
    for (int64_t d = 0; d < to.dim(); ++d) {
      if (d != 1 && to.size(d) != from.size(d)) return false;
    }
    return true;
  };
  auto copy = [&](const torch::OrderedDict<std::string, torch::Tensor>& to,
                  const torch::OrderedDict<std::string, torch::Tensor>& from) {
    for (const auto& item : to) {
      const torch::Tensor* source = from.find(item.key());
      if (source == nullptr || !include(item.key())) continue;
      torch::Tensor target = item.value();
      if (source->sizes() == target.sizes()) {
        target.copy_(*source);
      } else if (widen_inputs && wider(target, *source)) {
        target.zero_();
        target.slice(/*dim=*/1, 0, source->size(1)).copy_(*source);
      } else {
        continue;
      }
      copied.push_back(item.key());
    }
  };
  copy(model_->named_parameters(), other.model_->named_parameters());
  copy(model_->named_buffers(), other.model_->named_buffers());
  return copied;
}

std::vector<std::string> VPNetModel::Freeze(
    const std::function<bool(const std::string&)>& which) {
  std::vector<std::string> frozen;
  for (auto& item : model_->named_parameters()) {
    if (which(item.key())) {
      item.value().set_requires_grad(false);
      frozen.push_back(item.key());
    }
  }
  for (const auto& item : model_->named_modules()) {
    if (item.value()->children().empty() && which(item.key())) {
      frozen_modules_.push_back(item.value());
    }
  }
  return frozen;
}

void VPNetModel::LoadWeights(const std::string& path) {
  torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

double VPNetModel::LearningRate() {
  return static_cast<torch::optim::AdamOptions&>(
             model_optimizer_.param_groups()[0].options())
      .lr();
}

std::string VPNetModel::SaveCheckpoint(int step) {
  std::string full_path = absl::StrCat(path_, "/checkpoint-", step);

  torch::save(model_, absl::StrCat(full_path, ".pt"));
  torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));

  return full_path;
}

void VPNetModel::LoadCheckpoint(int step) {
  // Load checkpoint from the path given at its initialization.
  LoadCheckpoint(absl::StrCat(path_, "/checkpoint-", step));
}

void VPNetModel::LoadCheckpoint(const std::string& path) {
  torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
  torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"),
              torch_device_);
}

std::vector<torch::Tensor> VPNetModel::Activations(
    const std::vector<InferenceInputs>& inputs) {
  const int batch_size = inputs.size();
  std::vector<float> raw_observations(batch_size * flat_input_size_);
  std::vector<uint8_t> raw_legal_mask(batch_size * num_actions_, 0);
  for (int batch = 0; batch < batch_size; ++batch) {
    for (Action action : inputs[batch].legal_actions) {
      raw_legal_mask[batch * num_actions_ + action] = 1;
    }
    ModelInput(inputs[batch].observations,
               raw_observations.data() + batch * flat_input_size_);  // Our change.
  }
  torch::Tensor observations =
      torch::from_blob(raw_observations.data(), {batch_size, flat_input_size_})
          .clone()
          .to(torch_device_);
  torch::Tensor legal_mask =
      torch::from_blob(raw_legal_mask.data(), {batch_size, num_actions_},
                       torch::TensorOptions().dtype(torch::kByte))
          .clone()
          .to(torch_device_);
  model_->eval();
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> activations = model_->Activations(observations, legal_mask);
  for (torch::Tensor& a : activations) a = a.to(torch::kCPU);
  return activations;
}

std::vector<VPNetModel::InferenceOutputs> VPNetModel::Inference(
    const std::vector<InferenceInputs>& inputs) {
  int inference_batch_size = inputs.size();

  // Format the data outside of torch. Random assignments can be very slow on
  // torch::Tensor objects and this approach is _much_ faster.
  std::vector<float> raw_observations(inference_batch_size * flat_input_size_);
  std::vector<uint8_t> raw_legal_mask(inference_batch_size * num_actions_, 0);

  for (int batch = 0; batch < inference_batch_size; ++batch) {
    for (Action action : inputs[batch].legal_actions) {
      raw_legal_mask[batch * num_actions_ + action] = 1;
    }
    // Our change (the input planes): the network's input, not only the observation.
    ModelInput(inputs[batch].observations,
               raw_observations.data() + batch * flat_input_size_);
  }

  // Torch tensors by default use a dense, row-aligned memory layout.
  //   - Their default data type is a 32-bit float
  //   - Use the byte data type for boolean

  // Clone first to take ownership from the raw blob, then move to device.
  // This avoids the previous pattern of .to(device).clone() which performed
  // two allocations: one for the device transfer and one for the clone.
  torch::Tensor torch_inf_inputs =
      torch::from_blob(raw_observations.data(),
                       {inference_batch_size, flat_input_size_})
          .clone()
          .to(torch_device_);
  torch::Tensor torch_inf_legal_mask =
      torch::from_blob(raw_legal_mask.data(),
                       {inference_batch_size, num_actions_},
                       torch::TensorOptions().dtype(torch::kByte))
          .clone()
          .to(torch_device_);

  // Run the inference with gradient tracking disabled.
  // NoGradGuard prevents LibTorch from building the autograd computation
  // graph, saving memory and compute since we never backprop through
  // inference.
  model_->eval();
  torch::NoGradGuard no_grad;
  std::vector<torch::Tensor> torch_outputs =
      model_(torch_inf_inputs, torch_inf_legal_mask);

  // Move outputs to CPU in a single transfer, then use accessors for
  // zero-overhead element access. This replaces the previous per-element
  // .item<>() pattern which triggered a GPU-to-CPU sync on every call.
  torch::Tensor value_cpu = torch_outputs[0].to(torch::kCPU).contiguous();
  torch::Tensor policy_cpu = torch_outputs[1].to(torch::kCPU).contiguous();

  auto value_acc = value_cpu.accessor<float, 2>();
  auto policy_acc = policy_cpu.accessor<float, 2>();

  // Copy the Torch tensor output to the appropriate structure.
  std::vector<InferenceOutputs> output;
  output.reserve(inference_batch_size);
  for (int batch = 0; batch < inference_batch_size; ++batch) {
    double value = static_cast<double>(value_acc[batch][0]);

    ActionsAndProbs state_policy;
    state_policy.reserve(inputs[batch].legal_actions.size());
    for (Action action : inputs[batch].legal_actions) {
      state_policy.push_back({action, policy_acc[batch][action]});
    }

    output.push_back({value, state_policy});
  }

  return output;
}

VPNetModel::LossInfo VPNetModel::Learn(const std::vector<TrainInputs>& inputs) {
  return Losses(inputs, /*learn=*/true);
}

VPNetModel::LossInfo VPNetModel::Loss(const std::vector<TrainInputs>& inputs) {
  return Losses(inputs, /*learn=*/false);
}

// Our change: Learn's code, which can also stop before the learning step.
VPNetModel::LossInfo VPNetModel::Losses(const std::vector<TrainInputs>& inputs,
                                        bool learn) {
  int training_batch_size = inputs.size();

  std::vector<float> raw_train_inputs(training_batch_size * flat_input_size_);
  std::vector<uint8_t> raw_legal_mask(training_batch_size * num_actions_, 0);
  std::vector<float> raw_policy_targets(training_batch_size * num_actions_, 0);
  std::vector<float> raw_value_targets(training_batch_size);

  for (int batch = 0; batch < training_batch_size; ++batch) {
    // Our change (the input planes): the network's input, not only the observation —
    // computed here, after any symmetry the caller applied to the observation.
    ModelInput(inputs[batch].observations,
               raw_train_inputs.data() + batch * flat_input_size_);
    for (Action action : inputs[batch].legal_actions) {
      raw_legal_mask[num_actions_ * batch + action] = 1;
    }
    for (const auto &[action, probability] : inputs[batch].policy) {
      raw_policy_targets[num_actions_ * batch + action] = probability;
    }
    raw_value_targets[batch] = inputs[batch].value;
  }

  // Torch tensors by default use a dense, row-aligned memory layout.
  //   - Their default data type is a 32-bit float
  //   - Use the byte data type for boolean
  // Clone first to take ownership from the raw blob, then move to device.
  torch::Tensor torch_train_inputs =
      torch::from_blob(raw_train_inputs.data(),
                       {training_batch_size, flat_input_size_})
          .clone()
          .to(torch_device_);
  torch::Tensor torch_train_legal_mask =
      torch::from_blob(raw_legal_mask.data(),
                       {training_batch_size, num_actions_},
                       torch::TensorOptions().dtype(torch::kByte))
          .clone()
          .to(torch_device_);
  torch::Tensor torch_policy_targets =
      torch::from_blob(raw_policy_targets.data(),
                       {training_batch_size, num_actions_})
          .clone()
          .to(torch_device_);
  torch::Tensor torch_value_targets =
      torch::from_blob(raw_value_targets.data(), {training_batch_size, 1})
          .clone()
          .to(torch_device_);

  // Only the losses, as the network plays (Inference): no learning step.
  if (!learn) {
    model_->eval();
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> torch_outputs =
        model_->losses(torch_train_inputs, torch_train_legal_mask,
                       torch_policy_targets, torch_value_targets);
    LossInfo info(torch_outputs[0].item<float>(), torch_outputs[1].item<float>(),
                  torch_outputs[2].item<float>());
    AddLossesBySide(inputs, torch_outputs[3], torch_outputs[4], &info);
    return info;
  }

  // Run a training step and get the losses.
  model_->train();
  for (auto& module : frozen_modules_) module->eval();  // Our change: Freeze.
  model_->zero_grad();

  std::vector<torch::Tensor> torch_outputs =
      model_->losses(torch_train_inputs, torch_train_legal_mask,
                     torch_policy_targets, torch_value_targets);

  torch::Tensor total_loss =
      torch_outputs[0] + torch_outputs[1] + torch_outputs[2];

  total_loss.backward();

  model_optimizer_.step();

  LossInfo info(torch_outputs[0].item<float>(), torch_outputs[1].item<float>(),
                torch_outputs[2].item<float>());
  AddLossesBySide(inputs, torch_outputs[3], torch_outputs[4], &info);
  return info;
}

void VPNetModel::AddLossesBySide(const std::vector<TrainInputs>& inputs,
                                 torch::Tensor policy_each, torch::Tensor value_each,
                                 LossInfo* info) const {
  if (side_to_move_index_ < 0) return;
  policy_each = policy_each.to(torch::kCPU).contiguous();
  value_each = value_each.to(torch::kCPU).contiguous();
  const float* policy = policy_each.data_ptr<float>();
  const float* value = value_each.data_ptr<float>();
  for (int i = 0; i < inputs.size(); ++i) {
    const int side = inputs[i].observations[side_to_move_index_] > 0.5 ? 1 : 0;
    info->AddPosition(side, policy[i], value[i]);
  }
}

VPNetModel::TrainInputs SymmetricTrainInputs(const VPNetModel::TrainInputs& inputs,
                                             int symmetry) {
  VPNetModel::TrainInputs transformed;
  for (Action action : inputs.legal_actions) {
    transformed.legal_actions.push_back(thud::SymmetricAction(action, symmetry));
  }
  std::sort(transformed.legal_actions.begin(), transformed.legal_actions.end());
  transformed.observations =
      thud::SymmetricObservation(inputs.observations, symmetry);
  for (const auto& [action, probability] : inputs.policy) {
    transformed.policy.push_back(
        {thud::SymmetricAction(action, symmetry), probability});
  }
  std::sort(transformed.policy.begin(), transformed.policy.end());
  transformed.value = inputs.value;
  return transformed;
}

}  // namespace torch_az
}  // namespace thud_az
}  // namespace open_spiel
