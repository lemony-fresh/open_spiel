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
// open_spiel/algorithms/alpha_zero_torch/model.h at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#ifndef THUD_AZ_MODEL_H_
#define THUD_AZ_MODEL_H_

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace open_spiel {
namespace thud_az {
namespace torch_az {

struct ResInputBlockConfig {
  int input_channels;
  int input_height;
  int input_width;
  int filters;
  int kernel_size;
  int padding;
};

struct ResTorsoBlockConfig {
  int input_channels;
  int filters;
  int kernel_size;
  int padding;
  int layer;
};

struct ResOutputBlockConfig {
  int input_channels;
  int value_filters;
  int policy_filters;
  int kernel_size;
  int padding;
  int value_linear_in_features;
  int value_linear_out_features;
  int policy_linear_in_features;
  int policy_linear_out_features;
  int value_observation_size;
  int policy_observation_size;
};

// Information for the model. This should be enough for any type of model
// (residual, convultional, or MLP). It needs to be saved/loaded to/from
// a file so the input and output stream operators are overload.
struct ModelConfig {
  std::vector<int> observation_tensor_shape;
  int number_of_actions;
  int nn_depth;
  int nn_width;
  double learning_rate;
  double weight_decay;
  std::string nn_model = "resnet";
  // Our change, for nn_model "resnet_conv_policy": the number of policy
  // planes, and each action's entry in them, (plane * height + row) * width +
  // col. They are not saved with the rest: VPNetModel takes them from the game.
  int policy_planes = 0;
  std::vector<int64_t> policy_map;
};
std::istream& operator>>(std::istream& stream, ModelConfig& config);
std::ostream& operator<<(std::ostream& stream, const ModelConfig& config);

// A block of the residual model's network that handles the input. It consists
// of one convolutional layer (CONV) and one batch normalization (BN) layer, and
// the output is passed through a rectified linear unit function (RELU).
//
// Illustration:
//   [Input Tensor] --> CONV --> BN --> RELU
//
// There is only one input block per model.
class ResInputBlockImpl : public torch::nn::Module {
 public:
  ResInputBlockImpl(const ResInputBlockConfig& config);
  torch::Tensor forward(torch::Tensor x);

 private:
  int channels_;
  int height_;
  int width_;
  torch::nn::Conv2d conv_;
  torch::nn::BatchNorm2d batch_norm_;
};
TORCH_MODULE(ResInputBlock);

// A block of the residual model's network that makes up the 'torso'. It
// consists of two convolutional layers (CONV) and two batchnormalization layers
// (BN). The activation function is rectified linear unit (RELU). The input to
// the layer is added to the output before the final activation function.
//
// Illustration:
//   [Input Tensor] --> CONV --> BN --> RELU --> CONV --> BN --> + --> RELU
//          \___________________________________________________/
//
// Unlike the input and output blocks, one can specify how many of these torso
// blocks they want in their model.
class ResTorsoBlockImpl : public torch::nn::Module {
 public:
  ResTorsoBlockImpl(const ResTorsoBlockConfig& config, int layer);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Conv2d conv1_;
  torch::nn::Conv2d conv2_;
  torch::nn::BatchNorm2d batch_norm1_;
  torch::nn::BatchNorm2d batch_norm2_;
};
TORCH_MODULE(ResTorsoBlock);

// A block of the residual model's network that creates the output. It consists
// of a value and policy head. The value head takes the input through one
// convoluational layer (CONV), one batch normalization layers (BN), and two
// linear layers (LIN). The output activation function is tanh (TANH), the
// rectified linear activation function (RELU) is within. The policy head
// consists of one convolutional layer, batch normalization layer, and linear
// layer. There is no softmax activation function in this layer. The softmax
// on the output is applied in the forward function of the residual model.
// This design was chosen because the loss function of the residual model
// requires the policy logits, not the policy distribution. By providing the
// policy logits as output, the residual model can either apply the softmax
// activation function, or calculate the loss using Torch's log softmax
// function.
//
// Illustration:
//                    --> CONV --> BN --> RELU --> LIN --> RELU --> LIN --> TANH
//   [Input Tensor] --
//                    --> CONV --> BN --> RELU --> LIN (no SOFTMAX here)
//
// There is only one output block per model.
class ResOutputBlockImpl : public torch::nn::Module {
 public:
  ResOutputBlockImpl(const ResOutputBlockConfig& config);
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask);

 private:
  torch::nn::Conv2d value_conv_;
  torch::nn::BatchNorm2d value_batch_norm_;
  torch::nn::Linear value_linear1_;
  torch::nn::Linear value_linear2_;
  int value_observation_size_;
  torch::nn::Conv2d policy_conv_;
  torch::nn::BatchNorm2d policy_batch_norm_;
  torch::nn::Linear policy_linear_;
  int policy_observation_size_;
};
TORCH_MODULE(ResOutputBlock);

// Our change: an output block whose policy head is convolutional, built as
// AlphaZero's for chess and shogi ("an additional rectified, batch-normalized
// convolutional layer, followed by a final convolution of 73 filters",
// Silver et al., Science 2018, supplementary materials) and Leela Chess Zero's
// (lczero-training, tf/tfprocess.py: two 3x3 convolutions, the second with a
// bias, then a fixed map from its 80 planes to its 1,858 moves). The value head
// is the residual model's.
//
// Illustration:
//                    --> CONV --> BN --> RELU --> LIN --> RELU --> LIN --> TANH
//   [Input Tensor] --
//                    --> CONV --> BN --> RELU --> CONV --> MAP
//
// The policy convolutions are 3x3; the last one gives policy_planes planes
// and MAP takes each action's logit from its own entry in them
// (ModelConfig::policy_map). There is no weight per action: for Thud, at
// width 64, the head has ~0.1M weights where the linear layer has ~8.9M.
class ResConvPolicyOutputBlockImpl : public torch::nn::Module {
 public:
  ResConvPolicyOutputBlockImpl(const ResOutputBlockConfig& config,
                               int policy_planes,
                               const std::vector<int64_t>& policy_map);
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask);
  // The planes the policy logits are taken from, for checks.
  torch::Tensor PolicyPlanes(torch::Tensor x);

 private:
  torch::nn::Conv2d value_conv_;
  torch::nn::BatchNorm2d value_batch_norm_;
  torch::nn::Linear value_linear1_;
  torch::nn::Linear value_linear2_;
  int value_observation_size_;
  torch::nn::Conv2d policy_conv1_;
  torch::nn::BatchNorm2d policy_batch_norm_;
  torch::nn::Conv2d policy_conv2_;
  // Not a registered buffer, so checkpoints hold only what is learned.
  torch::Tensor policy_map_;
};
TORCH_MODULE(ResConvPolicyOutputBlock);

// A dense block with ReLU activation.
class MLPBlockImpl : public torch::nn::Module {
 public:
  MLPBlockImpl(const int in_features, const int out_features);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Linear linear_;
};
TORCH_MODULE(MLPBlock);

class MLPOutputBlockImpl : public torch::nn::Module {
 public:
  MLPOutputBlockImpl(const int nn_width, const int policy_linear_out_features);
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask);

 private:
  torch::nn::Linear value_linear1_;
  torch::nn::Linear value_linear2_;
  torch::nn::Linear policy_linear1_;
  torch::nn::Linear policy_linear2_;
};
TORCH_MODULE(MLPOutputBlock);

// The model class that interacts with the VPNet. The ResInputBlock,
// ResTorsoBlock, and ResOutputBlock are not to be used by the VPNet directly.
class ModelImpl : public torch::nn::Module {
 public:
  ModelImpl(const ModelConfig& config, const std::string& device);
  std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor mask);
  std::vector<torch::Tensor> losses(torch::Tensor inputs, torch::Tensor masks,
                                    torch::Tensor policy_targets,
                                    torch::Tensor value_targets);

 private:
  std::vector<torch::Tensor> forward_(torch::Tensor x, torch::Tensor mask);
  torch::nn::ModuleList layers_;
  torch::Device device_;
  int num_torso_blocks_;
  double weight_decay_;
  std::string nn_model_;
};
TORCH_MODULE(Model);

}  // namespace torch_az
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_MODEL_H_
