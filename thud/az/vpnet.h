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
// open_spiel/algorithms/alpha_zero_torch/vpnet.h at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#ifndef THUD_AZ_VPNET_H_
#define THUD_AZ_VPNET_H_

#include <torch/torch.h>

#include <nop/structure.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "thud/az/model.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace thud_az {
namespace torch_az {

// To avoid having to change function calls and the flow of the AlphaZero setup,
// this function is still called, but rather than create a TensorFlow graph
// definition, it simply creates a struct that the Libtorch model can use to
// initialize from. This struct is then saved and then loaded again when needed.
// Our change (thud/PLAN.md, the input planes): extra_input_planes more input planes than
// the game's observation has — for Thud, thud_az::kNumFeaturePlanes, the planes from the
// rules, which VPNetModel then computes from each observation (thud/az/thud_features.h).
bool CreateGraphDef(const Game& game, double learning_rate, double weight_decay,
                    const std::string& path, const std::string& filename,
                    std::string nn_model, int nn_width, int nn_depth,
                    bool verbose = false, int extra_input_planes = 0);

// Our change: `config` with the convolutional policy head's layout of the
// actions (ModelConfig::policy_map) filled in for `game`, if it has that head.
ModelConfig WithPolicyMap(const Game& game, ModelConfig config);

class VPNetModel {
 public:
  // A class to handle the network's loss.
  class LossInfo {
   public:
    LossInfo() {}
    LossInfo(double policy, double value, double l2)
        : policy_(policy), value_(value), l2_(l2), batches_(1) {}

    // Merge another LossInfo into this one.
    LossInfo& operator+=(const LossInfo& other) {
      policy_ += other.policy_;
      value_ += other.value_;
      l2_ += other.l2_;
      batches_ += other.batches_;
      for (int side = 0; side < 2; ++side) {
        side_policy_[side] += other.side_policy_[side];
        side_value_[side] += other.side_value_[side];
        side_positions_[side] += other.side_positions_[side];
      }
      return *this;
    }

    // Return the average losses over all merged into this one.
    double Policy() const { return policy_ / batches_; }
    double Value() const { return value_ / batches_; }
    double L2() const { return l2_ / batches_; }
    double Total() const { return Policy() + Value() + L2(); }

    // Our change (thud/PLAN.md, Instrumentation): the losses per side to move,
    // averaged over positions (with batches of equal size, the two sides'
    // averages weighted by their positions give Policy() and Value()).
    void AddPosition(int side, double policy, double value) {
      side_policy_[side] += policy;
      side_value_[side] += value;
      side_positions_[side] += 1;
    }
    int Positions(int side) const { return side_positions_[side]; }
    double Policy(int side) const {
      return side_policy_[side] / side_positions_[side];
    }
    double Value(int side) const { return side_value_[side] / side_positions_[side]; }

   private:
    double policy_ = 0;
    double value_ = 0;
    double l2_ = 0;
    int batches_ = 0;
    double side_policy_[2] = {0, 0};
    double side_value_[2] = {0, 0};
    int side_positions_[2] = {0, 0};
  };

  // A struct to handle the inputs for inference.
  struct InferenceInputs {
    std::vector<Action> legal_actions;
    std::vector<float> observations;

    bool operator==(const InferenceInputs& other) const {
      return legal_actions == other.legal_actions &&
             observations == other.observations;
    }

    template <typename H>
    friend H AbslHashValue(H h, const InferenceInputs& in) {
      return H::combine(std::move(h), in.legal_actions, in.observations);
    }
  };

  // A struct to hold the outputs of the inference (value and policy).
  struct InferenceOutputs {
    double value;
    ActionsAndProbs policy;
  };

  // A struct to hold the inputs for training.
  struct TrainInputs {
    std::vector<Action> legal_actions;
    std::vector<float> observations;
    ActionsAndProbs policy;
    double value;

    NOP_STRUCTURE(TrainInputs, legal_actions, observations, policy, value);
  };

  enum CheckpointStep {
    kMostRecentCheckpointStep = -1,
    kInvalidCheckpointStep = -2
  };

  VPNetModel(const Game &game, const std::string &path,
             const std::string &file_name,
             const std::string &device = "/cpu:0");

  // Move only, not copyable.
  VPNetModel(VPNetModel&& other) = default;
  VPNetModel& operator=(VPNetModel&& other) = default;
  VPNetModel(const VPNetModel&) = delete;
  VPNetModel& operator=(const VPNetModel&) = delete;

  // Inference: Get both at the same time.
  std::vector<InferenceOutputs> Inference(
      const std::vector<InferenceInputs>& inputs);

  // Training: do one (batch) step of neural net training
  LossInfo Learn(const std::vector<TrainInputs>& inputs);
  // Our change: the losses Learn would report, with the network as it plays
  // (Inference's mode), and no learning step.
  LossInfo Loss(const std::vector<TrainInputs>& inputs);

  // Our change (thud/PLAN.md, the feature probe): ModelImpl::Activations on the
  // CPU for these positions — the torso's output, the policy head's planes, the
  // values and the policies — with gradients off, in evaluation mode.
  std::vector<torch::Tensor> Activations(const std::vector<InferenceInputs>& inputs);

  // Our change (thud/PLAN.md, a tool to restructure a network): copies into this
  // network every parameter and buffer of `other` whose name passes `include` and
  // whose shape matches here; returns the names copied.
  // With widen_inputs, a tensor that has more input channels here (its dimension 1)
  // than in `other` and is otherwise alike — the first convolution, when input planes
  // are added — gets `other`'s channels first and zeros in the rest, so that this
  // network computes what `other` does (net surgery; Net2Net's function-preserving
  // start).
  std::vector<std::string> CopyParametersFrom(
      VPNetModel& other, const std::function<bool(const std::string&)>& include,
      bool widen_inputs = false);
  // Our change (the same): a parameter by name, sharing its storage (writing to it,
  // under torch::NoGradGuard, changes the network) — for checks and diagnostics.
  torch::Tensor Parameter(const std::string& name);
  // Our change (the same): whether the network takes the input planes from the rules.
  bool TakesInputPlanes() const { return input_planes_; }
  // Our change (the same): parameters whose names pass `which` get no gradient, and
  // the leaf modules whose names pass it stay in evaluation mode while learning (a
  // frozen batch norm neither uses batch statistics nor updates its running ones).
  // Returns the parameters frozen.
  std::vector<std::string> Freeze(const std::function<bool(const std::string&)>& which);

  // Our change (thud/PLAN.md, run L): the network's weights alone from a checkpoint
  // (PATH.pt), the optimizer left as it is; and the optimizer's learning rate.
  void LoadWeights(const std::string& path);
  double LearningRate();

  std::string SaveCheckpoint(int step);
  void LoadCheckpoint(int step);
  void LoadCheckpoint(const std::string& path);

  std::string Device() const { return device_; }

 private:
  LossInfo Losses(const std::vector<TrainInputs>& inputs, bool learn);
  // Our change: writes the network's input for `observation` to out[0, flat_input_size_).
  void ModelInput(const std::vector<float>& observation, float* out) const;
  // Our change: adds each position's losses to `info` by side to move, if the
  // game says which side moves in its observation (Thud: kTrollsToMovePlane).
  void AddLossesBySide(const std::vector<TrainInputs>& inputs,
                       torch::Tensor policy_each, torch::Tensor value_each,
                       LossInfo* info) const;

  std::string device_;
  std::string path_;

  // Store the full model metagraph file
  // for writing python compatible checkpoints.
  std::string model_meta_graph_contents_;

  // Our change: the network's input for one position, as many floats as the model's
  // input shape — the observation, then, if the model takes them, the input planes
  // from the rules computed from it (thud/az/thud_features.h); the observation's own
  // size; and whether the planes are added.
  int flat_input_size_;
  int observation_size_;
  bool input_planes_ = false;
  int num_actions_;

  // NOTE:
  // The member model_ takes an already initialized model_config_,
  // and model_optimizer_ takes an already initialized model_
  // parameters and model_config_ learning rate. Therefore, keep the
  // members' (model_config_, model_, model_optimizer_) declaration in
  // the order shown below so the member initialization list works.
  ModelConfig model_config_;
  Model model_;
  torch::optim::Adam model_optimizer_;
  torch::Device torch_device_;
  // Our change: where the observation says which side moves (Thud: the first
  // entry of kTrollsToMovePlane, 1 when the trolls move), or -1.
  int side_to_move_index_;
  // Our change (Freeze): leaf modules kept in evaluation mode while learning.
  std::vector<std::shared_ptr<torch::nn::Module>> frozen_modules_;
};

// Our change, for training on randomly turned and mirrored positions: `inputs` as
// they look after one of Thud's 8 board symmetries (thud::SymmetricObservation and
// thud::SymmetricAction; symmetry 0 changes nothing). The value stays: a symmetry
// changes neither the side to move nor the outcome.
VPNetModel::TrainInputs SymmetricTrainInputs(const VPNetModel::TrainInputs& inputs,
                                             int symmetry);

}  // namespace torch_az
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_VPNET_H_
