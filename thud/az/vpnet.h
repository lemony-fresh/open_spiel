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
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
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
  // under torch::NoGradGuard, changes the network) — for tests and diagnostics.
  torch::Tensor Parameter(const std::string& name);
  // Our change (thud/PLAN.md, the GPU readiness checklist: az_grow, 2026-10-15): every
  // parameter and buffer by name (the batch norms' running statistics included), sharing
  // their storage — writing to them, under torch::NoGradGuard, changes the network.
  std::vector<std::pair<std::string, torch::Tensor>> NamedTensors();
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
  // Our change (thud/PLAN.md, the GPU readiness checklist: a learning-rate schedule):
  // the optimizer's learning rate set for the next steps, and the rate the network's
  // file (vpnet.pb) gives — the base a schedule multiplies, whatever a loaded optimizer
  // holds.
  void SetLearningRate(double learning_rate);
  double BaseLearningRate() const { return model_config_.learning_rate; }
  // Our change (the value-loss weight; thud/DESIGN_dwarf_signal.md, E1): a learning
  // step minimises policy + weight x value + L2 (upstream: weight 1). The losses
  // reported (LossInfo) stay unweighted, comparable across weights.
  void SetValueLossWeight(double weight) { value_loss_weight_ = weight; }
  double ValueLossWeight() const { return value_loss_weight_; }
  // Our change (the same; the value-loss weight's sweep, 2026-10-17): the sizes (L2
  // norms) of the gradients of the policy loss, the value loss (unweighted) and the L2
  // term with respect to the parameters whose names pass `which` (the trunk: neither
  // "policy" nor "value" in the name), and the cosine between the first two — computed
  // as a learning step computes them (training mode, the batch's own statistics), but
  // with no step: the parameters and the batch norms' running statistics stay as they
  // were. Frozen parameters are left out (none left: every size 0).
  // Our change (the same, 2026-10-17): with true, a learning step keeps every batch norm
  // in evaluation mode — normalising by its running statistics, which then stay as they
  // are, instead of the batch's own — while their scales and shifts still learn. This is
  // how the trainer has in effect learnt on a single device: its inference threads call
  // Inference, which sets the shared network to evaluation mode, while the learner's step
  // runs (upstream's DeviceManager does not keep them apart; found 2026-10-17,
  // thud/DESIGN_dwarf_signal.md, E1). Off by default (upstream's training mode).
  void SetBatchNormsInEvaluation(bool on) { batch_norms_in_evaluation_ = on; }
  bool BatchNormsInEvaluation() const { return batch_norms_in_evaluation_; }

  // Our change (the race's fix, 2026-10-17; thud/DESIGN_dwarf_signal.md §10, E1's offline
  // sweep): the network is shared safely between threads. Inference, Activations, Loss
  // and SaveCheckpoint read it under a shared lock, side by side; Learn, Gradients,
  // LossInTrainingMode, LoadCheckpoint, LoadWeights, CopyParametersFrom and
  // ConvertBatchNorms change it under an exclusive one, and put it back in evaluation
  // mode before releasing it — so it is in evaluation mode whenever it is read, and the
  // readers no longer set the mode (upstream's Inference sets it, and its Learn sets
  // training mode, on one module that the trainer's inference threads share with the
  // learner: their calls switched the learner's steps to evaluation mode). Every learning
  // step then checks that each batch norm in training mode counted exactly its one batch
  // (num_batches_tracked) and every other none, and stops the program if not.
  //
  // The batch norms' counts of batches seen in training mode, in module order; and their
  // momenta (-1 for none: a cumulative average), for tests.
  std::vector<int64_t> BatchNormCounts();
  std::vector<double> BatchNormMomenta();
  // The check's comparison: "" when each batch norm `training` counted one batch from
  // `before` to `after` and each other none; otherwise the layers that did not, by name.
  static std::string BatchNormCountsDiffer(const std::vector<int64_t>& before,
                                           const std::vector<int64_t>& after,
                                           const std::vector<bool>& training,
                                           const std::vector<std::string>& names);
  // The losses Learn would report, in training mode (the batch's own statistics), with
  // no learning step: the parameters and the running statistics stay as they were.
  LossInfo LossInTrainingMode(const std::vector<TrainInputs>& inputs);

  // Our change (the same): each batch norm made to normalise by the true statistics of
  // what reaches it, the function unchanged in evaluation mode — Net2Net's way of
  // setting a batch norm from activation statistics estimated on training data. For one
  // batch norm at a time, the others in evaluation mode: its input's mean and variance
  // over `batches` (a cumulative average of the batches' own); then its running
  // statistics set to them and its scale and shift adjusted so that its output stays the
  // same (w' = w s'/s, b' = b + w (m' - m)/s, s = sqrt(var + eps)). Afterwards training
  // mode, which normalises by each batch's statistics, computes nearly what evaluation
  // mode does. Its count of batches and its momentum stay as they were. Returns per
  // batch norm the largest shift of a channel's mean in units of its old deviation, and
  // the smallest and largest ratio of new to old deviation.
  struct BatchNormConversion {
    std::string name;
    double largest_mean_shift = 0, smallest_ratio = 0, largest_ratio = 0;
  };
  std::vector<BatchNormConversion> ConvertBatchNorms(
      const std::vector<std::vector<TrainInputs>>& batches);
  struct GradientSizes {
    double policy = 0, value = 0, l2 = 0, cosine = 0;
  };
  GradientSizes Gradients(const std::vector<TrainInputs>& inputs,
                          const std::function<bool(const std::string&)>& which);
  // Our change (the legal policy; thud/PLAN.md, the batching path): Inference's priors
  // from the logits of each position's legal moves alone, normalised in double — not
  // upstream's mask and softmax over every distinct action (19,800 in Thud, of which a
  // position has 20 to 400 legal), whose illegal entries come out 0. The same
  // distribution up to rounding (~1e-7: thud/az/legal_policy_test.cc); the values are
  // bit for bit the same. On by default; false is upstream's computation, bit for bit
  // (identity_test compares with it).
  void SetLegalPolicy(bool on) { legal_policy_ = on; }
  bool LegalPolicy() const { return legal_policy_; }

  std::string SaveCheckpoint(int step);
  void LoadCheckpoint(int step);
  void LoadCheckpoint(const std::string& path);

  std::string Device() const { return device_; }

 private:
  // Our change: with `gradients_of` (learn true), Gradients' computation instead of the
  // learning step, into `sizes`.
  LossInfo Losses(const std::vector<TrainInputs>& inputs, bool learn,
                  const std::function<bool(const std::string&)>* gradients_of = nullptr,
                  GradientSizes* sizes = nullptr);
  // Our change: Inference with the legal policy (SetLegalPolicy).
  std::vector<InferenceOutputs> InferenceAtLegalMoves(const std::vector<InferenceInputs>& inputs);
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
  bool batch_norms_in_evaluation_ = false;  // Our change: SetBatchNormsInEvaluation.
  // Our change (the race's fix): shared for reading the network, exclusive for changing
  // it. A pointer, so that the model stays movable.
  std::unique_ptr<absl::Mutex> lock_ = std::make_unique<absl::Mutex>();
  // Our change: the network's input and legal-move mask for `inputs`, on the device.
  std::pair<torch::Tensor, torch::Tensor> InputsAndMasks(
      const std::vector<TrainInputs>& inputs);
  double value_loss_weight_ = 1;  // Our change: SetValueLossWeight.
  bool legal_policy_ = true;  // Our change: SetLegalPolicy.
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
