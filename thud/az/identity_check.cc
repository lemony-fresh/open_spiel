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

// Does our copy of OpenSpiel's C++ AlphaZero (thud/az/, namespace open_spiel::thud_az)
// behave exactly like upstream's (thud/PLAN.md Phase 6, roadmap stage 2)? For the same
// network, positions, settings and seeds it compares:
//
//   1. network outputs: upstream's VPNetModel and ours, loaded from the same checkpoint,
//      on every position — values and priors must be equal;
//   2. searches: upstream's MCTSBot and ours (PUCT, the trainer's settings) on every
//      position, without root noise and with it — every root child's visits and total
//      value must be equal. Ours runs with UntriedMoveValue::kUpstream: its default
//      values untried moves differently (thud/az/mcts.h), which a control checks;
//   3. checkpoints: a checkpoint saved by our copy, loaded by upstream's code, must give
//      the same outputs;
//   4. training: one learning step on the same batch must leave both networks with the
//      same losses and outputs (single-threaded LibTorch is deterministic);
//   5. the evaluation games' search: UCT with random rollouts and the solver, as the
//      trainer's evaluators play MCTS, with equal seeds.
//
// And controls that must differ, so that the check cannot pass by comparing nothing: the
// search with another noise seed, the outputs of another network, the outputs before and
// after the learning step, and the rollout search with another seed.
//
//   thud/az/build.sh thud/az/identity_check.cc \
//     open_spiel/algorithms/alpha_zero_torch/{model,vpnet,vpevaluator}.cc
//   OMP_NUM_THREADS=1 build-shared/identity_check RUN_DIR STEP OTHER_STEP [positions=50]
//
// RUN_DIR is a trainer run with checkpoints STEP and OTHER_STEP. Exits non-zero on any
// difference in 1-5, or if a control shows none.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

namespace up = open_spiel::algorithms;
namespace ours = open_spiel::thud_az;
using open_spiel::Action;

// Every 3rd position of random games.
std::vector<std::unique_ptr<open_spiel::State>> Positions(const open_spiel::Game& game,
                                                          int count) {
  std::mt19937 rng(7);
  std::vector<std::unique_ptr<open_spiel::State>> positions;
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  for (int turn = 0; positions.size() < count; ++turn) {
    if (state->IsTerminal()) state = game.NewInitialState();
    if (turn % 3 == 0) positions.push_back(state->Clone());
    std::vector<Action> legal = state->LegalActions();
    state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
  }
  return positions;
}

// A search's root as comparable numbers: per child its action, visits and total value.
template <typename Node>
std::vector<double> Summary(const Node& root) {
  std::vector<double> out = {static_cast<double>(root.explore_count), root.total_reward};
  for (const auto& child : root.children) {
    out.push_back(child.action);
    out.push_back(child.explore_count);
    out.push_back(child.total_reward);
  }
  return out;
}

template <typename Outputs>
std::vector<double> Flat(const std::vector<Outputs>& outputs) {
  std::vector<double> out;
  for (const auto& o : outputs) {
    out.push_back(o.value);
    for (const auto& [action, probability] : o.policy) {
      out.push_back(action);
      out.push_back(probability);
    }
  }
  return out;
}

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " RUN_DIR STEP OTHER_STEP [positions=50]"
              << std::endl;
    return 1;
  }
  const std::string dir = argv[1];
  const int step = std::stoi(argv[2]), other_step = std::stoi(argv[3]);
  const int count = argc > 4 ? std::stoi(std::string(argv[4]).substr(10)) : 50;
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const auto positions = Positions(*game, count);

  // 1. Network outputs, and the control with another network.
  up::torch_az::VPNetModel up_model(*game, dir, "vpnet.pb", "/cpu:0");
  up_model.LoadCheckpoint(step);
  ours::torch_az::VPNetModel our_model(*game, dir, "vpnet.pb", "/cpu:0");
  our_model.LoadCheckpoint(step);
  up::torch_az::VPNetModel up_other(*game, dir, "vpnet.pb", "/cpu:0");
  up_other.LoadCheckpoint(other_step);
  std::vector<up::torch_az::VPNetModel::InferenceInputs> up_inputs;
  std::vector<ours::torch_az::VPNetModel::InferenceInputs> our_inputs;
  for (const auto& state : positions) {
    up_inputs.push_back({state->LegalActions(), state->ObservationTensor()});
    our_inputs.push_back({state->LegalActions(), state->ObservationTensor()});
  }
  const std::vector<double> up_out = Flat(up_model.Inference(up_inputs));
  Expect(up_out == Flat(our_model.Inference(our_inputs)),
         absl::StrFormat("network outputs equal on %d positions", count));
  Expect(up_out != Flat(up_other.Inference(up_inputs)),
         absl::StrFormat("control: step %d's outputs differ from step %d's", other_step,
                         step));

  // 3. A checkpoint saved by our copy, loaded by upstream's code.
  const std::string tmp = absl::StrCat(std::filesystem::temp_directory_path().string(),
                                       "/identity_check_", getpid());
  std::filesystem::create_directories(tmp);
  std::filesystem::copy_file(dir + "/vpnet.pb", tmp + "/vpnet.pb");
  ours::torch_az::VPNetModel our_saver(*game, tmp, "vpnet.pb", "/cpu:0");
  our_saver.LoadCheckpoint(absl::StrCat(dir, "/checkpoint-", step));
  our_saver.SaveCheckpoint(1);
  up::torch_az::VPNetModel up_loader(*game, tmp, "vpnet.pb", "/cpu:0");
  up_loader.LoadCheckpoint(1);
  Expect(up_out == Flat(up_loader.Inference(up_inputs)),
         "a checkpoint saved by our copy gives upstream's code the same outputs");
  std::filesystem::remove_all(tmp);

  // 4. Training: one learning step on the same batch must leave equal networks.
  {
    up::torch_az::VPNetModel up_learner(*game, dir, "vpnet.pb", "/cpu:0");
    up_learner.LoadCheckpoint(step);
    ours::torch_az::VPNetModel our_learner(*game, dir, "vpnet.pb", "/cpu:0");
    our_learner.LoadCheckpoint(step);
    std::vector<up::torch_az::VPNetModel::TrainInputs> up_batch;
    std::vector<ours::torch_az::VPNetModel::TrainInputs> our_batch;
    for (int i = 0; i < positions.size(); ++i) {
      std::vector<Action> legal = positions[i]->LegalActions();
      open_spiel::ActionsAndProbs policy;
      for (Action a : legal) policy.push_back({a, 1.0 / legal.size()});
      const double value = i % 2 ? 1.0 : -1.0;
      up_batch.push_back({legal, positions[i]->ObservationTensor(), policy, value});
      our_batch.push_back({legal, positions[i]->ObservationTensor(), policy, value});
    }
    const auto up_loss = up_learner.Learn(up_batch);
    const auto our_loss = our_learner.Learn(our_batch);
    Expect(up_loss.Total() == our_loss.Total(),
           absl::StrFormat("one learning step: equal losses (%.6f)", up_loss.Total()));
    const std::vector<double> learned = Flat(up_learner.Inference(up_inputs));
    Expect(learned == Flat(our_learner.Inference(our_inputs)),
           "one learning step: equal network outputs afterwards");
    Expect(learned != up_out, "control: the learning step changed the outputs");
  }

  // 2. Searches, as the trainer's actors run them (alpha_zero.cc, InitAZBot), without
  // and with root noise; each side with its own evaluator on its own model.
  up::torch_az::DeviceManager up_devices;
  up_devices.AddDevice(std::move(up_model));
  ours::torch_az::DeviceManager our_devices;
  our_devices.AddDevice(std::move(our_model));
  auto up_eval = std::make_shared<up::torch_az::VPNetEvaluator>(&up_devices, 1, 0, 1 << 18, 1);
  auto our_eval =
      std::make_shared<ours::torch_az::VPNetEvaluator>(&our_devices, 1, 0, 1 << 18, 1);
  for (double epsilon : {0.0, 0.25}) {
    const double alpha = epsilon > 0 ? 0.1 : 0;
    up::MCTSBot up_bot(*game, up_eval, 2, 100, 10, false, /*seed=*/0, false,
                       up::ChildSelectionPolicy::PUCT, alpha, epsilon, true);
    ours::MCTSBot our_bot(*game, our_eval, 2, 100, 10, false, /*seed=*/0, false,
                          ours::ChildSelectionPolicy::PUCT, alpha, epsilon, true, -1,
                          ours::UntriedMoveValue::kUpstream);
    int equal = 0;
    for (const auto& state : positions) {
      equal += Summary(*up_bot.MCTSearch(*state)) == Summary(*our_bot.MCTSearch(*state));
    }
    Expect(equal == count, absl::StrFormat("searches equal %s root noise: %d of %d",
                                           epsilon > 0 ? "with" : "without", equal, count));
    if (epsilon == 0) {  // Control: our default rule for untried moves must differ.
      ours::MCTSBot our_default(*game, our_eval, 2, 100, 10, false, /*seed=*/0, false,
                                ours::ChildSelectionPolicy::PUCT, alpha, epsilon, true);
      up::MCTSBot up_again(*game, up_eval, 2, 100, 10, false, /*seed=*/0, false,
                           up::ChildSelectionPolicy::PUCT, alpha, epsilon, true);
      int differ = 0;
      for (const auto& state : positions) {
        differ += Summary(*up_again.MCTSearch(*state)) !=
                  Summary(*our_default.MCTSearch(*state));
      }
      Expect(differ > 0, absl::StrFormat("control: our default rule for untried moves "
                                         "changes %d of %d searches", differ, count));
    }
    if (epsilon > 0) {  // Control: another seed must change the noisy searches.
      ours::MCTSBot other_seed(*game, our_eval, 2, 100, 10, false, /*seed=*/1, false,
                               ours::ChildSelectionPolicy::PUCT, alpha, epsilon, true,
                               -1, ours::UntriedMoveValue::kUpstream);
      up::MCTSBot up_again(*game, up_eval, 2, 100, 10, false, /*seed=*/0, false,
                           up::ChildSelectionPolicy::PUCT, alpha, epsilon, true);
      int differ = 0;
      for (const auto& state : positions) {
        differ += Summary(*up_again.MCTSearch(*state)) !=
                  Summary(*other_seed.MCTSearch(*state));
      }
      Expect(differ > 0, absl::StrFormat("control: another noise seed changes %d of %d "
                                         "searches", differ, count));
    }
  }
  // 5. The evaluation games' search (alpha_zero.cc, evaluator): UCT with random
  // rollouts and the solver, equal seeds; control: another rollout seed.
  {
    std::vector<std::vector<double>> up_rollouts, our_rollouts, other_rollouts;
    up::MCTSBot up_bot(*game, std::make_shared<up::RandomRolloutEvaluator>(1, 3), 2, 100,
                       1000, /*solve=*/true, /*seed=*/3, false, up::ChildSelectionPolicy::UCT,
                       0, 0, true);
    ours::MCTSBot our_bot(*game, std::make_shared<ours::RandomRolloutEvaluator>(1, 3), 2,
                          100, 1000, /*solve=*/true, /*seed=*/3, false,
                          ours::ChildSelectionPolicy::UCT, 0, 0, true, -1,
                          ours::UntriedMoveValue::kUpstream);
    ours::MCTSBot other_bot(*game, std::make_shared<ours::RandomRolloutEvaluator>(1, 4), 2,
                            100, 1000, /*solve=*/true, /*seed=*/4, false,
                            ours::ChildSelectionPolicy::UCT, 0, 0, true, -1,
                            ours::UntriedMoveValue::kUpstream);
    for (const auto& state : positions) {
      up_rollouts.push_back(Summary(*up_bot.MCTSearch(*state)));
      our_rollouts.push_back(Summary(*our_bot.MCTSearch(*state)));
      other_rollouts.push_back(Summary(*other_bot.MCTSearch(*state)));
    }
    int equal = 0, differ = 0;
    for (int i = 0; i < positions.size(); ++i) {
      equal += up_rollouts[i] == our_rollouts[i];
      differ += up_rollouts[i] != other_rollouts[i];
    }
    Expect(equal == count, absl::StrFormat("evaluation searches (UCT, random rollouts, "
                                           "solver) equal: %d of %d", equal, count));
    Expect(differ > 0, absl::StrFormat("control: another rollout seed changes %d of %d",
                                       differ, count));
  }
  std::cout << (failures ? "IDENTITY CHECK FAILED" : "identity check passed") << std::endl;
  return failures ? 1 : 0;
}
