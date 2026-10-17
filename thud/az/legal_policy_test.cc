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

// Tests the legal policy (VPNetModel::SetLegalPolicy, on by default; thud/PLAN.md, the
// batching path): the priors from the logits of each position's legal moves alone,
// against upstream's computation — the mask and the softmax over all 19,800 distinct
// actions (SetLegalPolicy(false)). On networks built here with random weights, batch
// norms and a policy layer scaled by 5 (a peaked prior, as a trained network's: a mix-up
// of moves would show), for each model — the linear head ("resnet"), the convolutional
// head with the input planes ("resnet_conv_policy", as our runs), "mlp" — on 150
// positions of random games, both sides, in batches of 1 and of 32:
//
//   1. The values are the same, bit for bit.
//   2. The priors: the same moves in the same order, each probability within 1e-6 of
//      upstream's computation, summing to 1 within 1e-12.
//   Control: the same comparison against another network's priors (other weights) must
//   differ by more than 1e-3, so 2 can fail.
//   3. Reported, not required: searches on 20 positions (our MCTSBot, 100 simulations, no noise, through
//      the trainer's batching evaluator) with the legal policy on and off — on how many
//      positions the most visited move, and every root child's visits, agree (a prior
//      ~1e-7 apart can tip a near tie).
//
//   thud/run_tests.sh legal_policy_test   # builds and runs it
//   build-tests/bin/legal_policy_test

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az;
using az::torch_az::VPNetModel;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// A network with random weights (seeded), random batch norms and a peaked prior.
std::unique_ptr<VPNetModel> Network(const open_spiel::Game& game, const std::string& dir,
                                    unsigned seed) {
  torch::manual_seed(seed);
  auto model = std::make_unique<VPNetModel>(game, dir, "vpnet.pb", "/cpu:0");
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> scale(0.5, 1.5), shift(-0.3, 0.3), mean(-0.5, 0.5),
      variance(0.2, 3.0);
  torch::NoGradGuard no_grad;
  for (auto& [name, t] : model->NamedTensors()) {
    if (absl::StrContains(name, ".policy_linear") || absl::StrContains(name, ".policy_conv_2.")) {
      t.mul_(5);
    }
    if (!absl::StrContains(name, "batch_norm")) continue;
    std::uniform_real_distribution<float>* d =
        absl::EndsWith(name, ".weight")         ? &scale
        : absl::EndsWith(name, ".bias")         ? &shift
        : absl::EndsWith(name, ".running_mean") ? &mean
        : absl::EndsWith(name, ".running_var")  ? &variance
                                                : nullptr;
    if (d == nullptr) continue;
    std::vector<float> v(t.numel());
    for (float& x : v) x = (*d)(rng);
    t.copy_(torch::tensor(v).view(t.sizes()));
  }
  return model;
}

std::vector<std::unique_ptr<open_spiel::State>> Positions(const open_spiel::Game& game, int count) {
  std::mt19937 rng(17);
  std::vector<std::unique_ptr<open_spiel::State>> out;
  while (out.size() < count) {
    std::unique_ptr<open_spiel::State> s = game.NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 400 && out.size() < count; ++ply) {
      if (ply % 5 == 1) out.push_back(s->Clone());
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      s->ApplyAction(legal[rng() % legal.size()]);
    }
  }
  return out;
}

std::vector<VPNetModel::InferenceOutputs> Infer(
    VPNetModel& model, bool legal, int batch,
    const std::vector<std::unique_ptr<open_spiel::State>>& positions) {
  model.SetLegalPolicy(legal);
  std::vector<VPNetModel::InferenceOutputs> out;
  for (size_t i = 0; i < positions.size(); i += batch) {
    std::vector<VPNetModel::InferenceInputs> in;
    for (size_t j = i; j < std::min(positions.size(), i + batch); ++j) {
      in.push_back({positions[j]->LegalActions(), positions[j]->ObservationTensor()});
    }
    for (auto& o : model.Inference(in)) out.push_back(std::move(o));
  }
  return out;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const std::string base =
      (std::filesystem::temp_directory_path() / ("legal_policy_test_" + std::to_string(getpid())))
          .string();
  const auto positions = Positions(*game, 150);
  struct Kind {
    std::string nn_model;
    bool planes;
  };
  for (const Kind& kind : {Kind{"resnet", false}, Kind{"resnet_conv_policy", true},
                           Kind{"mlp", false}}) {
    const std::string dir = base + "/" + kind.nn_model;
    std::filesystem::create_directories(dir);
    SPIEL_CHECK_TRUE(az::torch_az::CreateGraphDef(
        *game, 0.001, 0.0001, dir, "vpnet.pb", kind.nn_model, 16, 2, /*verbose=*/false,
        kind.planes ? az::kNumFeaturePlanes : 0));
    auto model = Network(*game, dir, 1), other = Network(*game, dir, 2);
    for (int batch : {1, 32}) {
      const auto exact = Infer(*model, false, batch, positions);
      const auto fast = Infer(*model, true, batch, positions);
      bool values = true, moves = true;
      double worst = 0, worst_sum = 0;
      for (size_t i = 0; i < positions.size(); ++i) {
        values = values && exact[i].value == fast[i].value;
        moves = moves && exact[i].policy.size() == fast[i].policy.size();
        double sum = 0;
        for (size_t j = 0; moves && j < fast[i].policy.size(); ++j) {
          moves = exact[i].policy[j].first == fast[i].policy[j].first;
          worst = std::max(worst, std::abs(exact[i].policy[j].second - fast[i].policy[j].second));
          sum += fast[i].policy[j].second;
        }
        worst_sum = std::max(worst_sum, std::abs(sum - 1));
      }
      const std::string what = absl::StrFormat("%s, batches of %d", kind.nn_model, batch);
      Check(values, "1. " + what + ": the values bit for bit the same");
      Check(moves && worst <= 1e-6 && worst_sum <= 1e-12,
            absl::StrFormat("2. %s: the same moves in order, the priors within %.2g of "
                            "upstream's computation, summing to 1 within %.2g",
                            what, worst, worst_sum));
      const auto theirs = Infer(*other, false, batch, positions);
      double apart = 0;
      for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = 0; j < fast[i].policy.size(); ++j) {
          apart = std::max(apart, std::abs(theirs[i].policy[j].second - fast[i].policy[j].second));
        }
      }
      Check(apart > 1e-3, absl::StrFormat("2. control, %s: against another network's priors "
                                          "the comparison fails (%.2g apart)",
                                          what, apart));
    }

    // 3. Searches with the legal policy on and off.
    int same_move = 0, same_visits = 0, searched = 0;
    std::vector<std::vector<int>> visits[2];
    std::vector<open_spiel::Action> best[2];
    for (int on = 0; on < 2; ++on) {
      az::torch_az::DeviceManager devices;
      auto m = Network(*game, dir, 1);
      m->SetLegalPolicy(on == 1);
      devices.AddDevice(std::move(*m));
      auto eval = std::make_shared<az::torch_az::VPNetEvaluator>(&devices, 1, 0, 1 << 16, 1);
      az::MCTSBot bot(*game, eval, /*uct_c=*/2, /*max_simulations=*/100, /*max_memory_mb=*/100,
                      /*solve=*/false, /*seed=*/5, /*verbose=*/false,
                      az::ChildSelectionPolicy::PUCT, 0, 0, /*dont_return_chance_node=*/true);
      for (int i = 0; i < 20; ++i) {
        auto root = bot.MCTSearch(*positions[i]);
        std::vector<int> v;
        for (const auto& child : root->children) v.push_back(child.explore_count);
        visits[on].push_back(v);
        best[on].push_back(root->BestChild().action);
      }
    }
    for (int i = 0; i < 20; ++i) {
      ++searched;
      same_move += best[0][i] == best[1][i];
      same_visits += visits[0][i] == visits[1][i];
    }
    std::cout << absl::StrFormat(
                     "NOTE 3. %s: searches with the legal policy on and off — the same move "
                     "on %d of %d positions, every child's visits the same on %d",
                     kind.nn_model, same_move, searched, same_visits)
              << std::endl;
  }
  std::filesystem::remove_all(base);
  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
