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

// Tests the inference precision (VPNetModel::SetInferencePrecision, 2026-10-19; thud/PLAN.md,
// mixed precision) on networks built here (the convolutional head with the input planes,
// as our runs, width 16, depth 2; random weights and batch norms, a peaked prior), on 120
// positions of random games, one torch thread (so that equal inputs give equal results):
//
//   1. float32, set explicitly, infers exactly as by default (values and priors bit for
//      bit), through the legal policy and upstream's computation alike.
//   2. mixed and bfloat16 infer close to float32 — values within 0.01, priors within 0.01
//      (found 0.0012 and 0.0005 for mixed, 0.0017 and 0.0038 for bfloat16) — and through
//      either policy computation alike (within 1e-3). Controls: neither is bit for bit
//      float32 (the copy is used), they differ from each other, and another network (other
//      weights) is more than 0.01 away, so the tolerance tells networks apart.
//   3. The copy follows the weights: a mixed network after 3 learning steps infers exactly
//      as a fresh one that loads its saved checkpoint and then sets mixed, and one that
//      sets mixed first and then loads it (LoadCheckpoint refreshes the copy). Control: its
//      inference before learning differs.
//   4. Learning stays float32: a mixed network and a float32 one, from the same weights,
//      report the same losses, bit for bit, over the same 3 learning steps, and end with
//      the same weights, tensor by tensor.
//
//   thud/run_tests.sh precision_test   # builds and runs it
//   build-tests/bin/precision_test

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az;
using az::torch_az::VPNetModel;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// legal_policy_test's network: random weights (seeded), random batch norms, a peaked prior.
std::unique_ptr<VPNetModel> Network(const open_spiel::Game& game, const std::string& dir,
                                    unsigned seed) {
  torch::manual_seed(seed);
  auto model = std::make_unique<VPNetModel>(game, dir, "vpnet.pb", "/cpu:0");
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> scale(0.5, 1.5), shift(-0.3, 0.3), mean(-0.5, 0.5),
      variance(0.2, 3.0);
  torch::NoGradGuard no_grad;
  for (auto& [name, t] : model->NamedTensors()) {
    if (absl::StrContains(name, ".policy_conv_2.")) t.mul_(5);
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
  model->RefreshInference();
  return model;
}

std::vector<std::unique_ptr<open_spiel::State>> Positions(const open_spiel::Game& game, int count) {
  std::mt19937 rng(23);
  std::vector<std::unique_ptr<open_spiel::State>> out;
  while (static_cast<int>(out.size()) < count) {
    std::unique_ptr<open_spiel::State> s = game.NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 400 && static_cast<int>(out.size()) < count;
         ++ply) {
      if (ply % 5 == 1) out.push_back(s->Clone());
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      s->ApplyAction(legal[rng() % legal.size()]);
    }
  }
  return out;
}

std::vector<VPNetModel::InferenceOutputs> Infer(
    VPNetModel& model, bool legal, const std::vector<std::unique_ptr<open_spiel::State>>& ps) {
  model.SetLegalPolicy(legal);
  std::vector<VPNetModel::InferenceOutputs> out;
  for (size_t i = 0; i < ps.size(); i += 32) {
    std::vector<VPNetModel::InferenceInputs> in;
    for (size_t j = i; j < std::min(ps.size(), i + 32); ++j) {
      in.push_back({ps[j]->LegalActions(), ps[j]->ObservationTensor()});
    }
    for (auto& o : model.Inference(in)) out.push_back(std::move(o));
  }
  return out;
}

// The largest difference in values and in priors (same moves required: else infinite).
std::pair<double, double> Apart(const std::vector<VPNetModel::InferenceOutputs>& a,
                                const std::vector<VPNetModel::InferenceOutputs>& b) {
  double value = 0, prior = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    value = std::max(value, std::abs(a[i].value - b[i].value));
    if (a[i].policy.size() != b[i].policy.size()) return {INFINITY, INFINITY};
    for (size_t j = 0; j < a[i].policy.size(); ++j) {
      if (a[i].policy[j].first != b[i].policy[j].first) return {INFINITY, INFINITY};
      prior = std::max(prior, std::abs(a[i].policy[j].second - b[i].policy[j].second));
    }
  }
  return {value, prior};
}

std::vector<VPNetModel::TrainInputs> Batch(
    const std::vector<std::unique_ptr<open_spiel::State>>& ps, int first, int count) {
  std::vector<VPNetModel::TrainInputs> out;
  for (int i = first; i < first + count; ++i) {
    const auto legal = ps[i % ps.size()]->LegalActions();
    open_spiel::ActionsAndProbs policy;
    for (auto a : legal) policy.push_back({a, 1.0 / legal.size()});
    out.push_back({legal, ps[i % ps.size()]->ObservationTensor(), policy, (i % 9 - 4) / 4.0});
  }
  return out;
}

}  // namespace

int main() {
  torch::set_num_threads(1);
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const std::string dir =
      (std::filesystem::temp_directory_path() / ("precision_test_" + std::to_string(getpid())))
          .string();
  std::filesystem::create_directories(dir);
  SPIEL_CHECK_TRUE(az::torch_az::CreateGraphDef(*game, 0.001, 0.0001, dir, "vpnet.pb",
                                                "resnet_conv_policy", 16, 2,
                                                /*verbose=*/false, az::kNumFeaturePlanes));
  const auto positions = Positions(*game, 120);

  // 1. float32.
  {
    auto reference = Network(*game, dir, 1), explicit_ = Network(*game, dir, 1);
    explicit_->SetInferencePrecision("float32");
    bool same = true;
    for (bool legal : {true, false}) {
      const auto [v, p] = Apart(Infer(*reference, legal, positions),
                                Infer(*explicit_, legal, positions));
      same = same && v == 0 && p == 0;
    }
    Check(same, "1. float32 set explicitly infers bit for bit as by default (both policy "
                "computations)");
  }

  // 2. mixed and bfloat16.
  {
    auto f32 = Network(*game, dir, 1), mixed = Network(*game, dir, 1),
         b16 = Network(*game, dir, 1), other = Network(*game, dir, 9);
    mixed->SetInferencePrecision("mixed");
    b16->SetInferencePrecision("bfloat16");
    const auto ref = Infer(*f32, true, positions);
    const auto m_legal = Infer(*mixed, true, positions), m_upstream = Infer(*mixed, false, positions);
    const auto b_legal = Infer(*b16, true, positions), b_upstream = Infer(*b16, false, positions);
    const auto [mv, mp] = Apart(ref, m_legal);
    const auto [bv, bp] = Apart(ref, b_legal);
    const auto [mpv, mpp] = Apart(m_legal, m_upstream);
    const auto [bpv, bpp] = Apart(b_legal, b_upstream);
    const auto [xv, xp] = Apart(m_legal, b_legal);
    const auto [ov, op] = Apart(ref, Infer(*other, true, positions));
    Check(mv <= 0.01 && mp <= 0.01 && bv <= 0.01 && bp <= 0.01 &&
              std::max({mpv, mpp, bpv, bpp}) <= 1e-3,
          absl::StrFormat("2. mixed infers within %.2g (values) and %.2g (priors) of float32, "
                          "bfloat16 within %.2g and %.2g (0.01 allowed); the two policy "
                          "computations within %.2g (1e-3 allowed)",
                          mv, mp, bv, bp, std::max({mpv, mpp, bpv, bpp})));
    Check(std::max(mv, mp) > 0 && std::max(bv, bp) > 0 && std::max(xv, xp) > 0 &&
              std::max(ov, op) > 0.01,
          absl::StrFormat("2. controls: neither mixed nor bfloat16 is bit for bit float32, "
                          "they differ from each other, and another network is %.2g away "
                          "(more than 0.01 required)",
                          std::max(ov, op)));
  }

  // 3. The copy follows the weights.
  {
    auto learner = Network(*game, dir, 2);
    learner->SetInferencePrecision("mixed");
    const auto before = Infer(*learner, true, positions);
    for (int step = 0; step < 3; ++step) learner->Learn(Batch(positions, step * 40, 40));
    const std::string path = learner->SaveCheckpoint(7);
    const auto after = Infer(*learner, true, positions);
    VPNetModel loaded_then_set(*game, dir, "vpnet.pb", "/cpu:0");
    loaded_then_set.LoadCheckpoint(7);
    loaded_then_set.SetInferencePrecision("mixed");
    VPNetModel set_then_loaded(*game, dir, "vpnet.pb", "/cpu:0");
    set_then_loaded.SetInferencePrecision("mixed");
    set_then_loaded.LoadCheckpoint(7);
    const auto [v1, p1] = Apart(after, Infer(loaded_then_set, true, positions));
    const auto [v2, p2] = Apart(after, Infer(set_then_loaded, true, positions));
    const auto [cv, cp] = Apart(before, after);
    Check(v1 == 0 && p1 == 0 && v2 == 0 && p2 == 0 && std::max(cv, cp) > 1e-4,
          absl::StrFormat("3. after 3 learning steps the mixed copy infers bit for bit as a "
                          "fresh network's that loads the checkpoint, before or after setting "
                          "mixed; control: it moved by %.2g from before learning",
                          std::max(cv, cp)));
  }

  // 4. Learning stays float32.
  {
    auto f32 = Network(*game, dir, 3), mixed = Network(*game, dir, 3);
    mixed->SetInferencePrecision("mixed");
    bool same = true;
    for (int step = 0; step < 3; ++step) {
      const auto batch = Batch(positions, step * 40, 40);
      const auto a = f32->Learn(batch), b = mixed->Learn(batch);
      same = same && a.Policy() == b.Policy() && a.Value() == b.Value() && a.L2() == b.L2();
    }
    // The weights tensor by tensor (a saved file holds its own name, so files differ).
    bool weights = true;
    const auto a = f32->NamedTensors(), b = mixed->NamedTensors();
    for (size_t i = 0; i < a.size() && weights; ++i) {
      weights = a[i].first == b[i].first && torch::equal(a[i].second, b[i].second);
    }
    Check(same && weights,
          absl::StrFormat("4. a mixed network learns as a float32 one: the same losses bit for "
                          "bit over 3 steps (%s), the same weights tensor by tensor (%s)",
                          same ? "yes" : "no", weights ? "yes" : "no"));
  }

  std::filesystem::remove_all(dir);
  std::cout << (failures == 0 ? "ALL PASSED" : std::to_string(failures) + " FAILED")
            << std::endl;
  return failures == 0 ? 0 : 1;
}
