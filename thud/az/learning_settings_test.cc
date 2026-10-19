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

// Tests the learner's settings added for the GPU phase (thud/PLAN.md, the GPU readiness
// checklist), on a small network built here (width 8, depth 1, no weight decay, so that a
// parameter without a gradient stays exactly as it is):
//
//   1. LearningRateFactor against values worked out by hand: a warm-up of 4 steps from
//      step 10 (0.25, 0.5, 0.75, 1), the schedule "20:0.3,40:0.1" (each factor from its
//      step on), both overlapping (a warm-up from 18: step 20 is 0.75 x 0.3); without
//      either, 1 everywhere and no schedule at all.
//   2. The config: the four fields survive ToJson / FromJson; a config.json from before
//      them gives the defaults (0, 1, "", 1), so older runs resume as they trained.
//   3. SetLearningRate: the optimizer takes the rate (LearningRate reads it back), the
//      network's own (BaseLearningRate, vpnet.pb's) stays; a learning step at rate 0
//      leaves every parameter exactly as it was (the control: at the network's rate they
//      move) — its outputs still move a little, through the batch norms' running
//      statistics, which a step in training mode updates at any rate. And a fresh model
//      loading a checkpoint saved at another rate takes the saved rate: LibTorch restores
//      an optimizer's options with its state (first shown here, 2026-10-15), so without a
//      schedule a run trains at its checkpoint's rate, not vpnet.pb's; the schedule sets
//      the rate every step from vpnet.pb's.
//   4. The value-loss weight, one learning step on the same batch from the same network:
//      at weight 0 every parameter of the value head alone stays exactly as it was, at
//      weight 1 they move (the control); the policy head's move at both; the losses
//      reported are the same at both (unweighted); weight 2 moves the shared trunk
//      differently from weight 1 (the weight reaches the trunk).
//   5. Saving under a temporary name: SaveCheckpoint and ReplayBuffer::SaveBuffer leave
//      no ".tmp" file, and what they wrote loads (the network's outputs, the buffer's
//      contents equal).
//   6. Gradients, the gradients' sizes (2026-10-17, the value-loss weight's sweep): each
//      loss's own — over the value head's parameters the policy loss's gradient is
//      exactly 0 and the value loss's not, over the policy head's the reverse, over the
//      trunk both non-zero; the L2 term's equals its definition, weight decay x the size
//      of the trunk's weights without the biases (a network with weight decay 0.01), and
//      over the biases alone 0; the value's unweighted (the same at weight 5); and no
//      step: every parameter and running statistic, so every output, exactly as before
//      — the control: a learning step at rate 0, which only the running statistics
//      change, moves the outputs; and over no parameter (a frozen trunk) every size 0,
//      not LibTorch's exception (found 2026-10-17 by trainer_test's frozen trunk).
//   7. SetBatchNormsInEvaluation: a learning step at rate 0 then leaves every output
//      exactly as it was (the running statistics untouched; the control is part 3's step,
//      which moves them), its reported losses equal Loss()'s on the same batch (the batch
//      norms normalised as the network plays; the control: without it they differ), and
//      at the network's rate the batch norms' scales still learn.
//
//   thud/run_tests.sh learning_settings_test   # builds and runs it
//   build-tests/bin/learning_settings_test

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
#include "thud/az/alpha_zero.h"
#include "thud/az/position_buffer.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az::torch_az;
using az::AlphaZeroConfig;
using az::VPNetModel;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

std::vector<VPNetModel::TrainInputs> Batch(const open_spiel::Game& game, std::mt19937* rng,
                                           int count) {
  std::vector<VPNetModel::TrainInputs> out;
  std::uniform_real_distribution<double> value(-1, 1);
  while (out.size() < count) {
    std::unique_ptr<open_spiel::State> s = game.NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 200 && out.size() < count; ++ply) {
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      if (ply % 5 == 2) {
        open_spiel::ActionsAndProbs policy;
        for (open_spiel::Action a : legal) policy.push_back({a, 1.0 / legal.size()});
        out.push_back({legal, s->ObservationTensor(), policy, value(*rng)});
      }
      s->ApplyAction(legal[(*rng)() % legal.size()]);
    }
  }
  return out;
}

std::vector<VPNetModel::InferenceInputs> AsInference(
    const std::vector<VPNetModel::TrainInputs>& batch) {
  std::vector<VPNetModel::InferenceInputs> out;
  for (const auto& b : batch) out.push_back({b.legal_actions, b.observations});
  return out;
}

double MaxDifference(VPNetModel& a, VPNetModel& b,
                     const std::vector<VPNetModel::InferenceInputs>& in) {
  const auto va = a.Inference(in), vb = b.Inference(in);
  double d = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    d = std::max(d, std::abs(va[i].value - vb[i].value));
    for (size_t j = 0; j < va[i].policy.size(); ++j) {
      d = std::max(d, std::abs(va[i].policy[j].second - vb[i].policy[j].second));
    }
  }
  return d;
}

// The largest change of the parameters whose names `which` accepts, from `before`.
double Moved(VPNetModel& model, const std::vector<std::pair<std::string, torch::Tensor>>& before,
             const std::function<bool(const std::string&)>& which) {
  torch::NoGradGuard no_grad;
  double d = 0;
  const auto now = model.NamedTensors();
  for (size_t i = 0; i < now.size(); ++i) {
    const std::string& name = now[i].first;
    if (absl::StrContains(name, "running_") || absl::StrContains(name, "num_batches")) continue;
    if (!which(name)) continue;
    d = std::max(d, (now[i].second - before[i].second).abs().max().item<double>());
  }
  return d;
}

std::vector<std::pair<std::string, torch::Tensor>> Snapshot(VPNetModel& model) {
  std::vector<std::pair<std::string, torch::Tensor>> out;
  for (auto& [name, t] : model.NamedTensors()) out.push_back({name, t.detach().clone()});
  return out;
}

bool IsValueHead(const std::string& n) { return absl::StrContains(n, "value_"); }
bool IsPolicyHead(const std::string& n) { return absl::StrContains(n, "policy_"); }
bool IsTrunk(const std::string& n) { return !IsValueHead(n) && !IsPolicyHead(n); }

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  torch::manual_seed(3);
  const std::string dir =
      (std::filesystem::temp_directory_path() /
       ("learning_settings_test_" + std::to_string(getpid())))
          .string();
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  // 1. The factor.
  {
    AlphaZeroConfig c;
    bool ok = !az::HasLearningRateSchedule(c);
    for (int s : {1, 10, 1000}) ok = ok && az::LearningRateFactor(c, s) == 1;
    Check(ok, "1. no warm-up and no schedule: no schedule, the factor 1 everywhere");
    c.lr_warmup_steps = 4;
    c.lr_warmup_start = 10;
    c.lr_schedule = "20:0.3,40:0.1";
    const std::vector<std::pair<int, double>> expected = {
        {9, 1},    {10, 0.25}, {11, 0.5}, {12, 0.75}, {13, 1},  {19, 1},
        {20, 0.3}, {39, 0.3},  {40, 0.1}, {1000, 0.1}};
    ok = az::HasLearningRateSchedule(c);
    std::string got;
    for (auto [s, f] : expected) {
      const double x = az::LearningRateFactor(c, s);
      ok = ok && std::abs(x - f) < 1e-15;
      got += absl::StrFormat(" %d:%g", s, x);
    }
    Check(ok, "1. a warm-up of 4 from step 10 and \"20:0.3,40:0.1\":" + got);
    c.lr_warmup_start = 18;
    Check(std::abs(az::LearningRateFactor(c, 19) - 0.5) < 1e-15 &&
              std::abs(az::LearningRateFactor(c, 20) - 0.75 * 0.3) < 1e-15,
          absl::StrFormat("1. overlapping, a warm-up from 18: step 19 %g, step 20 %g (0.75 x "
                          "0.3)",
                          az::LearningRateFactor(c, 19), az::LearningRateFactor(c, 20)));
  }

  // 2. The config.
  {
    AlphaZeroConfig c;
    c.game = "thud";
    c.lr_warmup_steps = 5;
    c.lr_warmup_start = 103;
    c.lr_schedule = "150:0.5";
    c.value_loss_weight = 2.5;
    open_spiel::json::Object j = c.ToJson();
    AlphaZeroConfig back;
    back.FromJson(j);
    Check(back.lr_warmup_steps == 5 && back.lr_warmup_start == 103 &&
              back.lr_schedule == "150:0.5" && back.value_loss_weight == 2.5,
          "2. the four fields survive ToJson / FromJson");
    for (const char* key : {"lr_warmup_steps", "lr_warmup_start", "lr_schedule",
                            "value_loss_weight"}) {
      j.erase(key);
    }
    AlphaZeroConfig old;
    old.FromJson(j);
    Check(old.lr_warmup_steps == 0 && old.lr_warmup_start == 1 && old.lr_schedule.empty() &&
              old.value_loss_weight == 1 && !az::HasLearningRateSchedule(old),
          "2. a config.json from before them gives the defaults (0, 1, \"\", 1)");
  }

  SPIEL_CHECK_TRUE(az::CreateGraphDef(*game, /*learning_rate=*/0.01, /*weight_decay=*/0,
                                      dir, "vpnet.pb", "resnet_conv_policy", 8, 1,
                                      /*verbose=*/false));
  std::mt19937 rng(5);
  const auto batch = Batch(*game, &rng, 64);
  const auto probe = AsInference(Batch(*game, &rng, 30));
  {  // The starting network, saved for every part below.
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    model.SaveCheckpoint(0);
  }
  auto fresh = [&]() {
    auto m = std::make_unique<VPNetModel>(*game, dir, "vpnet.pb", "/cpu:0");
    m->LoadCheckpoint(0);
    return m;
  };

  // 3. SetLearningRate.
  {
    auto a = fresh(), reference = fresh();
    a->SetLearningRate(0.002);
    Check(std::abs(a->LearningRate() - 0.002) < 1e-15 && a->BaseLearningRate() == 0.01,
          absl::StrFormat("3. SetLearningRate(0.002): the optimizer's rate %g, the network's "
                          "own still %g",
                          a->LearningRate(), a->BaseLearningRate()));
    a->SetLearningRate(0);
    const auto before = Snapshot(*a);
    a->Learn(batch);
    auto all = [](const std::string&) { return true; };
    const double p0 = Moved(*a, before, all), out0 = MaxDifference(*a, *reference, probe);
    a->SetLearningRate(a->BaseLearningRate());
    a->Learn(batch);
    const double p1 = Moved(*a, before, all);
    Check(p0 == 0 && p1 > 1e-4,
          absl::StrFormat("3. a step at rate 0 leaves every parameter as it was (%.2g; the "
                          "outputs move by %.2g through the batch norms' running statistics); "
                          "the control, a step at the network's rate, moves them (%.2g)",
                          p0, out0, p1));
    a->SetLearningRate(0.002);
    a->SaveCheckpoint(1);
    VPNetModel loaded(*game, dir, "vpnet.pb", "/cpu:0");
    loaded.LoadCheckpoint(1);
    Check(std::abs(loaded.LearningRate() - 0.002) < 1e-12 && loaded.BaseLearningRate() == 0.01,
          absl::StrFormat("3. a checkpoint saved at rate 0.002 loads at %g (LibTorch restores "
                          "the optimizer's options); vpnet.pb's rate stays %g",
                          loaded.LearningRate(), loaded.BaseLearningRate()));
  }

  // 4. The value-loss weight.
  {
    auto w0 = fresh(), w1 = fresh(), w2 = fresh();
    const auto before = Snapshot(*w0);
    w0->SetValueLossWeight(0);
    w2->SetValueLossWeight(2);
    const VPNetModel::LossInfo l0 = w0->Learn(batch), l1 = w1->Learn(batch);
    w2->Learn(batch);
    const double value0 = Moved(*w0, before, IsValueHead), value1 = Moved(*w1, before, IsValueHead);
    Check(value0 == 0 && value1 > 1e-5,
          absl::StrFormat("4. weight 0: the value head's own parameters exactly unchanged "
                          "(%.2g); the control, weight 1: they move (%.2g)",
                          value0, value1));
    const double policy0 = Moved(*w0, before, IsPolicyHead), policy1 = Moved(*w1, before, IsPolicyHead);
    Check(policy0 > 1e-5 && policy1 > 1e-5,
          absl::StrFormat("4. the policy head moves at both weights (%.2g, %.2g)", policy0,
                          policy1));
    Check(l0.Value() == l1.Value() && l0.Policy() == l1.Policy(),
          absl::StrFormat("4. the losses reported are unweighted: value %.6f at both, policy "
                          "%.6f",
                          l0.Value(), l0.Policy()));
    const auto after1 = Snapshot(*w1);
    const double trunk = Moved(*w2, after1, IsTrunk);
    Check(trunk > 1e-6,
          absl::StrFormat("4. weight 2 moves the shared trunk unlike weight 1 (by %.2g)", trunk));
  }

  // 5. Saving under a temporary name.
  {
    auto m = fresh();
    m->SaveCheckpoint(7);
    bool tmp = false;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
      tmp = tmp || absl::EndsWith(e.path().string(), ".tmp");
    }
    VPNetModel loaded(*game, dir, "vpnet.pb", "/cpu:0");
    loaded.LoadCheckpoint(7);
    const double d = MaxDifference(*m, loaded, probe);
    open_spiel::thud_az::PositionBuffer<VPNetModel::TrainInputs> buffer(100), back(100);
    for (VPNetModel::TrainInputs b : batch) buffer.Add(std::move(b));
    const std::string path = dir + "/replay_buffer.data";
    buffer.SaveBuffer(path);
    bool tmp_buffer = std::filesystem::exists(path + ".tmp");
    back.LoadBuffer(path);
    bool same = back.Size() == buffer.Size() && back.TotalAdded() == buffer.TotalAdded();
    std::mt19937 r1(9), r2(9);
    const auto s1 = buffer.Sample(&r1, 20), s2 = back.Sample(&r2, 20);
    for (int i = 0; i < 20 && same; ++i) {
      same = s1[i].legal_actions == s2[i].legal_actions &&
             s1[i].observations == s2[i].observations && s1[i].policy == s2[i].policy &&
             s1[i].value == s2[i].value;
    }
    Check(!tmp && d == 0 && !tmp_buffer && same,
          absl::StrFormat("5. SaveCheckpoint and SaveBuffer leave no .tmp file; what they "
                          "wrote loads (outputs equal: %.2g; the buffer's samples equal: %s)",
                          d, same ? "yes" : "no"));
  }

  // 6. Gradients.
  {
    auto m = fresh(), reference = fresh();
    auto everything = [](const std::string&) { return true; };
    const auto before = Snapshot(*m);
    const VPNetModel::GradientSizes value_head = m->Gradients(batch, IsValueHead);
    const VPNetModel::GradientSizes policy_head = m->Gradients(batch, IsPolicyHead);
    const VPNetModel::GradientSizes trunk = m->Gradients(batch, IsTrunk);
    Check(value_head.policy == 0 && value_head.value > 0 && policy_head.value == 0 &&
              policy_head.policy > 0 && trunk.policy > 0 && trunk.value > 0 &&
              std::abs(trunk.cosine) <= 1,
          absl::StrFormat("6. each loss's own gradient: value head policy %.2g, value %.2g; "
                          "policy head policy %.2g, value %.2g; trunk policy %.3g, value "
                          "%.3g, cosine %.3f",
                          value_head.policy, value_head.value, policy_head.policy,
                          policy_head.value, trunk.policy, trunk.value, trunk.cosine));
    const VPNetModel::GradientSizes none =
        m->Gradients(batch, [](const std::string&) { return false; });
    Check(none.policy == 0 && none.value == 0 && none.l2 == 0 && none.cosine == 0,
          "6. over no parameter (a frozen trunk) every size 0");
    m->SetValueLossWeight(5);
    const VPNetModel::GradientSizes weighted = m->Gradients(batch, IsTrunk);
    Check(weighted.value == trunk.value && weighted.policy == trunk.policy,
          absl::StrFormat("6. the value's gradient unweighted: %.6g at weight 5 and 1",
                          weighted.value));
    m->Gradients(batch, everything);
    double moved = 0;
    {
      torch::NoGradGuard no_grad;
      const auto now = m->NamedTensors();
      for (size_t i = 0; i < now.size(); ++i) {
        moved = std::max(moved, (now[i].second.to(torch::kDouble) -
                                 before[i].second.to(torch::kDouble))
                                    .abs()
                                    .max()
                                    .item<double>());
      }
    }
    const double outputs = MaxDifference(*m, *reference, probe);
    auto control = fresh();
    control->SetLearningRate(0);
    control->Learn(batch);
    const double control_outputs = MaxDifference(*control, *reference, probe);
    Check(moved == 0 && outputs == 0 && control_outputs > 0,
          absl::StrFormat("6. no step: every tensor (running statistics included) moved by "
                          "%.2g, the outputs by %.2g; the control, a step at rate 0, moves "
                          "the outputs by %.2g",
                          moved, outputs, control_outputs));

    const std::string decay_dir = dir + "/decay";
    std::filesystem::create_directories(decay_dir);
    SPIEL_CHECK_TRUE(az::CreateGraphDef(*game, /*learning_rate=*/0.01, /*weight_decay=*/0.01,
                                        decay_dir, "vpnet.pb", "resnet_conv_policy", 8, 1,
                                        /*verbose=*/false));
    VPNetModel decay(*game, decay_dir, "vpnet.pb", "/cpu:0");
    double weights = 0;
    {
      torch::NoGradGuard no_grad;
      for (auto& [name, t] : decay.NamedTensors()) {
        if (IsTrunk(name) && !absl::StrContains(name, "bias") &&
            !absl::StrContains(name, "running_") && !absl::StrContains(name, "num_batches")) {
          weights += t.to(torch::kDouble).square().sum().item<double>();
        }
      }
    }
    const double expected = 0.01 * std::sqrt(weights);
    const VPNetModel::GradientSizes l2 = decay.Gradients(batch, IsTrunk);
    auto biases = [](const std::string& n) { return absl::StrContains(n, "bias"); };
    const VPNetModel::GradientSizes l2_biases = decay.Gradients(batch, biases);
    Check(std::abs(l2.l2 - expected) <= 1e-5 * expected && l2_biases.l2 == 0 &&
              l2_biases.value > 0,
          absl::StrFormat("6. the L2 term's gradient over the trunk %.7g, its definition "
                          "0.01 x |weights| %.7g; over the biases %.2g (their value "
                          "gradient %.2g)",
                          l2.l2, expected, l2_biases.l2, l2_biases.value));
  }

  // 7. SetBatchNormsInEvaluation.
  {
    auto m = fresh(), reference = fresh(), control = fresh();
    m->SetBatchNormsInEvaluation(true);
    m->SetLearningRate(0);
    control->SetLearningRate(0);
    const VPNetModel::LossInfo plays = reference->Loss(batch);
    const VPNetModel::LossInfo learnt = m->Learn(batch);
    const VPNetModel::LossInfo control_learnt = control->Learn(batch);
    const double outputs = MaxDifference(*m, *reference, probe);
    const double same = std::max(std::abs(learnt.Value() - plays.Value()),
                                 std::abs(learnt.Policy() - plays.Policy()));
    const double differs = std::max(std::abs(control_learnt.Value() - plays.Value()),
                                    std::abs(control_learnt.Policy() - plays.Policy()));
    Check(outputs == 0 && same < 1e-5 && differs > 1e-3,
          absl::StrFormat("7. batch norms in evaluation: a step at rate 0 moves the outputs "
                          "by %.2g; its losses differ from Loss()'s by %.2g; the control, "
                          "training mode, by %.2g",
                          outputs, same, differs));
    const auto before = Snapshot(*m);
    m->SetLearningRate(m->BaseLearningRate());
    m->Learn(batch);
    auto scales = [](const std::string& n) {
      return absl::StrContains(n, "batch_norm") && absl::EndsWith(n, ".weight");
    };
    const double moved = Moved(*m, before, scales);
    Check(moved > 1e-5,
          absl::StrFormat("7. ... and at the network's rate the batch norms' scales learn "
                          "(by %.2g)",
                          moved));
  }

  std::filesystem::remove_all(dir);
  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
