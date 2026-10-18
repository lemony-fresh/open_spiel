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

// Tests the race's fix in our copy (thud/az/vpnet.h, 2026-10-17; thud/DESIGN_dwarf_signal.md
// §10, E1's offline sweep), on small networks (width 16, depth 2) and one torch thread:
//
//   1. The race and its fix. Two copies of one network take the same 6 learning steps on
//      the same batches, one alone, the other while a second thread runs Inference on it
//      without pause, as the trainer's inference threads do. Ours: the two report the same
//      losses at every step (to 1e-6), and every batch norm of the shared one counted
//      exactly one batch a step. The control: upstream's VPNetModel, the same way — the
//      copy beside the inference thread departs from the one alone (by more than 1e-3).
//   2. BatchNormCountsDiffer: "" for exact counts; it names each layer that counted 0 or 2
//      batches in training mode, or 1 in evaluation mode.
//   3. After a learning step the network is in evaluation mode again: its Inference equals
//      that of the same state loaded into a fresh copy, exactly.
//   4. ConvertBatchNorms, on a network whose running statistics were made stale (each batch
//      norm's means raised by 0.5, its variances by 4x — as the race left N161's):
//      evaluation-mode outputs unchanged (1e-4); in training mode its losses then within a
//      fifth of their former distance from evaluation mode's (the control: that distance
//      before, at least 0.01); the batch norms' counts and momenta as before; and
//      LossInTrainingMode changes nothing (the outputs after it exactly the same).
//
//   thud/run_tests.sh batch_norm_test   # builds and runs it
//   build-tests/bin/batch_norm_test

#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <torch/torch.h>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/spiel.h"
#include "thud/az/vpnet.h"

namespace {

namespace up = open_spiel::algorithms::torch_az;
namespace ours = open_spiel::thud_az::torch_az;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// Positions of random games, with uniform policy targets and random values.
template <typename TrainInputs>
std::vector<TrainInputs> Positions(const open_spiel::Game& game, std::mt19937* rng, int count) {
  std::vector<TrainInputs> out;
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

template <typename Model>
std::vector<typename Model::InferenceInputs> AsInference(
    const std::vector<typename Model::TrainInputs>& positions) {
  std::vector<typename Model::InferenceInputs> out;
  for (const auto& p : positions) out.push_back({p.legal_actions, p.observations});
  return out;
}

template <typename Model>
double MaxDifference(Model& a, Model& b,
                     const std::vector<typename Model::InferenceInputs>& in) {
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

// Part 1's trial: the largest difference between the losses the two copies report, one
// learning alone, the other beside a thread running Inference on it without pause.
// `after_step` sees the shared copy after each step.
template <typename Model>
double Departure(Model& alone, Model& beside,
                 const std::vector<std::vector<typename Model::TrainInputs>>& batches,
                 const std::vector<typename Model::InferenceInputs>& probe,
                 const std::function<void(Model&)>& after_step) {
  double most = 0;
  for (const auto& batch : batches) {
    const auto a = alone.Learn(batch);
    std::atomic<bool> stop(false);
    std::atomic<int> calls(0);
    std::thread inference([&]() {
      while (!stop) {
        beside.Inference(probe);
        ++calls;
      }
    });
    while (calls == 0) std::this_thread::yield();  // The thread is running.
    const auto b = beside.Learn(batch);
    stop = true;
    inference.join();
    most = std::max({most, std::abs(a.Value() - b.Value()), std::abs(a.Policy() - b.Policy())});
    after_step(beside);
  }
  return most;
}

}  // namespace

int main() {
  torch::set_num_threads(1);  // One thread per operation: the same results every run.
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const std::string dir =
      (std::filesystem::temp_directory_path() / absl::StrFormat("batch_norm_test_%d", getpid()))
          .string();
  std::filesystem::create_directories(dir + "/up");
  std::filesystem::create_directories(dir + "/ours");
  std::filesystem::create_directories(dir + "/stale");
  std::mt19937 rng(7);

  // 1. The race and its fix.
  {
    std::vector<std::vector<ours::VPNetModel::TrainInputs>> batches;
    for (int i = 0; i < 6; ++i) {
      batches.push_back(Positions<ours::VPNetModel::TrainInputs>(*game, &rng, 128));
    }
    std::vector<ours::VPNetModel::InferenceInputs> probe =
        AsInference<ours::VPNetModel>(batches[0]);
    probe.resize(8);  // Small, so that the thread calls Inference often.

    SPIEL_CHECK_TRUE(ours::CreateGraphDef(*game, /*learning_rate=*/0.01, /*weight_decay=*/1e-4,
                                          dir + "/ours", "vpnet.pb", "resnet", 16, 2,
                                          /*verbose=*/false));
    ours::VPNetModel alone(*game, dir + "/ours", "vpnet.pb", "/cpu:0");
    alone.SaveCheckpoint(0);
    ours::VPNetModel beside(*game, dir + "/ours", "vpnet.pb", "/cpu:0");
    beside.LoadCheckpoint(0);
    bool exact = true;
    std::vector<int64_t> last = beside.BatchNormCounts();
    const double ours_departure = Departure<ours::VPNetModel>(
        alone, beside, batches, probe, [&](ours::VPNetModel& m) {
          const std::vector<int64_t> now = m.BatchNormCounts();
          for (size_t i = 0; i < now.size(); ++i) exact = exact && now[i] - last[i] == 1;
          last = now;
        });

    // The control: upstream's class, the same positions and steps.
    std::vector<std::vector<up::VPNetModel::TrainInputs>> up_batches;
    for (const auto& batch : batches) {
      std::vector<up::VPNetModel::TrainInputs> b;
      for (const auto& p : batch) b.push_back({p.legal_actions, p.observations, p.policy, p.value});
      up_batches.push_back(std::move(b));
    }
    std::vector<up::VPNetModel::InferenceInputs> up_probe;
    for (const auto& p : probe) up_probe.push_back({p.legal_actions, p.observations});
    SPIEL_CHECK_TRUE(up::CreateGraphDef(*game, /*learning_rate=*/0.01, /*weight_decay=*/1e-4,
                                        dir + "/up", "vpnet.pb", "resnet", 16, 2,
                                        /*verbose=*/false));
    up::VPNetModel up_alone(*game, dir + "/up", "vpnet.pb", "/cpu:0");
    up_alone.SaveCheckpoint(0);
    up::VPNetModel up_beside(*game, dir + "/up", "vpnet.pb", "/cpu:0");
    up_beside.LoadCheckpoint(0);
    const double up_departure = Departure<up::VPNetModel>(
        up_alone, up_beside, up_batches, up_probe, [](up::VPNetModel&) {});

    Check(ours_departure <= 1e-6 && exact && up_departure > 1e-3,
          absl::StrFormat("1. beside an inference thread ours learns as alone (losses differ "
                          "by %.2g at most; its %d batch norms each counted one batch a step: "
                          "%s); the control, upstream's class, departs by %.3g",
                          ours_departure, last.size(), exact ? "yes" : "no", up_departure));
  }

  // 2. BatchNormCountsDiffer.
  {
    const std::vector<std::string> names = {"a", "b", "c"};
    const std::string none = ours::VPNetModel::BatchNormCountsDiffer(
        {5, 7, 9}, {6, 8, 9}, {true, true, false}, names);
    const std::string some = ours::VPNetModel::BatchNormCountsDiffer(
        {5, 7, 9}, {5, 9, 10}, {true, true, false}, names);
    Check(none.empty() && absl::StrContains(some, "a counted 0 (1 expected)") &&
              absl::StrContains(some, "b counted 2 (1 expected)") &&
              absl::StrContains(some, "c counted 1 (0 expected)"),
          "2. BatchNormCountsDiffer: exact counts \"\"; otherwise each layer named: " + some);
  }

  // 3. Evaluation mode after a learning step.
  {
    ours::VPNetModel m(*game, dir + "/ours", "vpnet.pb", "/cpu:0");
    m.LoadCheckpoint(0);
    const auto positions = Positions<ours::VPNetModel::TrainInputs>(*game, &rng, 64);
    m.Learn(positions);
    m.SaveCheckpoint(1);
    ours::VPNetModel loaded(*game, dir + "/ours", "vpnet.pb", "/cpu:0");
    loaded.LoadCheckpoint(1);
    const double d = MaxDifference(m, loaded, AsInference<ours::VPNetModel>(positions));
    Check(d == 0, absl::StrFormat("3. after a learning step Inference equals a fresh copy's of "
                                  "the same state (differs by %.2g)",
                                  d));
  }

  // 4. ConvertBatchNorms.
  {
    SPIEL_CHECK_TRUE(ours::CreateGraphDef(*game, /*learning_rate=*/0.01, /*weight_decay=*/1e-4,
                                          dir + "/stale", "vpnet.pb", "resnet", 16, 2,
                                          /*verbose=*/false));
    ours::VPNetModel m(*game, dir + "/stale", "vpnet.pb", "/cpu:0");
    // Statistics of a few learning steps first, then made stale.
    for (int i = 0; i < 3; ++i) m.Learn(Positions<ours::VPNetModel::TrainInputs>(*game, &rng, 128));
    {
      torch::NoGradGuard no_grad;
      for (auto& [name, t] : m.NamedTensors()) {
        if (absl::EndsWith(name, "running_mean")) t.add_(0.5);
        if (absl::EndsWith(name, "running_var")) t.mul_(4);
      }
    }
    m.SaveCheckpoint(0);
    ours::VPNetModel reference(*game, dir + "/stale", "vpnet.pb", "/cpu:0");
    reference.LoadCheckpoint(0);
    const auto test = Positions<ours::VPNetModel::TrainInputs>(*game, &rng, 256);
    const auto probe = AsInference<ours::VPNetModel>(test);
    auto distance = [&](ours::VPNetModel& model) {
      const auto trained = model.LossInTrainingMode(test), plays = model.Loss(test);
      return std::abs(trained.Value() - plays.Value()) + std::abs(trained.Policy() - plays.Policy());
    };
    const double before = distance(m);
    const double unchanged = MaxDifference(m, reference, probe);  // LossInTrainingMode's.
    const std::vector<int64_t> counts = m.BatchNormCounts();
    const std::vector<double> momenta = m.BatchNormMomenta();
    std::vector<std::vector<ours::VPNetModel::TrainInputs>> batches;
    for (int i = 0; i < 8; ++i) {
      batches.push_back(Positions<ours::VPNetModel::TrainInputs>(*game, &rng, 256));
    }
    const auto conversion = m.ConvertBatchNorms(batches);
    const double outputs = MaxDifference(m, reference, probe);
    const double after = distance(m);
    double shift = 0;
    for (const auto& c : conversion) shift = std::max(shift, c.largest_mean_shift);
    Check(unchanged == 0 && outputs <= 1e-4 && before >= 0.01 && after <= before / 5 &&
              m.BatchNormCounts() == counts && m.BatchNormMomenta() == momenta &&
              conversion.size() == counts.size(),
          absl::StrFormat("4. ConvertBatchNorms on stale statistics (%d batch norms, means off "
                          "by up to %.2f deviations): evaluation-mode outputs move by %.2g; "
                          "training mode's losses from evaluation mode's %.4f before (the "
                          "control), %.4f after; counts and momenta kept; LossInTrainingMode "
                          "moved the outputs by %.2g",
                          conversion.size(), shift, outputs, before, after, unchanged));
  }

  std::filesystem::remove_all(dir);
  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
