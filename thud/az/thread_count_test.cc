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

// Tests the LibTorch thread counts per thread (VPNetModel::SetThreadCount, VPNetEvaluator's
// thread_count: the trainer's learner_threads, 2026-10-19; thud/PLAN.md, the GPU readiness
// checklist). Its own process, which has run nothing in parallel before part 1:
//
//   1. this thread's count set to 1, then 3: ThreadCount reports each, and a parallel
//      operation grows the process by OpenMP's 2 worker threads at 3 and by none at 1 —
//      the control that the count reaches the operations, not only the report;
//   2. the value LibTorch shares between threads: a new thread that sets nothing takes the
//      last count set anywhere (3) — the reason the inference threads set their own; one
//      that sets 2 reports 2, and this thread still reports 3;
//   3. the trainer's order: an evaluator built with thread_count 2, then this thread's
//      count set to 3 (the learner's): every inference thread reports 2 after its first
//      batch, this thread 3; the control, thread_count 0 (upstream's), the same order:
//      the inference threads take this thread's 3.
//
// Prints PASS or FAIL per part, then ALL PASSED or the count of failures; exits non-zero on
// a failure.

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/spiel.h"
#include "thud/az/device_manager.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az;
using az::torch_az::VPNetEvaluator;
using az::torch_az::VPNetModel;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// The process's threads, as the kernel lists them.
int ProcessThreads() {
  return std::distance(std::filesystem::directory_iterator("/proc/self/task"),
                       std::filesystem::directory_iterator());
}

// An operation LibTorch runs in parallel: an elementwise add over 4 million floats, split
// by at::parallel_for into chunks of 32,768 (ATen's grain size), with no library of its own
// that could start threads.
void ParallelOperation() {
  torch::Tensor t = torch::ones({1 << 22});
  t.add_(1);
  SPIEL_CHECK_EQ(t[0].item<float>(), 2);
}

}  // namespace

int main() {
  // 1. This thread's count reaches its operations.
  {
    VPNetModel::SetThreadCount(1);
    const int at_1 = VPNetModel::ThreadCount();
    const int before_1 = ProcessThreads();
    ParallelOperation();
    const int grown_1 = ProcessThreads() - before_1;
    VPNetModel::SetThreadCount(3);
    const int at_3 = VPNetModel::ThreadCount();
    const int before_3 = ProcessThreads();  // After setting: a pool LibTorch makes then.
    ParallelOperation();
    const int grown_3 = ProcessThreads() - before_3;
    Check(at_1 == 1 && at_3 == 3 && grown_1 == 0 && grown_3 == 2,
          absl::StrFormat("1. this thread's count reported as set (%d, %d; 1 and 3 "
                          "expected), and its parallel operation grew the process by %d "
                          "threads at 1 and %d at 3 (0 and 2 expected: OpenMP's workers)",
                          at_1, at_3, grown_1, grown_3));
  }

  // 2. The shared value, and counts per thread.
  {
    int unset = 0, set = 0;
    std::thread([&] { ParallelOperation(); unset = VPNetModel::ThreadCount(); }).join();
    std::thread([&] {
      VPNetModel::SetThreadCount(2);
      ParallelOperation();
      set = VPNetModel::ThreadCount();
    }).join();
    const int here = VPNetModel::ThreadCount();
    Check(unset == 3 && set == 2 && here == 3,
          absl::StrFormat("2. a new thread that sets nothing takes the last count set "
                          "anywhere (%d; 3 expected), one that sets 2 reports %d, and this "
                          "thread still %d (3 expected)",
                          unset, set, here));
  }

  // 3. The evaluator in the trainer's order.
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const std::string dir =
      (std::filesystem::temp_directory_path() /
       ("thread_count_test_" + std::to_string(getpid())))
          .string();
  std::filesystem::create_directories(dir);
  SPIEL_CHECK_TRUE(az::torch_az::CreateGraphDef(*game, 0.001, 0.0001, dir, "vpnet.pb",
                                                "resnet_conv_policy", 16, 2,
                                                /*verbose=*/false, az::kNumFeaturePlanes));
  std::vector<std::unique_ptr<open_spiel::State>> states;
  std::mt19937 rng(1);
  for (std::unique_ptr<open_spiel::State> s = game->NewInitialState();
       states.size() < 64 && !s->IsTerminal();) {
    const std::vector<open_spiel::Action> legal = s->LegalActions();
    s->ApplyAction(legal[rng() % legal.size()]);
    states.push_back(s->Clone());
  }
  // Each case: the inference threads' counts and this thread's after evaluating.
  auto run = [&](int thread_count) {
    az::torch_az::DeviceManager devices;
    devices.AddDevice(VPNetModel(*game, dir, "vpnet.pb", "/cpu:0"));
    std::vector<int> counts;
    {
      // Batches of up to 4, 2 inference threads, no cache: every request infers.
      VPNetEvaluator eval(&devices, 4, 2, 0, 1, thread_count);
      VPNetModel::SetThreadCount(3);  // The learner's, as the trainer sets it.
      std::vector<std::thread> callers;
      for (int c = 0; c < 8; ++c) {
        callers.emplace_back([&, c] {
          for (size_t i = c; i < states.size(); i += 8) eval.Evaluate(*states[i]);
        });
      }
      for (auto& t : callers) t.join();
      counts = eval.InferenceThreadCounts();
    }
    return std::make_pair(counts, VPNetModel::ThreadCount());
  };
  const auto [set, here_set] = run(2);
  const auto [unset, here_unset] = run(0);
  auto all = [](const std::vector<int>& v, int n) {
    if (v.empty()) return false;
    for (int x : v) if (x != n) return false;
    return true;
  };
  Check(all(set, 2) && here_set == 3 && all(unset, 3) && here_unset == 3,
        absl::StrFormat("3. with thread_count 2 the inference threads reported [%s] after "
                        "their first batch (2 each expected) and this thread %d (3); the "
                        "control, thread_count 0: [%s] (this thread's 3 each expected)",
                        absl::StrJoin(set, ", "), here_set, absl::StrJoin(unset, ", ")));

  std::filesystem::remove_all(dir);
  if (failures == 0) {
    std::cout << "ALL PASSED" << std::endl;
    return 0;
  }
  std::cout << failures << " FAILED" << std::endl;
  return 1;
}
