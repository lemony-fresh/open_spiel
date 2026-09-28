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
// open_spiel/algorithms/alpha_zero_torch/vpevaluator.h at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#ifndef THUD_AZ_VPEVALUATOR_H_
#define THUD_AZ_VPEVALUATOR_H_

#include <future>  // NOLINT
#include <vector>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "thud/az/device_manager.h"
#include "thud/az/vpnet.h"
#include "thud/az/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"

namespace open_spiel {
namespace thud_az {
namespace torch_az {

class VPNetEvaluator : public Evaluator {
 public:
  explicit VPNetEvaluator(DeviceManager* device_manager, int batch_size,
                          int threads, int cache_size, int cache_shards = 1);
  ~VPNetEvaluator() override;

  // Return a value of this state for each player.
  std::vector<double> Evaluate(const State& state) override;

  // Return a policy: the probability of the current player playing each action.
  ActionsAndProbs Prior(const State& state) override;

  void ClearCache();
  LRUCacheInfo CacheInfo();

  void ResetBatchSizeStats();
  open_spiel::BasicStats BatchSizeStats();
  open_spiel::HistogramNumbered BatchSizeHistogram();

 private:
  VPNetModel::InferenceOutputs Inference(const State& state);

  void Runner();

  DeviceManager& device_manager_;
  std::vector<std::unique_ptr<LRUCache<uint64_t, VPNetModel::InferenceOutputs>>>
      cache_;
  const int batch_size_;

  struct QueueItem {
    VPNetModel::InferenceInputs inputs;
    std::promise<VPNetModel::InferenceOutputs>* prom;
  };

  ThreadedQueue<QueueItem> queue_;
  StopToken stop_;
  std::vector<Thread> inference_threads_;
  absl::Mutex inference_queue_m_;  // Only one thread at a time should pop.

  absl::Mutex stats_m_;
  open_spiel::BasicStats batch_size_stats_;
  open_spiel::HistogramNumbered batch_size_hist_;
};

}  // namespace torch_az
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_VPEVALUATOR_H_
