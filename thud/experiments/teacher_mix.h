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

#ifndef THUD_EXPERIMENTS_TEACHER_MIX_H_
#define THUD_EXPERIMENTS_TEACHER_MIX_H_

// az_match's teacher_mix_a (thud/experiments/az_match.cc), in a header so that
// teacher_mix_test.cc can check it.

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/mcts.h"

namespace open_spiel {
namespace thud_az {

// Network A's prior mixed with kickstarting's teacher at some players' nodes
// (`teacher_mix_a`); its values are the network's. The teacher's policies are kept for
// positions searched again (a search's tree is rebuilt every move), up to kCacheSize.
class TeacherMixEvaluator : public open_spiel::thud_az::Evaluator {
 public:
  TeacherMixEvaluator(const open_spiel::Game& game,
                      std::shared_ptr<open_spiel::thud_az::Evaluator> network,
                      std::vector<double> mix, int depth, double temperature)
      : network_(std::move(network)),
        mix_(std::move(mix)),
        teacher_(game, depth),
        temperature_(temperature) {}

  std::vector<double> Evaluate(const open_spiel::State& state) override {
    return network_->Evaluate(state);
  }

  open_spiel::ActionsAndProbs Prior(const open_spiel::State& state) override {
    open_spiel::ActionsAndProbs prior = network_->Prior(state);
    const open_spiel::Player player = state.CurrentPlayer();
    const double w = player >= 0 && player < mix_.size() ? mix_[player] : 0;
    if (w <= 0) return prior;
    const std::map<open_spiel::Action, double> teacher = TeacherOf(state);
    double sum = 0;
    for (auto& [action, p] : prior) {
      p = (1 - w) * p + w * teacher.at(action);
      sum += p;
    }
    for (auto& [action, p] : prior) p /= sum;  // Only rounding: both sum to 1.
    return prior;
  }

 private:
  static constexpr int kCacheSize = 20000;

  std::map<open_spiel::Action, double> TeacherOf(const open_spiel::State& state) {
    const std::string key = state.ToString();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = cache_.find(key);
      if (it != cache_.end()) return it->second;
    }
    std::map<open_spiel::Action, double> policy;
    for (const auto& [action, p] :
         open_spiel::thud_az::torch_az::TeacherPolicy(state, teacher_, temperature_)) {
      policy[action] = p;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_.size() >= kCacheSize) cache_.clear();
    cache_[key] = policy;
    return policy;
  }

  std::shared_ptr<open_spiel::thud_az::Evaluator> network_;
  const std::vector<double> mix_;
  const open_spiel::thud_az::AlphaBetaPlayer teacher_;
  const double temperature_;
  std::mutex mutex_;
  std::map<std::string, std::map<open_spiel::Action, double>> cache_;
};

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_EXPERIMENTS_TEACHER_MIX_H_
