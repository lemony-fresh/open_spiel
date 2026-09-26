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

// How fast OpenSpiel's C++ AlphaZero searches and learns on this machine, for any game,
// with upstream's own pieces (MCTSBot with PUCT, VPNetEvaluator, VPNetModel), unmodified
// and freshly initialised: the speed does not depend on the weights.
//
//   search  N searcher threads share one evaluator and its cache, as the trainer's actors
//           do; each plays its own random games and searches every 3rd position, so no
//           position repeats. Counts simulations after a warm-up, separately for the two
//           halves of the window (a drop in the second half means the CPU throttles).
//   learn   Times VPNetModel::Learn on batches of positions from random games.
//   infer   Times the network alone (VPNetModel::Inference) per position, one at a time
//           and in batches of 64, on positions from random games.
//
//   thud/experiments/build_az_program.sh az_throughput
//   OMP_NUM_THREADS=1 build-shared/az_throughput search game=thud width=64 depth=4 \
//     sims=100 searchers=10 batch=1 inference_threads=1 seconds=30
//   OMP_NUM_THREADS=10 build-shared/az_throughput learn game=thud width=64 depth=4 \
//     batch=1024 steps=5
//   OMP_NUM_THREADS=1 build-shared/az_throughput infer game=thud width=64 depth=4
//
// Prints one JSON line.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
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

namespace {

using open_spiel::Action;
using open_spiel::algorithms::torch_az::DeviceManager;
using open_spiel::algorithms::torch_az::VPNetEvaluator;
using open_spiel::algorithms::torch_az::VPNetModel;
using Clock = std::chrono::steady_clock;

// key=value arguments, each with a default.
class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      const size_t eq = arg.find('=');
      if (eq == std::string::npos) {
        open_spiel::SpielFatalError(absl::StrCat("Expected key=value, got ", arg));
      }
      values_[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
  }
  std::string Get(const std::string& key, const std::string& fallback) {
    used_.push_back(key);
    return values_.count(key) ? values_[key] : fallback;
  }
  int GetInt(const std::string& key, int fallback) {
    return std::stoi(Get(key, std::to_string(fallback)));
  }
  void CheckAllUsed() const {
    for (const auto& [key, value] : values_) {
      if (std::find(used_.begin(), used_.end(), key) == used_.end()) {
        open_spiel::SpielFatalError(absl::StrCat("Unknown argument ", key));
      }
    }
  }

 private:
  std::map<std::string, std::string> values_;
  std::vector<std::string> used_;
};

// A fresh network of the given size, as the trainer creates it.
VPNetModel NewModel(const open_spiel::Game& game, int width, int depth) {
  const std::string dir = absl::StrCat(std::filesystem::temp_directory_path().string(),
                                       "/az_throughput_", game.GetType().short_name,
                                       "_", width, "x", depth, "_", getpid());
  std::filesystem::create_directories(dir);
  SPIEL_CHECK_TRUE(open_spiel::algorithms::torch_az::CreateGraphDef(
      game, /*learning_rate=*/1e-3, /*weight_decay=*/1e-4, dir, "vpnet.pb", "resnet",
      width, depth));
  torch::manual_seed(0);
  VPNetModel model(game, dir, "vpnet.pb", "/cpu:0");
  std::filesystem::remove_all(dir);
  return model;
}

void Search(const open_spiel::Game& game, Args& args) {
  const int width = args.GetInt("width", 64), depth = args.GetInt("depth", 4);
  const int sims = args.GetInt("sims", 100);
  const int searchers = args.GetInt("searchers", 1);
  const int batch = args.GetInt("batch", 1);
  const int inference_threads = args.GetInt("inference_threads", 1);
  const int seconds = args.GetInt("seconds", 30);
  const int warmup = args.GetInt("warmup", 5);
  const int cache = args.GetInt("cache", 1 << 18);  // The trainer's default.
  args.CheckAllUsed();

  DeviceManager devices;
  devices.AddDevice(NewModel(game, width, depth));
  auto evaluator = std::make_shared<VPNetEvaluator>(&devices, batch, inference_threads,
                                                    cache, /*cache_shards=*/1);

  // Each searcher records (seconds since start, simulations) per finished search.
  std::vector<std::vector<std::pair<double, int>>> done(searchers);
  std::vector<std::vector<int>> branching(searchers);
  const auto start = Clock::now();
  const double end = warmup + seconds;
  std::vector<std::thread> threads;
  for (int t = 0; t < searchers; ++t) {
    threads.emplace_back([&, t]() {
      // As the trainer's actors are set up (alpha_zero.cc, InitAZBot), without noise.
      open_spiel::algorithms::MCTSBot bot(
          game, evaluator, /*uct_c=*/2, sims, /*max_memory_mb=*/10, /*solve=*/false,
          /*seed=*/t, /*verbose=*/false, open_spiel::algorithms::ChildSelectionPolicy::PUCT);
      std::mt19937 rng(1000 + t);
      std::unique_ptr<open_spiel::State> state = game.NewInitialState();
      for (int turn = 0;; ++turn) {
        if (state->IsTerminal()) state = game.NewInitialState();
        if (state->IsChanceNode()) {
          state->ApplyAction(open_spiel::SampleAction(state->ChanceOutcomes(), rng).first);
          continue;
        }
        std::vector<Action> legal = state->LegalActions();
        if (turn % 3 == 0) {
          std::unique_ptr<open_spiel::algorithms::SearchNode> root = bot.MCTSearch(*state);
          const double now = std::chrono::duration<double>(Clock::now() - start).count();
          done[t].push_back({now, root->explore_count});
          branching[t].push_back(legal.size());
          if (now > end) break;
        }
        state->ApplyAction(legal[std::uniform_int_distribution<int>(
            0, legal.size() - 1)(rng)]);
      }
    });
  }
  for (std::thread& t : threads) t.join();

  // Simulations finished in each half of the measuring window.
  int64_t halves[2] = {0, 0}, searches = 0;
  double moves = 0;
  for (int t = 0; t < searchers; ++t) {
    for (int i = 0; i < done[t].size(); ++i) {
      const auto& [when, count] = done[t][i];
      if (when <= warmup || when > end) continue;
      halves[when <= warmup + seconds / 2.0 ? 0 : 1] += count;
      searches += 1;
      moves += branching[t][i];
    }
  }
  const open_spiel::LRUCacheInfo cache_info = evaluator->CacheInfo();
  std::cout << absl::StrFormat(
                   "{\"mode\": \"search\", \"game\": \"%s\", \"width\": %d, \"depth\": %d, "
                   "\"sims\": %d, \"searchers\": %d, \"batch\": %d, "
                   "\"inference_threads\": %d, \"seconds\": %d, \"sims_per_s\": %.1f, "
                   "\"first_half\": %.1f, \"second_half\": %.1f, \"searches\": %d, "
                   "\"mean_legal_moves\": %.1f, \"cache_hit_rate\": %.3f, "
                   "\"mean_batch\": %.2f}",
                   game.GetType().short_name, width, depth, sims, searchers, batch,
                   batch <= 1 ? 0 : inference_threads,  // As VPNetEvaluator uses them.
                   seconds, (halves[0] + halves[1]) / double(seconds),
                   halves[0] / (seconds / 2.0), halves[1] / (seconds / 2.0), searches,
                   searches ? moves / searches : 0, cache_info.HitRate(),
                   evaluator->BatchSizeStats().Avg())
            << std::endl;
}

// The first `count` positions of random games, one after another.
std::vector<std::unique_ptr<open_spiel::State>> RandomPositions(
    const open_spiel::Game& game, int count) {
  std::mt19937 rng(0);
  std::vector<std::unique_ptr<open_spiel::State>> positions;
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  while (positions.size() < count) {
    if (state->IsTerminal()) state = game.NewInitialState();
    if (state->IsChanceNode()) {
      state->ApplyAction(open_spiel::SampleAction(state->ChanceOutcomes(), rng).first);
      continue;
    }
    positions.push_back(state->Clone());
    std::vector<Action> legal = state->LegalActions();
    state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
  }
  return positions;
}

void Infer(const open_spiel::Game& game, Args& args) {
  const int width = args.GetInt("width", 64), depth = args.GetInt("depth", 4);
  const int count = args.GetInt("positions", 256);
  args.CheckAllUsed();
  std::vector<VPNetModel::InferenceInputs> inputs;
  for (const auto& state : RandomPositions(game, count)) {
    inputs.push_back({state->LegalActions(), state->ObservationTensor()});
  }
  VPNetModel model = NewModel(game, width, depth);
  model.Inference({inputs[0]});  // Warm-up.
  auto start = Clock::now();
  for (const auto& input : inputs) model.Inference({input});
  const double one = std::chrono::duration<double>(Clock::now() - start).count() / count;
  start = Clock::now();
  for (int i = 0; i + 64 <= count; i += 64) {
    model.Inference({inputs.begin() + i, inputs.begin() + i + 64});
  }
  const double batched =
      std::chrono::duration<double>(Clock::now() - start).count() / (count / 64 * 64);
  std::cout << absl::StrFormat(
                   "{\"mode\": \"infer\", \"game\": \"%s\", \"width\": %d, "
                   "\"depth\": %d, \"ms_one_at_a_time\": %.3f, \"ms_in_batches_of_64\": %.3f}",
                   game.GetType().short_name, width, depth, one * 1e3, batched * 1e3)
            << std::endl;
}

void Learn(const open_spiel::Game& game, Args& args) {
  const int width = args.GetInt("width", 64), depth = args.GetInt("depth", 4);
  const int batch = args.GetInt("batch", 1024);
  const int steps = args.GetInt("steps", 5);
  args.CheckAllUsed();

  // Positions from random games, with a uniform policy and a draw as targets: the time
  // a training step takes does not depend on the targets.
  std::mt19937 rng(0);
  std::vector<VPNetModel::TrainInputs> inputs;
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  while (inputs.size() < batch) {
    if (state->IsTerminal()) state = game.NewInitialState();
    if (state->IsChanceNode()) {
      state->ApplyAction(open_spiel::SampleAction(state->ChanceOutcomes(), rng).first);
      continue;
    }
    std::vector<Action> legal = state->LegalActions();
    open_spiel::ActionsAndProbs policy;
    for (Action a : legal) policy.push_back({a, 1.0 / legal.size()});
    inputs.push_back({legal, state->ObservationTensor(), policy, 0.0});
    state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
  }
  VPNetModel model = NewModel(game, width, depth);
  model.Learn(inputs);  // Warm-up.
  const auto start = Clock::now();
  for (int i = 0; i < steps; ++i) model.Learn(inputs);
  const double per_step =
      std::chrono::duration<double>(Clock::now() - start).count() / steps;
  std::cout << absl::StrFormat(
                   "{\"mode\": \"learn\", \"game\": \"%s\", \"width\": %d, \"depth\": %d, "
                   "\"batch\": %d, \"steps\": %d, \"seconds_per_step\": %.3f, "
                   "\"positions_per_s\": %.0f}",
                   game.GetType().short_name, width, depth, batch, steps, per_step,
                   batch / per_step)
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc < 2 ? "" : argv[1];
  if (mode != "search" && mode != "learn" && mode != "infer") {
    std::cerr << "Usage: " << argv[0] << " search|learn|infer game=NAME [key=value ...]"
              << std::endl;
    return 1;
  }
  Args args(argc, argv);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(args.Get("game", "thud"));
  if (mode == "search") {
    Search(*game, args);
  } else if (mode == "learn") {
    Learn(*game, args);
  } else {
    Infer(*game, args);
  }
  return 0;
}
