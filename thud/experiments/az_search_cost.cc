// Copyright 2026 The Thud-on-OpenSpiel authors.
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

// How much CPU does our search need per simulation, apart from the network?
// (thud/PLAN.md, Cloud GPU options: it says how many CPU cores a GPU needs.) On
// positions of random games (every 7th move, both sides), our copy's search as
// self-play runs it (PUCT, root noise 0.1 / 0.25, the default rule for untried
// moves), timed by a steady clock:
//
//   1. With a stand-in for the network that costs nothing (value 0, a uniform
//      prior): the tree search and the game alone — simulations a second on one
//      core, per side to move, and with 1, 2, 4, ... threads searching different
//      positions at once (how it scales over the cores).
//   2. With a tiny real network (an MLP of width 8, nearly free to evaluate) behind
//      the trainer's batching evaluator (VPNetEvaluator, batches of `batch`, two
//      inference threads, the trainer's cache of 262,144 — without it each
//      simulation asks twice, for the value and then the prior, which the trainer's
//      cache answers), many threads searching at once: adds what the batching
//      queue costs.
//
// Cores a GPU needs ≈ the GPU's evaluations a second / part 2's simulations a
// second per core (a simulation asks for one evaluation).
//
//   thud/az/build.sh thud/experiments/az_search_cost.cc
//   OMP_NUM_THREADS=1 build-shared/az_search_cost [positions=200 sims=100
//     max_threads=10 batch=32 batched_threads=8,32,64 inference_threads=2
//     cache_shards=1 scratch=/tmp/az_search_cost]
//
// Prints one JSON line per measurement.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::MCTSBot;

// A network that costs nothing: every position even, every legal move alike.
class FreeEvaluator : public open_spiel::thud_az::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override { return {0, 0}; }
  open_spiel::ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    open_spiel::ActionsAndProbs prior;
    for (Action a : legal) prior.push_back({a, 1.0 / legal.size()});
    return prior;
  }
};

std::map<std::string, std::string> Args(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    args[arg.substr(0, arg.find('='))] = arg.substr(arg.find('=') + 1);
  }
  return args;
}

// Searches every position once with `threads` threads, each with its own bot from
// `make_bot`; returns the wall-clock seconds.
template <typename MakeBot>
double SearchAll(const std::vector<std::unique_ptr<State>>& positions, int threads,
                 MakeBot make_bot) {
  std::atomic<int> next{0};
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t]() {
      std::unique_ptr<MCTSBot> bot = make_bot(t);
      for (int i = next++; i < positions.size(); i = next++) bot->MCTSearch(*positions[i]);
    });
  }
  for (auto& w : workers) w.join();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  auto args = Args(argc, argv);
  auto get = [&](const std::string& k, const std::string& d) {
    return args.count(k) ? args[k] : d;
  };
  const int num_positions = std::stoi(get("positions", "200"));
  const int sims = std::stoi(get("sims", "100"));
  const int max_threads = std::stoi(get("max_threads", "10"));
  const int batch = std::stoi(get("batch", "32"));
  const int inference_threads = std::stoi(get("inference_threads", "2"));
  const int cache_shards = std::stoi(get("cache_shards", "1"));
  const std::string scratch = get("scratch", "/tmp/az_search_cost");
  std::vector<int> batched_threads;
  for (const std::string& t :
       std::vector<std::string>(absl::StrSplit(get("batched_threads", "8,32,64"), ','))) {
    batched_threads.push_back(std::stoi(t));
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // Positions of random games: every 7th move (odd, so both sides to move).
  std::mt19937 rng(20261004);
  std::vector<std::unique_ptr<State>> positions, side[2];
  while (positions.size() < num_positions) {
    std::unique_ptr<State> state = game->NewInitialState();
    for (int move = 0; !state->IsTerminal() && positions.size() < num_positions; ++move) {
      if (move > 0 && move % 7 == 0) {
        positions.push_back(state->Clone());
        side[state->CurrentPlayer()].push_back(state->Clone());
      }
      std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
  }
  auto bot_with = [&](std::shared_ptr<open_spiel::thud_az::Evaluator> eval, int seed) {
    return std::make_unique<MCTSBot>(
        *game, eval, /*uct_c=*/2, sims, /*max_memory_mb=*/1000, /*solve=*/false, seed,
        /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT, 0.1, 0.25,
        /*dont_return_chance_node=*/true);
  };

  // 1. The free stand-in: per side on one core, then threads.
  auto free_eval = std::make_shared<FreeEvaluator>();
  const char* names[2] = {"dwarfs", "trolls"};
  for (int p = 0; p < 2; ++p) {
    const double seconds =
        SearchAll(side[p], 1, [&](int t) { return bot_with(free_eval, 100 + t); });
    std::cout << absl::StrFormat(
                     "{\"part\": \"free network\", \"side\": \"%s\", \"threads\": 1, "
                     "\"positions\": %d, \"simulations_per_s\": %.0f, "
                     "\"microseconds_per_simulation\": %.1f}",
                     names[p], side[p].size(), side[p].size() * sims / seconds,
                     1e6 * seconds / (side[p].size() * sims))
              << std::endl;
  }
  for (int threads = 1; threads <= max_threads; threads *= 2) {
    const double seconds =
        SearchAll(positions, threads, [&](int t) { return bot_with(free_eval, 200 + t); });
    std::cout << absl::StrFormat(
                     "{\"part\": \"free network\", \"side\": \"both\", \"threads\": %d, "
                     "\"simulations_per_s\": %.0f, \"per_thread\": %.0f}",
                     threads, positions.size() * sims / seconds,
                     positions.size() * sims / seconds / threads)
              << std::endl;
  }

  // 2. A tiny real network behind the batching evaluator.
  std::filesystem::create_directories(scratch);
  if (!open_spiel::thud_az::torch_az::CreateGraphDef(*game, 1e-3, 1e-4, scratch,
                                                     "vpnet.pb", "mlp", 8, 1)) {
    std::cerr << "CreateGraphDef failed" << std::endl;
    return 1;
  }
  for (int threads : batched_threads) {
    open_spiel::thud_az::torch_az::DeviceManager devices;
    devices.AddDevice(
        open_spiel::thud_az::torch_az::VPNetModel(*game, scratch, "vpnet.pb", "/cpu:0"));
    auto eval = std::make_shared<open_spiel::thud_az::torch_az::VPNetEvaluator>(
        &devices, batch, inference_threads, /*cache_size=*/1 << 18, cache_shards);
    const double seconds =
        SearchAll(positions, threads, [&](int t) { return bot_with(eval, 300 + t); });
    std::cout << absl::StrFormat(
                     "{\"part\": \"tiny network, batched\", \"threads\": %d, \"batch\": %d, "
                     "\"inference_threads\": %d, \"cache_shards\": %d, \"simulations_per_s\": %.0f}",
                     threads, batch, inference_threads, cache_shards,
                     positions.size() * sims / seconds)
              << std::endl;
  }
  return 0;
}
