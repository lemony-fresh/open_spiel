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
// The throughput sweep (2026-10-15; thud/PLAN.md, the GPU readiness checklist): part 2
// with a real network, `net=DIR:STEP` (instead of the tiny one), on `devices` (one copy
// per entry, as the trainer's --devices: cuda:0,cuda:0), over the grid of searching
// threads (`batched_threads`, the trainer's actors) x `batch` (its inference_batch_size)
// x `inference_threads`, each a list; then a line naming the fastest. `parts=2` skips
// part 1. The searches are self-play's on fixed positions, not whole games: a measure of
// the machine settings, not of a run's speed.
//
//   thud/az/build.sh thud/experiments/az_search_cost.cc
//   OMP_NUM_THREADS=1 build-shared/az_search_cost [positions=200 sims=100
//     max_threads=10 batch=32 batched_threads=8,32,64 inference_threads=2
//     cache_shards=1 scratch=/tmp/az_search_cost parts=1,2 net=DIR:STEP devices=/cpu:0]
//   OMP_NUM_THREADS=4 build-shared/az_search_cost parts=2 net=RUN_DIR:-1 \
//     batched_threads=32,64,128 batch=16,32,64 inference_threads=1,2,4 devices=cuda:0
//
// `legal_policy=0` computes the priors as upstream does, over every distinct action
// (VPNetModel::SetLegalPolicy; default 1, ours: the legal moves alone), to measure what
// the legal policy gains. `game=chess` (any game OpenSpiel's AlphaZero takes) with
// `nn_model`, `nn_width` and `nn_depth` (default the tiny mlp, 8 x 1) measures part 2 for
// another game, on a network built fresh.
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
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
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
  auto ints = [&](const std::string& k, const std::string& d) {
    std::vector<int> out;
    for (const std::string& t : std::vector<std::string>(absl::StrSplit(get(k, d), ','))) {
      out.push_back(std::stoi(t));
    }
    return out;
  };
  const std::vector<int> batches = ints("batch", "32");
  const std::vector<int> inference_threads_list = ints("inference_threads", "2");
  const int cache_shards = std::stoi(get("cache_shards", "1"));
  const std::string scratch = get("scratch", "/tmp/az_search_cost");
  const std::vector<int> batched_threads = ints("batched_threads", "8,32,64");
  const std::vector<int> parts = ints("parts", "1,2");
  auto part = [&](int p) { return std::find(parts.begin(), parts.end(), p) != parts.end(); };
  const std::string net = get("net", "");
  const bool legal_policy = get("legal_policy", "1") == "1";
  const std::string nn_model = get("nn_model", "mlp");
  const int nn_width = std::stoi(get("nn_width", "8")), nn_depth = std::stoi(get("nn_depth", "1"));
  const std::vector<std::string> device_names = absl::StrSplit(get("devices", "/cpu:0"), ',');
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(get("game", "thud"));

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
  for (int p = 0; p < 2 && part(1); ++p) {
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
  for (int threads = 1; threads <= max_threads && part(1); threads *= 2) {
    const double seconds =
        SearchAll(positions, threads, [&](int t) { return bot_with(free_eval, 200 + t); });
    std::cout << absl::StrFormat(
                     "{\"part\": \"free network\", \"side\": \"both\", \"threads\": %d, "
                     "\"simulations_per_s\": %.0f, \"per_thread\": %.0f}",
                     threads, positions.size() * sims / seconds,
                     positions.size() * sims / seconds / threads)
              << std::endl;
  }

  // 2. A network behind the batching evaluator: the tiny one, or `net`.
  if (!part(2)) return 0;
  std::string dir = scratch;
  int step = -2;  // None: the tiny network is used as it is created.
  if (net.empty()) {
    std::filesystem::create_directories(scratch);
    if (!open_spiel::thud_az::torch_az::CreateGraphDef(*game, 1e-3, 1e-4, scratch, "vpnet.pb",
                                                       nn_model, nn_width, nn_depth)) {
      std::cerr << "CreateGraphDef failed" << std::endl;
      return 1;
    }
  } else {
    dir = net.substr(0, net.rfind(':'));
    step = std::stoi(net.substr(net.rfind(':') + 1));
  }
  const std::string what = net.empty() ? "tiny network, batched" : "network, batched";
  double best = 0;
  std::string best_line;
  for (int threads : batched_threads) {
    for (int batch : batches) {
      for (int inference_threads : inference_threads_list) {
        open_spiel::thud_az::torch_az::DeviceManager devices;
        for (const std::string& d : device_names) {
          open_spiel::thud_az::torch_az::VPNetModel model(*game, dir, "vpnet.pb", d);
          if (step != -2) model.LoadCheckpoint(step);
          model.SetLegalPolicy(legal_policy);
          devices.AddDevice(std::move(model));
        }
        auto eval = std::make_shared<open_spiel::thud_az::torch_az::VPNetEvaluator>(
            &devices, batch, inference_threads, /*cache_size=*/1 << 18, cache_shards);
        const double seconds =
            SearchAll(positions, threads, [&](int t) { return bot_with(eval, 300 + t); });
        const double rate = positions.size() * sims / seconds;
        const std::string line = absl::StrFormat(
            "{\"part\": \"%s\", \"threads\": %d, \"batch\": %d, "
            "\"inference_threads\": %d, \"cache_shards\": %d, \"simulations_per_s\": %.0f%s}",
            what, threads, batch, inference_threads, cache_shards, rate,
            (net.empty() && device_names.size() == 1 && device_names[0] == "/cpu:0"
                 ? std::string()
                 : absl::StrFormat(", \"net\": \"%s\", \"devices\": \"%s\"", net,
                                   absl::StrJoin(device_names, ","))) +
                (legal_policy ? "" : ", \"legal_policy\": false") +
                (game->GetType().short_name == "thud" && nn_model == "mlp" && nn_width == 8 &&
                         nn_depth == 1
                     ? ""
                     : absl::StrFormat(", \"game\": \"%s\", \"nn\": \"%s %d x %d\"",
                                       game->GetType().short_name, net.empty() ? nn_model : "net",
                                       nn_width, nn_depth)));
        std::cout << line << std::endl;
        if (rate > best) {
          best = rate;
          best_line = line;
        }
      }
    }
  }
  if (batched_threads.size() * batches.size() * inference_threads_list.size() > 1) {
    std::cout << "{\"fastest\": " << best_line << "}" << std::endl;
  }
  return 0;
}
