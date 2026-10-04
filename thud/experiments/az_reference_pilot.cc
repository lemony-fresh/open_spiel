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

// Which search should make the validation set's targets? (thud/PLAN.md, How we
// evaluate networks: the pilot.) Plays self-play games as the trainer does with
// the networks in `play` (root noise, the first `drop` moves sampled from the
// visit counts, then the most visited), takes every `every`-th position, and
// searches each one, without root noise, with every reference in `refs` at each of
// its budgets. A reference is NAME@RUN_DIR:STEP@RULE@BUDGETS — a network's PUCT
// search with that rule for untried moves (upstream or
// sibling_mean_minus_reduction) — or NAME@mcts@@BUDGETS, plain UCT with random
// rollouts and the solver, as the trainer's evaluator plays (no network).
//
//   thud/az/build.sh thud/experiments/az_reference_pilot.cc
//   OMP_NUM_THREADS=4 build-shared/az_reference_pilot play=RUN_DIR:STEP[,RUN_DIR:STEP]
//     refs='A14@DIR:14@upstream@2000,4000;mcts@mcts@@25000' [games=8 every=7 drop=10
//      threads=32 batch=32 mcts_threads=6 mcts_memory_mb=1000 seed=1]
//
// Prints one JSON line per position: its text (ThudGame::NewInitialState reads it
// back), the side to move, the legal moves, and for each reference and budget the
// root's visit counts (every visited move), the root value and the seconds taken.
// thud/experiments/az_reference_pilot.py compares them.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
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

// A network for searching: its evaluator, with a batching queue of its own.
std::shared_ptr<VPNetEvaluator> Network(const open_spiel::Game& game,
                                        const std::string& spec, int batch,
                                        std::vector<std::unique_ptr<DeviceManager>>* keep) {
  const size_t colon = spec.rfind(':');
  if (colon == std::string::npos) open_spiel::SpielFatalError("Expected RUN_DIR:STEP");
  keep->push_back(std::make_unique<DeviceManager>());
  VPNetModel model(game, spec.substr(0, colon), "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
  keep->back()->AddDevice(std::move(model));
  return std::make_shared<VPNetEvaluator>(keep->back().get(), batch, /*threads=*/2,
                                          /*cache_size=*/1 << 18, /*shards=*/1);
}

struct Reference {
  std::string name;
  bool mcts;  // Plain UCT with random rollouts, no network.
  std::shared_ptr<VPNetEvaluator> eval;
  open_spiel::thud_az::UntriedMoveValue rule;
  std::vector<int> budgets;
};

}  // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  const std::string play = args.Get("play", "");
  const std::string refs_arg = args.Get("refs", "");
  const int games = args.GetInt("games", 8);
  const int every = args.GetInt("every", 7);
  const int drop = args.GetInt("drop", 10);
  const int threads = args.GetInt("threads", 32);
  const int batch = args.GetInt("batch", 32);
  const int mcts_threads = args.GetInt("mcts_threads", 6);
  const int mcts_memory_mb = args.GetInt("mcts_memory_mb", 1000);
  const int seed = args.GetInt("seed", 1);
  args.CheckAllUsed();
  if (play.empty() || refs_arg.empty()) {
    std::cerr << "Usage: " << argv[0] << " play=RUN_DIR:STEP refs=NAME@...;..." << std::endl;
    return 1;
  }
  if (every % 2 == 0) {  // The dwarfs move on even turns: an even step samples one side.
    std::cerr << "every must be odd, so that both sides to move are sampled" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::vector<std::unique_ptr<DeviceManager>> devices;

  std::vector<Reference> refs;
  for (absl::string_view r : absl::StrSplit(refs_arg, ';', absl::SkipEmpty())) {
    std::vector<std::string> f = absl::StrSplit(r, '@');
    if (f.size() != 4) open_spiel::SpielFatalError(absl::StrCat("Bad reference: ", r));
    Reference ref{f[0], f[1] == "mcts", nullptr,
                  open_spiel::thud_az::UntriedMoveValue::kUpstream, {}};
    if (!ref.mcts) {
      ref.eval = Network(*game, f[1], batch, &devices);
      ref.rule = open_spiel::thud_az::UntriedMoveValueFromString(f[2]);
    }
    for (absl::string_view b : absl::StrSplit(f[3], ',')) {
      int v;
      if (!absl::SimpleAtoi(b, &v)) open_spiel::SpielFatalError("Bad budget");
      ref.budgets.push_back(v);
    }
    refs.push_back(std::move(ref));
  }

  // Phase 1: self-play games of each network in `play`, as the trainer plays them
  // (100 simulations, root noise 0.1 / 0.25, the default rule).
  std::vector<std::string> players = absl::StrSplit(play, ',');
  std::vector<std::unique_ptr<open_spiel::State>> positions;
  std::vector<std::string> sources;
  std::mutex mu;
  for (const std::string& player : players) {
    std::shared_ptr<VPNetEvaluator> eval = Network(*game, player, batch, &devices);
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < std::min(threads, games); ++t) {
      workers.emplace_back([&, t]() {
        MCTSBot bot(*game, eval, /*uct_c=*/2, /*max_simulations=*/100,
                    /*max_memory_mb=*/1000, /*solve=*/false, seed * 1000 + t,
                    /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT,
                    /*dirichlet_alpha=*/0.1, /*dirichlet_epsilon=*/0.25,
                    /*dont_return_chance_node=*/true, /*max_wall_clock_time=*/-1,
                    open_spiel::thud_az::UntriedMoveValue::kSiblingMeanMinusReduction);
        std::mt19937 rng(seed * 7919 + t);
        for (int g = next++; g < games; g = next++) {
          std::unique_ptr<open_spiel::State> state = game->NewInitialState();
          for (int move = 0; !state->IsTerminal(); ++move) {
            if (move > 0 && move % every == 0) {
              std::lock_guard<std::mutex> lock(mu);
              positions.push_back(state->Clone());
              sources.push_back(player);
            }
            std::unique_ptr<SearchNode> root = bot.MCTSearch(*state);
            Action action;
            if (move < drop) {
              std::vector<double> w;
              for (const SearchNode& c : root->children) w.push_back(c.explore_count);
              std::discrete_distribution<int> pick(w.begin(), w.end());
              action = root->children[pick(rng)].action;
            } else {
              action = root->BestChild().action;
            }
            state->ApplyAction(action);
          }
        }
      });
    }
    for (auto& w : workers) w.join();
  }
  std::cerr << positions.size() << " positions" << std::endl;

  // Phase 2: every reference at every budget on every position. Network references
  // with `threads` workers (their evaluators batch across them), rollouts with
  // `mcts_threads` (each tree up to mcts_memory_mb).
  std::vector<std::vector<std::string>> results(positions.size());
  for (const Reference& ref : refs) {
    for (int budget : ref.budgets) {
      std::atomic<int> next{0};
      std::vector<std::thread> workers;
      const int n = ref.mcts ? mcts_threads : threads;
      for (int t = 0; t < n; ++t) {
        workers.emplace_back([&, t]() {
          std::unique_ptr<MCTSBot> bot;
          if (ref.mcts) {
            bot = std::make_unique<MCTSBot>(
                *game,
                std::make_shared<open_spiel::thud_az::RandomRolloutEvaluator>(
                    1, seed * 100 + t),
                /*uct_c=*/2, budget, mcts_memory_mb, /*solve=*/true, seed + t,
                /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::UCT,
                /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0,
                /*dont_return_chance_node=*/true);
          } else {
            bot = std::make_unique<MCTSBot>(
                *game, ref.eval, /*uct_c=*/2, budget, /*max_memory_mb=*/2000,
                /*solve=*/false, seed, /*verbose=*/false,
                open_spiel::thud_az::ChildSelectionPolicy::PUCT,
                /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0,
                /*dont_return_chance_node=*/true, /*max_wall_clock_time=*/-1, ref.rule);
          }
          for (int i = next++; i < positions.size(); i = next++) {
            const auto start = std::chrono::steady_clock::now();
            std::unique_ptr<SearchNode> root = bot->MCTSearch(*positions[i]);
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
            std::vector<std::string> visits;
            for (const SearchNode& c : root->children) {
              if (c.explore_count > 0) {
                visits.push_back(absl::StrFormat("\"%d\": %d", c.action, c.explore_count));
              }
            }
            const std::string entry = absl::StrFormat(
                "\"%s@%d\": {\"value\": %.4f, \"seconds\": %.2f, \"visits\": {%s}}",
                ref.name, budget, root->total_reward / root->explore_count, seconds,
                absl::StrJoin(visits, ", "));
            std::lock_guard<std::mutex> lock(mu);
            results[i].push_back(entry);
          }
        });
      }
      for (auto& w : workers) w.join();
      std::cerr << ref.name << "@" << budget << " done" << std::endl;
    }
  }
  for (int i = 0; i < positions.size(); ++i) {
    const open_spiel::State& state = *positions[i];
    std::cout << absl::StrFormat(
                     "{\"position\": %d, \"source\": \"%s\", \"side\": \"%s\", "
                     "\"legal\": %d, \"text\": \"%s\", \"searches\": {%s}}",
                     i, sources[i], state.CurrentPlayer() == 0 ? "dwarfs" : "trolls",
                     state.LegalActions().size(),
                     absl::StrReplaceAll(state.ToString(), {{"\n", "\\n"}}),
                     absl::StrJoin(results[i], ", "))
              << std::endl;
  }
  return 0;
}
