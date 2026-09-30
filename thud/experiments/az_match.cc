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

// Head-to-head matches between two OpenSpiel C++ AlphaZero networks, in Thud's own match
// format: pairs of battles with the sides swapped, won on the summed margin.
//
// Each pair starts both battles from the same opening, a few uniformly random moves from
// the initial position (so pairs differ); swapping sides cancels whatever advantage an
// opening gives one side. Both networks search with our copy of OpenSpiel's MCTSBot
// (PUCT; thud/az/), the same number of simulations (`sims`; `sims_a` and `sims_b` give
// each network its own, e.g. to measure what more simulations gain on each side) and the
// same rule for untried moves (`untried`, default upstream's — with it our copy searches
// exactly as upstream's does, thud/az/identity_check.cc; `untried_a` and `untried_b` give
// each network its own, e.g. the rule it was trained with), no root noise, and always play
// their most visited move.
// Each network has its own batched evaluator. Prints one JSON line per pair, then a summary:
// network A's summed margin per pair (in points), its mean with a 95% interval (Student's
// t), pairs won, drawn and lost, and each network's mean margin per side. With `progress`,
// it also reports the simulations per second to stderr every `progress` seconds — a
// throughput measure that needs no finished pairs — with the network requests by kind
// (values, move probabilities) and how many the cache answered. `cache` sets each
// evaluator's cache size (0: none); `share=1` gives a network playing itself one
// evaluator for both sides, as in self-play, so that each side's search can hit the
// positions the other side's evaluated.
//
//   thud/az/build.sh thud/experiments/az_match.cc
//   OMP_NUM_THREADS=4 build-shared/az_match a=RUN_DIR:STEP b=RUN_DIR:STEP sims=100 \
//     pairs=50 [sims_a=SIMS sims_b=SIMS opening=4 threads=16 batch=16 seed=1 \
//     untried=upstream untried_a=UNTRIED untried_b=UNTRIED progress=0 cache=262144 share=0]
//
// STEP is a checkpoint step of that run (-1: its most recent checkpoint).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
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

constexpr double kMaxMargin = 32;  // Thud's returns are the margin over 32.

std::atomic<int64_t> simulations_done{0};  // For the progress reports.

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

// A network from a trainer's directory, "DIR:STEP" (STEP -1: the most recent).
std::shared_ptr<VPNetEvaluator> LoadEvaluator(const open_spiel::Game& game,
                                              const std::string& spec,
                                              DeviceManager* devices, int batch,
                                              int inference_threads, int cache_size) {
  const size_t colon = spec.rfind(':');
  if (colon == std::string::npos) {
    open_spiel::SpielFatalError(absl::StrCat("Expected DIR:STEP, got ", spec));
  }
  VPNetModel model(game, spec.substr(0, colon), "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
  devices->AddDevice(std::move(model));
  return std::make_shared<VPNetEvaluator>(devices, batch, inference_threads,
                                          cache_size, /*cache_shards=*/1);
}

struct Battle {
  double a_return;  // Network A's return, -1 to 1.
  int moves;
};

// One battle from `opening`, network A playing `a_player`.
Battle Play(const open_spiel::Game& game, const std::vector<Action>& opening,
            MCTSBot* a, MCTSBot* b, int a_player) {
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  for (Action action : opening) state->ApplyAction(action);
  int moves = opening.size();
  while (!state->IsTerminal()) {
    MCTSBot* bot = state->CurrentPlayer() == a_player ? a : b;
    std::unique_ptr<SearchNode> root = bot->MCTSearch(*state);
    simulations_done += root->explore_count;
    state->ApplyAction(root->BestChild().action);
    ++moves;
  }
  return {state->Returns()[a_player], moves};
}

}  // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(args.Get("game", "thud"));
  const std::string spec_a = args.Get("a", ""), spec_b = args.Get("b", "");
  const int sims = args.GetInt("sims", 100);
  const int sims_a = args.GetInt("sims_a", sims), sims_b = args.GetInt("sims_b", sims);
  const int pairs = args.GetInt("pairs", 50);
  const int opening_moves = args.GetInt("opening", 4);
  const int threads = args.GetInt("threads", 16);
  const int batch = args.GetInt("batch", 16);
  const int inference_threads = args.GetInt("inference_threads", 1);
  const int seed = args.GetInt("seed", 1);
  const std::string untried = args.Get("untried", "upstream");
  const std::string untried_a = args.Get("untried_a", untried);
  const std::string untried_b = args.Get("untried_b", untried);
  const open_spiel::thud_az::UntriedMoveValue rule_a =
      open_spiel::thud_az::UntriedMoveValueFromString(untried_a);
  const open_spiel::thud_az::UntriedMoveValue rule_b =
      open_spiel::thud_az::UntriedMoveValueFromString(untried_b);
  const int progress = args.GetInt("progress", 0);
  const int cache_size = args.GetInt("cache", 1 << 18);
  const bool share = args.GetInt("share", 0) != 0;
  args.CheckAllUsed();
  if (spec_a.empty() || spec_b.empty()) {
    std::cerr << "Usage: " << argv[0] << " a=DIR:STEP b=DIR:STEP [key=value ...]"
              << std::endl;
    return 1;
  }

  if (share && spec_a != spec_b) {
    std::cerr << "share=1 needs the same network on both sides" << std::endl;
    return 1;
  }
  DeviceManager devices_a, devices_b;
  auto eval_a =
      LoadEvaluator(*game, spec_a, &devices_a, batch, inference_threads, cache_size);
  auto eval_b = share ? eval_a
                      : LoadEvaluator(*game, spec_b, &devices_b, batch,
                                      inference_threads, cache_size);
  // The requests of both evaluators (one if shared).
  auto requests = [&]() {
    VPNetEvaluator::RequestCounts sum = eval_a->GetRequestCounts();
    if (!share) {
      const VPNetEvaluator::RequestCounts b = eval_b->GetRequestCounts();
      sum.value_hits += b.value_hits;
      sum.value_misses += b.value_misses;
      sum.prior_hits += b.prior_hits;
      sum.prior_misses += b.prior_misses;
    }
    return sum;
  };
  auto percent = [](int64_t hits, int64_t misses) {
    return hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0;
  };

  // The openings, fixed by the seed so that reruns play the same pairs.
  std::vector<std::vector<Action>> openings(pairs);
  std::mt19937 rng(seed);
  for (auto& opening : openings) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int i = 0; i < opening_moves && !state->IsTerminal(); ++i) {
      std::vector<Action> legal = state->LegalActions();
      opening.push_back(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
      state->ApplyAction(opening.back());
    }
  }

  std::atomic<bool> finished{false};
  std::thread reporter;
  if (progress > 0) {
    reporter = std::thread([&]() {
      using Clock = std::chrono::steady_clock;
      const Clock::time_point start = Clock::now();
      Clock::time_point last = start;
      int64_t last_count = 0;
      while (!finished) {
        for (int i = 0; i < progress * 10 && !finished; ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        const Clock::time_point now = Clock::now();
        const int64_t count = simulations_done;
        const VPNetEvaluator::RequestCounts r = requests();
        std::cerr << absl::StrFormat(
                         "progress: %.0f s, %d simulations, %.0f simulations/s, "
                         "values %d (%.1f%% cached), move probabilities %d (%.1f%% "
                         "cached)\n",
                         std::chrono::duration<double>(now - start).count(), count,
                         (count - last_count) /
                             std::chrono::duration<double>(now - last).count(),
                         r.value_hits + r.value_misses,
                         percent(r.value_hits, r.value_misses),
                         r.prior_hits + r.prior_misses,
                         percent(r.prior_hits, r.prior_misses))
                  << std::flush;
        last = now;
        last_count = count;
      }
    });
  }

  std::vector<double> pair_margin(pairs), a_dwarfs(pairs), a_trolls(pairs);
  std::atomic<int> next{0};
  std::mutex out;
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t]() {
      auto make_bot = [&](std::shared_ptr<VPNetEvaluator> eval, int simulations,
                          open_spiel::thud_az::UntriedMoveValue untried_rule) {
        return std::make_unique<MCTSBot>(
            *game, eval, /*uct_c=*/2, simulations, /*max_memory_mb=*/1000, /*solve=*/false,
            /*seed=*/t, /*verbose=*/false,
            open_spiel::thud_az::ChildSelectionPolicy::PUCT, 0, 0,
            /*dont_return_chance_node=*/false, /*max_wall_clock_time=*/-1, untried_rule);
      };
      std::unique_ptr<MCTSBot> a = make_bot(eval_a, sims_a, rule_a),
                               b = make_bot(eval_b, sims_b, rule_b);
      for (int i = next++; i < pairs; i = next++) {
        const Battle as_dwarfs = Play(*game, openings[i], a.get(), b.get(), 0);
        const Battle as_trolls = Play(*game, openings[i], a.get(), b.get(), 1);
        a_dwarfs[i] = as_dwarfs.a_return * kMaxMargin + 0.0;  // + 0.0: no "-0".
        a_trolls[i] = as_trolls.a_return * kMaxMargin + 0.0;
        pair_margin[i] = a_dwarfs[i] + a_trolls[i];
        std::lock_guard<std::mutex> lock(out);
        std::cout << absl::StrFormat(
                         "{\"pair\": %d, \"a_as_dwarfs\": %.0f, \"a_as_trolls\": %.0f, "
                         "\"a_pair_margin\": %.0f, \"moves\": [%d, %d]}",
                         i, a_dwarfs[i], a_trolls[i], pair_margin[i], as_dwarfs.moves,
                         as_trolls.moves)
                  << std::endl;
      }
    });
  }
  for (std::thread& w : workers) w.join();
  finished = true;
  if (reporter.joinable()) reporter.join();

  auto mean = [](const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;
    return sum / v.size();
  };
  const double m = mean(pair_margin);
  double var = 0;
  for (double x : pair_margin) var += (x - m) * (x - m);
  // Student's t for pairs - 1 degrees of freedom (Cornish-Fisher expansion around 1.96;
  // 2.093 for 19). Until 2026-09-28 this used 1.96, slightly too narrow for 20 pairs.
  const double z = 1.959964, df = pairs - 1;
  const double t = df > 0 ? z + (z * z * z + z) / (4 * df) +
                                (5 * std::pow(z, 5) + 16 * z * z * z + 3 * z) / (96 * df * df)
                          : 0;
  const double half_width = pairs > 1 ? t * std::sqrt(var / df) / std::sqrt(pairs) : 0;
  const VPNetEvaluator::RequestCounts r = requests();
  int won = 0, drawn = 0, lost = 0;
  for (double x : pair_margin) (x > 0 ? won : x < 0 ? lost : drawn) += 1;
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"a\": \"%s\", \"b\": \"%s\", "
                   "\"untried_a\": \"%s\", \"untried_b\": \"%s\", "
                   "\"sims_a\": %d, \"sims_b\": %d, "
                   "\"pairs\": %d, \"a_mean_pair_margin\": %.2f, \"ci95\": [%.2f, %.2f], "
                   "\"a_pairs_won\": %d, \"drawn\": %d, \"lost\": %d, "
                   "\"a_mean_as_dwarfs\": %.2f, \"a_mean_as_trolls\": %.2f, "
                   "\"b_mean_as_dwarfs\": %.2f, \"b_mean_as_trolls\": %.2f, "
                   "\"cache\": %d, \"shared\": %s, \"value_requests\": %d, "
                   "\"value_cached\": %.1f, \"prior_requests\": %d, "
                   "\"prior_cached\": %.1f}",
                   spec_a, spec_b, untried_a, untried_b, sims_a, sims_b, pairs, m,
                   m - half_width, m + half_width, won, drawn, lost, mean(a_dwarfs),
                   mean(a_trolls), -mean(a_trolls), -mean(a_dwarfs), cache_size,
                   share ? "true" : "false", r.value_hits + r.value_misses,
                   percent(r.value_hits, r.value_misses), r.prior_hits + r.prior_misses,
                   percent(r.prior_hits, r.prior_misses))
            << std::endl;
  return 0;
}
