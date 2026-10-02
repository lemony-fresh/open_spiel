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

// How much better are the training targets of a deeper search? (thud/PLAN.md Phase 6,
// stage 3b: whether playout caps' full searches are worth recording ~7x fewer positions.)
// The trainer's policy target is the root's visit counts. This plays self-play games as
// the trainer does (root noise, the first `drop` moves sampled from the visit counts, then
// the most visited), takes every `every`-th position, and searches each one with root noise
// at each budget in `sims` — the targets the trainer would record — and once more without
// noise at `ref` simulations, the reference. Per side to move, for each budget: how often
// its most visited move is the reference's, the share of its visits on the reference's most
// visited move, and the total variation distance of its visit distribution from the
// reference's (half the summed absolute differences: the share of visits that would have
// to move). Means with 95% intervals over positions, and each budget against the first,
// paired per position. Control: the numbers should improve steadily with the budget.
//
//   thud/az/build.sh thud/experiments/az_target_quality.cc
//   OMP_NUM_THREADS=4 build-shared/az_target_quality a=RUN_DIR:STEP \
//     [sims=100,400,1000 ref=2000 games=40 every=7 drop=10 threads=32 batch=32 seed=1
//      untried=sibling_mean_minus_reduction alpha=0.1 epsilon=0.25]
//
// Prints one JSON line per position, then one summary line per side and budget.

#include <algorithm>
#include <atomic>
#include <cmath>
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
  double GetDouble(const std::string& key, double fallback) {
    return std::stod(Get(key, absl::StrCat(fallback)));
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

// The root's visit counts, normalised: the trainer's policy target.
std::map<Action, double> Visits(const SearchNode& root) {
  std::map<Action, double> visits;
  double total = 0;
  for (const SearchNode& c : root.children) total += c.explore_count;
  for (const SearchNode& c : root.children) {
    if (c.explore_count > 0) visits[c.action] = c.explore_count / total;
  }
  return visits;
}

Action MostVisited(const std::map<Action, double>& v) {
  return std::max_element(v.begin(), v.end(), [](const auto& a, const auto& b) {
           return a.second < b.second;
         })->first;
}

struct Comparison {
  double agrees, mass_on_best, distance;
};

Comparison Compare(const std::map<Action, double>& target,
                   const std::map<Action, double>& ref) {
  const Action best = MostVisited(ref);
  double distance = 0;
  for (const auto& [a, p] : target) {
    auto it = ref.find(a);
    distance += std::abs(p - (it == ref.end() ? 0 : it->second));
  }
  for (const auto& [a, p] : ref) {
    if (!target.count(a)) distance += p;
  }
  auto it = target.find(best);
  return {MostVisited(target) == best ? 1.0 : 0.0, it == target.end() ? 0.0 : it->second,
          distance / 2};
}

struct Stats {
  double mean, low, high;
};

Stats MeanInterval(const std::vector<double>& v) {
  const int n = v.size();
  double m = 0;
  for (double x : v) m += x;
  m /= std::max(n, 1);
  double var = 0;
  for (double x : v) var += (x - m) * (x - m);
  const double half = n > 1 ? 1.96 * std::sqrt(var / (n - 1) / n) : 0;
  return {m, m - half, m + half};
}

}  // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  const std::string spec = args.Get("a", "");
  std::vector<int> sims;
  for (absl::string_view s : absl::StrSplit(args.Get("sims", "100,400,1000"), ',')) {
    int v;
    if (!absl::SimpleAtoi(s, &v)) open_spiel::SpielFatalError("sims=N,N,...");
    sims.push_back(v);
  }
  const int ref_sims = args.GetInt("ref", 2000);
  const int games = args.GetInt("games", 40);
  const int every = args.GetInt("every", 7);
  const int drop = args.GetInt("drop", 10);
  const int threads = args.GetInt("threads", 32);
  const int batch = args.GetInt("batch", 32);
  const int seed = args.GetInt("seed", 1);
  const auto rule = open_spiel::thud_az::UntriedMoveValueFromString(
      args.Get("untried", "sibling_mean_minus_reduction"));
  const double alpha = args.GetDouble("alpha", 0.1);
  const double epsilon = args.GetDouble("epsilon", 0.25);
  args.CheckAllUsed();
  if (every % 2 == 0) {  // The dwarfs move on even turns: an even step samples one side.
    std::cerr << "every must be odd, so that both sides to move are sampled" << std::endl;
    return 1;
  }
  const size_t colon = spec.rfind(':');
  if (colon == std::string::npos) {
    std::cerr << "Usage: " << argv[0] << " a=RUN_DIR:STEP [key=value ...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  DeviceManager devices;
  {
    VPNetModel model(*game, spec.substr(0, colon), "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
    devices.AddDevice(std::move(model));
  }
  auto eval = std::make_shared<VPNetEvaluator>(&devices, batch, /*threads=*/2,
                                               /*cache_size=*/1 << 18, /*shards=*/1);
  auto make_bot = [&](int simulations, bool noise, int bot_seed) {
    return std::make_unique<MCTSBot>(
        *game, eval, /*uct_c=*/2, simulations, /*max_memory_mb=*/1000, /*solve=*/false,
        bot_seed, /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT,
        noise ? alpha : 0, noise ? epsilon : 0, /*dont_return_chance_node=*/true,
        /*max_wall_clock_time=*/-1, rule);
  };

  // Phase 1: self-play games as the trainer plays them; every `every`-th position.
  std::vector<std::unique_ptr<open_spiel::State>> positions;
  std::mutex mu;
  {
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        auto bot = make_bot(sims[0], /*noise=*/true, seed * 1000 + t);
        std::mt19937 rng(seed * 7919 + t);
        for (int g = next++; g < games; g = next++) {
          std::unique_ptr<open_spiel::State> state = game->NewInitialState();
          for (int move = 0; !state->IsTerminal(); ++move) {
            if (move > 0 && move % every == 0) {
              std::lock_guard<std::mutex> lock(mu);
              positions.push_back(state->Clone());
            }
            std::unique_ptr<SearchNode> root = bot->MCTSearch(*state);
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
  std::cerr << positions.size() << " positions from " << games << " games" << std::endl;

  // Phase 2: each position searched at every budget with noise, and as the reference.
  const int k = sims.size();
  std::vector<std::vector<Comparison>> results(positions.size());
  std::vector<int> side(positions.size());
  {
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        std::vector<std::unique_ptr<MCTSBot>> bots;
        for (int j = 0; j < k; ++j) bots.push_back(make_bot(sims[j], true, seed * 31 + t * 7 + j));
        auto ref_bot = make_bot(ref_sims, /*noise=*/false, seed);
        for (int i = next++; i < positions.size(); i = next++) {
          const open_spiel::State& state = *positions[i];
          const std::map<Action, double> ref = Visits(*ref_bot->MCTSearch(state));
          side[i] = state.CurrentPlayer();
          std::string line = absl::StrFormat("{\"position\": %d, \"side\": \"%s\", "
                                             "\"legal\": %d",
                                             i, side[i] == 0 ? "dwarfs" : "trolls",
                                             state.LegalActions().size());
          for (int j = 0; j < k; ++j) {
            const Comparison c = Compare(Visits(*bots[j]->MCTSearch(state)), ref);
            results[i].push_back(c);
            absl::StrAppendFormat(&line,
                                  ", \"s%d\": {\"agrees\": %.0f, \"mass_on_best\": %.4f, "
                                  "\"distance\": %.4f}",
                                  sims[j], c.agrees, c.mass_on_best, c.distance);
          }
          std::lock_guard<std::mutex> lock(mu);
          std::cout << line << "}" << std::endl;
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  // Summaries per side and budget, and each budget against the first, paired.
  for (int s = 0; s < 2; ++s) {
    for (int j = 0; j < k; ++j) {
      std::vector<double> agrees, mass, distance, d_agrees, d_mass, d_distance;
      for (int i = 0; i < positions.size(); ++i) {
        if (side[i] != s) continue;
        const Comparison& c = results[i][j];
        const Comparison& base = results[i][0];
        agrees.push_back(c.agrees);
        mass.push_back(c.mass_on_best);
        distance.push_back(c.distance);
        d_agrees.push_back(c.agrees - base.agrees);
        d_mass.push_back(c.mass_on_best - base.mass_on_best);
        d_distance.push_back(c.distance - base.distance);
      }
      const Stats a = MeanInterval(agrees), m = MeanInterval(mass),
                  d = MeanInterval(distance), da = MeanInterval(d_agrees),
                  dm = MeanInterval(d_mass), dd = MeanInterval(d_distance);
      std::cout << absl::StrFormat(
                       "{\"summary\": true, \"side\": \"%s\", \"sims\": %d, \"ref\": %d, "
                       "\"positions\": %d, \"agrees\": [%.3f, %.3f, %.3f], "
                       "\"mass_on_best\": [%.3f, %.3f, %.3f], \"distance\": [%.3f, %.3f, "
                       "%.3f], \"minus_%d\": {\"agrees\": [%.3f, %.3f, %.3f], "
                       "\"mass_on_best\": [%.3f, %.3f, %.3f], \"distance\": [%.3f, %.3f, "
                       "%.3f]}}",
                       s == 0 ? "dwarfs" : "trolls", sims[j], ref_sims, agrees.size(),
                       a.mean, a.low, a.high, m.mean, m.low, m.high, d.mean, d.low, d.high,
                       sims[0], da.mean, da.low, da.high, dm.mean, dm.low, dm.high,
                       dd.mean, dd.low, dd.high)
                << std::endl;
    }
  }
  return 0;
}
