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

// Does root noise decide what a self-play search records? (thud/PLAN.md, runs G and
// G': the dwarfs' prior stays almost uniform; the hypothesis that the noise, which
// outweighs a near-flat prior, picks the moves a search visits, so the policy target
// records noise.) For each position of an az_reference_pilot output (its "text"),
// and each network: the search the trainer records (100 simulations, root noise
// 0.1 / 0.25, the default rule for untried moves) with two noise seeds, and once
// without noise. Per side: the share of visits on moves the noise boosted (noised
// prior over twice the network's own), the same share for a uniform spread over the
// moves visited (its chance level), how often the two seeds' most visited moves
// agree, and how often each agrees with the noise-free search's. Then the same
// with KataGo's forced playouts and policy target pruning (thud/PLAN.md, policy
// target pruning): for the raw targets (visit counts) and the pruned ones, the
// share of target mass on noise-boosted moves, the targets' effective moves
// (exp of the entropy), and the distance between the two seeds' targets (half the
// summed absolute differences).
//
//   thud/az/build.sh thud/experiments/az_noise_check.cc
//   OMP_NUM_THREADS=2 build-shared/az_noise_check PILOT.jsonl [sims=100] RUN_DIR:STEP
//     [sims=N] [RUN_DIR:STEP ...]
//
// sims=N sets the simulations of every search for the networks after it (default
// 100, the trainer's).

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
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

Action MostVisited(const SearchNode& root) { return root.BestChild().action; }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " PILOT.jsonl RUN_DIR:STEP [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::vector<std::string> texts, sides;
  std::ifstream in(argv[1]);
  for (std::string line; std::getline(in, line);) {
    if (line.empty()) continue;
    open_spiel::json::Object o = open_spiel::json::FromString(line)->GetObject();
    texts.push_back(o["text"].GetString());
    sides.push_back(o["side"].GetString());
  }
  int sims = 100;
  for (int n = 2; n < argc; ++n) {
    const std::string spec = argv[n];
    if (spec.rfind("sims=", 0) == 0) {
      sims = std::stoi(spec.substr(5));
      continue;
    }
    const size_t colon = spec.rfind(':');
    DeviceManager devices;
    VPNetModel model(*game, spec.substr(0, colon), "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
    devices.AddDevice(std::move(model));
    auto eval = std::make_shared<VPNetEvaluator>(&devices, /*batch_size=*/1,
                                                 /*threads=*/1, /*cache_size=*/1 << 16,
                                                 /*shards=*/1);
    auto bot = [&](double epsilon, int seed) {
      return std::make_unique<MCTSBot>(
          *game, eval, /*uct_c=*/2, /*max_simulations=*/sims, /*max_memory_mb=*/1000,
          /*solve=*/false, seed, /*verbose=*/false,
          open_spiel::thud_az::ChildSelectionPolicy::PUCT, /*dirichlet_alpha=*/0.1,
          epsilon, /*dont_return_chance_node=*/true);
    };
    std::map<std::string, std::vector<double>> boosted_share, chance_share, seeds_agree,
        noise_vs_clean, prior_gap;
    // [0] raw targets, [1] pruned (searches with forced playouts).
    std::map<std::string, std::vector<double>> target_boosted[2], target_effective[2],
        target_distance[2];
    auto forced_bot = [&](int seed) {
      return std::make_unique<MCTSBot>(
          *game, eval, /*uct_c=*/2, /*max_simulations=*/sims, /*max_memory_mb=*/1000,
          /*solve=*/false, seed, /*verbose=*/false,
          open_spiel::thud_az::ChildSelectionPolicy::PUCT, /*dirichlet_alpha=*/0.1,
          /*dirichlet_epsilon=*/0.25, /*dont_return_chance_node=*/true, -1,
          open_spiel::thud_az::UntriedMoveValue::kSiblingMeanMinusReduction, 0.2,
          /*forced_playouts_k=*/2);
    };
    for (int i = 0; i < texts.size(); ++i) {
      std::unique_ptr<open_spiel::State> state = game->NewInitialState(texts[i]);
      std::map<Action, double> prior;
      double entropy = 0;
      for (const auto& [a, p] : eval->Prior(*state)) {
        prior[a] = p;
        if (p > 0) entropy -= p * std::log(p);
      }
      prior_gap[sides[i]].push_back(std::log(prior.size()) - entropy);
      Action top[2];
      for (int s = 0; s < 2; ++s) {
        std::unique_ptr<SearchNode> root = bot(0.25, 1000 * n + 7 * i + s)->MCTSearch(*state);
        double visits = 0, on_boosted = 0, visited = 0, boosted_visited = 0;
        for (const SearchNode& c : root->children) {
          const bool boosted = c.prior > 2 * prior[c.action];
          visits += c.explore_count;
          if (boosted) on_boosted += c.explore_count;
          if (c.explore_count > 0) {
            visited += 1;
            boosted_visited += boosted;
          }
        }
        boosted_share[sides[i]].push_back(on_boosted / visits);
        chance_share[sides[i]].push_back(boosted_visited / visited);
        top[s] = MostVisited(*root);
      }
      const Action clean = MostVisited(*bot(0, 1)->MCTSearch(*state));
      seeds_agree[sides[i]].push_back(top[0] == top[1]);
      for (int pruned = 0; pruned < 2; ++pruned) {
        std::map<Action, double> target[2];
        for (int s = 0; s < 2; ++s) {
          std::unique_ptr<SearchNode> root =
              pruned ? forced_bot(5000 * n + 7 * i + s)->MCTSearch(*state)
                     : bot(0.25, 5000 * n + 7 * i + s)->MCTSearch(*state);
          std::map<Action, double> noised;
          for (const SearchNode& c : root->children) noised[c.action] = c.prior;
          double total = 0;
          if (pruned) {
            for (const auto& [a, v] : open_spiel::thud_az::PrunedRootVisits(*root, 2, 2)) {
              target[s][a] = v;
              total += v;
            }
          } else {
            for (const SearchNode& c : root->children) {
              target[s][c.action] = c.explore_count;
              total += c.explore_count;
            }
          }
          double boosted = 0, entropy = 0;
          for (auto& [a, v] : target[s]) {
            v /= total;
            if (noised[a] > 2 * prior[a]) boosted += v;
            if (v > 0) entropy -= v * std::log(v);
          }
          target_boosted[pruned][sides[i]].push_back(boosted);
          target_effective[pruned][sides[i]].push_back(std::exp(entropy));
        }
        double distance = 0;
        for (const auto& [a, v] : target[0]) distance += std::abs(v - target[1][a]);
        target_distance[pruned][sides[i]].push_back(0.5 * distance);
      }
      noise_vs_clean[sides[i]].push_back(0.5 * ((top[0] == clean) + (top[1] == clean)));
    }
    auto mean = [](const std::vector<double>& v) {
      double s = 0;
      for (double x : v) s += x;
      return v.empty() ? 0.0 : s / v.size();
    };
    for (const std::string side : {"dwarfs", "trolls"}) {
      std::cout << absl::StrFormat(
                       "%s @%d %s: %d positions; visits on noise-boosted moves %.0f%% (their "
                       "share of the moves visited %.0f%%); two seeds' most visited "
                       "agree %.0f%%, with the noise-free search %.0f%%; prior below "
                       "uniform by %.3f nats",
                       spec.substr(spec.rfind('/') + 1), sims, side, seeds_agree[side].size(),
                       100 * mean(boosted_share[side]), 100 * mean(chance_share[side]),
                       100 * mean(seeds_agree[side]), 100 * mean(noise_vs_clean[side]),
                       mean(prior_gap[side]))
                << std::endl;
      for (int pruned = 0; pruned < 2; ++pruned) {
        std::cout << absl::StrFormat(
                         "  %s targets: mass on noise-boosted moves %.0f%%, effective "
                         "moves %.1f, distance between two seeds' targets %.3f",
                         pruned ? "pruned" : "raw   ",
                         100 * mean(target_boosted[pruned][side]),
                         mean(target_effective[pruned][side]),
                         mean(target_distance[pruned][side]))
                  << std::endl;
      }
    }
  }
  return 0;
}
