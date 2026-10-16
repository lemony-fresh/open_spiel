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

// The taper's condition (c) (thud/DESIGN_dwarf_signal.md, section 5; the arithmetic in
// thud/experiments/unsafe_targets.h): does the dwarfs' search amplify unsafe moves? From a
// run's actor logs, the dwarf positions of self-play games (games against alpha-beta left
// out; the last `last` games of each logged actor) with an unsafe move among the legal
// ones; each searched as self-play searches the dwarfs (our copy of MCTSBot, `sims`, the
// run's rule for untried moves and policy temperature, root noise as given: run N's
// dwarfs have none, epsilon=0, the default). Reports per position the network's prior
// mass on the unsafe moves and the search's share of visits on them; then R (their means'
// ratio), the upper end of its 95% interval (a bootstrap over the positions), and, for the
// dwarfs' teacher weight `weight`, whether condition (c) allows a step of 0.1: the
// loop's gain after it, (1 - (weight - 0.1)) x R's upper end, at most 0.8.
//
//   thud/az/build.sh thud/experiments/az_unsafe_targets.cc
//   OMP_NUM_THREADS=4 build-shared/az_unsafe_targets a=RUN_DIR:STEP logs=RUN_DIR \
//     [weight=0.7 temperature=1.5,1 untried=sibling_mean_minus_reduction sims=100 \
//     alpha=0.1 epsilon=0 last=3 max=200 seed=1]
//
// Prints one JSON line per position, then a summary line.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"
#include "thud/experiments/unsafe_targets.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;
namespace unsafe = open_spiel::thud_az::unsafe_targets;

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value: " + arg);
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  auto get = [&](const std::string& k, const std::string& d) {
    return args.count(k) ? args[k] : d;
  };
  const std::string spec = get("a", ""), logs = get("logs", "");
  const std::string untried = get("untried", "sibling_mean_minus_reduction");
  const std::string temperature = get("temperature", "1.5,1");
  const int sims = std::stoi(get("sims", "100")), last = std::stoi(get("last", "3"));
  const int max_positions = std::stoi(get("max", "200")), seed = std::stoi(get("seed", "1"));
  const double alpha = std::stod(get("alpha", "0.1")), epsilon = std::stod(get("epsilon", "0"));
  const double weight = std::stod(get("weight", "0.7"));
  if (spec.empty() || logs.empty()) {
    std::cerr << "Usage: az_unsafe_targets a=DIR:STEP logs=RUN_DIR [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The dwarf positions with an unsafe move, from each actor's last self-play games.
  std::vector<std::unique_ptr<State>> positions;
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(logs)) {
    const std::string name = entry.path().filename().string();
    if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  int games = 0, dwarf_positions = 0;
  for (const std::string& path : paths) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
      if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
        lines.push_back(line);
      }
    }
    for (int i = std::max<int>(0, lines.size() - last); i < lines.size(); ++i) {
      ++games;
      const std::string actions = lines[i].substr(lines[i].find("Actions: ") + 9);
      std::unique_ptr<State> state = game->NewInitialState();
      for (absl::string_view move : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
        if (state->IsTerminal()) break;
        if (state->CurrentPlayer() == thud::kDwarfPlayer) {
          ++dwarf_positions;
          for (Action a : state->LegalActions()) {
            if (unsafe::IsUnsafe(*state, a)) {
              positions.push_back(state->Clone());
              break;
            }
          }
        }
        state->ApplyAction(Find(*state, std::string(move)));
      }
    }
  }
  std::cerr << absl::StrFormat(
      "%d self-play games, %d dwarf positions, %d with an unsafe move\n", games,
      dwarf_positions, positions.size());
  if (positions.empty()) open_spiel::SpielFatalError("No dwarf position with an unsafe move");

  DeviceManager devices;
  const size_t colon = spec.rfind(':');
  VPNetModel model(*game, spec.substr(0, colon), "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
  devices.AddDevice(std::move(model));
  auto eval = std::make_shared<VPNetEvaluator>(&devices, /*batch=*/1, /*threads=*/1,
                                               /*cache=*/1 << 16, /*cache_shards=*/1);
  MCTSBot bot(*game, eval, /*uct_c=*/2, sims, /*max_memory_mb=*/1000, /*solve=*/false,
              seed, /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT,
              alpha, epsilon, /*dont_return_chance_node=*/false,
              /*max_wall_clock_time=*/-1,
              open_spiel::thud_az::UntriedMoveValueFromString(untried));
  bot.SetPolicyTemperatures(open_spiel::thud_az::torch_az::PerPlayerValues(temperature));

  std::vector<double> prior_mass, visit_share;
  for (int i = 0; i < positions.size(); ++i) {
    if (positions.size() > max_positions &&
        i * max_positions / positions.size() == (i + 1) * max_positions / positions.size()) {
      continue;  // An even sample.
    }
    const State& s = *positions[i];
    double pm = 0;
    int count = 0;
    for (const auto& [a, p] : eval->Prior(s)) {  // The network's own prior.
      if (unsafe::IsUnsafe(s, a)) {
        pm += p;
        ++count;
      }
    }
    std::unique_ptr<SearchNode> root = bot.MCTSearch(s);
    double visits = 0;
    for (const SearchNode& c : root->children) {
      if (unsafe::IsUnsafe(s, c.action)) visits += c.explore_count;
    }
    const double vs = visits / root->explore_count;
    prior_mass.push_back(pm);
    visit_share.push_back(vs);
    std::cout << absl::StrFormat(
                     "{\"unsafe\": %d, \"legal\": %d, \"prior_on_unsafe\": %.4f, "
                     "\"visits_on_unsafe\": %.4f}",
                     count, root->children.size(), pm, vs)
              << std::endl;
  }
  const double r = unsafe::Ratio(prior_mass, visit_share);
  const double r_upper = unsafe::RatioUpper(prior_mass, visit_share, 2000, seed);
  const double gain = (1 - (weight - 0.1)) * r_upper;
  double mean_prior = 0, mean_visits = 0;
  for (size_t i = 0; i < prior_mass.size(); ++i) {
    mean_prior += prior_mass[i] / prior_mass.size();
    mean_visits += visit_share[i] / prior_mass.size();
  }
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"a\": \"%s\", \"logs\": \"%s\", \"games\": %d, "
                   "\"dwarf_positions\": %d, \"with_unsafe\": %d, \"examined\": %d, "
                   "\"sims\": %d, \"epsilon\": %g, \"prior_on_unsafe\": %.4f, "
                   "\"visits_on_unsafe\": %.4f, \"R\": %.3f, \"R_upper95\": %.3f, "
                   "\"weight\": %g, \"gain_after_step\": %.3f, \"taper_allowed\": %s}",
                   spec, logs, games, dwarf_positions, positions.size(), prior_mass.size(),
                   sims, epsilon, mean_prior, mean_visits, r, r_upper, weight, gain,
                   unsafe::TaperAllowed(weight, r_upper) ? "true" : "false")
            << std::endl;
  return 0;
}
