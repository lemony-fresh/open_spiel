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

// Do the trolls' training targets ask for captures that their prior does not learn?
// (thud/PLAN.md, input features from the rules: the check that decides between input
// features and kickstarting.)
//
// Takes self-play positions from a run's actor logs (games against alpha-beta left
// out): every troll position with a capture among its legal moves (a capture step or a
// shove, written with an "x"). For each, a network searches it as self-play does —
// our copy of MCTSBot, root noise on (Dirichlet `alpha`, weight `epsilon`), the run's
// rule for untried moves and policy temperature — and reports, summed over the capture
// moves:
//
//   prior  the network's own prior on them (before temperature and noise): what it
//          has learnt;
//   visits their share of the root's visits — the policy target self-play would record:
//          what training asks for;
//   and whether the most visited move is a capture, against the captures' share of the
//   legal moves (what a uniform prior would give them).
//
// A large gap of visits over prior that training has not closed (compare a network
// before and after training on such positions, e.g. G'64 and I83 on run I's positions)
// says the network does not easily represent "this troll move captures": input
// features first. Little visit mass says the targets never asked for it: kickstarting.
//
//   thud/az/build.sh thud/experiments/az_capture_targets.cc
//   OMP_NUM_THREADS=4 build-shared/az_capture_targets a=RUN_DIR:STEP logs=RUN_DIR \
//     [temperature=1.5,1 untried=sibling_mean_minus_reduction sims=100 alpha=0.1 \
//     epsilon=0.25 last=3 max=200 seed=1]
//
// `last`: the most recent self-play games of each logged actor. Prints one JSON line per
// position, then a summary line.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
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

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

bool IsCapture(const State& state, Action a) {
  return absl::EndsWith(state.ActionToString(a), "x");
}

double Mean(const std::vector<double>& v) {
  return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
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
  const std::string temperature = get("temperature", "");
  const int sims = std::stoi(get("sims", "100")), last = std::stoi(get("last", "3"));
  const int max_positions = std::stoi(get("max", "200")), seed = std::stoi(get("seed", "1"));
  const double alpha = std::stod(get("alpha", "0.1")), epsilon = std::stod(get("epsilon", "0.25"));
  if (spec.empty() || logs.empty()) {
    std::cerr << "Usage: az_capture_targets a=DIR:STEP logs=RUN_DIR [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The troll positions with a capture available, from each actor's last games.
  std::vector<std::unique_ptr<State>> positions;
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(logs)) {
    const std::string name = entry.path().filename().string();
    if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  int games = 0, troll_positions = 0;
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
        if (state->CurrentPlayer() == thud::kTrollPlayer) {
          ++troll_positions;
          for (Action a : state->LegalActions()) {
            if (IsCapture(*state, a)) {
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
      "%d self-play games, %d troll positions, %d with a capture available\n", games,
      troll_positions, positions.size());

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

  std::vector<double> prior_mass, visit_share, top_capture, legal_share, prior_top_capture;
  for (int i = 0; i < positions.size(); ++i) {
    if (positions.size() > max_positions &&
        i * max_positions / positions.size() == (i + 1) * max_positions / positions.size()) {
      continue;  // An even sample.
    }
    const State& s = *positions[i];
    double pm = 0, best_p = -1;
    bool best_p_capture = false;
    for (const auto& [a, p] : eval->Prior(s)) {
      const bool capture = IsCapture(s, a);
      if (capture) pm += p;
      if (p > best_p) {
        best_p = p;
        best_p_capture = capture;
      }
    }
    std::unique_ptr<SearchNode> root = bot.MCTSearch(s);
    double visits = 0;
    int captures = 0;
    for (const SearchNode& c : root->children) {
      if (IsCapture(s, c.action)) {
        visits += c.explore_count;
        ++captures;
      }
    }
    const double vs = visits / root->explore_count;
    const bool top = IsCapture(s, root->BestChild().action);
    prior_mass.push_back(pm);
    visit_share.push_back(vs);
    top_capture.push_back(top);
    prior_top_capture.push_back(best_p_capture);
    legal_share.push_back(static_cast<double>(captures) / root->children.size());
    std::cout << absl::StrFormat(
                     "{\"captures\": %d, \"legal\": %d, \"prior_on_captures\": %.4f, "
                     "\"visits_on_captures\": %.4f, \"most_visited_captures\": %d, "
                     "\"prior_top_captures\": %d}",
                     captures, root->children.size(), pm, vs, static_cast<int>(top),
                     static_cast<int>(best_p_capture))
              << std::endl;
  }
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"a\": \"%s\", \"logs\": \"%s\", \"games\": %d, "
                   "\"troll_positions\": %d, \"with_capture\": %d, \"examined\": %d, "
                   "\"sims\": %d, \"prior_on_captures\": %.4f, "
                   "\"visits_on_captures\": %.4f, \"most_visited_is_capture\": %.3f, "
                   "\"prior_top_is_capture\": %.3f, \"captures_share_of_legal\": %.4f}",
                   spec, logs, games, troll_positions, positions.size(), prior_mass.size(),
                   sims, Mean(prior_mass), Mean(visit_share), Mean(top_capture),
                   Mean(prior_top_capture), Mean(legal_share))
            << std::endl;
  return 0;
}
