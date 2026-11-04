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

// The teacher's threat term against a search one ply deeper (thud/PLAN.md, *The teacher's
// finer evaluation*, gate 2). The term stands in for the opponent's captures one ply beyond
// the search's horizon, so the search one ply deeper, without it, is what it should agree
// with. On an even sample of each side's self-play positions from a run's actor logs (games
// against alpha-beta left out), replayed in the training's game (no-capture limit 100, so that
// the battle's end near it is as in the trainer), the teacher at its depth for that side to
// move — plain, with Threat::kScan and with Threat::kCaptures at each weight of `weights`
// (the same weight for both sides at the frontier) — against the plain search at that depth
// + 1, per side:
//
//   mae      the root moves' mean |score - the reference's score|, in points;
//   best     the teacher's policy mass (softmax of its scores over `temperature` points, as
//            kickstarting's) on the moves the reference scores best;
//   kl       KL(the reference's policy || the teacher's), the same softmax of each;
//   ms       the teacher's search time a position (on a loaded machine only roughly; the
//            timings that decide are taken at a stop — the plan's gate 3);
// and for best and kl the difference from plain with a 95% interval (the positions
// resampled, paired). One JSON line per side and variant.
//
//   thud/az/build.sh thud/experiments/az_threat_probe.cc
//   build-shared/az_threat_probe logs=RUN_DIR [last=8 max=150 depths=5,4
//     weights=0,0.25,0.5,0.75,1 temperature=0.25 game=thud(max_turns_without_capture=100)
//     sides=both]
//
// `last`: the most recent self-play games of each logged actor; `max`: positions a side;
// `depths`: the teacher's depth for the dwarfs and the trolls to move (the trainer's 5,4;
// with the dwarfs at the frontier in both; 5,5 puts the trolls there for their positions);
// `sides`: both, dwarfs or trolls.

#include <algorithm>
#include <array>
#include <chrono>  // NOLINT
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/material_search.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::MaterialSearch;
using open_spiel::thud_az::Threat;

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(state.CurrentPlayer(), a) == move) return a;
  }
  open_spiel::SpielFatalError("az_threat_probe: no legal move " + move + " in\n" +
                              state.ToString());
}

std::vector<double> Softmax(const std::vector<double>& scores, double temperature) {
  const double top = *std::max_element(scores.begin(), scores.end());
  std::vector<double> p;
  double sum = 0;
  for (double s : scores) {
    p.push_back(std::exp((s - top) / temperature));
    sum += p.back();
  }
  for (double& x : p) x /= sum;
  return p;
}

struct Variant {
  std::string name;
  Threat threat;
  double weight;
};

// The mean of `x` and a 95% interval of it, the entries resampled.
std::array<double, 3> MeanInterval(const std::vector<double>& x, std::mt19937* rng) {
  double mean = 0;
  for (double v : x) mean += v / x.size();
  std::vector<double> draws;
  for (int b = 0; b < 2000; ++b) {
    double m = 0;
    for (size_t i = 0; i < x.size(); ++i) m += x[(*rng)() % x.size()] / x.size();
    draws.push_back(m);
  }
  std::sort(draws.begin(), draws.end());
  return {mean, draws[50], draws[1949]};
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      std::cerr << "az_threat_probe: arguments are key=value, not " << arg << std::endl;
      return 1;
    }
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  auto get = [&args](const std::string& key, const std::string& fallback) {
    auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
  };
  const std::string logs = get("logs", "");
  const int last = std::stoi(get("last", "8"));
  const int max_positions = std::stoi(get("max", "150"));
  const double temperature = std::stod(get("temperature", "0.25"));
  const std::string sides = get("sides", "both");
  if (sides != "both" && sides != "dwarfs" && sides != "trolls") {
    std::cerr << "az_threat_probe: sides=both|dwarfs|trolls" << std::endl;
    return 1;
  }
  std::vector<int> depths;
  for (absl::string_view d : absl::StrSplit(get("depths", "5,4"), ',')) {
    depths.push_back(std::stoi(std::string(d)));
  }
  std::vector<double> weights;
  for (absl::string_view w : absl::StrSplit(get("weights", "0,0.25,0.5,0.75,1"), ',')) {
    weights.push_back(std::stod(std::string(w)));
  }
  if (logs.empty() || depths.size() != 2) {
    std::cerr << "Usage: az_threat_probe logs=RUN_DIR [last= max= depths=D,T weights= "
                 "temperature= game=]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(get("game", "thud(max_turns_without_capture=100)"));

  // Each side's positions from each actor's last self-play games, then an even sample.
  std::vector<std::unique_ptr<State>> positions[2];
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(logs)) {
    const std::string name = entry.path().filename().string();
    if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  int games = 0;
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
        positions[state->CurrentPlayer()].push_back(state->Clone());
        state->ApplyAction(Find(*state, std::string(move)));
      }
    }
  }
  std::cerr << absl::StrFormat("%d self-play games, %d dwarf and %d troll positions\n", games,
                               positions[0].size(), positions[1].size());

  std::vector<Variant> variants = {{"plain", Threat::kOff, 0}};
  for (Threat t : {Threat::kScan, Threat::kCaptures}) {
    for (double w : weights) {
      if (t == Threat::kScan && w == 0) continue;  // The plain search.
      variants.push_back({absl::StrFormat("%s:%g", t == Threat::kScan ? "scan" : "captures", w),
                          t, w});
    }
  }
  std::mt19937 rng(1);
  for (int side = 0; side < 2; ++side) {
    if (sides != "both" && sides != (side == 0 ? "dwarfs" : "trolls")) continue;
    const auto& all = positions[side];
    const int n = std::min<int>(max_positions, all.size());
    const int depth = depths[side];
    MaterialSearch reference(open_spiel::thud_az::Evaluation::kMaterial, true, true,
                             MaterialSearch::kTeacherTableBits);
    std::vector<MaterialSearch> searches;
    searches.reserve(variants.size());
    for (const Variant& v : variants) {
      searches.emplace_back(open_spiel::thud_az::Evaluation::kMaterial, true, true,
                            MaterialSearch::kTeacherTableBits);
      if (v.threat != Threat::kOff) searches.back().SetThreat(v.threat, {v.weight, v.weight});
    }
    // Per variant, per position.
    std::vector<std::vector<double>> mae(variants.size()), best(variants.size()),
        kl(variants.size());
    std::vector<double> seconds(variants.size()), reference_seconds(1);
    for (int k = 0; k < n; ++k) {
      const State& s = *all[static_cast<size_t>(k) * all.size() / n];
      auto start = std::chrono::steady_clock::now();
      const std::vector<double> ref = reference.Scores(s, depth + 1);
      reference_seconds[0] +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      const double ref_best = *std::max_element(ref.begin(), ref.end());
      const std::vector<double> ref_policy = Softmax(ref, temperature);
      for (size_t v = 0; v < variants.size(); ++v) {
        start = std::chrono::steady_clock::now();
        const std::vector<double> got = searches[v].Scores(s, depth);
        seconds[v] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const std::vector<double> policy = Softmax(got, temperature);
        double error = 0, on_best = 0, divergence = 0;
        for (size_t m = 0; m < got.size(); ++m) {
          error += std::abs(got[m] - ref[m]) / got.size();
          if (ref[m] == ref_best) on_best += policy[m];
          if (ref_policy[m] > 0) {
            divergence += ref_policy[m] * std::log(ref_policy[m] / std::max(policy[m], 1e-300));
          }
        }
        mae[v].push_back(error);
        best[v].push_back(on_best);
        kl[v].push_back(divergence);
      }
      if ((k + 1) % 25 == 0) {
        std::cerr << absl::StrFormat("side %d: %d of %d positions\n", side, k + 1, n);
      }
    }
    for (size_t v = 0; v < variants.size(); ++v) {
      std::vector<double> best_gain, kl_gain;
      for (int k = 0; k < n; ++k) {
        best_gain.push_back(best[v][k] - best[0][k]);
        kl_gain.push_back(kl[v][k] - kl[0][k]);
      }
      const auto m = MeanInterval(mae[v], &rng), b = MeanInterval(best[v], &rng),
                 d = MeanInterval(kl[v], &rng), bg = MeanInterval(best_gain, &rng),
                 dg = MeanInterval(kl_gain, &rng);
      std::cout << absl::StrFormat(
                       "{\"side\": \"%s\", \"depth\": %d, \"reference_depth\": %d, "
                       "\"variant\": \"%s\", \"positions\": %d, \"mae\": %.4f, \"best\": %.4f, "
                       "\"kl\": %.4f, \"best_minus_plain\": [%.4f, %.4f, %.4f], "
                       "\"kl_minus_plain\": [%.4f, %.4f, %.4f], \"ms\": %.1f, "
                       "\"reference_ms\": %.1f}",
                       side == 0 ? "dwarfs" : "trolls", depth, depth + 1, variants[v].name, n,
                       m[0], b[0], d[0], bg[0], bg[1], bg[2], dg[0], dg[1], dg[2],
                       1000 * seconds[v] / n, 1000 * reference_seconds[0] / n)
                << std::endl;
    }
  }
  return 0;
}
