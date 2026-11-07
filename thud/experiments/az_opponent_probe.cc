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

// The training opponent, plain against finer (thud/PLAN.md, *The teacher's finer evaluation*,
// the opponent's switch): how many moves it chooses among, and what a move costs. The finer
// evaluation breaks most of the plain teacher's ties for its best score, so the finer opponent
// draws its move among fewer moves and its games are less varied; its threat term makes its
// search slower. On an even sample of the positions where the opponent was to move in a run's
// games against alpha-beta (its actor logs, "Opponent: alphabeta:D as SIDE"), replayed in the
// training's game (no-capture limit 100), the opponent as the trainer has it (the teacher as a
// player, AlphaBetaPlayer::ChooseByScores at `depths`) plain and with the finer evaluation
// (the threat term and the positional terms, as az_match's b=teacher:N takes them), per side:
//
//   tied       the moves tied for its best score (mean, median);
//   effective  the effective number of its draw among them (exp of the draw's entropy; the
//              tied count when the draw is even, as the plain opponent's) (mean, median);
//   single     the share of positions where it had a single move to choose;
//   below2     the share where its draw had fewer than 2 effective moves;
//   ms         its time a move (ChooseByScores; one transposition table a setting, kept from
//              position to position as in the trainer, but the positions are spread over many
//              games, so it is reused less than in one game — both settings alike);
// and the finer against the plain, paired over the positions with 95% intervals (the positions
// resampled): ms_ratio (the ratio of the mean times), effective_difference, and
// finer_in_plain_best, the share of positions where the finer opponent's move is one the
// plain one scores best. One JSON line per side and setting, and one per side comparing them.
//
//   thud/build_programs.sh az_opponent_probe
//   build-shared/az_opponent_probe logs=RUN_DIR [last=8 max=300 depths=5,5 sides=both
//     teacher_threat=captures teacher_threat_weights=0.25,0 teacher_terms=0.104,-0.197
//     teacher_terms_temperature=0.25 game=thud(max_turns_without_capture=100) seed=1]
//
// `last`: the most recent games against alpha-beta of each logged actor; `max`: positions a
// side; `depths`: the opponent's depth as the dwarfs and as the trolls (the trainer's
// alphabeta_depth); `sides`: both, dwarfs or trolls (two processes, one a side, run in
// parallel). teacher_threat=off and no teacher_terms: the finer setting is the plain one.

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
#include "thud/az/alphabeta_player.h"
#include "thud/az/material_search.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
using open_spiel::thud_az::Threat;

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(state.CurrentPlayer(), a) == move) return a;
  }
  open_spiel::SpielFatalError("az_opponent_probe: no legal move " + move + " in\n" +
                              state.ToString());
}

std::vector<double> Numbers(const std::string& text) {
  std::vector<double> out;
  for (absl::string_view x : absl::StrSplit(text, ',', absl::SkipEmpty())) {
    out.push_back(std::stod(std::string(x)));
  }
  return out;
}

double Mean(const std::vector<double>& x) {
  double m = 0;
  for (double v : x) m += v / x.size();
  return m;
}

double Median(std::vector<double> x) {
  std::sort(x.begin(), x.end());
  const size_t n = x.size();
  return n % 2 == 1 ? x[n / 2] : (x[n / 2 - 1] + x[n / 2]) / 2;
}

double Share(const std::vector<double>& x, double below) {
  double n = 0;
  for (double v : x) n += v < below;
  return n / x.size();
}

// A statistic of paired samples (a[i], b[i]) and a 95% interval of it, the pairs resampled.
template <typename F>
std::array<double, 3> PairedInterval(const std::vector<double>& a, const std::vector<double>& b,
                                     F statistic, std::mt19937* rng) {
  std::vector<double> draws;
  std::vector<double> ra(a.size()), rb(b.size());
  for (int k = 0; k < 2000; ++k) {
    for (size_t i = 0; i < a.size(); ++i) {
      const size_t j = (*rng)() % a.size();
      ra[i] = a[j];
      rb[i] = b[j];
    }
    draws.push_back(statistic(ra, rb));
  }
  std::sort(draws.begin(), draws.end());
  return {statistic(a, b), draws[50], draws[1949]};
}

struct Measured {
  std::vector<double> tied, effective, ms;
  std::vector<Action> move;
};

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      std::cerr << "az_opponent_probe: arguments are key=value, not " << arg << std::endl;
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
  const int max_positions = std::stoi(get("max", "300"));
  const std::string sides = get("sides", "both");
  const std::vector<double> depths = Numbers(get("depths", "5,5"));
  const std::string threat_name = get("teacher_threat", "captures");
  const std::vector<double> threat_weights = Numbers(get("teacher_threat_weights", "0.25,0"));
  const std::string terms_text = get("teacher_terms", "0.104,-0.197");
  const std::vector<double> terms = Numbers(terms_text);
  const double terms_temperature = std::stod(get("teacher_terms_temperature", "0.25"));
  const uint64_t seed = std::stoull(get("seed", "1"));
  if (logs.empty() || depths.size() != 2 ||
      (sides != "both" && sides != "dwarfs" && sides != "trolls") ||
      (threat_name != "off" && threat_name != "scan" && threat_name != "captures") ||
      (threat_name == "off") != threat_weights.empty() || threat_weights.size() > 2 ||
      !(terms.empty() || terms.size() == 2)) {
    std::cerr << "Usage: az_opponent_probe logs=RUN_DIR [last= max= depths=D,T "
                 "sides=both|dwarfs|trolls teacher_threat=off|scan|captures "
                 "teacher_threat_weights=WD,WT teacher_terms=REACH,PAIRS "
                 "teacher_terms_temperature= game= seed=]" << std::endl;
    return 1;
  }
  const Threat threat = threat_name == "scan"       ? Threat::kScan
                        : threat_name == "captures" ? Threat::kCaptures
                                                    : Threat::kOff;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(get("game", "thud(max_turns_without_capture=100)"));
  const std::array<int, 2> depth = {static_cast<int>(depths[0]), static_cast<int>(depths[1])};

  // The positions where the opponent was to move, by the side it played, from each actor's
  // last games against alpha-beta; then an even sample of each side's.
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
      if (absl::StrContains(line, "Actions: ") && absl::StrContains(line, "Opponent: alphabeta:")) {
        lines.push_back(line);
      }
    }
    for (int i = std::max<int>(0, lines.size() - last); i < lines.size(); ++i) {
      const int opponent = absl::StrContains(lines[i], " as dwarfs;") ? 0
                           : absl::StrContains(lines[i], " as trolls;") ? 1
                                                                        : -1;
      if (opponent < 0) continue;
      ++games;
      const std::string actions = lines[i].substr(lines[i].find("Actions: ") + 9);
      std::unique_ptr<State> state = game->NewInitialState();
      for (absl::string_view move : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
        if (state->IsTerminal()) break;
        if (state->CurrentPlayer() == opponent) positions[opponent].push_back(state->Clone());
        state->ApplyAction(Find(*state, std::string(move)));
      }
    }
  }
  std::cerr << absl::StrFormat(
      "az_opponent_probe: %d games against alpha-beta, %d dwarf and %d troll positions where "
      "it moved; depths %d,%d; finer: threat %s (%s), positional terms %s at %g\n",
      games, positions[0].size(), positions[1].size(), depth[0], depth[1], threat_name,
      get("teacher_threat_weights", "0.25,0"), terms.empty() ? "off" : terms_text,
      terms_temperature);

  AlphaBetaPlayer plain(*game, depth), finer(*game, depth);
  for (AlphaBetaPlayer* p : {&plain, &finer}) {
    p->SetMode(AlphaBetaPlayer::Mode::kTeacher);
    p->NewGame(seed);
  }
  if (threat != Threat::kOff) {
    finer.SetThreat(threat, {threat_weights[0], threat_weights.back()});
  }
  if (!terms.empty()) finer.SetPositionalTerms({terms[0], terms[1]}, terms_temperature);

  std::mt19937 rng(1);
  for (int side = 0; side < 2; ++side) {
    if (sides != "both" && sides != (side == 0 ? "dwarfs" : "trolls")) continue;
    const auto& all = positions[side];
    const int n = std::min<int>(max_positions, all.size());
    if (n == 0) {
      std::cerr << "az_opponent_probe: no positions for the " << (side == 0 ? "dwarfs" : "trolls")
                << std::endl;
      continue;
    }
    Measured measured[2];  // Plain, finer.
    std::vector<double> in_plain_best;
    for (int k = 0; k < n; ++k) {
      const State& s = *all[static_cast<size_t>(k) * all.size() / n];
      for (int v : {k % 2, 1 - k % 2}) {  // Alternating which goes first.
        AlphaBetaPlayer::Choice choice;
        const auto start = std::chrono::steady_clock::now();
        const Action a = (v == 0 ? plain : finer).ChooseByScores(s, &choice);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();
        measured[v].tied.push_back(choice.tied);
        measured[v].effective.push_back(choice.effective);
        measured[v].ms.push_back(ms);
        measured[v].move.push_back(a);
      }
      // Is the finer opponent's move one the plain one scores best?
      const std::vector<double> scores = plain.Scores(s);  // Every score exact.
      const double best = *std::max_element(scores.begin(), scores.end());
      const std::vector<Action> legal = s.LegalActions();
      const size_t i = std::find(legal.begin(), legal.end(), measured[1].move.back()) -
                       legal.begin();
      in_plain_best.push_back(scores[i] == best ? 1 : 0);
    }
    const char* side_name = side == 0 ? "dwarfs" : "trolls";
    for (int v = 0; v < 2; ++v) {
      const Measured& m = measured[v];
      std::cout << absl::StrFormat(
          "{\"side\": \"%s\", \"setting\": \"%s\", \"positions\": %d, \"tied_mean\": %.3f, "
          "\"tied_median\": %.1f, \"effective_mean\": %.3f, \"effective_median\": %.3f, "
          "\"single\": %.4f, \"below2\": %.4f, \"ms\": %.2f}\n",
          side_name, v == 0 ? "plain" : "finer", n, Mean(m.tied), Median(m.tied),
          Mean(m.effective), Median(m.effective), Share(m.tied, 1.5), Share(m.effective, 2),
          Mean(m.ms));
    }
    const auto ratio = PairedInterval(
        measured[1].ms, measured[0].ms,
        [](const std::vector<double>& a, const std::vector<double>& b) { return Mean(a) / Mean(b); },
        &rng);
    const auto difference = PairedInterval(
        measured[1].effective, measured[0].effective,
        [](const std::vector<double>& a, const std::vector<double>& b) { return Mean(a) - Mean(b); },
        &rng);
    std::cout << absl::StrFormat(
        "{\"side\": \"%s\", \"comparison\": \"finer against plain\", \"positions\": %d, "
        "\"ms_ratio\": [%.3f, %.3f, %.3f], \"effective_difference\": [%.3f, %.3f, %.3f], "
        "\"finer_in_plain_best\": %.4f}\n",
        side_name, n, ratio[0], ratio[1], ratio[2], difference[0], difference[1], difference[2],
        Mean(in_plain_best));
  }
  return 0;
}
