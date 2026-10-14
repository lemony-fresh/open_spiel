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

// Does a network see a dwarf move's risk at its destination, at every distance?
// (thud/PLAN.md, the input planes, as decided: re-evaluate the destination risk — the
// user, 2026-10-11. A move is scored at its origin, but whether its landing square is
// dangerous lies up to 14 squares away for a dwarf, which moves like a queen; the trigger
// asks whether long moves' risks are missed more than short ones'.)
//
// On real positions (the dwarfs to move, from the run's actors' last self-play games),
// every dwarf move that captures nothing is judged exactly, by the game's own move
// generator: is it RISKY — can a troll take the moved dwarf with the very next move? —
// or SAFE. For each distance k (1-14 squares), in each position with both a risky and a
// safe move of distance k, the ratio of the prior's mean probability on the risky ones to
// that on the safe ones; reported per distance as the geometric mean of those ratios over
// positions (below 1: the network avoids the risk; 1: it cannot tell), with the share of
// positions where the risky moves get less, and the counts. Control: an untrained network
// of the same shape (`random` as the step) — ~1 at every distance.
//
// judge=teacher (2026-10-12: "risky" proved too blunt — a dwarf next to a troll may be
// takeable and still the better move, threatening a hurl worth 4): instead of RISKY, a
// move is LOSING if kickstarting's teacher (AlphaBetaPlayer::Scores at depth 2, material
// after the trolls' best reply) scores it below the best non-capturing move's score —
// the risk counted only when it costs material.
//
//   thud/az/build.sh thud/experiments/az_risk_probe.cc
//   OMP_NUM_THREADS=4 build-shared/az_risk_probe logs=RUN_DIR \
//     nets=NAME=DIR:STEP,NAME=DIR:random,... [last=3 max=600 judge=capture|teacher]
//
// `max`: dwarf positions (an even sample). Prints one JSON line per network and distance.

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
#include "thud/az/alphabeta_player.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

struct DwarfMove {
  Action action;
  int k;       // Distance.
  bool risky;  // A troll can take the moved dwarf with the very next move.
};

// The dwarfs' non-capturing moves of `state` (dwarfs to move), each judged: by whether a
// troll can take the moved dwarf next move, or with `teacher`, by whether the teacher
// scores it below the best non-capturing move.
std::vector<DwarfMove> Judge(const State& state,
                             const open_spiel::thud_az::AlphaBetaPlayer* teacher) {
  std::vector<DwarfMove> out;
  const std::vector<Action> legal = state.LegalActions();
  std::vector<double> scores;
  double best = -1e9;
  if (teacher != nullptr) {
    scores = teacher->Scores(state);
    for (int i = 0; i < legal.size(); ++i) {
      if (!absl::EndsWith(state.ActionToString(legal[i]), "x")) best = std::max(best, scores[i]);
    }
  }
  for (int i = 0; i < legal.size(); ++i) {
    const Action a = legal[i];
    const std::string text = state.ActionToString(a);
    if (absl::EndsWith(text, "x")) continue;
    int r0, c0, r1, c1;
    SPIEL_CHECK_EQ(std::sscanf(text.c_str(), "(%d,%d)-(%d,%d)", &r0, &c0, &r1, &c1), 4);
    std::unique_ptr<State> after = state.Child(a);
    bool risky = false;
    if (teacher != nullptr) {
      risky = scores[i] < best - 1e-9;
    } else if (!after->IsTerminal()) {
      for (Action t : after->LegalActions()) {
        if (!absl::EndsWith(after->ActionToString(t), "x")) continue;
        const auto& next =
            open_spiel::down_cast<const thud::ThudState&>(*after->Child(t));
        if (next.CellAt(r1, c1) != thud::Cell::kDwarf) {
          risky = true;
          break;
        }
      }
    }
    out.push_back({a, std::max(std::abs(r1 - r0), std::abs(c1 - c0)), risky});
  }
  return out;
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
  const std::string logs = get("logs", ""), nets = get("nets", "");
  const int last = std::stoi(get("last", "3")), max_positions = std::stoi(get("max", "600"));
  const bool by_teacher = get("judge", "capture") == "teacher";
  if (logs.empty() || nets.empty()) {
    std::cerr << "Usage: az_risk_probe logs=RUN_DIR nets=NAME=DIR:STEP,... [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The dwarfs' positions of each logged actor's last self-play games.
  std::vector<std::unique_ptr<State>> all;
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(logs)) {
    const std::string name = entry.path().filename().string();
    if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  for (const std::string& path : paths) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
      if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
        lines.push_back(line);
      }
    }
    for (int i = std::max<int>(0, lines.size() - last); i < lines.size(); ++i) {
      const std::string actions = lines[i].substr(lines[i].find("Actions: ") + 9);
      std::unique_ptr<State> s = game->NewInitialState();
      for (absl::string_view move : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
        if (s->IsTerminal()) break;
        if (s->CurrentPlayer() == thud::kDwarfPlayer) all.push_back(s->Clone());
        s->ApplyAction(Find(*s, std::string(move)));
      }
    }
  }
  std::vector<const State*> positions;
  for (int i = 0; i < all.size(); ++i) {
    if (all.size() > max_positions &&
        i * static_cast<int64_t>(max_positions) / all.size() ==
            (i + 1) * static_cast<int64_t>(max_positions) / all.size()) {
      continue;
    }
    positions.push_back(all[i].get());
  }
  std::vector<std::vector<DwarfMove>> judged;
  int risky = 0, moves = 0;
  const open_spiel::thud_az::AlphaBetaPlayer teacher(*game, 2);
  for (const State* s : positions) {
    judged.push_back(Judge(*s, by_teacher ? &teacher : nullptr));
    for (const DwarfMove& m : judged.back()) {
      ++moves;
      risky += m.risky;
    }
  }
  std::cerr << absl::StrFormat("%d dwarf positions of %d; %d non-capturing moves, %d %s\n",
                               positions.size(), all.size(), moves, risky,
                               by_teacher ? "losing (the teacher)" : "risky");

  for (absl::string_view spec : absl::StrSplit(nets, ',', absl::SkipEmpty())) {
    const size_t eq = spec.find('='), colon = spec.rfind(':');
    const std::string name(spec.substr(0, eq));
    const std::string dir(spec.substr(eq + 1, colon - eq - 1));
    const std::string step(spec.substr(colon + 1));
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    if (step != "random") model.LoadCheckpoint(std::stoi(step));
    // Per distance: log ratios over positions, positions where risky < safe, counts.
    std::map<int, std::vector<double>> log_ratio;
    std::map<int, int> avoided, n_risky, n_safe;
    for (size_t i = 0; i < positions.size(); ++i) {
      const auto out = model.Inference({{positions[i]->LegalActions(),
                                         positions[i]->ObservationTensor()}});
      std::map<Action, double> prior(out[0].policy.begin(), out[0].policy.end());
      std::map<int, std::pair<double, int>> r, s;  // k -> (sum of prior, count)
      for (const DwarfMove& m : judged[i]) {
        auto& slot = m.risky ? r[m.k] : s[m.k];
        slot.first += prior[m.action];
        slot.second += 1;
        (m.risky ? n_risky : n_safe)[m.k] += 1;
      }
      for (const auto& [k, rs] : r) {
        if (!s.count(k)) continue;
        const double mean_risky = rs.first / rs.second;
        const double mean_safe = s[k].first / s[k].second;
        if (mean_risky <= 0 || mean_safe <= 0) continue;
        log_ratio[k].push_back(std::log(mean_risky / mean_safe));
        avoided[k] += mean_risky < mean_safe;
      }
    }
    for (int k = 1; k <= thud::kMaxDistance; ++k) {
      if (log_ratio[k].empty()) continue;
      double sum = 0;
      for (double x : log_ratio[k]) sum += x;
      std::cout << absl::StrFormat(
                       "{\"net\": \"%s\", \"k\": %d, \"positions\": %d, \"risky_moves\": %d, "
                       "\"safe_moves\": %d, \"risk_ratio\": %.4f, \"avoided_share\": %.3f}",
                       name, k, log_ratio[k].size(), n_risky[k], n_safe[k],
                       std::exp(sum / log_ratio[k].size()),
                       static_cast<double>(avoided[k]) / log_ratio[k].size())
                << std::endl;
    }
  }
  return 0;
}
