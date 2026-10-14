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

// Does a network's value register a dwarf, and where? (thud/DESIGN_dwarf_signal.md, check
// Q5.) A quarter of the replay buffer is our trolls' positions from the training games
// against alpha-beta's dwarfs — games in which dwarfs survive — while our own dwarfs'
// positions come from games in which every dwarf is lost. If the value has learnt a
// dwarf's worth at the first kind (trolls to move) but not at dwarf-to-move positions, the
// gap is by side to move (approach C7 targets it); if at neither, the data are not the
// limit (approach E1).
//
// From a run's actor logs, four kinds of positions: trolls to move and dwarfs to move,
// each from the games against alpha-beta's dwarfs ("Opponent: alphabeta:D as dwarfs")
// and from self-play; every `every`-th ply from ply 10, at most `max` a kind (an even
// sample). In each, one dwarf at least `far` squares (king's moves) from every troll is
// removed — chosen at random, fixed by `seed` — and the value (the dwarfs' return, -1 to
// 1) is compared: dwarf = v(position) - v(one dwarf fewer), a dwarf's worth to the
// value (its material: 1/32 = 0.031). The positive control: a troll as far from every
// dwarf removed, troll = v(one troll fewer) - v(position) (material 4/32 = 0.125). The
// placebo: the same far dwarf moved to an empty neighbouring square as far from every
// troll, shift = v(position) - v(moved) — a change of no material, whose mean should be
// ~0. Each also divided by 1 - v^2 (tanh's slope, "pre-tanh"). (An untrained network is
// no control here: its value does not vary with the input.)
//
//   thud/az/build.sh thud/experiments/az_dwarf_value_probe.cc
//   OMP_NUM_THREADS=4 build-shared/az_dwarf_value_probe logs=RUN_DIR \
//     nets=NAME=DIR:STEP,... [every=7 max=800 far=4 seed=1]
//
// Prints one JSON line per network and kind of position.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
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

// A position and its variants: one far dwarf fewer, one far troll fewer, the far dwarf
// moved one square (the last two empty if there is no such piece or square).
struct Probe {
  std::unique_ptr<State> position, no_dwarf, no_troll, shifted;
};

// The squares of `kind` at least `far` king's moves from every piece of `other`.
std::vector<std::pair<int, int>> FarPieces(const thud::ThudState& state, thud::Cell kind,
                                           thud::Cell other, int far) {
  std::vector<std::pair<int, int>> mine, theirs, out;
  for (int r = 0; r < thud::kBoardSize; ++r) {
    for (int c = 0; c < thud::kBoardSize; ++c) {
      if (!thud::IsOnBoard(r, c)) continue;
      if (state.CellAt(r, c) == kind) mine.push_back({r, c});
      if (state.CellAt(r, c) == other) theirs.push_back({r, c});
    }
  }
  if (mine.size() < 2) return out;  // Never remove a side's last piece.
  for (const auto& [r, c] : mine) {
    bool ok = true;
    for (const auto& [r2, c2] : theirs) {
      if (std::max(std::abs(r - r2), std::abs(c - c2)) < far) ok = false;
    }
    if (ok) out.push_back({r, c});
  }
  return out;
}

// `state` with the piece at (row, col) removed, through the text format.
std::unique_ptr<State> Without(const open_spiel::Game& game, const State& state, int row,
                               int col) {
  std::string text = state.ToString();
  const size_t at = row * (thud::kBoardSize + 1) + col;  // Rows of 15 and a newline.
  SPIEL_CHECK_TRUE(text[at] == 'd' || text[at] == 'T');
  text[at] = '.';
  return game.NewInitialState(text);
}

// `state` with the dwarf at (row, col) moved to an empty neighbouring square at least
// `far` king's moves from every troll, or null if there is none.
std::unique_ptr<State> Shifted(const open_spiel::Game& game, const thud::ThudState& state,
                               int row, int col, int far) {
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const int r = row + dr, c = col + dc;
      if ((dr == 0 && dc == 0) || !thud::IsOnBoard(r, c) ||
          state.CellAt(r, c) != thud::Cell::kEmpty) {
        continue;
      }
      bool ok = true;
      for (int r2 = 0; r2 < thud::kBoardSize && ok; ++r2) {
        for (int c2 = 0; c2 < thud::kBoardSize && ok; ++c2) {
          if (thud::IsOnBoard(r2, c2) && state.CellAt(r2, c2) == thud::Cell::kTroll &&
              std::max(std::abs(r - r2), std::abs(c - c2)) < far) {
            ok = false;
          }
        }
      }
      if (!ok) continue;
      std::string text = state.ToString();
      text[row * (thud::kBoardSize + 1) + col] = '.';
      text[r * (thud::kBoardSize + 1) + c] = 'd';
      return game.NewInitialState(text);
    }
  }
  return nullptr;
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
  const int every = std::stoi(get("every", "7")), max_positions = std::stoi(get("max", "800"));
  const int far = std::stoi(get("far", "4"));
  std::mt19937 rng(std::stoi(get("seed", "1")));
  if (logs.empty() || nets.empty()) {
    std::cerr << "Usage: az_dwarf_value_probe logs=RUN_DIR nets=NAME=DIR:STEP,... [...]"
              << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The positions, by kind.
  const std::vector<std::string> kinds = {"ab_games_trolls_to_move",
                                          "ab_games_dwarfs_to_move",
                                          "selfplay_trolls_to_move",
                                          "selfplay_dwarfs_to_move"};
  std::map<std::string, std::vector<std::unique_ptr<State>>> all;
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(logs)) {
    if (absl::StartsWith(entry.path().filename().string(), "log-actor-")) {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  int games_read[2] = {0, 0};
  for (const std::string& path : paths) {
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
      if (!absl::StrContains(line, "Actions: ")) continue;
      const bool self_play = !absl::StrContains(line, "Opponent:");
      const bool ab_dwarfs = absl::StrContains(line, "Opponent: alphabeta:") &&
                             absl::StrContains(line, " as dwarfs;");
      if (!self_play && !ab_dwarfs) continue;
      ++games_read[self_play];
      const std::string prefix = self_play ? "selfplay_" : "ab_games_";
      std::unique_ptr<State> s = game->NewInitialState();
      int ply = 0;
      for (absl::string_view move :
           absl::StrSplit(line.substr(line.find("Actions: ") + 9), ' ', absl::SkipEmpty())) {
        if (s->IsTerminal()) break;
        if (ply >= 10 && ply % every == 0) {
          all[prefix + (s->CurrentPlayer() == thud::kDwarfPlayer ? "dwarfs_to_move"
                                                                 : "trolls_to_move")]
              .push_back(s->Clone());
        }
        s->ApplyAction(Find(*s, std::string(move)));
        ++ply;
      }
    }
  }
  std::cerr << absl::StrFormat("%d games against alpha-beta's dwarfs, %d self-play\n",
                               games_read[0], games_read[1]);

  std::map<std::string, std::vector<Probe>> probes;
  for (const std::string& kind : kinds) {
    std::vector<std::unique_ptr<State>>& list = all[kind];
    for (size_t i = 0; i < list.size(); ++i) {
      if (list.size() > max_positions &&
          i * static_cast<int64_t>(max_positions) / list.size() ==
              (i + 1) * static_cast<int64_t>(max_positions) / list.size()) {
        continue;
      }
      const auto& state = open_spiel::down_cast<const thud::ThudState&>(*list[i]);
      Probe probe;
      const auto dwarfs = FarPieces(state, thud::Cell::kDwarf, thud::Cell::kTroll, far);
      const auto trolls = FarPieces(state, thud::Cell::kTroll, thud::Cell::kDwarf, far);
      if (dwarfs.empty()) continue;  // The question needs a far dwarf.
      const auto [dr, dc] = dwarfs[std::uniform_int_distribution<int>(
          0, dwarfs.size() - 1)(rng)];
      probe.no_dwarf = Without(*game, state, dr, dc);
      probe.shifted = Shifted(*game, state, dr, dc, far);
      if (!trolls.empty()) {
        const auto [tr, tc] = trolls[std::uniform_int_distribution<int>(
            0, trolls.size() - 1)(rng)];
        probe.no_troll = Without(*game, state, tr, tc);
      }
      probe.position = std::move(list[i]);
      probes[kind].push_back(std::move(probe));
    }
    std::cerr << absl::StrFormat("%s: %d positions, %d with a far dwarf\n", kind,
                                 list.size(), probes[kind].size());
  }

  for (absl::string_view spec : absl::StrSplit(nets, ',', absl::SkipEmpty())) {
    const size_t eq = spec.find('='), colon = spec.rfind(':');
    const std::string name(spec.substr(0, eq));
    const std::string dir(spec.substr(eq + 1, colon - eq - 1));
    const std::string step(spec.substr(colon + 1));
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    if (step != "random") model.LoadCheckpoint(std::stoi(step));
    auto value = [&](const State& s) {
      return model.Inference({{s.LegalActions(), s.ObservationTensor()}})[0].value;
    };
    for (const std::string& kind : kinds) {
      std::vector<double> dwarf, dwarf_pre, troll, troll_pre, shift, v_all;
      for (const Probe& p : probes[kind]) {
        const double v = value(*p.position);
        const double slope = std::max(1 - v * v, 1e-3);
        v_all.push_back(v);
        const double d = v - value(*p.no_dwarf);
        dwarf.push_back(d);
        dwarf_pre.push_back(d / slope);
        if (p.shifted) shift.push_back(v - value(*p.shifted));
        if (p.no_troll) {
          const double t = value(*p.no_troll) - v;
          troll.push_back(t);
          troll_pre.push_back(t / slope);
        }
      }
      auto mean = [](const std::vector<double>& x) -> double {
        double s = 0;
        for (double y : x) s += y;
        return x.empty() ? NAN : s / x.size();
      };
      auto se = [&](const std::vector<double>& x) -> double {
        if (x.size() < 2) return NAN;
        const double m = mean(x);
        double s = 0;
        for (double y : x) s += (y - m) * (y - m);
        return std::sqrt(s / (x.size() - 1) / x.size());
      };
      auto num = [](double x) {
        return std::isnan(x) ? std::string("null") : absl::StrFormat("%.4f", x);
      };
      std::cout << absl::StrFormat(
                       "{\"net\": \"%s\", \"kind\": \"%s\", \"positions\": %d, "
                       "\"v_mean\": %s, \"dwarf\": %s, \"dwarf_se\": %s, "
                       "\"dwarf_pretanh\": %s, \"troll_positions\": %d, \"troll\": %s, "
                       "\"troll_se\": %s, \"troll_pretanh\": %s, \"shift_positions\": %d, "
                       "\"shift\": %s, \"shift_se\": %s}",
                       name, kind, dwarf.size(), num(mean(v_all)), num(mean(dwarf)),
                       num(se(dwarf)), num(mean(dwarf_pre)), troll.size(),
                       num(mean(troll)), num(se(troll)), num(mean(troll_pre)), shift.size(),
                       num(mean(shift)), num(se(shift)))
                << std::endl;
    }
  }
  return 0;
}
