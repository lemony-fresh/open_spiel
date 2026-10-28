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

// What would kickstarting's teacher ask for? (thud/PLAN.md, kickstarting: choosing its
// temperature and weight.)
//
// Takes self-play positions from a run's actor logs (games against alpha-beta left
// out), an even sample of each side's, and the teacher's policy at each temperature
// (TeacherPolicy: the softmax of AlphaBetaPlayer::Scores at `depth` over T points), and
// reports per side:
//
//   legal       the legal moves;
//   best_share  the share of them with the best score: the teacher's policy at any
//               temperature is uniform over these and puts at least its best mass there;
//   best_mass   the teacher's mass on them, at each T;
//   trolls, at positions with a capture available (a capture step or a shove, written
//   with an "x"; the capture measurement's sample, az_capture_targets.cc): the captures'
//   share of the legal moves, the share of them tied for the best score, and the
//   teacher's mass on them;
//   dwarfs, at positions with an unsafe move (the blunder probe's: one that captures
//   nothing and ends next to a troll, az_blunder_probe.cc): the same for those.
//   ms_a_position  the teacher's search time, its Scores alone, once a position
//               (2026-10-23: the depth's cost, per side; `table_bits`, the search's
//               transposition table, default the teacher's, 0 none);
//   table_differs  with check=1: the positions whose scores differ without the table.
//
// And per side a line with "gaps": true (2026-10-27, for the root window's margin): the root
// moves' share at each gap below the best score (`gap_moves`, 0 to 6+ points), the policy's
// mass there at the first temperature (`gap_mass`), per margin m the share of moves m or more
// points below (`cut_share`) and the bound on how much a root window at m changes the policy
// (`bound_mean`, `bound_max`: those moves' count times e^(-m/T), over the normaliser), and
// the search's nodes a position (`nodes_a_position`, unchanged by the machine's load).
// `margin=M`: the timed search with the root window M; `window_violations` counts the moves
// where it breaks its promise (0 expected), `window_bounds` those it scored by a bound.
// `quick=1`: only the timed search — a line a side with its nodes and time a position.
//
//   thud/az/build.sh thud/experiments/az_teacher_stats.cc
//   build-shared/az_teacher_stats logs=RUN_DIR [depth=2 temperatures=0.25,0.5,1,2 \
//     last=3 max=200]
//
// `last`: the most recent self-play games of each logged actor; `max`: positions a side.
// Prints one summary line a side and temperature.

#include <algorithm>
#include <chrono>
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
#include "thud/az/alphabeta_player.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
using open_spiel::thud_az::torch_az::TeacherPolicy;
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

// A dwarf move that captures nothing and ends next to a troll (az_blunder_probe.cc).
bool Unsafe(const State& before, Action action) {
  if (IsCapture(before, action)) return false;
  const std::string move = before.ActionToString(action);
  const std::vector<std::string> squares = absl::StrSplit(move, '-');
  SPIEL_CHECK_EQ(squares.size(), 2);
  int row, col;
  SPIEL_CHECK_EQ(std::sscanf(squares[1].c_str(), "(%d,%d)", &row, &col), 2);
  std::unique_ptr<State> after = before.Child(action);
  const auto& t = open_spiel::down_cast<const thud::ThudState&>(*after);
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      if ((dr || dc) && thud::IsOnBoard(row + dr, col + dc) &&
          t.CellAt(row + dr, col + dc) == thud::Cell::kTroll) {
        return true;
      }
    }
  }
  return false;
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
  const std::string logs = get("logs", "");
  const int depth = std::stoi(get("depth", "2")), last = std::stoi(get("last", "3"));
  const int max_positions = std::stoi(get("max", "200"));
  // The timed search's transposition table (material_search.h; 0: none), kept from one
  // position to the next as the teacher's is.
  const int table_bits = std::stoi(
      get("table_bits", std::to_string(open_spiel::thud_az::MaterialSearch::kTeacherTableBits)));
  // check=1: every position also searched without a table (a fresh search each), and the
  // positions whose scores differ counted (`table_differs`; 0 expected: the table changes no
  // score, teacher_identity_test part 6).
  const bool check = get("check", "0") == "1";
  // margin=M (2026-10-27): the timed search with the root window M (MaterialSearch::Scores);
  // the statistics then come from an exact search of its own, and every position is checked
  // against the window's promise — the moves less than M below the best exact, the others a
  // bound between their score and best - M (`window_violations`, 0 expected).
  const int margin = std::stoi(get("margin", "0"));
  // quick=1: only the timed search, its nodes and time a position (one line a side, "quick":
  // true) — for deep searches, where the statistics' own exact searches cost the most.
  const bool quick = get("quick", "0") == "1";
  std::vector<double> temperatures;
  for (absl::string_view t : absl::StrSplit(get("temperatures", "0.25,0.5,1,2"), ',')) {
    temperatures.push_back(std::stod(std::string(t)));
  }
  if (logs.empty()) {
    std::cerr << "Usage: az_teacher_stats logs=RUN_DIR [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // Each side's positions from each actor's last self-play games.
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
  std::cerr << absl::StrFormat("%d self-play games, %d dwarf and %d troll positions\n",
                               games, positions[0].size(), positions[1].size());

  const AlphaBetaPlayer teacher(*game, depth);
  open_spiel::thud_az::MaterialSearch search(open_spiel::thud_az::Evaluation::kMaterial,
                                             /*captures_first=*/true, /*killers=*/true,
                                             table_bits);
  for (int side = 0; side < 2; ++side) {
    const auto& all = positions[side];
    std::vector<double> legal, best_share, marked_share, marked_best;
    double search_seconds = 0;  // The teacher's Scores alone, once a position.
    int64_t nodes = 0;          // Its nodes, which the machine's load does not change.
    int window_violations = 0, window_bounds = 0;  // margin > 0: the promise's breaks; bounds.
    // Added 2026-10-27, for the root window's margin: the root moves by their gap below the
    // best score (0, 1, ..., kGaps - 1 or more points), and the policy's mass on them at the
    // first temperature; per margin m, the share of the moves m or more below and the bound on
    // what the window would change in the policy (their count times e^(-m/T), over the
    // policy's normaliser: a fail-low move gets at most that weight), on average and at most.
    constexpr int kGaps = 7, kMargins = 4;
    std::vector<double> gap_moves(kGaps), gap_mass(kGaps);
    double moves_seen = 0;
    std::vector<double> cut_share(kMargins + 1), bound_mean(kMargins + 1), bound_max(kMargins + 1);
    int table_differs = 0;
    // Per temperature.
    std::vector<std::vector<double>> best_mass(temperatures.size()),
        marked_mass(temperatures.size());
    int examined = 0;
    for (int i = 0; i < all.size(); ++i) {
      if (all.size() > max_positions &&
          i * max_positions / all.size() == (i + 1) * max_positions / all.size()) {
        continue;  // An even sample.
      }
      ++examined;
      const State& s = *all[i];
      const std::vector<Action> actions = s.LegalActions();
      const auto start = std::chrono::steady_clock::now();
      search.ResetNodes();
      const std::vector<double> timed = search.Scores(s, depth, margin);
      search_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      nodes += search.nodes();
      if (quick) continue;
      // Equal to the teacher's at margin 0.
      const std::vector<double> scores =
          margin == 0 ? timed
                      : open_spiel::thud_az::MaterialSearch(
                            open_spiel::thud_az::Evaluation::kMaterial)
                            .Scores(s, depth);
      if (margin > 0) {
        const double top = *std::max_element(scores.begin(), scores.end());
        for (size_t k = 0; k < scores.size(); ++k) {
          const bool exact_expected = scores[k] > top - margin;
          window_violations += exact_expected ? timed[k] != scores[k]
                                              : timed[k] < scores[k] || timed[k] > top - margin;
          window_bounds += !exact_expected && timed[k] != scores[k];
        }
      }
      if (check) {
        table_differs +=
            open_spiel::thud_az::MaterialSearch(open_spiel::thud_az::Evaluation::kMaterial)
                .Scores(s, depth) != scores;
      }
      const double best = *std::max_element(scores.begin(), scores.end());
      {
        const double t0 = temperatures[0];
        double z = 0;
        for (double score : scores) z += std::exp((score - best) / t0);
        for (double score : scores) {
          const int gap = std::min<int>(std::lround(best - score), kGaps - 1);
          gap_moves[gap] += 1;
          gap_mass[gap] += std::exp((score - best) / t0) / z;
        }
        moves_seen += scores.size();
        for (int m = 1; m <= kMargins; ++m) {
          int cut = 0;
          for (double score : scores) cut += best - score >= m - 1e-9;
          const double bound = cut * std::exp(-m / t0) / z;
          cut_share[m] += static_cast<double>(cut) / scores.size();
          bound_mean[m] += bound;
          bound_max[m] = std::max(bound_max[m], bound);
        }
      }
      std::map<Action, bool> is_best, marked;
      int n_best = 0, n_marked = 0, n_marked_best = 0;
      for (int k = 0; k < actions.size(); ++k) {
        is_best[actions[k]] = scores[k] >= best - 1e-9;
        n_best += is_best[actions[k]];
        marked[actions[k]] = side == thud::kTrollPlayer ? IsCapture(s, actions[k])
                                                        : Unsafe(s, actions[k]);
        n_marked += marked[actions[k]];
        n_marked_best += marked[actions[k]] && is_best[actions[k]];
      }
      legal.push_back(actions.size());
      best_share.push_back(static_cast<double>(n_best) / actions.size());
      if (n_marked > 0) {
        marked_share.push_back(static_cast<double>(n_marked) / actions.size());
        marked_best.push_back(static_cast<double>(n_marked_best) / n_marked);
      }
      for (int t = 0; t < temperatures.size(); ++t) {
        double on_best = 0, on_marked = 0;
        for (const auto& [a, p] : TeacherPolicy(s, teacher, temperatures[t])) {
          if (is_best[a]) on_best += p;
          if (marked[a]) on_marked += p;
        }
        best_mass[t].push_back(on_best);
        if (n_marked > 0) marked_mass[t].push_back(on_marked);
      }
    }
    if (quick) {
      std::cout << absl::StrFormat(
                       "{\"side\": \"%s\", \"depth\": %d, \"positions\": %d, \"margin\": %d, "
                       "\"nodes_a_position\": %.0f, \"ms_a_position\": %.2f, \"quick\": true}",
                       side == thud::kTrollPlayer ? "trolls" : "dwarfs", depth, examined, margin,
                       static_cast<double>(nodes) / examined, 1000 * search_seconds / examined)
                << std::endl;
      continue;
    }
    const std::string what = side == thud::kTrollPlayer ? "captures" : "unsafe";
    std::string gaps = "\"gap_moves\": [", masses = "\"gap_mass\": [", margins = "\"margins\": [";
    for (int g = 0; g < kGaps; ++g) {
      gaps += absl::StrFormat("%s%.4f", g ? ", " : "", gap_moves[g] / moves_seen);
      masses += absl::StrFormat("%s%.6f", g ? ", " : "", gap_mass[g] / examined);
    }
    for (int m = 1; m <= kMargins; ++m) {
      margins += absl::StrFormat(
          "%s{\"margin\": %d, \"cut_share\": %.4f, \"bound_mean\": %.2e, \"bound_max\": %.2e}",
          m > 1 ? ", " : "", m, cut_share[m] / examined, bound_mean[m] / examined, bound_max[m]);
    }
    std::cout << absl::StrFormat("{\"side\": \"%s\", \"depth\": %d, \"positions\": %d, "
                                 "\"temperature\": %g, %s], %s], %s], \"nodes_a_position\": %.0f, "
                                 "\"ms_a_position\": %.2f, \"margin\": %d, \"window_violations\": %d, "
                                 "\"window_bounds\": %d, \"gaps\": true}",
                                 side == thud::kTrollPlayer ? "trolls" : "dwarfs", depth, examined,
                                 temperatures[0], gaps, masses, margins,
                                 static_cast<double>(nodes) / examined,
                                 1000 * search_seconds / examined, margin, window_violations,
                                 window_bounds)
              << std::endl;
    for (int t = 0; t < temperatures.size(); ++t) {
      std::cout << absl::StrFormat(
                       "{\"side\": \"%s\", \"depth\": %d, \"temperature\": %g, "
                       "\"positions\": %d, \"legal\": %.1f, \"best_share\": %.4f, "
                       "\"best_mass\": %.4f, \"with_%s\": %d, \"%s_share\": %.4f, "
                       "\"%s_tied_for_best\": %.4f, \"%s_mass\": %.4f, \"ms_a_position\": %.2f%s}",
                       side == thud::kTrollPlayer ? "trolls" : "dwarfs", depth,
                       temperatures[t], examined, Mean(legal), Mean(best_share),
                       Mean(best_mass[t]), what, marked_share.size(), what,
                       Mean(marked_share), what, Mean(marked_best), what,
                       Mean(marked_mass[t]), 1000 * search_seconds / examined,
                       check ? absl::StrFormat(", \"table_differs\": %d", table_differs)
                             : std::string())
                << std::endl;
    }
  }
  return 0;
}
