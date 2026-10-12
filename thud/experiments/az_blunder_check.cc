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

// Where do our dwarfs' blunders come from? (thud/PLAN.md, training against alpha-beta)
//
// Reads an az_match games file (games=PATH) of a network against a player without a
// network and finds the network's blunders as the dwarfs: a move, not a capture, that
// ends next to a troll, after which the opponent's very next move takes that dwarf.
// For each, it asks the network again, searching as it played (our copy of MCTSBot,
// the run's rule for untried moves and policy temperature, no root noise):
//
//   1. the dwarfs' prior: the blunder's probability and rank, whether it is the prior's
//      top move, and the prior's mass on all such moves (non-capturing dwarf moves that
//      end next to a troll) against their share of the legal moves;
//   2. the trolls' prior inside the dwarfs' search: at the node after the blunder, the
//      prior and the visits of the capture the opponent played, and whether the search
//      visited it at all — at `sims` and at `deep_sims` simulations;
//   3. the value head, from the dwarfs' side (its output is player 0's whoever is to
//      move): before the blunder, after it (trolls to move, the capture available) and
//      after the capture, against the material it costs (one dwarf, 1/32 of the
//      return); as a control, its response to removing the troll the dwarf threatened
//      from the position after the blunder (a hurl's worth, 4/32), which a working
//      value head must register.
//
//   thud/az/build.sh thud/experiments/az_blunder_check.cc
//   OMP_NUM_THREADS=4 build-shared/az_blunder_check a=RUN_DIR:STEP \
//     games=~/thud-runs/outside_2026-10-07/NAME_games.jsonl [untried=... \
//     temperature=1.5,1 sims=100 deep_sims=800 max=150]
//
// Prints one JSON line per blunder, then a summary line.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
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

std::map<std::string, std::string> ParseArgs(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value: " + arg);
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  return args;
}

// The action of `state` whose text is `move`.
Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

// The destination square of "(r,c)-(r,c)[x]".
std::pair<int, int> Destination(const std::string& move) {
  std::smatch m;
  static const std::regex pattern(R"(\(\d+,\d+\)-\((\d+),(\d+)\))");
  SPIEL_CHECK_TRUE(std::regex_search(move, m, pattern));
  return {std::stoi(m[1]), std::stoi(m[2])};
}

bool NextToTroll(const State& state, int row, int col) {
  const auto& t = open_spiel::down_cast<const thud::ThudState&>(state);
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

// A dwarf move that captures nothing and ends next to a troll, judged in `before`.
bool Unsafe(const State& before, Action action) {
  const std::string move = before.ActionToString(action);
  if (move.back() == 'x') return false;
  const auto [row, col] = Destination(move);
  std::unique_ptr<State> after = before.Child(action);
  return NextToTroll(*after, row, col);
}

struct Blunder {
  std::unique_ptr<State> before;  // Dwarfs to move.
  Action blunder;
  Action capture;
};

double Mean(const std::vector<double>& v) {
  return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double Median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args = ParseArgs(argc, argv);
  auto get = [&](const std::string& k, const std::string& d) {
    return args.count(k) ? args[k] : d;
  };
  const std::string spec = get("a", ""), games_path = get("games", "");
  const std::string untried = get("untried", "sibling_mean_minus_reduction");
  const std::string temperature = get("temperature", "");
  const int sims = std::stoi(get("sims", "100")), deep_sims = std::stoi(get("deep_sims", "800"));
  const int max_positions = std::stoi(get("max", "150"));
  if (spec.empty() || games_path.empty()) {
    std::cerr << "Usage: az_blunder_check a=DIR:STEP games=PATH [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The blunders, in the order played.
  std::vector<Blunder> blunders;
  int battles = 0, dwarf_moves = 0;
  std::ifstream in(games_path);
  static const std::regex player_re(R"("a_player": (\d))");
  for (std::string line; std::getline(in, line);) {
    std::smatch m;
    if (!std::regex_search(line, m, player_re) || m[1] != "0") continue;  // As the dwarfs.
    ++battles;
    const size_t start = line.find("\"actions\": [\"") + 13;
    const size_t end = line.rfind("\"]");
    const std::vector<std::string> moves =
        absl::StrSplit(line.substr(start, end - start), "\", \"");
    std::unique_ptr<State> state = game->NewInitialState();
    for (int i = 0; i + 1 < moves.size(); ++i) {
      const Action a = Find(*state, moves[i]);
      const bool ours = i >= 4 && state->CurrentPlayer() == thud::kDwarfPlayer;
      if (ours) {
        ++dwarf_moves;
        if (Unsafe(*state, a)) {
          std::unique_ptr<State> after = state->Child(a);
          const Action reply = Find(*after, moves[i + 1]);
          const auto [row, col] = Destination(moves[i]);
          std::unique_ptr<State> replied = after->Child(reply);
          if (open_spiel::down_cast<const thud::ThudState&>(*replied).CellAt(row, col) !=
              thud::Cell::kDwarf) {
            blunders.push_back({state->Clone(), a, reply});
          }
        }
      }
      state->ApplyAction(a);
    }
  }
  std::cerr << absl::StrFormat("%d battles as the dwarfs, %d dwarf moves, %d blunders\n",
                               battles, dwarf_moves, blunders.size());
  // An even sample if there are more than max_positions.
  std::vector<int> chosen;
  for (int i = 0; i < blunders.size(); ++i) {
    if (blunders.size() <= max_positions ||
        i * max_positions / blunders.size() != (i + 1) * max_positions / blunders.size()) {
      chosen.push_back(i);
    }
  }

  DeviceManager devices;
  const size_t colon = spec.rfind(':');
  VPNetModel model(*game, spec.substr(0, colon), "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
  devices.AddDevice(std::move(model));
  auto eval = std::make_shared<VPNetEvaluator>(&devices, /*batch=*/1, /*threads=*/1,
                                               /*cache=*/1 << 16, /*cache_shards=*/1);
  auto make_bot = [&](int simulations) {
    auto bot = std::make_unique<MCTSBot>(
        *game, eval, /*uct_c=*/2, simulations, /*max_memory_mb=*/1000, /*solve=*/false,
        /*seed=*/0, /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT, 0,
        0, /*dont_return_chance_node=*/false, /*max_wall_clock_time=*/-1,
        open_spiel::thud_az::UntriedMoveValueFromString(untried));
    bot->SetPolicyTemperatures(
        open_spiel::thud_az::torch_az::PerPlayerValues(temperature));
    return bot;
  };
  std::unique_ptr<MCTSBot> bot = make_bot(sims), deep_bot = make_bot(deep_sims);

  std::vector<double> p_blunder, rank, top, unsafe_mass, unsafe_share, chosen_again,
      chosen_deep, cap_prior, cap_visited, cap_visit_share, cap_visited_deep, q_blunder,
      v_before, v_after, v_captured, v_drop, v_troll_removed;
  for (int i : chosen) {
    const Blunder& b = blunders[i];
    const State& s = *b.before;
    // 1. The dwarfs' prior (the network's own, before any temperature).
    const open_spiel::ActionsAndProbs prior = eval->Prior(s);
    double pb = 0, mass = 0;
    int unsafe = 0, better = 0;
    for (const auto& [a, p] : prior) {
      if (a == b.blunder) pb = p;
    }
    for (const auto& [a, p] : prior) {
      if (p > pb) ++better;
      if (Unsafe(s, a)) {
        mass += p;
        ++unsafe;
      }
    }
    // 2. The search as played, and deeper.
    auto examine = [&](MCTSBot* searcher, double* chose, double* visited, double* share,
                       double* prior_of_capture, double* q) {
      std::unique_ptr<SearchNode> root = searcher->MCTSearch(s);
      *chose = root->BestChild().action == b.blunder;
      for (const SearchNode& c : root->children) {
        if (c.action != b.blunder) continue;
        if (q) *q = c.explore_count ? c.total_reward / c.explore_count : NAN;
        *visited = 0;
        *share = 0;
        for (const SearchNode& t : c.children) {
          if (t.action != b.capture) continue;
          if (prior_of_capture) *prior_of_capture = t.prior;
          *visited = t.explore_count > 0;
          *share = c.explore_count > 1 ? static_cast<double>(t.explore_count) /
                                             (c.explore_count - 1)
                                       : 0;
        }
      }
    };
    double chose = 0, visited = 0, share = 0, pc = NAN, q = NAN, chose_d = 0,
           visited_d = 0, share_d = 0;
    examine(bot.get(), &chose, &visited, &share, &pc, &q);
    examine(deep_bot.get(), &chose_d, &visited_d, &share_d, nullptr, nullptr);
    // 3. The value head, from the dwarfs' side.
    std::unique_ptr<State> after = s.Child(b.blunder);
    std::unique_ptr<State> captured = after->Child(b.capture);
    // A finished battle (the capture took the last dwarf) is worth its result.
    auto value = [&](const State& v) {
      return v.IsTerminal() ? v.Returns()[0] : eval->Evaluate(v)[0];
    };
    const double vb = value(s), va = value(*after), vc = value(*captured);
    // Control: the position after the blunder with one troll removed (the one the
    // dwarf threatened: next to the dwarf), the trolls still to move.
    double vt = NAN;
    {
      const auto [row, col] = Destination(s.ActionToString(b.blunder));
      std::string text = after->ToString();
      for (int dr = -1; dr <= 1 && std::isnan(vt); ++dr) {
        for (int dc = -1; dc <= 1 && std::isnan(vt); ++dc) {
          const int r = row + dr, c = col + dc;
          if ((dr || dc) && thud::IsOnBoard(r, c) &&
              open_spiel::down_cast<const thud::ThudState&>(*after).CellAt(r, c) ==
                  thud::Cell::kTroll) {
            text[r * (thud::kBoardSize + 1) + c] = '.';
            vt = value(*game->NewInitialState(text));
          }
        }
      }
    }
    p_blunder.push_back(pb);
    rank.push_back(better + 1);
    top.push_back(better == 0);
    unsafe_mass.push_back(mass);
    unsafe_share.push_back(static_cast<double>(unsafe) / prior.size());
    chosen_again.push_back(chose);
    chosen_deep.push_back(chose_d);
    cap_prior.push_back(pc);
    cap_visited.push_back(visited);
    cap_visit_share.push_back(share);
    cap_visited_deep.push_back(visited_d);
    q_blunder.push_back(q);
    v_before.push_back(vb);
    v_after.push_back(va);
    v_captured.push_back(vc);
    v_drop.push_back(vc - va);
    v_troll_removed.push_back(vt - va);
    std::cout << absl::StrFormat(
                     "{\"blunder\": \"%s\", \"capture\": \"%s\", \"prior\": %.4f, "
                     "\"rank\": %d, \"unsafe_mass\": %.3f, \"unsafe_share\": %.3f, "
                     "\"chosen\": %d, \"chosen_deep\": %d, \"capture_prior\": %.4f, "
                     "\"capture_visited\": %d, \"capture_visited_deep\": %d, "
                     "\"q_blunder\": %.3f, \"v_before\": %.3f, \"v_after\": %.3f, "
                     "\"v_captured\": %.3f, \"v_troll_removed\": %.3f}",
                     s.ActionToString(b.blunder), after->ActionToString(b.capture), pb,
                     better + 1, mass, unsafe_share.back(), static_cast<int>(chose),
                     static_cast<int>(chose_d), pc, static_cast<int>(visited),
                     static_cast<int>(visited_d), q, vb, va, vc, vt)
              << std::endl;
  }
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"a\": \"%s\", \"games\": \"%s\", \"battles\": %d, "
                   "\"dwarf_moves\": %d, \"blunders\": %d, \"examined\": %d, "
                   "\"sims\": %d, \"deep_sims\": %d, "
                   "\"prior_blunder_median\": %.4f, \"prior_rank_median\": %.0f, "
                   "\"prior_top_share\": %.3f, \"unsafe_prior_mass_mean\": %.3f, "
                   "\"unsafe_legal_share_mean\": %.3f, \"chosen_again\": %.3f, "
                   "\"chosen_deep\": %.3f, \"capture_prior_median\": %.4f, "
                   "\"capture_visited\": %.3f, \"capture_visit_share_mean\": %.3f, "
                   "\"capture_visited_deep\": %.3f, \"q_blunder_mean\": %.3f, "
                   "\"v_before_mean\": %.3f, \"v_after_mean\": %.3f, "
                   "\"v_captured_mean\": %.3f, \"v_drop_mean\": %.3f, "
                   "\"v_drop_median\": %.3f, \"v_troll_removed_mean\": %.3f, "
                   "\"one_dwarf\": %.4f, \"one_troll\": %.4f}",
                   spec, games_path, battles, dwarf_moves, blunders.size(), chosen.size(),
                   sims, deep_sims, Median(p_blunder), Median(rank), Mean(top),
                   Mean(unsafe_mass), Mean(unsafe_share), Mean(chosen_again),
                   Mean(chosen_deep), Median(cap_prior), Mean(cap_visited),
                   Mean(cap_visit_share), Mean(cap_visited_deep), Mean(q_blunder),
                   Mean(v_before), Mean(v_after), Mean(v_captured), Mean(v_drop),
                   Median(v_drop), Mean(v_troll_removed), 1.0 / thud::kMaxMargin,
                   static_cast<double>(thud::kTrollValue) / thud::kMaxMargin)
            << std::endl;
  return 0;
}
