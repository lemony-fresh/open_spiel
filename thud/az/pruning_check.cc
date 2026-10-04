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

// Checks KataGo's forced playouts and policy target pruning in our copy of
// OpenSpiel's MCTS (thud/az/mcts.h: MCTSBot's forced_playouts_k, PrunedRootVisits;
// arXiv 1902.10565, section 3.2; thud/PLAN.md, policy target pruning):
//
//   1. PrunedRootVisits on a hand-built root, against values worked out by hand
//      (uct_c 2, k 2, the root visited 64 times, its children 63): the most visited
//      child keeps its playouts; a child whose PUCT value would pass the best's at
//      once keeps all; a child pruned within its n_forced down to one playout drops
//      to 0; a child stopped by the PUCT bound before its n_forced is used up;
//      unvisited children stay 0.
//   2. Forced playouts in real searches (Thud positions, 100 simulations, root noise
//      0.1 / 0.25, a network): with k 2, root children with playouts but fewer than
//      sqrt(2 P N) are rare — at most about one a search, since a child's n_forced
//      grows with N after its last visit and the final simulations can leave it just
//      short — against clearly more (at least 3x) with k 0 (off, upstream's search),
//      the control. At 100 simulations n_forced is only ~1-4 playouts, so forcing
//      binds for few children; first measured 2026-10-04: 2 of 214 against 18 of
//      232 in 12 searches.
//
//   thud/az/build.sh thud/az/pruning_check.cc
//   build-shared/pruning_check RUN_DIR STEP
//
// Exits non-zero on any failure.

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::PrunedRootVisits;
using open_spiel::thud_az::SearchNode;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

SearchNode Child(Action action, double prior, int playouts, double mean) {
  SearchNode child(action, /*player=*/0, prior);
  child.explore_count = playouts;
  child.total_reward = mean * playouts;
  return child;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " RUN_DIR STEP" << std::endl;
    return 1;
  }
  // 1. By hand. The best child: 0.5 + 2 * 0.2 * sqrt(64) / 51 = 0.5627.
  //    1: n_forced sqrt(2 * 0.3 * 63) = 6.15, but at 9 playouts its PUCT is
  //       0.4 + 2 * 0.3 * 8 / 10 = 0.88 >= 0.5627: keeps 10.
  //    2: n_forced sqrt(2 * 0.05 * 63) = 2.51: at 2, -0.2 + 0.8 / 3 = 0.067; at 1,
  //       -0.2 + 0.8 / 2 = 0.2, both below; a third would exceed 2.51; left with
  //       one playout, it drops to 0.
  //    3: n_forced sqrt(2 * 0.1 * 63) = 3.55 (up to 3): at 7, 0.3 + 1.6 / 8 = 0.5;
  //       at 6, 0.3 + 1.6 / 7 = 0.529; at 5, 0.3 + 1.6 / 6 = 0.567 >= 0.5627: 6.
  //    4: unvisited, 0.
  SearchNode root(open_spiel::kInvalidAction, 0, 1);
  root.explore_count = 64;
  root.children.push_back(Child(10, 0.2, 50, 0.5));
  root.children.push_back(Child(11, 0.3, 10, 0.4));
  root.children.push_back(Child(12, 0.05, 3, -0.2));
  root.children.push_back(Child(13, 0.1, 8, 0.3));
  root.children.push_back(Child(14, 0.35, 0, 0));
  std::map<Action, double> pruned;
  for (const auto& [a, n] : PrunedRootVisits(root, /*uct_c=*/2, /*k=*/2)) pruned[a] = n;
  Expect(pruned[10] == 50, absl::StrCat("the most visited child keeps 50: ", pruned[10]));
  Expect(pruned[11] == 10,
         absl::StrCat("a child at once above the best's PUCT keeps 10: ", pruned[11]));
  Expect(pruned[12] == 0,
         absl::StrCat("a child pruned to one playout drops to 0: ", pruned[12]));
  Expect(pruned[13] == 6,
         absl::StrCat("the PUCT bound stops pruning at 6, before n_forced: ", pruned[13]));
  Expect(pruned[14] == 0, absl::StrCat("an unvisited child stays 0: ", pruned[14]));

  // 2. Real searches.
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  open_spiel::thud_az::torch_az::DeviceManager devices;
  open_spiel::thud_az::torch_az::VPNetModel model(*game, argv[1], "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(std::stoi(argv[2]));
  devices.AddDevice(std::move(model));
  auto eval = std::make_shared<open_spiel::thud_az::torch_az::VPNetEvaluator>(
      &devices, 1, 1, 1 << 16, 1);
  std::mt19937 rng(20261004);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();
  int short_on[2] = {0, 0}, visited_on[2] = {0, 0}, searches = 0;
  for (int move = 0; move < 120 && !state->IsTerminal(); ++move) {
    if (move % 3 == 0) {
      ++searches;
      for (int on = 0; on < 2; ++on) {
        MCTSBot bot(*game, eval, /*uct_c=*/2, /*max_simulations=*/100,
                    /*max_memory_mb=*/1000, /*solve=*/false, 7 * move + on,
                    /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT,
                    0.1, 0.25, true, -1,
                    open_spiel::thud_az::UntriedMoveValue::kSiblingMeanMinusReduction,
                    0.2, /*forced_playouts_k=*/on ? 2 : 0);
        std::unique_ptr<SearchNode> searched = bot.MCTSearch(*state);
        double playouts = 0;
        for (const SearchNode& c : searched->children) playouts += c.explore_count;
        for (const SearchNode& c : searched->children) {
          if (c.explore_count == 0) continue;
          ++visited_on[on];
          if (c.explore_count < std::sqrt(2 * c.prior * playouts)) ++short_on[on];
        }
      }
    }
    std::vector<Action> legal = state->LegalActions();
    state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
  }
  Expect(short_on[1] <= searches && short_on[0] >= 3 * short_on[1] && short_on[0] > 0,
         absl::StrCat("visited root children below n_forced, in ", searches,
                      " searches: ", short_on[1], " of ", visited_on[1],
                      " with forced playouts, ", short_on[0], " of ", visited_on[0],
                      " without (at most one a search with them; control: at least 3x "
                      "more without)"));
  std::cout << (failures ? "PRUNING CHECK FAILED" : "pruning check passed") << std::endl;
  return failures ? 1 : 0;
}
