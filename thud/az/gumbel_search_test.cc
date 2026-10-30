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

// Tests the Gumbel search (thud/az/mcts.h: MCTSBot::SetGumbel and its rules; Gumbel AlphaZero,
// Danihelka et al., ICLR 2022), 2026-10-29:
//
//   1. The halving schedule (GumbelConsideredVisits) equals mctx 0.0.71's
//      get_sequence_of_considered_visits for eight (actions, simulations): (16, 200), (16, 100),
//      (4, 7), (1, 5), (16, 16), (2, 9), (8, 50), (16, 400).
//   2. The root rule (GumbelRootSelect) picks mctx's action (score_considered on its own
//      completed Q, then argmax) on three hand-built roots: the first simulation (no child
//      visited: the Gumbel draw and the prior decide), the second level, the third.
//   3. The rule below the root (GumbelInteriorSelect) picks mctx's action
//      (softmax(logits + completed Q) - N / (1 + sum N), argmax) on three hand-built nodes.
//   4. Searches of Thud positions with a network (NET_DIR STEP; positions every 5 plies of a
//      random 60-ply game, Gumbel searches of 16 actions at 200 and at 100 simulations below the
//      root): the root's playouts sum to the budget; its sorted visit counts are exactly those
//      the schedule implies (each simulation adds one to a child with the considered count,
//      whichever it is); min(16, legal) children visited; the move (GumbelAction) among the
//      most visited. With the Gumbel scale 0 the children visited are the 16 of highest prior.
//      The controls: PUCT's searches of the same positions break the schedule's counts in most;
//      another seed at scale 1 visits another set of children in most (the draws explore).
//
//   thud/run_tests.sh gumbel_search_test   # builds and runs it
//   build-tests/bin/gumbel_search_test [NET_DIR STEP]   # part 4 only with a network
//
// Exits non-zero on any failure.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <set>
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
using open_spiel::thud_az::GumbelConsideredVisits;
using open_spiel::thud_az::GumbelInteriorSelect;
using open_spiel::thud_az::GumbelRootSelect;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

struct Schedule {
  int actions, simulations;
  std::vector<int> want;
};
// mctx 0.0.71's get_sequence_of_considered_visits (the script: thud/PLAN.md, Gumbel).
const std::vector<Schedule> kSchedules = {
    {16, 200, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48}},
    {16, 100, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27}},
    {4, 7, {0, 0, 0, 0, 1, 1, 2}},
    {1, 5, {0, 1, 2, 3, 4}},
    {16, 16, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {2, 9, {0, 0, 1, 1, 2, 2, 3, 3, 4}},
    {8, 50, {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14}},
    {16, 400, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96, 96}},
};

// A node with these children (actions 100, 101, ...).
SearchNode Node(const std::vector<int>& visits, const std::vector<double>& q,
                const std::vector<double>& prior) {
  SearchNode node(open_spiel::kInvalidAction, 0, 1);
  for (size_t i = 0; i < visits.size(); ++i) {
    SearchNode child(100 + i, 0, prior[i]);
    child.explore_count = visits[i];
    child.total_reward = q[i] * visits[i];
    node.children.push_back(child);
    node.explore_count += visits[i];
  }
  return node;
}

// The sorted visit counts the schedule implies: each simulation adds one to a child with the
// considered count (the multiset does not depend on which).
std::vector<int> ImpliedCounts(const std::vector<int>& schedule, int considered) {
  std::vector<int> counts(considered, 0);
  for (int c : schedule) {
    auto it = std::find(counts.begin(), counts.end(), c);
    if (it == counts.end()) return {};  // No child with that count: the schedule is broken.
    ++*it;
  }
  std::sort(counts.begin(), counts.end());
  return counts;
}

std::vector<int> SortedVisited(const SearchNode& root) {
  std::vector<int> counts;
  for (const SearchNode& c : root.children) {
    if (c.explore_count > 0) counts.push_back(c.explore_count);
  }
  std::sort(counts.begin(), counts.end());
  return counts;
}

std::set<Action> Visited(const SearchNode& root) {
  std::set<Action> s;
  for (const SearchNode& c : root.children) {
    if (c.explore_count > 0) s.insert(c.action);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  // 1. The schedule.
  for (const Schedule& s : kSchedules) {
    Expect(GumbelConsideredVisits(s.actions, s.simulations) == s.want,
           absl::StrCat("1. the schedule for ", s.actions, " actions, ", s.simulations,
                        " simulations, as mctx's"));
  }
  // 2. The root rule, against mctx's actions.
  {
    const std::vector<double> prior = {0.3, 0.2, 0.15, 0.05, 0.1, 0.2};
    struct RootCase {
      std::vector<int> visits;
      std::vector<double> q;
      double raw;
      std::vector<double> gumbel;
      int considered, want;
    };
    const std::vector<RootCase> cases = {
        {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, 0.1, {0.5, -0.2, 1.3, 0.0, 2.0, -1.0}, 0, 4},
        {{1, 1, 1, 1, 0, 0}, {0.2, -0.1, 0.4, 0.0, 0, 0}, 0.1, {0.5, -0.2, 1.3, 0.0, 2.0, -1.0}, 1, 2},
        {{3, 3, 1, 1, 0, 0}, {0.2, 0.3, 0.4, 0.0, 0, 0}, 0.0, {0.1, 0.0, 0.9, 0.0, 0.0, 0.0}, 3, 1},
    };
    for (size_t k = 0; k < cases.size(); ++k) {
      const RootCase& c = cases[k];
      const SearchNode root = Node(c.visits, c.q, prior);
      const int got = GumbelRootSelect(root, c.gumbel, c.considered, c.raw);
      Expect(got == c.want, absl::StrCat("2. the root rule, case ", k + 1, ": action ", got,
                                         " (mctx's ", c.want, ")"));
    }
    const SearchNode root = Node({1, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, prior);
    Expect(GumbelRootSelect(root, std::vector<double>(6, 0), 5, 0) == -1,
           "2. no child with the considered count: -1");
  }
  // 3. The rule below the root.
  {
    struct InteriorCase {
      std::vector<int> visits;
      std::vector<double> q, prior;
      double raw;
      int want;
    };
    const std::vector<InteriorCase> cases = {
        {{5, 2, 0, 1}, {0.1, 0.5, 0, -0.3}, {0.4, 0.3, 0.2, 0.1}, 0.0, 1},
        {{0, 0, 0}, {0, 0, 0}, {0.2, 0.5, 0.3}, 0.3, 1},
        {{10, 10, 1}, {0.2, 0.2, 0.9}, {0.45, 0.45, 0.1}, 0.2, 2},
    };
    for (size_t k = 0; k < cases.size(); ++k) {
      const InteriorCase& c = cases[k];
      const int got = GumbelInteriorSelect(Node(c.visits, c.q, c.prior), c.raw);
      Expect(got == c.want, absl::StrCat("3. the rule below the root, case ", k + 1, ": action ",
                                         got, " (mctx's ", c.want, ")"));
    }
  }
  // 4. Searches with a network.
  if (argc == 3) {
    namespace az = open_spiel::thud_az::torch_az;
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
    az::DeviceManager devices;
    az::VPNetModel model(*game, argv[1], "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(std::stoi(argv[2]));
    devices.AddDevice(std::move(model));
    auto eval = std::make_shared<az::VPNetEvaluator>(&devices, 1, 1, 1 << 16, 1);
    auto bot = [&](int simulations, int seed, bool gumbel, double scale) {
      auto b = std::make_unique<MCTSBot>(
          *game, eval, /*uct_c=*/2, simulations, /*max_memory_mb=*/1000, /*solve=*/false, seed,
          /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT, 0.1, 0.25, true,
          -1, open_spiel::thud_az::UntriedMoveValue::kSiblingMeanMinusReduction, 0.2, 0);
      if (gumbel) b->SetGumbel(16, scale);
      return b;
    };
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    std::mt19937 moves(31);
    int searches = 0, budget = 0, structured = 0, breadth = 0, chosen = 0, top_prior = 0,
        puct_broken = 0, other_set = 0;
    for (int ply = 0; ply < 60 && !state->IsTerminal(); ++ply) {
      if (ply % 5 == 0) {
        for (int simulations : {200, 100}) {
          auto g = bot(simulations, 3 * ply + 1, true, 1.0);
          std::unique_ptr<SearchNode> root = g->MCTSearch(*state);
          ++searches;
          int playouts = 0;
          for (const SearchNode& c : root->children) playouts += c.explore_count;
          budget += playouts == simulations;
          const int considered = std::min<int>(16, root->children.size());
          structured +=
              SortedVisited(*root) ==
              ImpliedCounts(GumbelConsideredVisits(considered, simulations), considered);
          breadth += Visited(*root).size() == considered;
          const Action a = g->GumbelAction(*root);
          int most = 0, of_a = -1;
          for (const SearchNode& c : root->children) {
            most = std::max(most, c.explore_count);
            if (c.action == a) of_a = c.explore_count;
          }
          chosen += of_a == most;
          // Scale 0: the children visited are those of highest prior.
          auto z = bot(simulations, 3 * ply + 2, true, 0.0);
          std::unique_ptr<SearchNode> zroot = z->MCTSearch(*state);
          std::vector<std::pair<double, Action>> by_prior;
          for (const SearchNode& c : zroot->children) by_prior.push_back({c.prior, c.action});
          std::sort(by_prior.rbegin(), by_prior.rend());
          std::set<Action> highest;
          for (int i = 0; i < considered; ++i) highest.insert(by_prior[i].second);
          top_prior += Visited(*zroot) == highest;
          // The controls.
          auto p = bot(simulations, 3 * ply + 1, false, 0);
          puct_broken += SortedVisited(*p->MCTSearch(*state)) !=
                         ImpliedCounts(GumbelConsideredVisits(considered, simulations), considered);
          auto other = bot(simulations, 3 * ply + 7, true, 1.0);
          other_set += Visited(*other->MCTSearch(*state)) != Visited(*root);
        }
      }
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(moves)]);
    }
    Expect(searches >= 20 && budget == searches,
           absl::StrCat("4. the root's playouts sum to the budget in ", budget, " of ", searches));
    Expect(structured == searches,
           absl::StrCat("4. the root's visit counts follow the halving schedule exactly in ",
                        structured, " of ", searches));
    Expect(breadth == searches,
           absl::StrCat("4. min(16, legal) children visited in ", breadth, " of ", searches));
    Expect(chosen == searches,
           absl::StrCat("4. the move among the most visited in ", chosen, " of ", searches));
    Expect(top_prior == searches,
           absl::StrCat("4. at Gumbel scale 0 the children of highest prior visited in ", top_prior,
                        " of ", searches));
    Expect(2 * puct_broken > searches,
           absl::StrCat("4. control: PUCT's root counts break the schedule in ", puct_broken,
                        " of ", searches, " (most required)"));
    Expect(2 * other_set > searches,
           absl::StrCat("4. control: another seed visits another set in ", other_set, " of ",
                        searches, " (most required)"));
  } else {
    std::cout << "skip    4. no network given (NET_DIR STEP)" << std::endl;
  }
  std::cout << (failures ? "GUMBEL SEARCH TEST FAILED" : "gumbel search test passed") << std::endl;
  return failures ? 1 : 0;
}
