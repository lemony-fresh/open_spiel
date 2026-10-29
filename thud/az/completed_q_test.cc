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

// Tests the completed-Q policy target (thud/az/mcts.h, CompletedQPolicy: Gumbel AlphaZero's,
// Danihelka et al., ICLR 2022), 2026-10-29:
//
//   1. Against mctx (0.0.71, its own functions — _compute_mixed_value, _complete_qvalues,
//      _rescale_qvalues, then softmax(prior logits + the transformed values) — in float64,
//      its defaults value_scale 0.1 and maxvisit_init 50) on five hand-built roots: a typical
//      one, every child unvisited, one visited, two visited with equal values, one child with
//      most of 200 playouts. Every target within 1e-12 of mctx's (the mixed value through
//      its effect: in one_visited it sets the unvisited children's mass).
//   2. Properties: with no child visited the target is the prior exactly; an unvisited
//      child's completed Q is the mixed value (two unvisited children with equal priors get
//      equal mass); the control — value_scale 0.2 instead of 0.1 moves the typical target by
//      more than 1e-3 (so the comparison in 1 can fail).
//   3. Real searches (MCTSBot with random rollouts, 40 simulations, on 6 Thud positions of
//      random games): every root child in the target, the target a distribution (each
//      finite and in [0, 1], their sum 1 within 1e-12), and the most valued visited child
//      (its mean value the highest of those with the most playouts' half or more) given more
//      mass by the target than by the prior.
//   4. The trainer's wiring (CompletedQTarget, thud/az/alpha_zero.h), with a network (NET_DIR
//      STEP: make_test_network's) on positions every 6 plies of a 60-ply game, its searches as
//      self-play's until 2026-10-28 (40 simulations, root noise 0.1 / 0.25, the dwarfs' policy
//      temperature 1.5): the target is CompletedQPolicy with the evaluator's prior and the
//      value for the player to move, exactly; the controls — the root children's priors
//      (noised, tempered) give another target at every position, and the other player's value
//      at most (where unvisited children take the mixed value).
//
//   thud/run_tests.sh completed_q_test   # builds and runs it
//   build-tests/bin/completed_q_test [NET_DIR STEP]   # part 4 only with a network
//
// Exits non-zero on any failure.

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::thud_az::CompletedQPolicy;
using open_spiel::thud_az::CompletedQSettings;
using open_spiel::thud_az::SearchNode;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

struct Case {
  std::string name;
  std::vector<int> visits;
  std::vector<double> q, prior;
  double raw_value, mixed;
  std::vector<double> want;
};

// mctx 0.0.71's results (the script that made them: thud/PLAN.md, Gumbel targets).
const std::vector<Case> kCases = {
    {"typical", {10, 3, 0, 1, 0, 6}, {0.2, -0.1, 0.0, 0.5, 0.0, 0.3},
     {0.3, 0.2, 0.15, 0.05, 0.1, 0.2}, 0.1, 0.16349206349206347,
     {0.1476795107366726, 0.0049016865983862751, 0.051255242840504402, 0.49437037761662089,
      0.034170161893669608, 0.26762302031414614}},
    {"unvisited", {0, 0, 0, 0}, {0.0, 0.0, 0.0, 0.0}, {0.4, 0.3, 0.2, 0.1}, -0.3, -0.3,
     {0.40000000000000002, 0.29999999999999993, 0.20000000000000001, 0.10000000000000003}},
    {"one_visited", {5, 0, 0}, {-0.4, 0.0, 0.0}, {0.2, 0.5, 0.3}, 0.2, -0.3000000000000001,
     {0.0010206500687286308, 0.62436209370704454, 0.37461725622422676}},
    {"equal_q", {4, 4, 0}, {0.25, 0.25, 0.0}, {0.5, 0.25, 0.25}, 0.25, 0.25, {0.5, 0.25, 0.25}},
    {"many_visits", {180, 15, 5, 0, 0}, {0.62, 0.55, 0.1, 0.0, 0.0},
     {0.35, 0.3, 0.15, 0.15, 0.05}, 0.5, 0.49626865671641784,
     {0.96046509215502507, 0.037229933889734584, 4.2240759283484399e-11,
      0.0017287304347496789, 0.00057624347824989329}},
};

// A root with the case's children (actions 100, 101, ...).
SearchNode Root(const Case& c, std::vector<std::pair<Action, double>>* prior) {
  SearchNode root(open_spiel::kInvalidAction, 0, 1);
  for (size_t i = 0; i < c.visits.size(); ++i) {
    SearchNode child(100 + i, 0, c.prior[i]);
    child.explore_count = c.visits[i];
    child.total_reward = c.q[i] * c.visits[i];
    root.children.push_back(child);
    root.explore_count += c.visits[i];
    prior->emplace_back(100 + i, c.prior[i]);
  }
  return root;
}

}  // namespace

double LargestDifference(const std::vector<std::pair<Action, double>>& a,
                         const std::vector<std::pair<Action, double>>& b) {
  if (a.size() != b.size()) return 1;
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, a[i].first == b[i].first ? std::abs(a[i].second - b[i].second) : 1);
  }
  return worst;
}

int main(int argc, char** argv) {
  // 1. Against mctx.
  for (const Case& c : kCases) {
    std::vector<std::pair<Action, double>> prior;
    const SearchNode root = Root(c, &prior);
    const auto got = CompletedQPolicy(root, prior, c.raw_value);
    double worst = 0;
    for (size_t i = 0; i < c.want.size(); ++i) {
      worst = std::max(worst, std::abs(got[i].second - c.want[i]));
      worst = std::max(worst, got[i].first == static_cast<Action>(100 + i) ? 0.0 : 1.0);
    }
    Expect(got.size() == c.want.size() && worst < 1e-12,
           absl::StrCat("1. ", c.name, ": the target as mctx's (largest difference ", worst, ")"));
  }
  // The mixed value, through two unvisited children of equal prior and its effect: in the
  // typical case both unvisited children's completed Q is the mixed value 0.1635, so their
  // targets' ratio is their priors' (0.15 : 0.1).
  {
    std::vector<std::pair<Action, double>> prior;
    const SearchNode root = Root(kCases[0], &prior);
    const auto got = CompletedQPolicy(root, prior, kCases[0].raw_value);
    Expect(std::abs(got[2].second / got[4].second - 1.5) < 1e-12,
           absl::StrCat("2. unvisited children share the mixed value: their targets' ratio ",
                        got[2].second / got[4].second, " (their priors' 1.5)"));
  }
  // 2. Properties and the control.
  {
    std::vector<std::pair<Action, double>> prior;
    const SearchNode root = Root(kCases[1], &prior);
    const auto got = CompletedQPolicy(root, prior, kCases[1].raw_value);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(got[i].second - kCases[1].prior[i]));
    }
    Expect(worst < 1e-15, absl::StrCat("2. no child visited: the target is the prior (", worst, ")"));
  }
  {
    std::vector<std::pair<Action, double>> prior;
    const SearchNode root = Root(kCases[0], &prior);
    CompletedQSettings other;
    other.value_scale = 0.2;
    const auto got = CompletedQPolicy(root, prior, kCases[0].raw_value, other);
    double moved = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      moved = std::max(moved, std::abs(got[i].second - kCases[0].want[i]));
    }
    Expect(moved > 1e-3, absl::StrCat("2. control: value_scale 0.2 moves the typical target by ",
                                      moved, " (more than 1e-3 required)"));
  }
  // 3. Real searches.
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  auto evaluator = std::make_shared<open_spiel::thud_az::RandomRolloutEvaluator>(1, 7);
  std::mt19937 rng(3);
  int positions = 0, valid = 0, sharpened = 0, compared = 0;
  for (int g = 0; g < 6; ++g) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    const int plies = std::uniform_int_distribution<int>(0, 60)(rng);
    for (int k = 0; k < plies && !state->IsTerminal(); ++k) {
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
    if (state->IsTerminal()) continue;
    open_spiel::thud_az::MCTSBot bot(*game, evaluator, /*uct_c=*/2, /*max_simulations=*/40,
                                     /*max_memory_mb=*/100, /*solve=*/false, /*seed=*/g,
                                     /*verbose=*/false);
    std::unique_ptr<SearchNode> root = bot.MCTSearch(*state);
    const open_spiel::ActionsAndProbs prior = evaluator->Prior(*state);
    const auto target = CompletedQPolicy(*root, prior, 0.0);
    ++positions;
    double sum = 0;
    bool ok = target.size() == root->children.size();
    for (size_t i = 0; ok && i < target.size(); ++i) {
      ok &= target[i].first == root->children[i].action && std::isfinite(target[i].second) &&
            target[i].second >= 0 && target[i].second <= 1;
      sum += target[i].second;
    }
    valid += ok && std::abs(sum - 1) < 1e-12;
    // The most valued child among the well visited ones gets more than its prior.
    int most = 0;
    for (const SearchNode& c : root->children) most = std::max(most, c.explore_count);
    int best = -1;
    for (size_t i = 0; i < root->children.size(); ++i) {
      const SearchNode& c = root->children[i];
      if (2 * c.explore_count < most || c.explore_count == 0) continue;
      if (best < 0 || c.total_reward / c.explore_count >
                          root->children[best].total_reward / root->children[best].explore_count) {
        best = i;
      }
    }
    if (best >= 0) {
      ++compared;
      double p = 0;
      for (const auto& [a, pr] : prior) {
        if (a == root->children[best].action) p = pr;
      }
      sharpened += target[best].second > p;
    }
  }
  Expect(positions >= 4 && valid == positions,
         absl::StrCat("3. real searches: ", valid, " of ", positions,
                      " targets distributions over every root child"));
  Expect(compared > 0 && sharpened == compared,
         absl::StrCat("3. the most valued well-visited child gains mass over its prior in ",
                      sharpened, " of ", compared, " searches"));

  // 4. The trainer's wiring, with a network.
  if (argc == 3) {
    namespace az = open_spiel::thud_az::torch_az;
    az::DeviceManager devices;
    az::VPNetModel model(*game, argv[1], "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(std::stoi(argv[2]));
    devices.AddDevice(std::move(model));
    auto eval = std::make_shared<az::VPNetEvaluator>(&devices, 1, 1, 1 << 16, 1);
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    std::mt19937 moves(29);
    int checked = 0, wired = 0, noised_differs = 0, value_differs = 0;
    for (int ply = 0; ply < 60 && !state->IsTerminal(); ++ply) {
      if (ply % 6 == 0) {
        open_spiel::thud_az::MCTSBot bot(
            *game, eval, /*uct_c=*/2, /*max_simulations=*/40, /*max_memory_mb=*/1000,
            /*solve=*/false, 11 * ply, /*verbose=*/false,
            open_spiel::thud_az::ChildSelectionPolicy::PUCT, 0.1, 0.25, true, -1,
            open_spiel::thud_az::UntriedMoveValue::kSiblingMeanMinusReduction, 0.2, 0);
        bot.SetPolicyTemperatures({1.5, 1.0});
        std::unique_ptr<SearchNode> root = bot.MCTSearch(*state);
        const open_spiel::Player player = state->CurrentPlayer();
        const auto got = az::CompletedQTarget(*root, *state, eval.get());
        const auto want =
            CompletedQPolicy(*root, eval->Prior(*state), eval->Evaluate(*state)[player]);
        std::vector<std::pair<Action, double>> noised;
        for (const SearchNode& c : root->children) noised.emplace_back(c.action, c.prior);
        ++checked;
        wired += LargestDifference(got, want) == 0;
        noised_differs +=
            LargestDifference(got, CompletedQPolicy(*root, noised,
                                                    eval->Evaluate(*state)[player])) > 1e-6;
        value_differs +=
            LargestDifference(got, CompletedQPolicy(*root, eval->Prior(*state),
                                                    eval->Evaluate(*state)[1 - player])) > 1e-9;
      }
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(moves)]);
    }
    Expect(checked >= 8 && wired == checked,
           absl::StrCat("4. the trainer's target is CompletedQPolicy with the network's prior and "
                        "the mover's value at ", wired, " of ", checked, " positions"));
    Expect(noised_differs == checked,
           absl::StrCat("4. control: the root's noised, tempered priors give another target at ",
                        noised_differs, " of ", checked));
    Expect(2 * value_differs >= checked,
           absl::StrCat("4. control: the other player's value gives another target at ",
                        value_differs, " of ", checked, " (at least half required)"));
  } else {
    std::cout << "skip    4. no network given (NET_DIR STEP)" << std::endl;
  }

  std::cout << (failures ? "COMPLETED-Q TEST FAILED" : "completed-q test passed") << std::endl;
  return failures ? 1 : 0;
}
