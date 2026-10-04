// Copyright 2021 DeepMind Technologies Limited
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

// Changed by the Thud-on-OpenSpiel authors, 2026: copied from
// open_spiel/algorithms/mcts.h at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#ifndef THUD_AZ_MCTS_H_
#define THUD_AZ_MCTS_H_

#include <stdint.h>

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

// A vanilla Monte Carlo Tree Search algorithm.
//
// This algorithm searches the game tree from the given state.
// At the leaf, the evaluator is called if the game state is not terminal.
// A total of max_simulations states are explored.
//
// At every node, the algorithm chooses the action with the highest PUCT value
// defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
// reward after the action, and N is the number of times the action was
// explored in this position. The input parameter c controls the balance
// between exploration and exploitation; higher values of c encourage
// exploration of under-explored nodes. Unseen actions are always explored
// first.
//
// At the end of the search, the chosen action is the action that has been
// explored most often. This is the action that is returned.
//
// This implementation supports sequential n-player games, with or without
// chance nodes. All players maximize their own reward and ignore the other
// players' rewards. This corresponds to max^n for n-player games. It is the
// norm for zero-sum games, but doesn't have any special handling for
// non-zero-sum games. It doesn't have any special handling for imperfect
// information games.
//
// The implementation also supports backing up solved states, i.e. MCTS-Solver.
// The implementation is general in that it is based on a max^n backup (each
// player greedily chooses their maximum among proven children values, or there
// exists one child whose proven value is Game::MaxUtility()), so it will work
// for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
// draw games). Also chance nodes are considered proven only if all children
// have the same value.
//
// Some references:
// - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
//   https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
// - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
//   https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
// - Silver, AlphaGo Zero: Starting from scratch, 2017
//   https://deepmind.com/blog/article/alphago-zero-starting-scratch
// - Winands, Bjornsson, and Saito, Monte-Carlo Tree Search Solver, 2008.
//   https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

namespace open_spiel {
namespace thud_az {

// How PUCT values a child that has not been visited yet ("first-play urgency"). Added
// by the Thud-on-OpenSpiel authors (thud/PLAN.md Phase 6, roadmap stage 3a): upstream
// counts such a child as 0, an even game, so a side whose tried moves are worth less
// spends every simulation on a new move and never looks deeper.
enum class UntriedMoveValue {
  // Upstream OpenSpiel's rule: 0.
  kUpstream,
  // The visit-weighted mean value of the already visited siblings, minus
  // reduction * sqrt(the prior mass of those siblings) — KataGo's rule (arXiv
  // 1902.10565, section 2: V(parent) - c_FPU * sqrt(P_explored), c_FPU = 0.2), with the
  // siblings' mean for the parent's value. While no sibling is visited it is 0; then all
  // untried children tie on it and the prior decides.
  kSiblingMeanMinusReduction,
  // The game's minimum utility, a loss (AlphaZero's rule).
  kLoss,
};

// "upstream", "sibling_mean_minus_reduction" or "loss".
UntriedMoveValue UntriedMoveValueFromString(const std::string& name);

enum class ChildSelectionPolicy {
  UCT,
  PUCT,
};

// Abstract class representing an evaluation function for a game.
// The evaluation function takes in an intermediate state in the game and
// returns an evaluation of that state, which should correlate with chances of
// winning the game for player 0.
class Evaluator {
 public:
  virtual ~Evaluator() = default;

  // Return a value of this state for each player.
  virtual std::vector<double> Evaluate(const State& state) = 0;

  // Return a policy: the probability of the current player playing each action.
  virtual ActionsAndProbs Prior(const State& state) = 0;
};

// A simple evaluator that returns the average outcome of playing random actions
// from the given state until the end of the game.
// n_rollouts is the number of random outcomes to be considered.
class RandomRolloutEvaluator : public Evaluator {
 public:
  explicit RandomRolloutEvaluator(int n_rollouts, int seed)
      : n_rollouts_(n_rollouts), rng_(seed) {}

  // Runs random games, returning the average returns.
  std::vector<double> Evaluate(const State& state) override;

  // Returns equal probability for each action.
  ActionsAndProbs Prior(const State& state) override;

 private:
  int n_rollouts_;
  std::mt19937 rng_;
};

// A node in the search tree for MCTS
struct SearchNode {
  Action action = 0;            // The action taken to get to this node.
  double prior = 0;             // The prior probability of playing this action.
  Player player = 0;            // Which player gets to make this action.
  int explore_count = 0;        // Number of times this node was explored.
  double total_reward = 0;      // Total reward passing through this node.
  std::vector<double> outcome;  // The reward if each players plays perfectly.
  std::vector<SearchNode> children;  // The successors to this state.

  SearchNode() {}

  SearchNode(Action action_, Player player_, double prior_)
      : action(action_), prior(prior_), player(player_) {}

  // The value as returned by the UCT formula.
  double UCTValue(int parent_explore_count, double uct_c) const;

  // The value as returned by the PUCT formula.
  double PUCTValue(int parent_explore_count, double uct_c) const;
  // The same, valuing this child at `untried_value` while it is not visited yet.
  double PUCTValue(int parent_explore_count, double uct_c,
                   double untried_value) const;

  // The sort order for the BestChild.
  bool CompareFinal(const SearchNode& b) const;
  const SearchNode& BestChild() const;

  // Return a string representation of this node, or all its children.
  // The state is needed to convert the action to a string.
  std::string ToString(const State& state) const;
  std::string ChildrenStr(const State& state) const;

  Action SampleFromPrior(const State& state,
                         Evaluator* evaluator,
                         std::mt19937* rng) const;
};

// A SpielBot that uses the MCTS algorithm as its policy.
// Our change (thud/PLAN.md, policy target pruning): KataGo's policy target pruning
// (arXiv 1902.10565, section 3.2). The root's visit counts with the forced
// playouts taken out again: from every child but the most visited (BestChild), up
// to n_forced(c) = sqrt(k P(c) N) playouts are subtracted — P(c) the child's prior
// as the search used it (root noise included), N the sum of the children's
// playouts — as long as the child's PUCT value, its mean value held constant,
// stays below the most visited child's; a child left with one playout gets 0. The
// counts are for the policy target only; the move played is chosen as before.
std::vector<std::pair<Action, double>> PrunedRootVisits(const SearchNode& root,
                                                        double uct_c, double k);

class MCTSBot : public Bot {
 public:
  // The evaluator is passed as a shared pointer to make it explicit that
  // the same evaluator instance can be passed to multiple bots and to
  // make the MCTSBot Python interface work regardless of the scope of the
  // Python evaluator object.
  //
  // TODO(author5): The second parameter needs to be a const reference at the
  // moment, even though it gets assigned to a member of type
  // std::shared_ptr<Evaluator>. This is because using a
  // std::shared_ptr<Evaluator> in the constructor leads to the Julia API test
  // failing. We don't know why right now, but intend to fix this.
  MCTSBot(
      const Game& game, std::shared_ptr<Evaluator> evaluator, double uct_c,
      int max_simulations,
      int64_t max_memory_mb,  // Max memory use in megabytes.
      bool solve,             // Whether to back up solved states.
      int seed, bool verbose,
      ChildSelectionPolicy child_selection_policy = ChildSelectionPolicy::UCT,
      double dirichlet_alpha = 0, double dirichlet_epsilon = 0,
      bool dont_return_chance_node = false, double max_wall_clock_time = -1,
      UntriedMoveValue untried_move_value =
          UntriedMoveValue::kSiblingMeanMinusReduction,
      double untried_move_reduction = 0.2,
      // Our change (thud/PLAN.md, policy target pruning): KataGo's forced
      // playouts, k in n_forced(c) = sqrt(k P(c) sum of the root's playouts); a
      // root child with playouts but fewer than that is chosen next. 0 (the
      // default) turns them off, as upstream.
      double forced_playouts_k = 0);
  ~MCTSBot() = default;

  void Restart() override {}
  void RestartAt(const State& state) override {}
  // Run MCTS for one step, choosing the action, and printing some information.
  Action Step(const State& state) override;

  // The value PUCT gives the unvisited children of `node` (public for tests).
  double UntriedValue(const SearchNode& node) const;

  // Implements StepWithPolicy. This is equivalent to calling Step, but wraps
  // the action as an ActionsAndProbs with 100% probability assigned to the
  // lone action.
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override;

  // Run MCTS on a given state, and return the resulting search tree.
  std::unique_ptr<SearchNode> MCTSearch(const State& state);

 private:
  // Applies the UCT policy to play the game until reaching a leaf node.
  //
  // A leaf node is defined as a node that is terminal or has not been evaluated
  // yet. If it reaches a node that has been evaluated before but hasn't been
  // expanded, then expand it's children and continue.
  //
  // Args:
  //   root: The root node in the search tree.
  //   state: The state of the game at the root node.
  //   visit_path: A vector of nodes to be filled in descending from the root
  //     node to a leaf node.
  //
  // Returns: The state of the game at the leaf node.
  std::unique_ptr<State> ApplyTreePolicy(SearchNode* root, const State& state,
                                         std::vector<SearchNode*>* visit_path);

  void GarbageCollect(SearchNode* node);

  double uct_c_;
  int max_simulations_;
  double max_wall_clock_time_;
  int max_nodes_;  // Max nodes allowed in the tree
  int nodes_;  // Nodes used in the tree.
  int gc_limit_;
  bool verbose_;
  bool solve_;
  double max_utility_;
  double min_utility_;
  UntriedMoveValue untried_move_value_;
  double untried_move_reduction_;
  double forced_playouts_k_;
  double dirichlet_alpha_;
  double dirichlet_epsilon_;
  bool dont_return_chance_node_;
  std::mt19937 rng_;
  const ChildSelectionPolicy child_selection_policy_;
  std::shared_ptr<Evaluator> evaluator_;
};

// Returns a vector of noise sampled from a dirichlet distribution. See:
// https://en.wikipedia.org/wiki/Dirichlet_process
std::vector<double> dirichlet_noise(int count, double alpha, std::mt19937* rng);

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_MCTS_H_
