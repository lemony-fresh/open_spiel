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

#include <functional>
#include <limits>
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
  // Our change (2026-10-29; Gumbel search, MCTSBot::SetGumbel): the evaluator's value of this
  // node for the player to move at it, from its first visit; NaN until then, or if terminal.
  double raw_value = std::numeric_limits<double>::quiet_NaN();

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

// Our change (2026-10-29; the user: Gumbel targets, part one): the policy target from the
// root's search values instead of its visit counts — Gumbel AlphaZero's "completed Q"
// (Danihelka, Guez, Schrittwieser and Silver, "Policy improvement by planning with Gumbel",
// ICLR 2022, appendix D; mctx's qtransform_completed_by_mix_value with its defaults, and
// gumbel_muzero_policy's action_weights): softmax(log prior + sigma(completed Q)), where a
// visited child's Q is its mean value for the player to move at the root, an unvisited
// child's the mixed value — (raw_value + N x the prior-weighted mean Q of the visited
// children) / (N + 1), N the root's children's playouts — the completed Qs rescaled to
// [0, 1] over the children (max - min at least epsilon), and sigma(q) = (maxvisit_init +
// the most playouts of a child) x value_scale x q. `prior` is the network's, before root
// noise and any policy temperature (a missing action gets the smallest positive double);
// `raw_value` the network's value of the root for its player to move. In the order of the
// root's children; matches mctx to ~1e-12 (completed_q_test).
struct CompletedQSettings {
  double value_scale = 0.1, maxvisit_init = 50, epsilon = 1e-8;  // mctx's defaults.
};
// The completed Qs of `node`'s children, transformed as above — sigma(completed Q), mctx's
// qtransform_completed_by_mix_value — with `child_priors` in the order of the children (for
// the mixed value) and `raw_value` the node's own value for the player to move at it.
std::vector<double> TransformedCompletedQ(const SearchNode& node,
                                          const std::vector<double>& child_priors,
                                          double raw_value,
                                          const CompletedQSettings& settings = {});

// Our change (2026-10-29; the user: Gumbel, part two — Danihelka et al., ICLR 2022; mctx's
// gumbel_muzero_policy): the search's rules, free so that they can be tested against mctx.
// The visit counts sequential halving considers at each of `num_simulations` root
// simulations, for `max_considered` sampled actions (mctx's
// get_sequence_of_considered_visits).
std::vector<int> GumbelConsideredVisits(int max_considered, int num_simulations);
// The root child to simulate next (mctx's gumbel_muzero_root_action_selection): among the
// children with `considered_visit` playouts, the highest gumbel + log prior (less its
// maximum) + sigma(completed Q); `gumbel` in the order of the children; their priors as the
// search's (child.prior). Its index; -1 if no child has that count.
int GumbelRootSelect(const SearchNode& root, const std::vector<double>& gumbel,
                     int considered_visit, double raw_value,
                     const CompletedQSettings& settings = {});
// A child below the root (mctx's gumbel_muzero_interior_action_selection): the highest
// softmax(log prior + sigma(completed Q)) - playouts / (1 + the children's playouts), the
// deterministic rule whose visits follow the improved policy. Its index.
int GumbelInteriorSelect(const SearchNode& node, double raw_value,
                         const CompletedQSettings& settings = {});
std::vector<std::pair<Action, double>> CompletedQPolicy(
    const SearchNode& root, const std::vector<std::pair<Action, double>>& prior,
    double raw_value, const CompletedQSettings& settings = {});

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

  // Our change (thud/PLAN.md, training the dwarfs): Leela Chess Zero's policy
  // softmax temperature, per player — at every node where player p is to move, the
  // prior is raised to 1 / temperatures[p] and renormalised before the search uses
  // it (before root noise): above 1 it flattens the prior, keeping its order, so the
  // search spreads over more moves. Empty, or 1 for a player, leaves the prior as it
  // is (upstream's behaviour).
  // Our change (2026-10-29; Gumbel, part two): searches the root by sequential halving over
  // `max_considered` actions sampled by Gumbel-top-k (gumbel noise at `scale`: 1 in self-play,
  // 0 deterministic) and every other node by GumbelInteriorSelect, instead of PUCT; no root
  // noise (the Gumbel draws explore); max_simulations below the root, after the root's own
  // evaluation (mctx's num_simulations). GumbelAction picks the move: the halving's winner.
  void SetGumbel(int max_considered, double scale) {
    gumbel_ = true;
    gumbel_considered_ = max_considered;
    gumbel_scale_ = scale;
  }
  bool gumbel() const { return gumbel_; }
  // Our change (2026-11-01): the last MCTSearch's tree — the most nodes it held (each
  // NodeBytes(), as max_memory_mb counts them) and how often it reached that limit and
  // garbage-collected (dropping the subtrees of rarely visited nodes; the root's children
  // and their counts stay).
  int last_search_peak_nodes() const { return last_peak_nodes_; }
  int last_search_collections() const { return last_collections_; }
  static constexpr int NodeBytes() { return sizeof(SearchNode); }
  // The move after a Gumbel search of `root` (the root of the last MCTSearch): among the
  // children with the most playouts, the highest gumbel + log prior + sigma(completed Q).
  Action GumbelAction(const SearchNode& root) const;

  void SetPolicyTemperatures(std::vector<double> temperatures) {
    policy_temperatures_ = std::move(temperatures);
  }

  // Our change (thud/PLAN.md, the blunder diagnosis — a diagnostic, not for training):
  // at every node where player p is to move, the actions `marked` picks out get at
  // least floors[p] of the prior's mass between them (ApplyPriorFloor), after the
  // policy temperature and before root noise. Empty, or 0 for a player, leaves the
  // prior as it is.
  void SetPriorFloors(std::vector<double> floors,
                      std::function<bool(const State&, Action)> marked) {
    prior_floors_ = std::move(floors);
    prior_floor_marked_ = std::move(marked);
  }

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
  int last_peak_nodes_ = 0;    // Our change (2026-11-01): see last_search_peak_nodes.
  int last_collections_ = 0;
  bool verbose_;
  bool solve_;
  double max_utility_;
  double min_utility_;
  UntriedMoveValue untried_move_value_;
  double untried_move_reduction_;
  double forced_playouts_k_;
  std::vector<double> policy_temperatures_;  // Our change: SetPolicyTemperatures.
  // Our change (SetGumbel): its settings, and the last search's draws and schedule.
  bool gumbel_ = false;
  int gumbel_considered_ = 16;
  double gumbel_scale_ = 1;
  std::vector<double> root_gumbel_;
  std::vector<int> gumbel_schedule_;
  std::vector<double> prior_floors_;         // Our change: SetPriorFloors.
  std::function<bool(const State&, Action)> prior_floor_marked_;
  double dirichlet_alpha_;
  double dirichlet_epsilon_;
  bool dont_return_chance_node_;
  std::mt19937 rng_;
  const ChildSelectionPolicy child_selection_policy_;
  std::shared_ptr<Evaluator> evaluator_;
};

// Our change (see MCTSBot::SetPolicyTemperatures): raises each probability of
// `prior` to 1 / temperature and renormalises; zeros stay zero.
void ApplyPolicyTemperature(double temperature, ActionsAndProbs* prior);

// Our change (see MCTSBot::SetPriorFloors): if the actions with marked[i] hold less
// than `floor` of `prior`'s mass, scales them up to `floor` between them (equal shares
// if they held none) and the others down to 1 - floor, keeping each group's
// proportions; otherwise, or with no marked action, leaves `prior` as it is.
void ApplyPriorFloor(double floor, const std::vector<bool>& marked,
                     ActionsAndProbs* prior);

// Returns a vector of noise sampled from a dirichlet distribution. See:
// https://en.wikipedia.org/wiki/Dirichlet_process
std::vector<double> dirichlet_noise(int count, double alpha, std::mt19937* rng);

}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_MCTS_H_
