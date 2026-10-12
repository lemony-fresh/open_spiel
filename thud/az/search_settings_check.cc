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

// Checks the per-player search settings (thud/PLAN.md, training the dwarfs): the
// policy softmax temperature (MCTSBot::SetPolicyTemperatures, ApplyPolicyTemperature)
// and the trainer's player_simulations / player_policy_temperature (InitAZBot). With
// a deterministic stand-in for the network (a fixed uneven prior, values from a hash
// of the position) and no root noise, every search is reproducible:
//
//   1. ApplyPolicyTemperature against hand-worked numbers; 1 changes nothing.
//   2. Temperatures {1, 1} give exactly the visits of no setting (the control that
//      it changes nothing it should not); {2, 1} gives the dwarfs' roots the prior
//      flattened by 2 and leaves the trolls' roots' prior as it was. The control
//      that it does something: with a sharp prior and values that single out no
//      move — our dwarfs' situation, the prior on ~5% of the legal moves — it spreads
//      the dwarfs' searches over more moves. (With an only mildly uneven prior and
//      uneven values a flatter prior need not: the search then follows the values;
//      the first stand-in shows that, and the control was changed to this one on
//      2026-10-06, to review with the user.)
//   3. InitAZBot: each self-play player's search has its simulations and the
//      temperatures; the evaluation search max_simulations.
//   4. The config's JSON keeps both settings, and a config without them loads as
//      empty (old runs' config.json).
//   5. The prior floor (MCTSBot::SetPriorFloors, ApplyPriorFloor; 2026-10-08, a
//      diagnostic): hand-worked numbers, including marked moves with no prior and a
//      floor already met; floors {0, 0} give exactly the visits of no setting; with
//      {0, 0.5} and the captures marked, the trolls' roots with a capture get the
//      floored prior, the dwarfs' roots keep theirs — and, the control that it reaches
//      inside a search, some dwarf searches change (their trolls' replies now weigh
//      captures).
//
//   thud/az/build.sh thud/az/search_settings_check.cc
//   build-shared/search_settings_check

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/mcts.h"

namespace {

using open_spiel::Action;
using open_spiel::ActionsAndProbs;
using open_spiel::State;
using open_spiel::thud_az::ApplyPolicyTemperature;
using open_spiel::thud_az::ChildSelectionPolicy;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;

int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  failures += !ok;
}

// A fixed uneven prior over the legal moves, and a value from a hash of the position.
class StandIn : public open_spiel::thud_az::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override {
    const double v = 0.3 * std::sin(std::hash<std::string>{}(state.ToString()) % 1000);
    return {v, -v};
  }
  ActionsAndProbs Prior(const State& state) override {
    ActionsAndProbs prior;
    double sum = 0;
    for (Action a : state.LegalActions()) {
      const double w = 1 + (a * 7919) % 13;
      prior.push_back({a, w});
      sum += w;
    }
    for (auto& [a, p] : prior) p /= sum;
    return prior;
  }
};

// A sharp prior (each move half as likely as the one before it, in a fixed order) and
// values that single out no move: the search's spread then follows the prior.
class SharpPrior : public open_spiel::thud_az::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override { return {0, 0}; }
  ActionsAndProbs Prior(const State& state) override {
    std::vector<Action> legal = state.LegalActions();
    std::sort(legal.begin(), legal.end(), [](Action a, Action b) {
      return (a * 7919) % 104729 < (b * 7919) % 104729;
    });
    ActionsAndProbs prior;
    double w = 1, sum = 0;
    for (Action a : legal) {
      prior.push_back({a, w});
      sum += w;
      w *= 0.5;
    }
    for (auto& [a, p] : prior) p /= sum;
    return prior;
  }
};

std::unique_ptr<MCTSBot> Bot(const open_spiel::Game& game,
                             std::shared_ptr<open_spiel::thud_az::Evaluator> eval,
                             int sims) {
  return std::make_unique<MCTSBot>(game, eval, /*uct_c=*/2, sims, /*max_memory_mb=*/100,
                                   /*solve=*/false, /*seed=*/1, /*verbose=*/false,
                                   ChildSelectionPolicy::PUCT, /*alpha=*/0,
                                   /*epsilon=*/0, /*dont_return_chance_node=*/true);
}

std::vector<int> Visits(const SearchNode& root) {
  std::vector<int> v;
  for (const SearchNode& c : root.children) v.push_back(c.explore_count);
  return v;
}

double EffectiveMoves(const SearchNode& root) {
  double entropy = 0;
  for (const SearchNode& c : root.children) {
    if (c.explore_count == 0) continue;
    const double q = static_cast<double>(c.explore_count) / root.explore_count;
    entropy -= q * std::log(q);
  }
  return std::exp(entropy);
}

// The prior each root child got, by action.
bool PriorsMatch(const SearchNode& root, const ActionsAndProbs& expected) {
  for (const SearchNode& c : root.children) {
    bool found = false;
    for (const auto& [a, p] : expected) {
      if (a == c.action) {
        if (std::abs(c.prior - p) > 1e-12) return false;
        found = true;
      }
    }
    if (!found) return false;
  }
  return root.children.size() == expected.size();
}

}  // namespace

int main(int argc, char** argv) {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  auto eval = std::make_shared<StandIn>();

  // 1. The transform.
  ActionsAndProbs p = {{1, 0.7}, {2, 0.2}, {3, 0.1}, {4, 0.0}};
  ApplyPolicyTemperature(2, &p);
  const double s = std::sqrt(0.7) + std::sqrt(0.2) + std::sqrt(0.1);
  Check(std::abs(p[0].second - std::sqrt(0.7) / s) < 1e-12 &&
            std::abs(p[1].second - std::sqrt(0.2) / s) < 1e-12 &&
            std::abs(p[2].second - std::sqrt(0.1) / s) < 1e-12 && p[3].second == 0,
        "1. temperature 2: p^(1/2) renormalised (0.523, 0.279, 0.198), zero stays zero");
  ActionsAndProbs q = {{1, 0.7}, {2, 0.2}, {3, 0.1}};
  ApplyPolicyTemperature(1, &q);
  Check(q[0].second == 0.7 && q[1].second == 0.2 && q[2].second == 0.1,
        "1. temperature 1 changes nothing");

  // Positions of random games, both sides to move.
  std::mt19937 rng(20261006);
  std::vector<std::unique_ptr<State>> positions;
  while (positions.size() < 40) {
    std::unique_ptr<State> state = game->NewInitialState();
    for (int move = 0; !state->IsTerminal() && positions.size() < 40; ++move) {
      if (move > 0 && move % 9 == 0) positions.push_back(state->Clone());
      std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
  }

  // 2. The search.
  bool identical = true, dwarf_priors = true, troll_priors = true;
  double spread_plain = 0, spread_flat = 0;
  int dwarf_roots = 0;
  for (const auto& position : positions) {
    auto plain = Bot(*game, eval, 100);
    auto ones = Bot(*game, eval, 100);
    ones->SetPolicyTemperatures({1, 1});
    auto flat = Bot(*game, eval, 100);
    flat->SetPolicyTemperatures({2, 1});
    const std::unique_ptr<SearchNode> a = plain->MCTSearch(*position);
    const std::unique_ptr<SearchNode> b = ones->MCTSearch(*position);
    const std::unique_ptr<SearchNode> c = flat->MCTSearch(*position);
    identical = identical && Visits(*a) == Visits(*b);
    ActionsAndProbs raw = eval->Prior(*position);
    if (position->CurrentPlayer() == 0) {
      ActionsAndProbs flattened = raw;
      ApplyPolicyTemperature(2, &flattened);
      dwarf_priors = dwarf_priors && PriorsMatch(*c, flattened) && !PriorsMatch(*c, raw);
      spread_plain += EffectiveMoves(*a);
      spread_flat += EffectiveMoves(*c);
      ++dwarf_roots;
    } else {
      troll_priors = troll_priors && PriorsMatch(*c, raw);
    }
  }
  Check(identical, "2. temperatures {1, 1}: the same visits as no setting, in all " +
                       std::to_string(positions.size()) + " positions");
  Check(dwarf_priors, "2. temperatures {2, 1}: the dwarfs' roots get the prior flattened by 2");
  Check(troll_priors, "2. temperatures {2, 1}: the trolls' roots keep the prior as it was");
  std::cout << "   the dwarfs' searches spread over " << spread_plain / dwarf_roots
            << " effective moves without, " << spread_flat / dwarf_roots
            << " with temperature 2 (" << dwarf_roots << " positions)" << std::endl;
  // The control, with a sharp prior and values that single out no move.
  auto sharp = std::make_shared<SharpPrior>();
  double sharp_plain = 0, sharp_flat = 0;
  for (const auto& position : positions) {
    if (position->CurrentPlayer() != 0) continue;
    auto plain = Bot(*game, sharp, 100);
    auto flat = Bot(*game, sharp, 100);
    flat->SetPolicyTemperatures({2, 1});
    sharp_plain += EffectiveMoves(*plain->MCTSearch(*position));
    sharp_flat += EffectiveMoves(*flat->MCTSearch(*position));
  }
  std::cout << "   with a sharp prior and even values: " << sharp_plain / dwarf_roots
            << " effective moves without, " << sharp_flat / dwarf_roots
            << " with temperature 2" << std::endl;
  Check(sharp_flat > 1.5 * sharp_plain,
        "2. control: with a sharp prior, temperature 2 spreads the dwarfs' searches "
        "(by more than half again)");

  // 3. The trainer's searches.
  open_spiel::thud_az::torch_az::AlphaZeroConfig config;
  config.uct_c = 2;
  config.max_simulations = 100;
  config.policy_alpha = 0;
  config.policy_epsilon = 0;
  config.untried_move_value = "sibling_mean_minus_reduction";
  config.untried_move_reduction = 0.2;
  config.policy_target_pruning = false;
  config.player_simulations = "300,50";
  config.player_policy_temperature = "1.5,1";
  const State* dwarfs_to_move = nullptr;
  const State* trolls_to_move = nullptr;
  for (const auto& position : positions) {
    if (position->CurrentPlayer() == 0 && !dwarfs_to_move) dwarfs_to_move = position.get();
    if (position->CurrentPlayer() == 1 && !trolls_to_move) trolls_to_move = position.get();
  }
  auto dwarf_bot = open_spiel::thud_az::torch_az::InitAZBot(config, *game, eval, false, 0);
  auto troll_bot = open_spiel::thud_az::torch_az::InitAZBot(config, *game, eval, false, 1);
  auto eval_bot = open_spiel::thud_az::torch_az::InitAZBot(config, *game, eval, true);
  const std::unique_ptr<SearchNode> d = dwarf_bot->MCTSearch(*dwarfs_to_move);
  const std::unique_ptr<SearchNode> t = troll_bot->MCTSearch(*trolls_to_move);
  const std::unique_ptr<SearchNode> e = eval_bot->MCTSearch(*dwarfs_to_move);
  Check(d->explore_count == 300 && t->explore_count == 50 && e->explore_count == 100,
        "3. simulations: the dwarfs' search 300, the trolls' 50, evaluation 100 (" +
            std::to_string(d->explore_count) + ", " + std::to_string(t->explore_count) +
            ", " + std::to_string(e->explore_count) + ")");
  ActionsAndProbs flattened = eval->Prior(*dwarfs_to_move);
  ApplyPolicyTemperature(1.5, &flattened);
  Check(PriorsMatch(*d, flattened) && PriorsMatch(*e, flattened),
        "3. temperatures: the self-play and evaluation searches flatten the dwarfs' "
        "prior by 1.5");
  Check(PriorsMatch(*t, eval->Prior(*trolls_to_move)),
        "3. temperatures: the trolls' prior stays as it was");

  // 4. The config's JSON.
  open_spiel::thud_az::torch_az::AlphaZeroConfig full = config;
  full.game = "thud";
  open_spiel::json::Object json = full.ToJson();
  Check(json.at("player_simulations").GetString() == "300,50" &&
            json.at("player_policy_temperature").GetString() == "1.5,1",
        "4. ToJson keeps both settings");
  json.erase("player_simulations");
  json.erase("player_policy_temperature");
  open_spiel::thud_az::torch_az::AlphaZeroConfig old;
  old.FromJson(json);
  Check(old.player_simulations.empty() && old.player_policy_temperature.empty(),
        "4. a config without them (an old run's) loads them as empty");

  // 5. The prior floor.
  {
    using open_spiel::thud_az::ApplyPriorFloor;
    ActionsAndProbs f = {{1, 0.7}, {2, 0.2}, {3, 0.1}};
    ApplyPriorFloor(0.5, {false, true, true}, &f);
    ActionsAndProbs g = {{1, 0.6}, {2, 0.4}, {3, 0}, {4, 0}};
    ApplyPriorFloor(0.5, {false, false, true, true}, &g);
    ActionsAndProbs met = {{1, 0.7}, {2, 0.2}, {3, 0.1}};
    ApplyPriorFloor(0.2, {false, true, true}, &met);
    ActionsAndProbs none = {{1, 0.7}, {2, 0.3}};
    ApplyPriorFloor(0.5, {false, false}, &none);
    auto near = [](double x, double y) { return std::abs(x - y) < 1e-12; };
    Check(near(f[0].second, 0.5) && near(f[1].second, 0.2 * 0.5 / 0.3) &&
              near(f[2].second, 0.1 * 0.5 / 0.3) && near(g[0].second, 0.3) &&
              near(g[1].second, 0.2) && near(g[2].second, 0.25) && near(g[3].second, 0.25) &&
              met[1].second == 0.2 && met[2].second == 0.1 && none[0].second == 0.7,
          "5. ApplyPriorFloor: hand-worked numbers; a floor already met and no marked move "
          "change nothing");
    auto captures = [](const State& state, Action a) {
      const std::string text = state.ActionToString(a);
      return !text.empty() && text.back() == 'x';
    };
    bool zero_identical = true, troll_floored = true, dwarf_kept = true;
    int troll_roots_with_capture = 0, dwarf_searches_changed = 0, dwarf_searches = 0;
    for (const auto& position : positions) {
      auto plain = Bot(*game, eval, 100);
      auto zero = Bot(*game, eval, 100);
      zero->SetPriorFloors({0, 0}, captures);
      auto floored = Bot(*game, eval, 100);
      floored->SetPriorFloors({0, 0.5}, captures);
      const std::unique_ptr<SearchNode> a = plain->MCTSearch(*position);
      const std::unique_ptr<SearchNode> b = zero->MCTSearch(*position);
      const std::unique_ptr<SearchNode> c = floored->MCTSearch(*position);
      zero_identical = zero_identical && Visits(*a) == Visits(*b);
      ActionsAndProbs raw = eval->Prior(*position);
      if (position->CurrentPlayer() == 1) {
        std::vector<bool> marked;
        for (const auto& [action, p] : raw) marked.push_back(captures(*position, action));
        if (std::find(marked.begin(), marked.end(), true) == marked.end()) continue;
        ++troll_roots_with_capture;
        ActionsAndProbs expected = raw;
        ApplyPriorFloor(0.5, marked, &expected);
        troll_floored = troll_floored && PriorsMatch(*c, expected);
      } else {
        ++dwarf_searches;
        dwarf_kept = dwarf_kept && PriorsMatch(*c, raw);
        dwarf_searches_changed += Visits(*a) != Visits(*c);
      }
    }
    Check(zero_identical, "5. floors {0, 0}: the same visits as no setting, in all " +
                              std::to_string(positions.size()) + " positions");
    Check(troll_roots_with_capture > 0 && troll_floored,
          "5. floors {0, 0.5}: the trolls' roots with a capture get the floored prior (" +
              std::to_string(troll_roots_with_capture) + " roots)");
    Check(dwarf_kept, "5. floors {0, 0.5}: the dwarfs' roots keep their prior");
    Check(dwarf_searches_changed > 0,
          "5. control: the floor reaches inside the dwarfs' searches (" +
              std::to_string(dwarf_searches_changed) + " of " +
              std::to_string(dwarf_searches) + " changed)");
  }

  std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
  return failures ? 1 : 0;
}
