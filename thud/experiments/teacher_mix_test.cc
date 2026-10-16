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

// Tests az_match's teacher_mix_a, TeacherMixEvaluator (thud/experiments/teacher_mix.h;
// thud/DESIGN_dwarf_signal.md, measurement 2): network A's prior at a player's nodes mixed with
// kickstarting's teacher, (1 - w) x the network's + w x TeacherPolicy — with a stand-in
// network (an uneven prior, fixed values), so that no model is needed. At a dwarf and a
// troll position of a random game, each with a capture to make (so the teacher is far
// from the stand-in's prior: the control, below):
//
//   1. With the mix "0,0.5": at the dwarfs' node the network's prior unchanged; at the
//      trolls' node half the network's and half the teacher's, within 1e-12, summing to 1.
//   2. With "0.3,1": the dwarfs' node 0.7 x the network's + 0.3 x the teacher's; the
//      trolls' node the teacher's alone.
//   3. The values are the network's, unchanged; a position asked for again (from the
//      evaluator's cache) gets the same prior.
//   Control: at both nodes the teacher differs from the network's prior by more than 0.05
//   in some move, so 1 and 2 can tell a mixed prior from an unmixed one.
//
//   build-tests/bin/teacher_mix_test

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/experiments/teacher_mix.h"

namespace {

using open_spiel::Action;
using open_spiel::ActionsAndProbs;
using open_spiel::State;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

// An uneven prior (the i-th legal move weighs i + 1) and fixed values.
class StandIn : public open_spiel::thud_az::Evaluator {
 public:
  std::vector<double> Evaluate(const State& state) override { return {0.25, -0.25}; }
  ActionsAndProbs Prior(const State& state) override {
    const std::vector<Action> legal = state.LegalActions();
    const double n = legal.size();
    ActionsAndProbs prior;
    for (int i = 0; i < legal.size(); ++i) prior.push_back({legal[i], (i + 1) / (n * (n + 1) / 2)});
    return prior;
  }
};

std::map<Action, double> AsMap(const ActionsAndProbs& policy) {
  std::map<Action, double> m;
  for (const auto& [a, p] : policy) m[a] = p;
  return m;
}

// The largest difference between `got` and (1 - w) x `network` + w x `teacher`, and the sum.
std::pair<double, double> Deviation(const ActionsAndProbs& got, const ActionsAndProbs& network,
                                    const ActionsAndProbs& teacher, double w) {
  const std::map<Action, double> n = AsMap(network), t = AsMap(teacher);
  double d = 0, sum = 0;
  for (const auto& [a, p] : got) {
    d = std::max(d, std::abs(p - ((1 - w) * n.at(a) + w * t.at(a))));
    sum += p;
  }
  if (got.size() != n.size()) d = 1;
  return {d, sum};
}

bool HasCapture(const State& state) {
  for (Action a : state.LegalActions()) {
    if (absl::StrContains(state.ActionToString(state.CurrentPlayer(), a), "x")) return true;
  }
  return false;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const int depth = 2;
  const double temperature = 0.25;
  const open_spiel::thud_az::AlphaBetaPlayer teacher(*game, depth);
  auto network = std::make_shared<StandIn>();

  // A dwarf and a troll position with a capture to make, from a random game.
  std::unique_ptr<State> node[2];
  std::mt19937 rng(3);
  while (!node[0] || !node[1]) {
    std::unique_ptr<State> s = game->NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 400 && (!node[0] || !node[1]); ++ply) {
      const int p = s->CurrentPlayer();
      if (ply > 10 && !node[p] && HasCapture(*s)) node[p] = s->Clone();
      const std::vector<Action> legal = s->LegalActions();
      s->ApplyAction(legal[rng() % legal.size()]);
    }
  }

  ActionsAndProbs net[2], teach[2];
  for (int p = 0; p < 2; ++p) {
    net[p] = network->Prior(*node[p]);
    teach[p] = open_spiel::thud_az::torch_az::TeacherPolicy(*node[p], teacher, temperature);
    const auto [d, sum] = Deviation(teach[p], net[p], net[p], 0);
    Check(d > 0.05, absl::StrFormat("control: at the %s' node the teacher differs from the "
                                    "network's prior (by %.3g)",
                                    p == 0 ? "dwarfs" : "trolls", d));
  }

  {
    open_spiel::thud_az::TeacherMixEvaluator mix(*game, network, {0, 0.5}, depth, temperature);
    for (int p = 0; p < 2; ++p) {
      const double w = p == 0 ? 0 : 0.5;
      const auto [d, sum] = Deviation(mix.Prior(*node[p]), net[p], teach[p], w);
      Check(d <= 1e-12 && std::abs(sum - 1) <= 1e-12,
            absl::StrFormat("1. mix 0,0.5: the %s' node is %g x the teacher's (deviation "
                            "%.2g, sum %.15g)",
                            p == 0 ? "dwarfs" : "trolls", w, d, sum));
    }
  }
  {
    open_spiel::thud_az::TeacherMixEvaluator mix(*game, network, {0.3, 1}, depth, temperature);
    for (int p = 0; p < 2; ++p) {
      const double w = p == 0 ? 0.3 : 1;
      const auto [d, sum] = Deviation(mix.Prior(*node[p]), net[p], teach[p], w);
      Check(d <= 1e-12 && std::abs(sum - 1) <= 1e-12,
            absl::StrFormat("2. mix 0.3,1: the %s' node is %g x the teacher's (deviation "
                            "%.2g)",
                            p == 0 ? "dwarfs" : "trolls", w, d));
    }
    const std::vector<double> v = mix.Evaluate(*node[1]);
    Check(v == network->Evaluate(*node[1]), "3. the values are the network's");
    Check(AsMap(mix.Prior(*node[1])) == AsMap(mix.Prior(*node[1])) &&
              Deviation(mix.Prior(*node[1]), net[1], teach[1], 1).first <= 1e-12,
          "3. a position asked for again (from the cache) gets the same prior");
  }

  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
