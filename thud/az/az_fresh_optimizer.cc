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

// A network's weights with a fresh optimizer, for a run that starts from a network
// trained elsewhere (thud/PLAN.md, run L: K83's trunk with the convolutional head
// az_restructure trained at learning rate 0.001, to go on at run K's 0.0001).
//
// Builds the network from `to`/vpnet.pb (its structure and learning rate), loads the
// weights alone from `from`'s checkpoint, and saves them with the new, fresh Adam
// optimizer as `to`'s checkpoint `step`. Prints the learning rate a full load of the
// source checkpoint would leave (LibTorch restores an optimizer's options with its state:
// not shown either way on 2026-10-11, the source's file and checkpoint having the same
// rate; shown since by learning_settings_check, 2026-10-15 — which is why a new rate
// needs a fresh optimizer). Checks, fatal on failure: the saved checkpoint, loaded in full into a new
// network, gives exactly the source's outputs on positions of random games, and has
// `to`'s learning rate.
//
//   thud/az/build.sh thud/az/az_fresh_optimizer.cc
//   build-shared/az_fresh_optimizer from=DIR:STEP to=DIR [step=STEP seed=1]

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "thud/az/vpnet.h"

using open_spiel::thud_az::torch_az::VPNetModel;

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value: " + arg);
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  const std::string from = args["from"], to = args["to"];
  if (from.empty() || to.empty()) {
    std::cerr << "Usage: az_fresh_optimizer from=DIR:STEP to=DIR [step=STEP]" << std::endl;
    return 1;
  }
  const size_t colon = from.rfind(':');
  const std::string from_dir = from.substr(0, colon), from_step = from.substr(colon + 1);
  const int step = std::stoi(args.count("step") ? args["step"] : from_step);
  std::mt19937 rng(std::stoull(args.count("seed") ? args["seed"] : "1"));
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  VPNetModel source(*game, from_dir, "vpnet.pb", "/cpu:0");
  const double source_rate = source.LearningRate();
  source.LoadCheckpoint(absl::StrCat(from_dir, "/checkpoint-", from_step));
  std::cerr << absl::StrFormat(
      "the source's learning rate: %g in its vpnet.pb, %g after a full load of its "
      "checkpoint\n",
      source_rate, source.LearningRate());

  VPNetModel fresh(*game, to, "vpnet.pb", "/cpu:0");
  const double rate = fresh.LearningRate();
  fresh.LoadWeights(absl::StrCat(from_dir, "/checkpoint-", from_step));
  fresh.SaveCheckpoint(step);

  // The checks: the saved checkpoint, loaded in full.
  VPNetModel saved(*game, to, "vpnet.pb", "/cpu:0");
  saved.LoadCheckpoint(step);
  std::vector<VPNetModel::InferenceInputs> positions;
  for (int g = 0; positions.size() < 200; ++g) {
    std::unique_ptr<open_spiel::State> s = game->NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 200; ++ply) {
      if (ply % 10 == 3) positions.push_back({s->LegalActions(), s->ObservationTensor()});
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      s->ApplyAction(legal[rng() % legal.size()]);
    }
  }
  const auto a = source.Inference(positions), b = saved.Inference(positions);
  double worst = 0;
  for (size_t i = 0; i < positions.size(); ++i) {
    worst = std::max(worst, std::abs(a[i].value - b[i].value));
    for (size_t j = 0; j < a[i].policy.size(); ++j) {
      worst = std::max(worst, std::abs(a[i].policy[j].second - b[i].policy[j].second));
    }
  }
  std::cerr << absl::StrFormat(
      "check: %d positions, outputs differ from the source's by %g at most; the saved "
      "checkpoint's learning rate %g (to's vpnet.pb: %g)\n",
      positions.size(), worst, saved.LearningRate(), rate);
  if (worst > 0) open_spiel::SpielFatalError("The weights changed");
  if (saved.LearningRate() != rate) open_spiel::SpielFatalError("The learning rate changed");
  std::cout << absl::StrCat(to, "/checkpoint-", step) << std::endl;
  return 0;
}
