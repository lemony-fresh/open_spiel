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

// A bigger network that computes what a trained one does — Net2Net (Chen, Goodfellow and
// Shlens, "Net2Net: accelerating learning via knowledge transfer", ICLR 2016) for our
// residual networks (thud/PLAN.md, the GPU readiness checklist; the user, 2026-10-15:
// grow the network on the GPU without starting over). Both models, `resnet` (the linear
// policy head) and `resnet_conv_policy`, with or without the input planes.
//
// Wider (nn_width W -> W'): each new channel of the trunk copies a channel chosen at
// random — the same choice g everywhere the trunk's width appears: the input block's
// and every residual block's convolutions and batch norms (their running statistics
// too), the policy head's first convolution and batch norm, and the value head's hidden
// layer (as wide as the trunk) — and every weight that reads those channels is divided
// by how many copies its channel now has, so sums are unchanged. The residual additions
// stay consistent because every block uses the same g. Deeper (nn_depth D -> D'): the new
// residual blocks come after the old ones, with their last batch norm's scale and shift
// at 0 — the block then adds nothing and, its input being the previous block's ReLU
// output, passes it on unchanged (Goyal et al. 2017's zero-initialised residual); its
// convolutions keep their fresh initialisation and learn once the scale moves. Copies
// would stay copies under training (identical weights get identical gradients), so
// `noise` adds noise of that many times each tensor's standard deviation to the new
// channels' incoming weights — 0 keeps the function exact, the test below.
//
// Tests, printed and fatal if they fail: on positions of random games the grown network's
// values and policies equal the source's within `tolerance` (with noise=0); the control
// that the test can fail: the same growth without the division by the copies must differ
// by more than `tolerance`; the saved network reloads with the same outputs. With
// noise > 0 the differences are reported, not required to vanish.
//
// Output: `to` (must not exist) with vpnet.pb (the new size, `learning_rate`, by default
// the source's vpnet.pb rate — not its optimizer's, which a learning-rate schedule may
// have scaled) and checkpoint-STEP (with a fresh optimizer), which the trainer, az_match
// and the probes read like any run's.
//
//   thud/az/build.sh thud/az/az_grow.cc
//   OMP_NUM_THREADS=4 build-shared/az_grow from=RUN_DIR:STEP to=NEW_DIR [nn_width=W' \
//     nn_depth=D' noise=0 seed=1 learning_rate=LR step=STEP tolerance=1e-4 positions=200 \
//     device=/cpu:0]

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/model.h"
#include "thud/az/thud_features.h"
#include "thud/az/net2net.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::torch_az::ModelConfig;
using open_spiel::thud_az::torch_az::VPNetModel;
using open_spiel::thud_az::net2net::GrowInto;
using open_spiel::thud_az::net2net::MakeMapping;
using open_spiel::thud_az::net2net::Mapping;

std::vector<VPNetModel::InferenceInputs> RandomPositions(const open_spiel::Game& game,
                                                         std::mt19937* rng, int count) {
  std::vector<VPNetModel::InferenceInputs> out;
  while (out.size() < count) {
    std::unique_ptr<open_spiel::State> s = game.NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 300 && out.size() < count; ++ply) {
      if (ply % 7 == 3) out.push_back({s->LegalActions(), s->ObservationTensor()});
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      s->ApplyAction(legal[(*rng)() % legal.size()]);
    }
  }
  return out;
}

// The largest differences of the values and of the policies' probabilities.
std::pair<double, double> Differences(VPNetModel& a, VPNetModel& b,
                                      const std::vector<VPNetModel::InferenceInputs>& in) {
  const auto va = a.Inference(in), vb = b.Inference(in);
  double value = 0, policy = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    value = std::max(value, std::abs(va[i].value - vb[i].value));
    for (size_t j = 0; j < va[i].policy.size(); ++j) {
      policy = std::max(policy, std::abs(va[i].policy[j].second - vb[i].policy[j].second));
    }
  }
  return {value, policy};
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
  const std::string from = get("from", ""), to = get("to", "");
  if (from.empty() || to.empty()) {
    std::cerr << "Usage: az_grow from=RUN_DIR:STEP to=NEW_DIR [nn_width= nn_depth= noise=0 "
                 "seed=1 learning_rate= step= tolerance=1e-4 positions=200 device=/cpu:0]"
              << std::endl;
    return 1;
  }
  const size_t colon = from.rfind(':');
  const std::string from_dir = from.substr(0, colon);
  const int from_step = std::stoi(from.substr(colon + 1));
  if (std::filesystem::exists(to)) open_spiel::SpielFatalError(to + " exists already");
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  ModelConfig source_config;
  {
    std::ifstream in(from_dir + "/vpnet.pb");
    in >> source_config;
  }
  const int old_width = source_config.nn_width, old_depth = source_config.nn_depth;
  const int new_width = std::stoi(get("nn_width", std::to_string(old_width)));
  const int new_depth = std::stoi(get("nn_depth", std::to_string(old_depth)));
  const double noise = std::stod(get("noise", "0"));
  const double tolerance = std::stod(get("tolerance", "1e-4"));
  const int count = std::stoi(get("positions", "200"));
  const std::string device = get("device", "/cpu:0");
  const int step = std::stoi(get("step", std::to_string(from_step)));
  const unsigned seed = std::stoul(get("seed", "1"));
  if (source_config.nn_model != "resnet" && source_config.nn_model != "resnet_conv_policy") {
    open_spiel::SpielFatalError("az_grow grows residual networks only, not " +
                                source_config.nn_model);
  }
  if (new_width < old_width || new_depth < old_depth) {
    open_spiel::SpielFatalError("az_grow only grows: nn_width and nn_depth at least the source's");
  }
  // Other dimensions must not be mistaken for the width (a size equal to W elsewhere).
  for (int other : {1, 2, 3, 15 * 15, 2 * 15 * 15, open_spiel::thud::kNumObservationPlanes,
                    open_spiel::thud::kNumObservationPlanes +
                        open_spiel::thud_az::kNumFeaturePlanes}) {
    if (old_width == other && new_width != old_width) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "width %d is also another dimension's size; az_grow cannot tell them apart",
          old_width));
    }
  }

  VPNetModel source(*game, from_dir, "vpnet.pb", device);
  source.LoadCheckpoint(from_step);
  const double learning_rate =
      std::stod(get("learning_rate", absl::StrFormat("%.17g", source.BaseLearningRate())));
  const int extra_planes = source.TakesInputPlanes() ? open_spiel::thud_az::kNumFeaturePlanes : 0;
  std::filesystem::create_directories(to);
  SPIEL_CHECK_TRUE(open_spiel::thud_az::torch_az::CreateGraphDef(
      *game, learning_rate, source_config.weight_decay, to, "vpnet.pb",
      source_config.nn_model, new_width, new_depth, /*verbose=*/false, extra_planes));
  VPNetModel model(*game, to, "vpnet.pb", device);
  VPNetModel control(*game, to, "vpnet.pb", device);

  std::mt19937 rng(seed);
  const Mapping m = MakeMapping(old_width, new_width, &rng);
  std::mt19937 noise_rng(seed + 1), control_rng(seed + 1);
  const std::string what = GrowInto(source, model, old_width, new_width, old_depth, new_depth,
                                    m, /*divide=*/true, noise, &noise_rng);
  GrowInto(source, control, old_width, new_width, old_depth, new_depth, m, /*divide=*/false,
           0, &control_rng);
  std::cerr << absl::StrFormat("%s (%s, %d x %d) -> %s (%d x %d): %s\n", from,
                               source_config.nn_model, old_width, old_depth, to, new_width,
                               new_depth, what);

  std::mt19937 positions_rng(seed + 2);
  const auto check = RandomPositions(*game, &positions_rng, count);
  const auto [value, policy] = Differences(model, source, check);
  const auto [control_value, control_policy] = Differences(control, source, check);
  std::cerr << absl::StrFormat(
      "test: on %d positions of random games the grown network's values differ from the "
      "source's by %.3g at most, its policies by %.3g; the control (no division by the "
      "copies) by %.3g and %.3g\n",
      check.size(), value, policy, control_value, control_policy);
  const bool widened = new_width > old_width;
  if (noise == 0 && (value > tolerance || policy > tolerance)) {
    open_spiel::SpielFatalError("The grown network does not compute what the source does");
  }
  if (widened && control_value <= tolerance && control_policy <= tolerance) {
    open_spiel::SpielFatalError("The control did not fail: the test cannot see the copies");
  }
  model.SaveCheckpoint(step);
  VPNetModel reloaded(*game, to, "vpnet.pb", device);
  reloaded.LoadCheckpoint(step);
  const auto [reload_value, reload_policy] = Differences(reloaded, model, check);
  if (reload_value > 0 || reload_policy > 0) {
    open_spiel::SpielFatalError("The saved network does not reload with the same outputs");
  }
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"from\": \"%s\", \"to\": \"%s\", \"nn_model\": \"%s\", "
                   "\"width\": [%d, %d], \"depth\": [%d, %d], \"input_planes\": %s, "
                   "\"noise\": %g, \"seed\": %u, \"learning_rate\": %g, \"positions\": %d, "
                   "\"value_difference\": %.3g, \"policy_difference\": %.3g, "
                   "\"control_value_difference\": %.3g, \"control_policy_difference\": %.3g, "
                   "\"step\": %d}",
                   from, to, source_config.nn_model, old_width, new_width, old_depth, new_depth,
                   extra_planes ? "true" : "false", noise, seed, model.LearningRate(),
                   check.size(), value, policy, control_value, control_policy, step)
            << std::endl;
  return 0;
}
