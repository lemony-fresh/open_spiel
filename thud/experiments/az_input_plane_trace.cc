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

// How a network's input-plane weights develop over a run's checkpoints (thud/PLAN.md, the
// input planes, as decided: learning, over many steps — the user, 2026-10-11, asked to
// see that the zero-initialised weights learn and do not collapse over many training
// steps; run M started them at zero by net surgery).
//
// For each checkpoint of a run whose network takes the input planes, from the first
// convolution's channels past the observation's (filters x 3 x 3 weights a plane):
//   norm_median, norm_max   the size of each plane's weights, over the 50 planes;
//   by_distance             the mean size of the move planes of each distance 1-7;
//   maps                    the size of each map's weights (kTrollLandsNear, kHurlLands);
//   spread                  how differently the filters use the planes — the mean over
//                           planes of the standard deviation, over filters, of each
//                           filter's weights' size (0 if all filters moved alike);
//   policy_effect,          how much the outputs move when those weights are zeroed, on a
//   value_effect            fixed set of positions (half the policies' L1 distance; the
//                           values' mean absolute difference) — how much the network
//                           leans on the planes.
// The positions: every 5th of each logged actor's last self-play game in the run.
//
//   thud/az/build.sh thud/experiments/az_input_plane_trace.cc
//   OMP_NUM_THREADS=4 build-shared/az_input_plane_trace run=RUN_DIR steps=83,84,...,102

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;
namespace az = open_spiel::thud_az;

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value: " + arg);
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  const std::string run = args["run"];
  if (run.empty() || args["steps"].empty()) {
    std::cerr << "Usage: az_input_plane_trace run=RUN_DIR steps=S1,S2,..." << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The positions.
  std::vector<VPNetModel::InferenceInputs> in;
  for (const auto& entry : std::filesystem::directory_iterator(run)) {
    if (!absl::StartsWith(entry.path().filename().string(), "log-actor-")) continue;
    std::string last;
    std::ifstream file(entry.path());
    for (std::string line; std::getline(file, line);) {
      if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
        last = line;
      }
    }
    if (last.empty()) continue;
    std::unique_ptr<State> s = game->NewInitialState();
    int ply = 0;
    for (absl::string_view move :
         absl::StrSplit(last.substr(last.find("Actions: ") + 9), ' ', absl::SkipEmpty())) {
      if (s->IsTerminal()) break;
      if (ply++ % 5 == 2) in.push_back({s->LegalActions(), s->ObservationTensor()});
      for (Action a : s->LegalActions()) {
        if (s->ActionToString(a) == move) {
          s->ApplyAction(a);
          break;
        }
      }
    }
  }
  std::cerr << in.size() << " positions" << std::endl;

  for (absl::string_view step_text : absl::StrSplit(args["steps"], ',', absl::SkipEmpty())) {
    const int step = std::stoi(std::string(step_text));
    VPNetModel model(*game, run, "vpnet.pb", "/cpu:0");
    if (!model.TakesInputPlanes()) open_spiel::SpielFatalError(run + " takes no input planes");
    model.LoadCheckpoint(step);
    torch::Tensor w = model.Parameter("layers.0.input_conv.weight");
    const int64_t first = thud::kNumObservationPlanes;
    std::vector<double> norms;
    double spread = 0;
    {
      torch::NoGradGuard no_grad;
      for (int64_t c = first; c < w.size(1); ++c) {
        torch::Tensor channel = w.select(1, c);  // filters x 3 x 3
        norms.push_back(channel.norm().item<double>());
        spread += channel.flatten(1).norm(2, 1).std().item<double>();
      }
    }
    spread /= norms.size();
    std::vector<double> by_distance(az::kMaxCaptureDistance + 1, 0), count(az::kMaxCaptureDistance + 1, 0);
    for (int p = 0; p < az::kNumMovePlanes; ++p) {
      by_distance[az::MoveOfPlane(p).k] += norms[p];
      count[az::MoveOfPlane(p).k] += 1;
    }
    std::vector<double> sorted = norms;
    std::sort(sorted.begin(), sorted.end());
    // The planes' effect: the outputs with those weights zeroed.
    const auto with = model.Inference(in);
    {
      torch::NoGradGuard no_grad;
      w.slice(1, first).zero_();
    }
    const auto without = model.Inference(in);
    double policy = 0, value = 0;
    for (size_t i = 0; i < in.size(); ++i) {
      double l1 = 0;
      for (size_t j = 0; j < with[i].policy.size(); ++j) {
        l1 += std::abs(with[i].policy[j].second - without[i].policy[j].second);
      }
      policy += l1 / 2;
      value += std::abs(with[i].value - without[i].value);
    }
    std::string distances;
    for (int k = 1; k <= az::kMaxCaptureDistance; ++k) {
      distances += absl::StrFormat("%s%.4f", k > 1 ? ", " : "", by_distance[k] / count[k]);
    }
    std::cout << absl::StrFormat(
                     "{\"step\": %d, \"norm_median\": %.4f, \"norm_max\": %.4f, "
                     "\"by_distance\": [%s], \"maps\": [%.4f, %.4f], \"spread\": %.5f, "
                     "\"policy_effect\": %.4f, \"value_effect\": %.4f}",
                     step, sorted[sorted.size() / 2], sorted.back(), distances,
                     norms[az::kTrollLandsNear], norms[az::kHurlLands], spread,
                     policy / in.size(), value / in.size())
              << std::endl;
  }
  return 0;
}
