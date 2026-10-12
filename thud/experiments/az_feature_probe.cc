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

// Does a network already compute the input features from the rules? (thud/PLAN.md,
// input features from the rules: the probe; the user asked, 2026-10-09, for
// experiments that tell whether we need the extra input planes.)
//
// Linear probes (Alain and Bengio, "Understanding intermediate layers using linear
// classifier probes", arXiv 1610.01644; McGrath et al., "Acquisition of chess
// knowledge in AlphaZero", PNAS 2022, who probed AlphaZero's layers for concepts
// against an untrained network's): for each feature plane of thud/az/thud_features.h,
// a logistic regression shared by all squares reads the plane's fact at a square
// from the network's activations there, trained on the positions of two thirds of
// the games and scored by its AUC on the rest (0.5 chance, 1 perfect). Two places:
//
//   torso   the trunk's output at the square (nn_width numbers; each sees a
//           receptive field far wider than the facts need) — what the network
//           computes;
//   policy  the policy head's 2 rectified planes in the 5 x 5 squares around it (50
//           numbers) — what of it reaches the linear policy layer nearby (that layer
//           reads every square, so this is a lower bound; thud/az/model.cc).
//
// Each fact is probed only on the squares it can concern (takeable dwarfs on dwarfs,
// danger on empty squares, hurls on dwarfs, troll captures on trolls). Controls: a
// fact the input holds outright (a dwarf on the square, on all squares: ~1 in an
// untrained network's torso — the probe can learn) and the danger plane with its
// training labels shuffled, 5 times (their mean ~0.5 — no leak; one alone strays,
// a random direction in a space where the fact is encoded); and, per McGrath et al., the same probes on an
// untrained network of the same shape (`random` as the step), what probes find in
// random features. The activations are checked to be the ones that make the
// network's outputs: the value and policy from the same pass equal Inference's.
//
//   thud/az/build.sh thud/experiments/az_feature_probe.cc
//   OMP_NUM_THREADS=4 build-shared/az_feature_probe logs=RUN_DIR \
//     nets=NAME=DIR:STEP,NAME=DIR:random,... [last=3 max=4000 iters=400 seed=1]
//
// Prints one JSON line per network, place and fact.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
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
constexpr int kN = thud::kBoardSize;

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

// The area under the ROC curve of `scores` for `labels` (ties share their ranks).
double Auc(const std::vector<float>& scores, const std::vector<bool>& labels) {
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] < scores[b]; });
  double rank_sum = 0;
  int64_t positives = 0;
  for (int i = 0; i < order.size();) {
    int j = i;
    while (j < order.size() && scores[order[j]] == scores[order[i]]) ++j;
    const double rank = (i + 1 + j) / 2.0;  // The mean of ranks i+1 .. j.
    for (int k = i; k < j; ++k) {
      if (labels[order[k]]) {
        rank_sum += rank;
        ++positives;
      }
    }
    i = j;
  }
  const int64_t negatives = scores.size() - positives;
  if (positives == 0 || negatives == 0) return NAN;
  return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives);
}

struct ProbeResult {
  int64_t train = 0, test = 0, test_positives = 0;
  double auc = NAN;
};

// A logistic regression from `x` (samples x features) to `y`, trained on the rows
// with !test, scored on the rest. `shuffle`: the training labels permuted first.
ProbeResult Probe(torch::Tensor x, torch::Tensor y, torch::Tensor test, int iters,
                  bool shuffle, uint64_t seed) {
  ProbeResult r;
  torch::Tensor xtr = x.index({~test}), xte = x.index({test});
  torch::Tensor ytr = y.index({~test}).to(torch::kFloat), yte = y.index({test});
  r.train = xtr.size(0);
  r.test = xte.size(0);
  r.test_positives = yte.sum().item<int64_t>();
  if (r.train == 0 || r.test == 0) return r;
  if (shuffle) {
    torch::manual_seed(seed);
    ytr = ytr.index({torch::randperm(ytr.size(0))});
  }
  const torch::Tensor mean = xtr.mean(0), sd = xtr.std(0).clamp_min(1e-6);
  xtr = (xtr - mean) / sd;
  xte = (xte - mean) / sd;
  torch::Tensor w = torch::zeros({x.size(1)}, torch::requires_grad());
  torch::Tensor b = torch::zeros({1}, torch::requires_grad());
  torch::optim::Adam optimizer({w, b}, torch::optim::AdamOptions(0.05));
  for (int i = 0; i < iters; ++i) {
    optimizer.zero_grad();
    torch::Tensor loss =
        torch::binary_cross_entropy_with_logits(torch::mv(xtr, w) + b, ytr) +
        1e-4 * w.pow(2).sum();
    loss.backward();
    optimizer.step();
  }
  torch::NoGradGuard no_grad;
  torch::Tensor scores = (torch::mv(xte, w) + b).contiguous();
  std::vector<float> s(scores.data_ptr<float>(), scores.data_ptr<float>() + scores.numel());
  torch::Tensor labels = yte.to(torch::kBool).contiguous();
  std::vector<bool> l(labels.numel());
  for (int64_t i = 0; i < labels.numel(); ++i) l[i] = labels.data_ptr<bool>()[i];
  r.auc = Auc(s, l);
  return r;
}

// Per square, the channels of `planes` (positions x C x kN x kN) in the window x
// window squares around it (zero beyond the board): positions*kN*kN x C*window^2.
torch::Tensor PerSquare(torch::Tensor planes, int window) {
  const int64_t n = planes.size(0);
  if (window > 1) {
    const int pad = window / 2;
    torch::Tensor padded = torch::constant_pad_nd(planes, {pad, pad, pad, pad}, 0);
    std::vector<torch::Tensor> shifted;
    for (int dr = 0; dr < window; ++dr) {
      for (int dc = 0; dc < window; ++dc) {
        shifted.push_back(padded.slice(2, dr, dr + kN).slice(3, dc, dc + kN));
      }
    }
    planes = torch::cat(shifted, 1);
  }
  return planes.permute({0, 2, 3, 1}).reshape({n * kN * kN, planes.size(1)});
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
  const std::string logs = get("logs", ""), nets = get("nets", "");
  const int last = std::stoi(get("last", "3")), max_positions = std::stoi(get("max", "4000"));
  const int iters = std::stoi(get("iters", "400"));
  const uint64_t seed = std::stoull(get("seed", "1"));
  if (logs.empty() || nets.empty()) {
    std::cerr << "Usage: az_feature_probe logs=RUN_DIR nets=NAME=DIR:STEP,... [...]"
              << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // Positions from each logged actor's last self-play games, with their game.
  std::vector<std::unique_ptr<State>> all;
  std::vector<int> all_games;
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(logs)) {
    const std::string name = entry.path().filename().string();
    if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  int games = 0;
  for (const std::string& path : paths) {
    std::vector<std::string> lines;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
      if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
        lines.push_back(line);
      }
    }
    for (int i = std::max<int>(0, lines.size() - last); i < lines.size(); ++i, ++games) {
      const std::string actions = lines[i].substr(lines[i].find("Actions: ") + 9);
      std::unique_ptr<State> state = game->NewInitialState();
      for (absl::string_view move : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
        if (state->IsTerminal()) break;
        all.push_back(state->Clone());
        all_games.push_back(games);
        state->ApplyAction(Find(*state, std::string(move)));
      }
    }
  }
  // An even sample.
  std::vector<int> chosen;
  for (int i = 0; i < all.size(); ++i) {
    if (all.size() > max_positions &&
        i * static_cast<int64_t>(max_positions) / all.size() ==
            (i + 1) * static_cast<int64_t>(max_positions) / all.size()) {
      continue;
    }
    chosen.push_back(i);
  }
  const int64_t n = chosen.size();
  std::cerr << absl::StrFormat("%d self-play games, %d positions, %d probed\n", games,
                               all.size(), n);

  // Inputs, feature planes, squares and the test split (every third game).
  std::vector<VPNetModel::InferenceInputs> inputs;
  std::vector<float> features;
  std::vector<uint8_t> is_test;
  for (int i : chosen) {
    const State& s = *all[i];
    inputs.push_back({s.LegalActions(), s.ObservationTensor()});
    az::AppendFeaturePlanes(inputs.back().observations, &features);
    is_test.push_back(all_games[i] % 3 == 0);
  }
  std::vector<float> observations;
  for (const auto& in : inputs) {
    observations.insert(observations.end(), in.observations.begin(), in.observations.end());
  }
  const torch::Tensor obs =
      torch::from_blob(observations.data(), {n, thud::kNumObservationPlanes, kN, kN})
          .clone();
  const torch::Tensor feat =
      torch::from_blob(features.data(), {n, az::kNumFeaturePlanes, kN, kN}).clone();
  const torch::Tensor dwarf = obs.select(1, thud::kDwarfPlane) > 0.5;
  const torch::Tensor troll = obs.select(1, thud::kTrollPlane) > 0.5;
  const torch::Tensor empty = obs.select(1, thud::kEmptyPlane) > 0.5;
  const torch::Tensor test =
      torch::from_blob(is_test.data(), {n, 1, 1}, torch::kUInt8).clone().to(torch::kBool)
          .expand({n, kN, kN}).reshape({n * kN * kN});

  // The facts: name, label plane, the squares it concerns, labels shuffled.
  struct Fact {
    std::string name;
    torch::Tensor label, squares;
    bool shuffle;
  };
  const char* kDirections[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
  // The facts, from the 50 planes (2026-10-11; first run on the 20 planes of 2026-10-08,
  // which stated the same facts): takeable and danger are each map on the pieces and on
  // the empty squares; a capturing hurl (troll move) in direction d is any of that
  // direction's move planes on a dwarf (troll).
  const torch::Tensor lands_near = feat.select(1, az::kTrollLandsNear) > 0;
  const torch::Tensor hurl_lands = feat.select(1, az::kHurlLands) > 0;
  std::vector<Fact> facts = {
      {"control_dwarf_here", dwarf, dwarf | troll | empty, false},
      {"control_dwarf_danger_shuffled", lands_near, empty, true},
      {"dwarf_takeable", lands_near, dwarf, false},
      {"troll_takeable", hurl_lands, troll, false},
      {"dwarf_danger", lands_near, empty, false},
      {"troll_danger", hurl_lands, empty, false},
  };
  std::vector<torch::Tensor> captures_toward;  // Per direction: any move plane nonzero.
  for (int d = 0; d < thud::kNumDirections; ++d) {
    torch::Tensor any = torch::zeros_like(dwarf);
    for (int k = 1; k <= az::MaxCaptureDistance(d); ++k) {
      any = any | (feat.select(1, az::MovePlane(d, k)) > 0);
    }
    captures_toward.push_back(any);
  }
  for (int d = 0; d < thud::kNumDirections; ++d) {
    facts.push_back({absl::StrCat("hurl_", kDirections[d]), captures_toward[d], dwarf, false});
  }
  for (int d = 0; d < thud::kNumDirections; ++d) {
    facts.push_back({absl::StrCat("troll_capture_", kDirections[d]), captures_toward[d], troll,
                     false});
  }

  for (absl::string_view spec : absl::StrSplit(nets, ',', absl::SkipEmpty())) {
    const size_t eq = spec.find('='), colon = spec.rfind(':');
    const std::string name(spec.substr(0, eq));
    const std::string dir(spec.substr(eq + 1, colon - eq - 1));
    const std::string step(spec.substr(colon + 1));
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    if (step != "random") model.LoadCheckpoint(std::stoi(step));

    std::vector<torch::Tensor> torso, policy;
    double worst = 0;  // The largest difference from Inference's outputs.
    for (int64_t start = 0; start < n; start += 256) {
      std::vector<VPNetModel::InferenceInputs> batch(
          inputs.begin() + start, inputs.begin() + std::min<int64_t>(n, start + 256));
      std::vector<torch::Tensor> a = model.Activations(batch);
      torso.push_back(a[0]);
      policy.push_back(a[1]);
      if (start == 0) {
        std::vector<VPNetModel::InferenceOutputs> out = model.Inference(batch);
        for (int b = 0; b < batch.size(); ++b) {
          worst = std::max(worst, std::abs(out[b].value - a[2][b][0].item<double>()));
          for (const auto& [action, p] : out[b].policy) {
            worst = std::max(worst, std::abs(p - a[3][b][action].item<double>()));
          }
        }
      }
    }
    std::cerr << absl::StrFormat("%s: activations' outputs differ from Inference's by %g "
                                 "at most (first batch)\n",
                                 name, worst);
    if (worst > 1e-5) open_spiel::SpielFatalError("Activations do not match Inference");

    const torch::Tensor torso_x = PerSquare(torch::cat(torso, 0), 1);
    const torch::Tensor policy_x = PerSquare(torch::cat(policy, 0), 5);
    for (const Fact& f : facts) {
      const torch::Tensor squares = f.squares.reshape({n * kN * kN});
      const torch::Tensor y = f.label.reshape({n * kN * kN}).index({squares});
      const torch::Tensor t = test.index({squares});
      for (const auto& [place, x] : {std::pair{"torso", torso_x}, {"policy", policy_x}}) {
        // A shuffled control: 5 shuffles, their mean (and range) reported.
        std::vector<double> aucs;
        ProbeResult r;
        for (int k = 0; k < (f.shuffle ? 5 : 1); ++k) {
          r = Probe(x.index({squares}), y, t, iters, f.shuffle, seed + k);
          aucs.push_back(r.auc);
        }
        const double mean = std::accumulate(aucs.begin(), aucs.end(), 0.0) / aucs.size();
        std::cout << absl::StrFormat(
                         "{\"net\": \"%s\", \"place\": \"%s\", \"fact\": \"%s\", "
                         "\"train\": %d, \"test\": %d, \"test_positives\": %d, "
                         "\"auc\": %.4f, \"auc_min\": %.4f, \"auc_max\": %.4f}",
                         name, place, f.name, r.train, r.test, r.test_positives, mean,
                         *std::min_element(aucs.begin(), aucs.end()),
                         *std::max_element(aucs.begin(), aucs.end()))
                  << std::endl;
      }
    }
  }
  return 0;
}
