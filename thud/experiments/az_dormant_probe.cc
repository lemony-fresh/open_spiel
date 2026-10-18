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

// Has a network lost capacity to channels that no longer fire? (The user, 2026-10-18:
// the batch-norm race, thud/DESIGN_dwarf_signal.md §10, may have left parts of the
// network unused, and a dead rectifier gets no gradient to recover.) For every rectifier
// of the residual network (VPNetModel::ReluOutputs: the input block's, each torso block's
// two, the heads' three) and each of its channels, over `positions` positions of a
// replay buffer, in evaluation mode (as the network plays): its mean activation, the
// largest, and the share of its entries above 0. A channel is dead if it is never above 0;
// τ-dormant if its mean activation is at most τ times its layer's mean over channels
// (Sokar, Agarwal, Castro and Evci, "The Dormant Neuron Phenomenon in Deep Reinforcement
// Learning", ICML 2023). The first rectifier of a torso block and the heads' can only
// recover through their inputs' weights, and those get no gradient while it is dead (nor
// does its batch norm's shift, which the L2 term leaves out: model.cc); a block's second
// rectifier adds the residual stream.
//
// Tests, fatal, for the first network:
//   - the values ReluOutputs computes after its rectifiers equal Inference's (to 1e-5):
//     the activations measured are the ones that make the outputs;
//   - the positive control: a copy with one live channel killed (its batch norm's shift
//     set to -1e4) in the first torso block's first rectifier and in the value head's
//     convolution must show those channels dead, and the first layer exactly one dead
//     channel more (its inputs are unchanged).
// Reported for comparison: a fresh network of the same structure (LibTorch's
// initialisation), its batch norms' running statistics set to the true statistics of these
// positions (VPNetModel::ConvertBatchNorms without compensation: what training mode
// computes; a fresh network's are placeholders — mean 0, variance 1 — that leave it
// unnormalised in evaluation mode).
//
// Prints one JSON line per network and rectifier, then one summary per network.
//
//   thud/build_programs.sh az_dormant_probe
//   OMP_NUM_THREADS=8 build-shared/az_dormant_probe nets=NAME=DIR:STEP,... \
//     buffer=RUN_DIR/replay_buffer.data [buffer_size=458752 positions=4096 batch=256 \
//     seed=1 tau=0.025,0.1 device=/cpu:0]

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "thud/az/replay_buffer.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::ReplayBuffer;
using open_spiel::thud_az::torch_az::VPNetModel;

// Per channel of one rectifier, over every position and square.
struct Channels {
  torch::Tensor sum, max, positive;  // float64, one entry per channel.
  double entries = 0;
};

struct Layer {
  std::string name;
  int channels = 0;
  int dead = 0;
  std::vector<int> dormant;  // Per τ.
  double median_active = 0;  // The median channel's share of entries above 0.
  std::vector<double> mean;  // Per channel.
};

std::vector<Layer> Measure(VPNetModel& model,
                           const std::vector<VPNetModel::InferenceInputs>& positions,
                           int batch, const std::vector<double>& taus) {
  std::map<std::string, Channels> stats;
  std::vector<std::string> order;
  for (size_t start = 0; start < positions.size(); start += batch) {
    std::vector<VPNetModel::InferenceInputs> part(
        positions.begin() + start,
        positions.begin() + std::min(positions.size(), start + batch));
    for (auto& [name, t] : model.ReluOutputs(part)) {
      if (name == "value") continue;
      torch::Tensor a = (t.dim() == 4 ? t : t.view({t.size(0), t.size(1), 1, 1}))
                            .to(torch::kFloat64);
      Channels& c = stats[name];
      if (!c.sum.defined()) {
        order.push_back(name);
        c.sum = torch::zeros({a.size(1)}, torch::kFloat64);
        c.max = torch::zeros({a.size(1)}, torch::kFloat64);
        c.positive = torch::zeros({a.size(1)}, torch::kFloat64);
      }
      c.sum += a.sum({0, 2, 3});
      c.max = torch::maximum(c.max, a.amax({0, 2, 3}));
      c.positive += (a > 0).to(torch::kFloat64).sum({0, 2, 3});
      c.entries += a.size(0) * a.size(2) * a.size(3);
    }
  }
  std::vector<Layer> layers;
  for (const std::string& name : order) {
    const Channels& c = stats[name];
    Layer l;
    l.name = name;
    l.channels = c.sum.size(0);
    torch::Tensor mean = c.sum / c.entries;
    const double layer_mean = mean.mean().item<double>();
    l.dead = (c.max <= 0).sum().item<int>();
    for (double tau : taus) {
      l.dormant.push_back(layer_mean > 0 ? (mean <= tau * layer_mean).sum().item<int>()
                                         : l.channels);
    }
    std::vector<double> active(l.channels);
    for (int i = 0; i < l.channels; ++i) {
      active[i] = c.positive[i].item<double>() / c.entries;
      l.mean.push_back(mean[i].item<double>());
    }
    std::sort(active.begin(), active.end());
    l.median_active = active[l.channels / 2];
    layers.push_back(l);
  }
  return layers;
}

void Report(const std::string& net, const std::vector<Layer>& layers,
            const std::vector<double>& taus) {
  int channels = 0, dead = 0;
  std::vector<int> dormant(taus.size(), 0);
  for (const Layer& l : layers) {
    std::string d;
    for (size_t i = 0; i < taus.size(); ++i) {
      absl::StrAppend(&d, i ? ", " : "", absl::StrFormat("\"%g\": %d", taus[i], l.dormant[i]));
      dormant[i] += l.dormant[i];
    }
    std::cout << absl::StrFormat(
                     "{\"net\": \"%s\", \"layer\": \"%s\", \"channels\": %d, \"dead\": %d, "
                     "\"dormant\": {%s}, \"median_active_share\": %.4f}",
                     net, l.name, l.channels, l.dead, d, l.median_active)
              << std::endl;
    channels += l.channels;
    dead += l.dead;
  }
  std::string d;
  for (size_t i = 0; i < taus.size(); ++i) {
    absl::StrAppend(&d, i ? ", " : "", absl::StrFormat("\"%g\": %d", taus[i], dormant[i]));
  }
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"net\": \"%s\", \"channels\": %d, \"dead\": %d, "
                   "\"dormant\": {%s}}",
                   net, channels, dead, d)
            << std::endl;
}

const Layer& Find(const std::vector<Layer>& layers, const std::string& name) {
  for (const Layer& l : layers) {
    if (l.name == name) return l;
  }
  open_spiel::SpielFatalError("No rectifier " + name);
}

// Kills `channel` of the rectifier after the batch norm whose name ends in `suffix`.
void Kill(VPNetModel& model, const std::string& suffix, int channel) {
  for (auto& [name, t] : model.NamedTensors()) {
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      torch::NoGradGuard no_grad;
      t[channel].fill_(-1e4);
      return;
    }
  }
  open_spiel::SpielFatalError("No tensor ending in " + suffix);
}

int Largest(const Layer& l) {
  return std::max_element(l.mean.begin(), l.mean.end()) - l.mean.begin();
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
  const std::string nets_arg = get("nets", ""), buffer_path = get("buffer", "");
  if (nets_arg.empty() || buffer_path.empty()) {
    std::cerr << "Usage: az_dormant_probe nets=NAME=DIR:STEP,... buffer=PATH [...]" << std::endl;
    return 1;
  }
  const int count = std::stoi(get("positions", "4096"));
  const int batch = std::stoi(get("batch", "256"));
  const std::string device = get("device", "/cpu:0");
  std::vector<double> taus;
  for (absl::string_view t : absl::StrSplit(get("tau", "0.025,0.1"), ',')) {
    double v;
    SPIEL_CHECK_TRUE(absl::SimpleAtod(t, &v));
    taus.push_back(v);
  }
  std::mt19937 rng(std::stoull(get("seed", "1")));

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  ReplayBuffer<VPNetModel::TrainInputs> buffer(std::stoi(get("buffer_size", "458752")));
  buffer.LoadBuffer(buffer_path);
  if (buffer.Size() < count) {
    open_spiel::SpielFatalError(absl::StrFormat("The buffer has %d positions, %d needed",
                                                buffer.Size(), count));
  }
  std::vector<VPNetModel::InferenceInputs> positions;
  for (const auto& p : buffer.Sample(&rng, count)) {
    positions.push_back({p.legal_actions, p.observations});
  }

  struct Net {
    std::string name, dir;
    int step;
  };
  std::vector<Net> nets;
  for (absl::string_view spec : absl::StrSplit(nets_arg, ',')) {
    const std::vector<std::string> parts = absl::StrSplit(spec, absl::MaxSplits('=', 1));
    SPIEL_CHECK_EQ(parts.size(), 2);
    const size_t colon = parts[1].rfind(':');
    nets.push_back({parts[0], parts[1].substr(0, colon),
                    std::stoi(parts[1].substr(colon + 1))});
  }

  for (size_t n = 0; n < nets.size(); ++n) {
    VPNetModel model(*game, nets[n].dir, "vpnet.pb", device);
    model.LoadCheckpoint(nets[n].step);
    const std::vector<Layer> layers = Measure(model, positions, batch, taus);
    Report(nets[n].name, layers, taus);
    if (n > 0) continue;

    // The tests, on the first network.
    std::vector<VPNetModel::InferenceInputs> first(positions.begin(),
                                                   positions.begin() + std::min(count, batch));
    const auto outputs = model.Inference(first);
    const torch::Tensor values = model.ReluOutputs(first).back().second.flatten();
    double differ = 0;
    for (size_t i = 0; i < first.size(); ++i) {
      differ = std::max(differ, std::abs(values[i].item<double>() - outputs[i].value));
    }
    std::cerr << absl::StrFormat(
        "test: ReluOutputs' values differ from Inference's by %g at most (1e-5 allowed)\n",
        differ);
    if (differ > 1e-5) open_spiel::SpielFatalError("ReluOutputs computes another network");

    VPNetModel killed(*game, nets[n].dir, "vpnet.pb", device);
    killed.LoadCheckpoint(nets[n].step);
    const Layer& torso = Find(layers, "torso_1_a");
    const Layer& value = Find(layers, "value_conv");
    const int torso_channel = Largest(torso), value_channel = Largest(value);
    Kill(killed, "res_0_batch_norm_1.bias", torso_channel);
    Kill(killed, "value_batch_norm.bias", value_channel);
    const std::vector<Layer> after = Measure(killed, positions, batch, taus);
    const Layer& torso_after = Find(after, "torso_1_a");
    const Layer& value_after = Find(after, "value_conv");
    const bool found = torso_after.mean[torso_channel] == 0 &&
                       value_after.mean[value_channel] == 0 &&
                       torso_after.dead == torso.dead + 1;
    std::cerr << absl::StrFormat(
        "test: the positive control — channel %d of torso_1_a (dead channels %d -> %d, one "
        "more required) and channel %d of value_conv (dead channels %d -> %d) killed: %s\n",
        torso_channel, torso.dead, torso_after.dead, value_channel, value.dead,
        value_after.dead, found ? "found dead" : "NOT FOUND");
    if (!found) open_spiel::SpielFatalError("The probe misses a killed channel");

    VPNetModel fresh(*game, nets[n].dir, "vpnet.pb", device);  // LibTorch's initialisation.
    std::vector<std::vector<VPNetModel::TrainInputs>> batches;
    for (size_t start = 0; start < positions.size(); start += batch) {
      batches.emplace_back();
      for (size_t i = start; i < std::min(positions.size(), start + batch); ++i) {
        batches.back().push_back({positions[i].legal_actions, positions[i].observations, {}, 0});
      }
    }
    fresh.ConvertBatchNorms(batches, /*compensate=*/false);
    Report("fresh", Measure(fresh, positions, batch, taus), taus);
  }
  return 0;
}
