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

// Can a network still learn as much as a fresh one of its size? (The user, 2026-10-18: the
// batch-norm race, thud/DESIGN_dwarf_signal.md §10, may have cost capacity that a larger
// network would not bring back.) After the capacity probe of Lyle, Rowland and Dabney
// ("Understanding and Preventing Capacity Loss in Reinforcement Learning", ICLR 2022,
// arXiv 2204.09560: agents' checkpoints fitted to random targets — a random network's
// outputs — fit them worse as training went on): each network learns random targets on
// the same `positions` positions of a replay buffer — a random legal move as the policy
// target, a random -1 or 1 as the value — which no knowledge of the game helps with, so
// how far the losses fall in `batches` learning steps measures what it can still fit.
// (It cannot tell lost capacity from features too specialised to the game to fit
// arbitrary targets — the user, 2026-10-18; a network trained as long on the game from a
// fresh start is the control.) Every network gets a fresh optimizer (Adam at `learning_rate`) and
// learns as the trainer does (training mode, the trainer's loss, no symmetries); the
// trained ones keep their weights (VPNetModel::LoadWeights). Fresh networks of other widths
// (`NAME=fresh:WIDTH`, the first network's structure otherwise) give the scale: a trained
// network that fits like a fresh one of width w has the capacity of width w, in effect.
//
// Every `report` steps one JSON line per network: the mean policy and value loss of the
// steps since the last report; then a summary per network (the last report's losses, the
// mean over all reports, and the size — L2 norm — of its weights at the start: with batch
// norms a larger weight learns more slowly at the same rate, one known cause of lost
// plasticity). Test, fatal, when fresh networks of two widths are measured:
// the wider fits better (else the test cannot tell capacities apart at these settings).
//
//   thud/build_programs.sh az_plasticity
//   OMP_NUM_THREADS=8 build-shared/az_plasticity nets=NAME=DIR:STEP,NAME=fresh:64,... \
//     buffer=RUN_DIR/replay_buffer.data scratch=NEW_DIR [buffer_size=458752 \
//     positions=8192 batch=256 batches=2000 report=100 learning_rate=0.001 seed=1 \
//     device=/cpu:0]
//
// `scratch` (must not exist) holds the fresh networks' vpnet.pb files.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "thud/az/model.h"
#include "thud/az/position_buffer.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::PositionBuffer;
using open_spiel::thud_az::torch_az::ModelConfig;
using open_spiel::thud_az::torch_az::VPNetModel;

struct Result {
  std::string name;
  double last_policy = 0, last_value = 0, mean_policy = 0, mean_value = 0, norm = 0;
};

// The L2 norm of the network's learnt parameters (not the running statistics).
double WeightNorm(VPNetModel& model) {
  double sum = 0;
  for (const auto& [name, t] : model.NamedTensors()) {
    if (name.find("running") != std::string::npos || name.find("num_batches") != std::string::npos) {
      continue;
    }
    sum += t.to(torch::kFloat64).square().sum().item<double>();
  }
  return std::sqrt(sum);
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
  const std::string nets_arg = get("nets", ""), buffer_path = get("buffer", ""),
                    scratch = get("scratch", "");
  if (nets_arg.empty() || buffer_path.empty() || scratch.empty()) {
    std::cerr << "Usage: az_plasticity nets=NAME=DIR:STEP|NAME=fresh:WIDTH,... buffer=PATH "
                 "scratch=NEW_DIR [...]"
              << std::endl;
    return 1;
  }
  const int count = std::stoi(get("positions", "8192"));
  const int batch = std::stoi(get("batch", "256"));
  const int batches = std::stoi(get("batches", "2000"));
  const int report = std::stoi(get("report", "100"));
  const double learning_rate = std::stod(get("learning_rate", "0.001"));
  const std::string device = get("device", "/cpu:0");
  const unsigned seed = std::stoul(get("seed", "1"));
  if (std::filesystem::exists(scratch)) open_spiel::SpielFatalError(scratch + " exists already");
  std::filesystem::create_directories(scratch);

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::vector<VPNetModel::TrainInputs> positions;
  {
    PositionBuffer<VPNetModel::TrainInputs> buffer(std::stoi(get("buffer_size", "458752")));
    buffer.LoadBuffer(buffer_path);
    if (buffer.Size() < count) {
      open_spiel::SpielFatalError(absl::StrFormat("The buffer has %d positions, %d needed",
                                                  buffer.Size(), count));
    }
    std::mt19937 rng(seed);
    positions = buffer.Sample(&rng, count);
  }
  // The random targets, the same for every network.
  {
    std::mt19937 rng(seed + 1);
    for (auto& p : positions) {
      p.policy = {{p.legal_actions[std::uniform_int_distribution<int>(
                       0, p.legal_actions.size() - 1)(rng)],
                   1.0}};
      p.value = std::uniform_int_distribution<int>(0, 1)(rng) ? 1 : -1;
    }
  }

  struct Net {
    std::string name, dir;
    int step = -1, width = 0;  // width > 0: fresh.
  };
  std::vector<Net> nets;
  for (absl::string_view spec : absl::StrSplit(nets_arg, ',')) {
    const std::vector<std::string> parts = absl::StrSplit(spec, absl::MaxSplits('=', 1));
    SPIEL_CHECK_EQ(parts.size(), 2);
    const size_t colon = parts[1].rfind(':');
    Net n{parts[0], parts[1].substr(0, colon)};
    if (n.dir == "fresh") {
      n.width = std::stoi(parts[1].substr(colon + 1));
    } else {
      n.step = std::stoi(parts[1].substr(colon + 1));
    }
    nets.push_back(n);
  }
  if (nets[0].width > 0) open_spiel::SpielFatalError("The first network must be a trained one");
  ModelConfig structure;
  {
    std::ifstream in(nets[0].dir + "/vpnet.pb");
    in >> structure;
  }
  const bool input_planes =
      structure.observation_tensor_shape[0] != game->ObservationTensorShape()[0];

  std::vector<Result> results;
  std::map<int, double> fresh_by_width;  // The last report's summed losses.
  for (const Net& n : nets) {
    std::string dir = n.dir;
    if (n.width > 0) {
      dir = absl::StrCat(scratch, "/", n.name);
      std::filesystem::create_directories(dir);
      SPIEL_CHECK_TRUE(open_spiel::thud_az::torch_az::CreateGraphDef(
          *game, learning_rate, structure.weight_decay, dir, "vpnet.pb", structure.nn_model,
          n.width, structure.nn_depth, /*verbose=*/false,
          input_planes ? open_spiel::thud_az::kNumFeaturePlanes : 0));
    }
    VPNetModel model(*game, dir, "vpnet.pb", device);  // A fresh optimizer.
    if (n.width == 0) model.LoadWeights(absl::StrCat(n.dir, "/checkpoint-", n.step));
    model.SetLearningRate(learning_rate);
    const double norm = WeightNorm(model);

    std::mt19937 rng(seed + 2);  // The same batches for every network.
    std::vector<int> order(positions.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    size_t next = order.size();
    double policy = 0, value = 0, sum_policy = 0, sum_value = 0;
    int reports = 0;
    Result r{n.name};
    r.norm = norm;
    for (int b = 1; b <= batches; ++b) {
      std::vector<VPNetModel::TrainInputs> part;
      for (int i = 0; i < batch; ++i) {
        if (next == order.size()) {
          std::shuffle(order.begin(), order.end(), rng);
          next = 0;
        }
        part.push_back(positions[order[next++]]);
      }
      const VPNetModel::LossInfo loss = model.Learn(part);
      policy += loss.Policy();
      value += loss.Value();
      if (b % report == 0) {
        policy /= report;
        value /= report;
        std::cout << absl::StrFormat(
                         "{\"net\": \"%s\", \"batches\": %d, \"policy\": %.4f, \"value\": %.4f}",
                         n.name, b, policy, value)
                  << std::endl;
        r.last_policy = policy;
        r.last_value = value;
        sum_policy += policy;
        sum_value += value;
        ++reports;
        policy = value = 0;
      }
    }
    r.mean_policy = sum_policy / std::max(reports, 1);
    r.mean_value = sum_value / std::max(reports, 1);
    std::cout << absl::StrFormat(
                     "{\"summary\": true, \"net\": \"%s\", \"fresh\": %s, \"width\": %d, "
                     "\"last_policy\": %.4f, "
                     "\"last_value\": %.4f, \"mean_policy\": %.4f, \"mean_value\": %.4f, "
                     "\"weight_norm\": %.2f}",
                     n.name, n.width > 0 ? "true" : "false",
                     n.width > 0 ? n.width : structure.nn_width, r.last_policy,
                     r.last_value, r.mean_policy, r.mean_value, r.norm)
              << std::endl;
    if (n.width > 0) fresh_by_width[n.width] = r.last_policy + r.last_value;
    results.push_back(r);
  }
  if (fresh_by_width.size() >= 2) {
    const auto narrow = fresh_by_width.begin();
    const auto wide = std::prev(fresh_by_width.end());
    std::cerr << absl::StrFormat(
        "test: fresh width %d fits to %.4f (policy + value), width %d to %.4f: the wider "
        "must fit better\n",
        wide->first, wide->second, narrow->first, narrow->second);
    if (wide->second >= narrow->second) {
      open_spiel::SpielFatalError("These settings do not tell the widths apart");
    }
  }
  return 0;
}
