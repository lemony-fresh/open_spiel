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

// A network of a new structure from a trained one and its replay buffer (thud/PLAN.md,
// a tool to restructure a network; the user, 2026-10-10: build it, to test whether a
// convolutional policy head lets the network act on the multi-square hurls that the
// linear head ignores).
//
// Builds the new network (`nn_model`, by default the source's width and depth), copies
// into it every parameter and buffer of the source whose name holds no "policy" and
// whose shape still fits (the trunk and the value head: identical names in both
// models' output blocks), and, with freeze=copied, freezes them — no gradient, and
// their batch norms kept in evaluation mode — so that only the new policy head learns,
// on the trunk the source trained. Then trains it on the saved replay buffer (the
// trainer's loss, learning rate `learning_rate`, Thud's 8 symmetries at random, as the
// trainer's augmentation), holding out every 20th block of 1,024 positions; every
// `eval_every` batches it reports the held-out losses (per side), and keeps the best
// network by the held-out policy loss, stopping after `patience` reports without a
// better one or at `max_batches`.
//
// Tests, printed and fatal if they fail: before training the new network's values
// equal the source's on held-out positions (the copy); after it, with freeze=copied,
// again (the freeze, batch norms included). Reported for comparison: the source's own
// held-out policy loss (its linear head on the same positions).
//
// Output: `to` (must not exist) with vpnet.pb (the new structure, `learning_rate`) and
// checkpoint-STEP (the best), which az_match and the probes read like any run's.
//
// Net surgery (mode 1, thud/PLAN.md, the input planes; 2026-10-11): input_planes=1 builds
// the new network with the 50 input planes from the rules (thud/az/thud_features.h)
// besides the observation; copy=all copies the policy head too; the first convolution,
// wider now, gets the source's channels and zeros for the new ones (CopyParametersFrom's
// widen_inputs). The test then requires the values and the policies to equal the
// source's exactly. With max_batches=0 nothing is trained (no buffer needed: the test
// uses positions of random games) and the network is saved with a fresh optimizer at
// `learning_rate`. When the new network takes the input planes and is trained, each
// report adds how its new input weights grow: their size per plane (the median and the
// largest over the planes), their spread across the filters, and how much the outputs
// move when they are zeroed (the planes' effect: half the policies' L1 distance, and
// the values' mean absolute difference, on the test positions).
//
//   thud/az/build.sh thud/az/az_restructure.cc
//   OMP_NUM_THREADS=8 build-shared/az_restructure from=RUN_DIR:STEP to=NEW_DIR \
//     buffer=RUN_DIR/replay_buffer.data [buffer_size=458752 nn_model=resnet_conv_policy \
//     freeze=copied learning_rate=0.001 batch_size=1024 eval_every=250 patience=4 \
//     max_batches=20000 step=STEP seed=1 input_planes=0 copy=not_policy device=/cpu:0]
//   build-shared/az_restructure from=RUN_DIR:STEP to=NEW_DIR input_planes=1 copy=all \
//     freeze=none max_batches=0 learning_rate=0.0001   # net surgery alone

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/model.h"
#include "thud/az/replay_buffer.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::ReplayBuffer;
using open_spiel::thud_az::torch_az::ModelConfig;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;

// The trainer's buffer, its positions readable in place.
class Buffer : public ReplayBuffer<VPNetModel::TrainInputs> {
 public:
  using ReplayBuffer<VPNetModel::TrainInputs>::ReplayBuffer;
  const std::vector<VPNetModel::TrainInputs>& Positions() const { return this->data_; }
};

// The held-out losses over `positions`, in batches.
VPNetModel::LossInfo HeldOut(VPNetModel& model,
                             const std::vector<VPNetModel::TrainInputs>& positions) {
  VPNetModel::LossInfo total;
  for (size_t start = 0; start < positions.size(); start += 1024) {
    std::vector<VPNetModel::TrainInputs> batch(
        positions.begin() + start,
        positions.begin() + std::min(positions.size(), start + 1024));
    total += model.Loss(batch);
  }
  return total;
}

// The largest differences between two networks' values and policies on `positions`.
std::pair<double, double> Differences(VPNetModel& a, VPNetModel& b,
                                      const std::vector<VPNetModel::TrainInputs>& positions) {
  std::vector<VPNetModel::InferenceInputs> in;
  for (const auto& p : positions) in.push_back({p.legal_actions, p.observations});
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

// Positions of random games, for the tests when there is no buffer.
std::vector<VPNetModel::TrainInputs> RandomPositions(const open_spiel::Game& game,
                                                     std::mt19937* rng, int count) {
  std::vector<VPNetModel::TrainInputs> out;
  while (out.size() < count) {
    std::unique_ptr<open_spiel::State> s = game.NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 300 && out.size() < count; ++ply) {
      if (ply % 7 == 3) out.push_back({s->LegalActions(), s->ObservationTensor(), {}, 0});
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      s->ApplyAction(legal[(*rng)() % legal.size()]);
    }
  }
  return out;
}

// The new input weights (the first convolution's channels past the observation's):
// their size per plane (median, largest), their spread across the filters (the mean
// over planes of the standard deviation over filters of each filter's weights' size),
// and the planes' effect — how much the outputs move when those weights are zeroed.
std::string InputPlaneStats(VPNetModel& model,
                            const std::vector<VPNetModel::TrainInputs>& positions) {
  torch::Tensor w = model.Parameter("layers.0.input_conv.weight");
  const int64_t first = thud::kNumObservationPlanes;
  std::vector<double> norms;
  double spread = 0;
  {
    torch::NoGradGuard no_grad;
    for (int64_t c = first; c < w.size(1); ++c) {
      torch::Tensor channel = w.select(1, c);                    // filters x 3 x 3
      norms.push_back(channel.norm().item<double>());
      spread += channel.flatten(1).norm(2, 1).std().item<double>();
    }
  }
  spread /= norms.size();
  std::vector<double> sorted = norms;
  std::sort(sorted.begin(), sorted.end());
  std::vector<VPNetModel::InferenceInputs> in;
  for (const auto& p : positions) in.push_back({p.legal_actions, p.observations});
  const auto with = model.Inference(in);
  torch::Tensor saved;
  {
    torch::NoGradGuard no_grad;
    saved = w.slice(1, first).clone();
    w.slice(1, first).zero_();
  }
  const auto without = model.Inference(in);
  {
    torch::NoGradGuard no_grad;
    w.slice(1, first).copy_(saved);
  }
  double policy = 0, value = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    double l1 = 0;
    for (size_t j = 0; j < with[i].policy.size(); ++j) {
      l1 += std::abs(with[i].policy[j].second - without[i].policy[j].second);
    }
    policy += l1 / 2;
    value += std::abs(with[i].value - without[i].value);
  }
  return absl::StrFormat(
      "\"input_planes\": {\"norm_median\": %.5f, \"norm_max\": %.5f, \"spread\": %.5f, "
      "\"policy_effect\": %.5f, \"value_effect\": %.5f}",
      sorted[sorted.size() / 2], sorted.back(), spread, policy / in.size(),
      value / in.size());
}

std::string Losses(const VPNetModel::LossInfo& l) {
  return absl::StrFormat(
      "\"policy\": %.4f, \"value\": %.4f, \"policy_dwarfs\": %.4f, \"policy_trolls\": %.4f",
      l.Policy(), l.Value(), l.Policy(0), l.Policy(1));
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
  const std::string from = get("from", ""), to = get("to", ""), buffer_path = get("buffer", "");
  const int max_batches = std::stoi(get("max_batches", "20000"));
  if (from.empty() || to.empty() || (buffer_path.empty() && max_batches > 0)) {
    std::cerr << "Usage: az_restructure from=DIR:STEP to=NEW_DIR buffer=PATH [...]"
              << std::endl;
    return 1;
  }
  const size_t colon = from.rfind(':');
  const std::string from_dir = from.substr(0, colon);
  const int from_step = std::stoi(from.substr(colon + 1));
  const std::string nn_model = get("nn_model", "resnet_conv_policy");
  const bool freeze = get("freeze", "copied") == "copied";
  const double learning_rate = std::stod(get("learning_rate", "0.001"));
  const int batch_size = std::stoi(get("batch_size", "1024"));
  const int eval_every = std::stoi(get("eval_every", "250"));
  const int patience = std::stoi(get("patience", "4"));
  const int buffer_size = std::stoi(get("buffer_size", "458752"));
  const int step = std::stoi(get("step", std::to_string(from_step)));
  const bool input_planes = get("input_planes", "0") == "1";
  const bool copy_all = get("copy", "not_policy") == "all";
  // Added 2026-10-15 (thud/PLAN.md, the GPU readiness checklist): where the networks run
  // and train (cpu, cuda:0, ...).
  const std::string device = get("device", "/cpu:0");
  std::mt19937 rng(std::stoull(get("seed", "1")));
  if (std::filesystem::exists(to)) open_spiel::SpielFatalError(to + " exists already");

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  ModelConfig source_config;
  {
    std::ifstream in(from_dir + "/vpnet.pb");
    in >> source_config;
  }
  const int nn_width = std::stoi(get("nn_width", std::to_string(source_config.nn_width)));
  const int nn_depth = std::stoi(get("nn_depth", std::to_string(source_config.nn_depth)));
  VPNetModel source(*game, from_dir, "vpnet.pb", device);
  source.LoadCheckpoint(from_step);

  std::filesystem::create_directories(to);
  SPIEL_CHECK_TRUE(open_spiel::thud_az::torch_az::CreateGraphDef(
      *game, learning_rate, source_config.weight_decay, to, "vpnet.pb", nn_model, nn_width,
      nn_depth, /*verbose=*/false,
      input_planes ? open_spiel::thud_az::kNumFeaturePlanes : 0));
  VPNetModel model(*game, to, "vpnet.pb", device);
  auto not_policy = [](const std::string& name) {
    return name.find("policy") == std::string::npos;
  };
  auto everything = [](const std::string&) { return true; };
  const std::function<bool(const std::string&)> include =
      copy_all ? std::function<bool(const std::string&)>(everything)
               : std::function<bool(const std::string&)>(not_policy);
  const std::vector<std::string> copied =
      model.CopyParametersFrom(source, include, /*widen_inputs=*/true);
  std::cerr << absl::StrFormat("copied %d tensors from %s:%d (%s, %d x %d) into %s (%s)\n",
                               copied.size(), from_dir, from_step, source_config.nn_model,
                               source_config.nn_width, source_config.nn_depth, to, nn_model);
  if (freeze) {
    std::cerr << absl::StrFormat("froze %d parameters\n", model.Freeze(include).size());
  }

  if (max_batches == 0) {  // Net surgery alone: test and save.
    const auto check = RandomPositions(*game, &rng, 200);
    const auto [value, policy] = Differences(model, source, check);
    std::cerr << absl::StrFormat(
        "test: on %d positions of random games the values differ from the source's by %g "
        "at most, the policies by %g%s\n",
        check.size(), value, policy, copy_all ? " (copy=all: both must be 0)" : "");
    if (value > 0 || (copy_all && policy > 0)) {
      open_spiel::SpielFatalError("The new network does not compute what the source does");
    }
    model.SaveCheckpoint(step);
    std::cout << absl::StrFormat(
                     "{\"summary\": true, \"from\": \"%s\", \"to\": \"%s\", \"nn_model\": "
                     "\"%s\", \"input_planes\": %s, \"copied\": %d, \"learning_rate\": %g}",
                     from, to, nn_model, input_planes ? "true" : "false", copied.size(),
                     model.LearningRate())
              << std::endl;
    return 0;
  }

  Buffer buffer(buffer_size);
  buffer.LoadBuffer(buffer_path);
  const auto& positions = buffer.Positions();
  std::vector<int> train;
  std::vector<VPNetModel::TrainInputs> held_out;
  for (int i = 0; i < positions.size(); ++i) {
    if ((i / 1024) % 20 == 0) {
      held_out.push_back(positions[i]);
    } else {
      train.push_back(i);
    }
  }
  std::cerr << absl::StrFormat("%d positions: %d to train on, %d held out\n",
                               positions.size(), train.size(), held_out.size());
  const std::vector<VPNetModel::TrainInputs> check(
      held_out.begin(), held_out.begin() + std::min<size_t>(1024, held_out.size()));

  // The copy's test: the same values as the source (and policies, with copy=all).
  const auto [copy_value, copy_policy] = Differences(model, source, check);
  std::cerr << absl::StrFormat(
      "test: values differ from the source's by %g at most, policies by %g\n", copy_value,
      copy_policy);
  if (copy_value > 1e-5 || (copy_all && copy_policy > 1e-5)) {
    open_spiel::SpielFatalError("The copy changed the network's outputs");
  }

  const VPNetModel::LossInfo source_loss = HeldOut(source, held_out);
  VPNetModel::LossInfo best = HeldOut(model, held_out);
  std::cout << absl::StrFormat("{\"source\": true, %s}", Losses(source_loss)) << std::endl;
  std::cout << absl::StrFormat("{\"batches\": 0, \"held_out\": {%s}%s}", Losses(best),
                               model.TakesInputPlanes()
                                   ? ", " + InputPlaneStats(model, check)
                                   : std::string())
            << std::endl;
  model.SaveCheckpoint(step);
  int best_batches = 0, since_best = 0;
  std::uniform_int_distribution<int> pick(0, train.size() - 1);
  std::uniform_int_distribution<int> symmetry(0, thud::kNumSymmetries - 1);
  VPNetModel::LossInfo trained;
  const auto start = std::chrono::steady_clock::now();
  for (int b = 1; b <= max_batches; ++b) {
    std::vector<VPNetModel::TrainInputs> batch;
    batch.reserve(batch_size);
    for (int j = 0; j < batch_size; ++j) {
      batch.push_back(open_spiel::thud_az::torch_az::SymmetricTrainInputs(
          positions[train[pick(rng)]], symmetry(rng)));
    }
    trained += model.Learn(batch);
    if (b % eval_every == 0) {
      const VPNetModel::LossInfo now = HeldOut(model, held_out);
      const bool better = now.Policy() < best.Policy();
      std::cout << absl::StrFormat(
                       "{\"batches\": %d, \"seconds\": %.0f, \"trained\": {%s}, "
                       "\"held_out\": {%s}, \"best\": %s%s}",
                       b,
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                           .count(),
                       Losses(trained), Losses(now), better ? "true" : "false",
                       model.TakesInputPlanes() ? ", " + InputPlaneStats(model, check)
                                                : std::string())
                << std::endl;
      trained = VPNetModel::LossInfo();
      if (better) {
        best = now;
        best_batches = b;
        since_best = 0;
        model.SaveCheckpoint(step);
      } else if (++since_best >= patience) {
        break;
      }
    }
  }

  // The best network, and the freeze's test.
  VPNetModel saved(*game, to, "vpnet.pb", device);
  saved.LoadCheckpoint(step);
  const double freeze_difference = Differences(saved, source, check).first;
  std::cerr << absl::StrFormat(
      "test: the saved network's values differ from the source's by %g at most%s\n",
      freeze_difference, freeze ? " (frozen: must be ~0)" : "");
  if (freeze && freeze_difference > 1e-5) {
    open_spiel::SpielFatalError("The frozen trunk or value head changed");
  }
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"from\": \"%s\", \"to\": \"%s\", \"nn_model\": "
                   "\"%s\", \"freeze\": %s, \"best_batches\": %d, \"source_held_out\": {%s}, "
                   "\"held_out\": {%s}}",
                   from, to, nn_model, freeze ? "true" : "false", best_batches,
                   Losses(source_loss), Losses(best))
            << std::endl;
  return 0;
}
