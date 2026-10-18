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

// A trained network's batch norms made to normalise by the true statistics of what reaches
// them, its function unchanged (VPNetModel::ConvertBatchNorms; 2026-10-17,
// thud/DESIGN_dwarf_signal.md §10): for a network that learnt with its batch norms in
// evaluation mode — the race on a shared device — so that it can go on learning in
// training mode without a jump. The statistics come from `batches` batches of the run's
// buffer, Thud's symmetries at random, as the trainer samples them.
//
// Tests, fatal, on 4 further batches (`batch_size` positions each, not among those):
//   - evaluation mode: the converted network's values and policies equal the source's to
//     1e-4;
//   - training mode: its losses come within a fifth of their distance from evaluation
//     mode's that the source's had (the source's distance reported: the control).
//
// Writes `to`/vpnet.pb (the source's) and `to`/checkpoint-STEP (the converted weights, the
// source's optimizer), which the trainer and every probe read like any run's.
//
//   thud/build_programs.sh az_batch_norm_convert
//   OMP_NUM_THREADS=8 build-shared/az_batch_norm_convert from=RUN_DIR:STEP to=NEW_DIR \
//     buffer=RUN_DIR/replay_buffer.data [buffer_size=458752 batches=32 batch_size=1024 \
//     seed=1 device=/cpu:0]

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/replay_buffer.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::ReplayBuffer;
using open_spiel::thud_az::torch_az::VPNetModel;

double MaxDifference(VPNetModel& a, VPNetModel& b,
                     const std::vector<VPNetModel::TrainInputs>& positions) {
  std::vector<VPNetModel::InferenceInputs> in;
  for (const auto& p : positions) in.push_back({p.legal_actions, p.observations});
  const auto va = a.Inference(in), vb = b.Inference(in);
  double d = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    d = std::max(d, std::abs(va[i].value - vb[i].value));
    for (size_t j = 0; j < va[i].policy.size(); ++j) {
      d = std::max(d, std::abs(va[i].policy[j].second - vb[i].policy[j].second));
    }
  }
  return d;
}

// Training mode's losses against evaluation mode's, summed over the test batches.
struct Distance {
  double value = 0, policy = 0;
  double Total() const { return value + policy; }
};
Distance ModeDistance(VPNetModel& model,
                      const std::vector<std::vector<VPNetModel::TrainInputs>>& tests) {
  Distance d;
  for (const auto& batch : tests) {
    const auto trained = model.LossInTrainingMode(batch), plays = model.Loss(batch);
    d.value += std::abs(trained.Value() - plays.Value()) / tests.size();
    d.policy += std::abs(trained.Policy() - plays.Policy()) / tests.size();
  }
  return d;
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
  if (from.empty() || to.empty() || buffer_path.empty()) {
    std::cerr << "Usage: az_batch_norm_convert from=DIR:STEP to=NEW_DIR buffer=PATH [...]"
              << std::endl;
    return 1;
  }
  const size_t colon = from.rfind(':');
  const std::string from_dir = from.substr(0, colon);
  const int step = std::stoi(from.substr(colon + 1));
  const int batches = std::stoi(get("batches", "32"));
  const int batch_size = std::stoi(get("batch_size", "1024"));
  const int buffer_size = std::stoi(get("buffer_size", "458752"));
  const std::string device = get("device", "/cpu:0");
  std::mt19937 rng(std::stoull(get("seed", "1")));
  if (std::filesystem::exists(to)) open_spiel::SpielFatalError(to + " exists already");

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  ReplayBuffer<VPNetModel::TrainInputs> buffer(buffer_size);
  buffer.LoadBuffer(buffer_path);
  const int tests = 4;
  if (buffer.Size() < (batches + tests) * batch_size) {
    open_spiel::SpielFatalError(absl::StrFormat("The buffer has %d positions, %d needed",
                                                buffer.Size(), (batches + tests) * batch_size));
  }
  std::vector<VPNetModel::TrainInputs> drawn =
      buffer.Sample(&rng, (batches + tests) * batch_size);  // Distinct positions.
  std::shuffle(drawn.begin(), drawn.end(), rng);
  std::uniform_int_distribution<int> symmetry(0, open_spiel::thud::kNumSymmetries - 1);
  std::vector<std::vector<VPNetModel::TrainInputs>> statistics, test_batches;
  for (int b = 0; b < batches + tests; ++b) {
    std::vector<VPNetModel::TrainInputs> batch;
    for (int i = 0; i < batch_size; ++i) {
      batch.push_back(open_spiel::thud_az::torch_az::SymmetricTrainInputs(
          drawn[b * batch_size + i], symmetry(rng)));
    }
    (b < batches ? statistics : test_batches).push_back(std::move(batch));
  }

  VPNetModel source(*game, from_dir, "vpnet.pb", device);
  source.LoadCheckpoint(step);
  std::filesystem::create_directories(to);
  std::filesystem::copy_file(from_dir + "/vpnet.pb", to + "/vpnet.pb");
  VPNetModel model(*game, to, "vpnet.pb", device);
  model.LoadCheckpoint(absl::StrCat(from_dir, "/checkpoint-", step));  // With its optimizer.

  const Distance before = ModeDistance(source, test_batches);
  const auto conversion = model.ConvertBatchNorms(statistics);
  for (const auto& c : conversion) {
    std::cout << absl::StrFormat(
                     "{\"batch_norm\": \"%s\", \"largest_mean_shift\": %.4f, "
                     "\"deviation_ratio\": [%.4f, %.4f]}",
                     c.name, c.largest_mean_shift, c.smallest_ratio, c.largest_ratio)
              << std::endl;
  }
  double outputs = 0;
  for (const auto& batch : test_batches) {
    outputs = std::max(outputs, MaxDifference(model, source, batch));
  }
  const Distance after = ModeDistance(model, test_batches);
  std::cerr << absl::StrFormat(
      "test: evaluation-mode outputs differ from the source's by %g at most (1e-4 allowed)\n"
      "test: training mode's losses from evaluation mode's: the source's %.4f (value %.4f, "
      "policy %.4f), the converted network's %.4f (value %.4f, policy %.4f); a fifth of the "
      "source's allowed\n",
      outputs, before.Total(), before.value, before.policy, after.Total(), after.value,
      after.policy);
  if (outputs > 1e-4) open_spiel::SpielFatalError("The conversion changed the network");
  if (after.Total() > before.Total() / 5) {
    open_spiel::SpielFatalError("Training mode still computes unlike evaluation mode");
  }
  model.SaveCheckpoint(step);
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"from\": \"%s\", \"to\": \"%s\", \"batches\": %d, "
                   "\"batch_size\": %d, \"outputs_differ\": %.3g, \"mode_distance_before\": "
                   "{\"value\": %.5f, \"policy\": %.5f}, \"mode_distance_after\": "
                   "{\"value\": %.5f, \"policy\": %.5f}}",
                   from, to, batches, batch_size, outputs, before.value, before.policy,
                   after.value, after.policy)
            << std::endl;
  return 0;
}
