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

// Has a network forgotten what it learned earlier in its run? (thud/PLAN.md Phase 6,
// Step 2's quick check.) A trainer run keeps every self-play position in its archive
// (archive/replay_buffer_stepN.data: the replay buffer after step N, the positions of
// steps N-2 to N). From each archive this takes the same random sample of positions and
// measures, for each network, per side to move, how well it predicts what was recorded
// there: the value error against the game's final margin (squared, the trainer's value
// loss) and the move-probability loss against the search's visit counts (cross-entropy,
// the trainer's policy loss), and how often its most likely move is the most visited.
// A network that forgot would do clearly worse on the old archives than an earlier
// network that had just trained on them, while doing better on the new ones.
//
//   thud/az/build.sh thud/experiments/az_forgetting.cc
//   OMP_NUM_THREADS=4 build-shared/az_forgetting RUN_DIR steps=0,16,29 \
//     archives=3,6,9,12,15,18,21,24,27,30 [sample=4096 seed=1 archive_dir=RUN_DIR]
//
// archive_dir takes the archives from another run, e.g. to compare a branch's networks
// on the positions of the run it branched from.
//
// Prints one JSON line per archive, network and side.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>

#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/serializable_circular_buffer.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::torch_az::VPNetModel;

constexpr int kPlaneSize = 15 * 15;  // Thud's observation: 6 planes of 15 x 15.
constexpr int kTrollsToMovePlane = 3;
constexpr int kBatch = 256;

std::vector<int> IntList(const std::string& text) {
  std::vector<int> out;
  for (absl::string_view part : absl::StrSplit(text, ',')) {
    int value;
    if (!absl::SimpleAtoi(part, &value)) {
      open_spiel::SpielFatalError(absl::StrCat("Not a number: ", part));
    }
    out.push_back(value);
  }
  return out;
}

// The positions of one archive: a fixed random sample.
std::vector<VPNetModel::TrainInputs> Sample(const std::string& path, int sample,
                                            int seed) {
  int max_size;
  {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    if (!deserializer.Read(&max_size)) {
      open_spiel::SpielFatalError(absl::StrCat("Cannot read ", path));
    }
  }
  open_spiel::SerializableCircularBuffer<VPNetModel::TrainInputs> buffer(max_size);
  buffer.LoadBuffer(path);
  std::vector<VPNetModel::TrainInputs> all = buffer.Data();
  std::mt19937 rng(seed);
  std::shuffle(all.begin(), all.end(), rng);
  if (all.size() > sample) all.resize(sample);
  return all;
}

struct Totals {
  double value_error = 0, policy_loss = 0;
  int top_agree = 0, count = 0;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " RUN_DIR steps=S1,S2,... archives=A1,A2,... [sample=4096 seed=1]"
              << std::endl;
    return 1;
  }
  const std::string dir = argv[1];
  std::vector<int> steps, archives;
  int sample = 4096, seed = 1;
  std::string archive_dir = dir;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("steps=", 0) == 0) steps = IntList(arg.substr(6));
    else if (arg.rfind("archives=", 0) == 0) archives = IntList(arg.substr(9));
    else if (arg.rfind("sample=", 0) == 0) sample = std::stoi(arg.substr(7));
    else if (arg.rfind("seed=", 0) == 0) seed = std::stoi(arg.substr(5));
    else if (arg.rfind("archive_dir=", 0) == 0) archive_dir = arg.substr(12);
    else open_spiel::SpielFatalError(absl::StrCat("Unknown argument ", arg));
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  std::vector<std::vector<VPNetModel::TrainInputs>> samples;
  for (int archive : archives) {
    samples.push_back(Sample(
        absl::StrCat(archive_dir, "/archive/replay_buffer_step", archive, ".data"), sample,
        seed));
  }

  for (int step : steps) {
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(step);
    for (int a = 0; a < archives.size(); ++a) {
      const std::vector<VPNetModel::TrainInputs>& positions = samples[a];
      Totals side[2];
      for (int start = 0; start < positions.size(); start += kBatch) {
        const int end = std::min<int>(start + kBatch, positions.size());
        std::vector<VPNetModel::InferenceInputs> inputs;
        for (int i = start; i < end; ++i) {
          inputs.push_back({positions[i].legal_actions, positions[i].observations});
        }
        const std::vector<VPNetModel::InferenceOutputs> outputs = model.Inference(inputs);
        for (int i = start; i < end; ++i) {
          const VPNetModel::TrainInputs& p = positions[i];
          const VPNetModel::InferenceOutputs& out = outputs[i - start];
          Totals& t = side[p.observations[kTrollsToMovePlane * kPlaneSize] > 0 ? 1 : 0];
          t.value_error += (out.value - p.value) * (out.value - p.value);
          // Cross-entropy of the network's move probabilities against the visit counts.
          double loss = 0;
          for (const auto& [action, target] : p.policy) {
            if (target <= 0) continue;
            double predicted = 0;
            for (const auto& [a2, prob] : out.policy) {
              if (a2 == action) {
                predicted = prob;
                break;
              }
            }
            loss -= target * std::log(std::max(predicted, 1e-12));
          }
          t.policy_loss += loss;
          auto most = [](const open_spiel::ActionsAndProbs& v) {
            return std::max_element(v.begin(), v.end(), [](const auto& x, const auto& y) {
                     return x.second < y.second;
                   })->first;
          };
          t.top_agree += most(out.policy) == most(p.policy);
          ++t.count;
        }
      }
      for (int s = 0; s < 2; ++s) {
        const Totals& t = side[s];
        if (t.count == 0) continue;
        std::cout << absl::StrFormat(
                         "{\"archive\": %d, \"network\": %d, \"side\": \"%s\", "
                         "\"positions\": %d, \"value_error\": %.4f, \"policy_loss\": %.4f, "
                         "\"top_move_agrees\": %.3f}",
                         archives[a], step, s == 0 ? "dwarfs" : "trolls", t.count,
                         t.value_error / t.count, t.policy_loss / t.count,
                         static_cast<double>(t.top_agree) / t.count)
                  << std::endl;
      }
    }
  }
  return 0;
}
