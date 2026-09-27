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

// How widely do the searches in a Thud self-play run spread their visits? Reads the
// replay buffer the OpenSpiel C++ AlphaZero trainer saves after every learning step
// (replay_buffer.data): each position's policy target is its root's visit counts,
// normalised (alpha_zero.cc, PlayGame). Reports per side to move: the legal moves, the
// moves that got any visit, the most visited move's share, the entropy of the visit
// distribution as an effective number of moves (exp(entropy)), and the mean value
// target, which is the dwarfs' return for every position (alpha_zero.cc:363-369).
//
//   thud/experiments/build_az_program.sh az_buffer_stats
//   build-shared/az_buffer_stats RUN_DIR/replay_buffer.data
//
// Prints one JSON line.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serializable_circular_buffer.h"

namespace {

using open_spiel::algorithms::torch_az::VPNetModel;

constexpr int kPlaneSize = 15 * 15;         // Thud's observation: 6 planes of 15 x 15.
constexpr int kTrollsToMovePlane = 3;

double Median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

double Mean(const std::vector<double>& v) {
  double sum = 0;
  for (double x : v) sum += x;
  return v.empty() ? 0 : sum / v.size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " RUN_DIR/replay_buffer.data" << std::endl;
    return 1;
  }
  const std::string path = argv[1];
  int max_size;
  {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    if (!deserializer.Read(&max_size)) {
      open_spiel::SpielFatalError(absl::StrCat("Cannot read ", path));
    }
  }
  open_spiel::SerializableCircularBuffer<VPNetModel::TrainInputs> buffer(max_size);
  buffer.LoadBuffer(path);

  struct Side {
    std::vector<double> legal, visited, top_share, effective, value;
  } sides[2];  // 0: dwarfs to move, 1: trolls.
  for (const VPNetModel::TrainInputs& p : buffer.Data()) {
    Side& side = sides[p.observations[kTrollsToMovePlane * kPlaneSize] > 0 ? 1 : 0];
    int visited = 0;
    double top = 0, entropy = 0;
    for (const auto& [action, probability] : p.policy) {
      if (probability <= 0) continue;
      ++visited;
      top = std::max(top, probability);
      entropy -= probability * std::log(probability);
    }
    side.legal.push_back(p.legal_actions.size());
    side.visited.push_back(visited);
    side.top_share.push_back(top);
    side.effective.push_back(std::exp(entropy));
    side.value.push_back(p.value);
  }
  std::string out = absl::StrFormat("{\"positions\": %d, \"max_size\": %d",
                                    buffer.Size(), max_size);
  const char* names[2] = {"dwarfs", "trolls"};
  for (int s = 0; s < 2; ++s) {
    const Side& side = sides[s];
    absl::StrAppend(
        &out, absl::StrFormat(
                  ", \"%s\": {\"positions\": %d, \"median_legal\": %.0f, "
                  "\"median_visited\": %.0f, \"mean_visited\": %.1f, "
                  "\"median_top_share\": %.3f, \"median_effective_moves\": %.1f, "
                  "\"mean_dwarfs_return_target\": %.3f}",
                  names[s], side.legal.size(), Median(side.legal), Median(side.visited),
                  Mean(side.visited), Median(side.top_share), Median(side.effective),
                  Mean(side.value)));
  }
  std::cout << out << "}" << std::endl;
  return 0;
}
