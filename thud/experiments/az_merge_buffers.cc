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

// Builds one replay buffer file from archived ones, so that a trainer run can resume
// with a larger buffer (thud/PLAN.md Phase 6, Step 2): upstream's LoadBuffer refuses a
// file of another size, but loads this one. Takes each input's positions in the order
// they were added (a full circular buffer starts at total_added % size) and adds them,
// inputs in the order given, to a buffer of MAX_SIZE, which keeps the last MAX_SIZE.
// A position already added (the same observation, legal moves, targets) is skipped, so
// overlapping inputs — a run's archive and its later buffer — give each position once.
// Then reads the file back and checks that it holds exactly those positions.
//
//   thud/az/build.sh thud/experiments/az_merge_buffers.cc
//   build-shared/az_merge_buffers OUT MAX_SIZE IN1 IN2 ...

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serializable_circular_buffer.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::torch_az::VPNetModel;
using Buffer = open_spiel::SerializableCircularBuffer<VPNetModel::TrainInputs>;

int SavedMaxSize(const std::string& path) {
  int max_size;
  nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
  if (!deserializer.Read(&max_size)) {
    open_spiel::SpielFatalError(absl::StrCat("Cannot read ", path));
  }
  return max_size;
}

uint64_t Fingerprint(const VPNetModel::TrainInputs& p) {
  return absl::Hash<std::tuple<std::vector<open_spiel::Action>, std::vector<float>,
                               open_spiel::ActionsAndProbs, double>>{}(
      std::make_tuple(p.legal_actions, p.observations, p.policy, p.value));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " OUT MAX_SIZE IN1 IN2 ..." << std::endl;
    return 1;
  }
  const std::string out_path = argv[1];
  const int max_size = std::stoi(argv[2]);
  std::vector<uint64_t> expected;  // Fingerprints, in the order added.
  std::unordered_set<uint64_t> seen;
  int64_t skipped = 0;
  {
    Buffer merged(max_size);
    for (int i = 3; i < argc; ++i) {
      Buffer in(SavedMaxSize(argv[i]));
      in.LoadBuffer(argv[i]);
      const std::vector<VPNetModel::TrainInputs>& data = in.Data();
      const int n = data.size();
      // A full buffer overwrote in place: its oldest position is at total_added % n.
      const int start = n == SavedMaxSize(argv[i]) ? in.TotalAdded() % n : 0;
      for (int k = 0; k < n; ++k) {
        const VPNetModel::TrainInputs& p = data[(start + k) % n];
        const uint64_t fingerprint = Fingerprint(p);
        if (!seen.insert(fingerprint).second) {
          ++skipped;  // Already added from an earlier input.
          continue;
        }
        merged.Add(p);
        expected.push_back(fingerprint);
      }
      std::cerr << absl::StrFormat("%s: %d positions (total added %d)\n", argv[i], n,
                                   in.TotalAdded());
    }
    merged.SaveBuffer(out_path);
  }
  if (expected.size() > max_size) {
    expected.erase(expected.begin(), expected.end() - max_size);
  }
  // Read it back as the trainer will, and compare the positions.
  Buffer check(max_size);
  check.LoadBuffer(out_path);
  std::vector<uint64_t> got;
  for (const VPNetModel::TrainInputs& p : check.Data()) got.push_back(Fingerprint(p));
  std::vector<uint64_t> a = expected, b = got;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  const bool same = a == b;
  // Control: the fingerprints tell positions apart (a constant one would pass anything).
  const int distinct = std::unique(a.begin(), a.end()) - a.begin();
  std::cout << absl::StrFormat(
                   "{\"out\": \"%s\", \"max_size\": %d, \"positions\": %d, "
                   "\"total_added\": %d, \"distinct_fingerprints\": %d, "
                   "\"duplicates_skipped\": %d, \"same_positions_as_inputs\": %s}",
                   out_path, max_size, check.Size(), check.TotalAdded(), distinct, skipped,
                   same ? "true" : "false")
            << std::endl;
  return same ? 0 : 1;
}
