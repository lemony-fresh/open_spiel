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
// Since 2026-10-19 (thud/az/position_buffer.h) the inputs may be in either format — the
// compact one or upstream's — and OUT is written compact, or with --legacy in upstream's
// format (for programs on upstream's AlphaZero and az_trainer_d1). A position's policy
// target comes through as the floats the learner reads (position_buffer.h), so a
// position of an upstream file and the same position of a compact one are one position.
// One input is a conversion: az_merge_buffers OUT SIZE IN [--legacy].
//
//   thud/build_programs.sh az_merge_buffers
//   build-shared/az_merge_buffers OUT MAX_SIZE IN1 IN2 ... [--legacy]

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/position_buffer.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::thud_az::torch_az::VPNetModel;
using Buffer = open_spiel::thud_az::PositionBuffer<VPNetModel::TrainInputs>;

uint64_t Fingerprint(const VPNetModel::TrainInputs& p) {
  return absl::Hash<std::tuple<std::vector<open_spiel::Action>, std::vector<float>,
                               open_spiel::ActionsAndProbs, double>>{}(
      std::make_tuple(p.legal_actions, p.observations, p.policy, p.value));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  const bool legacy = std::find(args.begin(), args.end(), "--legacy") != args.end();
  args.erase(std::remove(args.begin(), args.end(), "--legacy"), args.end());
  if (args.size() < 3) {
    std::cerr << "Usage: " << argv[0] << " OUT MAX_SIZE IN1 IN2 ... [--legacy]" << std::endl;
    return 1;
  }
  const std::string out_path = args[0];
  const int max_size = std::stoi(args[1]);
  std::vector<uint64_t> expected;  // Fingerprints, in the order added.
  std::unordered_set<uint64_t> seen;
  int64_t skipped = 0;
  {
    Buffer merged(max_size);
    for (size_t i = 2; i < args.size(); ++i) {
      Buffer in(Buffer::FileMaxSize(args[i]));
      in.LoadBuffer(args[i]);
      const int n = in.Size();
      for (int64_t slot : in.OldestFirst()) {
        const VPNetModel::TrainInputs p = in.At(slot);
        const uint64_t fingerprint = Fingerprint(p);
        if (!seen.insert(fingerprint).second) {
          ++skipped;  // Already added from an earlier input.
          continue;
        }
        merged.Add(p);
        expected.push_back(fingerprint);
      }
      std::cerr << absl::StrFormat("%s: %d positions (total added %d)\n", args[i], n,
                                   in.TotalAdded());
    }
    if (legacy) {
      merged.SaveLegacyBuffer(out_path);
    } else {
      merged.SaveBuffer(out_path);
    }
  }
  if (expected.size() > max_size) {
    expected.erase(expected.begin(), expected.end() - max_size);
  }
  // Read it back as the trainer will, and compare the positions.
  Buffer check(max_size);
  check.LoadBuffer(out_path);
  std::vector<uint64_t> got;
  for (int i = 0; i < check.Size(); ++i) got.push_back(Fingerprint(check.At(i)));
  std::vector<uint64_t> a = expected, b = got;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  const bool same = a == b;
  // Control: the fingerprints tell positions apart (a constant one would pass anything).
  const int distinct = std::unique(a.begin(), a.end()) - a.begin();
  std::cout << absl::StrFormat(
                   "{\"out\": \"%s\", \"max_size\": %d, \"positions\": %d, "
                   "\"total_added\": %d, \"distinct_fingerprints\": %d, "
                   "\"duplicates_skipped\": %d, \"same_positions_as_inputs\": %s, "
                   "\"format\": \"%s\", \"compact_bytes\": %d}",
                   out_path, max_size, check.Size(), check.TotalAdded(), distinct, skipped,
                   same ? "true" : "false", legacy ? "legacy" : "compact", check.Bytes())
            << std::endl;
  return same ? 0 : 1;
}
