// Copyright 2026 The Thud-on-OpenSpiel authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A run's replay buffer as it was earlier (thud/PLAN.md, runs G and G': branching from
// G'64 when only step 73's buffer file is left — the trainer overwrites it each step).
// A full buffer of max_size M with T positions added holds in slot s the position added
// last there, number i = s + M floor((T - 1 - s) / M); the buffer as it was after K
// positions held those with i < K that are still there — all but the T - K added since,
// which overwrote the oldest. This keeps the positions with i < K, in the order they
// were added, in a new buffer of the same max_size (its count of positions added: how
// many it keeps). The positions are read and written as the trainer's TrainInputs
// (thud/az/vpnet.h), mirrored field for field so that no LibTorch is needed.
//
//   build-shared/az_buffer_truncate self_test
//       the index arithmetic on a small buffer, with wrap-around, against hand-worked
//       values
//   build-shared/az_buffer_truncate copy IN OUT
//       load and save unchanged: OUT must equal IN byte for byte (the control that the
//       mirror reads the trainer's format exactly)
//   build-shared/az_buffer_truncate keep IN OUT K
//
// Built like replay_buffer_test (thud/az/replay_buffer_test.cc), without LibTorch:
//   clang++ -std=gnu++20 -O2 -I. -Iopen_spiel -Iopen_spiel/abseil-cpp \
//     -Iopen_spiel/json/include -Iopen_spiel/libnop/libnop/include -DOPEN_SPIEL_BUILD_WITH_LIBNOP \
//     thud/experiments/az_buffer_truncate.cc -Lbuild-shared -lopen_spiel \
//     -Wl,-rpath,$PWD/build-shared -o build-shared/az_buffer_truncate

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "nop/structure.h"
// libnop's -Wmissing-template-arg-list-after-template-kw, as in replay_buffer_test.cc.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-template-arg-list-after-template-kw"
#include "open_spiel/utils/serializable_circular_buffer.h"
#pragma clang diagnostic pop
#include "thud/az/replay_buffer.h"

namespace {

using Action = int64_t;
using ActionsAndProbs = std::vector<std::pair<Action, double>>;

// The trainer's VPNetModel::TrainInputs, field for field (thud/az/vpnet.h).
struct TrainInputs {
  std::vector<Action> legal_actions;
  std::vector<float> observations;
  ActionsAndProbs policy;
  double value;

  NOP_STRUCTURE(TrainInputs, legal_actions, observations, policy, value);
};

// Exposes the stored elements and the buffer's size, which upstream keeps protected.
template <class T>
class Buffer : public open_spiel::thud_az::ReplayBuffer<T> {
 public:
  using open_spiel::thud_az::ReplayBuffer<T>::ReplayBuffer;
  const std::vector<T>& Elements() const { return this->data_; }
  int MaxSize() const { return this->max_size_; }
};

// The slots of a buffer of `max_size` with `total` added whose positions were added
// before the first `keep`, in the order they were added.
std::vector<int64_t> KeptSlots(int64_t max_size, int64_t size, int64_t total,
                               int64_t keep) {
  std::vector<std::pair<int64_t, int64_t>> by_number;  // (number added, slot)
  for (int64_t slot = 0; slot < size; ++slot) {
    const int64_t number =
        size < max_size ? slot : slot + max_size * ((total - 1 - slot) / max_size);
    if (number < keep) by_number.push_back({number, slot});
  }
  std::sort(by_number.begin(), by_number.end());
  std::vector<int64_t> slots;
  for (const auto& [number, slot] : by_number) slots.push_back(slot);
  return slots;
}

int SelfTest() {
  // max_size 10, 27 added (numbers 0-26): slots 0-6 hold 20-26, slots 7-9 hold 17-19.
  const std::vector<int64_t> kept = KeptSlots(10, 10, 27, 22);
  const std::vector<int64_t> expected = {7, 8, 9, 0, 1};  // Numbers 17, 18, 19, 20, 21.
  bool ok = kept == expected;
  // Not yet full: max_size 10, 6 added, keep 4 — slots 0-3.
  ok = ok && KeptSlots(10, 6, 6, 4) == std::vector<int64_t>({0, 1, 2, 3});
  // Keeping everything added so far keeps every slot, oldest first: 27 added, keep 27.
  ok = ok && KeptSlots(10, 10, 27, 27) ==
                 std::vector<int64_t>({7, 8, 9, 0, 1, 2, 3, 4, 5, 6});
  // And through a real buffer: values are the numbers added.
  Buffer<TrainInputs> buffer(10);
  for (int i = 0; i < 27; ++i) {
    TrainInputs p;
    p.value = i;
    buffer.Add(std::move(p));
  }
  std::vector<double> values;
  for (int64_t slot : KeptSlots(10, buffer.Size(), buffer.TotalAdded(), 22)) {
    values.push_back(buffer.Elements()[slot].value);
  }
  ok = ok && values == std::vector<double>({17, 18, 19, 20, 21});
  std::cout << (ok ? "PASS" : "FAIL") << " self test: the kept slots and their order"
            << std::endl;
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "self_test") return SelfTest();
  if ((mode != "copy" && mode != "keep") || argc < (mode == "keep" ? 5 : 4)) {
    std::cerr << "usage: az_buffer_truncate self_test | copy IN OUT | keep IN OUT K"
              << std::endl;
    return 2;
  }
  // The size the file was saved with: LoadBuffer checks it, so read it from a probe.
  int max_size = 0;
  {
    nop::Deserializer<nop::StreamReader<std::ifstream>> probe{argv[2]};
    probe.Read(&max_size);
  }
  Buffer<TrainInputs> in(max_size);
  in.LoadBuffer(argv[2]);
  std::cout << "loaded " << in.Size() << " positions of max_size " << in.MaxSize()
            << ", " << in.TotalAdded() << " added" << std::endl;
  if (mode == "copy") {
    in.SaveBuffer(argv[3]);
    std::cout << "saved unchanged to " << argv[3] << std::endl;
    return 0;
  }
  const int64_t keep = std::stoll(argv[4]);
  const std::vector<int64_t> slots = KeptSlots(in.MaxSize(), in.Size(), in.TotalAdded(), keep);
  Buffer<TrainInputs> out(in.MaxSize());
  for (int64_t slot : slots) out.Add(in.Elements()[slot]);
  out.SaveBuffer(argv[3]);
  std::cout << "kept " << out.Size() << " of " << in.Size()
            << " positions, those among the first " << keep << " added; dropped the "
            << in.Size() - out.Size() << " added after them; saved to " << argv[3]
            << std::endl;
  return 0;
}
