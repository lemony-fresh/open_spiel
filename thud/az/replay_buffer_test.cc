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

// Tests ReplayBuffer (thud/az/replay_buffer.h), whose Add moves each position into
// its slot, against upstream's SerializableCircularBuffer, whose Add copies it in. Both
// get the same positions, shaped like the trainer's (the legal moves, a 6 x 15 x 15
// observation, the policy over the legal moves, a value): games of 50-400 positions,
// the two sides alternating, dwarf positions with 150-330 legal moves and troll
// positions with 10-40. A buffer of 2,000 is filled, saved and loaded into fresh
// buffers (as a resumed run starts), then gets 3 more buffers' worth.
//
//   1. Same contents: every slot's legal moves, observation, policy and value equal,
//      and the same size and number added.
//   2. Same samples: Sample with equal seeds draws equal positions.
//   3. Same saved file, byte for byte.
//   0. Loading: ours holds exactly what its positions need right after LoadBuffer;
//      upstream's holds more (the control: libnop's spare capacity).
//   4. Memory: in ours every slot's vectors hold exactly their sizes; in upstream's
//      the slots keep larger storage (the control: the test sees the ratchet).
//   5. A file cut short (2026-10-15; a run killed while writing its buffer in place):
//      ours refuses to load the saved file cut to 99.9% and to half (each load in a
//      child process, which must stop with an error); upstream's loads them without
//      an error — the control, and the reason ours verifies its reads — keeping only the positions
//      before the cut, its count of positions added still the file's. The file whole
//      loads in a child too (the control that the child can succeed).
//
// Needs only headers and libopen_spiel.so (no LibTorch):
//   thud/run_tests.sh replay_buffer_test   # builds and runs it
//   build-tests/bin/replay_buffer_test [DIR]   # DIR for its files (default /tmp)

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "nop/structure.h"
// libnop (third-party, reached through this header) writes calls like
// `value_.template Construct(args...)`: the `template` keyword without a template
// argument list, which clang here rejects by default
// (-Wmissing-template-arg-list-after-template-kw). The arguments are deduced from the
// call, so the code means the same with or without the keyword; upstream builds
// libnop with every warning off (open_spiel/CMakeLists.txt). Ignored for this include
// only.
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

bool Equal(const TrainInputs& a, const TrainInputs& b) {
  return a.legal_actions == b.legal_actions && a.observations == b.observations &&
         a.policy == b.policy && a.value == b.value;
}

// The positions of one game: sides alternate, dwarfs first.
std::vector<TrainInputs> Game(std::mt19937* rng) {
  std::vector<TrainInputs> game;
  const int length = std::uniform_int_distribution<int>(50, 400)(*rng);
  const double value = std::uniform_real_distribution<double>(-1, 1)(*rng);
  for (int i = 0; i < length; ++i) {
    const bool dwarfs = i % 2 == 0;
    const int legal = dwarfs ? std::uniform_int_distribution<int>(150, 330)(*rng)
                             : std::uniform_int_distribution<int>(10, 40)(*rng);
    TrainInputs p;
    for (int a = 0; a < legal; ++a) {  // push_back: vectors with spare capacity.
      p.legal_actions.push_back((*rng)() % 19800);
      p.policy.push_back({p.legal_actions.back(), 1.0 / legal});
    }
    p.observations.assign(6 * 15 * 15, 0.0f);
    p.observations[(*rng)() % p.observations.size()] = 1.0f;
    p.value = value;
    game.push_back(std::move(p));
  }
  return game;
}

// The bytes the slots' vectors hold, and the bytes their elements need.
template <class Buffer>
std::pair<int64_t, int64_t> Bytes(const Buffer& buffer) {
  int64_t held = 0, needed = 0;
  for (const TrainInputs& p : buffer.Data()) {
    held += p.legal_actions.capacity() * sizeof(Action) +
            p.observations.capacity() * sizeof(float) +
            p.policy.capacity() * sizeof(p.policy[0]);
    needed += p.legal_actions.size() * sizeof(Action) +
              p.observations.size() * sizeof(float) +
              p.policy.size() * sizeof(p.policy[0]);
  }
  return {held, needed};
}

std::string Read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "/tmp";
  constexpr int kSize = 2000;
  std::mt19937 rng(20261005);
  int failures = 0;
  auto check = [&](bool ok, const std::string& what) {
    std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
    failures += !ok;
  };

  // Fill, save, load into fresh buffers: as a resumed run starts.
  open_spiel::SerializableCircularBuffer<TrainInputs> first(kSize);
  while (first.Size() < kSize) {
    for (TrainInputs& p : Game(&rng)) first.Add(p);
  }
  first.SaveBuffer(dir + "/replay_buffer_test_start.data");
  open_spiel::SerializableCircularBuffer<TrainInputs> upstream(kSize);
  open_spiel::thud_az::ReplayBuffer<TrainInputs> ours(kSize);
  upstream.LoadBuffer(dir + "/replay_buffer_test_start.data");
  ours.LoadBuffer(dir + "/replay_buffer_test_start.data");
  const auto [loaded_held, loaded_needed] = Bytes(ours);
  const auto [up_loaded_held, up_loaded_needed] = Bytes(upstream);
  std::cout << "loaded, upstream's: " << up_loaded_held << " bytes held for "
            << up_loaded_needed << " needed (+"
            << 100.0 * (up_loaded_held - up_loaded_needed) / up_loaded_needed << "%)\n"
            << "loaded, ours:       " << loaded_held << " bytes held for "
            << loaded_needed << " needed" << std::endl;
  check(loaded_held == loaded_needed, "0. ours holds exactly what it needs once loaded");
  check(up_loaded_held > up_loaded_needed,
        "0. control: upstream's loaded buffer holds more");
  bool same_loaded = upstream.Size() == ours.Size();
  for (int i = 0; same_loaded && i < ours.Size(); ++i) {
    same_loaded = Equal(upstream[i], ours[i]);
  }
  check(same_loaded, "0. both loaded the same contents");

  // Three more buffers' worth, the same positions to both: a copy to upstream's (its
  // Add copies), a copy moved into ours (as the trainer moves its fresh copy in).
  while (ours.TotalAdded() < 4 * kSize) {
    for (const TrainInputs& p : Game(&rng)) {
      upstream.Add(p);
      TrainInputs copy = p;
      ours.Add(std::move(copy));
    }
  }

  // 1. Contents.
  bool same = upstream.Size() == ours.Size() && upstream.TotalAdded() == ours.TotalAdded();
  for (int i = 0; same && i < ours.Size(); ++i) same = Equal(upstream[i], ours[i]);
  check(same, "1. same contents (" + std::to_string(ours.Size()) + " slots, " +
                  std::to_string(ours.TotalAdded()) + " added)");

  // 2. Samples.
  std::mt19937 rng_a(7), rng_b(7);
  const std::vector<TrainInputs> a = upstream.Sample(&rng_a, 500);
  const std::vector<TrainInputs> b = ours.Sample(&rng_b, 500);
  bool same_sample = a.size() == b.size();
  for (int i = 0; same_sample && i < a.size(); ++i) same_sample = Equal(a[i], b[i]);
  check(same_sample, "2. same samples (500, equal seeds)");

  // 3. Saved files.
  upstream.SaveBuffer(dir + "/replay_buffer_test_upstream.data");
  ours.SaveBuffer(dir + "/replay_buffer_test_ours.data");
  const std::string file_a = Read(dir + "/replay_buffer_test_upstream.data");
  const std::string file_b = Read(dir + "/replay_buffer_test_ours.data");
  check(!file_a.empty() && file_a == file_b,
        "3. same saved file (" + std::to_string(file_a.size()) + " bytes)");

  // 4. Memory.
  const auto [up_held, up_needed] = Bytes(upstream);
  const auto [our_held, our_needed] = Bytes(ours);
  std::cout << "upstream's: " << up_held << " bytes held for " << up_needed
            << " needed (+" << 100.0 * (up_held - up_needed) / up_needed << "%)\n"
            << "ours:       " << our_held << " bytes held for " << our_needed
            << " needed" << std::endl;
  check(our_held == our_needed, "4. ours holds exactly what its positions need");
  check(up_held > up_needed, "4. control: upstream's slots keep larger storage");

  // 5. A file cut short.
  const std::string saved = dir + "/replay_buffer_test_ours.data";
  const std::string cut = dir + "/replay_buffer_test_cut.data";
  // Loads `path` into a fresh buffer of ours in a child process; its exit status.
  auto load_in_child = [&](const std::string& path) {
    std::cout.flush();
    const pid_t pid = fork();
    if (pid == 0) {
      open_spiel::thud_az::ReplayBuffer<TrainInputs> fresh(kSize);
      fresh.LoadBuffer(path);
      _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  };
  check(load_in_child(saved) == 0, "5. control: the whole file loads in a child process");
  const auto size = std::filesystem::file_size(saved);
  for (double fraction : {0.999, 0.5}) {
    std::filesystem::copy_file(saved, cut, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(cut, static_cast<uintmax_t>(size * fraction));
    const int status = load_in_child(cut);
    open_spiel::SerializableCircularBuffer<TrainInputs> theirs(kSize);
    theirs.LoadBuffer(cut);
    check(status != 0, "5. ours refuses the file cut to " + std::to_string(fraction) +
                           " (the child's exit " + std::to_string(status) + ")");
    check(theirs.Size() < kSize && theirs.TotalAdded() == ours.TotalAdded(),
          "5. control: upstream's loads it without an error, " +
              std::to_string(theirs.Size()) + " of " + std::to_string(kSize) +
              " positions, its count still " + std::to_string(theirs.TotalAdded()));
  }
  std::filesystem::remove(cut);
  std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
  return failures ? 1 : 0;
}
