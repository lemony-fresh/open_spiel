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

// Tests the compact replay buffer (thud/az/position_buffer.h) against upstream's
// CircularBuffer and SerializableCircularBuffer, on positions shaped like the trainer's
// (replay_buffer_test's: games of 50-400 positions, dwarf positions with 150-330 legal
// moves, troll positions with 10-40) and on positions of random games of Thud:
//
//   1. The round trip: Decompress(Compress(p)) gives every field back exactly but the
//      policy's probabilities, which come back as static_cast<float> of them — for
//      observations of 1, 2, 3, 5, 17 and 300 distinct values (the last stored raw), 0.0
//      and -0.0 apart, policies on the legal moves in their order and otherwise (a subset
//      in another order, as pruning leaves), no legal moves at all. Controls: a position
//      with one observation entry changed compares unequal; a compact position with one
//      index byte changed decompresses to another observation.
//   2. Real positions: 2,000 positions of random games of Thud come back exactly (their
//      uniform policies as floats), each compact position at most a quarter of the bytes
//      the position's vectors need (the limit for many legal moves: 6 bytes against 24).
//   3. Sampling: the same 2,750 positions added to ours and to upstream's CircularBuffer
//      (of 1,000: it wraps around, its oldest at slot 750): the same contents slot by slot,
//      and Sample with equal seeds draws the same positions, 20 times.
//   4. Files: ours saved and loaded holds the same positions in the same slots and draws
//      the same samples; a file upstream's SerializableCircularBuffer saved loads into
//      ours with the same contents; SaveLegacyBuffer's file loads into upstream's with the
//      same contents (policies as floats). Control: a file of other positions loads
//      unequal.
//   5. Growth: a full buffer of 1,000 saved and loaded into one of 2,500 holds its
//      positions oldest first, and after 3,000 more holds exactly the last 2,500 added
//      (the oldest replaced first), its newest the newest; saved and loaded again (the
//      slots' offset kept) it adds on the same. A file of 2,500
//      into a buffer of 1,000 is refused (in a child process, which must stop with an
//      error; the file of 1,000 loads in a child — the control that a child can succeed).
//   6. A file cut short (as replay_buffer_test's part 5): ours refuses the compact file
//      cut to 99.9% and to half, in a child process each; whole it loads.
//   7. Memory: the compact positions of part 3's buffer hold at most a quarter of the
//      bytes its positions' vectors need.
//
// Needs only headers and libopen_spiel.so (no LibTorch):
//   thud/run_tests.sh position_buffer_test   # builds and runs it
//   build-tests/bin/position_buffer_test [DIR]   # DIR for its files (default /tmp)

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "thud/az/position_buffer.h"
// libnop again, through upstream's buffer: see position_buffer.h.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-template-arg-list-after-template-kw"
#include "open_spiel/utils/serializable_circular_buffer.h"
#pragma clang diagnostic pop
#include "open_spiel/spiel.h"

namespace {

using open_spiel::thud_az::CompactPosition;
using open_spiel::thud_az::Compress;
using open_spiel::thud_az::Decompress;
using open_spiel::thud_az::PositionBuffer;
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

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// What the round trip must give back: everything, the probabilities as floats.
bool Restored(const TrainInputs& original, const TrainInputs& back) {
  if (original.legal_actions != back.legal_actions || original.value != back.value ||
      !SameBits(original.observations, back.observations) ||
      original.policy.size() != back.policy.size()) {
    return false;
  }
  for (size_t i = 0; i < original.policy.size(); ++i) {
    if (original.policy[i].first != back.policy[i].first ||
        static_cast<float>(original.policy[i].second) != back.policy[i].second) {
      return false;
    }
  }
  return true;
}

// Equal as stored: two positions that came back from the compact form.
bool Equal(const TrainInputs& a, const TrainInputs& b) {
  return a.legal_actions == b.legal_actions && SameBits(a.observations, b.observations) &&
         a.policy == b.policy && a.value == b.value;
}

int64_t Needed(const TrainInputs& p) {
  return p.legal_actions.size() * sizeof(Action) + p.observations.size() * sizeof(float) +
         p.policy.size() * sizeof(p.policy[0]) + sizeof(p.value);
}

// replay_buffer_test's positions: one game, sides alternating, dwarfs first.
std::vector<TrainInputs> Game(std::mt19937* rng) {
  std::vector<TrainInputs> game;
  const int length = std::uniform_int_distribution<int>(50, 400)(*rng);
  const double value = std::uniform_real_distribution<double>(-1, 1)(*rng);
  for (int i = 0; i < length; ++i) {
    const bool dwarfs = i % 2 == 0;
    const int legal = dwarfs ? std::uniform_int_distribution<int>(150, 330)(*rng)
                             : std::uniform_int_distribution<int>(10, 40)(*rng);
    TrainInputs p;
    for (int a = 0; a < legal; ++a) {
      p.legal_actions.push_back((*rng)() % 19800);
      p.policy.push_back({p.legal_actions.back(),
                          std::uniform_real_distribution<double>(0, 1)(*rng)});
    }
    p.observations.assign(6 * 15 * 15, 0.0f);
    for (int k = 0; k < 40; ++k) p.observations[(*rng)() % p.observations.size()] = 1.0f;
    std::fill(p.observations.begin() + 4 * 225, p.observations.begin() + 5 * 225,
              static_cast<float>(i % 200) / 200);  // The no-capture fraction's plane.
    p.value = value;
    game.push_back(std::move(p));
  }
  return game;
}

std::vector<TrainInputs> Positions(std::mt19937* rng, int count) {
  std::vector<TrainInputs> out;
  while (static_cast<int>(out.size()) < count) {
    for (TrainInputs& p : Game(rng)) {
      if (static_cast<int>(out.size()) < count) out.push_back(std::move(p));
    }
  }
  return out;
}

// Runs `load` in a child process: its exit status (0: it loaded).
template <class F>
int InChild(F load) {
  std::cout.flush();
  const pid_t pid = fork();
  if (pid == 0) {
    load();
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "/tmp";
  std::mt19937 rng(20261019);

  // 1. The round trip.
  {
    bool all = true;
    std::string failed;
    for (int distinct : {1, 2, 3, 5, 17, 300}) {
      TrainInputs p;
      p.observations.resize(1350);
      for (size_t i = 0; i < p.observations.size(); ++i) {
        p.observations[i] = static_cast<float>(i % distinct) * 0.37f;
      }
      if (distinct >= 2) p.observations[1] = -0.0f;  // Not 0.0 bit for bit.
      for (int a = 0; a < 25; ++a) {
        p.legal_actions.push_back(19799 - 7 * a);
        p.policy.push_back({p.legal_actions.back(), 1.0 / 3 + a * 1e-9});
      }
      p.value = -0.71875;
      TrainInputs subset = p;  // Another order, fewer moves: as pruning leaves.
      subset.policy = {{p.legal_actions[5], 0.6}, {p.legal_actions[2], 0.4}};
      TrainInputs none;  // No legal moves.
      none.observations = p.observations;
      none.value = 1;
      for (const TrainInputs* q : {&p, &subset, &none}) {
        if (!Restored(*q, Decompress<TrainInputs>(Compress(*q)))) {
          all = false;
          failed += " " + std::to_string(distinct);
        }
      }
    }
    Check(all, "1. the round trip: every field back (the probabilities as floats), for 1, 2, "
               "3, 5, 17 and 300 distinct observation values, -0.0 apart, aligned, subset "
               "and empty policies" + (failed.empty() ? "" : "; failed for" + failed));
    TrainInputs p = Positions(&rng, 1)[0];
    TrainInputs changed = Decompress<TrainInputs>(Compress(p));
    changed.observations[100] = changed.observations[100] == 0 ? 1 : 0;
    CompactPosition c = Compress(p);
    c.bytes[40] ^= 0x01;  // In the packed indices (after the header and the palette).
    const TrainInputs flipped = Decompress<TrainInputs>(c);
    Check(!Restored(p, changed) && !SameBits(p.observations, flipped.observations),
          "1. controls: one observation entry changed compares unequal; one index byte "
          "changed decompresses to another observation");
  }

  // 2. Real positions.
  {
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
    int count = 0;
    bool exact = true;
    int64_t compact = 0, needed = 0;
    double worst = 0;
    while (count < 2000) {
      std::unique_ptr<open_spiel::State> s = game->NewInitialState();
      for (int ply = 0; ply < 300 && !s->IsTerminal() && count < 2000; ++ply, ++count) {
        TrainInputs p;
        p.legal_actions = s->LegalActions();
        p.observations = s->ObservationTensor();
        for (Action a : p.legal_actions) p.policy.push_back({a, 1.0 / p.legal_actions.size()});
        p.value = std::uniform_int_distribution<int>(-32, 32)(rng) / 32.0;
        const CompactPosition c = Compress(p);
        exact = exact && Restored(p, Decompress<TrainInputs>(c));
        compact += c.bytes.size();
        needed += Needed(p);
        worst = std::max(worst, static_cast<double>(c.bytes.size()) / Needed(p));
        s->ApplyAction(p.legal_actions[rng() % p.legal_actions.size()]);
      }
    }
    Check(exact && worst <= 0.25,
          absl::StrFormat("2. 2,000 positions of random games of Thud come back exactly; "
                          "compact %.1f%% of the bytes their vectors need (%.1fx fewer), at "
                          "most %.1f%% (a quarter allowed)",
                          100.0 * compact / needed, static_cast<double>(needed) / compact,
                          100 * worst));
  }

  // 3. Sampling, against upstream's CircularBuffer.
  const std::vector<TrainInputs> positions = Positions(&rng, 2750);
  PositionBuffer<TrainInputs> ours(1000);
  {
    open_spiel::CircularBuffer<TrainInputs> theirs(1000);
    for (const TrainInputs& p : positions) {
      ours.Add(p);
      theirs.Add(p);
    }
    bool same = ours.Size() == theirs.Size() && ours.TotalAdded() == theirs.TotalAdded();
    for (int i = 0; i < ours.Size() && same; ++i) same = Restored(theirs[i], ours.At(i));
    bool samples = true;
    for (int trial = 0; trial < 20 && samples; ++trial) {
      std::mt19937 r1(trial), r2(trial);
      const std::vector<TrainInputs> a = ours.Sample(&r1, 256), b = theirs.Sample(&r2, 256);
      samples = a.size() == b.size();
      for (size_t i = 0; i < a.size() && samples; ++i) samples = Restored(b[i], a[i]);
    }
    Check(same && samples, "3. sampling: the same contents slot by slot as upstream's "
                           "CircularBuffer after 2,750 adds to 1,000, and the same 20 "
                           "samples of 256 with equal seeds");
  }

  // 4. Files.
  {
    const std::string compact = dir + "/position_buffer_test.data";
    const std::string legacy = dir + "/position_buffer_test_legacy.data";
    const std::string back = dir + "/position_buffer_test_back.data";
    ours.SaveBuffer(compact);
    PositionBuffer<TrainInputs> loaded(1000);
    loaded.LoadBuffer(compact);
    bool same = loaded.Size() == ours.Size() && loaded.TotalAdded() == ours.TotalAdded();
    for (int i = 0; i < ours.Size() && same; ++i) same = Equal(ours.At(i), loaded.At(i));
    std::mt19937 r1(5), r2(5);
    const auto s1 = ours.Sample(&r1, 300), s2 = loaded.Sample(&r2, 300);
    for (size_t i = 0; i < s1.size() && same; ++i) same = Equal(s1[i], s2[i]);

    open_spiel::SerializableCircularBuffer<TrainInputs> upstream(1000);
    for (const TrainInputs& p : positions) upstream.Add(p);
    upstream.SaveBuffer(legacy);
    PositionBuffer<TrainInputs> from_legacy(1000);
    from_legacy.LoadBuffer(legacy);
    bool legacy_same = from_legacy.Size() == upstream.Size() &&
                       from_legacy.TotalAdded() == upstream.TotalAdded();
    for (int i = 0; i < upstream.Size() && legacy_same; ++i) {
      legacy_same = Restored(upstream[i], from_legacy.At(i));
    }
    from_legacy.SaveLegacyBuffer(back);
    open_spiel::SerializableCircularBuffer<TrainInputs> upstream_back(1000);
    upstream_back.LoadBuffer(back);
    bool back_same = upstream_back.Size() == upstream.Size() &&
                     upstream_back.TotalAdded() == upstream.TotalAdded();
    for (int i = 0; i < upstream.Size() && back_same; ++i) {
      back_same = Restored(upstream[i], upstream_back[i]);
    }
    PositionBuffer<TrainInputs> other(1000);  // The control: other positions.
    for (const TrainInputs& p : Positions(&rng, 1000)) other.Add(p);
    other.SaveBuffer(back);
    PositionBuffer<TrainInputs> other_loaded(1000);
    other_loaded.LoadBuffer(back);
    bool control_differs = false;
    for (int i = 0; i < 1000 && !control_differs; ++i) {
      control_differs = !Equal(other_loaded.At(i), ours.At(i));
    }
    Check(same && legacy_same && back_same && control_differs,
          "4. files: ours saved and loaded the same (contents, samples); upstream's file "
          "loads into ours the same; SaveLegacyBuffer's file loads into upstream's the "
          "same; control: a file of other positions loads unequal");
    std::filesystem::remove(legacy);
    std::filesystem::remove(back);
  }

  // 5. Growth.
  {
    const std::string small = dir + "/position_buffer_test.data";  // Part 4's, of 1,000.
    PositionBuffer<TrainInputs> grown(2500);
    grown.LoadBuffer(small);
    // Its positions oldest first: positions 1,750-2,749.
    bool oldest_first = grown.Size() == 1000 && grown.TotalAdded() == 2750;
    for (int i = 0; i < 1000 && oldest_first; ++i) {
      oldest_first = Restored(positions[1750 + i], grown.At(i));
    }
    // 3,000 more: it must hold the last 2,500 added, as a fresh buffer of 2,500 given every
    // position does (in other slots).
    const std::vector<TrainInputs> more = Positions(&rng, 3000);
    std::vector<TrainInputs> history(positions.begin(), positions.end());
    history.insert(history.end(), more.begin(), more.end());
    PositionBuffer<TrainInputs> fresh(2500);
    for (const TrainInputs& p : history) fresh.Add(p);
    for (const TrainInputs& p : more) grown.Add(p);
    std::vector<TrainInputs> held;
    for (int i = 0; i < grown.Size(); ++i) held.push_back(grown.At(i));
    std::vector<TrainInputs> last(history.end() - 2500, history.end());
    auto key = [](const TrainInputs& p) { return std::make_pair(p.value, p.legal_actions); };
    auto sorted = [&](std::vector<TrainInputs> v) {
      std::sort(v.begin(), v.end(), [&](const TrainInputs& a, const TrainInputs& b) {
        return key(a) < key(b);
      });
      return v;
    };
    const std::vector<TrainInputs> held_sorted = sorted(held), last_sorted = sorted(last);
    bool holds_last = held_sorted.size() == last_sorted.size();
    for (size_t i = 0; i < held_sorted.size() && holds_last; ++i) {
      holds_last = Restored(last_sorted[i], held_sorted[i]);
    }
    // The newest is where a fresh buffer of 2,500 would put it.
    bool newest = Equal(grown.SampleNewest(&rng, 1, 1)[0], fresh.SampleNewest(&rng, 1, 1)[0]);
    const std::string grown_file = dir + "/position_buffer_test_grown.data";
    grown.SaveBuffer(grown_file);
    PositionBuffer<TrainInputs> reloaded(2500);
    reloaded.LoadBuffer(grown_file);
    for (const TrainInputs& p : Positions(&rng, 500)) {
      reloaded.Add(p);
      grown.Add(p);
    }
    bool kept = reloaded.Size() == grown.Size() && reloaded.TotalAdded() == grown.TotalAdded();
    for (int i = 0; i < grown.Size() && kept; ++i) kept = Equal(grown.At(i), reloaded.At(i));
    const int refused = InChild([&]() {
      PositionBuffer<TrainInputs> smaller(1000);
      smaller.LoadBuffer(grown_file);
    });
    const int control = InChild([&]() {
      PositionBuffer<TrainInputs> same_size(1000);
      same_size.LoadBuffer(small);
    });
    Check(oldest_first && holds_last && newest && kept && refused != 0 && control == 0,
          "5. growth: a buffer of 1,000 loaded into 2,500 holds its positions oldest first, "
          "after 3,000 more the last 2,500 added, the newest where a fresh buffer puts it; "
          "saved and loaded again it adds on the same; a file of 2,500 into 1,000 refused "
          "(the file of 1,000 loads: the control)");
    std::filesystem::remove(grown_file);
  }

  // 6. A file cut short.
  {
    const std::string saved = dir + "/position_buffer_test.data";
    const std::string cut = dir + "/position_buffer_test_cut.data";
    const auto size = std::filesystem::file_size(saved);
    for (double fraction : {0.999, 0.5}) {
      std::filesystem::copy_file(saved, cut, std::filesystem::copy_options::overwrite_existing);
      std::filesystem::resize_file(cut, static_cast<uintmax_t>(size * fraction));
      const int status = InChild([&]() {
        PositionBuffer<TrainInputs> b(1000);
        b.LoadBuffer(cut);
      });
      Check(status != 0, "6. ours refuses the compact file cut to " + std::to_string(fraction));
    }
    const int whole = InChild([&]() {
      PositionBuffer<TrainInputs> b(1000);
      b.LoadBuffer(saved);
    });
    Check(whole == 0, "6. the whole file loads (the control)");
    std::filesystem::remove(cut);
    std::filesystem::remove(saved);
  }

  // 7. Memory.
  {
    int64_t needed = 0;
    for (int i = 0; i < ours.Size(); ++i) needed += Needed(ours.At(i));
    Check(ours.Bytes() * 4 <= needed,
          absl::StrFormat("7. memory: part 3's compact positions hold %.1f%% of the bytes "
                          "their vectors need (%.1fx fewer; a quarter allowed)",
                          100.0 * ours.Bytes() / needed,
                          static_cast<double>(needed) / ours.Bytes()));
  }

  std::cout << (failures == 0 ? "ALL PASSED" : std::to_string(failures) + " FAILED")
            << std::endl;
  return failures == 0 ? 0 : 1;
}
