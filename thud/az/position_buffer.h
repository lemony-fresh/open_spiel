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

#ifndef THUD_AZ_POSITION_BUFFER_H_
#define THUD_AZ_POSITION_BUFFER_H_

// The trainer's replay buffer, its positions stored compactly (the user, 2026-10-19: more
// positions in the same memory — a larger buffer, as KataGo's window grows, without more
// RAM). A position as VPNetModel::TrainInputs holds its observation as 1,350 floats (5.4
// KB) and 24 bytes per legal move (the move as an int64 in the legal moves, the move and a
// double in the policy target): 9.4 KB on average in run R's buffer. Compact (one byte
// vector, CompactPosition):
//
//   - the observation: its distinct values (bit for bit; Thud's are 0, 1 and the
//     no-capture fraction) once, then each entry's index into them in 1, 2, 4 or 8 bits —
//     every float exactly as it was (more than 256 distinct values: the floats as they
//     are);
//   - the legal moves as 16-bit numbers (Thud has 19,800 distinct actions; more than
//     65,535 is refused);
//   - the policy target's probabilities as floats, the moves not repeated when they are
//     the legal moves in their order (else as 16-bit numbers too);
//   - the value as a double.
//
// Every field comes back exactly but the policy's probabilities, which come back as the
// floats the learner makes of them (VPNetModel's losses read them as float: learning on a
// decompressed position computes exactly what it does on the original).
//
// PositionBuffer has the interface the trainer and the tools used of ReplayBuffer
// (thud/az/replay_buffer.h): Add, Sample — the same positions upstream's CircularBuffer
// draws with the same random numbers (std::sample over as many elements) — SampleNewest
// (alpha_zero.h's, the same draws), Size, TotalAdded, MaxSize, At. Its file: a header
// ("thud-compact-buffer", a version), the size, the count added, the slots' offset and the
// compact positions; LoadBuffer also reads upstream's format (every buffer saved before),
// both checked as ReplayBuffer's (a read that fails, or fewer positions than the count
// says, stops with an error); SaveBuffer writes under a temporary name and renames, and
// SaveLegacyBuffer writes upstream's format (for programs on upstream's AlphaZero and
// az_trainer_d1). A file of a smaller buffer loads into a larger one — its positions
// oldest first, the next ones added after them, the oldest replaced first once full — so
// a run can resume with a larger buffer; a larger file into a smaller buffer is refused.
//
// Tests: thud/az/position_buffer_test.cc.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// libnop (third-party) writes calls like `value_.template Construct(args...)`: the
// `template` keyword without a template argument list, which clang here rejects by
// default (-Wmissing-template-arg-list-after-template-kw). The arguments are deduced from
// the call, so the code means the same with or without the keyword; upstream builds
// libnop with every warning off (open_spiel/CMakeLists.txt). Ignored for these includes
// only (as thud/az/replay_buffer_test.cc does).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-template-arg-list-after-template-kw"
#include <nop/serializer.h>
#include <nop/structure.h>
#include <nop/utility/stream_reader.h>
#include <nop/utility/stream_writer.h>
#pragma clang diagnostic pop

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::thud_az {

// One position, compact (the format above).
struct CompactPosition {
  std::vector<uint8_t> bytes;
  NOP_STRUCTURE(CompactPosition, bytes);
};

namespace compact_internal {

inline constexpr uint8_t kVersion = 1;

template <class V>
void Put(std::vector<uint8_t>* out, V value) {
  const size_t at = out->size();
  out->resize(at + sizeof(V));
  std::memcpy(out->data() + at, &value, sizeof(V));
}

class Reader {
 public:
  explicit Reader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}
  template <class V>
  V Get() {
    SPIEL_CHECK_LE(at_ + sizeof(V), bytes_.size());
    V value;
    std::memcpy(&value, bytes_.data() + at_, sizeof(V));
    at_ += sizeof(V);
    return value;
  }
  const uint8_t* Take(size_t n) {
    SPIEL_CHECK_LE(at_ + n, bytes_.size());
    const uint8_t* p = bytes_.data() + at_;
    at_ += n;
    return p;
  }
  bool Done() const { return at_ == bytes_.size(); }

 private:
  const std::vector<uint8_t>& bytes_;
  size_t at_ = 0;
};

// Bits per index for a palette of `values` distinct values: 0, 1, 2, 4 or 8.
inline int IndexBits(int values) {
  if (values <= 1) return 0;
  if (values <= 2) return 1;
  if (values <= 4) return 2;
  if (values <= 16) return 4;
  return 8;
}

inline uint16_t Action16(int64_t action) {
  if (action < 0 || action > 65535) {
    SpielFatalError(absl::StrFormat("CompactPosition: action %d outside 0-65535", action));
  }
  return static_cast<uint16_t>(action);
}

}  // namespace compact_internal

// T has legal_actions (integers), observations (floats), policy (pairs of an action and a
// probability) and value — VPNetModel::TrainInputs, or a test's copy of it.
template <class T>
CompactPosition Compress(const T& p) {
  using namespace compact_internal;  // NOLINT: this function's helpers.
  CompactPosition c;
  std::vector<uint8_t>& out = c.bytes;
  Put<uint8_t>(&out, kVersion);

  // The observation: a palette of its distinct floats, bit for bit, and an index each.
  const uint32_t n = p.observations.size();
  Put<uint32_t>(&out, n);
  std::vector<uint32_t> palette;
  std::vector<uint8_t> index(n);
  bool raw = false;
  for (uint32_t i = 0; i < n && !raw; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &p.observations[i], sizeof(bits));
    auto it = std::find(palette.begin(), palette.end(), bits);
    if (it == palette.end()) {
      if (palette.size() == 256) {
        raw = true;
        break;
      }
      palette.push_back(bits);
      it = palette.end() - 1;
    }
    index[i] = static_cast<uint8_t>(it - palette.begin());
  }
  if (raw) {  // More than 256 distinct values: the floats as they are.
    Put<uint16_t>(&out, 0);
    for (float f : p.observations) Put<float>(&out, f);
  } else {
    Put<uint16_t>(&out, palette.size());
    for (uint32_t bits : palette) Put<uint32_t>(&out, bits);
    const int b = IndexBits(palette.size());
    if (b > 0) {
      const int per_byte = 8 / b;
      for (uint32_t i = 0; i < n; i += per_byte) {
        uint8_t packed = 0;
        for (int k = 0; k < per_byte && i + k < n; ++k) packed |= index[i + k] << (k * b);
        Put<uint8_t>(&out, packed);
      }
    }
  }

  // The legal moves, then the policy target (its moves only if not the legal moves).
  Put<uint32_t>(&out, p.legal_actions.size());
  for (auto a : p.legal_actions) Put<uint16_t>(&out, Action16(a));
  bool aligned = p.policy.size() == p.legal_actions.size();
  for (size_t i = 0; i < p.policy.size() && aligned; ++i) {
    aligned = p.policy[i].first == p.legal_actions[i];
  }
  Put<uint8_t>(&out, aligned ? 1 : 0);
  if (!aligned) {
    Put<uint32_t>(&out, p.policy.size());
    for (const auto& [a, prob] : p.policy) Put<uint16_t>(&out, Action16(a));
  }
  for (const auto& [a, prob] : p.policy) Put<float>(&out, static_cast<float>(prob));
  Put<double>(&out, p.value);
  out.shrink_to_fit();
  return c;
}

template <class T>
T Decompress(const CompactPosition& c) {
  using namespace compact_internal;  // NOLINT: this function's helpers.
  Reader in(c.bytes);
  const uint8_t version = in.Get<uint8_t>();
  if (version != kVersion) {
    SpielFatalError(absl::StrFormat("CompactPosition: version %d, not %d", version, kVersion));
  }
  T p;
  const uint32_t n = in.Get<uint32_t>();
  const uint16_t values = in.Get<uint16_t>();
  p.observations.resize(n);
  if (values == 0) {
    for (uint32_t i = 0; i < n; ++i) p.observations[i] = in.Get<float>();
  } else {
    std::vector<float> palette(values);
    for (float& f : palette) {
      const uint32_t bits = in.Get<uint32_t>();
      std::memcpy(&f, &bits, sizeof(f));
    }
    const int b = IndexBits(values);
    if (b == 0) {
      std::fill(p.observations.begin(), p.observations.end(), palette[0]);
    } else {
      const int per_byte = 8 / b;
      const uint8_t mask = (1 << b) - 1;
      for (uint32_t i = 0; i < n; i += per_byte) {
        const uint8_t packed = in.Get<uint8_t>();
        for (int k = 0; k < per_byte && i + k < n; ++k) {
          const int idx = (packed >> (k * b)) & mask;
          SPIEL_CHECK_LT(idx, values);
          p.observations[i + k] = palette[idx];
        }
      }
    }
  }
  const uint32_t legal = in.Get<uint32_t>();
  p.legal_actions.resize(legal);
  for (auto& a : p.legal_actions) a = in.Get<uint16_t>();
  const bool aligned = in.Get<uint8_t>() == 1;
  if (aligned) {
    p.policy.resize(legal);
    for (uint32_t i = 0; i < legal; ++i) p.policy[i].first = p.legal_actions[i];
  } else {
    p.policy.resize(in.Get<uint32_t>());
    for (auto& entry : p.policy) entry.first = in.Get<uint16_t>();
  }
  for (auto& entry : p.policy) entry.second = in.Get<float>();
  p.value = in.Get<double>();
  SPIEL_CHECK_TRUE(in.Done());
  return p;
}

template <class T>
class PositionBuffer {
 public:
  explicit PositionBuffer(int max_size) : max_size_(max_size) {}

  void Add(const T& p) { Place(Compress(p)); }

  // `num` distinct positions — the ones upstream's CircularBuffer::Sample draws with the
  // same generator (std::sample over as many elements).
  std::vector<T> Sample(std::mt19937* rng, int num) const {
    std::vector<int64_t> picked;
    picked.reserve(num);
    std::sample(Indices().begin(), Indices().end(), std::back_inserter(picked), num, *rng);
    std::vector<T> out;
    out.reserve(picked.size());
    for (int64_t i : picked) out.push_back(Decompress<T>(data_[i]));
    return out;
  }

  // `num` distinct positions of the `newest` added last, each set equally likely — the
  // ones alpha_zero.h's SampleNewest draws on upstream's buffer with the same generator.
  std::vector<T> SampleNewest(std::mt19937* rng, int64_t newest, int num) const {
    const int64_t size = data_.size(), added = total_added_ - slot_offset_;
    newest = std::min(newest, size);
    std::unordered_set<int64_t> offsets;  // Floyd's algorithm, as SampleNewest.
    for (int64_t j = newest - std::min<int64_t>(num, newest); j < newest; ++j) {
      const int64_t t = std::uniform_int_distribution<int64_t>(0, j)(*rng);
      offsets.insert(offsets.count(t) ? j : t);
    }
    std::vector<T> out;
    out.reserve(offsets.size());
    for (int64_t offset : offsets) {
      out.push_back(Decompress<T>(data_[(added - 1 - offset) % size]));
    }
    return out;
  }

  T At(int i) const { return Decompress<T>(data_.at(i)); }
  // The slots' indices, the oldest position's first.
  std::vector<int64_t> OldestFirst() const {
    const int64_t size = data_.size();
    const int64_t start = size == max_size_ ? (total_added_ - slot_offset_) % size : 0;
    std::vector<int64_t> order(size);
    for (int64_t k = 0; k < size; ++k) order[k] = (start + k) % size;
    return order;
  }
  // The max size a saved buffer was saved at, in either format.
  static int FileMaxSize(const std::string& path) {
    std::string magic;
    {
      nop::Deserializer<nop::StreamReader<std::ifstream>> probe{path};
      if (!probe.Read(&magic)) magic.clear();
    }
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    int version = 0, max_size = 0;
    const bool ok = magic == kMagic ? deserializer.Read(&magic) && deserializer.Read(&version) &&
                                          deserializer.Read(&max_size)
                                    : static_cast<bool>(deserializer.Read(&max_size));
    if (!ok) SpielFatalError(absl::StrFormat("Cannot read the buffer %s's size", path));
    return max_size;
  }
  int Size() const { return data_.size(); }
  bool Empty() const { return data_.empty(); }
  int64_t TotalAdded() const { return total_added_; }
  int MaxSize() const { return max_size_; }
  // The bytes the compact positions hold, for measurements.
  int64_t Bytes() const {
    int64_t sum = 0;
    for (const auto& c : data_) sum += c.bytes.size();
    return sum;
  }

  void LoadBuffer(const std::string& path) {
    std::string magic;
    {
      nop::Deserializer<nop::StreamReader<std::ifstream>> probe{path};
      if (!probe.Read(&magic)) magic.clear();
    }
    if (magic == kMagic) {
      LoadCompact(path);
    } else {
      LoadLegacy(path);
    }
  }

  void SaveBuffer(const std::string& path) const {
    const std::string temporary = path + ".tmp";
    {
      nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{temporary};
      Check(serializer.Write(std::string(kMagic)), temporary);
      Check(serializer.Write(static_cast<int>(kFileVersion)), temporary);
      Check(serializer.Write(max_size_), temporary);
      Check(serializer.Write(total_added_), temporary);
      Check(serializer.Write(slot_offset_), temporary);
      Check(serializer.Write(data_), temporary);
    }
    std::filesystem::rename(temporary, path);
  }

  // Upstream's format (SerializableCircularBuffer<T>::SaveBuffer), in the slots' order and
  // count upstream's Add would have — a buffer loaded at its own size, never grown (else
  // refused: upstream's slot arithmetic could not continue it).
  void SaveLegacyBuffer(const std::string& path) const {
    if (slot_offset_ != 0) {
      SpielFatalError("SaveLegacyBuffer: a grown buffer has no upstream layout");
    }
    std::vector<T> plain;
    plain.reserve(data_.size());
    for (const auto& c : data_) plain.push_back(Decompress<T>(c));
    const std::string temporary = path + ".tmp";
    {
      nop::Serializer<nop::StreamWriter<std::ofstream>> serializer{temporary};
      Check(serializer.Write(max_size_), temporary);
      Check(serializer.Write(total_added_), temporary);
      Check(serializer.Write(plain), temporary);
    }
    std::filesystem::rename(temporary, path);
  }

 private:
  static constexpr const char* kMagic = "thud-compact-buffer";
  static constexpr int kFileVersion = 1;

  template <class Status>
  static void Check(const Status& status, const std::string& path) {
    if (!status) {
      SpielFatalError(absl::StrFormat("Cannot write the buffer %s (%s)", path,
                                      status.GetErrorMessage()));
    }
  }

  void Place(CompactPosition&& c) {
    if (static_cast<int>(data_.size()) < max_size_) {
      data_.push_back(std::move(c));  // Indices() follows the new size when sampled.
    } else {
      data_[(total_added_ - slot_offset_) % max_size_] = std::move(c);
    }
    total_added_ += 1;
  }

  const std::vector<int64_t>& Indices() const {
    if (static_cast<int64_t>(indices_.size()) != static_cast<int64_t>(data_.size())) {
      indices_.resize(data_.size());
      for (size_t i = 0; i < indices_.size(); ++i) indices_[i] = i;
    }
    return indices_;
  }

  // The file's positions placed in this buffer: as they are at the file's own size; oldest
  // first in a larger buffer (the slots then counted from `total_added_ - n`).
  template <class Get>
  void Arrange(int file_max, int64_t file_total, int64_t file_offset, int64_t n,
               const std::string& path, Get&& get) {
    if (file_max > max_size_) {
      SpielFatalError(absl::StrFormat("Cannot load the buffer %s, of max size %d, into a "
                                      "buffer of max size %d",
                                      path, file_max, max_size_));
    }
    const int64_t expected = std::min<int64_t>(file_total - file_offset, file_max);
    if (n != expected) {
      SpielFatalError(absl::StrFormat("Cannot load the buffer %s: %d positions read, %d "
                                      "expected (the file cut short?)",
                                      path, n, expected));
    }
    data_.clear();
    data_.reserve(std::min<int64_t>(max_size_, std::max<int64_t>(n, 1)));
    total_added_ = file_total;
    if (file_max == max_size_) {
      for (int64_t i = 0; i < n; ++i) data_.push_back(get(i));
      slot_offset_ = file_offset;
    } else {  // Grown: oldest first.
      const int64_t start = n == file_max ? (file_total - file_offset) % file_max : 0;
      for (int64_t k = 0; k < n; ++k) data_.push_back(get((start + k) % n));
      slot_offset_ = file_total - n;
    }
    indices_.clear();
  }

  void LoadCompact(const std::string& path) {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    auto read = [&](auto* value, const char* what) {
      if (!deserializer.Read(value)) {
        SpielFatalError(absl::StrFormat("Cannot load the buffer %s: reading %s failed", path,
                                        what));
      }
    };
    std::string magic;
    int version, file_max;
    int64_t file_total, file_offset;
    std::vector<CompactPosition> positions;
    read(&magic, "its header");
    read(&version, "its version");
    if (version != kFileVersion) {
      SpielFatalError(absl::StrFormat("Cannot load the buffer %s: version %d", path, version));
    }
    read(&file_max, "its size");
    read(&file_total, "its count of positions added");
    read(&file_offset, "its slots' offset");
    read(&positions, "its positions");
    Arrange(file_max, file_total, file_offset, positions.size(), path,
            [&](int64_t i) { return std::move(positions[i]); });
  }

  void LoadLegacy(const std::string& path) {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    auto read = [&](auto* value, const char* what) {
      const auto status = deserializer.Read(value);
      if (!status) {
        SpielFatalError(absl::StrFormat("Cannot load the buffer %s: reading %s failed (%s)",
                                        path, what, status.GetErrorMessage()));
      }
    };
    int file_max;
    int64_t file_total;
    std::vector<T> positions;
    read(&file_max, "its size");
    read(&file_total, "its count of positions added");
    read(&positions, "its positions");
    Arrange(file_max, file_total, 0, positions.size(), path, [&](int64_t i) {
      CompactPosition c = Compress(positions[i]);
      positions[i] = T();  // Its memory freed as it goes.
      return c;
    });
  }

  const int max_size_;
  int64_t total_added_ = 0;
  // Slots are counted from here: 0 unless the buffer was loaded from a smaller one.
  int64_t slot_offset_ = 0;
  std::vector<CompactPosition> data_;
  mutable std::vector<int64_t> indices_;  // 0, 1, ..., Size() - 1: what Sample draws from.
};

}  // namespace open_spiel::thud_az

#endif  // THUD_AZ_POSITION_BUFFER_H_
