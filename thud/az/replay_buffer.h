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

#ifndef THUD_AZ_REPLAY_BUFFER_H_
#define THUD_AZ_REPLAY_BUFFER_H_

#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/serializable_circular_buffer.h"

namespace open_spiel::thud_az {

// The trainer's replay buffer (thud/PLAN.md, Instrumentation: the buffer's memory):
// upstream's SerializableCircularBuffer plus an Add that moves the new element into
// its slot. Upstream's Add copies it in, and a copy-assigned std::vector keeps its
// old storage whenever that is large enough, so once the buffer is full every slot
// keeps the largest position it ever held — in Thud a slot that held a dwarf position
// (a median ~240 legal moves, 24 bytes each in the legal moves and the policy target)
// keeps that size when a troll position (~20) replaces it. Moving frees the old
// storage and takes the new element's, so each slot holds its own position's size.
// And upstream's LoadBuffer leaves spare storage in every vector it reads (libnop
// reads them element by element: 45% more than the elements need in
// replay_buffer_check), so ours compacts the loaded buffer: each element replaced by
// a copy of itself, which holds exactly its size. The contents, samples and saved
// files are the same either way (thud/az/replay_buffer_check.cc).
template <class T>
class ReplayBuffer : public SerializableCircularBuffer<T> {
 public:
  using SerializableCircularBuffer<T>::SerializableCircularBuffer;
  using SerializableCircularBuffer<T>::Add;  // Upstream's, which copies.

  void Add(T&& value) {
    if (this->data_.size() < this->max_size_) {
      this->data_.push_back(std::move(value));
    } else {
      this->data_[this->total_added_ % this->max_size_] = std::move(value);
    }
    this->total_added_ += 1;
  }

  // Upstream's LoadBuffer, checked (thud/PLAN.md, the GPU readiness checklist: restarting
  // after a crash or a preemption): upstream's ignores the status of each read, so a file
  // cut short — written in place by a run killed while saving — loads without an error:
  // only the positions before the cut, with total_added_ still the file's, which breaks
  // Add's slot arithmetic (found 2026-10-15, thud/az/replay_buffer_check.cc). Ours stops
  // on a read that fails or a count of positions that does not match the file's.
  void LoadBuffer(const std::string& path) {
    nop::Deserializer<nop::StreamReader<std::ifstream>> deserializer{path};
    auto read = [&](auto* value, const char* what) {
      const auto status = deserializer.Read(value);
      if (!status) {
        SpielFatalError(absl::StrFormat("Cannot load the buffer %s: reading %s failed (%s)",
                                        path, what, status.GetErrorMessage()));
      }
    };
    // Ensure this buffer's max size equals the max size of the saved buffer.
    int max_size;
    read(&max_size, "its size");
    if (max_size != this->max_size_) {  // Upstream's message.
      SpielFatalError(absl::StrFormat("Cannot load data from a buffer with max"
                                      "size %d into a buffer with max size %d.",
                                      max_size, this->max_size_));
    }
    read(&(this->total_added_), "its count of positions added");
    read(&(this->data_), "its positions");
    const int64_t expected = std::min<int64_t>(this->total_added_, this->max_size_);
    if (static_cast<int64_t>(this->data_.size()) != expected) {
      SpielFatalError(absl::StrFormat("Cannot load the buffer %s: %d positions read, %d "
                                      "expected (the file cut short?)",
                                      path, this->data_.size(), expected));
    }
    for (T& element : this->data_) element = T(element);  // One at a time.
  }

  // Upstream's file, written under a temporary name and renamed into place (thud/PLAN.md,
  // the GPU readiness checklist: restarting after a crash or a preemption): a run
  // killed while saving keeps the previous buffer whole. Upstream's writer has closed
  // its stream when it returns.
  void SaveBuffer(const std::string& path) const {
    const std::string temporary = path + ".tmp";
    SerializableCircularBuffer<T>::SaveBuffer(temporary);
    std::filesystem::rename(temporary, path);
  }
};

}  // namespace open_spiel::thud_az

#endif  // THUD_AZ_REPLAY_BUFFER_H_
