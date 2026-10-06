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

#include <string>
#include <utility>

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

  void LoadBuffer(const std::string& path) {
    SerializableCircularBuffer<T>::LoadBuffer(path);
    for (T& element : this->data_) element = T(element);  // One at a time.
  }
};

}  // namespace open_spiel::thud_az

#endif  // THUD_AZ_REPLAY_BUFFER_H_
