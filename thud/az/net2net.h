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

// Net2Net for our residual networks — the growth az_grow.cc does, here so that
// grow_test.cc can test it on small networks (see az_grow.cc for the method).

#ifndef THUD_AZ_NET2NET_H_
#define THUD_AZ_NET2NET_H_

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/vpnet.h"

namespace open_spiel {
namespace thud_az {
namespace net2net {

using torch_az::VPNetModel;

// "layers.N.rest" -> {N, "rest"}; {-1, name} if the name has no layer index.
inline std::pair<int, std::string> SplitLayer(const std::string& name) {
  if (!absl::StartsWith(name, "layers.")) return {-1, name};
  const size_t dot = name.find('.', 7);
  return {std::stoi(name.substr(7, dot - 7)), name.substr(dot + 1)};
}

// The trunk's width mapping: new channel c copies old channel g[c].
struct Mapping {
  std::vector<int64_t> g;       // Size W'.
  std::vector<double> copies;   // Per old channel, how many new channels copy it.
};

inline Mapping MakeMapping(int old_width, int new_width, std::mt19937* rng) {
  Mapping m;
  m.copies.assign(old_width, 0);
  for (int c = 0; c < new_width; ++c) {
    const int64_t from =
        c < old_width ? c : std::uniform_int_distribution<int64_t>(0, old_width - 1)(*rng);
    m.g.push_back(from);
    m.copies[from] += 1;
  }
  return m;
}

// `dst` (the grown network's tensor) from `src` (the source's): every dimension of size W
// in src and W' in dst is the trunk's width — dimension 0 (outputs) copies by g,
// dimension 1 (inputs) copies by g and divides by the copies (unless `divide` is off: the
// control). New output channels get noise of `noise` x src's standard deviation.
inline void Grow(const torch::Tensor& src, torch::Tensor dst, int old_width, int new_width,
          const Mapping& m, bool divide, double noise, std::mt19937* rng) {
  torch::Tensor t = src.detach().clone();
  const torch::Tensor index = torch::tensor(m.g, torch::kInt64).to(t.device());
  bool out_mapped = false;
  for (int64_t d = 0; d < std::min<int64_t>(t.dim(), 2); ++d) {
    if (old_width == new_width || t.size(d) != old_width || dst.size(d) != new_width) continue;
    t = t.index_select(d, index);
    if (d == 0) out_mapped = true;
    if (d == 1 && divide) {
      std::vector<float> scale;
      for (int64_t c : m.g) scale.push_back(1.0f / static_cast<float>(m.copies[c]));
      std::vector<int64_t> shape(t.dim(), 1);
      shape[1] = new_width;
      t = t * torch::tensor(scale).to(t.device()).view(shape);
    }
  }
  if (noise > 0 && out_mapped && t.is_floating_point() && t.dim() >= 2) {
    const double sd = src.std().item<double>() * noise;
    torch::Tensor rows = t.narrow(0, old_width, new_width - old_width);
    std::normal_distribution<float> n(0.0f, static_cast<float>(sd));
    std::vector<float> values(rows.numel());
    for (float& v : values) v = n(*rng);
    rows.add_(torch::tensor(values).to(t.device()).view(rows.sizes()));
  }
  SPIEL_CHECK_TRUE(t.sizes() == dst.sizes());
  dst.copy_(t);
}

// Fills `model` (W' x D') from `source` (W x D). Returns how many tensors were grown,
// copied and set to the identity block.
inline std::string GrowInto(VPNetModel& source, VPNetModel& model, int old_width, int new_width,
                     int old_depth, int new_depth, const Mapping& m, bool divide,
                     double noise, std::mt19937* rng) {
  torch::NoGradGuard no_grad;
  std::map<std::string, torch::Tensor> from;
  for (auto& [name, t] : source.NamedTensors()) from[name] = t;
  int grown = 0, copied = 0, identity = 0, fresh = 0;
  for (auto& [name, t] : model.NamedTensors()) {
    auto [layer, rest] = SplitLayer(name);
    std::string source_name = name;
    if (layer == new_depth + 1) {  // The output block.
      source_name = absl::StrFormat("layers.%d.%s", old_depth + 1, rest);
    } else if (layer > old_depth) {  // A new residual block: the identity.
      if (absl::StrContains(rest, "batch_norm_2.weight") ||
          absl::StrContains(rest, "batch_norm_2.bias")) {
        t.zero_();
        ++identity;
      } else {
        ++fresh;
      }
      continue;
    }
    auto it = from.find(source_name);
    if (it == from.end()) SpielFatalError("No source tensor for " + name);
    if (it->second.sizes() == t.sizes()) {
      t.copy_(it->second);
      ++copied;
    } else {
      Grow(it->second, t, old_width, new_width, m, divide, noise, rng);
      ++grown;
    }
  }
  return absl::StrFormat("%d tensors grown, %d copied, %d set for the identity, %d fresh",
                         grown, copied, identity, fresh);
}

}  // namespace net2net
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_NET2NET_H_
