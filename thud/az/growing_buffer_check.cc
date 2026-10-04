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

// Checks the growing buffer of our copy's learner (thud/az/alpha_zero.h:
// GrowingBufferSize, SampleNewest; thud/PLAN.md, runs G and G'):
//
//   1. GrowingBufferSize against values worked out by hand (start size 65,536): all positions
//      up to the start, then 65,536 (1 + 0.4 ((N / 65,536)^0.75 - 1) / 0.75).
//   2. SampleNewest on a buffer of numbered elements, before and after it wraps
//      around: every sample among the newest `newest`, no element twice, as many as
//      asked; `newest` beyond what the buffer holds samples all it holds; over many
//      draws every one of the newest is drawn, equally often by a chi-square test
//      (statistic below its 99.9% point for 299 degrees of freedom, ~375).
//      Control: CircularBuffer::Sample (upstream's, the whole buffer) draws older
//      elements too.
//
//   thud/az/build.sh thud/az/growing_buffer_check.cc
//   build-shared/growing_buffer_check
//
// Exits non-zero on any failure.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/utils/circular_buffer.h"
#include "thud/az/alpha_zero.h"

namespace {

using open_spiel::thud_az::torch_az::GrowingBufferSize;
using open_spiel::thud_az::torch_az::SampleNewest;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

}  // namespace

int main() {
  // 1. By hand: (350,000 / 65,536)^0.75 = 5.3406^0.75 = 3.5126; (3.5126 - 1) / 0.75
  //    = 3.3501; 1 + 0.4 * 3.3501 = 2.3400; * 65,536 = 153,355 (to rounding).
  //    1,000,000: 15.2588^0.75 = 7.7160; 6.7160 / 0.75 * 0.4 = 3.5819; 4.5819 * 65,536
  //    = 300,280 (to rounding).
  Expect(GrowingBufferSize(10000, 65536) == 10000, "below the start: every position");
  Expect(GrowingBufferSize(65536, 65536) == 65536, "at the start: every position");
  Expect(std::abs(GrowingBufferSize(350000, 65536) - 153355) < 200,
         absl::StrCat("at 350,000 generated: ", GrowingBufferSize(350000, 65536),
                      " (by hand ~153,355)"));
  Expect(std::abs(GrowingBufferSize(1000000, 65536) - 300280) < 400,
         absl::StrCat("at 1,000,000 generated: ", GrowingBufferSize(1000000, 65536),
                      " (by hand ~300,280)"));

  // 2. Buffers of numbered elements.
  std::mt19937 rng(20261004);
  for (int added : {500, 2500}) {  // Before and after a buffer of 1,000 wraps.
    open_spiel::CircularBuffer<int> buffer(1000);
    for (int i = 0; i < added; ++i) buffer.Add(i);
    const int newest = 300;
    std::vector<int> drawn = SampleNewest(buffer, &rng, newest, 100);
    const std::set<int> distinct(drawn.begin(), drawn.end());
    const bool among_newest = std::all_of(drawn.begin(), drawn.end(), [&](int x) {
      return x >= added - newest && x < added;
    });
    Expect(drawn.size() == 100 && distinct.size() == 100 && among_newest,
           absl::StrCat(added, " added: 100 distinct samples, all among the newest ",
                        newest));
    std::vector<int> all = SampleNewest(buffer, &rng, 5000, 2000);
    Expect(all.size() == std::min(added, 1000),
           absl::StrCat(added, " added: newest beyond what the buffer holds samples all of it: ",
                        all.size()));
    std::map<int, int> counts;
    for (int draw = 0; draw < 3000; ++draw) {
      for (int x : SampleNewest(buffer, &rng, newest, 10)) ++counts[x];
    }
    // 30,000 draws over 300: 100 each expected. Chi-square with 299 degrees of
    // freedom: mean 299, 99.9% point 299 + 3.09 sqrt(2 * 299) = 375.
    double chi2 = 0;
    for (const auto& [x, n] : counts) chi2 += (n - 100.0) * (n - 100.0) / 100.0;
    Expect(counts.size() == newest && chi2 < 375,
           absl::StrCat(added, " added: every one of the newest drawn, uniformly: "
                        "chi-square ", static_cast<int>(chi2), " (299 expected, < 375)"));
    std::vector<int> upstream = buffer.Sample(&rng, 100);
    Expect(std::any_of(upstream.begin(), upstream.end(),
                       [&](int x) { return x < added - newest; }),
           absl::StrCat(added, " added: control, upstream's Sample draws older ones too"));
  }
  std::cout << (failures ? "GROWING BUFFER CHECK FAILED" : "growing buffer check passed")
            << std::endl;
  return failures ? 1 : 0;
}
