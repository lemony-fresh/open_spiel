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

#ifndef THUD_EXPERIMENTS_UNSAFE_TARGETS_H_
#define THUD_EXPERIMENTS_UNSAFE_TARGETS_H_

// The taper's condition (c) (thud/DESIGN_dwarf_signal.md, section 5; decided by the
// user 2026-10-15): the loop that amplifies unsafe dwarf moves must keep shrinking after
// a taper step. R = the search's share of visits on unsafe dwarf moves / the network's
// prior mass on them, both averaged over the positions; the loop's gain after a step from
// teacher weight w to w' is (1 - w') x R; a step is allowed only if that gain, with R at
// the upper end of its 95% interval, is at most 0.8. In a header so that
// unsafe_targets_test.cc can test it; az_unsafe_targets.cc measures it.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace thud_az {
namespace unsafe_targets {

// A move's destination square, from its notation "(r,c)-(r,c)" or "(r,c)-(r,c)x".
inline std::pair<int, int> Destination(const std::string& move) {
  static const std::regex pattern(R"(\(\d+,\d+\)-\((\d+),(\d+)\))");
  std::smatch m;
  SPIEL_CHECK_TRUE(std::regex_search(move, m, pattern));
  return {std::stoi(m[1]), std::stoi(m[2])};
}

// An unsafe dwarf move, as az_blunder_probe and the regression gates define it: it
// captures nothing and ends next to a troll (king's moves). `state` has the dwarfs to move.
inline bool IsUnsafe(const State& state, Action action) {
  const std::string move = state.ActionToString(action);
  if (move.back() == 'x') return false;
  const auto [row, col] = Destination(move);
  std::unique_ptr<State> after = state.Child(action);
  const auto& t = down_cast<const thud::ThudState&>(*after);
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      if ((dr || dc) && thud::IsOnBoard(row + dr, col + dc) &&
          t.CellAt(row + dr, col + dc) == thud::Cell::kTroll) {
        return true;
      }
    }
  }
  return false;
}

// R: the mean visit share on unsafe moves over the mean prior mass on them.
inline double Ratio(const std::vector<double>& prior, const std::vector<double>& visits) {
  double p = 0, v = 0;
  for (size_t i = 0; i < prior.size(); ++i) {
    p += prior[i];
    v += visits[i];
  }
  return p > 0 ? v / p : NAN;
}

// The upper end of R's 95% interval: the 97.5th percentile of R over `resamples`
// bootstrap resamples of the positions (seeded).
inline double RatioUpper(const std::vector<double>& prior, const std::vector<double>& visits,
                         int resamples = 2000, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, prior.size() - 1);
  std::vector<double> rs;
  rs.reserve(resamples);
  for (int b = 0; b < resamples; ++b) {
    double p = 0, v = 0;
    for (size_t i = 0; i < prior.size(); ++i) {
      const size_t j = pick(rng);
      p += prior[j];
      v += visits[j];
    }
    rs.push_back(p > 0 ? v / p : INFINITY);
  }
  std::sort(rs.begin(), rs.end());
  return rs[std::min(rs.size() - 1, static_cast<size_t>(std::ceil(0.975 * rs.size())) - 1)];
}

// Whether condition (c) allows a step from weight `w` to `w - step`: the loop's gain after
// it, (1 - (w - step)) x r_upper, at most `max_gain` (0.8).
inline bool TaperAllowed(double w, double r_upper, double step = 0.1, double max_gain = 0.8) {
  return (1 - (w - step)) * r_upper <= max_gain + 1e-12;
}

}  // namespace unsafe_targets
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_EXPERIMENTS_UNSAFE_TARGETS_H_
