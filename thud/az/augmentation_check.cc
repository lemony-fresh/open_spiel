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

// Checks the symmetry augmentation of our copy of OpenSpiel's C++ AlphaZero
// (SymmetricTrainInputs in thud/az/vpnet.h; thud/PLAN.md Phase 6) against positions
// mirrored independently, as text: every 7th position of random games, under each of
// the 8 symmetries, with a random policy target over some legal moves.
//
//   1. The transformed observation and legal moves are those of the mirrored position.
//   2. The policy keeps its probabilities, and each of its moves leads in the mirrored
//      position to the mirror image of where it led: the target goes to the same moves.
//   3. The value stays, and symmetry 0 changes nothing.
//   Control: without the transformation, the observations of mirrored positions
//   differ (for symmetries 1-7), so the comparisons above can fail.
//
//   thud/az/build.sh thud/az/augmentation_check.cc
//   build-shared/augmentation_check
//
// Exits non-zero on any failure.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az::torch_az;
using open_spiel::Action;
using open_spiel::Game;
using open_spiel::State;

int failures = 0;

void Expect(bool ok, const std::string& what) {
  std::cout << (ok ? "ok      " : "FAILED  ") << what << std::endl;
  if (!ok) ++failures;
}

// The symmetries in thud.h's order, each a matrix on (row, col) offsets from the
// Thudstone, (7, 7): the identity, the quarter turns, the mirror images.
constexpr std::array<std::array<int, 4>, 8> kMatrices = {{{1, 0, 0, 1},
                                                          {0, 1, -1, 0},
                                                          {-1, 0, 0, -1},
                                                          {0, -1, 1, 0},
                                                          {1, 0, 0, -1},
                                                          {-1, 0, 0, 1},
                                                          {0, 1, 1, 0},
                                                          {0, -1, -1, 0}}};

// The position's text (15 rows, then the status line) with the board mirrored.
std::string MirrorText(const std::string& text, int symmetry) {
  std::vector<std::string> lines = absl::StrSplit(text, '\n');
  SPIEL_CHECK_EQ(lines.size(), 16);
  std::vector<std::string> rows(15, std::string(15, '?'));
  const std::array<int, 4>& m = kMatrices[symmetry];
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      rows[7 + m[0] * (r - 7) + m[1] * (c - 7)][7 + m[2] * (r - 7) + m[3] * (c - 7)] =
          lines[r][c];
    }
  }
  return absl::StrCat(absl::StrJoin(rows, "\n"), "\n", lines[15]);
}

}  // namespace

int main(int argc, char** argv) {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("thud");
  std::mt19937 rng(20261002);
  int positions = 0, observation_ok = 0, legal_ok = 0, policy_ok = 0, value_ok = 0,
      identity_ok = 0, differ_untransformed = 0, mirrored = 0;
  for (int g = 0; g < 4; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    for (int turn = 0; !state->IsTerminal(); ++turn) {
      const std::vector<Action> legal = state->LegalActions();
      if (turn % 7 == 0) {  // 7 is odd, so both sides are to move.
        ++positions;
        // A random target over up to 5 legal moves.
        az::VPNetModel::TrainInputs inputs{legal, state->ObservationTensor(), {}, 0};
        double total = 0;
        for (int i = 0; i < std::min<int>(5, legal.size()); ++i) {
          const double p = std::uniform_real_distribution<double>(0.1, 1)(rng);
          inputs.policy.push_back(
              {legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)], p});
          total += p;
        }
        for (auto& [action, p] : inputs.policy) p /= total;
        inputs.value = std::uniform_real_distribution<double>(-1, 1)(rng);

        const az::VPNetModel::TrainInputs same = az::SymmetricTrainInputs(inputs, 0);
        open_spiel::ActionsAndProbs sorted_policy = inputs.policy;  // As it sorts.
        std::sort(sorted_policy.begin(), sorted_policy.end());
        identity_ok += same.observations == inputs.observations &&
                       same.legal_actions == inputs.legal_actions &&
                       same.value == inputs.value && same.policy == sorted_policy;
        for (int s = 0; s < 8; ++s) {
          ++mirrored;
          const az::VPNetModel::TrainInputs t = az::SymmetricTrainInputs(inputs, s);
          std::unique_ptr<State> image =
              game->NewInitialState(MirrorText(state->ToString(), s));
          observation_ok += t.observations == image->ObservationTensor();
          legal_ok += t.legal_actions == image->LegalActions();
          value_ok += t.value == inputs.value;
          if (s > 0) differ_untransformed += inputs.observations != image->ObservationTensor();
          // Each target move, mirrored, leads to the mirror image of its child.
          bool ok = t.policy.size() == inputs.policy.size();
          double sum = 0;
          for (const auto& [action, p] : inputs.policy) {
            const Action image_action = open_spiel::thud::SymmetricAction(action, s);
            bool found = false;
            for (const auto& [a, q] : t.policy) found |= a == image_action && q == p;
            ok &= found &&
                  image->Child(image_action)->ToString() ==
                      MirrorText(state->Child(action)->ToString(), s);
            sum += p;
          }
          policy_ok += ok && std::abs(sum - 1) < 1e-9;
        }
      }
      state->ApplyAction(
          legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
    }
  }
  const int n = mirrored;
  Expect(observation_ok == n, absl::StrFormat("observations are the mirrored "
                                              "positions': %d of %d", observation_ok, n));
  Expect(legal_ok == n, absl::StrFormat("legal moves are the mirrored positions': "
                                        "%d of %d", legal_ok, n));
  Expect(policy_ok == n, absl::StrFormat("policy targets keep their probabilities and "
                                         "go to the mirrored moves: %d of %d",
                                         policy_ok, n));
  Expect(value_ok == n, absl::StrFormat("values stay: %d of %d", value_ok, n));
  Expect(identity_ok == positions,
         absl::StrFormat("symmetry 0 changes nothing: %d of %d", identity_ok,
                         positions));
  Expect(differ_untransformed > 0.9 * 7 * positions,
         absl::StrFormat("control: untransformed observations differ from the "
                         "mirrored positions' in %d of %d",
                         differ_untransformed, 7 * positions));
  std::cout << (failures ? "AUGMENTATION CHECK FAILED" : "augmentation check passed")
            << std::endl;
  return failures ? 1 : 0;
}
