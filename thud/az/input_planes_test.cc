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

// Tests the input planes from the rules inside the network (thud/PLAN.md, the input
// planes, as decided; our copy's VPNetModel builds each input from the observation and,
// if the model takes them, the 50 planes of thud/az/thud_features.h — the planes
// themselves are checked by thud/az/features_test.cc):
//
//   1. net surgery: a network with the planes, its weights copied from one without
//      (CopyParametersFrom with widen_inputs: the new input channels zero), gives exactly
//      the other's values and policies — for the convolutional and the linear policy
//      head;
//   2. each plane reaches the network on its own channel: with weight on that plane's
//      input channel only, the outputs change in exactly the positions where the plane
//      is nonzero somewhere, and nowhere else (control: judged against the next
//      plane's positions instead, it must fail);
//   3. learning reaches exactly the planes present: one learning step on a batch moves
//      the input weights of every plane nonzero somewhere in the batch and leaves every
//      other plane's weights at zero (a zero weight with zero input has no gradient, and
//      its weight decay is zero too).
//
// Positions: one capture of each kind, direction and distance the board has room for
// (thud/az/capture_patterns.h; part 2 then requires every plane the board allows to be
// probed, the planes never lit being exactly those without room), random games, the
// opening and, with LOG_DIR, run self-play (all positions of each actor's last game).
//
//   thud/run_tests.sh input_planes_test   # builds and runs it
//   build-tests/bin/input_planes_test SCRATCH_DIR [LOG_DIR]

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/capture_patterns.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::torch_az::CreateGraphDef;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;
namespace az = open_spiel::thud_az;
constexpr int n = thud::kBoardSize;
const char* kInputWeight = "layers.0.input_conv.weight";

int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  failures += !ok;
}

std::vector<VPNetModel::InferenceInputs> Inputs(
    const std::vector<std::unique_ptr<State>>& positions) {
  std::vector<VPNetModel::InferenceInputs> in;
  for (const auto& s : positions) in.push_back({s->LegalActions(), s->ObservationTensor()});
  return in;
}

// Per position, whether the two networks' outputs differ at all.
std::vector<bool> Moved(VPNetModel& a, VPNetModel& b,
                        const std::vector<VPNetModel::InferenceInputs>& in) {
  const auto va = a.Inference(in), vb = b.Inference(in);
  std::vector<bool> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    bool moved = va[i].value != vb[i].value;
    for (size_t j = 0; j < va[i].policy.size(); ++j) {
      moved |= va[i].policy[j].second != vb[i].policy[j].second;
    }
    out[i] = moved;
  }
  return out;
}

// Per position and plane, whether the plane is nonzero anywhere.
std::vector<std::vector<bool>> PlanesPresent(const std::vector<VPNetModel::InferenceInputs>& in) {
  std::vector<std::vector<bool>> out;
  for (const auto& x : in) {
    std::vector<float> f;
    az::AppendFeaturePlanes(x.observations, &f);
    std::vector<bool> present(az::kNumFeaturePlanes, false);
    for (int p = 0; p < az::kNumFeaturePlanes; ++p) {
      for (int i = 0; i < n * n; ++i) present[p] = present[p] || f[p * n * n + i] != 0;
    }
    out.push_back(present);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: input_planes_test SCRATCH_DIR [LOG_DIR]" << std::endl;
    return 1;
  }
  const std::string scratch = argv[1];
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::vector<std::unique_ptr<State>> positions;
  positions.push_back(game->NewInitialState());
  // The planes the board has room for: lit by some pattern.
  std::vector<bool> possible(az::kNumFeaturePlanes, false);
  possible[az::kTrollLandsNear] = possible[az::kHurlLands] = true;
  for (const az::CapturePattern& p : az::AllCapturePatterns(az::kMaxCaptureDistance)) {
    positions.push_back(game->NewInitialState(p.board + "to_move=dwarfs"));
    possible[az::MovePlane(p.d, p.k)] = true;
  }
  std::mt19937 rng(20261011);
  for (int g = 0; g < 20; ++g) {
    std::unique_ptr<State> s = game->NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 300; ++ply) {
      if (ply % 9 == 4) positions.push_back(s->Clone());
      const std::vector<Action> legal = s->LegalActions();
      s->ApplyAction(legal[rng() % legal.size()]);
    }
  }
  if (argc > 2) {
    for (const auto& entry : std::filesystem::directory_iterator(argv[2])) {
      if (!absl::StartsWith(entry.path().filename().string(), "log-actor-")) continue;
      std::string last;
      std::ifstream in(entry.path());
      for (std::string line; std::getline(in, line);) {
        if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
          last = line;
        }
      }
      if (last.empty()) continue;
      std::unique_ptr<State> s = game->NewInitialState();
      for (absl::string_view move :
           absl::StrSplit(last.substr(last.find("Actions: ") + 9), ' ', absl::SkipEmpty())) {
        if (s->IsTerminal()) break;
        positions.push_back(s->Clone());
        for (Action a : s->LegalActions()) {
          if (s->ActionToString(a) == move) {
            s->ApplyAction(a);
            break;
          }
        }
      }
    }
  }
  const auto in = Inputs(positions);
  const auto present = PlanesPresent(in);
  std::cout << positions.size() << " positions" << std::endl;

  for (const std::string nn_model : {"resnet_conv_policy", "resnet"}) {
    // Two networks of the same shape, one without the planes and one with.
    const std::string without_dir = absl::StrCat(scratch, "/", nn_model, "_without");
    const std::string with_dir = absl::StrCat(scratch, "/", nn_model, "_with");
    std::filesystem::create_directories(without_dir);
    std::filesystem::create_directories(with_dir);
    SPIEL_CHECK_TRUE(CreateGraphDef(*game, 0.001, 0.0001, without_dir, "vpnet.pb", nn_model,
                                    32, 2));
    SPIEL_CHECK_TRUE(CreateGraphDef(*game, 0.001, 0.0001, with_dir, "vpnet.pb", nn_model, 32,
                                    2, /*verbose=*/false, az::kNumFeaturePlanes));
    torch::manual_seed(1);
    VPNetModel without(*game, without_dir, "vpnet.pb", "/cpu:0");
    VPNetModel with(*game, with_dir, "vpnet.pb", "/cpu:0");
    Check(!without.TakesInputPlanes() && with.TakesInputPlanes(),
          nn_model + ": the second network takes the input planes, the first does not");

    // 1. Net surgery.
    const auto copied = with.CopyParametersFrom(
        without, [](const std::string&) { return true; }, /*widen_inputs=*/true);
    const auto moved = Moved(with, without, in);
    int differ = 0;
    for (bool m : moved) differ += m;
    Check(differ == 0, nn_model + ": 1. after the surgery (" + std::to_string(copied.size()) +
                           " tensors) the outputs are exactly the network's without the "
                           "planes (" + std::to_string(differ) + " positions differ)");

    // 2. Each plane on its own channel.
    torch::Tensor w = with.Parameter(kInputWeight);
    int mismatches = 0, control_mismatches = 0, planes_seen = 0;
    std::vector<int> never;
    for (int p = 0; p < az::kNumFeaturePlanes; ++p) {
      const int channel = thud::kNumObservationPlanes + p;
      {
        torch::NoGradGuard no_grad;
        w.select(1, channel).fill_(0.5);
      }
      const auto m = Moved(with, without, in);
      {
        torch::NoGradGuard no_grad;
        w.select(1, channel).zero_();
      }
      int positives = 0;
      for (size_t i = 0; i < in.size(); ++i) {
        positives += present[i][p];
        mismatches += m[i] != present[i][p];
        control_mismatches += m[i] != present[i][(p + 1) % az::kNumFeaturePlanes];
      }
      if (positives == 0) {
        never.push_back(p);
      } else {
        ++planes_seen;
      }
    }
    std::string never_text;
    for (int p : never) {
      never_text += p < az::kNumMovePlanes
                        ? absl::StrCat(" (", az::MoveOfPlane(p).d, ",", az::MoveOfPlane(p).k, ")")
                        : absl::StrCat(" ", p);
    }
    std::cout << "   " << planes_seen << " planes nonzero in some position; never:"
              << (never_text.empty() ? " none" : never_text) << " (direction, distance)"
              << std::endl;
    Check(mismatches == 0 && planes_seen > 0,
          nn_model + ": 2. weight on one plane's channel moves the outputs exactly where "
                     "that plane is nonzero (" + std::to_string(mismatches) +
              " position-plane mismatches)");
    int never_possible = 0, n_possible = 0;
    for (int p = 0; p < az::kNumFeaturePlanes; ++p) n_possible += possible[p];
    for (int p : never) never_possible += possible[p];
    Check(never_possible == 0,
          nn_model + ": 2. every plane the board has room for (" + std::to_string(n_possible) +
              ") was probed; those never lit are exactly the ones without room");
    Check(control_mismatches > 0,
          nn_model + ": 2. control: judged against the next plane, it fails (" +
              std::to_string(control_mismatches) + " mismatches)");

    // 3. One learning step reaches exactly the planes present in the batch.
    std::vector<VPNetModel::TrainInputs> batch;
    std::vector<bool> in_batch(az::kNumFeaturePlanes, false);
    for (size_t i = 0; i < in.size(); i += 3) {
      VPNetModel::TrainInputs t{in[i].legal_actions, in[i].observations, {}, 0.0};
      for (Action a : t.legal_actions) t.policy.push_back({a, 1.0 / t.legal_actions.size()});
      batch.push_back(t);
      for (int p = 0; p < az::kNumFeaturePlanes; ++p) in_batch[p] = in_batch[p] || present[i][p];
      if (batch.size() == 256) break;
    }
    with.Learn(batch);
    int moved_present = 0, n_present = 0, moved_absent = 0, n_absent = 0;
    {
      torch::NoGradGuard no_grad;
      for (int p = 0; p < az::kNumFeaturePlanes; ++p) {
        const bool changed =
            w.select(1, thud::kNumObservationPlanes + p).abs().max().item<float>() > 0;
        if (in_batch[p]) {
          ++n_present;
          moved_present += changed;
        } else {
          ++n_absent;
          moved_absent += changed;
        }
      }
    }
    Check(n_present > 0 && moved_present == n_present && moved_absent == 0,
          nn_model + ": 3. one learning step moved the weights of all " +
              std::to_string(n_present) + " planes present in the batch (" +
              std::to_string(moved_present) + ") and of none of the " +
              std::to_string(n_absent) + " absent (" + std::to_string(moved_absent) + ")");
  }
  std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
  return failures ? 1 : 0;
}
