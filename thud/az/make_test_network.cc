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

// A small untrained network for the checks (thud/run_checks.sh), so that checks which read
// a run folder (identity_check, pruning_check, untried_move_check, az_match_check.sh)
// need no trained run: `dir` (must not exist) gets vpnet.pb and checkpoints 0 to
// `checkpoints` - 1, each with its own random weights (LibTorch's initialisation, seeded
// by `seed` + the step, so reruns write the same networks), and random batch-norm running
// statistics and scales (fresh ones are trivial: mean 0, variance 1, scale 1, shift 0).
// `policy_scale` multiplies the policy head's last layer (its logits), so that the prior
// is peaked as a trained network's is, not almost uniform as a fresh one's: what
// pruning_check's forced playouts need to bind (sqrt(2 P N) of at least 1 playout).
// `value_bias` is added to the value head's last bias (player 0's value is the tanh of
// it): -1 puts the dwarfs' values near tanh(-1) = -0.76, as a trained network has them
// (the dwarfs lose), so that untried_move_check's control holds — upstream's rule, which
// gives an untried move 0, spreads the dwarfs' searches only when their values are below
// it.
//
//   build-checks/bin/make_test_network dir=DIR [nn_model=resnet nn_width=8 nn_depth=1 \
//     input_planes=0 checkpoints=2 seed=1 policy_scale=1 value_bias=0]

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

int main(int argc, char** argv) {
  namespace az = open_spiel::thud_az;
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value: " + arg);
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  auto get = [&](const std::string& k, const std::string& d) {
    return args.count(k) ? args[k] : d;
  };
  const std::string dir = get("dir", "");
  if (dir.empty()) {
    std::cerr << "Usage: make_test_network dir=DIR [nn_model=resnet nn_width=8 nn_depth=1 "
                 "input_planes=0 checkpoints=2 seed=1 policy_scale=1 value_bias=0]"
              << std::endl;
    return 1;
  }
  if (std::filesystem::exists(dir)) open_spiel::SpielFatalError(dir + " exists already");
  const int checkpoints = std::stoi(get("checkpoints", "2"));
  const unsigned seed = std::stoul(get("seed", "1"));
  const double policy_scale = std::stod(get("policy_scale", "1"));
  const double value_bias = std::stod(get("value_bias", "0"));
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  std::filesystem::create_directories(dir);
  SPIEL_CHECK_TRUE(az::torch_az::CreateGraphDef(
      *game, 0.001, 0.0001, dir, "vpnet.pb", get("nn_model", "resnet"),
      std::stoi(get("nn_width", "8")), std::stoi(get("nn_depth", "1")), /*verbose=*/false,
      get("input_planes", "0") == "1" ? az::kNumFeaturePlanes : 0));
  for (int step = 0; step < checkpoints; ++step) {
    torch::manual_seed(seed + step);
    az::torch_az::VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    std::mt19937 rng(seed + step);
    std::uniform_real_distribution<float> scale(0.5, 1.5), shift(-0.3, 0.3),
        mean(-0.5, 0.5), variance(0.2, 3.0);
    {
      torch::NoGradGuard no_grad;
      for (auto& [name, t] : model.NamedTensors()) {
        if (absl::StrContains(name, ".policy_linear.") ||
            absl::StrContains(name, ".policy_conv_2.")) {
          t.mul_(policy_scale);
        }
        if (absl::EndsWith(name, ".value_linear_2.bias")) t.add_(value_bias);
        if (!absl::StrContains(name, "batch_norm")) continue;
        std::uniform_real_distribution<float>* d =
            absl::EndsWith(name, ".weight")         ? &scale
            : absl::EndsWith(name, ".bias")         ? &shift
            : absl::EndsWith(name, ".running_mean") ? &mean
            : absl::EndsWith(name, ".running_var")  ? &variance
                                                    : nullptr;
        if (d == nullptr) continue;
        std::vector<float> v(t.numel());
        for (float& x : v) x = (*d)(rng);
        t.copy_(torch::tensor(v).view(t.sizes()));
      }
    }
    model.SaveCheckpoint(step);
  }
  std::cout << "Wrote " << dir << " with checkpoints 0 to " << checkpoints - 1 << std::endl;
  return 0;
}
