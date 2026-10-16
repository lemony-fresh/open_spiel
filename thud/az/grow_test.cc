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

// Tests Net2Net (thud/az/net2net.h, used by az_grow; thud/PLAN.md, the GPU readiness
// checklist) on small networks built here — width 8, depth 2 — whose weights, batch-norm
// scales and shifts and running statistics are all random (fresh batch norms have mean
// 0, variance 1, scale 1 and shift 0, which would hide a batch norm left uncopied):
//
//   1. The mapping: the first W channels map to themselves, every old channel is copied
//      at least once, the copies add up to W'.
//   2. Function preserved, the outputs (values and every legal move's probability) equal
//      the source's within 1e-5 on positions of random games, for: wider 8 -> 13 (uneven
//      copies) with the convolutional head and the input planes; deeper 2 -> 4; both, 8 x 2
//      -> 16 x 3; the linear head without the input planes, 8 x 2 -> 12 x 3; and no growth
//      at all (every tensor copied).
//   3. Controls, each must differ by more than 10 x the tolerance (so the test would fail
//      them by a wide margin): the widening without the division by
//      the copies; the widening with the new channels' batch-norm variances reset to 1
//      (a batch norm left uncopied); the deepening with the new blocks' last batch-norm
//      scale not zeroed.
//   4. Every tensor of the grown network accounted for (grown, copied, set for the
//      identity, or fresh in a new block — nothing else).
//   5. Symmetry: after 30 learning steps on random positions, a copied channel's incoming
//      weights still equal its original's without noise (copies stay copies — why the
//      noise exists) and differ with noise 0.01; and with noise the outputs move only a
//      little before learning (< 0.05).
//   6. A grown network saves and reloads with identical outputs.
//
//   thud/run_tests.sh grow_test   # builds and runs it
//   build-tests/bin/grow_test

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/net2net.h"
#include "thud/az/thud_features.h"
#include "thud/az/vpnet.h"

namespace {

namespace az = open_spiel::thud_az;
using az::net2net::GrowInto;
using az::net2net::MakeMapping;
using az::net2net::Mapping;
using az::torch_az::VPNetModel;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++failures;
}

std::vector<VPNetModel::InferenceInputs> RandomPositions(const open_spiel::Game& game,
                                                         std::mt19937* rng, int count) {
  std::vector<VPNetModel::InferenceInputs> out;
  while (out.size() < count) {
    std::unique_ptr<open_spiel::State> s = game.NewInitialState();
    for (int ply = 0; !s->IsTerminal() && ply < 300 && out.size() < count; ++ply) {
      if (ply % 7 == 3) out.push_back({s->LegalActions(), s->ObservationTensor()});
      const std::vector<open_spiel::Action> legal = s->LegalActions();
      s->ApplyAction(legal[(*rng)() % legal.size()]);
    }
  }
  return out;
}

double MaxDifference(VPNetModel& a, VPNetModel& b,
                     const std::vector<VPNetModel::InferenceInputs>& in) {
  const auto va = a.Inference(in), vb = b.Inference(in);
  double d = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    d = std::max(d, std::abs(va[i].value - vb[i].value));
    for (size_t j = 0; j < va[i].policy.size(); ++j) {
      d = std::max(d, std::abs(va[i].policy[j].second - vb[i].policy[j].second));
    }
  }
  return d;
}

std::string Dir(const std::string& base, const std::string& name) {
  return base + "/" + name;
}

// A new network in its own folder.
std::unique_ptr<VPNetModel> Network(const open_spiel::Game& game, const std::string& dir,
                                    const std::string& nn_model, int width, int depth,
                                    bool planes) {
  std::filesystem::create_directories(dir);
  SPIEL_CHECK_TRUE(az::torch_az::CreateGraphDef(game, 0.001, 0.0001, dir, "vpnet.pb",
                                                nn_model, width, depth, /*verbose=*/false,
                                                planes ? az::kNumFeaturePlanes : 0));
  return std::make_unique<VPNetModel>(game, dir, "vpnet.pb", "/cpu:0");
}

// Random batch-norm scales, shifts and running statistics (fresh ones are trivial).
void RandomiseBatchNorms(VPNetModel& model, std::mt19937* rng) {
  torch::NoGradGuard no_grad;
  std::uniform_real_distribution<float> scale(0.5, 1.5), shift(-0.3, 0.3), mean(-0.5, 0.5),
      variance(0.2, 3.0);
  for (auto& [name, t] : model.NamedTensors()) {
    if (!absl::StrContains(name, "batch_norm")) continue;
    auto fill = [&](std::uniform_real_distribution<float>& d) {
      std::vector<float> v(t.numel());
      for (float& x : v) x = d(*rng);
      t.copy_(torch::tensor(v).view(t.sizes()));
    };
    if (absl::EndsWith(name, ".weight")) fill(scale);
    if (absl::EndsWith(name, ".bias")) fill(shift);
    if (absl::EndsWith(name, ".running_mean")) fill(mean);
    if (absl::EndsWith(name, ".running_var")) fill(variance);
  }
}

struct Grown {
  std::unique_ptr<VPNetModel> model;
  Mapping mapping;
  std::string summary;
};

// `source` (W x D) grown to W' x D' in `dir`; with `divide` off, or `noise`, as given.
Grown GrowNetwork(const open_spiel::Game& game, VPNetModel& source, const std::string& dir,
                  const std::string& nn_model, bool planes, int old_width, int old_depth,
                  int width, int depth, bool divide, double noise, unsigned seed) {
  Grown g;
  g.model = Network(game, dir, nn_model, width, depth, planes);
  std::mt19937 rng(seed), noise_rng(seed + 1);
  g.mapping = MakeMapping(old_width, width, &rng);
  g.summary = GrowInto(source, *g.model, old_width, width, old_depth, depth, g.mapping, divide,
                       noise, &noise_rng);
  return g;
}

// The counts in GrowInto's summary, summed.
int Accounted(const std::string& summary) {
  int a = 0, b = 0, c = 0, d = 0;
  SPIEL_CHECK_EQ(std::sscanf(summary.c_str(),
                             "%d tensors grown, %d copied, %d set for the identity, %d fresh",
                             &a, &b, &c, &d),
                 4);
  return a + b + c + d;
}

// A batch of training inputs for random positions: uniform policies, random values.
std::vector<VPNetModel::TrainInputs> TrainBatch(const open_spiel::Game& game, std::mt19937* rng,
                                                int count) {
  std::vector<VPNetModel::TrainInputs> out;
  std::uniform_real_distribution<double> value(-1, 1);
  for (const auto& p : RandomPositions(game, rng, count)) {
    open_spiel::ActionsAndProbs policy;
    for (open_spiel::Action a : p.legal_actions) {
      policy.push_back({a, 1.0 / p.legal_actions.size()});
    }
    out.push_back({p.legal_actions, p.observations, policy, value(*rng)});
  }
  return out;
}

}  // namespace

int main() {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const std::string base =
      (std::filesystem::temp_directory_path() / ("grow_test_" + std::to_string(getpid())))
          .string();
  std::filesystem::remove_all(base);
  torch::manual_seed(7);  // LibTorch's default generator is seeded nondeterministically.
  std::mt19937 rng(7);
  const auto positions = RandomPositions(*game, &rng, 40);
  const double tolerance = 1e-5, control_floor = 10 * tolerance;

  // The sources: the convolutional head with the input planes, and the linear head without.
  auto conv = Network(*game, Dir(base, "conv"), "resnet_conv_policy", 8, 2, true);
  auto linear = Network(*game, Dir(base, "linear"), "resnet", 8, 2, false);
  RandomiseBatchNorms(*conv, &rng);
  RandomiseBatchNorms(*linear, &rng);

  // 1. The mapping.
  {
    std::mt19937 r(3);
    const Mapping m = MakeMapping(8, 13, &r);
    bool ok = m.g.size() == 13;
    double copies = 0;
    for (int c = 0; c < 8; ++c) ok = ok && m.g[c] == c && m.copies[c] >= 1;
    for (double x : m.copies) copies += x;
    Check(ok && copies == 13, "1. the mapping: the first W to themselves, every old channel "
                              "at least once, the copies adding up to W'");
  }

  // 2-4. Function preserved; the controls; every tensor accounted for.
  struct Case {
    std::string name, nn_model;
    VPNetModel* source;
    bool planes;
    int width, depth;
  };
  const std::vector<Case> cases = {
      {"wider 8 -> 13 (convolutional head, input planes)", "resnet_conv_policy", conv.get(),
       true, 13, 2},
      {"deeper 2 -> 4", "resnet_conv_policy", conv.get(), true, 8, 4},
      {"both, 8 x 2 -> 16 x 3", "resnet_conv_policy", conv.get(), true, 16, 3},
      {"linear head, no input planes, 8 x 2 -> 12 x 3", "resnet", linear.get(), false, 12, 3},
      {"no growth", "resnet_conv_policy", conv.get(), true, 8, 2},
  };
  int n = 0;
  for (const Case& c : cases) {
    Grown g = GrowNetwork(*game, *c.source, Dir(base, "case" + std::to_string(n++)),
                          c.nn_model, c.planes, 8, 2, c.width, c.depth, /*divide=*/true,
                          /*noise=*/0, 11);
    const double d = MaxDifference(*g.model, *c.source, positions);
    Check(d <= tolerance, absl::StrFormat("2. %s: the outputs equal the source's (%.2g)",
                                          c.name, d));
    Check(Accounted(g.summary) == static_cast<int>(g.model->NamedTensors().size()),
          absl::StrFormat("4. %s: every tensor accounted for (%s)", c.name, g.summary));
    if (c.width > 8) {
      Grown control = GrowNetwork(*game, *c.source, Dir(base, "control" + std::to_string(n)),
                                  c.nn_model, c.planes, 8, 2, c.width, c.depth,
                                  /*divide=*/false, 0, 11);
      const double cd = MaxDifference(*control.model, *c.source, positions);
      Check(cd > control_floor,
            absl::StrFormat("3. %s, control: without the division it differs (%.2g)", c.name,
                            cd));
    }
  }
  {
    // Control: a batch norm left uncopied (the new channels' variances reset to 1).
    Grown g = GrowNetwork(*game, *conv, Dir(base, "bn_control"), "resnet_conv_policy", true, 8,
                          2, 13, 2, true, 0, 11);
    {
      torch::NoGradGuard no_grad;
      for (auto& [name, t] : g.model->NamedTensors()) {
        if (absl::EndsWith(name, "running_var") && t.size(0) == 13) {
          t.narrow(0, 8, 5).fill_(1.0);
        }
      }
    }
    const double d = MaxDifference(*g.model, *conv, positions);
    Check(d > control_floor,
          absl::StrFormat("3. control: with a batch norm's statistics left uncopied it "
                          "differs (%.2g)", d));
  }
  {
    // Control: the new blocks' last batch-norm scale not zeroed.
    Grown g = GrowNetwork(*game, *conv, Dir(base, "id_control"), "resnet_conv_policy", true, 8,
                          2, 8, 4, true, 0, 11);
    {
      torch::NoGradGuard no_grad;
      for (auto& [name, t] : g.model->NamedTensors()) {
        auto [layer, rest] = az::net2net::SplitLayer(name);
        if (layer > 2 && layer <= 4 && absl::StrContains(rest, "batch_norm_2.weight")) {
          t.fill_(1.0);
        }
      }
    }
    const double d = MaxDifference(*g.model, *conv, positions);
    Check(d > control_floor,
          absl::StrFormat("3. control: with the new blocks not the identity it differs "
                          "(%.2g)", d));
  }

  // 5. Symmetry, and the noise.
  {
    // Channel pairs (original, copy) in the grown trunk.
    auto incoming_gap = [&](VPNetModel& model, const Mapping& m) {
      torch::NoGradGuard no_grad;
      torch::Tensor w = model.Parameter("layers.1.res_0_conv_1.weight");
      double gap = 0;
      for (int c = 8; c < static_cast<int>(m.g.size()); ++c) {
        gap = std::max(gap, (w[c] - w[m.g[c]]).abs().max().item<double>());
      }
      return gap;
    };
    std::mt19937 train_rng(5);
    const auto batch = TrainBatch(*game, &train_rng, 32);
    for (double noise : {0.0, 0.01}) {
      Grown g = GrowNetwork(*game, *conv, Dir(base, absl::StrFormat("sym%g", noise)),
                            "resnet_conv_policy", true, 8, 2, 13, 2, true, noise, 11);
      const double before = MaxDifference(*g.model, *conv, positions);
      for (int i = 0; i < 30; ++i) g.model->Learn(batch);
      const double gap = incoming_gap(*g.model, g.mapping);
      if (noise == 0) {
        Check(gap <= 1e-6, absl::StrFormat("5. without noise, after 30 learning steps the "
                                           "copies still equal their originals (%.2g)", gap));
      } else {
        Check(gap > 1e-4, absl::StrFormat("5. with noise 0.01 the copies differ after "
                                          "learning (%.2g)", gap));
        Check(before < 0.05, absl::StrFormat("5. with noise 0.01 the outputs move only a "
                                             "little before learning (%.2g)", before));
      }
    }
  }

  // 6. Save and reload.
  {
    Grown g = GrowNetwork(*game, *conv, Dir(base, "saved"), "resnet_conv_policy", true, 8, 2,
                          16, 3, true, 0, 11);
    g.model->SaveCheckpoint(1);
    VPNetModel reloaded(*game, Dir(base, "saved"), "vpnet.pb", "/cpu:0");
    reloaded.LoadCheckpoint(1);
    const double d = MaxDifference(reloaded, *g.model, positions);
    Check(d == 0, absl::StrFormat("6. a grown network reloads with identical outputs (%.2g)", d));
  }

  std::filesystem::remove_all(base);
  std::cout << (failures == 0 ? "ALL PASSED" : absl::StrFormat("%d FAILED", failures))
            << std::endl;
  return failures == 0 ? 0 : 1;
}
