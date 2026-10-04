// Copyright 2021 DeepMind Technologies Limited
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

// Changed by the Thud-on-OpenSpiel authors, 2026: copied from
// open_spiel/algorithms/alpha_zero_torch/alpha_zero.h at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#ifndef THUD_AZ_ALPHA_ZERO_H_
#define THUD_AZ_ALPHA_ZERO_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel {
namespace thud_az {
namespace torch_az {

struct AlphaZeroConfig {
  std::string game;
  std::string path;
  std::string graph_def;
  std::string nn_model;
  int nn_width;
  int nn_depth;
  std::string devices;

  bool explicit_learning;
  double learning_rate;
  double weight_decay;
  int train_batch_size;
  int inference_batch_size;
  int inference_threads;
  int inference_cache;
  int replay_buffer_size;
  int replay_buffer_reuse;
  // Added: the batches each learning step trains on; 0 is upstream's one pass over the
  // buffer (its size / train_batch_size). Separates how much the network remembers
  // (the buffer) from how much it trains per step (thud/PLAN.md Phase 6, Step 2).
  int learner_batches;
  // Added: train on each sampled position turned or mirrored by a random one of
  // Thud's 8 board symmetries (SymmetricTrainInputs in thud/az/vpnet.h), as AlphaGo
  // Zero did with Go's; false is upstream's behaviour.
  bool symmetry_augmentation;
  // Added: KataGo's forced playouts and policy target pruning in self-play
  // (arXiv 1902.10565, section 3.2; PrunedRootVisits in thud/az/mcts.h), so that
  // the policy target is what the search concluded rather than where root noise
  // sent it; false is upstream's behaviour.
  bool policy_target_pruning;
  // Added: a growing buffer (thud/PLAN.md, runs G and G'): the learner samples only
  // the newest GrowingBufferSize(positions generated, replay_buffer_start_size)
  // positions stored, the buffer growing towards replay_buffer_size; 0 samples
  // every position stored, as upstream.
  int replay_buffer_start_size;
  int checkpoint_freq;
  int evaluation_window;

  double uct_c;
  int max_simulations;
  double policy_alpha;
  double policy_epsilon;
  std::string untried_move_value;  // Added: see UntriedMoveValue in thud/az/mcts.h.
  double untried_move_reduction;
  double temperature;
  double temperature_drop;
  double cutoff_probability;
  double cutoff_value;

  int actors;
  int evaluators;
  int eval_levels;
  int max_steps;

  json::Object ToJson() const {
    return json::Object({
        {"game", game},
        {"path", path},
        {"graph_def", graph_def},
        {"nn_model", nn_model},
        {"nn_width", nn_width},
        {"nn_depth", nn_depth},
        {"devices", devices},
        {"explicit_learning", explicit_learning},
        {"learning_rate", learning_rate},
        {"weight_decay", weight_decay},
        {"train_batch_size", train_batch_size},
        {"inference_batch_size", inference_batch_size},
        {"inference_threads", inference_threads},
        {"inference_cache", inference_cache},
        {"replay_buffer_size", replay_buffer_size},
        {"replay_buffer_reuse", replay_buffer_reuse},
        {"learner_batches", learner_batches},
        {"symmetry_augmentation", symmetry_augmentation},
        {"policy_target_pruning", policy_target_pruning},
        {"replay_buffer_start_size", replay_buffer_start_size},
        {"checkpoint_freq", checkpoint_freq},
        {"evaluation_window", evaluation_window},
        {"uct_c", uct_c},
        {"max_simulations", max_simulations},
        {"policy_alpha", policy_alpha},
        {"policy_epsilon", policy_epsilon},
        {"untried_move_value", untried_move_value},
        {"untried_move_reduction", untried_move_reduction},
        {"temperature", temperature},
        {"temperature_drop", temperature_drop},
        {"cutoff_probability", cutoff_probability},
        {"cutoff_value", cutoff_value},
        {"actors", actors},
        {"evaluators", evaluators},
        {"eval_levels", eval_levels},
        {"max_steps", max_steps},
    });
  }

  void FromJson(const json::Object& config_json) {
    game = config_json.at("game").GetString();
    path = config_json.at("path").GetString();
    graph_def = config_json.at("graph_def").GetString();
    nn_model = config_json.at("nn_model").GetString();
    nn_width = config_json.at("nn_width").GetInt();
    nn_depth = config_json.at("nn_depth").GetInt();
    devices = config_json.at("devices").GetString();
    explicit_learning = config_json.at("explicit_learning").GetBool();
    learning_rate = config_json.at("learning_rate").GetDouble();
    weight_decay = config_json.at("weight_decay").GetDouble();
    train_batch_size = config_json.at("train_batch_size").GetInt();
    inference_batch_size = config_json.at("inference_batch_size").GetInt();
    inference_threads = config_json.at("inference_threads").GetInt();
    inference_cache = config_json.at("inference_cache").GetInt();
    replay_buffer_size = config_json.at("replay_buffer_size").GetInt();
    replay_buffer_reuse = config_json.at("replay_buffer_reuse").GetInt();
    // A run from before the setting existed trained as upstream does.
    learner_batches = config_json.count("learner_batches")
                          ? config_json.at("learner_batches").GetInt()
                          : 0;
    // A run from before the setting existed trained without it.
    symmetry_augmentation = config_json.count("symmetry_augmentation") &&
                            config_json.at("symmetry_augmentation").GetBool();
    policy_target_pruning = config_json.count("policy_target_pruning") &&
                            config_json.at("policy_target_pruning").GetBool();
    replay_buffer_start_size = config_json.count("replay_buffer_start_size")
                                   ? config_json.at("replay_buffer_start_size").GetInt()
                                   : 0;
    checkpoint_freq = config_json.at("checkpoint_freq").GetInt();
    evaluation_window = config_json.at("evaluation_window").GetInt();
    uct_c = config_json.at("uct_c").GetDouble();
    max_simulations = config_json.at("max_simulations").GetInt();
    policy_alpha = config_json.at("policy_alpha").GetDouble();
    policy_epsilon = config_json.at("policy_epsilon").GetDouble();
    // A run from before the setting existed searched as upstream does.
    untried_move_value = config_json.count("untried_move_value")
                             ? config_json.at("untried_move_value").GetString()
                             : "upstream";
    untried_move_reduction =
        config_json.count("untried_move_reduction")
            ? config_json.at("untried_move_reduction").GetDouble()
            : 0.2;
    temperature = config_json.at("temperature").GetDouble();
    temperature_drop = config_json.at("temperature_drop").GetDouble();
    cutoff_probability = config_json.at("cutoff_probability").GetDouble();
    cutoff_value = config_json.at("cutoff_value").GetDouble();
    actors = config_json.at("actors").GetInt();
    evaluators = config_json.at("evaluators").GetInt();
    eval_levels = config_json.at("eval_levels").GetInt();
    max_steps = config_json.at("max_steps").GetInt();
  }
};

bool AlphaZero(AlphaZeroConfig config, StopToken* stop, bool resuming);

// Our change (thud/PLAN.md, runs G and G'): the size of a growing buffer — KataGo's
// rule (arXiv 1902.10565, section 3 and appendix C; KataGo calls it a "growing moving
// window") with the start size c = `start`: all positions while at most c have been
// generated, then c (1 + 0.4 ((N / c)^0.75 - 1) / 0.75) for N generated. Our small
// runs learnt the dwarfs' policy early only with a small buffer (run C) and kept
// improving later only with a large one (runs E, G').
inline int64_t GrowingBufferSize(int64_t generated, int64_t start) {
  if (generated <= start) return generated;
  const double alpha = 0.75, beta = 0.4;
  return static_cast<int64_t>(
      start * (1 + beta * (std::pow(static_cast<double>(generated) / start, alpha) - 1) /
                       alpha));
}

// Our change: `num` distinct elements drawn uniformly from the newest `newest` that
// `buffer` holds (all it holds if fewer) — as CircularBuffer::Sample over all.
template <typename T>
std::vector<T> SampleNewest(const open_spiel::CircularBuffer<T>& buffer,
                            std::mt19937* rng, int64_t newest, int num) {
  const std::vector<T>& data = buffer.Data();
  const int64_t size = data.size(), added = buffer.TotalAdded();
  newest = std::min(newest, size);
  // Floyd's algorithm: num distinct offsets in [0, newest), each set equally likely.
  std::unordered_set<int64_t> offsets;
  for (int64_t j = newest - std::min<int64_t>(num, newest); j < newest; ++j) {
    const int64_t t = std::uniform_int_distribution<int64_t>(0, j)(*rng);
    offsets.insert(offsets.count(t) ? j : t);
  }
  std::vector<T> out;
  out.reserve(offsets.size());
  // The newest is the element added last, at index (added - 1) % size once full.
  for (int64_t offset : offsets) out.push_back(data[(added - 1 - offset) % size]);
  return out;
}

// Our change (thud/PLAN.md, Instrumentation): how a finished game ended, as the
// trainer logs it — "cutoff" if the trainer cut it off by its value, for Thud one
// of "turn_limit", "no_capture_limit", "dwarfs_gone", "trolls_gone",
// "no_legal_move" (THUD_RULES.md section 6), for other games "terminal".
std::string GameEnding(const open_spiel::State& state, bool cut_off);

}  // namespace torch_az
}  // namespace thud_az
}  // namespace open_spiel

#endif  // THUD_AZ_ALPHA_ZERO_H_
