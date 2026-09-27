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

// How much of its search tree could OpenSpiel's AlphaZero carry from one move to the
// next? OpenSpiel builds a fresh tree for every move (mcts.cc:356); this measures what
// reusing it would inherit, with upstream's pieces unmodified.
//
// Plays self-play games as the trainer's actors do (alpha_zero.cc, InitAZBot and
// PlayGame: MCTSBot with PUCT, root noise, moves sampled from the visit counts for the
// first `temperature_drop` moves of a game, the most visited move after that). For every
// search it records the share of the root's visits that went into the move played —
// what the next search, by the other side, would start with if both sides shared one
// tree — and the share that went into the move played and then the opponent's actual
// reply — what the same side would start with two plies later from its own tree.
//
// Each game can start after a random number (0 to `prefix`) of random moves, so that all
// phases of a game appear within minutes; temperature applies only to the first
// `temperature_drop` moves of the game, as in real self-play. Beware: positions after
// random moves are unlike self-play's, and the network's values there differ; with a
// prefix of up to 200 moves (2026-09-26) the dwarfs' searches concentrated their visits
// (10-40 of 100 on the move played), while the trainer's own searches spread them one
// visit per move (az_buffer_stats). So the default is no prefix, and for self-play the
// trainer's saved buffers are the better source (az_buffer_stats: the most visited
// move's share).
//
//   thud/experiments/build_az_program.sh az_reuse
//   OMP_NUM_THREADS=4 build-shared/az_reuse game=thud width=64 depth=4 sims=100 \
//     seconds=480 [model_dir=DIR]      # DIR holds vpnet.pb and checkpoint--1.pt
//
// Prints one JSON line per recorded move, then a summary line.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace {

using open_spiel::Action;
using open_spiel::algorithms::SearchNode;
using open_spiel::algorithms::torch_az::DeviceManager;
using open_spiel::algorithms::torch_az::VPNetEvaluator;
using open_spiel::algorithms::torch_az::VPNetModel;
using Clock = std::chrono::steady_clock;

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      const size_t eq = arg.find('=');
      if (eq == std::string::npos) {
        open_spiel::SpielFatalError(absl::StrCat("Expected key=value, got ", arg));
      }
      values_[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
  }
  std::string Get(const std::string& key, const std::string& fallback) {
    used_.push_back(key);
    return values_.count(key) ? values_[key] : fallback;
  }
  double GetDouble(const std::string& key, double fallback) {
    return std::stod(Get(key, std::to_string(fallback)));
  }
  int GetInt(const std::string& key, int fallback) {
    return std::stoi(Get(key, std::to_string(fallback)));
  }
  void CheckAllUsed() const {
    for (const auto& [key, value] : values_) {
      if (std::find(used_.begin(), used_.end(), key) == used_.end()) {
        open_spiel::SpielFatalError(absl::StrCat("Unknown argument ", key));
      }
    }
  }

 private:
  std::map<std::string, std::string> values_;
  std::vector<std::string> used_;
};

// A trained network from a trainer's directory, or a fresh one of the given size.
VPNetModel LoadModel(const open_spiel::Game& game, const std::string& model_dir,
                     int width, int depth) {
  if (!model_dir.empty()) {
    VPNetModel model(game, model_dir, "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(VPNetModel::kMostRecentCheckpointStep);
    return model;
  }
  const std::string dir = absl::StrCat(std::filesystem::temp_directory_path().string(),
                                       "/az_reuse_", getpid());
  std::filesystem::create_directories(dir);
  SPIEL_CHECK_TRUE(open_spiel::algorithms::torch_az::CreateGraphDef(
      game, /*learning_rate=*/1e-3, /*weight_decay=*/1e-4, dir, "vpnet.pb", "resnet",
      width, depth));
  torch::manual_seed(0);
  VPNetModel model(game, dir, "vpnet.pb", "/cpu:0");
  std::filesystem::remove_all(dir);
  return model;
}

const SearchNode* Child(const SearchNode& node, Action action) {
  for (const SearchNode& child : node.children) {
    if (child.action == action) return &child;
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(args.Get("game", "thud"));
  const int width = args.GetInt("width", 64), depth = args.GetInt("depth", 4);
  const std::string model_dir = args.Get("model_dir", "");
  const int sims = args.GetInt("sims", 100);
  const int searchers = args.GetInt("searchers", 32);
  const int batch = args.GetInt("batch", 32);
  const int inference_threads = args.GetInt("inference_threads", 2);
  const int seconds = args.GetInt("seconds", 480);
  const int prefix = args.GetInt("prefix", 0);
  // The trainer's settings: alpha_zero_torch_example defaults and our flagfile.
  const double uct_c = args.GetDouble("uct_c", 2);
  const double alpha = args.GetDouble("policy_alpha", 0.1);
  const double epsilon = args.GetDouble("policy_epsilon", 0.25);
  const int temperature_drop = args.GetInt("temperature_drop", 10);
  const int memory_mb = args.GetInt("max_memory_mb", 10);  // As InitAZBot.
  args.CheckAllUsed();

  DeviceManager devices;
  devices.AddDevice(LoadModel(*game, model_dir, width, depth));
  auto evaluator = std::make_shared<VPNetEvaluator>(&devices, batch, inference_threads,
                                                    /*cache_size=*/1 << 18,
                                                    /*cache_shards=*/1);
  std::mutex out;
  const auto start = Clock::now();
  std::vector<std::thread> threads;
  for (int t = 0; t < searchers; ++t) {
    threads.emplace_back([&, t]() {
      open_spiel::algorithms::MCTSBot bot(
          *game, evaluator, uct_c, sims, memory_mb, /*solve=*/false, /*seed=*/t,
          /*verbose=*/false, open_spiel::algorithms::ChildSelectionPolicy::PUCT, alpha,
          epsilon, /*dont_return_chance_node=*/true);
      std::mt19937 rng(1000 + t);
      for (int game_num = 0;; ++game_num) {
        std::unique_ptr<open_spiel::State> state = game->NewInitialState();
        const int random_moves = std::uniform_int_distribution<int>(0, prefix)(rng);
        for (int i = 0; i < random_moves && !state->IsTerminal(); ++i) {
          std::vector<Action> legal = state->LegalActions();
          state->ApplyAction(
              legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
        }
        // Each side's last root and the move it played, to measure two plies on.
        std::unique_ptr<SearchNode> last_root[2];
        Action last_played[2] = {open_spiel::kInvalidAction, open_spiel::kInvalidAction};
        std::string pending[2];  // A record waiting for the opponent's reply.
        for (int move = random_moves; !state->IsTerminal(); ++move) {
          const int player = state->CurrentPlayer();
          std::unique_ptr<SearchNode> root = bot.MCTSearch(*state);
          // As PlayGame: sample from the visit counts early, then the most visited.
          Action action;
          if (move < temperature_drop) {
            open_spiel::ActionsAndProbs policy;
            for (const SearchNode& c : root->children) {
              policy.push_back({c.action, static_cast<double>(c.explore_count)});
            }
            open_spiel::NormalizePolicy(&policy);
            action = open_spiel::SampleAction(policy, rng).first;
          } else {
            action = root->BestChild().action;
          }
          const SearchNode* played = Child(*root, action);
          const double shared_share =
              static_cast<double>(played->explore_count) / root->explore_count;
          const double now = std::chrono::duration<double>(Clock::now() - start).count();
          // This side's previous search is complete now that the opponent has replied:
          // its share of visits under its move and then that reply.
          std::string line;
          if (!pending[player].empty() && last_root[player]) {
            const SearchNode* mine = Child(*last_root[player], last_played[player]);
            const SearchNode* reply = mine ? Child(*mine, state->History().back()) : nullptr;
            const double own_share =
                reply ? static_cast<double>(reply->explore_count) /
                            last_root[player]->explore_count
                      : 0.0;
            line = absl::StrCat(pending[player],
                                absl::StrFormat(", \"own_tree_share\": %.4f}", own_share));
          }
          pending[player] = absl::StrFormat(
              "{\"game\": \"%d-%d\", \"move\": %d, \"side\": \"%s\", \"legal\": %d, "
              "\"root_visits\": %d, \"played_visits\": %d, \"shared_tree_share\": %.4f, "
              "\"sampled\": %s",
              t, game_num, move, player == 0 ? "dwarfs" : "trolls",
              static_cast<int>(root->children.size()), root->explore_count,
              played->explore_count, shared_share,
              move < temperature_drop ? "true" : "false");
          last_root[player] = std::move(root);
          last_played[player] = action;
          if (!line.empty()) {
            std::lock_guard<std::mutex> lock(out);
            std::cout << line << "\n";
          }
          state->ApplyAction(action);
          if (now > seconds) break;
        }
        if (std::chrono::duration<double>(Clock::now() - start).count() > seconds) break;
      }
    });
  }
  for (std::thread& t : threads) t.join();
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"sims\": %d, \"width\": %d, \"depth\": %d, "
                   "\"model_dir\": \"%s\", \"cache_hit_rate\": %.3f, \"mean_batch\": %.2f}",
                   sims, width, depth, model_dir, evaluator->CacheInfo().HitRate(),
                   evaluator->BatchSizeStats().Avg())
            << std::endl;
  return 0;
}
