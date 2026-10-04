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
// open_spiel/algorithms/alpha_zero_torch/alpha_zero.cc at upstream commit 540bba6e (2026-06-18)
// by thud/az/import_from_upstream.py, which renames the namespace
// open_spiel::algorithms to open_spiel::thud_az, points the includes at
// the copies and renames the header guard. Later changes: git history
// and thud/PLAN.md, Phase 6.

#include "thud/az/alpha_zero.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/random/uniform_real_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/thud/thud.h"
#include "thud/az/device_manager.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"
#include "thud/az/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/serializable_circular_buffer.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"

namespace open_spiel {
namespace thud_az {
namespace torch_az {

struct StartInfo {
  absl::Time start_time;
  int start_step;
  int model_checkpoint_step;
  int64_t total_trajectories;
};

StartInfo StartInfoFromLearnerJson(const std::string& path) {
  StartInfo start_info;
  file::File learner_file(path + "/learner.jsonl", "r");
  std::vector<std::string> learner_lines =
      absl::StrSplit(learner_file.ReadContents(), '\n');
  std::string last_learner_line;

  // Get the last non-empty line in learner.jsonl.
  for (int i = learner_lines.size() - 1; i >= 0; i--) {
    if (!learner_lines[i].empty()) {
      last_learner_line = learner_lines[i];
      break;
    }
  }

  json::Object last_learner_json = json::FromString(
      last_learner_line).value().GetObject();

  start_info.start_time = absl::Now() - absl::Seconds(
      last_learner_json["time_rel"].GetDouble());
  start_info.start_step = last_learner_json["step"].GetInt() + 1;
  start_info.model_checkpoint_step = VPNetModel::kMostRecentCheckpointStep;
  start_info.total_trajectories =
      last_learner_json["total_trajectories"].GetInt();

  return start_info;
}

struct Trajectory {
  struct State {
    std::vector<float> observation;
    open_spiel::Player current_player;
    std::vector<open_spiel::Action> legal_actions;
    open_spiel::Action action;
    open_spiel::ActionsAndProbs policy;
    double value;
    // Our change (thud/PLAN.md, Instrumentation): how the search moved the
    // network's own prior at the root, if the actor measured it (see PlayGame).
    double prior_entropy = std::numeric_limits<double>::quiet_NaN();
    double prior_kl = std::numeric_limits<double>::quiet_NaN();
    int prior_top_agrees = -1;  // 1 or 0; -1 if not measured.
  };

  std::vector<State> states;
  std::vector<double> returns;
  std::string ending;  // Our change: how the game ended (GameEnding).
};

// Our change (thud/PLAN.md, Instrumentation): how a finished game ended. For
// Thud the five ways of THUD_RULES.md section 6, checked in ThudState's order.
std::string GameEnding(const open_spiel::State& state, bool cut_off) {
  if (cut_off) return "cutoff";
  const auto* thud_state = dynamic_cast<const thud::ThudState*>(&state);
  const auto* thud_game = dynamic_cast<const thud::ThudGame*>(state.GetGame().get());
  if (thud_state == nullptr || thud_game == nullptr) return "terminal";
  if (thud_state->TurnsPlayed() >= thud_game->max_turns()) return "turn_limit";
  if (thud_state->TurnsWithoutCapture() >= thud_game->max_turns_without_capture()) {
    return "no_capture_limit";
  }
  int dwarfs = 0, trolls = 0;
  for (int row = 0; row < thud::kBoardSize; ++row) {
    for (int col = 0; col < thud::kBoardSize; ++col) {
      if (!thud::IsOnBoard(row, col)) continue;
      dwarfs += thud_state->CellAt(row, col) == thud::Cell::kDwarf;
      trolls += thud_state->CellAt(row, col) == thud::Cell::kTroll;
    }
  }
  if (dwarfs == 0) return "dwarfs_gone";
  if (trolls == 0) return "trolls_gone";
  return "no_legal_move";
}

Trajectory PlayGame(Logger* logger, int game_num, const open_spiel::Game& game,
                    std::vector<std::unique_ptr<MCTSBot>>* bots,
                    std::mt19937* rng, double temperature, int temperature_drop,
                    double cutoff_value, bool verbose = false,
                    VPNetEvaluator* prior_source = nullptr,
                    double pruning_k = 0, double uct_c = 0) {
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;
  Trajectory trajectory;

  while (true) {
    if (state->IsChanceNode()) {
      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      open_spiel::Action action =
          open_spiel::SampleAction(outcomes, *rng).first;
      history.push_back(state->ActionToString(state->CurrentPlayer(), action));
      state->ApplyAction(action);
    } else {
      open_spiel::Player player = state->CurrentPlayer();
      std::unique_ptr<SearchNode> root = (*bots)[player]->MCTSearch(*state);
      open_spiel::ActionsAndProbs policy;
      policy.reserve(root->children.size());
      for (const SearchNode& c : root->children) {
        policy.emplace_back(c.action,
                            std::pow(c.explore_count, 1.0 / temperature));
      }
      NormalizePolicy(&policy);
      open_spiel::Action action;
      if (history.size() >= temperature_drop) {
        action = root->BestChild().action;
      } else {
        action = open_spiel::SampleAction(policy, *rng).first;
      }
      // Our change (thud/PLAN.md, policy target pruning): the policy target is
      // the visit counts with KataGo's pruning, if on; the move above was still
      // chosen from the unpruned counts, as before.
      open_spiel::ActionsAndProbs target;
      if (pruning_k > 0) {
        for (const auto& [a, n] : PrunedRootVisits(*root, uct_c, pruning_k)) {
          target.emplace_back(a, std::pow(n, 1.0 / temperature));
        }
        NormalizePolicy(&target);
      }

      double root_value = root->total_reward / root->explore_count;
      trajectory.states.push_back(Trajectory::State{
          state->ObservationTensor(), player, state->LegalActions(), action,
          pruning_k > 0 ? std::move(target) : std::move(policy), root_value});
      // Our change (thud/PLAN.md, Instrumentation): the network's own prior
      // (before root noise: the evaluator's, cached since the search's first
      // evaluation), its entropy, the KL divergence of the root's visit counts
      // from it, and whether its likeliest move is the most visited.
      if (prior_source != nullptr && root->explore_count > 0) {
        std::unordered_map<open_spiel::Action, double> prior;
        double entropy = 0;
        open_spiel::Action prior_top = open_spiel::kInvalidAction;
        for (const auto& [a, p] : prior_source->Prior(*state)) {
          prior[a] = p;
          if (p > 0) entropy -= p * std::log(p);
          if (prior_top == open_spiel::kInvalidAction || p > prior[prior_top]) {
            prior_top = a;
          }
        }
        double kl = 0;
        for (const SearchNode& c : root->children) {
          if (c.explore_count == 0) continue;
          const double v = static_cast<double>(c.explore_count) / root->explore_count;
          kl += v * std::log(v / std::max(prior[c.action], 1e-30));
        }
        Trajectory::State& recorded = trajectory.states.back();
        recorded.prior_entropy = entropy;
        recorded.prior_kl = kl;
        recorded.prior_top_agrees = prior_top == root->BestChild().action;
      }
      std::string action_str = state->ActionToString(player, action);
      history.push_back(action_str);
      state->ApplyAction(action);
      if (verbose) {
        logger->Print("Player: %d, action: %s", player, action_str);
      }
      if (state->IsTerminal()) {
        trajectory.returns = state->Returns();
        trajectory.ending = GameEnding(*state, /*cut_off=*/false);
        break;
      } else if (std::abs(root_value) > cutoff_value) {
        trajectory.returns.resize(2);
        trajectory.returns[player] = root_value;
        trajectory.returns[1 - player] = -root_value;
        trajectory.ending = GameEnding(*state, /*cut_off=*/true);
        break;
      }
    }
  }

  logger->Print("Game %d: Returns: %s; Ending: %s; Actions: %s", game_num,
                absl::StrJoin(trajectory.returns, " "), trajectory.ending,
                absl::StrJoin(history, " "));
  return trajectory;
}

// Our change (thud/PLAN.md, policy target pruning): KataGo's k (arXiv 1902.10565,
// section 3.2).
constexpr double kForcedPlayoutsK = 2;

std::unique_ptr<MCTSBot> InitAZBot(const AlphaZeroConfig& config,
                                   const open_spiel::Game& game,
                                   std::shared_ptr<Evaluator> evaluator,
                                   bool evaluation) {
  return std::make_unique<MCTSBot>(
      game, std::move(evaluator), config.uct_c, config.max_simulations,
      /*max_memory_mb=*/10,
      /*solve=*/false,
      /*seed=*/0,
      /*verbose=*/false, ChildSelectionPolicy::PUCT,
      evaluation ? 0 : config.policy_alpha,
      evaluation ? 0 : config.policy_epsilon,
      /*dont_return_chance_node*/ true, /*max_wall_clock_time=*/-1,
      UntriedMoveValueFromString(config.untried_move_value),
      config.untried_move_reduction,
      // Our change: KataGo's forced playouts in self-play, with the pruning.
      !evaluation && config.policy_target_pruning ? kForcedPlayoutsK : 0);
}

// An actor thread runner that generates games and returns trajectories.
void actor(const open_spiel::Game& game, const AlphaZeroConfig& config, int num,
           ThreadedQueue<Trajectory>* trajectory_queue,
           std::shared_ptr<VPNetEvaluator> vp_eval, StopToken* stop) {
  std::unique_ptr<Logger> logger;
  if (num < 20) {  // Limit the number of open files.
    logger.reset(new FileLogger(config.path, absl::StrCat("actor-", num)));
  } else {
    logger.reset(new NoopLogger());
  }
  std::mt19937 rng(absl::ToUnixNanos(absl::Now()));
  absl::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<std::unique_ptr<MCTSBot>> bots;
  bots.reserve(2);
  for (int player = 0; player < 2; player++) {
    bots.push_back(InitAZBot(config, game, vp_eval, false));
  }
  for (int game_num = 1; !stop->StopRequested(); ++game_num) {
    double cutoff =
        (dist(rng) < config.cutoff_probability ? config.cutoff_value
                                               : game.MaxUtility() + 1);
    if (!trajectory_queue->Push(
            PlayGame(logger.get(), game_num, game, &bots, &rng,
                     config.temperature, config.temperature_drop, cutoff,
                     /*verbose=*/false, /*prior_source=*/vp_eval.get(),
                     config.policy_target_pruning ? kForcedPlayoutsK : 0,
                     config.uct_c),
            absl::Seconds(10))) {
      logger->Print("Failed to push a trajectory after 10 seconds.");
    }
  }
  logger->Print("Got a quit.");
}

class EvalResults {
 public:
  explicit EvalResults(int count, int evaluation_window) {
    results_.reserve(count);
    for (int i = 0; i < count; ++i) {
      results_.emplace_back(evaluation_window);
    }
  }

  // How many evals per difficulty.
  int EvalCount() {
    absl::MutexLock lock(m_);
    return eval_num_ / results_.size();
  }

  // Which eval to do next: difficulty, player0.
  std::pair<int, bool> Next() {
    absl::MutexLock lock(m_);
    int next = eval_num_ % (results_.size() * 2);
    eval_num_ += 1;
    return {next / 2, next % 2};
  }

  void Add(int i, double value) {
    absl::MutexLock lock(m_);
    results_[i].Add(value);
  }

  std::vector<double> AvgResults() {
    absl::MutexLock lock(m_);
    std::vector<double> out;
    out.reserve(results_.size());
    for (const auto& result : results_) {
      out.push_back(result.Empty() ? 0
                                   : (absl::c_accumulate(result.Data(), 0.0) /
                                      result.Size()));
    }
    return out;
  }

 private:
  std::vector<CircularBuffer<double>> results_;
  int eval_num_ = 0;
  absl::Mutex m_;
};

// A thread that plays vs standard MCTS.
void evaluator(const open_spiel::Game& game, const AlphaZeroConfig& config,
               int num, EvalResults* results,
               std::shared_ptr<VPNetEvaluator> vp_eval, StopToken* stop) {
  FileLogger logger(config.path, absl::StrCat("evaluator-", num));
  std::mt19937 rng;
  auto rand_evaluator = std::make_shared<RandomRolloutEvaluator>(1, num);

  for (int game_num = 1; !stop->StopRequested(); ++game_num) {
    auto [difficulty, first] = results->Next();
    int az_player = first ? 0 : 1;
    int rand_max_simulations =
        config.max_simulations * std::pow(10, difficulty / 2.0);
    std::vector<std::unique_ptr<MCTSBot>> bots;
    bots.reserve(2);
    bots.push_back(InitAZBot(config, game, vp_eval, true));
    bots.push_back(std::make_unique<MCTSBot>(
        game, rand_evaluator, config.uct_c, rand_max_simulations,
        /*max_memory_mb=*/1000,
        /*solve=*/true,
        /*seed=*/num * 1000 + game_num,
        /*verbose=*/false, ChildSelectionPolicy::UCT,
        /*dirichlet_alpha=*/0,
        /*dirichlet_epsilon=*/0,
        /*dont_return_chance_node=*/true));
    if (az_player == 1) {
      std::swap(bots[0], bots[1]);
    }

    logger.Print("Running MCTS with %d simulations", rand_max_simulations);
    Trajectory trajectory = PlayGame(
        &logger, game_num, game, &bots, &rng, /*temperature=*/1,
        /*temperature_drop=*/0, /*cutoff_value=*/game.MaxUtility() + 1);

    results->Add(difficulty, trajectory.returns[az_player]);
    logger.Print("Game %d: AZ: %5.2f, MCTS: %5.2f, MCTS-sims: %d, length: %d",
                 game_num, trajectory.returns[az_player],
                 trajectory.returns[1 - az_player], rand_max_simulations,
                 trajectory.states.size());
  }
  logger.Print("Got a quit.");
}

// Our change (thud/PLAN.md, Instrumentation): statistics of one learning step's
// new self-play games — the final margins (player 0's return; in Thud the
// dwarfs') and how the games ended; per side to move, the searches' breadth (as
// thud/experiments/az_buffer_stats.cc computes it), how far each search moved
// the network's prior, and the values' sign accuracy and size at 7 points of the
// side's own turns.
class StepStats {
 public:
  explicit StepStats(const open_spiel::Game& game)
      : sides_(game.GetType().short_name == "thud"
                   ? std::array<std::string, 2>{"dwarfs", "trolls"}
                   : std::array<std::string, 2>{"player0", "player1"}),
        endings_(kEndings) {}

  void Reset() {
    returns_.Reset();
    returns_hist_.Reset();
    endings_.Reset();
    for (Side& side : by_side_) side = Side();
  }

  void Add(const Trajectory& trajectory) {
    returns_.Add(trajectory.returns[0]);
    // 21 buckets of 0.1 over [-1, 1], centred on -1.0, -0.9, ..., 1.0.
    returns_hist_.Add(std::clamp<int>(
        static_cast<int>(std::lround((trajectory.returns[0] + 1) * 10)), 0, 20));
    const auto ending = absl::c_find(kEndings, trajectory.ending);
    endings_.Add(ending == kEndings.end() ? kEndings.size() - 1
                                          : ending - kEndings.begin());
    std::array<std::vector<const Trajectory::State*>, 2> turns;
    for (const Trajectory::State& state : trajectory.states) {
      if (state.current_player < 0 || state.current_player > 1) continue;
      Side& side = by_side_[state.current_player];
      turns[state.current_player].push_back(&state);
      double visited = 0, top = 0, entropy = 0;
      for (const auto& [action, p] : state.policy) {
        if (p <= 0) continue;
        visited += 1;
        top = std::max(top, p);
        entropy -= p * std::log(p);
      }
      side.legal.push_back(state.legal_actions.size());
      side.visited.push_back(visited);
      side.top_share.push_back(top);
      side.effective_moves.push_back(std::exp(entropy));
      if (state.prior_top_agrees >= 0) {
        side.prior_effective_moves.push_back(std::exp(state.prior_entropy));
        side.prior_kl.push_back(state.prior_kl);
        side.prior_top_agrees += state.prior_top_agrees;
      }
    }
    for (int p = 0; p < 2; ++p) {
      if (turns[p].empty()) continue;
      for (int stage = 0; stage < kStages; ++stage) {
        const Trajectory::State& s =
            *turns[p][(turns[p].size() - 1) * stage / (kStages - 1)];
        by_side_[p].value_accuracy[stage].Add(
            (s.value >= 0) == (trajectory.returns[p] >= 0));
        by_side_[p].value_prediction[stage].Add(std::abs(s.value));
      }
    }
  }

  json::Object SelfPlayJson() const {
    return json::Object({
        {"player0_return", returns_.ToJson()},
        {"player0_return_hist", returns_hist_.ToJson()},
        {"endings", endings_.ToJson()},
    });
  }

  json::Object BySideJson() const {
    json::Object result;
    for (int p = 0; p < 2; ++p) {
      const Side& side = by_side_[p];
      json::Object o({
          {"positions", static_cast<int>(side.visited.size())},
          {"median_legal", Median(side.legal)},
          {"median_visited", Median(side.visited)},
          {"mean_visited", Mean(side.visited)},
          {"median_top_share", Median(side.top_share)},
          {"median_effective_moves", Median(side.effective_moves)},
          {"value_accuracy",
           json::TransformToArray(side.value_accuracy,
                                  [](auto v) { return v.ToJson(); })},
          {"value_prediction",
           json::TransformToArray(side.value_prediction,
                                  [](auto v) { return v.ToJson(); })},
      });
      if (!side.prior_kl.empty()) {
        o.emplace("median_prior_effective_moves", Median(side.prior_effective_moves));
        o.emplace("mean_kl_visits_prior", Mean(side.prior_kl));
        o.emplace("prior_top_is_most_visited",
                  static_cast<double>(side.prior_top_agrees) / side.prior_kl.size());
      }
      result.emplace(sides_[p], std::move(o));
    }
    return result;
  }

  const std::string& SideName(int p) const { return sides_[p]; }

 private:
  static constexpr int kStages = 7;
  inline static const std::vector<std::string> kEndings = {
      "turn_limit", "no_capture_limit", "dwarfs_gone", "trolls_gone",
      "no_legal_move", "cutoff", "terminal", "other"};
  struct Side {
    std::vector<double> legal, visited, top_share, effective_moves,
        prior_effective_moves, prior_kl;
    int prior_top_agrees = 0;
    std::vector<open_spiel::BasicStats> value_accuracy =
        std::vector<open_spiel::BasicStats>(kStages);
    std::vector<open_spiel::BasicStats> value_prediction =
        std::vector<open_spiel::BasicStats>(kStages);
  };

  static double Median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 == 1) return *middle;
    return (*middle + *std::max_element(values.begin(), middle)) / 2;
  }
  static double Mean(const std::vector<double>& values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0;
    for (double v : values) sum += v;
    return sum / values.size();
  }

  std::array<std::string, 2> sides_;
  open_spiel::BasicStats returns_;
  open_spiel::HistogramNumbered returns_hist_{21};
  open_spiel::HistogramNamed endings_;
  std::array<Side, 2> by_side_;
};

// Our change: a LossInfo's losses per side, as JSON (empty if not measured).
json::Object LossesBySide(const VPNetModel::LossInfo& losses,
                          const StepStats& names) {
  json::Object result;
  for (int p = 0; p < 2; ++p) {
    if (losses.Positions(p) == 0) continue;
    result.emplace(names.SideName(p),
                   json::Object({{"policy", losses.Policy(p)},
                                 {"value", losses.Value(p)},
                                 {"positions", losses.Positions(p)}}));
  }
  return result;
}

void learner(const open_spiel::Game& game, const AlphaZeroConfig& config,
             DeviceManager* device_manager,
             std::shared_ptr<VPNetEvaluator> eval,
             ThreadedQueue<Trajectory>* trajectory_queue,
             EvalResults* eval_results, StopToken* stop,
             const StartInfo& start_info) {
  FileLogger logger(config.path, "learner", "a");
  DataLoggerJsonLines data_logger(
      config.path, "learner", true, "a", start_info.start_time);
  std::mt19937 rng;

  int device_id = 0;  // Do not change, the first device is the learner.
  logger.Print("Running the learner on device %d: %s", device_id,
               device_manager->Get(0, device_id)->Device());

  SerializableCircularBuffer<VPNetModel::TrainInputs> replay_buffer(
      config.replay_buffer_size);
  if (start_info.start_step > 1) {
    replay_buffer.LoadBuffer(config.path + "/replay_buffer.data");
  }
  int learn_rate = config.replay_buffer_size / config.replay_buffer_reuse;
  int64_t total_trajectories = start_info.total_trajectories;

  const int stage_count = 7;
  std::vector<open_spiel::BasicStats> value_accuracies(stage_count);
  std::vector<open_spiel::BasicStats> value_predictions(stage_count);
  open_spiel::BasicStats game_lengths;
  open_spiel::HistogramNumbered game_lengths_hist(game.MaxGameLength() + 1);

  open_spiel::HistogramNamed outcomes({"Player1", "Player2", "Draw"});
  StepStats step_stats(game);  // Added: thud/PLAN.md, Instrumentation.
  // Added, to see overfitting (thud/PLAN.md Phase 6): before each learning step,
  // the loss on a sample of the positions new since the last step against a
  // sample the network has trained on, taken after the last step. Same network,
  // same mode (as it plays). Both are empty in a run's first step. Their own
  // random numbers leave the batches the learner draws as they were.
  constexpr int kLossSample = 2048;
  std::vector<VPNetModel::TrainInputs> new_sample, trained_sample;
  std::mt19937 loss_rng;
  // Actor threads have likely been contributing for a while, so put `last` in
  // the past to avoid a giant spike on the first step.
  absl::Time last = absl::Now() - absl::Seconds(60);
  for (int step = start_info.start_step;
       !stop->StopRequested() &&
           (config.max_steps == 0 || step <= config.max_steps);
       ++step) {
    outcomes.Reset();
    game_lengths.Reset();
    game_lengths_hist.Reset();
    for (auto& value_accuracy : value_accuracies) {
      value_accuracy.Reset();
    }
    for (auto& value_prediction : value_predictions) {
      value_prediction.Reset();
    }
    step_stats.Reset();

    // Collect trajectories
    int queue_size = trajectory_queue->Size();
    int num_states = 0;
    int num_trajectories = 0;
    while (!stop->StopRequested() && num_states < learn_rate) {
      absl::optional<Trajectory> trajectory = trajectory_queue->Pop();
      if (trajectory) {
        num_trajectories += 1;
        total_trajectories += 1;
        game_lengths.Add(trajectory->states.size());
        game_lengths_hist.Add(trajectory->states.size());

        double p1_outcome = trajectory->returns[0];
        outcomes.Add(p1_outcome > 0 ? 0 : (p1_outcome < 0 ? 1 : 2));
        step_stats.Add(*trajectory);

        for (const Trajectory::State& state : trajectory->states) {
          const VPNetModel::TrainInputs inputs{state.legal_actions,
                                               state.observation, state.policy,
                                               p1_outcome};
          replay_buffer.Add(inputs);
          // Added: a uniform sample of them (reservoir sampling).
          if (num_states < kLossSample) {
            new_sample.push_back(inputs);
          } else {
            const int j =
                std::uniform_int_distribution<int>(0, num_states)(loss_rng);
            if (j < kLossSample) new_sample[j] = inputs;
          }
          num_states += 1;
        }

        for (int stage = 0; stage < stage_count; ++stage) {
          // Scale for the length of the game
          int index = (trajectory->states.size() - 1) *
                      static_cast<double>(stage) / (stage_count - 1);
          const Trajectory::State& s = trajectory->states[index];
          value_accuracies[stage].Add(
              (s.value >= 0) == (trajectory->returns[s.current_player] >= 0));
          value_predictions[stage].Add(abs(s.value));
        }
      }
    }
    absl::Time now = absl::Now();
    double seconds = absl::ToDoubleSeconds(now - last);

    logger.Print("Step: %d", step);
    logger.Print(
        "Collected %5d states from %3d games, %.1f states/s; "
        "%.1f states/(s*actor), game length: %.1f",
        num_states, num_trajectories, num_states / seconds,
        num_states / (config.actors * seconds),
        static_cast<double>(num_states) / num_trajectories);
    logger.Print("Queue size: %d. Buffer size: %d. States seen: %d", queue_size,
                 replay_buffer.Size(), replay_buffer.TotalAdded());

    if (stop->StopRequested()) {
      break;
    }

    last = now;

    replay_buffer.SaveBuffer(config.path + "/replay_buffer.data");

    VPNetModel::LossInfo losses, new_loss, trained_loss;
    int64_t growing_buffer_size = 0;  // Added: the positions the learner sampled from.
    const bool compare_losses = !trained_sample.empty();  // Added.
    {  // Extra scope to return the device for use for inference asap.
      DeviceManager::DeviceLoan learn_model =
          device_manager->Get(config.train_batch_size, device_id);

      // Added: the losses on new and on trained positions, before learning.
      auto loss = [&](const std::vector<VPNetModel::TrainInputs>& sample) {
        VPNetModel::LossInfo info;
        for (int i = 0; i < sample.size(); i += config.train_batch_size) {
          info += learn_model->Loss(std::vector<VPNetModel::TrainInputs>(
              sample.begin() + i,
              sample.begin() +
                  std::min<int>(i + config.train_batch_size, sample.size())));
        }
        return info;
      };
      if (compare_losses) {
        new_loss = loss(new_sample);
        trained_loss = loss(trained_sample);
        logger.Print("Before learning, as it plays: new positions' losses: policy "
                     "%.4f, value %.4f; trained positions': policy %.4f, value %.4f",
                     new_loss.Policy(), new_loss.Value(), trained_loss.Policy(),
                     trained_loss.Value());
      }
      new_sample.clear();

      // Let the device manager know that the first device is now
      // off-limits for inference and should only be used for learning
      // (if config.explicit_learning == true).
      device_manager->SetLearning(config.explicit_learning);

      // Learn from them: upstream's one pass over the buffer, or a fixed number of
      // batches (learner_batches, thud/PLAN.md Phase 6, Step 2).
      const int batches = config.learner_batches > 0
                              ? config.learner_batches
                              : replay_buffer.Size() / config.train_batch_size;
      logger.Print("Learning on %d batches of %d%s", batches, config.train_batch_size,
                   config.symmetry_augmentation ? ", each position turned or "
                                                  "mirrored at random" : "");
      // Added: the growing buffer, if on (GrowingBufferSize in alpha_zero.h).
      const int64_t buffer_size =
          config.replay_buffer_start_size > 0
              ? std::min<int64_t>(GrowingBufferSize(replay_buffer.TotalAdded(),
                                                    config.replay_buffer_start_size),
                                  replay_buffer.Size())
              : replay_buffer.Size();
      if (config.replay_buffer_start_size > 0) {
        logger.Print("Growing buffer: sampling the newest %d of %d positions stored",
                     buffer_size, replay_buffer.Size());
      }
      auto sample = [&](std::mt19937* r, int num) {
        return config.replay_buffer_start_size > 0
                   ? SampleNewest(replay_buffer, r, buffer_size, num)
                   : replay_buffer.Sample(r, num);
      };
      std::uniform_int_distribution<int> symmetry(0, thud::kNumSymmetries - 1);
      for (int i = 0; i < batches; i++) {
        std::vector<VPNetModel::TrainInputs> batch =
            sample(&rng, config.train_batch_size);
        if (config.symmetry_augmentation) {  // Added: thud/PLAN.md Phase 6.
          for (VPNetModel::TrainInputs& inputs : batch) {
            inputs = SymmetricTrainInputs(inputs, symmetry(rng));
          }
        }
        losses += learn_model->Learn(batch);
      }
      trained_sample = sample(&loss_rng, std::min<int64_t>(kLossSample, buffer_size));
      growing_buffer_size = buffer_size;

      // The device manager can now once again use the first device for
      // inference (if it could not before).
      device_manager->SetLearning(false);
    }

    // Always save a checkpoint, either for keeping or for loading the weights
    // to the other sessions. It only allows numbers, so use -1 as "latest".
    std::string checkpoint_path = device_manager->Get(0, device_id)
        ->SaveCheckpoint(VPNetModel::kMostRecentCheckpointStep);
    if (step % config.checkpoint_freq == 0) {
      device_manager->Get(0, device_id)->SaveCheckpoint(step);
    }
    if (device_manager->Count() > 0) {
      for (int i = 0; i < device_manager->Count(); ++i) {
        if (i != device_id) {
          device_manager->Get(0, i)->LoadCheckpoint(checkpoint_path);
        }
      }
    }
    logger.Print("Checkpoint saved: %s", checkpoint_path);

    DataLogger::Record record = {
        {"step", step},
        {"total_states", replay_buffer.TotalAdded()},
        {"states_per_s", num_states / seconds},
        {"states_per_s_actor", num_states / (config.actors * seconds)},
        {"total_trajectories", total_trajectories},
        {"trajectories_per_s", num_trajectories / seconds},
        {"queue_size", queue_size},
        {"game_length", game_lengths.ToJson()},
        {"game_length_hist", game_lengths_hist.ToJson()},
        {"outcomes", outcomes.ToJson()},
        {"value_accuracy",
         json::TransformToArray(value_accuracies,
                                [](auto v) { return v.ToJson(); })},
        {"value_prediction",
         json::TransformToArray(value_predictions,
                                [](auto v) { return v.ToJson(); })},
        {"eval", json::Object({
                     {"count", eval_results->EvalCount()},
                     {"results", json::CastToArray(eval_results->AvgResults())},
                 })},
        {"batch_size", eval->BatchSizeStats().ToJson()},
        {"batch_size_hist", eval->BatchSizeHistogram().ToJson()},
        {"loss", json::Object({
                     {"policy", losses.Policy()},
                     {"value", losses.Value()},
                     {"l2reg", losses.L2()},
                     {"sum", losses.Total()},
                 })},
    };
    if (compare_losses) {  // Added: see new_sample above.
      record.emplace("loss_before_learning",
                     json::Object({{"new_policy", new_loss.Policy()},
                                   {"new_value", new_loss.Value()},
                                   {"trained_policy", trained_loss.Policy()},
                                   {"trained_value", trained_loss.Value()},
                                   {"new_by_side", LossesBySide(new_loss, step_stats)},
                                   {"trained_by_side",
                                    LossesBySide(trained_loss, step_stats)}}));
    }
    // Added: thud/PLAN.md, Instrumentation — the losses per side to move, the
    // step's self-play results, and the searches per side.
    record.emplace("loss_by_side", LossesBySide(losses, step_stats));
    if (config.replay_buffer_start_size > 0) {
      record.emplace("growing_buffer_size", static_cast<int>(growing_buffer_size));
    }
    record.emplace("selfplay", step_stats.SelfPlayJson());
    record.emplace("by_side", step_stats.BySideJson());
    eval->ResetBatchSizeStats();
    logger.Print("Losses: policy: %.4f, value: %.4f, l2: %.4f, sum: %.4f",
                 losses.Policy(), losses.Value(), losses.L2(), losses.Total());
    if (losses.Positions(0) > 0 && losses.Positions(1) > 0) {  // Added.
      logger.Print("Losses by side: %s policy %.4f, value %.4f; %s policy %.4f, "
                   "value %.4f", step_stats.SideName(0), losses.Policy(0),
                   losses.Value(0), step_stats.SideName(1), losses.Policy(1),
                   losses.Value(1));
    }

    LRUCacheInfo cache_info = eval->CacheInfo();
    if (cache_info.size > 0) {
      logger.Print(absl::StrFormat(
          "Cache size: %d/%d: %.1f%%, hits: %d, misses: %d, hit rate: %.3f%%",
          cache_info.size, cache_info.max_size, 100.0 * cache_info.Usage(),
          cache_info.hits, cache_info.misses, 100.0 * cache_info.HitRate()));
      eval->ClearCache();
    }
    const VPNetEvaluator::RequestCounts requests = eval->GetRequestCounts();
    eval->ResetRequestCounts();
    const auto share = [](int64_t hits, int64_t misses) {
      return hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0;
    };
    logger.Print(absl::StrFormat(
        "Requests: values %d (cache hits %.1f%%), move probabilities %d (cache hits "
        "%.1f%%)",
        requests.value_hits + requests.value_misses,
        share(requests.value_hits, requests.value_misses),
        requests.prior_hits + requests.prior_misses,
        share(requests.prior_hits, requests.prior_misses)));
    record.emplace("requests",
                   json::Object({
                       {"value_hits", requests.value_hits},
                       {"value_misses", requests.value_misses},
                       {"prior_hits", requests.prior_hits},
                       {"prior_misses", requests.prior_misses},
                   }));
    record.emplace("cache",
                   json::Object({
                       {"size", cache_info.size},
                       {"max_size", cache_info.max_size},
                       {"usage", cache_info.Usage()},
                       {"requests", cache_info.Total()},
                       {"requests_per_s", cache_info.Total() / seconds},
                       {"hits", cache_info.hits},
                       {"misses", cache_info.misses},
                       {"misses_per_s", cache_info.misses / seconds},
                       {"hit_rate", cache_info.HitRate()},
                   }));

    data_logger.Write(record);
    logger.Print("");
  }
}

bool AlphaZero(AlphaZeroConfig config, StopToken* stop, bool resuming) {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(config.game);

  open_spiel::GameType game_type = game->GetType();
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");
  if (config.symmetry_augmentation && game_type.short_name != "thud")
    open_spiel::SpielFatalError("symmetry_augmentation knows only Thud's symmetries.");

  file::Mkdirs(config.path);
  if (!file::IsDirectory(config.path)) {
    std::cerr << config.path << " is not a directory." << std::endl;
    return false;
  }

  std::cout << "Logging directory: " << config.path << std::endl;

  if (config.graph_def.empty()) {
    config.graph_def = "vpnet.pb";
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Overwriting existing model: " << model_path << std::endl;
    } else {
      std::cout << "Creating model: " << model_path << std::endl;
    }
    SPIEL_CHECK_TRUE(CreateGraphDef(
        *game, config.learning_rate, config.weight_decay, config.path,
        config.graph_def, config.nn_model, config.nn_width, config.nn_depth));
  } else {
    std::string model_path = absl::StrCat(config.path, "/", config.graph_def);
    if (file::Exists(model_path)) {
      std::cout << "Using existing model: " << model_path << std::endl;
    } else {
      std::cout << "Model not found: " << model_path << std::endl;
    }
  }

  std::cout << "Playing game: " << config.game << std::endl;

  config.inference_batch_size = std::max(
      1,
      std::min(config.inference_batch_size, config.actors + config.evaluators));

  config.inference_threads =
      std::max(1, std::min(config.inference_threads,
                           (1 + config.actors + config.evaluators) / 2));

  {
    file::File fd(config.path + "/config.json", "w");
    fd.Write(json::ToString(config.ToJson(), true) + "\n");
  }

  StartInfo start_info = {/*start_time=*/absl::Now(),
                          /*start_step=*/1,
                          /*model_checkpoint_step=*/0,
                          /*total_trajectories=*/0};
  if (resuming) {
    start_info = StartInfoFromLearnerJson(config.path);
  }

  DeviceManager device_manager;
  for (const absl::string_view& device : absl::StrSplit(config.devices, ',')) {
    device_manager.AddDevice(
        VPNetModel(*game, config.path, config.graph_def, std::string(device)));
  }

  if (device_manager.Count() == 0) {
    std::cerr << "No devices specified?" << std::endl;
    return false;
  }

  // The explicit_learning option should only be used when multiple
  // devices are available (so that inference can continue while
  // also undergoing learning).
  if (device_manager.Count() <= 1 && config.explicit_learning) {
    std::cerr << "Explicit learning can only be used with multiple devices."
              << std::endl;
    return false;
  }

  std::cerr << "Loading model from step " << start_info.model_checkpoint_step
            << std::endl;
  {  // Make sure they're all in sync.
    if (!resuming) {
      device_manager.Get(0)->SaveCheckpoint(start_info.model_checkpoint_step);
    }
    for (int i = 0; i < device_manager.Count(); ++i) {
      device_manager.Get(0, i)->LoadCheckpoint(
          start_info.model_checkpoint_step);
    }
  }

  auto eval = std::make_shared<VPNetEvaluator>(
      &device_manager, config.inference_batch_size, config.inference_threads,
      config.inference_cache, (config.actors + config.evaluators) / 16);

  ThreadedQueue<Trajectory> trajectory_queue(config.replay_buffer_size /
                                             config.replay_buffer_reuse);

  EvalResults eval_results(config.eval_levels, config.evaluation_window);

  std::vector<Thread> actors;
  actors.reserve(config.actors);
  for (int i = 0; i < config.actors; ++i) {
    actors.emplace_back(
        [&, i]() { actor(*game, config, i, &trajectory_queue, eval, stop); });
  }
  std::vector<Thread> evaluators;
  evaluators.reserve(config.evaluators);
  for (int i = 0; i < config.evaluators; ++i) {
    evaluators.emplace_back(
        [&, i]() { evaluator(*game, config, i, &eval_results, eval, stop); });
  }
  learner(*game, config, &device_manager, eval, &trajectory_queue,
          &eval_results, stop, start_info);

  if (!stop->StopRequested()) {
    stop->Stop();
  }

  // Empty the queue so that the actors can exit.
  trajectory_queue.BlockNewValues();
  trajectory_queue.Clear();

  std::cout << "Joining all the threads." << std::endl;
  for (auto& t : actors) {
    t.join();
  }
  for (auto& t : evaluators) {
    t.join();
  }
  std::cout << "Exiting cleanly." << std::endl;
  return true;
}

}  // namespace torch_az
}  // namespace thud_az
}  // namespace open_spiel
