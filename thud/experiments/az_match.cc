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

// Head-to-head matches between two OpenSpiel C++ AlphaZero networks, in Thud's own match
// format: pairs of battles with the sides swapped, won on the summed margin.
//
// Each pair starts both battles from the same opening, a few uniformly random moves from
// the initial position (so pairs differ); swapping sides cancels whatever advantage an
// opening gives one side. Both networks search with our copy of OpenSpiel's MCTSBot
// (PUCT; thud/az/), the same number of simulations (`sims`; `sims_a` and `sims_b` give
// each network its own, e.g. to measure what more simulations gain on each side) and the
// same rule for untried moves (`untried`, default upstream's — with it our copy searches
// exactly as upstream's does, thud/az/identity_test.cc; `untried_a` and `untried_b` give
// each network its own, e.g. the rule it was trained with), no root noise, and always play
// their most visited move. `temperature_a` and `temperature_b` give each network the policy
// softmax temperatures per player it was trained with (comma-separated, as the trainer's
// --player_policy_temperature, e.g. 1.5,1; empty: none, upstream's search).
// Each network has its own batched evaluator. Prints one JSON line per pair, then a summary:
// network A's summed margin per pair (in points), its mean with a 95% interval (Student's
// t), pairs won, drawn and lost, and each network's mean margin per side. With `progress`,
// it also reports the simulations per second to stderr every `progress` seconds — a
// throughput measure that needs no finished pairs — with the network requests by kind
// (values, move probabilities) and how many the cache answered. `cache` sets each
// evaluator's cache size (0: none); `share=1` gives a network playing itself one
// evaluator for both sides, as in self-play, so that each side's search can hit the
// positions the other side's evaluated.
//
// Side B can instead be a player without a network (2026-10-07):
//   b=mcts:SIMS           OpenSpiel's MCTSBot with random rollouts, as the trainer's
//                         evaluator plays it; `mcts_memory` caps each tree, in MB
//   b=alphabeta:DEPTH     OpenSpiel's alpha-beta search to DEPTH plies, scoring by
//                         material (AlphaBetaOpponent below)
//   b=hexparrot:LOOKAHEAD hexparrot/thudgame's AI (its default lookahead is 3), in a
//                         Python process per thread (thud/experiments/hexparrot_player.py;
//                         `hexparrot_player` gives another path to it)
// Each starts every battle from a seed fixed by `seed`, the pair and the side, so reruns
// play the same games whichever thread plays them. The pair lines then also give the
// pair's seconds and each side's seconds per move it chose (wall clock: on a loaded
// machine both sides' grow; network A's include waiting for its batches), the summary
// those over all battles and how often hexparrot's AI chose no move and a uniformly
// random one was played for it (`b_fallbacks`). Matches between networks are unchanged.
// `games=PATH` writes every battle's moves to PATH, one JSON line each, in any match.
// `capture_floor_a=LIST` (a diagnostic, 2026-10-08; thud/PLAN.md, the blunder
// diagnosis): per player, the least prior mass network A's search gives the captures
// (moves written with an "x") at that player's nodes ("0,0.5": at the trolls' nodes
// half the prior) — MCTSBot::SetPriorFloors.
// Self-play's conditions (2026-10-13; thud/DESIGN_dwarf_signal.md, measurement 3), off by
// default: `noise=LIST` gives the searches at each player's roots the trainer's root noise
// (per player 0 or 1, "1,0": at the dwarfs' roots, in both networks' searches; Dirichlet
// `noise_alpha`, default 0.1, mixed in at `noise_epsilon`, default 0.25), and
// `sample_plies=N` chooses each battle's first N plies, the opening's included, by
// sampling from the root's visit counts, as the trainer does before its temperature drop
// (temperature 1). With either, every battle gets fresh searches seeded from `seed`, the
// pair and the side, so reruns play the same battles whichever thread plays them (the
// searches' random tie-breaks and noise included); without them nothing changes.
// `teacher_mix_a=LIST` (measurement 2): per player, network A's prior at that player's nodes
// mixed with kickstarting's teacher ("0.5,0": at the dwarfs' nodes, half the teacher) —
// (1 - w) x the network's + w x the softmax of AlphaBetaPlayer::Scores at `teacher_depth`
// plies (default 2) over `teacher_temperature` points (default 0.25), the trainer's
// TeacherPolicy; in all of A's searches, as a network trained toward the teacher would
// have it.
//
//   thud/az/build.sh thud/experiments/az_match.cc
//   OMP_NUM_THREADS=4 build-shared/az_match a=RUN_DIR:STEP b=RUN_DIR:STEP sims=100 \
//     pairs=50 [sims_a=SIMS sims_b=SIMS opening=4 threads=16 batch=16 seed=1 \
//     untried=upstream untried_a=UNTRIED untried_b=UNTRIED temperature_a=LIST \
//     temperature_b=LIST progress=0 cache=262144 share=0 mcts_memory=1000 \
//     hexparrot_player=PATH games=PATH capture_floor_a=LIST noise=LIST noise_alpha=0.1 \
//     noise_epsilon=0.25 sample_plies=0 teacher_mix_a=LIST teacher_depth=2 \
//     teacher_temperature=0.25 devices=/cpu:0 precision_a=float32 precision_b=float32]
// `precision_a`, `precision_b` (2026-10-19; thud/PLAN.md, mixed precision): each network's
// inference — float32, mixed (the trunk in bfloat16, the heads in float32) or bfloat16
// (VPNetModel::SetInferencePrecision); with share=1 network A's for both sides.
// `devices=LIST` (2026-10-15): where each network runs, comma-separated, one copy of it on
// each (as the trainer's --devices: cpu, cuda:0, cuda:1, ...; cuda:0,cuda:0 puts two
// copies on one GPU). Default /cpu:0, as before.
//
// STEP is a checkpoint step of that run (-1: its most recent checkpoint).

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_replace.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"
#include "thud/experiments/teacher_mix.h"

namespace {

using open_spiel::Action;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::TeacherMixEvaluator;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;

constexpr double kMaxMargin = 32;  // Thud's returns are the margin over 32.

std::atomic<int64_t> simulations_done{0};  // For the progress reports.

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

// A network from a trainer's directory, "DIR:STEP" (STEP -1: the most recent), one copy
// of it on each of `device_names` (`devices=`: e.g. cuda:0, or cuda:0,cuda:0 for two copies
// on one GPU; the evaluator spreads its batches over them).
std::shared_ptr<VPNetEvaluator> LoadEvaluator(const open_spiel::Game& game,
                                              const std::string& spec,
                                              DeviceManager* devices, int batch,
                                              int inference_threads, int cache_size,
                                              const std::vector<std::string>& device_names,
                                              const std::string& precision) {
  const size_t colon = spec.rfind(':');
  if (colon == std::string::npos) {
    open_spiel::SpielFatalError(absl::StrCat("Expected DIR:STEP, got ", spec));
  }
  for (const std::string& device : device_names) {
    VPNetModel model(game, spec.substr(0, colon), "vpnet.pb", device);
    model.LoadCheckpoint(std::stoi(spec.substr(colon + 1)));
    model.SetInferencePrecision(precision);  // 2026-10-19: precision_a, precision_b.
    devices->AddDevice(std::move(model));
  }
  return std::make_shared<VPNetEvaluator>(devices, batch, inference_threads,
                                          cache_size, /*cache_shards=*/1);
}

struct Battle {
  double a_return;  // Network A's return, -1 to 1.
  int moves;
  std::vector<std::string> actions;  // The battle's moves, opening included, as text.
  double seconds[2] = {0, 0};  // A's and B's time choosing moves (wall clock).
  int choices[2] = {0, 0};     // The moves each chose, the opening's not counted.
};

// A player's move in a state.
using Chooser = std::function<Action(const open_spiel::State&)>;

// A network's move: its most visited.
Chooser NetworkChooser(MCTSBot* bot) {
  return [bot](const open_spiel::State& state) {
    std::unique_ptr<SearchNode> root = bot->MCTSearch(state);
    simulations_done += root->explore_count;
    return root->BestChild().action;
  };
}

// A network's move in self-play's conditions (`noise`, `sample_plies`): searched by
// bots[the player to move], and in the first `sample_plies` plies sampled from the root's
// visit counts (the trainer's temperature 1), else its most visited.
Chooser SelfPlayChooser(std::array<MCTSBot*, 2> bots, int sample_plies,
                        std::mt19937* rng) {
  return [bots, sample_plies, rng](const open_spiel::State& state) {
    std::unique_ptr<SearchNode> root = bots[state.CurrentPlayer()]->MCTSearch(state);
    simulations_done += root->explore_count;
    if (state.MoveNumber() >= sample_plies) return root->BestChild().action;
    open_spiel::ActionsAndProbs visits;
    for (const SearchNode& child : root->children) {
      visits.emplace_back(child.action, child.explore_count);
    }
    double sum = 0;
    for (const auto& [action, n] : visits) sum += n;
    for (auto& [action, n] : visits) n /= sum;
    return open_spiel::SampleAction(visits, *rng).first;
  };
}

// One battle from `opening`, network A playing `a_player`.
Battle Play(const open_spiel::Game& game, const std::vector<Action>& opening,
            const Chooser& a, const Chooser& b, int a_player) {
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> actions;
  for (Action action : opening) {
    actions.push_back(state->ActionToString(action));
    state->ApplyAction(action);
  }
  int moves = opening.size();
  double seconds[2] = {0, 0};
  int choices[2] = {0, 0};
  while (!state->IsTerminal()) {
    const int chooser = state->CurrentPlayer() == a_player ? 0 : 1;
    const auto start = std::chrono::steady_clock::now();
    const Action action = (chooser == 0 ? a : b)(*state);
    seconds[chooser] +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++choices[chooser];
    actions.push_back(state->ActionToString(action));
    state->ApplyAction(action);
    ++moves;
  }
  return {state->Returns()[a_player], moves, std::move(actions), {seconds[0], seconds[1]},
          {choices[0], choices[1]}};
}

// Side B's player when it is not a network (b=KIND:N).
class Opponent {
 public:
  virtual ~Opponent() = default;
  // Called before each battle, with a seed fixed by the pair and the side.
  virtual void NewBattle(int seed) = 0;
  virtual Action Choose(const open_spiel::State& state) = 0;
  // Moves played for it because it chose none.
  virtual int Fallbacks() const { return 0; }
};

int BattleSeed(int seed, int pair, int a_player) {
  return seed * 100003 + 2 * pair + a_player;
}

// OpenSpiel's MCTSBot with random rollouts (mcts:SIMS), set up as the trainer's evaluator
// sets up its opponent (thud/az/alpha_zero.cc, EvaluatorThread): UCT with uct_c 2, one
// rollout per evaluation, the solver on; `max_memory_mb` caps the tree, beyond which it
// prunes it (MCTSBot::MCTSearch).
class MctsOpponent : public Opponent {
 public:
  MctsOpponent(const open_spiel::Game& game, int simulations, int max_memory_mb)
      : game_(game), simulations_(simulations), max_memory_mb_(max_memory_mb) {}
  void NewBattle(int seed) override {
    bot_ = std::make_unique<open_spiel::algorithms::MCTSBot>(
        game_, std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(1, seed),
        /*uct_c=*/2, simulations_, max_memory_mb_, /*solve=*/true, seed,
        /*verbose=*/false, open_spiel::algorithms::ChildSelectionPolicy::UCT,
        /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0,
        /*dont_return_chance_node=*/true);
  }
  Action Choose(const open_spiel::State& state) override { return bot_->Step(state); }

 private:
  const open_spiel::Game& game_;
  const int simulations_;
  const int max_memory_mb_;
  std::unique_ptr<open_spiel::algorithms::MCTSBot> bot_;
};

// OpenSpiel's alpha-beta search (alphabeta:DEPTH, in plies), scoring by material with a
// random tie-break: thud/az/alphabeta_player.h, which the trainer's games against
// alpha-beta use too.
class AlphaBetaOpponent : public Opponent {
 public:
  AlphaBetaOpponent(const open_spiel::Game& game, int depth) : player_(game, depth) {}
  void NewBattle(int seed) override { player_.NewGame(seed); }
  Action Choose(const open_spiel::State& state) override { return player_.Choose(state); }

 private:
  open_spiel::thud_az::AlphaBetaPlayer player_;
};

// hexparrot/thudgame's AI (hexparrot:LOOKAHEAD), run by thud/experiments/hexparrot_player.py
// in a Python process of its own, which this talks to over pipes (the protocol is in that
// file). Positions cross, not moves: this sends the board and plays the one legal action
// that leads to the board hexparrot's move left, or stops the match if none does.
class HexparrotOpponent : public Opponent {
 public:
  HexparrotOpponent(int lookahead, const std::string& script) {
    int to_child[2], from_child[2];
    // O_CLOEXEC: no other thread's Python may inherit these, or it would hold them open
    // after this one closes them.
    SPIEL_CHECK_EQ(pipe2(to_child, O_CLOEXEC), 0);
    SPIEL_CHECK_EQ(pipe2(from_child, O_CLOEXEC), 0);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, to_child[0], 0);
    posix_spawn_file_actions_adddup2(&actions, from_child[1], 1);
    std::vector<std::string> words = {"python3", "-u", script, "--lookahead",
                                      std::to_string(lookahead)};
    std::vector<char*> argv;
    for (std::string& word : words) argv.push_back(word.data());
    argv.push_back(nullptr);
    const int error =
        posix_spawnp(&pid_, "python3", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (error != 0) {
      open_spiel::SpielFatalError(absl::StrCat("Cannot start python3 ", script));
    }
    close(to_child[0]);
    close(from_child[1]);
    to_ = fdopen(to_child[1], "w");
    from_ = fdopen(from_child[0], "r");
  }
  ~HexparrotOpponent() override {
    fclose(to_);  // End of input ends the player.
    fclose(from_);
    waitpid(pid_, nullptr, 0);
  }
  void NewBattle(int seed) override {
    rng_.seed(seed);
    if (Ask(absl::StrCat("seed ", seed)) != "ok") {
      open_spiel::SpielFatalError("hexparrot_player.py did not take its seed");
    }
  }
  Action Choose(const open_spiel::State& state) override {
    const std::string reply = Ask(absl::StrCat(
        "move ", state.CurrentPlayer() == open_spiel::thud::kDwarfPlayer ? "dwarf" : "troll",
        " ", Rows(state)));
    const std::vector<Action> legal = state.LegalActions();
    if (reply == "none") {
      ++fallbacks_;
      return legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng_)];
    }
    for (Action action : legal) {
      if (Rows(*state.Child(action)) == reply) return action;
    }
    open_spiel::SpielFatalError(absl::StrCat(
        "hexparrot's move leads to no position our rules allow, from\n", state.ToString(),
        "\nto\n", absl::StrReplaceAll(reply, {{"/", "\n"}})));
  }
  int Fallbacks() const override { return fallbacks_; }

 private:
  // The board: the 15 rows of thud.h's text format, joined by '/'.
  static std::string Rows(const open_spiel::State& state) {
    std::string text = state.ToString();
    text.resize(text.rfind('\n'));  // Drops the status line.
    std::replace(text.begin(), text.end(), '\n', '/');
    return text;
  }
  std::string Ask(const std::string& request) {
    std::fprintf(to_, "%s\n", request.c_str());
    std::fflush(to_);
    char* line = nullptr;
    size_t capacity = 0;
    const ssize_t length = getline(&line, &capacity, from_);
    std::string reply = length > 0 ? std::string(line, length) : "";
    std::free(line);
    if (!reply.empty() && reply.back() == '\n') reply.pop_back();
    if (reply.empty()) {
      open_spiel::SpielFatalError("hexparrot_player.py stopped (its error is above)");
    }
    return reply;
  }

  pid_t pid_;
  std::FILE* to_;
  std::FILE* from_;
  std::mt19937 rng_;
  int fallbacks_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(args.Get("game", "thud"));
  const std::string spec_a = args.Get("a", ""), spec_b = args.Get("b", "");
  const int sims = args.GetInt("sims", 100);
  const int sims_a = args.GetInt("sims_a", sims), sims_b = args.GetInt("sims_b", sims);
  const int pairs = args.GetInt("pairs", 50);
  const int opening_moves = args.GetInt("opening", 4);
  const int threads = args.GetInt("threads", 16);
  const int batch = args.GetInt("batch", 16);
  const int inference_threads = args.GetInt("inference_threads", 1);
  const int seed = args.GetInt("seed", 1);
  const std::string untried = args.Get("untried", "upstream");
  const std::string untried_a = args.Get("untried_a", untried);
  const std::string untried_b = args.Get("untried_b", untried);
  const std::string temperature_a = args.Get("temperature_a", "");
  const std::string temperature_b = args.Get("temperature_b", "");
  const std::vector<double> temperatures_a =
      open_spiel::thud_az::torch_az::PerPlayerValues(temperature_a);
  const std::vector<double> temperatures_b =
      open_spiel::thud_az::torch_az::PerPlayerValues(temperature_b);
  const open_spiel::thud_az::UntriedMoveValue rule_a =
      open_spiel::thud_az::UntriedMoveValueFromString(untried_a);
  const open_spiel::thud_az::UntriedMoveValue rule_b =
      open_spiel::thud_az::UntriedMoveValueFromString(untried_b);
  const int progress = args.GetInt("progress", 0);
  const int cache_size = args.GetInt("cache", 1 << 18);
  const bool share = args.GetInt("share", 0) != 0;
  const int mcts_memory = args.GetInt("mcts_memory", 1000);
  const std::string games_path = args.Get("games", "");
  // Added 2026-10-15 (thud/PLAN.md, the GPU readiness checklist): where the networks run.
  const std::string devices_list = args.Get("devices", "/cpu:0");
  // 2026-10-19 (thud/PLAN.md, mixed precision): each network's inference precision,
  // float32, mixed or bfloat16 (VPNetModel::SetInferencePrecision; share=1: A's).
  const std::string precision_a = args.Get("precision_a", "float32");
  const std::string precision_b = args.Get("precision_b", "float32");
  const std::vector<std::string> device_names =
      absl::StrSplit(devices_list, ',', absl::SkipEmpty());
  if (device_names.empty()) open_spiel::SpielFatalError("devices= lists no device");
  const std::string capture_floor_a = args.Get("capture_floor_a", "");
  const std::vector<double> capture_floors_a =
      open_spiel::thud_az::torch_az::PerPlayerValues(capture_floor_a);
  const std::string noise_list = args.Get("noise", "");
  const std::vector<double> noise =
      open_spiel::thud_az::torch_az::PerPlayerValues(noise_list);
  const double noise_alpha = std::stod(args.Get("noise_alpha", "0.1"));
  const double noise_epsilon = std::stod(args.Get("noise_epsilon", "0.25"));
  const int sample_plies = args.GetInt("sample_plies", 0);
  const std::string teacher_mix_list = args.Get("teacher_mix_a", "");
  const std::vector<double> teacher_mix_a =
      open_spiel::thud_az::torch_az::PerPlayerValues(teacher_mix_list);
  const int teacher_depth = args.GetInt("teacher_depth", 2);
  const double teacher_temperature = std::stod(args.Get("teacher_temperature", "0.25"));
  // Self-play's conditions: fresh searches every battle (SelfPlayChooser).
  const bool self_play_conditions =
      sample_plies > 0 || std::any_of(noise.begin(), noise.end(),
                                      [](double x) { return x != 0; });
  const std::string hexparrot_player = args.Get(
      "hexparrot_player",
      (std::filesystem::read_symlink("/proc/self/exe").parent_path().parent_path() /
       "thud/experiments/hexparrot_player.py")
          .string());
  args.CheckAllUsed();
  if (spec_a.empty() || spec_b.empty()) {
    std::cerr << "Usage: " << argv[0] << " a=DIR:STEP b=DIR:STEP [key=value ...]"
              << std::endl;
    return 1;
  }
  // Side B without a network: "mcts:SIMS", "alphabeta:DEPTH" or "hexparrot:LOOKAHEAD".
  const std::string kind_b = spec_b.substr(0, spec_b.find(':'));
  const bool outside = kind_b == "mcts" || kind_b == "alphabeta" || kind_b == "hexparrot";
  const int strength_b = outside ? std::stoi(spec_b.substr(kind_b.size() + 1)) : 0;
  if (outside && kind_b == "hexparrot") {
    // A Python player that stopped shows as an error reading its reply, not a SIGPIPE.
    std::signal(SIGPIPE, SIG_IGN);
  }

  if (share && spec_a != spec_b) {
    std::cerr << "share=1 needs the same network on both sides" << std::endl;
    return 1;
  }
  DeviceManager devices_a, devices_b;
  auto eval_a = LoadEvaluator(*game, spec_a, &devices_a, batch, inference_threads,
                              cache_size, device_names, precision_a);
  auto eval_b = share     ? eval_a
                : outside ? nullptr
                          : LoadEvaluator(*game, spec_b, &devices_b, batch,
                                          inference_threads, cache_size, device_names,
                                          precision_b);
  // Network A's searches, with the teacher in its prior if `teacher_mix_a` asks for it.
  std::shared_ptr<open_spiel::thud_az::Evaluator> search_eval_a = eval_a;
  if (std::any_of(teacher_mix_a.begin(), teacher_mix_a.end(),
                  [](double w) { return w > 0; })) {
    search_eval_a = std::make_shared<TeacherMixEvaluator>(
        *game, eval_a, teacher_mix_a, teacher_depth, teacher_temperature);
  }
  // The requests of both evaluators (one if shared or B has none).
  auto requests = [&]() {
    VPNetEvaluator::RequestCounts sum = eval_a->GetRequestCounts();
    if (!share && eval_b) {
      const VPNetEvaluator::RequestCounts b = eval_b->GetRequestCounts();
      sum.value_hits += b.value_hits;
      sum.value_misses += b.value_misses;
      sum.prior_hits += b.prior_hits;
      sum.prior_misses += b.prior_misses;
    }
    return sum;
  };
  auto percent = [](int64_t hits, int64_t misses) {
    return hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0;
  };

  // The openings, fixed by the seed so that reruns play the same pairs.
  std::vector<std::vector<Action>> openings(pairs);
  std::mt19937 rng(seed);
  for (auto& opening : openings) {
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    for (int i = 0; i < opening_moves && !state->IsTerminal(); ++i) {
      std::vector<Action> legal = state->LegalActions();
      opening.push_back(legal[std::uniform_int_distribution<int>(0, legal.size() - 1)(rng)]);
      state->ApplyAction(opening.back());
    }
  }

  std::atomic<bool> finished{false};
  std::thread reporter;
  if (progress > 0) {
    reporter = std::thread([&]() {
      using Clock = std::chrono::steady_clock;
      const Clock::time_point start = Clock::now();
      Clock::time_point last = start;
      int64_t last_count = 0;
      while (!finished) {
        for (int i = 0; i < progress * 10 && !finished; ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        const Clock::time_point now = Clock::now();
        const int64_t count = simulations_done;
        const VPNetEvaluator::RequestCounts r = requests();
        std::cerr << absl::StrFormat(
                         "progress: %.0f s, %d simulations, %.0f simulations/s, "
                         "values %d (%.1f%% cached), move probabilities %d (%.1f%% "
                         "cached)\n",
                         std::chrono::duration<double>(now - start).count(), count,
                         (count - last_count) /
                             std::chrono::duration<double>(now - last).count(),
                         r.value_hits + r.value_misses,
                         percent(r.value_hits, r.value_misses),
                         r.prior_hits + r.prior_misses,
                         percent(r.prior_hits, r.prior_misses))
                  << std::flush;
        last = now;
        last_count = count;
      }
    });
  }

  std::vector<double> pair_margin(pairs), a_dwarfs(pairs), a_trolls(pairs);
  std::atomic<int> next{0};
  std::atomic<int> fallbacks{0};
  double thinking_seconds[2] = {0, 0};  // A's and B's, over all battles (under `out`).
  int64_t thinking_moves[2] = {0, 0};
  std::mutex out;
  std::unique_ptr<std::ofstream> games;
  if (!games_path.empty()) games = std::make_unique<std::ofstream>(games_path);
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t]() {
      auto make_bot = [&](std::shared_ptr<open_spiel::thud_az::Evaluator> eval,
                          int simulations,
                          open_spiel::thud_az::UntriedMoveValue untried_rule,
                          const std::vector<double>& temperatures,
                          const std::vector<double>& capture_floors, int bot_seed,
                          double dirichlet_alpha, double dirichlet_epsilon) {
        auto bot = std::make_unique<MCTSBot>(
            *game, eval, /*uct_c=*/2, simulations, /*max_memory_mb=*/1000, /*solve=*/false,
            bot_seed, /*verbose=*/false,
            open_spiel::thud_az::ChildSelectionPolicy::PUCT, dirichlet_alpha,
            dirichlet_epsilon,
            /*dont_return_chance_node=*/false, /*max_wall_clock_time=*/-1, untried_rule);
        bot->SetPolicyTemperatures(temperatures);
        if (!capture_floors.empty()) {
          bot->SetPriorFloors(capture_floors, [](const open_spiel::State& s, Action a) {
            return absl::EndsWith(s.ActionToString(a), "x");
          });
        }
        return bot;
      };
      std::unique_ptr<MCTSBot> a = make_bot(search_eval_a, sims_a, rule_a, temperatures_a,
                                            capture_floors_a, /*bot_seed=*/t, 0, 0);
      std::unique_ptr<MCTSBot> b;
      std::unique_ptr<Opponent> opponent;
      if (!outside) {
        b = make_bot(eval_b, sims_b, rule_b, temperatures_b, {}, /*bot_seed=*/t, 0, 0);
      } else if (kind_b == "mcts") {
        opponent = std::make_unique<MctsOpponent>(*game, strength_b, mcts_memory);
      } else if (kind_b == "alphabeta") {
        opponent = std::make_unique<AlphaBetaOpponent>(*game, strength_b);
      } else {
        opponent = std::make_unique<HexparrotOpponent>(strength_b, hexparrot_player);
      }
      const Chooser choose_a = NetworkChooser(a.get());
      const Chooser choose_b =
          outside ? Chooser([&opponent](const open_spiel::State& state) {
            return opponent->Choose(state);
          })
                  : NetworkChooser(b.get());
      // One battle of pair i, network A playing a_player; in self-play's conditions with
      // searches of its own, per network and player to move, seeded from the battle.
      auto battle = [&](int i, int a_player) {
        if (outside) opponent->NewBattle(BattleSeed(seed, i, a_player));
        if (!self_play_conditions) {
          return Play(*game, openings[i], choose_a, choose_b, a_player);
        }
        const int base = 8 * BattleSeed(seed, i, a_player);
        std::array<std::unique_ptr<MCTSBot>, 2> bots_a, bots_b;
        for (int p : {0, 1}) {
          const bool noisy = p < noise.size() && noise[p] != 0;
          const double alpha = noisy ? noise_alpha : 0, epsilon = noisy ? noise_epsilon : 0;
          bots_a[p] = make_bot(search_eval_a, sims_a, rule_a, temperatures_a,
                               capture_floors_a, base + p, alpha, epsilon);
          if (!outside) {
            bots_b[p] = make_bot(eval_b, sims_b, rule_b, temperatures_b, {}, base + 2 + p,
                                 alpha, epsilon);
          }
        }
        std::mt19937 rng(base + 4);
        const Chooser self_play_a =
            SelfPlayChooser({bots_a[0].get(), bots_a[1].get()}, sample_plies, &rng);
        const Chooser self_play_b =
            outside ? choose_b
                    : SelfPlayChooser({bots_b[0].get(), bots_b[1].get()}, sample_plies, &rng);
        return Play(*game, openings[i], self_play_a, self_play_b, a_player);
      };
      for (int i = next++; i < pairs; i = next++) {
        const auto start = std::chrono::steady_clock::now();
        const Battle as_dwarfs = battle(i, 0);
        const Battle as_trolls = battle(i, 1);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        a_dwarfs[i] = as_dwarfs.a_return * kMaxMargin + 0.0;  // + 0.0: no "-0".
        a_trolls[i] = as_trolls.a_return * kMaxMargin + 0.0;
        pair_margin[i] = a_dwarfs[i] + a_trolls[i];
        std::lock_guard<std::mutex> lock(out);
        if (games) {
          for (const Battle* battle : {&as_dwarfs, &as_trolls}) {
            *games << absl::StrFormat(
                          "{\"pair\": %d, \"a_player\": %d, \"a_margin\": %.0f, "
                          "\"actions\": [\"%s\"]}",
                          i, battle == &as_trolls, battle->a_return * kMaxMargin + 0.0,
                          absl::StrJoin(battle->actions, "\", \""))
                   << std::endl;
          }
        }
        std::cout << absl::StrFormat(
                         "{\"pair\": %d, \"a_as_dwarfs\": %.0f, \"a_as_trolls\": %.0f, "
                         "\"a_pair_margin\": %.0f, \"moves\": [%d, %d]",
                         i, a_dwarfs[i], a_trolls[i], pair_margin[i], as_dwarfs.moves,
                         as_trolls.moves)
                  << (outside ? absl::StrFormat(
                                    ", \"seconds\": %.1f, \"seconds_per_move\": [%.3f, %.3f]",
                                    seconds,
                                    (as_dwarfs.seconds[0] + as_trolls.seconds[0]) /
                                        (as_dwarfs.choices[0] + as_trolls.choices[0]),
                                    (as_dwarfs.seconds[1] + as_trolls.seconds[1]) /
                                        (as_dwarfs.choices[1] + as_trolls.choices[1]))
                              : "")
                  << "}" << std::endl;
        for (const Battle* battle : {&as_dwarfs, &as_trolls}) {
          for (int side : {0, 1}) {
            thinking_seconds[side] += battle->seconds[side];
            thinking_moves[side] += battle->choices[side];
          }
        }
      }
      if (outside) fallbacks += opponent->Fallbacks();
    });
  }
  for (std::thread& w : workers) w.join();
  finished = true;
  if (reporter.joinable()) reporter.join();

  auto mean = [](const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;
    return sum / v.size();
  };
  const double m = mean(pair_margin);
  double var = 0;
  for (double x : pair_margin) var += (x - m) * (x - m);
  // Student's t for pairs - 1 degrees of freedom (Cornish-Fisher expansion around 1.96;
  // 2.093 for 19). Until 2026-09-28 this used 1.96, slightly too narrow for 20 pairs.
  const double z = 1.959964, df = pairs - 1;
  const double t = df > 0 ? z + (z * z * z + z) / (4 * df) +
                                (5 * std::pow(z, 5) + 16 * z * z * z + 3 * z) / (96 * df * df)
                          : 0;
  const double half_width = pairs > 1 ? t * std::sqrt(var / df) / std::sqrt(pairs) : 0;
  const VPNetEvaluator::RequestCounts r = requests();
  int won = 0, drawn = 0, lost = 0;
  for (double x : pair_margin) (x > 0 ? won : x < 0 ? lost : drawn) += 1;
  std::cout << absl::StrFormat(
                   "{\"summary\": true, \"a\": \"%s\", \"b\": \"%s\", "
                   "\"untried_a\": \"%s\", \"untried_b\": \"%s\", "
                   "\"temperature_a\": \"%s\", \"temperature_b\": \"%s\", "
                   "\"sims_a\": %d, \"sims_b\": %d, "
                   "\"pairs\": %d, \"a_mean_pair_margin\": %.2f, \"ci95\": [%.2f, %.2f], "
                   "\"a_pairs_won\": %d, \"drawn\": %d, \"lost\": %d, "
                   "\"a_mean_as_dwarfs\": %.2f, \"a_mean_as_trolls\": %.2f, "
                   "\"b_mean_as_dwarfs\": %.2f, \"b_mean_as_trolls\": %.2f, "
                   "\"cache\": %d, \"shared\": %s, \"value_requests\": %d, "
                   "\"value_cached\": %.1f, \"prior_requests\": %d, "
                   "\"prior_cached\": %.1f",
                   spec_a, spec_b, untried_a, untried_b, temperature_a, temperature_b,
                   sims_a, sims_b, pairs, m,
                   m - half_width, m + half_width, won, drawn, lost, mean(a_dwarfs),
                   mean(a_trolls), -mean(a_trolls), -mean(a_dwarfs), cache_size,
                   share ? "true" : "false", r.value_hits + r.value_misses,
                   percent(r.value_hits, r.value_misses), r.prior_hits + r.prior_misses,
                   percent(r.prior_hits, r.prior_misses))
            << (capture_floor_a.empty()
                    ? ""
                    : absl::StrFormat(", \"capture_floor_a\": \"%s\"", capture_floor_a))
            << (self_play_conditions
                    ? absl::StrFormat(", \"noise\": \"%s\", \"noise_alpha\": %g, "
                                      "\"noise_epsilon\": %g, \"sample_plies\": %d, "
                                      "\"opening\": %d, \"seed\": %d",
                                      noise_list, noise_alpha, noise_epsilon, sample_plies,
                                      opening_moves, seed)
                    : "")
            << (devices_list == "/cpu:0"
                    ? ""
                    : absl::StrFormat(", \"devices\": \"%s\"", devices_list))
            << (precision_a == "float32" && precision_b == "float32"
                    ? ""
                    : absl::StrFormat(", \"precision_a\": \"%s\", \"precision_b\": \"%s\"",
                                      precision_a, precision_b))
            << (teacher_mix_list.empty()
                    ? ""
                    : absl::StrFormat(", \"teacher_mix_a\": \"%s\", \"teacher_depth\": %d, "
                                      "\"teacher_temperature\": %g",
                                      teacher_mix_list, teacher_depth, teacher_temperature))
            << (outside ? absl::StrFormat(", \"seed\": %d, \"mcts_memory\": %d, "
                                          "\"b_fallbacks\": %d, "
                                          "\"seconds_per_move\": [%.3f, %.3f]",
                                          seed, mcts_memory, fallbacks.load(),
                                          thinking_seconds[0] / thinking_moves[0],
                                          thinking_seconds[1] / thinking_moves[1])
                        : "")
            << "}" << std::endl;
  return 0;
}
