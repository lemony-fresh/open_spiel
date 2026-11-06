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

// A run's own self-play positions relabelled with deeper searches' targets: the user's test
// (A) of better targets (2026-10-30; thud/PLAN.md, the playout cap row) — do deeper targets
// train a better network, and do deep searches picked by surprise beat randomly picked ones?
//
// Relabel mode (the default): from the newest `games` self-play games in RUN_DIR's actor
// logs (games against alpha-beta left out), every `held_out`-th game (in time order) held
// out, each training position searched as the trainer's self-play does (RUN_DIR/config.json's
// settings per side, root noise included; the network at `step`) at the run's own budget and
// at `deep` ("800,400": the dwarfs', the trolls'). Each recorded target is the trainer's —
// (1 - w) x the root's visits + w x the teacher's policy, w per side (with kickstart_ties=search
// the teacher's ties shared by the visits; with the run's finer evaluation, since 2026-11-06),
// over the root's children — and the value the
// game's result for player 0, as the trainer records it. Writes into OUT:
//   all_low.data    every position with its low target (the control: the trainer's own)
//   all_deep.data   every position with its deep target
//   rand_deep.data  the deep target on a random `share` of each side's positions, else low
//   surp_deep.data  the deep target on each side's `share` most surprised positions (the low
//                   search's KL(visits || prior), the signal of 2026-10-30), else low — the
//                   same cost as rand_deep
//   positions.jsonl a line per position: game, side, surprise, the low and deep targets'
//                   distance, picked by rand and by surp
//   held_out.txt    the held-out games' actions, a game a line
// The four buffers hold the same positions in the same order; only their targets differ.
// Every deep search is a separate search, not the low one continued: a deep search that
// extended the low one would cost 600 more simulations a dwarf move, this one 800.
//
// Probe mode (`probe=1`): on the held-out games' positions (an even sample of `max` a side),
// the network at `step`'s long search (`ref` simulations, no root noise) as the reference,
// and for each of `nets` (NAME=DIR:STEP,...) its raw prior against it — the distance (total
// variation), whether its likeliest move is the reference's most visited (ties shared), its
// mass on that move; per side the means with 95% intervals, each net paired against the first.
//
//   thud/build_programs.sh az_relabel
//   OMP_NUM_THREADS=4 build-shared/az_relabel run=RUN_DIR step=STEP out=DIR [games=150
//     held_out=10 deep=800,400 share=0.25 threads=32 batch=32 inference_threads=2 seed=1]
//   OMP_NUM_THREADS=4 build-shared/az_relabel probe=1 run=RUN_DIR step=STEP out=DIR
//     nets=NAME=DIR:STEP,... [ref=4000 max=100 threads=32 batch=32 inference_threads=2]

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/position_buffer.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::ActionsAndProbs;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::AlphaZeroConfig;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;
using Buffer = open_spiel::thud_az::PositionBuffer<VPNetModel::TrainInputs>;
namespace az = open_spiel::thud_az::torch_az;

// A logged self-play game: its actions and player 0's result.
struct Game {
  std::string time;  // The log line's time stamp, for the newest games.
  std::vector<std::string> actions;
  double return0 = 0;
};

// The self-play games of RUN_DIR's actor logs ("Game N: Returns: R0 R1; ...; Actions: ..."),
// games against alpha-beta ("Opponent:") left out; oldest first.
std::vector<Game> LoggedGames(const std::string& run) {
  std::vector<Game> games;
  for (const auto& entry : std::filesystem::directory_iterator(run)) {
    const std::string name = entry.path().filename().string();
    if (!absl::StartsWith(name, "log-actor-")) continue;
    std::ifstream in(entry.path());
    for (std::string line; std::getline(in, line);) {
      if (!absl::StrContains(line, "Actions: ") || absl::StrContains(line, "Opponent:") ||
          !absl::StrContains(line, "Returns: ")) {
        continue;
      }
      Game g;
      g.time = line.substr(0, line.find(']') + 1);
      const size_t r = line.find("Returns: ") + 9;
      g.return0 = std::stod(line.substr(r, line.find(' ', r) - r));
      const std::string actions = line.substr(line.find("Actions: ") + 9);
      for (absl::string_view m : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
        g.actions.push_back(std::string(m));
      }
      games.push_back(std::move(g));
    }
  }
  std::stable_sort(games.begin(), games.end(),
                   [](const Game& a, const Game& b) { return a.time < b.time; });
  return games;
}

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

// The game's positions before each move (a player to move, not terminal).
std::vector<std::unique_ptr<State>> Positions(const open_spiel::Game& game,
                                              const std::vector<std::string>& actions) {
  std::vector<std::unique_ptr<State>> out;
  std::unique_ptr<State> state = game.NewInitialState();
  for (const std::string& move : actions) {
    if (state->IsTerminal()) break;
    out.push_back(state->Clone());
    state->ApplyAction(Find(*state, move));
  }
  return out;
}

// The root's visit counts over all its children (unvisited ones at 0), normalised: the
// trainer's policy target at temperature 1 without pruning (PlayGame).
ActionsAndProbs ChildVisits(const SearchNode& root) {
  ActionsAndProbs visits;
  double total = 0;
  for (const SearchNode& c : root.children) total += c.explore_count;
  for (const SearchNode& c : root.children) {
    visits.emplace_back(c.action, total > 0 ? c.explore_count / total : 0);
  }
  return visits;
}

// The recorded target, as PlayGame mixes it: (1 - w) x target + w x the teacher's policy,
// its ties shared as the run shares them (TeacherTies: by the target's own probabilities with
// kickstart_ties=search, by the positional terms with kickstart_ties=positional).
ActionsAndProbs Recorded(ActionsAndProbs target, const ActionsAndProbs& teacher_policy,
                         double weight, const State& state, const AlphaBetaPlayer& player,
                         bool ties_by_search) {
  if (weight <= 0) return target;
  const ActionsAndProbs teacher =
      az::TeacherTies(teacher_policy, state, player, ties_by_search, target);
  std::map<Action, double> taught(teacher.begin(), teacher.end());
  for (auto& [a, p] : target) p = (1 - weight) * p + weight * taught[a];
  return target;
}

double Distance(const ActionsAndProbs& a, const ActionsAndProbs& b) {
  std::map<Action, double> diff;
  for (const auto& [x, p] : a) diff[x] += p;
  for (const auto& [x, p] : b) diff[x] -= p;
  double d = 0;
  for (const auto& [x, v] : diff) d += std::abs(v);
  return d / 2;
}

// KL(visits || prior) over the visited moves: the search's surprise (KataGo's signal).
double Surprise(const ActionsAndProbs& visits, const ActionsAndProbs& prior) {
  std::map<Action, double> p(prior.begin(), prior.end());
  double kl = 0;
  for (const auto& [a, v] : visits) {
    if (v > 0) kl += v * std::log(v / std::max(p[a], 1e-30));
  }
  return kl;
}

struct Comparison {
  double agrees = 0, mass_on_best = 0, distance = 0;
};

// `candidate` against `reference`: agreement of the likeliest moves (ties in the candidate
// shared: 1/k), the candidate's mass on the reference's most visited move, the distance.
Comparison Compare(const ActionsAndProbs& candidate, const ActionsAndProbs& reference) {
  Action best = reference[0].first;
  double best_p = -1;
  for (const auto& [a, p] : reference) {
    if (p > best_p) best = a, best_p = p;
  }
  double top = -1;
  for (const auto& [a, p] : candidate) top = std::max(top, p);
  int tied = 0;
  bool best_tied = false;
  Comparison c;
  for (const auto& [a, p] : candidate) {
    if (p >= top - 1e-12) {
      ++tied;
      best_tied |= a == best;
    }
    if (a == best) c.mass_on_best = p;
  }
  c.agrees = best_tied ? 1.0 / tied : 0;
  c.distance = Distance(candidate, reference);
  return c;
}

// Mean and 95% interval (t ~ 1.98) of `x`, as JSON.
std::string MeanInterval(const std::vector<double>& x) {
  const double n = x.size();
  const double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
  double ss = 0;
  for (double v : x) ss += (v - mean) * (v - mean);
  const double half = n > 1 ? 1.98 * std::sqrt(ss / (n - 1) / n) : 0;
  return absl::StrFormat("[%.4f, %.4f, %.4f]", mean, mean - half, mean + half);
}

// The self-tests of the arithmetic, on hand-made distributions; stops on a wrong result.
void SelfTest() {
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-12; };
  const ActionsAndProbs ref = {{1, 0.6}, {2, 0.3}, {3, 0.1}};
  const Comparison same = Compare(ref, ref);
  const Comparison tied = Compare({{1, 0.25}, {2, 0.25}, {4, 0.25}, {5, 0.25}}, ref);
  bool ok = near(same.agrees, 1) && near(same.distance, 0) && near(same.mass_on_best, 0.6) &&
            near(tied.agrees, 0.25) && near(tied.distance, 0.5);
  ok &= near(Surprise({{1, 0.5}, {2, 0.5}, {3, 0}}, {{1, 0.5}, {2, 0.5}, {3, 0}}), 0);
  ok &= near(Surprise({{1, 1}}, {{1, 0.5}, {2, 0.5}}), std::log(2.0));
  const std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  const AlphaBetaPlayer plain(*game, 1);  // No positional terms: the teacher's ties as they are.
  const ActionsAndProbs mixed = Recorded({{1, 1}, {2, 0}}, {{1, 0}, {2, 1}}, 0.3,
                                         *game->NewInitialState(), plain, false);
  ok &= near(mixed[0].second, 0.7) && near(mixed[1].second, 0.3);
  if (!ok) open_spiel::SpielFatalError("az_relabel: the self-test failed");
}

}  // namespace

int main(int argc, char** argv) {
  SelfTest();
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const size_t eq = a.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value, got " + a);
    args[a.substr(0, eq)] = a.substr(eq + 1);
  }
  std::vector<std::string> used;
  auto get = [&](const std::string& k, const std::string& d) {
    used.push_back(k);
    auto it = args.find(k);
    return it == args.end() ? d : it->second;
  };
  const bool probe = get("probe", "0") == "1";
  const std::string run = get("run", ""), out = get("out", "");
  const int step = std::stoi(get("step", "-1"));
  const int num_games = std::stoi(get("games", "150"));
  const int held_out = std::stoi(get("held_out", "10"));
  const std::vector<std::string> deep_list = absl::StrSplit(get("deep", "800,400"), ',');
  const double share = std::stod(get("share", "0.25"));
  const std::string nets_arg = get("nets", "");
  const int ref = std::stoi(get("ref", "4000")), max_positions = std::stoi(get("max", "100"));
  const int threads = std::stoi(get("threads", "32")), batch = std::stoi(get("batch", "32"));
  const int inference_threads = std::stoi(get("inference_threads", "2"));
  const int seed = std::stoi(get("seed", "1"));
  for (const auto& [k, v] : args) {
    if (std::find(used.begin(), used.end(), k) == used.end()) {
      open_spiel::SpielFatalError("Unknown argument " + k);
    }
  }
  if (run.empty() || out.empty() || deep_list.size() != 2 || held_out < 2) {
    std::cerr << "Usage: az_relabel run=RUN_DIR step=STEP out=DIR [...] (see the header)"
              << std::endl;
    return 1;
  }
  const int deep[2] = {std::stoi(deep_list[0]), std::stoi(deep_list[1])};

  AlphaZeroConfig config;
  {
    open_spiel::file::File file(run + "/config.json", "r");
    config.FromJson(open_spiel::json::FromString(file.ReadContents()).value().GetObject());
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(config.game);
  const std::vector<double> weights = az::KickstartWeights(config);
  AlphaBetaPlayer teacher(*game, az::KickstartDepths(config));
  // The run's finer evaluation, as its trainer sets it (2026-11-06; before, the plain teacher
  // whatever the run's config said).
  az::ConfigureTeacher(config, &teacher);
  const bool ties_by_search = config.kickstart_ties == "search";
  std::filesystem::create_directories(out);

  DeviceManager devices;
  {
    VPNetModel model(*game, run, config.graph_def, "/cpu:0");
    model.LoadCheckpoint(step);
    model.SetInferencePrecision(config.inference_precision);
    devices.AddDevice(std::move(model));
  }
  auto eval = std::make_shared<VPNetEvaluator>(&devices, batch, inference_threads,
                                               /*cache_size=*/1 << 18, /*shards=*/1);
  const std::vector<double> simulations = az::PerPlayerValues(config.player_simulations);
  const std::vector<double> alphas = az::PerPlayerValues(config.player_policy_alpha);
  const std::vector<double> epsilons = az::PerPlayerValues(config.player_policy_epsilon);
  // As InitAZBot (alpha_zero.cc) builds the trainer's self-play search for `player`: its root
  // noise unless `reference`; `budget` 0 the run's own simulations for the player.
  auto make_bot = [&](int player, bool reference, int budget, int bot_seed) {
    const int own = player < simulations.size() ? static_cast<int>(simulations[player])
                                                : config.max_simulations;
    const bool listed = player < epsilons.size();
    const double epsilon = reference ? 0 : listed ? epsilons[player] : config.policy_epsilon;
    const double alpha = reference || (listed && epsilon == 0) ? 0
                         : player < alphas.size()              ? alphas[player]
                                                               : config.policy_alpha;
    auto bot = std::make_unique<MCTSBot>(
        *game, eval, config.uct_c, budget > 0 ? budget : own,
        /*max_memory_mb=*/reference ? 2000 : 400, /*solve=*/false, bot_seed, /*verbose=*/false,
        open_spiel::thud_az::ChildSelectionPolicy::PUCT, alpha, epsilon,
        /*dont_return_chance_node=*/true, /*max_wall_clock_time=*/-1,
        open_spiel::thud_az::UntriedMoveValueFromString(config.untried_move_value),
        config.untried_move_reduction, /*forced_playouts_k=*/0);
    bot->SetPolicyTemperatures(az::PerPlayerValues(config.player_policy_temperature));
    return bot;
  };

  std::vector<Game> games = LoggedGames(run);
  if (games.size() > num_games) games.erase(games.begin(), games.end() - num_games);

  if (probe) {
    // The held-out positions, an even sample of `max` a side.
    std::vector<std::unique_ptr<State>> by_side[2];
    std::ifstream in(out + "/held_out.txt");
    for (std::string line; std::getline(in, line);) {
      std::vector<std::string> actions = absl::StrSplit(line, ' ', absl::SkipEmpty());
      for (auto& s : Positions(*game, actions)) by_side[s->CurrentPlayer()].push_back(std::move(s));
    }
    std::vector<std::unique_ptr<State>> positions;
    for (auto& all : by_side) {
      for (size_t i = 0; i < all.size(); ++i) {
        if (all.size() > max_positions &&
            i * max_positions / all.size() == (i + 1) * max_positions / all.size()) {
          continue;
        }
        positions.push_back(std::move(all[i]));
      }
    }
    std::vector<std::string> names;
    std::vector<std::unique_ptr<DeviceManager>> net_devices;
    std::vector<std::shared_ptr<VPNetEvaluator>> net_evals;
    for (absl::string_view spec : absl::StrSplit(nets_arg, ',', absl::SkipEmpty())) {
      const size_t eq = spec.find('='), colon = spec.rfind(':');
      if (eq == absl::string_view::npos || colon == absl::string_view::npos) {
        open_spiel::SpielFatalError("nets=NAME=DIR:STEP,...");
      }
      names.push_back(std::string(spec.substr(0, eq)));
      net_devices.push_back(std::make_unique<DeviceManager>());
      VPNetModel model(*game, std::string(spec.substr(eq + 1, colon - eq - 1)), "vpnet.pb",
                       "/cpu:0");
      model.LoadCheckpoint(std::stoi(std::string(spec.substr(colon + 1))));
      model.SetInferencePrecision(config.inference_precision);
      net_devices.back()->AddDevice(std::move(model));
      net_evals.push_back(std::make_shared<VPNetEvaluator>(net_devices.back().get(), 8, 1,
                                                           1 << 14, 1));
    }
    if (names.empty()) open_spiel::SpielFatalError("probe=1 needs nets=NAME=DIR:STEP,...");
    // results[i][n]: net n's prior at position i against the reference.
    std::vector<std::vector<Comparison>> results(positions.size());
    std::vector<int> side(positions.size());
    std::atomic<int> next{0};
    std::mutex mu;
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        std::unique_ptr<MCTSBot> refs[2] = {make_bot(0, true, ref, seed + 2 * t),
                                            make_bot(1, true, ref, seed + 2 * t + 1)};
        for (int i = next++; i < positions.size(); i = next++) {
          const State& state = *positions[i];
          const int p = state.CurrentPlayer();
          side[i] = p;
          const ActionsAndProbs reference = ChildVisits(*refs[p]->MCTSearch(state));
          std::string line = absl::StrFormat("{\"position\": %d, \"side\": \"%s\"", i,
                                             p == 0 ? "dwarfs" : "trolls");
          for (size_t n = 0; n < names.size(); ++n) {
            const Comparison c = Compare(net_evals[n]->Prior(state), reference);
            results[i].push_back(c);
            absl::StrAppendFormat(&line, ", \"%s\": {\"agrees\": %.3f, \"mass_on_best\": %.4f, "
                                  "\"distance\": %.4f}", names[n], c.agrees, c.mass_on_best,
                                  c.distance);
          }
          std::lock_guard<std::mutex> lock(mu);
          std::cout << line << "}" << std::endl;
        }
      });
    }
    for (auto& w : workers) w.join();
    for (int s = 0; s < 2; ++s) {
      for (size_t n = 0; n < names.size(); ++n) {
        std::vector<double> agrees, mass, distance, d_distance, d_mass;
        for (size_t i = 0; i < positions.size(); ++i) {
          if (side[i] != s) continue;
          agrees.push_back(results[i][n].agrees);
          mass.push_back(results[i][n].mass_on_best);
          distance.push_back(results[i][n].distance);
          d_distance.push_back(results[i][n].distance - results[i][0].distance);
          d_mass.push_back(results[i][n].mass_on_best - results[i][0].mass_on_best);
        }
        std::cout << absl::StrFormat(
                         "{\"summary\": true, \"side\": \"%s\", \"net\": \"%s\", \"positions\": "
                         "%d, \"ref\": %d, \"agrees\": %s, \"mass_on_best\": %s, \"distance\": "
                         "%s, \"minus_first\": {\"distance\": %s, \"mass_on_best\": %s}}",
                         s == 0 ? "dwarfs" : "trolls", names[n], distance.size(), ref,
                         MeanInterval(agrees), MeanInterval(mass), MeanInterval(distance),
                         MeanInterval(d_distance), MeanInterval(d_mass))
                  << std::endl;
      }
    }
    return 0;
  }

  // Relabel: the training games' positions, the held-out games' actions.
  struct Position {
    int game = 0, side = 0;
    std::unique_ptr<State> state;
    double value = 0, surprise = 0, change = 0;  // change: the low and deep targets' distance.
    VPNetModel::TrainInputs low, deep;
  };
  std::vector<Position> positions;
  {
    std::ofstream held(out + "/held_out.txt");
    for (size_t g = 0; g < games.size(); ++g) {
      if (g % held_out == 0) {
        held << absl::StrJoin(games[g].actions, " ") << "\n";
        continue;
      }
      for (auto& s : Positions(*game, games[g].actions)) {
        Position p;
        p.game = g;
        p.side = s->CurrentPlayer();
        p.value = games[g].return0;
        p.state = std::move(s);
        positions.push_back(std::move(p));
      }
    }
  }
  std::cerr << absl::StrFormat(
                   "az_relabel: %s step %d, %d games (%d held out), %d positions; the low "
                   "budget the run's own, the deep %d, %d; the teacher's depths %d, %d, weights "
                   "%g, %g, ties %s, threat %s (%s), positional terms %s; precision %s",
                   run, step, games.size(), (games.size() + held_out - 1) / held_out,
                   positions.size(), deep[0], deep[1], teacher.depth(0), teacher.depth(1),
                   weights[0], weights[1], config.kickstart_ties, config.teacher_threat,
                   config.teacher_threat_weights,
                   config.kickstart_ties == "positional" ? config.teacher_terms_weights : "off",
                   config.inference_precision)
            << std::endl;
  {
    std::atomic<int> next{0}, done{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        std::unique_ptr<MCTSBot> low[2], high[2];
        for (int p = 0; p < 2; ++p) {
          low[p] = make_bot(p, false, 0, seed * 1000 + 8 * t + p);
          high[p] = make_bot(p, false, deep[p], seed * 1000 + 8 * t + 4 + p);
        }
        for (int i = next++; i < positions.size(); i = next++) {
          Position& pos = positions[i];
          const State& state = *pos.state;
          const int p = pos.side;
          const ActionsAndProbs low_visits = ChildVisits(*low[p]->MCTSearch(state));
          const ActionsAndProbs deep_visits = ChildVisits(*high[p]->MCTSearch(state));
          pos.surprise = Surprise(low_visits, eval->Prior(state));
          const ActionsAndProbs teacher_policy =
              az::TeacherPolicy(state, teacher, config.kickstart_temperature);
          const double w = p < weights.size() ? weights[p] : config.kickstart_weight;
          const std::vector<float> observation = state.ObservationTensor();
          const std::vector<Action> legal = state.LegalActions();
          pos.low = {legal, observation,
                     Recorded(low_visits, teacher_policy, w, state, teacher, ties_by_search),
                     pos.value};
          pos.deep = {legal, observation,
                      Recorded(deep_visits, teacher_policy, w, state, teacher, ties_by_search),
                      pos.value};
          pos.change = Distance(pos.low.policy, pos.deep.policy);
          pos.state.reset();  // Its memory: ~40,000 positions.
          const int d = ++done;
          if (d % 1000 == 0) std::cerr << d << " positions relabelled" << std::endl;
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  // The picks, per side: a random share, and the share most surprised.
  const size_t n = positions.size();
  std::vector<bool> rand_pick(n), surp_pick(n);
  std::mt19937 rng(seed);
  for (int s = 0; s < 2; ++s) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < n; ++i) {
      if (positions[i].side == s) idx.push_back(i);
    }
    const size_t k = static_cast<size_t>(std::llround(share * idx.size()));
    std::vector<size_t> shuffled = idx;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    for (size_t j = 0; j < k; ++j) rand_pick[shuffled[j]] = true;
    std::vector<size_t> by_surprise = idx;
    std::stable_sort(by_surprise.begin(), by_surprise.end(), [&](size_t a, size_t b) {
      return positions[a].surprise > positions[b].surprise;
    });
    for (size_t j = 0; j < k; ++j) surp_pick[by_surprise[j]] = true;
  }
  // The four buffers, the same positions in the same order; each read back and counted.
  struct Variant {
    const char* name;
    std::function<bool(size_t)> deep;
  };
  const std::vector<Variant> variants = {
      {"all_low", [](size_t) { return false; }},
      {"all_deep", [](size_t) { return true; }},
      {"rand_deep", [&](size_t i) { return static_cast<bool>(rand_pick[i]); }},
      {"surp_deep", [&](size_t i) { return static_cast<bool>(surp_pick[i]); }}};
  for (const Variant& v : variants) {
    const std::string path = absl::StrCat(out, "/", v.name, ".data");
    {
      Buffer buffer(n);
      for (size_t i = 0; i < n; ++i) buffer.Add(v.deep(i) ? positions[i].deep : positions[i].low);
      buffer.SaveBuffer(path);
    }
    // Read back as the learner will: every position's target the one meant (as floats).
    Buffer check(n);
    check.LoadBuffer(path);
    SPIEL_CHECK_EQ(static_cast<size_t>(check.Size()), n);
    for (size_t i = 0; i < n; ++i) {
      const VPNetModel::TrainInputs got = check.At(i);
      const VPNetModel::TrainInputs& meant = v.deep(i) ? positions[i].deep : positions[i].low;
      SPIEL_CHECK_EQ(got.policy.size(), meant.policy.size());
      for (size_t j = 0; j < meant.policy.size(); ++j) {
        SPIEL_CHECK_EQ(got.policy[j].first, meant.policy[j].first);
        SPIEL_CHECK_LT(std::abs(got.policy[j].second - meant.policy[j].second), 1e-6);
      }
      SPIEL_CHECK_EQ(got.value, meant.value);
    }
  }
  {
    std::ofstream lines(out + "/positions.jsonl");
    for (size_t i = 0; i < n; ++i) {
      const Position& p = positions[i];
      lines << absl::StrFormat(
          "{\"position\": %d, \"game\": %d, \"side\": \"%s\", \"surprise\": %.5f, \"change\": "
          "%.5f, \"value\": %.5f, \"rand\": %d, \"surp\": %d}\n",
          i, p.game, p.side == 0 ? "dwarfs" : "trolls", p.surprise, p.change, p.value,
          static_cast<int>(rand_pick[i]), static_cast<int>(surp_pick[i]));
    }
  }
  // The summary: per side, how far the deep search moved the target, over all positions and
  // over each pick — the picks' share of the whole change (the share's own: `share`).
  for (int s = 0; s < 2; ++s) {
    double all = 0, rand = 0, surp = 0;
    int count = 0, picked = 0;
    for (size_t i = 0; i < n; ++i) {
      if (positions[i].side != s) continue;
      ++count;
      all += positions[i].change;
      if (rand_pick[i]) rand += positions[i].change;
      if (surp_pick[i]) surp += positions[i].change, ++picked;
    }
    std::cout << absl::StrFormat(
                     "{\"summary\": true, \"side\": \"%s\", \"positions\": %d, \"picked\": %d, "
                     "\"mean_change\": %.4f, \"rand_share_of_change\": %.3f, "
                     "\"surp_share_of_change\": %.3f}",
                     s == 0 ? "dwarfs" : "trolls", count, picked, count ? all / count : 0,
                     all > 0 ? rand / all : 0, all > 0 ? surp / all : 0)
              << std::endl;
  }
  return 0;
}
