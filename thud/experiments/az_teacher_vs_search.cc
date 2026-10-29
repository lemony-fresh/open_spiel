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

// Is kickstarting's teacher still ahead of the search? (thud/PLAN.md, the Phase 6 roadmap,
// *After the depth-5 teacher*: the trigger for tapering the teacher; the user, 2026-10-24.)
// On a run's own recent self-play positions (its actors' logs, games against alpha-beta
// left out: the last `last` games of each actor, an even sample of `max` positions a side),
// with the run's own settings (RUN_DIR/config.json: its game, simulations, root noise,
// policy temperatures and untried-move rule per side, the teacher's depths, weights and
// temperature, the inference precision) and its network at `step`, the candidate policy
// targets:
//   visitsN  the trainer's own search's visit counts at N simulations, each N in `sims`
//            (the first the trainer's own, default the run's) — what it records without a
//            teacher; more budgets answer whether more simulations bring the visits closer
//            to the reference or only spread them (the recheck of 2026-10-04's az_noise_probe,
//            whose question — two root-noise seeds disagreeing — no longer applies to the
//            dwarfs, who search without root noise since run N's recipe);
//   completed_q  (2026-10-29) the completed-Q target of the trainer's own search (Gumbel
//            AlphaZero's, CompletedQTarget: from the root's values, not its visit counts);
//   teacher  the teacher's policy (TeacherPolicy at the run's depths per side);
//   target   what the trainer records: (1 - w) x the first visits + w x teacher, w per side;
//   cq_target  (2026-10-29) what it would record with --policy_target=completed_q:
//            (1 - w) x completed_q + w x teacher;
//   uniform  every legal move alike — the floor: a candidate no closer than it says nothing;
// each compared with the references — the same network's search without root noise at each
// budget in `refs`, the run's settings otherwise (the first the reference proper; each deeper
// one is compared with it: how stable the reference is, the closest a candidate can be
// expected to come):
//   agrees        the chance that the candidate's most likely move is the reference's most
//                 visited one, ties shared: 1/k when the reference's move is among the k
//                 moves tied for the candidate's highest probability (the teacher ties many);
//   mass_on_best  the candidate's probability on the reference's most visited move;
//   distance      the total variation distance (half the summed absolute differences);
// and each candidate's effective moves (the exponential of its entropy: how spread it is).
// Per side and reference: means with 95% intervals, and paired per position each candidate
// minus the first visits. The reading: once the search's visits come closer to the
// reference than the teacher's policy (a smaller distance, more mass on its best move), the
// teacher no longer leads the search and holds the network back. The reference is the
// network's own longer search — the best estimate at hand, not the truth: it shares the
// network's blind spots, so it judges the search's direction more kindly than the teacher's.
//
//   thud/build_programs.sh az_teacher_vs_search
//   OMP_NUM_THREADS=4 build-shared/az_teacher_vs_search run=RUN_DIR step=STEP \
//     [sims=(the run's) refs=2000,4000 last=2 max=100 threads=32 batch=32
//      inference_threads=2 seed=1 precision=(the run's) depths=(the run's teacher's)]
//
// `depths` ("5,5"): the teacher's depths per side instead of the run's.
//
// Prints one JSON line per position, then one summary line per side and reference. Checks
// its comparison on hand-made distributions first and stops if they are wrong.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "thud/az/alpha_zero.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::AlphaBetaPlayer;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::AlphaZeroConfig;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;
using Policy = std::map<Action, double>;

// KataGo's forced playouts, as the trainer's (alpha_zero.cc's kForcedPlayoutsK).
constexpr double kForcedPlayoutsK = 2;

// The root's visit counts, normalised: the trainer's policy target without a teacher.
Policy Visits(const SearchNode& root) {
  Policy visits;
  double total = 0;
  for (const SearchNode& c : root.children) total += c.explore_count;
  for (const SearchNode& c : root.children) {
    if (c.explore_count > 0) visits[c.action] = c.explore_count / total;
  }
  return visits;
}

struct Comparison {
  double agrees, mass_on_best, distance;
};

// The exponential of the entropy: as many moves as an even spread with the same entropy.
double EffectiveMoves(const Policy& p) {
  double h = 0;
  for (const auto& [a, q] : p) {
    if (q > 0) h -= q * std::log(q);
  }
  return std::exp(h);
}

// `candidate` against `reference` (both summing to 1; moves missing have probability 0).
Comparison Compare(const Policy& candidate, const Policy& reference) {
  Action best = reference.begin()->first;  // The reference's most visited: the first such.
  for (const auto& [a, p] : reference) {
    if (p > reference.at(best)) best = a;
  }
  double top = 0;
  for (const auto& [a, p] : candidate) top = std::max(top, p);
  int tied = 0;
  bool best_tied = false;
  for (const auto& [a, p] : candidate) {
    if (p >= top - 1e-12) {
      ++tied;
      best_tied |= a == best;
    }
  }
  double distance = 0;
  for (const auto& [a, p] : candidate) {
    auto it = reference.find(a);
    distance += std::abs(p - (it == reference.end() ? 0 : it->second));
  }
  for (const auto& [a, p] : reference) {
    if (!candidate.count(a)) distance += p;
  }
  auto it = candidate.find(best);
  return {best_tied ? 1.0 / tied : 0.0, it == candidate.end() ? 0.0 : it->second, distance / 2};
}

// The comparison on hand-made distributions; stops on a wrong result.
void SelfTest() {
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-12; };
  const Policy reference = {{1, 0.6}, {2, 0.3}, {3, 0.1}};
  const Comparison same = Compare(reference, reference);
  const Comparison tied = Compare({{1, 0.25}, {2, 0.25}, {4, 0.25}, {5, 0.25}}, reference);
  const Comparison wrong = Compare({{2, 0.9}, {3, 0.1}}, reference);
  const bool ok = near(same.agrees, 1) && near(same.mass_on_best, 0.6) && near(same.distance, 0) &&
                  near(tied.agrees, 0.25) && near(tied.mass_on_best, 0.25) &&
                  near(tied.distance, 0.5) &&  // (0.35 + 0.05 + 0.1 + 0.25 + 0.25) / 2.
                  near(wrong.agrees, 0) && near(wrong.mass_on_best, 0) && near(wrong.distance, 0.6);
  const bool spread_ok = near(EffectiveMoves({{1, 0.5}, {2, 0.5}}), 2) &&
                         near(EffectiveMoves({{1, 1}}), 1) &&
                         near(EffectiveMoves({{1, 0.25}, {2, 0.25}, {3, 0.25}, {4, 0.25}}), 4);
  if (!ok || !spread_ok) {
    open_spiel::SpielFatalError("az_teacher_vs_search: the comparison's self-test failed");
  }
}

struct Stats {
  double mean, low, high;
};

Stats MeanInterval(const std::vector<double>& v) {
  const int n = v.size();
  double m = 0;
  for (double x : v) m += x;
  m /= std::max(n, 1);
  double var = 0;
  for (double x : v) var += (x - m) * (x - m);
  const double half = n > 1 ? 1.96 * std::sqrt(var / (n - 1) / n) : 0;
  return {m, m - half, m + half};
}

std::string Json(const Stats& s) { return absl::StrFormat("[%.4f, %.4f, %.4f]", s.mean, s.low, s.high); }

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

}  // namespace

int main(int argc, char** argv) {
  SelfTest();
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const size_t eq = arg.find('=');
    if (eq == std::string::npos) open_spiel::SpielFatalError("Expected key=value: " + arg);
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }
  std::vector<std::string> used;
  auto get = [&](const std::string& k, const std::string& d) {
    used.push_back(k);
    return args.count(k) ? args[k] : d;
  };
  const std::string run = get("run", "");
  const int step = std::stoi(get("step", "-1"));
  const std::string sims_arg = get("sims", "");
  const std::string depths_arg = get("depths", "");  // The teacher's depths per side instead.
  std::vector<int> refs;
  for (absl::string_view s : absl::StrSplit(get("refs", "2000,4000"), ',')) {
    int v;
    if (!absl::SimpleAtoi(s, &v)) open_spiel::SpielFatalError("refs=N,N,...");
    refs.push_back(v);
  }
  const int last = std::stoi(get("last", "2")), max_positions = std::stoi(get("max", "100"));
  const int threads = std::stoi(get("threads", "32")), batch = std::stoi(get("batch", "32"));
  const int inference_threads = std::stoi(get("inference_threads", "2"));
  const int seed = std::stoi(get("seed", "1"));
  const std::string precision_arg = get("precision", "");
  for (const auto& [k, v] : args) {
    if (std::find(used.begin(), used.end(), k) == used.end()) {
      open_spiel::SpielFatalError("Unknown argument " + k);
    }
  }
  if (run.empty()) {
    std::cerr << "Usage: az_teacher_vs_search run=RUN_DIR step=STEP [...]" << std::endl;
    return 1;
  }

  // The run's settings.
  AlphaZeroConfig config;
  {
    open_spiel::file::File file(run + "/config.json", "r");
    config.FromJson(open_spiel::json::FromString(file.ReadContents()).value().GetObject());
  }
  namespace az = open_spiel::thud_az::torch_az;
  if (!depths_arg.empty()) config.player_kickstart_depth = depths_arg;
  const std::vector<double> weights = az::KickstartWeights(config);
  const AlphaBetaPlayer teacher(*open_spiel::LoadGame(config.game), az::KickstartDepths(config));
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(config.game);
  const std::string precision = precision_arg.empty() ? config.inference_precision : precision_arg;
  // The budgets: the trainer's own first (as its config sets it per side), then the others.
  std::vector<int> sims = {0};  // 0: the run's own, per side.
  for (absl::string_view b : absl::StrSplit(sims_arg, ',', absl::SkipEmpty())) {
    int v;
    if (!absl::SimpleAtoi(b, &v) || v <= 0) open_spiel::SpielFatalError("sims=N,N,...");
    sims.push_back(v);
  }
  std::vector<std::string> candidates;
  for (int b : sims) candidates.push_back(b == 0 ? "visits" : absl::StrCat("visits", b));
  // Since 2026-10-29 (Gumbel targets, part one): completed_q, the completed-Q target of the
  // run's own search (CompletedQTarget), and cq_target, the target the trainer would record
  // with it (--policy_target=completed_q: it mixed with the teacher as `target` mixes the
  // visits).
  for (const char* c : {"completed_q", "teacher", "target", "cq_target", "uniform"}) {
    candidates.push_back(c);
  }
  std::cerr << absl::StrFormat("az_teacher_vs_search: %s step %d, game %s, the teacher's depths %d, %d, "
                               "weights %g, %g, temperature %g; precision %s",
                               run, step, config.game, teacher.depth(0), teacher.depth(1),
                               weights[0], weights[1], config.kickstart_temperature, precision)
            << std::endl;

  // The positions: each actor's last games, an even sample a side.
  std::vector<std::unique_ptr<State>> by_side[2];
  {
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(run)) {
      const std::string name = entry.path().filename().string();
      if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    for (const std::string& path : paths) {
      std::vector<std::string> lines;
      std::ifstream in(path);
      for (std::string line; std::getline(in, line);) {
        if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
          lines.push_back(line);
        }
      }
      for (int i = std::max<int>(0, lines.size() - last); i < lines.size(); ++i) {
        std::unique_ptr<State> state = game->NewInitialState();
        const std::string actions = lines[i].substr(lines[i].find("Actions: ") + 9);
        for (absl::string_view move : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
          if (state->IsTerminal()) break;
          by_side[state->CurrentPlayer()].push_back(state->Clone());
          state->ApplyAction(Find(*state, std::string(move)));
        }
      }
    }
  }
  std::vector<std::unique_ptr<State>> positions;
  for (auto& all : by_side) {
    for (size_t i = 0; i < all.size(); ++i) {
      if (all.size() > max_positions &&
          i * max_positions / all.size() == (i + 1) * max_positions / all.size()) {
        continue;  // An even sample.
      }
      positions.push_back(std::move(all[i]));
    }
  }
  std::cerr << positions.size() << " positions" << std::endl;

  // The network and the searches.
  DeviceManager devices;
  {
    VPNetModel model(*game, run, config.graph_def, "/cpu:0");
    model.LoadCheckpoint(step);
    model.SetInferencePrecision(precision);
    devices.AddDevice(std::move(model));
  }
  auto eval = std::make_shared<VPNetEvaluator>(&devices, batch, inference_threads,
                                               /*cache_size=*/1 << 18, /*shards=*/1);
  // As InitAZBot (alpha_zero.cc) builds the trainer's self-play search for `player`, with its
  // own seed and memory; `reference`: no root noise and `simulations` simulations.
  const std::vector<double> simulations = az::PerPlayerValues(config.player_simulations);
  const std::vector<double> alphas = az::PerPlayerValues(config.player_policy_alpha);
  const std::vector<double> epsilons = az::PerPlayerValues(config.player_policy_epsilon);
  // `budget` 0: the run's own simulations for `player`.
  auto make_bot = [&](int player, bool reference, int budget, int bot_seed) {
    const int own = player < simulations.size() ? static_cast<int>(simulations[player])
                                                : config.max_simulations;
    const int n = budget > 0 ? budget : own;
    const bool listed = player < epsilons.size();
    const double epsilon = reference ? 0 : listed ? epsilons[player] : config.policy_epsilon;
    const double alpha = reference || (listed && epsilon == 0) ? 0
                         : player < alphas.size()              ? alphas[player]
                                                               : config.policy_alpha;
    auto bot = std::make_unique<MCTSBot>(
        *game, eval, config.uct_c, n, /*max_memory_mb=*/reference ? 2000 : 200,
        /*solve=*/false, bot_seed, /*verbose=*/false,
        open_spiel::thud_az::ChildSelectionPolicy::PUCT, alpha, epsilon,
        /*dont_return_chance_node=*/true, /*max_wall_clock_time=*/-1,
        open_spiel::thud_az::UntriedMoveValueFromString(config.untried_move_value),
        config.untried_move_reduction,
        !reference && config.policy_target_pruning ? kForcedPlayoutsK : 0);
    bot->SetPolicyTemperatures(az::PerPlayerValues(config.player_policy_temperature));
    return bot;
  };

  // Each position: the references, then the candidates, each compared with every reference.
  const int nr = refs.size(), nc = candidates.size(), nb = sims.size();
  // results[i][r][c]: candidate c against reference r; stable[i][r]: reference r against 0.
  std::vector<std::vector<std::vector<Comparison>>> results(
      positions.size(), std::vector<std::vector<Comparison>>(nr));
  std::vector<std::vector<Comparison>> stable(positions.size());
  std::vector<std::vector<double>> spread(positions.size());  // Effective moves per candidate.
  std::vector<int> side(positions.size());
  std::mutex mu;
  {
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        std::vector<std::unique_ptr<MCTSBot>> searches[2];  // By side, then budget.
        for (int p = 0; p < 2; ++p) {
          for (int b = 0; b < nb; ++b) {
            searches[p].push_back(make_bot(p, false, sims[b], seed * 1000 + 64 * t + 8 * b + p));
          }
        }
        std::vector<std::unique_ptr<MCTSBot>> references[2];
        for (int p = 0; p < 2; ++p) {
          for (int r = 0; r < nr; ++r) references[p].push_back(make_bot(p, true, refs[r], seed));
        }
        for (int i = next++; i < positions.size(); i = next++) {
          const State& state = *positions[i];
          const int p = state.CurrentPlayer();
          side[i] = p;
          std::vector<Policy> ref;
          for (int r = 0; r < nr; ++r) ref.push_back(Visits(*references[p][r]->MCTSearch(state)));
          std::vector<Policy> visits;
          Policy completed_q, cq_target;
          for (int b = 0; b < nb; ++b) {
            std::unique_ptr<SearchNode> root = searches[p][b]->MCTSearch(state);
            visits.push_back(Visits(*root));
            if (b == 0) {
              for (const auto& [a, q] : az::CompletedQTarget(*root, state, eval.get())) {
                completed_q[a] = q;
              }
            }
          }
          Policy teacher_policy, target, uniform;
          for (const auto& [a, q] : az::TeacherPolicy(state, teacher, config.kickstart_temperature)) {
            teacher_policy[a] = q;
          }
          const std::vector<Action> legal = state.LegalActions();
          for (Action a : legal) {
            auto it = visits[0].find(a);
            target[a] = (1 - weights[p]) * (it == visits[0].end() ? 0 : it->second) +
                        weights[p] * teacher_policy[a];
            cq_target[a] = (1 - weights[p]) * completed_q[a] + weights[p] * teacher_policy[a];
            uniform[a] = 1.0 / legal.size();
          }
          std::vector<const Policy*> policies;
          for (const Policy& v : visits) policies.push_back(&v);
          for (const Policy* q : {&completed_q, &teacher_policy, &target, &cq_target, &uniform}) {
            policies.push_back(q);
          }
          for (const Policy* q : policies) spread[i].push_back(EffectiveMoves(*q));
          std::string line = absl::StrFormat("{\"position\": %d, \"side\": \"%s\", \"legal\": %d", i,
                                             p == 0 ? "dwarfs" : "trolls", legal.size());
          for (int r = 1; r < nr; ++r) {
            const Comparison c = Compare(ref[0], ref[r]);
            stable[i].push_back(c);
            absl::StrAppendFormat(&line, ", \"ref%d_vs_ref%d\": {\"agrees\": %.3f, \"distance\": %.4f}",
                                  refs[0], refs[r], c.agrees, c.distance);
          }
          for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
              const Comparison x = Compare(*policies[c], ref[r]);
              results[i][r].push_back(x);
              absl::StrAppendFormat(&line,
                                    ", \"%s_ref%d\": {\"agrees\": %.3f, \"mass_on_best\": %.4f, "
                                    "\"distance\": %.4f}",
                                    candidates[c], refs[r], x.agrees, x.mass_on_best, x.distance);
            }
          }
          std::lock_guard<std::mutex> lock(mu);
          std::cout << line << "}" << std::endl;
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  // Per side: the references' stability, then per reference each candidate, and the
  // teacher and the target against the visits, paired.
  for (int s = 0; s < 2; ++s) {
    for (int r = 1; r < nr; ++r) {
      std::vector<double> agrees, distance;
      for (size_t i = 0; i < positions.size(); ++i) {
        if (side[i] != s) continue;
        agrees.push_back(stable[i][r - 1].agrees);
        distance.push_back(stable[i][r - 1].distance);
      }
      std::cout << absl::StrFormat("{\"summary\": true, \"stability\": true, \"side\": \"%s\", "
                                   "\"ref\": %d, \"deeper_ref\": %d, \"positions\": %d, "
                                   "\"agrees\": %s, \"distance\": %s}",
                                   s == 0 ? "dwarfs" : "trolls", refs[0], refs[r], agrees.size(),
                                   Json(MeanInterval(agrees)), Json(MeanInterval(distance)))
                << std::endl;
    }
    for (int r = 0; r < nr; ++r) {
      for (int c = 0; c < nc; ++c) {
        std::vector<double> agrees, mass, distance, d_agrees, d_mass, d_distance, moves;
        for (size_t i = 0; i < positions.size(); ++i) {
          if (side[i] != s) continue;
          moves.push_back(spread[i][c]);
          const Comparison& x = results[i][r][c];
          const Comparison& visits = results[i][r][0];
          agrees.push_back(x.agrees);
          mass.push_back(x.mass_on_best);
          distance.push_back(x.distance);
          d_agrees.push_back(x.agrees - visits.agrees);
          d_mass.push_back(x.mass_on_best - visits.mass_on_best);
          d_distance.push_back(x.distance - visits.distance);
        }
        std::cout << absl::StrFormat(
                         "{\"summary\": true, \"side\": \"%s\", \"candidate\": \"%s\", \"ref\": %d, "
                         "\"positions\": %d, \"agrees\": %s, \"mass_on_best\": %s, \"distance\": %s, "
                         "\"effective_moves\": %s, "
                         "\"minus_visits\": {\"agrees\": %s, \"mass_on_best\": %s, \"distance\": %s}}",
                         s == 0 ? "dwarfs" : "trolls", candidates[c], refs[r], agrees.size(),
                         Json(MeanInterval(agrees)), Json(MeanInterval(mass)),
                         Json(MeanInterval(distance)), Json(MeanInterval(moves)),
                         Json(MeanInterval(d_agrees)),
                         Json(MeanInterval(d_mass)), Json(MeanInterval(d_distance)))
                  << std::endl;
      }
    }
  }
  return 0;
}
