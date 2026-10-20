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

// Does a network infer and search the same in lower precision? (The user, 2026-10-19:
// bfloat16 inference only if it barely ever changes a prediction; thud/PLAN.md, mixed
// precision.) On positions of real games (an az_match games file: every 7th ply from ply
// 10), the network in float32 against each of `precisions` (VPNetModel::
// SetInferencePrecision: mixed, bfloat16):
//
//   1. The outputs: the values' difference (mean, largest), the priors' total variation
//      distance (half the L1; mean, largest), how often the most likely move is the same;
//      where it is not, how far apart float32 put the two moves (its prior gap).
//   2. The searches, as the trainer searches (our MCTSBot, `sims` simulations, the untried
//      moves at their siblings' mean minus 0.2, the policy temperatures 1.5 for the dwarfs
//      and 1 for the trolls) but without root noise, each position searched by float32 and
//      by the lower precision with the same seed: how often the most visited move is the
//      same; where it is not, the user's near-tie measure — in float32's search, the two
//      moves' shares of the root's visits and their values (the mean return for the side to
//      move), and their gaps. The control: float32 searched again with another seed — how
//      often the move changes by the searches' random tie-breaks alone, the floor any
//      precision's changes are to be read against.
//
// Every evaluation alone (batches of 1), so that no batch's makeup changes a result.
// Prints one JSON line per precision and part, and with `each=PATH` one line per searched
// position.
//
//   thud/build_programs.sh az_precision_probe
//   OMP_NUM_THREADS=1 build-shared/az_precision_probe net=RUN_DIR:STEP games=GAMES.jsonl \
//     [positions=600 sims=100 precisions=mixed,bfloat16 threads=8 seed=1 each=PATH]

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "thud/az/device_manager.h"
#include "thud/az/mcts.h"
#include "thud/az/vpevaluator.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::MCTSBot;
using open_spiel::thud_az::SearchNode;
using open_spiel::thud_az::torch_az::DeviceManager;
using open_spiel::thud_az::torch_az::VPNetEvaluator;
using open_spiel::thud_az::torch_az::VPNetModel;

struct Searched {
  Action move = 0;
  std::map<Action, std::pair<double, double>> children;  // Visit share, value.
};

Searched Search(const open_spiel::Game& game, std::shared_ptr<VPNetEvaluator> eval,
                const State& state, int sims, int seed) {
  MCTSBot bot(game, eval, /*uct_c=*/2, sims, /*max_memory_mb=*/1000, /*solve=*/false, seed,
              /*verbose=*/false, open_spiel::thud_az::ChildSelectionPolicy::PUCT,
              /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0,
              /*dont_return_chance_node=*/false, /*max_wall_clock_time=*/-1,
              open_spiel::thud_az::UntriedMoveValue::kSiblingMeanMinusReduction);
  bot.SetPolicyTemperatures({1.5, 1});
  std::unique_ptr<SearchNode> root = bot.MCTSearch(state);
  Searched out;
  out.move = root->BestChild().action;
  for (const SearchNode& child : root->children) {
    out.children[child.action] = {
        static_cast<double>(child.explore_count) / std::max(root->explore_count, 1),
        child.explore_count > 0 ? child.total_reward / child.explore_count : NAN};
  }
  return out;
}

std::shared_ptr<VPNetEvaluator> Evaluator(const open_spiel::Game& game, const std::string& dir,
                                          int step, const std::string& precision, int threads,
                                          DeviceManager* devices) {
  VPNetModel model(game, dir, "vpnet.pb", "/cpu:0");
  model.LoadCheckpoint(step);
  model.SetInferencePrecision(precision);
  devices->AddDevice(std::move(model));
  return std::make_shared<VPNetEvaluator>(devices, /*batch_size=*/1, threads,
                                          /*cache_size=*/1 << 16, /*cache_shards=*/1);
}

double Quantile(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v[std::min<size_t>(v.size() - 1, q * v.size())];
}

}  // namespace

int main(int argc, char** argv) {
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
  const std::string net = get("net", ""), games_path = get("games", "");
  if (net.empty() || games_path.empty()) {
    std::cerr << "Usage: az_precision_probe net=DIR:STEP games=GAMES.jsonl [...]" << std::endl;
    return 1;
  }
  const size_t colon = net.rfind(':');
  const std::string dir = net.substr(0, colon);
  const int step = std::stoi(net.substr(colon + 1));
  const int count = std::stoi(get("positions", "600"));
  const int sims = std::stoi(get("sims", "100"));
  const int threads = std::stoi(get("threads", "8"));
  const int seed = std::stoi(get("seed", "1"));
  const std::vector<std::string> precisions =
      absl::StrSplit(get("precisions", "mixed,bfloat16"), ',');
  const std::string each_path = get("each", "");

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");
  // The positions: every 7th ply from ply 10 of each battle, up to `count`, spread over
  // the battles in order.
  std::vector<std::unique_ptr<State>> positions;
  {
    std::ifstream in(games_path);
    std::string line;
    while (std::getline(in, line) && static_cast<int>(positions.size()) < count) {
      const auto battle = open_spiel::json::FromString(line);
      if (!battle) continue;
      std::unique_ptr<State> s = game->NewInitialState();
      int ply = 0;
      for (const auto& a : battle->GetObject().at("actions").GetArray()) {
        if (s->IsTerminal()) break;
        if (ply >= 10 && ply % 7 == 3 && static_cast<int>(positions.size()) < count) {
          positions.push_back(s->Clone());
        }
        s->ApplyAction(s->StringToAction(a.GetString()));
        ++ply;
      }
    }
  }
  if (positions.empty()) open_spiel::SpielFatalError("No positions in " + games_path);

  std::ofstream each;
  if (!each_path.empty()) each.open(each_path);
  std::mutex each_mutex;

  DeviceManager reference_devices;
  auto reference = Evaluator(*game, dir, step, "float32", threads, &reference_devices);
  VPNetModel reference_model(*game, dir, "vpnet.pb", "/cpu:0");
  reference_model.LoadCheckpoint(step);

  // The control: float32 against float32 with another seed.
  std::vector<Searched> base(positions.size()), again(positions.size());
  {
    std::atomic<int> next(0);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&]() {
        for (int i = next++; i < static_cast<int>(positions.size()); i = next++) {
          base[i] = Search(*game, reference, *positions[i], sims, seed + i);
          again[i] = Search(*game, reference, *positions[i], sims, seed + i + 100000);
        }
      });
    }
    for (auto& w : workers) w.join();
  }
  auto report_search = [&](const std::string& name, const std::vector<Searched>& other) {
    int same = 0;
    std::vector<double> visit_gaps, value_gaps;
    int near = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
      if (other[i].move == base[i].move) {
        ++same;
        continue;
      }
      const auto& mine = base[i].children.at(base[i].move);
      const auto it = base[i].children.find(other[i].move);
      const double visits = it == base[i].children.end() ? 0 : it->second.first;
      const double value = it == base[i].children.end() ? NAN : it->second.second;
      visit_gaps.push_back(mine.first - visits);
      if (!std::isnan(value)) value_gaps.push_back(std::abs(mine.second - value));
      if (mine.first - visits < 0.1 && !std::isnan(value) && std::abs(mine.second - value) < 0.01) {
        ++near;
      }
      if (each.is_open()) {
        std::lock_guard<std::mutex> lock(each_mutex);
        each << absl::StrFormat(
                    "{\"against\": \"%s\", \"position\": %d, \"float32_move\": \"%s\", "
                    "\"other_move\": \"%s\", \"float32_visits\": [%.4f, %.4f], "
                    "\"float32_values\": [%.4f, %.4f]}",
                    name, i, positions[i]->ActionToString(base[i].move),
                    positions[i]->ActionToString(other[i].move), mine.first, visits, mine.second,
                    value)
             << std::endl;
      }
    }
    const int differ = positions.size() - same;
    std::cout << absl::StrFormat(
                     "{\"part\": \"searches\", \"against\": \"%s\", \"positions\": %d, "
                     "\"same_move\": %d, \"same_share\": %.4f, \"different\": %d, "
                     "\"near_ties\": %d, \"visit_gap_median\": %.4f, \"visit_gap_p90\": %.4f, "
                     "\"value_gap_median\": %.4f, \"value_gap_p90\": %.4f}",
                     name, positions.size(), same, static_cast<double>(same) / positions.size(),
                     differ, near, Quantile(visit_gaps, 0.5), Quantile(visit_gaps, 0.9),
                     Quantile(value_gaps, 0.5), Quantile(value_gaps, 0.9))
              << std::endl;
  };
  report_search("float32, another seed (the control)", again);

  for (const std::string& precision : precisions) {
    // 1. The outputs, every position alone.
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    model.LoadCheckpoint(step);
    model.SetInferencePrecision(precision);
    double value_sum = 0, value_max = 0, tv_sum = 0, tv_max = 0;
    int top_same = 0;
    std::vector<double> prior_gaps;
    for (const auto& s : positions) {
      const VPNetModel::InferenceInputs in{s->LegalActions(), s->ObservationTensor()};
      const auto a = reference_model.Inference({in})[0];
      const auto b = model.Inference({in})[0];
      const double dv = std::abs(a.value - b.value);
      value_sum += dv;
      value_max = std::max(value_max, dv);
      double tv = 0;
      size_t top_a = 0, top_b = 0;
      for (size_t j = 0; j < a.policy.size(); ++j) {
        tv += std::abs(a.policy[j].second - b.policy[j].second) / 2;
        if (a.policy[j].second > a.policy[top_a].second) top_a = j;
        if (b.policy[j].second > b.policy[top_b].second) top_b = j;
      }
      tv_sum += tv;
      tv_max = std::max(tv_max, tv);
      if (top_a == top_b) {
        ++top_same;
      } else {
        prior_gaps.push_back(a.policy[top_a].second - a.policy[top_b].second);
      }
    }
    const int n = positions.size();
    std::cout << absl::StrFormat(
                     "{\"part\": \"outputs\", \"precision\": \"%s\", \"positions\": %d, "
                     "\"value_diff_mean\": %.5f, \"value_diff_max\": %.5f, "
                     "\"prior_tv_mean\": %.5f, \"prior_tv_max\": %.5f, \"top_move_same\": %.4f, "
                     "\"prior_gap_where_different_median\": %.5f, "
                     "\"prior_gap_where_different_max\": %.5f}",
                     precision, n, value_sum / n, value_max, tv_sum / n, tv_max,
                     static_cast<double>(top_same) / n, Quantile(prior_gaps, 0.5),
                     Quantile(prior_gaps, 1.0))
              << std::endl;

    // 2. The searches, the same seeds as float32's.
    DeviceManager devices;
    auto eval = Evaluator(*game, dir, step, precision, threads, &devices);
    std::vector<Searched> other(positions.size());
    std::atomic<int> next(0);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&]() {
        for (int i = next++; i < static_cast<int>(positions.size()); i = next++) {
          other[i] = Search(*game, eval, *positions[i], sims, seed + i);
        }
      });
    }
    for (auto& w : workers) w.join();
    report_search(precision, other);
  }
  return 0;
}
