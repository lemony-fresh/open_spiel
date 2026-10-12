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

// Does a network see a capture at the end of a long hurl or shove? (thud/PLAN.md, input
// features from the rules: the long-capture probe; the user asked, 2026-10-10, to
// probe the longest hurls and shoves across the board, in every direction and in many
// places, to see whether the network represents them accurately.)
//
// Built positions, in minimal pairs. A *hurl* of k squares: a line of k dwarfs in
// direction d (the hurled dwarf at its front), k - 1 empty squares, then a troll; a
// *shove* of k squares (k >= 2): a line of k trolls, k - 1 empty squares, the landing
// square, and a dwarf beyond it (next to the landing square only). Around them, other
// pieces at random — none next to the line, its path, its target or the square behind
// the line; the extra trolls three squares or more from every dwarf and not next to
// each other, then the extra dwarfs three or more from every troll, so that no step,
// lone hurl or shove captures — and the position is kept only if that hurl or shove is
// the side to move's one capture. Its twin: the same position with the line's last piece removed, so
// that the line is one too short and the capture illegal (and kept only if it then has
// no capture at all). k runs to 7, the longest the board allows (2k squares in a
// line, and the Thudstone blocks the middle row and column), in all 8 directions and
// at up to `places` positions of the line on the board, each with `backgrounds`
// random surroundings. The dwarfs move in hurl positions, the trolls in shove ones.
//
// For each network, from its policy and value (no search):
//   p          the move's probability (the prior over the legal moves);
//   top        whether it is the likeliest move; rank, its rank (1 = first);
//   uniform    1 / the legal moves — what a prior blind to the capture gives;
//   gain       log p(move | positive) - log p(move | twin), the move made legal in the
//              twin for the network's eyes only (its mask) — how much more the
//              network likes the move when it captures; 0 if it cannot tell;
//   value      the side to move's value, positive minus twin (a troll is 4/32 of a
//              battle's return to the dwarfs, a dwarf 1/32 to the trolls).
// Control: an untrained network of the same shape (`random` as the step) — gain ~0
// (a little below: the twin has one piece fewer, so fewer legal moves).
//
// `logs=RUN_DIR`: also the captures in real positions — every capture of the side to
// move in the positions of each logged actor's last `last` self-play games: hurls, and
// troll moves of 2 or more squares (shoves) and of 1 (steps), by distance, with p, top,
// rank and uniform as above (no twin) — the same measure where the network trained,
// to tell whether the built positions, emptier and more evenly spread than real ones,
// mislead it.
//
//   thud/az/build.sh thud/experiments/az_long_capture_probe.cc
//   OMP_NUM_THREADS=4 build-shared/az_long_capture_probe \
//     nets=NAME=DIR:STEP,NAME=DIR:random,... [places=12 backgrounds=4 dwarfs=16 \
//     trolls=3 seed=1 show=0 logs=RUN_DIR last=3 teacher=0]
//
// `teacher=1`: for real hurls and shoves (k >= 2), also kickstarting's teacher's view
// (AlphaBetaPlayer::Scores at depth 2, in points): the move's score less the best
// legal move's (`teacher_gap`, 0 = among the best) — are these captures good moves?
//
// `show=N`: also prints the positions of every Nth case, with their twins, to stderr.
//
// Prints one JSON line per position and network, then summaries per network, kind and
// distance, and per network, kind, distance and direction (`summary`).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/alphabeta_player.h"
#include "thud/az/vpnet.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
using open_spiel::thud_az::torch_az::VPNetModel;
namespace thud = open_spiel::thud;
constexpr int kN = thud::kBoardSize;
const char* kDirections[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};

bool Placeable(int r, int c) {
  return thud::IsOnBoard(r, c) && !(r == thud::kThudstoneRow && c == thud::kThudstoneCol);
}

// A board as rows of characters ('-', 'O', '.', 'd', 'T').
struct Board {
  std::vector<std::string> rows;
  Board() {
    for (int r = 0; r < kN; ++r) {
      std::string row;
      for (int c = 0; c < kN; ++c) {
        row += !thud::IsOnBoard(r, c) ? '-' : !Placeable(r, c) ? 'O' : '.';
      }
      rows.push_back(row);
    }
  }
  char& at(int r, int c) { return rows[r][c]; }
  std::string Text(bool trolls_move) const {
    std::string text;
    for (const std::string& row : rows) text += row + "\n";
    return text + absl::StrCat("to_move=", trolls_move ? "trolls" : "dwarfs",
                               " turns=100 turns_without_capture=4\n");
  }
};

struct Sq {
  int r, c;
};

// The squares of one hurl or shove of `k` squares in direction `d` whose moving
// piece stands on `front`: the line (front first), the path, the target (the troll
// hurled onto, or the landing square), and for a shove the dwarf beyond it; the
// square behind the line last. Empty if any of them is off the board or the stone.
struct Pattern {
  std::vector<Sq> line, path;
  Sq target, dwarf_beyond, behind;
};

bool MakePattern(bool shove, int d, int k, Sq front, Pattern* p) {
  const int dr = thud::kRowStep[d], dc = thud::kColStep[d];
  auto ok = [](Sq s) { return Placeable(s.r, s.c); };
  p->line.clear();
  p->path.clear();
  for (int i = 0; i < k; ++i) {
    p->line.push_back({front.r - i * dr, front.c - i * dc});
    if (!ok(p->line.back())) return false;
  }
  for (int j = 1; j < k; ++j) {
    p->path.push_back({front.r + j * dr, front.c + j * dc});
    if (!ok(p->path.back())) return false;
  }
  p->target = {front.r + k * dr, front.c + k * dc};
  if (!ok(p->target)) return false;
  if (shove) {
    p->dwarf_beyond = {p->target.r + dr, p->target.c + dc};
    if (!ok(p->dwarf_beyond)) return false;
  }
  p->behind = {front.r - k * dr, front.c - k * dc};  // May be off the board.
  return true;
}

// A number for JSON: null if NaN.
std::string Json(double x, int digits) {
  return std::isnan(x) ? "null" : absl::StrFormat("%.*f", digits, x);
}

double Mean(const std::vector<double>& v) {
  return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double Median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

struct Case {
  std::string kind;  // "hurl", "shove" or "step".
  int d, k;
  std::unique_ptr<State> positive, twin;  // twin: null for k == 1 (no shorter line).
  Action move;
  bool natural;  // From a real game.
};

Action Find(const State& state, const std::string& move) {
  for (Action a : state.LegalActions()) {
    if (state.ActionToString(a) == move) return a;
  }
  open_spiel::SpielFatalError("No legal action " + move);
}

struct Stats {
  std::vector<double> p, top, rank, uniform, gain, value;
};

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
  const std::string nets = get("nets", "");
  const int places = std::stoi(get("places", "12")), backgrounds = std::stoi(get("backgrounds", "4"));
  const int extra_dwarfs = std::stoi(get("dwarfs", "16")), extra_trolls = std::stoi(get("trolls", "3"));
  std::mt19937 rng(std::stoull(get("seed", "1")));
  const bool use_teacher = get("teacher", "0") == "1";
  const int show = std::stoi(get("show", "0"));
  if (nets.empty()) {
    std::cerr << "Usage: az_long_capture_probe nets=NAME=DIR:STEP,... [...]" << std::endl;
    return 1;
  }
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // The cases.
  std::vector<Case> cases;
  int rejected = 0;
  for (bool shove : {false, true}) {
    for (int k = shove ? 2 : 1; k <= 7; ++k) {
      for (int d = 0; d < thud::kNumDirections; ++d) {
        std::vector<Sq> fronts;
        Pattern p;
        for (int r = 0; r < kN; ++r) {
          for (int c = 0; c < kN; ++c) {
            if (MakePattern(shove, d, k, {r, c}, &p)) fronts.push_back({r, c});
          }
        }
        for (int i = 0; i < fronts.size(); ++i) {
          if (fronts.size() > places &&
              i * places / fronts.size() == (i + 1) * places / fronts.size()) {
            continue;  // An even sample of the places.
          }
          MakePattern(shove, d, k, fronts[i], &p);
          for (int b = 0; b < backgrounds; ++b) {
            for (int attempt = 0; attempt < 50; ++attempt) {
              Board board;
              const char mover = shove ? 'T' : 'd';
              for (const Sq& s : p.line) board.at(s.r, s.c) = mover;
              board.at(p.target.r, p.target.c) = shove ? '.' : 'T';
              if (shove) board.at(p.dwarf_beyond.r, p.dwarf_beyond.c) = 'd';
              // Squares the surroundings keep clear: the pattern, its neighbours, the
              // square behind the line.
              std::set<std::pair<int, int>> clear;
              std::vector<Sq> all = p.line;
              all.insert(all.end(), p.path.begin(), p.path.end());
              all.push_back(p.target);
              if (shove) all.push_back(p.dwarf_beyond);
              for (const Sq& s : all) {
                for (int rr = -1; rr <= 1; ++rr) {
                  for (int cc = -1; cc <= 1; ++cc) clear.insert({s.r + rr, s.c + cc});
                }
              }
              clear.insert({p.behind.r, p.behind.c});
              // The nearest piece of a kind, in king moves.
              auto nearest = [&](int r, int c, char kind) {
                int best = 99;
                for (int rr = 0; rr < kN; ++rr) {
                  for (int cc = 0; cc < kN; ++cc) {
                    if (board.at(rr, cc) == kind) {
                      best = std::min(best, std::max(std::abs(rr - r), std::abs(cc - c)));
                    }
                  }
                }
                return best;
              };
              auto place = [&](char kind, int count) {
                for (int j = 0; j < count; ++j) {
                  std::vector<Sq> free;
                  for (int r = 0; r < kN; ++r) {
                    for (int c = 0; c < kN; ++c) {
                      if (!Placeable(r, c) || board.at(r, c) != '.' || clear.count({r, c})) {
                        continue;
                      }
                      if (kind == 'T' && (nearest(r, c, 'd') < 3 || nearest(r, c, 'T') < 2)) {
                        continue;
                      }
                      if (kind == 'd' && nearest(r, c, 'T') < 3) continue;
                      free.push_back({r, c});
                    }
                  }
                  if (free.empty()) return;
                  const Sq s = free[std::uniform_int_distribution<int>(0, free.size() - 1)(rng)];
                  board.at(s.r, s.c) = kind;
                }
              };
              const int trolls_here = shove ? k : 1;
              place('T', std::min(extra_trolls, 8 - trolls_here));
              place('d', extra_dwarfs);
              std::unique_ptr<State> positive = game->NewInitialState(board.Text(shove));
              const std::string wanted =
                  absl::StrFormat("(%d,%d)-(%d,%d)x", p.line[0].r, p.line[0].c, p.target.r,
                                  p.target.c);
              Action move = open_spiel::kInvalidAction;
              int captures = 0;
              for (Action a : positive->LegalActions()) {
                const std::string s = positive->ActionToString(a);
                if (absl::EndsWith(s, "x")) {
                  ++captures;
                  if (s == wanted) move = a;
                }
              }
              if (move == open_spiel::kInvalidAction || captures != 1) {
                ++rejected;
                continue;
              }
              std::unique_ptr<State> twin;
              if (k > 1) {
                board.at(p.line.back().r, p.line.back().c) = '.';
                twin = game->NewInitialState(board.Text(shove));
                bool any_capture = false, move_legal = false;
                for (Action a : twin->LegalActions()) {
                  any_capture |= absl::EndsWith(twin->ActionToString(a), "x");
                  move_legal |= a == move;
                }
                if (any_capture || move_legal) {
                  ++rejected;
                  continue;
                }
              }
              if (show > 0 && cases.size() % show == 0) {
                std::cerr << absl::StrFormat("case %d: %s of %d to the %s, %s\n",
                                             cases.size(), shove ? "shove" : "hurl", k,
                                             kDirections[d], positive->ActionToString(move))
                          << positive->ToString() << "\ntwin:\n"
                          << (twin ? twin->ToString() : std::string("none")) << "\n\n";
              }
              cases.push_back({shove ? "shove" : "hurl", d, k, std::move(positive),
                               std::move(twin), move, false});
              break;
            }
          }
        }
      }
    }
  }
  std::cerr << absl::StrFormat("%d cases (%d candidate positions rejected)\n", cases.size(),
                               rejected);

  // Real positions.
  const std::string logs = get("logs", "");
  if (!logs.empty()) {
    const int last = std::stoi(get("last", "3"));
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(logs)) {
      const std::string name = entry.path().filename().string();
      if (absl::StartsWith(name, "log-actor-")) paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    int natural = 0;
    for (const std::string& path : paths) {
      std::vector<std::string> lines;
      std::ifstream in(path);
      for (std::string line; std::getline(in, line);) {
        if (absl::StrContains(line, "Actions: ") && !absl::StrContains(line, "Opponent:")) {
          lines.push_back(line);
        }
      }
      for (int i = std::max<int>(0, lines.size() - last); i < lines.size(); ++i) {
        const std::string actions = lines[i].substr(lines[i].find("Actions: ") + 9);
        std::unique_ptr<State> state = game->NewInitialState();
        for (absl::string_view move : absl::StrSplit(actions, ' ', absl::SkipEmpty())) {
          if (state->IsTerminal()) break;
          const bool trolls = state->CurrentPlayer() == thud::kTrollPlayer;
          for (Action a : state->LegalActions()) {
            const std::string text = state->ActionToString(a);
            if (!absl::EndsWith(text, "x")) continue;
            int r0, c0, r1, c1;
            SPIEL_CHECK_EQ(std::sscanf(text.c_str(), "(%d,%d)-(%d,%d)", &r0, &c0, &r1, &c1), 4);
            const int k = std::max(std::abs(r1 - r0), std::abs(c1 - c0));
            int d = 0;
            while (thud::kRowStep[d] != (r1 > r0) - (r1 < r0) ||
                   thud::kColStep[d] != (c1 > c0) - (c1 < c0)) {
              ++d;
            }
            const std::string kind = !trolls ? "hurl" : k >= 2 ? "shove" : "step";
            cases.push_back({kind, d, k, state->Clone(), nullptr, a, true});
            ++natural;
          }
          state->ApplyAction(Find(*state, std::string(move)));
        }
      }
    }
    std::cerr << absl::StrFormat("%d captures in real positions\n", natural);
  }

  for (absl::string_view spec : absl::StrSplit(nets, ',', absl::SkipEmpty())) {
    const size_t eq = spec.find('='), colon = spec.rfind(':');
    const std::string name(spec.substr(0, eq));
    const std::string dir(spec.substr(eq + 1, colon - eq - 1));
    const std::string step(spec.substr(colon + 1));
    VPNetModel model(*game, dir, "vpnet.pb", "/cpu:0");
    if (step != "random") model.LoadCheckpoint(std::stoi(step));
    // (natural, kind, k, d), d = -1: all directions.
    std::map<std::tuple<bool, std::string, int, int>, Stats> by_k;
    const open_spiel::thud_az::AlphaBetaPlayer teacher(*game, 2);
    for (const Case& c : cases) {
      double teacher_gap = NAN;
      if (use_teacher && c.natural && c.kind != "step" && c.k >= 2) {
        const std::vector<Action> legal = c.positive->LegalActions();
        const std::vector<double> scores = teacher.Scores(*c.positive);
        const double best = *std::max_element(scores.begin(), scores.end());
        for (int i = 0; i < legal.size(); ++i) {
          if (legal[i] == c.move) teacher_gap = scores[i] - best;
        }
      }
      std::vector<VPNetModel::InferenceInputs> in = {
          {c.positive->LegalActions(), c.positive->ObservationTensor()}};
      if (c.twin) {
        std::vector<Action> legal = c.twin->LegalActions();
        legal.push_back(c.move);  // For the network's eyes only.
        std::sort(legal.begin(), legal.end());
        in.push_back({legal, c.twin->ObservationTensor()});
      }
      const std::vector<VPNetModel::InferenceOutputs> out = model.Inference(in);
      double p = 0;
      int rank = 1;
      for (const auto& [a, q] : out[0].policy) {
        if (a == c.move) p = q;
      }
      for (const auto& [a, q] : out[0].policy) rank += a != c.move && q > p;
      const double sign = c.kind == "hurl" ? 1 : -1;  // The value is the dwarfs'.
      double gain = NAN, value = NAN;
      if (c.twin) {
        double p_twin = 0;
        for (const auto& [a, q] : out[1].policy) {
          if (a == c.move) p_twin = q;
        }
        gain = std::log(p) - std::log(p_twin);
        value = sign * (out[0].value - out[1].value);
      }
      const double uniform = 1.0 / out[0].policy.size();
      std::cout << absl::StrFormat(
                       "{\"net\": \"%s\", \"source\": \"%s\", \"kind\": \"%s\", \"k\": %d, "
                       "\"direction\": \"%s\", \"move\": \"%s\", \"p\": %.5f, \"rank\": %d, "
                       "\"legal\": %d, \"gain\": %s, \"value\": %s, \"teacher_gap\": %s}",
                       name, c.natural ? "real" : "built", c.kind, c.k, kDirections[c.d],
                       c.positive->ActionToString(c.move), p, rank, out[0].policy.size(),
                       Json(gain, 4), Json(value, 4), Json(teacher_gap, 2))
                << std::endl;
      for (int d : {-1, c.d}) {
        Stats& s = by_k[{c.natural, c.kind, c.k, d}];
        s.p.push_back(p);
        s.top.push_back(rank == 1);
        s.rank.push_back(rank);
        s.uniform.push_back(uniform);
        if (c.twin) {
          s.gain.push_back(gain);
          s.value.push_back(value);
        }
      }
    }
    for (const auto& [key, s] : by_k) {
      const auto& [natural, kind, k, d] = key;
      std::cout << absl::StrFormat(
                       "{\"summary\": true, \"net\": \"%s\", \"source\": \"%s\", "
                       "\"kind\": \"%s\", \"k\": %d, \"direction\": \"%s\", \"cases\": %d, "
                       "\"p\": %.4f, \"uniform\": %.4f, \"top\": %.3f, \"rank_median\": %.1f, "
                       "\"gain_mean\": %s, \"gain_median\": %s, \"value_mean\": %s}",
                       name, natural ? "real" : "built", kind, k,
                       d < 0 ? "all" : kDirections[d],
                       s.p.size(), Mean(s.p), Mean(s.uniform), Mean(s.top), Median(s.rank),
                       Json(Mean(s.gain), 3), Json(Median(s.gain), 3), Json(Mean(s.value), 4))
                << std::endl;
    }
  }
  return 0;
}
