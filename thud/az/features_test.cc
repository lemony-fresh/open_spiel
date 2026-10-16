// Copyright 2026 The Thud-on-OpenSpiel authors.
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

// Tests the input planes from the rules (thud/az/thud_features.h; the planes decided with
// the user on 2026-10-11: 50 — 48 move planes and 2 maps) — written before the planes'
// code, as the user asked:
//
//   0. hand-built positions, each plane worked out by hand: the longest hurl (7, east)
//      and the longest diagonal hurl (5); a line one dwarf short; a path blocked by a
//      dwarf and by the Thudstone; two trolls in a line's path (only the first can be
//      hit); a lone dwarf next to a troll (both sides at once); a troll line shoving at
//      three distances, one of them taking two dwarfs; the longest shove (7); two
//      captures in one position; and the two maps, by hand, in two small positions.
//      The move planes are compared in full, the maps where the test gives them;
//   1. the move planes against the game's own move generator, both sides (the board
//      read with each side to move): every capturing action has its value in plane
//      (direction, distance) at its square — 1 for a hurl, the dwarfs it removes over 8
//      for a troll's move — and every nonzero entry is a capturing action; no capture
//      is longer than MaxCaptureDistance (7 straight, 5 diagonal);
//   2. the maps: on dwarfs, kTrollLandsNear exactly where some troll move would take
//      the dwarf; on trolls, kHurlLands exactly where some hurl would land; on empty
//      squares, against a piece placed there (a dwarf: would a troll move take it? a
//      troll: would a hurl land on it?) — judged on the board as it is, the placed
//      piece blocking no path, so a few disagreements are expected, reported and
//      required to be rare (under 1%);
//   3. symmetry: rotating (or reflecting) the board and then computing the planes gives
//      the same as computing them and then turning them — the move planes' directions
//      moved by SymmetricDirection, their distances kept — on all 7 symmetries other
//      than the identity, the hand-built positions included;
//   4. controls, each required to fail: a hand-built test with one entry wrong (part
//      0), two distance planes swapped (part 1), and the planes turned without moving
//      their directions (part 3);
//   5. every plane the board allows: a position with one capture of each kind,
//      direction and distance the board has room for (thud/az/capture_patterns.h) lights
//      that move plane at that piece — and joins parts 1-3; and the board has room for
//      exactly the distances the planes keep (up to 7 straight, 5 diagonal) and none of
//      those left out (6 and 7 on the diagonals), for hurls and troll moves alike — the
//      reason the 8 planes are left out, checked.
//
// Positions: the hand-built ones, the patterns of part 5, random games (every 7th ply)
// and, with LOG_DIR, run self-play (every 5th ply of each actor's last game).
//
//   thud/run_tests.sh features_test   # builds and runs it
//   build-tests/bin/features_test [LOG_DIR]

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/thud/thud.h"
#include "open_spiel/spiel.h"
#include "thud/az/capture_patterns.h"
#include "thud/az/thud_features.h"

namespace {

using open_spiel::Action;
using open_spiel::State;
namespace thud = open_spiel::thud;
namespace az = open_spiel::thud_az;
constexpr int n = thud::kBoardSize;
enum Direction { N, NE, E, SE, S, SW, W, NW };  // thud.h's order.

int failures = 0;
void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  failures += !ok;
}

std::vector<float> Features(const State& state) {
  std::vector<float> features;
  az::AppendFeaturePlanes(state.ObservationTensor(0), &features);
  return features;
}
float& At(std::vector<float>& f, int plane, int r, int c) { return f[(plane * n + r) * n + c]; }

// A board's text: the given pieces on an otherwise empty board.
struct Piece {
  char kind;  // 'd' or 'T'.
  int r, c;
};
std::string BoardText(const std::vector<Piece>& pieces) {
  std::string text;
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      text += !thud::IsOnBoard(r, c)                                   ? '-'
              : r == thud::kThudstoneRow && c == thud::kThudstoneCol ? 'O'
                                                                       : '.';
    }
    text += '\n';
  }
  for (const Piece& p : pieces) {
    SPIEL_CHECK_TRUE(thud::IsOnBoard(p.r, p.c));
    SPIEL_CHECK_EQ(text[p.r * (n + 1) + p.c], '.');
    text[p.r * (n + 1) + p.c] = p.kind;
  }
  return text;
}
std::vector<Piece> Row(char kind, int r, int c0, int c1) {
  std::vector<Piece> out;
  for (int c = c0; c <= c1; ++c) out.push_back({kind, r, c});
  return out;
}

// The position's board with `to_move` to move (no counters, so no limit applies).
std::unique_ptr<State> WithToMove(const open_spiel::Game& game, const State& state,
                                  const std::string& to_move) {
  std::string text = state.ToString();
  text.resize(text.rfind('\n') + 1);
  return game.NewInitialState(text + "to_move=" + to_move);
}

struct Move {
  int r1, c1, r2, c2, d, k;
  bool capture;
};
Move Parse(const std::string& text) {
  Move m;
  m.capture = absl::EndsWith(text, "x");
  std::sscanf(text.c_str(), "(%d,%d)-(%d,%d)", &m.r1, &m.c1, &m.r2, &m.c2);
  const int dr = (m.r2 > m.r1) - (m.r2 < m.r1), dc = (m.c2 > m.c1) - (m.c2 < m.c1);
  for (int d = 0; d < thud::kNumDirections; ++d) {
    if (thud::kRowStep[d] == dr && thud::kColStep[d] == dc) m.d = d;
  }
  m.k = std::max(std::abs(m.r2 - m.r1), std::abs(m.c2 - m.c1));
  return m;
}

std::set<std::pair<int, int>> Squares(const State& state, thud::Cell kind) {
  std::set<std::pair<int, int>> out;
  const auto& t = open_spiel::down_cast<const thud::ThudState&>(state);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      if (thud::IsOnBoard(r, c) && t.CellAt(r, c) == kind) out.insert({r, c});
    }
  }
  return out;
}

// From the move generator: the move planes, and the maps on pieces (takeable dwarfs on
// kTrollLandsNear, hurled-onto trolls on kHurlLands); `too_long` counts captures beyond
// MaxCaptureDistance (7 straight, 5 diagonal).
std::vector<float> Expected(const open_spiel::Game& game, const State& state, int* too_long) {
  std::vector<float> f(az::kNumFeaturePlanes * n * n, 0.0f);
  std::unique_ptr<State> trolls = WithToMove(game, state, "trolls");
  const auto dwarfs_before = Squares(*trolls, thud::Cell::kDwarf);
  for (Action a : trolls->LegalActions()) {
    const Move m = Parse(trolls->ActionToString(a));
    if (!m.capture) continue;
    if (m.k > az::MaxCaptureDistance(m.d)) {
      ++*too_long;
      continue;
    }
    const auto after = Squares(*trolls->Child(a), thud::Cell::kDwarf);
    int taken = 0;
    for (const auto& [r, c] : dwarfs_before) {
      if (!after.count({r, c})) {
        At(f, az::kTrollLandsNear, r, c) = 1;
        ++taken;
      }
    }
    At(f, az::MovePlane(m.d, m.k), m.r1, m.c1) = static_cast<float>(taken) / 8;
  }
  std::unique_ptr<State> dwarfs = WithToMove(game, state, "dwarfs");
  for (Action a : dwarfs->LegalActions()) {
    const Move m = Parse(dwarfs->ActionToString(a));
    if (!m.capture) continue;
    if (m.k > az::MaxCaptureDistance(m.d)) {
      ++*too_long;
      continue;
    }
    At(f, az::MovePlane(m.d, m.k), m.r1, m.c1) = 1;
    At(f, az::kHurlLands, m.r2, m.c2) = 1;
  }
  return f;
}

// The squares where planes `planes` of a and b differ.
int Differences(const std::vector<float>& a, const std::vector<float>& b,
                const std::vector<int>& planes) {
  int differ = 0;
  for (int p : planes) {
    for (int i = 0; i < n * n; ++i) differ += std::abs(a[p * n * n + i] - b[p * n * n + i]) > 1e-6;
  }
  return differ;
}

std::vector<int> MovePlanes() {
  std::vector<int> out(az::kNumMovePlanes);
  for (int p = 0; p < az::kNumMovePlanes; ++p) out[p] = p;
  return out;
}
std::vector<int> AllPlanes() {
  std::vector<int> out(az::kNumFeaturePlanes);
  for (int p = 0; p < az::kNumFeaturePlanes; ++p) out[p] = p;
  return out;
}

// The planes turned by `symmetry`, the move planes' directions moved with `direction`.
std::vector<float> Turned(const std::vector<float>& f, int symmetry,
                          int (*direction)(int, int)) {
  std::vector<float> out(f.size(), 0.0f);
  for (int p = 0; p < az::kNumFeaturePlanes; ++p) {
    int q = p;
    if (p < az::kNumMovePlanes) {
      const az::PlaneMove m = az::MoveOfPlane(p);
      q = az::MovePlane(direction(m.d, symmetry), m.k);
    }
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        if (!thud::IsOnBoard(r, c)) continue;
        const thud::Coord to = thud::SymmetricCoord({r, c}, symmetry);
        out[(q * n + to.row) * n + to.col] = f[(p * n + r) * n + c];
      }
    }
  }
  return out;
}
int Identity(int direction, int) { return direction; }

// 0. A hand-built position and its planes worked out by hand.
struct Entry {
  int plane, r, c;
  float value;
};
struct HandTest {
  std::string name;
  std::vector<Piece> pieces;
  std::vector<Entry> moves;  // Every nonzero move-plane entry.
  bool maps = false;         // Whether `map_entries` gives the maps in full.
  std::vector<Entry> map_entries;
};

// The squares where a hand test's planes differ from its hand-worked ones.
int HandDifferences(const open_spiel::Game& game, const HandTest& t) {
  std::unique_ptr<State> s = game.NewInitialState(BoardText(t.pieces) + "to_move=dwarfs");
  std::vector<float> f = Features(*s);
  std::vector<float> e(f.size(), 0.0f);
  for (const Entry& m : t.moves) At(e, m.plane, m.r, m.c) = m.value;
  for (const Entry& m : t.map_entries) At(e, m.plane, m.r, m.c) = m.value;
  int differ = Differences(f, e, MovePlanes());
  if (t.maps) differ += Differences(f, e, {az::kTrollLandsNear, az::kHurlLands});
  return differ;
}

std::vector<HandTest> HandTests() {
  using az::MovePlane;
  std::vector<HandTest> tests;
  auto join = [](std::vector<std::vector<Piece>> parts) {
    std::vector<Piece> out;
    for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
  };
  // Row 5 runs the board's full width, columns 0-14.
  tests.push_back({"the longest hurl: 7 dwarfs, 6 empty squares, a troll (east)",
                   join({Row('d', 5, 0, 6), {{'T', 5, 13}}}),
                   {{MovePlane(E, 7), 5, 6, 1}}});
  tests.push_back({"a line one dwarf short: 6 dwarfs, a troll 7 squares away",
                   join({Row('d', 5, 1, 6), {{'T', 5, 13}}}), {}});
  tests.push_back({"a path blocked by a dwarf",
                   join({Row('d', 5, 0, 6), {{'d', 5, 9}, {'T', 5, 13}}}), {}});
  tests.push_back({"two trolls in the path: only the first is hit (3, not 7)",
                   join({Row('d', 5, 0, 6), {{'T', 5, 9}, {'T', 5, 13}}}),
                   {{MovePlane(E, 3), 5, 6, 1}}});
  tests.push_back({"a path blocked by the Thudstone: 5 dwarfs, the troll 4 squares away",
                   join({Row('d', 7, 0, 4), {{'T', 7, 8}}}), {}});
  // The diagonal (0,5)-(9,14) holds 10 squares: room for a hurl of 5.
  tests.push_back({"the longest diagonal hurl: 5 dwarfs, 4 empty squares, a troll (south-east)",
                   {{'d', 0, 5}, {'d', 1, 6}, {'d', 2, 7}, {'d', 3, 8}, {'d', 4, 9},
                    {'T', 9, 14}},
                   {{MovePlane(SE, 5), 4, 9, 1}}});
  // The troll steps to four squares next to the dwarf; the dwarf hurls 1 onto it.
  tests.push_back({"a lone dwarf next to a troll: both sides",
                   {{'d', 9, 9}, {'T', 8, 9}},
                   {{MovePlane(N, 1), 9, 9, 1},
                    {MovePlane(W, 1), 8, 9, 0.125f},
                    {MovePlane(E, 1), 8, 9, 0.125f},
                    {MovePlane(SW, 1), 8, 9, 0.125f},
                    {MovePlane(SE, 1), 8, 9, 0.125f}}});
  // 4 trolls in line: shoves of 2, 3 and 4 land at (5,5), (5,6), (5,7); the dwarf at
  // (4,6) is next to all three, the dwarf at (6,8) next to (5,7) only.
  tests.push_back({"a troll line shoving at three distances, one taking two dwarfs",
                   join({Row('T', 5, 0, 3), {{'d', 4, 6}, {'d', 6, 8}}}),
                   {{MovePlane(E, 2), 5, 3, 0.125f},
                    {MovePlane(E, 3), 5, 3, 0.125f},
                    {MovePlane(E, 4), 5, 3, 0.25f}}});
  tests.push_back({"the longest shove: 7 trolls, landing at (5,13) next to the dwarf",
                   join({Row('T', 5, 0, 6), {{'d', 5, 14}}}),
                   {{MovePlane(E, 7), 5, 6, 0.125f}}});
  tests.push_back({"two captures at once: a hurl of 3 and a troll's steps",
                   join({Row('d', 9, 1, 3), {{'T', 9, 6}, {'T', 12, 6}, {'d', 12, 8}}}),
                   {{MovePlane(E, 3), 9, 3, 1},
                    {MovePlane(E, 1), 12, 6, 0.125f},
                    {MovePlane(NE, 1), 12, 6, 0.125f},
                    {MovePlane(SE, 1), 12, 6, 0.125f}}});
  // The maps. A lone troll's steps land on its 8 neighbours, so a dwarf would be taken
  // anywhere in the 5 x 5 block around it (the troll's own square aside); a lone
  // dwarf can be hurled 1 square any way, onto its 8 neighbours.
  {
    HandTest t{"the maps of a lone troll and a lone dwarf", {{'T', 9, 10}, {'d', 2, 6}}, {}};
    t.maps = true;
    for (int r = 7; r <= 11; ++r) {
      for (int c = 8; c <= 12; ++c) {
        if (r != 9 || c != 10) t.map_entries.push_back({az::kTrollLandsNear, r, c, 1});
      }
    }
    for (int r = 1; r <= 3; ++r) {
      for (int c = 5; c <= 7; ++c) {
        if (r != 2 || c != 6) t.map_entries.push_back({az::kHurlLands, r, c, 1});
      }
    }
    tests.push_back(t);
  }
  // 3 dwarfs in line on row 12 (columns 3-11 on the board): hurled 3 squares east from
  // (12,6) and west from (12,4) (only (12,3) on the board), 1 square any other way.
  {
    HandTest t{"the hurl map of a line of three", Row('d', 12, 4, 6), {}};
    t.maps = true;
    for (auto [r, c] : std::vector<std::pair<int, int>>{
             {12, 3}, {12, 7}, {12, 8}, {12, 9}, {11, 3}, {11, 4}, {11, 5}, {11, 6},
             {11, 7}, {13, 4}, {13, 5}, {13, 6}, {13, 7}}) {
      t.map_entries.push_back({az::kHurlLands, r, c, 1});
    }
    tests.push_back(t);
  }
  return tests;
}

}  // namespace

int main(int argc, char** argv) {
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("thud");

  // 0. The hand-built positions.
  const std::vector<HandTest> hand = HandTests();
  for (const HandTest& t : hand) {
    const int differ = HandDifferences(*game, t);
    Check(differ == 0, "0. " + t.name + (differ ? " (" + std::to_string(differ) + " squares differ)" : ""));
  }
  {
    HandTest wrong = hand[0];  // Control: the longest hurl written as 6, not 7.
    wrong.moves = {{az::MovePlane(E, 6), 5, 6, 1}};
    Check(HandDifferences(*game, wrong) > 0, "4. control: a hand-built test with one entry wrong fails");
  }

  std::vector<std::unique_ptr<State>> positions;
  for (const HandTest& t : hand) {
    positions.push_back(game->NewInitialState(BoardText(t.pieces) + "to_move=dwarfs"));
  }
  // 5. Every plane the board allows.
  {
    const char* kNames[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
    int lit = 0, wrong = 0, room_differs = 0;
    std::string no_room;
    for (bool troll : {false, true}) {
      for (int d = 0; d < thud::kNumDirections; ++d) {
        for (int k = 1; k <= az::kMaxCaptureDistance; ++k) {
          az::CapturePattern p;
          const bool room = az::FindCapturePattern(troll, d, k, &p);
          room_differs += room != (k <= az::MaxCaptureDistance(d));
          if (!room) {
            no_room += std::string(" ") + (troll ? "troll " : "hurl ") + kNames[d] +
                       std::to_string(k);
            continue;
          }
          positions.push_back(game->NewInitialState(p.board + "to_move=dwarfs"));
          std::vector<float> f = Features(*positions.back());
          const float value = At(f, az::MovePlane(d, k), p.row, p.col);
          ++lit;
          wrong += value != (troll ? 0.125f : 1.0f);
        }
      }
    }
    std::cout << "   no room on the board for:" << (no_room.empty() ? " none" : no_room)
              << std::endl;
    Check(wrong == 0 && lit > 0, "5. each of " + std::to_string(lit) +
                                     " patterns lights its move plane at its piece (" +
                                     std::to_string(wrong) + " wrong)");
    Check(room_differs == 0,
          "5. the board has room for exactly the distances the planes keep, for hurls and "
          "troll moves alike (" + std::to_string(room_differs) + " differ)");
  }
  // Random games, every 7th ply.
  std::mt19937 rng(20261008);
  for (int g = 0; g < 30; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    for (int ply = 0; !state->IsTerminal() && ply < 400; ++ply) {
      if (ply % 7 == 3) positions.push_back(state->Clone());
      const std::vector<Action> legal = state->LegalActions();
      state->ApplyAction(legal[rng() % legal.size()]);
    }
  }
  // Self-play, every 5th ply of each actor's last game.
  int real = 0;
  if (argc > 1) {
    for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
      if (!absl::StartsWith(entry.path().filename().string(), "log-actor-")) continue;
      std::string last;
      std::ifstream in(entry.path());
      for (std::string line; std::getline(in, line);) {
        if (absl::StrContains(line, "Actions: ")) last = line;
      }
      if (last.empty()) continue;
      std::unique_ptr<State> state = game->NewInitialState();
      int ply = 0;
      for (absl::string_view move :
           absl::StrSplit(last.substr(last.find("Actions: ") + 9), ' ', absl::SkipEmpty())) {
        if (state->IsTerminal()) break;
        if (ply++ % 5 == 2) {
          positions.push_back(state->Clone());
          ++real;
        }
        for (Action a : state->LegalActions()) {
          if (state->ActionToString(a) == move) {
            state->ApplyAction(a);
            break;
          }
        }
      }
    }
  }
  std::cout << positions.size() << " positions (" << hand.size()
            << " hand-built and the patterns, " << real << " from self-play)" << std::endl;

  int move_diff = 0, planted_diff = 0, too_long = 0, captures_seen = 0, longest = 0;
  int map_piece_diff = 0, map_pieces = 0;
  int danger_squares = 0, danger_disagree = 0, tdanger_squares = 0, tdanger_disagree = 0;
  int sym_diff = 0, sym_planted_diff = 0;
  for (const auto& state : positions) {
    const std::vector<float> f = Features(*state);
    const std::vector<float> e = Expected(*game, *state, &too_long);
    // 1. The move planes.
    move_diff += Differences(f, e, MovePlanes());
    for (int p = 0; p < az::kNumMovePlanes; ++p) {
      for (int i = 0; i < n * n; ++i) {
        if (e[p * n * n + i] > 0) {
          ++captures_seen;
          longest = std::max(longest, az::MoveOfPlane(p).k);
        }
      }
    }
    // Control: two distance planes swapped (east, 1 and 2).
    std::vector<float> planted = f;
    std::swap_ranges(planted.begin() + az::MovePlane(E, 1) * n * n,
                     planted.begin() + (az::MovePlane(E, 1) + 1) * n * n,
                     planted.begin() + az::MovePlane(E, 2) * n * n);
    planted_diff += Differences(planted, e, MovePlanes());
    // 2. The maps: exact on pieces, against a placed piece on empty squares.
    const auto dwarfs = Squares(*state, thud::Cell::kDwarf);
    const auto trolls = Squares(*state, thud::Cell::kTroll);
    for (const auto& [r, c] : dwarfs) {
      ++map_pieces;
      map_piece_diff += (f[(az::kTrollLandsNear * n + r) * n + c] > 0) !=
                        (e[(az::kTrollLandsNear * n + r) * n + c] > 0);
      map_piece_diff += f[(az::kHurlLands * n + r) * n + c] != 0;  // Never on a dwarf.
    }
    for (const auto& [r, c] : trolls) {
      ++map_pieces;
      map_piece_diff += (f[(az::kHurlLands * n + r) * n + c] > 0) !=
                        (e[(az::kHurlLands * n + r) * n + c] > 0);
      map_piece_diff += f[(az::kTrollLandsNear * n + r) * n + c] != 0;  // Never on a troll.
    }
    const std::string text = [&] {
      std::string t = state->ToString();
      t.resize(t.rfind('\n') + 1);
      return t;
    }();
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        if (!thud::IsOnBoard(r, c) || text[r * (n + 1) + c] != '.') continue;
        if (dwarfs.size() < thud::kNumDwarfs) {
          std::string placed = text;
          placed[r * (n + 1) + c] = 'd';
          std::unique_ptr<State> s = game->NewInitialState(placed + "to_move=trolls");
          bool taken = false;
          for (Action a : s->LegalActions()) {
            if (!absl::EndsWith(s->ActionToString(a), "x")) continue;
            if (open_spiel::down_cast<const thud::ThudState&>(*s->Child(a)).CellAt(r, c) !=
                thud::Cell::kDwarf) {
              taken = true;
              break;
            }
          }
          ++danger_squares;
          danger_disagree += taken != (f[(az::kTrollLandsNear * n + r) * n + c] > 0);
        }
        if (trolls.size() < thud::kNumTrolls) {
          std::string placed = text;
          placed[r * (n + 1) + c] = 'T';
          std::unique_ptr<State> s = game->NewInitialState(placed + "to_move=dwarfs");
          bool taken = false;
          for (Action a : s->LegalActions()) {
            const std::string m = s->ActionToString(a);
            if (absl::EndsWith(m, "x") && Parse(m).r2 == r && Parse(m).c2 == c) taken = true;
          }
          ++tdanger_squares;
          tdanger_disagree += taken != (f[(az::kHurlLands * n + r) * n + c] > 0);
        }
      }
    }
    // 3. Symmetry: rotate the board, then compute; against compute, then rotate.
    const std::vector<float> obs = state->ObservationTensor(0);
    for (int s = 1; s < thud::kNumSymmetries; ++s) {
      std::vector<float> turned_features;
      az::AppendFeaturePlanes(thud::SymmetricObservation(obs, s), &turned_features);
      sym_diff += Differences(turned_features, Turned(f, s, thud::SymmetricDirection), AllPlanes());
      sym_planted_diff += Differences(turned_features, Turned(f, s, Identity), AllPlanes());
    }
  }
  std::cout << "   " << captures_seen << " capturing moves across the positions, the longest "
            << longest << " squares" << std::endl;
  Check(move_diff == 0 && captures_seen > 0,
        "1. the move planes match the capturing moves of both sides (" +
            std::to_string(move_diff) + " squares differ)");
  Check(too_long == 0, "1. no capture is longer than MaxCaptureDistance (" +
                           std::to_string(too_long) + " are)");
  Check(planted_diff > 0, "4. control: two distance planes swapped differ at " +
                              std::to_string(planted_diff) + " squares");
  Check(map_piece_diff == 0 && map_pieces > 0,
        "2. the maps on pieces: takeable dwarfs and hurled-onto trolls exactly (" +
            std::to_string(map_piece_diff) + " of " + std::to_string(map_pieces) +
            " pieces differ)");
  std::cout << "   empty squares: a placed dwarf's fact disagrees with kTrollLandsNear on "
            << danger_disagree << " of " << danger_squares << ", a placed troll's with "
            << "kHurlLands on " << tdanger_disagree << " of " << tdanger_squares << std::endl;
  Check(danger_squares > 0 && danger_disagree * 100 < danger_squares,
        "2. kTrollLandsNear agrees with a placed dwarf on over 99% of empty squares");
  Check(tdanger_squares > 0 && tdanger_disagree * 100 < tdanger_squares,
        "2. kHurlLands agrees with a placed troll on over 99% of empty squares");
  Check(sym_diff == 0, "3. rotating then computing equals computing then rotating (" +
                           std::to_string(sym_diff) + " squares differ)");
  Check(sym_planted_diff > 0, "4. control: without moving the directions they differ at " +
                                  std::to_string(sym_planted_diff) + " squares");
  std::cout << (failures ? "FAILED" : "ALL PASSED") << std::endl;
  return failures ? 1 : 0;
}
