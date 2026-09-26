# Copyright 2026 The Thud-on-OpenSpiel authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarises Thud games that OpenSpiel's mcts_example played, one game per log file.

Phase 5 (thud/PLAN.md): MCTS with random rollouts against itself. Play the games first
with the Release build (CLAUDE.md), 10 at a time, one per core; seeds 1-20, 5,000
simulations per move took 400 s:

  cd build-release && mkdir -p /tmp/mcts5k && seq 1 20 | xargs -P 10 -I{} sh -c \
    's=$(date +%s); ./examples/mcts_example --game=thud --player1=mcts --player2=mcts \
     --max_simulations=5000 --rollout_count=1 --num_games=1 --seed={} \
     > /tmp/mcts5k/game_{}.log 2>&1; echo "wall_seconds=$(( $(date +%s) - s ))" \
     >> /tmp/mcts5k/game_{}.log'
  python3 ../thud/experiments/mcts_games.py /tmp/mcts5k

For each folder of logs it reports the results, lengths and material left, then replays
every game through pyspiel to report each side's number of legal moves and how often a
capture was available to it, and taken. Needs pyspiel, as random_endings.py does.
"""

import glob
import math
import re
import statistics as st
import sys

import pyspiel


def read_log(path):
  """The moves, final position and result of one game, and its wall time if logged."""
  text = open(path).read()
  board, turns = re.findall(r"State: \n((?:.*\n){15})to_move=\S+ turns=(\d+)", text)[-1]
  moves = re.search(r"Game actions: (.*)", text).group(1).split()
  wall = re.search(r"wall_seconds=(\d+)", text)
  game = {"moves": moves, "dwarfs": board.count("d"), "trolls": board.count("T"),
          "margin": round(float(re.search(r"Returns: ([-\d.e]+),", text).group(1)) * 32),
          "wall": int(wall.group(1)) if wall else None}
  assert int(turns) == len(moves) and game["margin"] == game["dwarfs"] - 4 * game["trolls"]
  return game


def turns_of(game, pyspiel_game):
  """(side to move, legal moves, captures among them, whether a capture was played)."""
  state, turns = pyspiel_game.new_initial_state(), []
  for move in game["moves"]:
    by_name = {state.action_to_string(a): a for a in state.legal_actions()}
    turns.append((state.current_player(), len(by_name),
                  sum(name.endswith("x") for name in by_name), move.endswith("x")))
    state.apply_action(by_name[move])
  assert state.is_terminal()
  return turns


def wilson(k, n, z=1.96):
  """95% Wilson score interval for a share of k in n."""
  centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
  half = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
  return centre - half, centre + half


def report(folder, pyspiel_game):
  games = [read_log(p) for p in glob.glob(folder + "/game_*.log")]
  n, wins = len(games), sum(g["margin"] > 0 for g in games)
  low, high = wilson(wins, n)
  lengths = [len(g["moves"]) for g in games]
  print(f"{folder}: {n} games")
  print(f"  dwarfs win {wins}, trolls {sum(g['margin'] < 0 for g in games)}, draws "
        f"{sum(g['margin'] == 0 for g in games)} (dwarfs' share, 95% interval "
        f"{low:.0%}-{high:.0%}); dwarf margins {sorted(g['margin'] for g in games)}, "
        f"mean {st.mean(g['margin'] for g in games):+.1f}")
  print(f"  no trolls left {sum(g['trolls'] == 0 for g in games)}, no dwarfs left "
        f"{sum(g['dwarfs'] == 0 for g in games)}; dwarfs left median "
        f"{st.median(g['dwarfs'] for g in games)}; turns min {min(lengths)}, median "
        f"{st.median(lengths):.0f}, max {max(lengths)}")
  walls = [g["wall"] for g in games if g["wall"]]
  if len(walls) == n:
    print(f"  wall time per game {min(walls)}-{max(walls)} s, "
          f"{sum(walls) / sum(lengths):.2f} s per move")
  turns = [t for g in games for t in turns_of(g, pyspiel_game)]
  for player, side in ((0, "dwarfs"), (1, "trolls")):
    mine = [t for t in turns if t[0] == player]
    open_ = [t for t in mine if t[2]]
    print(f"  {side}: legal moves median {st.median(t[1] for t in mine):.0f}; a capture "
          f"available on {len(open_) / len(mine):.0%} of turns, and then "
          f"{st.median(t[2] / t[1] for t in open_):.1%} of the legal moves (median), "
          f"taken {sum(t[3] for t in open_) / len(open_):.0%} of the time")


def main():
  pyspiel_game = pyspiel.load_game("thud")
  for folder in sys.argv[1:]:
    report(folder, pyspiel_game)


if __name__ == "__main__":
  main()
