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

"""How the end-of-battle limits (THUD_RULES.md §6) would affect game length.

Plays Thud games with hexparrot/thudgame's engine, whose rules match ours closely enough for
length statistics, with no limit of our own. It records the turn of every capture, then
computes what each (no-capture cap, hard cap) pair would have done to those games. This
produced the measurements in thud/PROGRESS.md, session 2 (2026-09-22):

  python3 limits_sim.py --player random --games 1000   # 148 s on 10 processes
  python3 limits_sim.py --player ai --games 3000       # 755 s on 10 processes

Setup: hexparrot (MIT-licensed) is not vendored. Clone it outside this repo, by default to
~/hexparrot_thudgame (or pass --hexparrot); the measurements used commit
7b171108ddb76c75a0b4177a57083d8e36d764cc:

  git clone https://github.com/hexparrot/thudgame.git ~/hexparrot_thudgame

Seeds are fixed (0..N-1), so a run is reproducible for a given hexparrot commit.
"""

import argparse
import logging
import multiprocessing as mp
import os
import random
import statistics as st
import sys
import time

SAFETY_CAP = 3000  # far beyond any limit we evaluate; no game reached it
LIMIT_PAIRS = [(50, 400), (100, 400), (100, 800), (200, 400), (200, 800), (None, None)]


def play_random(seed):
  """One game with uniformly random legal moves for both sides."""
  from thud.gameboard import Gameboard  # pylint: disable=g-import-not-at-top
  rng, board = random.Random(seed), Gameboard("classic")
  while len(board.ply_list) < SAFETY_CAP and not board.get_game_outcome():
    side = board.turn_to_act()
    moves = list(board.find_caps(side)) + list(board.find_moves(side))
    if not moves:
      break
    ply = rng.choice(moves)
    board.apply_ply(ply)
    board.ply_list.append(ply)
  return board.ply_list


def play_ai(seed, lookahead):
  """One game of hexparrot's heuristic AI against itself."""
  from thud import selfplay  # pylint: disable=g-import-not-at-top
  return selfplay.play_game("classic", seed=seed, max_plies=SAFETY_CAP,
                            lookahead=lookahead)["ply_list"]


def play(args):
  seed, player, lookahead, hexparrot = args
  sys.path.insert(0, hexparrot)
  logging.disable(logging.CRITICAL)
  plies = play_random(seed) if player == "random" else play_ai(seed, lookahead)
  return {"length": len(plies), "captures": [i for i, p in enumerate(plies) if p.captured]}


def turns_played(game, no_capture_cap, hard_cap):
  """Turns played before a limit ends the battle, and which limit (if any) did."""
  last = -1
  for turn in game["captures"]:
    if turn - last - 1 >= no_capture_cap:
      end = last + 1 + no_capture_cap
      return (end, "no-capture cap") if end <= hard_cap else (hard_cap, "hard cap")
    last = turn
  end = min(game["length"], last + 1 + no_capture_cap)
  if end > hard_cap:
    return hard_cap, "hard cap"
  return end, ("no-capture cap" if end < game["length"] else None)


def longest_gap(game):
  edges = [-1] + game["captures"] + [game["length"]]
  return max(b - a - 1 for a, b in zip(edges, edges[1:]))


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--player", choices=["random", "ai"], default="random")
  parser.add_argument("--games", type=int, default=1000)
  parser.add_argument("--lookahead", type=int, default=3,
                      help="hexparrot AI lookahead (its GUI/console default is 3)")
  parser.add_argument("--workers", type=int, default=os.cpu_count())
  parser.add_argument("--hexparrot", default=os.path.expanduser("~/hexparrot_thudgame"))
  args = parser.parse_args()

  start = time.time()
  jobs = [(seed, args.player, args.lookahead, args.hexparrot) for seed in range(args.games)]
  with mp.Pool(args.workers) as pool:
    games = pool.map(play, jobs, chunksize=10)
  print(f"{args.games} {args.player} games in {time.time() - start:.0f}s "
        f"on {args.workers} processes")

  lengths, gaps = [g["length"] for g in games], [longest_gap(g) for g in games]
  print(f"natural length: median {st.median(lengths):.0f}, max {max(lengths)}; "
        f"longest no-capture stretch: median {st.median(gaps):.0f}, max {max(gaps)}")
  print(f"{'limits':>9} {'mean turns':>11} {'cut by no-capture cap':>22} {'cut by hard cap':>16}")
  for cap, hard in LIMIT_PAIRS:
    label = f"{cap}/{hard}" if cap else "none"
    results = [turns_played(g, cap or SAFETY_CAP, hard or SAFETY_CAP) for g in games]
    by = lambda why: sum(r[1] == why for r in results)  # pylint: disable=cell-var-from-loop
    print(f"{label:>9} {st.mean(r[0] for r in results):>11.1f} "
          f"{by('no-capture cap'):>22} {by('hard cap'):>16}")


if __name__ == "__main__":
  main()
