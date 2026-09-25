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

"""How often each kind of move occurs in games of uniformly random moves.

Sizes TestRandomPlay in open_spiel/games/thud/thud_test.cc, whose random games must play
every kind of move at least once. Plays random games with hexparrot/thudgame's engine
(perft_reference.py explains how its moves map to our actions) and ends them at our default
limits: 200 turns without a capture, or 800 turns. This produced the measurement quoted in
TestRandomPlay (thud/PROGRESS.md, session 3, 2026-09-24):

  python3 move_kinds_sim.py --games 1000     # about 160 s on 10 processes

Seeds are fixed (0..N-1), so a run is reproducible for a given hexparrot commit. Setup as in
perft_reference.py.
"""

import argparse
import multiprocessing as mp
import os
import random

import perft_reference as ref

KINDS = ["move", "hurl", "step", "capture_step", "shove"]


def play(seed):
  """One random game: the number of moves of each kind, and the number of turns."""
  rng, board, side = random.Random(seed), ref.Gameboard("classic"), "dwarf"
  counts = dict.fromkeys(KINDS, 0)
  turns = without_capture = 0
  while turns < 800 and without_capture < 200:
    plies = ref.moves(board, side)
    if not plies:
      break
    ply = rng.choice(plies)
    counts[ref.kind(ply, side)] += 1
    board.apply_ply(ply)
    board.ply_list.append(ply)
    turns += 1
    without_capture = 0 if ply.captured else without_capture + 1
    side = ref.OTHER[side]
  return counts, turns


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--games", type=int, default=1000)
  parser.add_argument("--workers", type=int, default=os.cpu_count())
  parser.add_argument("--hexparrot", default=os.path.expanduser("~/hexparrot_thudgame"))
  args = parser.parse_args()

  with mp.Pool(args.workers, initializer=ref.load_hexparrot,
               initargs=(args.hexparrot,)) as pool:
    games = pool.map(play, range(args.games), chunksize=10)
  n = args.games
  print(f"{n} random games, mean length {sum(t for _, t in games) / n:.0f} turns")
  for k in KINDS:
    share = sum(1 for counts, _ in games if counts[k] > 0) / n
    print(f"  {k:13s} in {share:6.1%} of games, "
          f"{sum(counts[k] for counts, _ in games) / n:6.1f} per game; "
          f"missing from 10 games with a chance of about {(1 - share) ** 10:.1e}")
  blocks = [games[i:i + 10] for i in range(0, n, 10)]
  missing = sum(1 for block in blocks
                if any(all(counts[k] == 0 for counts, _ in block) for k in KINDS))
  print(f"  blocks of 10 consecutive games missing a kind: {missing} of {len(blocks)}")


if __name__ == "__main__":
  main()
