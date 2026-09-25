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

"""Reference perft counts for TestPerft in open_spiel/games/thud/thud_test.cc.

Counts the move sequences of 1-3 turns from fixed positions with hexparrot/thudgame's
independent Thud engine, so that our move generator is checked against code we did not
write. This produced the counts in TestPerft (thud/PROGRESS.md, session 3, 2026-09-23):

  python3 perft_reference.py        # about 20 s on 10 processes

hexparrot's plies map one-to-one to our actions: dwarf moves and hurls; troll steps without
a capture (its find_moves); one-square troll steps capturing all adjacent dwarfs, and shoves
of 2..N squares (its find_caps). One known difference: find_caps tries hurls and shoves of at
most 6 squares, where our rules allow 7 along a full row or column. So every position that
generates moves is checked for lines of 7 or more same-type pieces, where the difference
could matter; the counts are valid only if the longest line reported is at most 6.

The positions are kInitial and kTangled from thud_test.cc, each with both sides to move, and
a midgame taken from hexparrot's AI playing itself, which becomes kMidgame.

Setup: hexparrot (MIT-licensed) is not vendored. Clone it outside this repo at commit
7b171108ddb76c75a0b4177a57083d8e36d764cc, which every measurement used, by default to
~/.local/share/thud-openspiel/hexparrot_thudgame ($XDG_DATA_HOME, default ~/.local/share:
persistent user data, unlike ~/.cache, which is meant to be disposable); or pass
--hexparrot PATH:

  git clone https://github.com/hexparrot/thudgame.git ~/.local/share/thud-openspiel/hexparrot_thudgame
  git -C ~/.local/share/thud-openspiel/hexparrot_thudgame checkout 7b171108ddb76c75a0b4177a57083d8e36d764cc

Without it, every script here stops with these instructions. The other scripts reuse this
file's helpers and its default location.
"""

import argparse
import importlib
import logging
import multiprocessing as mp
import os
import sys
import time

OPENING = """
-----dd.dd-----
----d.....d----
---d.......d---
--d.........d--
-d...........d-
d.............d
d.....TTT.....d
......TOT......
d.....TTT.....d
d.............d
-d...........d-
--d.........d--
---d.......d---
----d.....d----
-----dd.dd-----
"""

TANGLED = """
-----d....-----
----d......----
---d........---
--d..........--
-d............-
.....d.......Td
d......T.......
.......Od....d.
.......ddd...d.
.TTTTTT......d.
-.........d..d-
--...........--
---.........---
----.......----
-----.ddd.-----
"""

# (name, diagram, side to move, perft depths)
POSITIONS = [
    ("opening", OPENING, "dwarf", [1, 2, 3]),
    ("opening", OPENING, "troll", [1, 2]),
    ("tangled", TANGLED, "dwarf", [1, 2]),
    ("tangled", TANGLED, "troll", [1, 2]),
]
OTHER = {"dwarf": "troll", "troll": "dwarf"}
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

# Where the scripts look for hexparrot's clone: persistent per-user data (XDG_DATA_HOME,
# default ~/.local/share), not the cache, which is meant to be disposable. And the commit
# every measurement used.
DEFAULT_HEXPARROT = os.path.join(
    os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
    "thud-openspiel", "hexparrot_thudgame")
HEXPARROT_COMMIT = "7b171108ddb76c75a0b4177a57083d8e36d764cc"

# hexparrot's engine, imported by load_hexparrot().
Bitboard = Gameboard = Ply = selfplay = None


def check_hexparrot(path):
  """Exits with instructions to restore it unless hexparrot is cloned at `path`.

  Without a clone there, `import thud` would find this repo's own thud/ folder (the repo
  is on PYTHONPATH) and fail with a confusing "No module named 'thud.bitboard'".
  """
  if os.path.isfile(os.path.join(path, "thud", "gameboard.py")):
    return
  sys.exit(f"""hexparrot/thudgame is not installed at {path}.

What it is: https://github.com/hexparrot/thudgame, an independent, MIT-licensed Python
engine for Thud. The scripts in thud/experiments/ use it as the reference that our own
implementation is checked against: crosscheck_tests.py (every hand-written move
expectation in open_spiel/games/thud/thud_test.cc), perft_reference.py (TestPerft's
counts), move_kinds_sim.py and limits_sim.py (measurements recorded in thud/PROGRESS.md).
Nothing in the build or in the C++ tests needs it.

Where it goes: outside this repository, never inside it (it is not vendored). The default
is $XDG_DATA_HOME/thud-openspiel/hexparrot_thudgame, with $XDG_DATA_HOME defaulting to
~/.local/share. Every measurement used commit {HEXPARROT_COMMIT}.

To restore it:
  git clone https://github.com/hexparrot/thudgame.git {path}
  git -C {path} checkout {HEXPARROT_COMMIT}

To check: {os.path.join(path, "thud", "gameboard.py")} must exist, and
`git -C {path} log --oneline -1` must show 7b17110. Then re-run the script. To use a
clone elsewhere, pass --hexparrot PATH. More in thud/experiments/perft_reference.py
(module docstring) and in thud/PROGRESS.md (Current status).""")


def load_hexparrot(path):
  """Imports hexparrot's engine from its clone at `path`; runs in every worker too."""
  global Bitboard, Gameboard, Ply, selfplay  # pylint: disable=global-statement
  check_hexparrot(path)
  sys.path.insert(0, path)
  logging.disable(logging.CRITICAL)
  Bitboard = importlib.import_module("thud.bitboard").Bitboard
  Gameboard = importlib.import_module("thud.gameboard").Gameboard
  Ply = importlib.import_module("thud.ply").Ply
  selfplay = importlib.import_module("thud.selfplay")


def pos(r, c):
  """hexparrot's square index for our (row, col); its board has a 1-square border."""
  return (r + 1) * 17 + (c + 1)


def rc(p):
  """Our (row, col) for hexparrot's square index."""
  file, rank = Ply.position_to_tuple(p)
  return rank - 1, file - 1


def board_from(diagram):
  """A hexparrot board holding the pieces of one of our diagrams."""
  rows = [line.strip() for line in diagram.strip().split("\n")]
  board = Gameboard("classic")
  board.dwarfs = Bitboard([pos(r, c) for r, row in enumerate(rows)
                           for c, cell in enumerate(row) if cell == "d"])
  board.trolls = Bitboard([pos(r, c) for r, row in enumerate(rows)
                           for c, cell in enumerate(row) if cell == "T"])
  assert set(board.thudstone.get_bits()) == {pos(7, 7)}
  return board


def diagram_of(board):
  """One of our diagrams for a hexparrot board."""
  dwarfs = {rc(p) for p in board.dwarfs.get_bits()}
  trolls = {rc(p) for p in board.trolls.get_bits()}
  rows = []
  for r in range(15):
    row = ""
    for c in range(15):
      if not board.playable[pos(r, c)]:
        row += "-"
      elif (r, c) == (7, 7):
        row += "O"
      else:
        row += "d" if (r, c) in dwarfs else "T" if (r, c) in trolls else "."
    rows.append(row)
  return "\n".join(rows)


def moves(board, side):
  """hexparrot's legal plies for `side`: one per action of ours."""
  plies = list(board.find_caps(side)) + list(board.find_moves(side))
  keys = [(p.origin, p.dest, bool(p.captured)) for p in plies]
  assert len(keys) == len(set(keys)), "duplicate plies"
  return plies


def kind(ply, side):
  """The kind of move, named as in thud_test.cc's Kind."""
  (r1, c1), (r2, c2) = rc(ply.origin), rc(ply.dest)
  if side == "dwarf":
    return "hurl" if ply.captured else "move"
  if not ply.captured:
    return "step"
  return "capture_step" if max(abs(r2 - r1), abs(c2 - c1)) == 1 else "shove"


def longest_line(board):
  """The longest unbroken line of dwarfs, or of trolls, in any direction."""
  best = 0
  for pieces in (board.dwarfs, board.trolls):
    squares = {rc(p) for p in pieces.get_bits()}
    for r, c in squares:
      for dr, dc in DIRECTIONS[:4]:
        if (r - dr, c - dc) in squares:
          continue  # Not the start of its line.
        n = 0
        while (r + n * dr, c + n * dc) in squares:
          n += 1
        best = max(best, n)
  return best


def perft(board, side, depth, stats):
  """The number of move sequences of `depth` turns; records the longest line seen."""
  stats["longest_line"] = max(stats["longest_line"], longest_line(board))
  plies = moves(board, side)
  if depth == 1:
    return len(plies)
  total = 0
  for ply in plies:
    snapshot = board.snapshot()
    board.apply_ply(ply)
    total += perft(board, OTHER[side], depth - 1, stats)
    board.restore(snapshot)
  return total


def subtree(args):
  """perft below the root's move `index`, for one worker process."""
  diagram, side, index, depth = args
  board = board_from(diagram)
  board.apply_ply(moves(board, side)[index])
  stats = {"longest_line": 0}
  return perft(board, OTHER[side], depth - 1, stats), stats["longest_line"]


def perft_parallel(pool, diagram, side, depth):
  """perft, with one job per move from the root; also the longest line seen."""
  root = board_from(diagram)
  if depth == 1:
    return len(moves(root, side)), longest_line(root)
  jobs = [(diagram, side, i, depth) for i in range(len(moves(root, side)))]
  results = pool.map(subtree, jobs)
  return (sum(n for n, _ in results),
          max([longest_line(root)] + [line for _, line in results]))


def pick_midgame():
  """The first position, after 16 or more turns of hexparrot's AI playing itself, where
  the dwarfs have a hurl and the trolls both a capturing step and a shove."""
  for seed in range(200):
    game = selfplay.play_game("classic", seed=seed, max_plies=400, lookahead=3)
    board = Gameboard("classic")
    for turn, ply in enumerate(game["ply_list"]):
      if turn >= 16:
        dwarf_kinds = [kind(p, "dwarf") for p in moves(board, "dwarf")]
        troll_kinds = [kind(p, "troll") for p in moves(board, "troll")]
        if ("hurl" in dwarf_kinds and "capture_step" in troll_kinds and
            "shove" in troll_kinds):
          side = "dwarf" if turn % 2 == 0 else "troll"
          return seed, turn, diagram_of(board), side
      board.apply_ply(ply)
      board.ply_list.append(ply)
  raise RuntimeError("no midgame found")


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--workers", type=int, default=os.cpu_count())
  parser.add_argument("--hexparrot", default=DEFAULT_HEXPARROT)
  args = parser.parse_args()
  load_hexparrot(args.hexparrot)

  start = time.time()
  seed, turn, midgame, side = pick_midgame()
  print(f"midgame: hexparrot's AI against itself (lookahead 3), seed {seed}, "
        f"after {turn} turns")
  print("\n".join("  " + row for row in midgame.split("\n")))
  positions = POSITIONS + [("midgame", midgame, side, [1, 2, 3])]
  with mp.Pool(args.workers, initializer=load_hexparrot,
               initargs=(args.hexparrot,)) as pool:
    for name, diagram, side, depths in positions:
      kinds = {}
      for ply in moves(board_from(diagram), side):
        kinds[kind(ply, side)] = kinds.get(kind(ply, side), 0) + 1
      print(f"{name}, {side}s to move: first turn {kinds}")
      for depth in depths:
        count, line = perft_parallel(pool, diagram, side, depth)
        print(f"  perft({depth}) = {count}  (longest line seen: {line})  "
              f"[{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
  main()
