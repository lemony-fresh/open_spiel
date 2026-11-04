#!/usr/bin/env python3
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

"""Tests thud/experiments/teacher_terms.py, the six positional terms of the teacher's finer
evaluation (thud/PLAN.md, *The teacher's finer evaluation*), on hand-built positions whose
values were worked out by hand from the definitions, not from the code:

  1. Two dwarfs side by side: one pair; a hurl reach of one square (the line's 2-square
     hurl east; west runs off the board); no trolls, so every troll term 0 and the distance 0.
  2. A line of three dwarfs: two pairs; reach 2 (east 2 and 3 squares; the inner dwarfs'
     hurls blocked by their own line).
  3. The Thudstone: a line of four pointing at it reaches only the square before it.
  4. A troll on a hurl square counts and ends the line's reach there; one lone troll: no
     pair, no shove reach, 8 steps, its distance 2 to the nearest dwarf; the distance in
     king moves (a diagonal 3, not 6), to the nearest of two dwarfs.
  5. A line of three trolls: two pairs, a shove reach of 4 (2 and 3 squares off each end),
     20 steps; a dwarf on a shove square blocks that end's reach (2 left) and sets the
     distance (the mean of 4, 3 and 2).
  6. The board's edge: a diagonal pair on the cut-off corner reaches one square inward and
     none outward.
  7. Two lines reaching the same square count it once.
  8. The starting position: dwarf pairs 28, hurl reach 0, troll pairs 12, shove reach 24,
     troll steps 32, distance 4.
  9. The terms read the board alone: the same board with the trolls to move gives the same
     values; parse() reads ThudState's text (pyspiel) and agrees with a hand-built board.

  python3 thud/experiments/teacher_terms_test.py
"""

import importlib.util
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("teacher_terms",
                                              os.path.join(here, "teacher_terms.py"))
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

failures = []


def check(name, got, want):
  if got != want:
    failures.append(f"{name}: got {got}, want {want}")


def expect(name, board, **want):
  got = tt.terms(board)
  for key, value in want.items():
    check(f"{name}: {key}", got[key], value)


D, T = "d", "T"
B = tt.board_with

# 1. Two dwarfs side by side on row 5 (columns 0-14 on the board).
expect("1", B({(5, 1): D, (5, 2): D}), dwarf_cohesion=1, dwarf_hurl_reach=1,
       troll_cohesion=0, troll_shove_reach=0, troll_mobility=0, distance=0.0)

# 2. A line of three on row 7, pointing east (the Thudstone at (7,7) is beyond its reach).
expect("2", B({(7, 1): D, (7, 2): D, (7, 3): D}), dwarf_cohesion=2, dwarf_hurl_reach=2)

# 3. A line of four pointing at the Thudstone: (7,5) is 1 square off, (7,6) 2, (7,7) the stone.
expect("3", B({(7, 1): D, (7, 2): D, (7, 3): D, (7, 4): D}), dwarf_cohesion=3,
       dwarf_hurl_reach=1)

# 4. A troll two squares in front of a line of three: the reach is that square alone.
expect("4", B({(5, 1): D, (5, 2): D, (5, 3): D, (5, 5): T}), dwarf_cohesion=2,
       dwarf_hurl_reach=1, troll_cohesion=0, troll_shove_reach=0, troll_mobility=8,
       distance=2.0)

# 4b. King moves, the nearest dwarf: (8,5) is 3 king moves from (5,2) (6 orthogonal steps)
# and 4 from (8,9).
expect("4b", B({(5, 2): D, (8, 9): D, (8, 5): T}), distance=3.0)

# 5. A line of three trolls on row 10 (columns 1-13 on the board).
trolls = {(10, 5): T, (10, 6): T, (10, 7): T}
expect("5", B(trolls), troll_cohesion=2, troll_shove_reach=4, troll_mobility=20,
       distance=0.0, dwarf_cohesion=0, dwarf_hurl_reach=0)
expect("5 blocked", B({**trolls, (10, 9): D}), troll_cohesion=2, troll_shove_reach=2,
       troll_mobility=20, distance=3.0, dwarf_cohesion=0, dwarf_hurl_reach=0)

# 6. The top-left cut corner: (1,4) and (2,3) on its diagonal edge.
expect("6", B({(1, 4): D, (2, 3): D}), dwarf_cohesion=1, dwarf_hurl_reach=1)

# 7. A row pair reaching (5,4) east, a column pair reaching (5,4) south.
expect("7", B({(5, 1): D, (5, 2): D, (2, 4): D, (3, 4): D}), dwarf_cohesion=2,
       dwarf_hurl_reach=1)

# 8. The starting position.
start = tt.start_board()
expect("8", start, dwarf_cohesion=28, dwarf_hurl_reach=0, troll_cohesion=12,
       troll_shove_reach=24, troll_mobility=32, distance=4.0)
check("8: 32 dwarfs", sum(row.count(D) for row in start), 32)
check("8: 8 trolls", sum(row.count(T) for row in start), 8)

# 9. The board alone; ThudState's text parsed.
board5 = B({**trolls, (10, 9): D})
text = "\n".join(board5) + "\nto_move=trolls turns=3 turns_without_capture=1"
check("9: parse", tt.parse(text), board5)
check("9: the side to move", tt.terms(tt.parse(text)), tt.terms(board5))
try:
  import pyspiel
  state = pyspiel.load_game("thud").new_initial_state()
  check("9: pyspiel's start", tt.parse(str(state)), start)
except ImportError:
  failures.append("9: pyspiel not importable (PYTHONPATH: CLAUDE.md)")

if failures:
  print("\n".join(failures))
  print(f"teacher_terms_test: {len(failures)} FAILED")
  sys.exit(1)
print("teacher_terms_test: all passed")
