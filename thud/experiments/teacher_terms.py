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

"""The six positional terms of the teacher's finer evaluation (thud/PLAN.md, *The teacher's
finer evaluation*), on a board as ThudState prints it: 15 rows of 15 characters, "d" a
dwarf, "T" a troll, "." an empty square, "O" the Thudstone, "-" a cut-off corner square.

  dwarf_cohesion     pairs of dwarfs on neighbouring squares, each pair once
  dwarf_hurl_reach   the distinct squares, empty or holding a troll, that a dwarf could
                     reach by a hurl of 2 or more squares: its line (it counting) at least
                     that long behind it, the squares between empty
  troll_cohesion     pairs of trolls on neighbouring squares, each pair once
  troll_shove_reach  the distinct empty squares a troll could reach by a shove of 2 or more
                     squares: its line at least that long behind it, the path empty —
                     whether a dwarf stands beside the square or not
  troll_mobility     the trolls' steps onto empty squares (a square two trolls can step to
                     counts twice)
  distance           the trolls' mean distance in king moves to their nearest dwarf
                     (0 if either side has no piece)

The terms read the board alone, not the side to move. The C++ teacher computes the same
(thud/az/teacher_terms.h; teacher_terms_cross_test compares the two).
"""

SIZE = 15
STONE = (7, 7)
# thud.h's order: N, NE, E, SE, S, SW, W, NW.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
# The four directions that visit each neighbouring pair once.
FORWARD = ((0, 1), (1, 1), (1, 0), (1, -1))
# The playable columns of each row (THUD_RULES.md section 1).
_COLUMNS = {0: (5, 9), 1: (4, 10), 2: (3, 11), 3: (2, 12), 4: (1, 13), 10: (1, 13),
            11: (2, 12), 12: (3, 11), 13: (4, 10), 14: (5, 9)}
TERMS = ("dwarf_cohesion", "dwarf_hurl_reach", "troll_cohesion", "troll_shove_reach",
         "troll_mobility", "distance")


def on_board(r, c):
  if not (0 <= r < SIZE and 0 <= c < SIZE):
    return False
  lo, hi = _COLUMNS.get(r, (0, 14))
  return lo <= c <= hi


def board_with(pieces):
  """A board with `pieces` ({(row, col): "d" or "T"}) and every other square empty."""
  rows = []
  for r in range(SIZE):
    row = []
    for c in range(SIZE):
      if (r, c) == STONE:
        row.append("O")
      elif not on_board(r, c):
        row.append("-")
      else:
        row.append(pieces.get((r, c), "."))
    rows.append("".join(row))
  for (r, c), piece in pieces.items():
    assert on_board(r, c) and (r, c) != STONE and piece in "dT", ((r, c), piece)
  return rows


def start_board():
  """The starting position (THUD_RULES.md section 2)."""
  pieces = {(7 + dr, 7 + dc): "T" for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc}
  for i in (5, 6, 8, 9):  # The straight edges, without the squares in line with the stone.
    for square in ((0, i), (14, i), (i, 0), (i, 14)):
      pieces[square] = "d"
  for i in range(4):  # The diagonal edges.
    for square in ((1 + i, 4 - i), (1 + i, 10 + i), (10 + i, 1 + i), (13 - i, 10 + i)):
      pieces[square] = "d"
  assert len(pieces) == 40
  return board_with(pieces)


def parse(text):
  """The 15 board rows of ThudState's text (its last line, the counters, dropped)."""
  rows = text.strip("\n").split("\n")[:SIZE]
  assert len(rows) == SIZE and all(len(row) == SIZE for row in rows), text
  return rows


def _at(board, r, c):
  return board[r][c] if on_board(r, c) else "-"


def _line_behind(board, r, c, dr, dc, piece):
  """The unbroken line of `piece` from (r, c) backwards (direction -d), counting it."""
  n = 0
  while _at(board, r, c) == piece:
    n += 1
    r, c = r - dr, c - dc
  return n


def _reach(board, piece, landing_ok):
  """The distinct squares 2 or more squares off that a line of `piece` could send its front
  piece to: the path empty, the square itself empty or allowed by landing_ok."""
  squares = set()
  for r in range(SIZE):
    for c in range(SIZE):
      if board[r][c] != piece:
        continue
      for dr, dc in DIRECTIONS:
        n = _line_behind(board, r, c, dr, dc, piece)
        for k in range(1, n + 1):
          q = (r + k * dr, c + k * dc)
          square = _at(board, *q)
          if square == "." or landing_ok(square):
            if k >= 2:
              squares.add(q)
          if square != ".":
            break
  return len(squares)


def _pairs(board, piece):
  return sum(1 for r in range(SIZE) for c in range(SIZE) if board[r][c] == piece
             for dr, dc in FORWARD if _at(board, r + dr, c + dc) == piece)


def terms(board):
  """The six terms of `board` (a list of 15 rows), as a dict."""
  dwarfs = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == "d"]
  trolls = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == "T"]
  mobility = sum(1 for r, c in trolls for dr, dc in DIRECTIONS
                 if _at(board, r + dr, c + dc) == ".")
  if dwarfs and trolls:
    distance = sum(min(max(abs(r - x), abs(c - y)) for x, y in dwarfs)
                   for r, c in trolls) / len(trolls)
  else:
    distance = 0.0
  return {
      "dwarf_cohesion": _pairs(board, "d"),
      "dwarf_hurl_reach": _reach(board, "d", lambda square: square == "T"),
      "troll_cohesion": _pairs(board, "T"),
      "troll_shove_reach": _reach(board, "T", lambda square: False),
      "troll_mobility": mobility,
      "distance": distance,
  }
