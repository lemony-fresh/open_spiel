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

"""Checks thud_test.cc's hand-written expectations against hexparrot/thudgame's engine.

The move expectations in open_spiel/games/thud/thud_test.cc were worked out by hand. This
script reads every position and move check out of the test file and recomputes it with
hexparrot's independent engine (perft_reference.py explains how its moves map to our
actions), mirroring the C++ helpers:

  CheckMoves, CheckAllLegalMoves  the exact set of legal moves (of one kind), as notation
  CheckLegal, CheckNotLegal       whether an action is legal; a one-square capture is read
                                  with the piece on its square, as the tests' Parse does
  CheckReach                      the distances of a piece's plain moves or steps
  CheckAfter                      the board and the status line after a move

It also reports "must not be legal" checks whose square holds no piece of the side to move
(they would pass trivially), and every diagram in the file that the reader must reject
(thud.h): more than 32 dwarfs or 8 trolls, '-' anywhere but exactly the cut-off corners, or
anything but the Thudstone at (7,7).

One known difference is reported separately: hexparrot tries hurls and shoves of at most 6
squares, so it cannot confirm the tests' 7-square ones. Written 2026-09-24 (session 4);
re-run it after changing any test. Exits with status 1 on any other disagreement.

  python3 crosscheck_tests.py          # a few seconds

Setup as in perft_reference.py (hexparrot cloned outside this repo).
"""

import argparse
import os
import re
import sys

import perft_reference as ref

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                         "open_spiel", "games", "thud", "thud_test.cc")
COLS = [(5, 9), (4, 10), (3, 11), (2, 12), (1, 13)] + [(0, 14)] * 5 + [
    (1, 13), (2, 12), (3, 11), (4, 10), (5, 9)]
SQUARE = {}
for _r in range(15):
  for _c in range(COLS[_r][0], COLS[_r][1] + 1):
    SQUARE[(_r, _c)] = len(SQUARE)
DR, DC = [-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]
DIRECTIONS = {"kNorth": 0, "kNorthEast": 1, "kEast": 2, "kSouthEast": 3,
              "kSouth": 4, "kSouthWest": 5, "kWest": 6, "kNorthWest": 7}
KINDS = {"kDwarfMove": "move", "kHurl": "hurl", "kTrollStep": "step",
         "kCaptureStep": "capture_step", "kShove": "shove"}
CHECKED = ("CheckMoves", "CheckAllLegalMoves", "CheckLegal", "CheckNotLegal",
           "CheckReach", "CheckAfter")


def sign(x):
  return (x > 0) - (x < 0)


def direction(dr, dc):
  return next(d for d in range(8) if DR[d] == sign(dr) and DC[d] == sign(dc))


def line_action(r, c, d, k):
  return (SQUARE[(r, c)] * 8 + d) * 14 + (k - 1)


def capture_step_action(r, c, d):
  return 18480 + SQUARE[(r, c)] * 8 + d


def rows_of(diagram):
  return [line.strip() for line in diagram.strip().split("\n") if line.strip()]


def diagram_problems(diagram):
  """Why the reader must reject this diagram (thud.h), if it must."""
  rows = rows_of(diagram)
  if len(rows) != 15 or any(len(row) != 15 for row in rows):
    return ["not 15 rows of 15 characters"]
  problems = []
  for r in range(15):
    for c in range(15):
      cell, on_board = rows[r][c], COLS[r][0] <= c <= COLS[r][1]
      if cell not in "-.dTO":
        problems.append(f"{cell!r} at ({r},{c})")
      if (cell == "-") == on_board:
        problems.append(f"'-' wrong at ({r},{c})")
      if cell == "O" and (r, c) != (7, 7):
        problems.append(f"a Thudstone at ({r},{c})")
  if rows[7][7] != "O":
    problems.append("no Thudstone at (7,7)")
  dwarfs, trolls = "".join(rows).count("d"), "".join(rows).count("T")
  if dwarfs > 32:
    problems.append(f"{dwarfs} dwarfs")
  if trolls > 8:
    problems.append(f"{trolls} trolls")
  return problems


class Position:
  """A test position and hexparrot's legal moves in it."""

  def __init__(self, diagram, status):
    self.diagram, self.rows = diagram, rows_of(diagram)
    fields = dict(field.split("=") for field in status.split())
    self.side = "troll" if fields.get("to_move") == "trolls" else "dwarf"
    self.turns = int(fields.get("turns", 0))
    self.without_capture = int(fields.get("turns_without_capture", 0))
    self.moves = [self.describe(ply)
                  for ply in ref.moves(ref.board_from(diagram), self.side)]

  def describe(self, ply):
    (r1, c1), (r2, c2) = ref.rc(ply.origin), ref.rc(ply.dest)
    k, d = max(abs(r2 - r1), abs(c2 - c1)), direction(r2 - r1, c2 - c1)
    capture_step = self.side == "troll" and bool(ply.captured) and k == 1
    return dict(
        r1=r1, c1=c1, d=d, k=k, ply=ply, kind=ref.kind(ply, self.side),
        action=(capture_step_action(r1, c1, d) if capture_step
                else line_action(r1, c1, d, k)),
        notation=f"({r1},{c1})-({r2},{c2})" + ("x" if ply.captured else ""))

  def parse(self, move):
    """The action a move string names, as the tests' Parse(state, move) reads it."""
    m = re.fullmatch(r"\((\d+),(\d+)\)-\((\d+),(\d+)\)(x?)", move)
    r1, c1, r2, c2 = map(int, m.groups()[:4])
    k, d = max(abs(r2 - r1), abs(c2 - c1)), direction(r2 - r1, c2 - c1)
    if m.group(5) and k == 1:
      mover = self.rows[r1][c1]
      if mover not in "dT":
        raise ValueError(f"one-square capture {move} from {mover!r}")
      if mover == "T":
        return capture_step_action(r1, c1, d)
    return line_action(r1, c1, d, k)


def split_args(text):
  """Splits at commas outside parentheses and braces."""
  out, depth, current = [], 0, ""
  for ch in text:
    depth += {"(": 1, "{": 1, ")": -1, "}": -1}.get(ch, 0)
    if ch == "," and depth == 0:
      out.append(current.strip())
      current = ""
    else:
      current += ch
  return out + [current.strip()] if current.strip() else out


def statements(body):
  """Splits a function body at ';' and at block braces, outside parentheses."""
  out, depth, current = [], 0, ""
  for ch in body:
    depth += {"(": 1, ")": -1}.get(ch, 0)
    ends = depth == 0 and (ch == ";" or (ch in "{}" and
                                        not re.search(r"[,(=]\s*$", current)))
    if ends:
      if current.strip():
        out.append(" ".join(current.split()))
      current = ""
    else:
      current += ch
  return out


def known(moves):
  """True if hexparrot's 6-square limit explains a difference in these moves."""
  return bool(moves) and all(max(abs(int(a) - int(b)) for a, b in zip(
      re.findall(r"\d+", m)[:2], re.findall(r"\d+", m)[2:4])) >= 7 for m in moves)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--hexparrot", default=os.path.expanduser("~/hexparrot_thudgame"))
  parser.add_argument("--test_file", default=TEST_FILE)
  args = parser.parse_args()
  ref.load_hexparrot(args.hexparrot)

  text = open(args.test_file).read()
  text = re.sub(r"//[^\n]*", "", text)
  text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
  diagrams = []

  def stash(m):
    diagrams.append(m.group(1))
    return f"__DIAGRAM{len(diagrams) - 1}__"

  text = re.sub(r'R"\((.*?)\)"', stash, text, flags=re.S)

  rejected = []
  for m in re.finditer(r"constexpr char (k\w+)\[\] = __DIAGRAM(\d+)__", text):
    problems = diagram_problems(diagrams[int(m.group(2))])
    if problems:
      rejected.append(f"{m.group(1)}: {', '.join(problems[:4])}")

  # File-level diagrams sit inside the three namespaces, outside any function.
  file_level, depth = {}, 0
  for m in re.finditer(r"[{}]|constexpr char (k\w+)\[\] = __DIAGRAM(\d+)__", text):
    if m.group(0) in "{}":
      depth += 1 if m.group(0) == "{" else -1
    elif depth <= 3:
      file_level[m.group(1)] = diagrams[int(m.group(2))]

  counts, failures, known_failures, trivial, unparsed = {}, [], [], [], []
  for fm in re.finditer(r"void (Test\w+)\(\) \{", text):
    test, start, depth, end = fm.group(1), fm.end(), 1, fm.end()
    while depth:
      depth += {"{": 1, "}": -1}.get(text[end], 0)
      end += 1
    names, strings, states = dict(file_level), {}, {}
    for st in statements(text[start:end - 1]):
      m = re.match(r"(?:static )?constexpr char (k\w+)\[\] = __DIAGRAM(\d+)__$", st)
      if m:
        names[m.group(1)] = diagrams[int(m.group(2))]
        continue
      m = re.match(r'const std::string (\w+) = "(.*)"$', st)
      if m:
        strings[m.group(1)] = m.group(2)
        continue
      m = re.match(r"(?:std::unique_ptr<ThudState> )?(\w+) = FromDiagram\((.*)\)$", st)
      if m:
        fargs = split_args(m.group(2))
        diagram = [a for a in fargs if a in names]
        status = [a[1:-1] if a.startswith('"') else strings[a]
                  for a in fargs if a.startswith('"') or a in strings]
        if len(diagram) == 1:
          states[m.group(1)] = (Position(names[diagram[0]], status[0] if status else ""),
                                f"{diagram[0]} {status[0] if status else ''}".strip())
        continue
      m = re.match(r"(Check\w+)\(\s*\*(\w+), (.*)\)$", st)
      if not m or m.group(1) not in CHECKED:
        if re.match(r"(%s)\(" % "|".join(CHECKED), st):
          unparsed.append(f"{test}: {st[:100]}")
        continue
      check, var, rest = m.groups()
      if var not in states:
        unparsed.append(f"{test}: no position for {st[:90]}")
        continue
      pos, where = states[var]
      counts[check] = counts.get(check, 0) + 1
      cargs = split_args(rest)
      label = f"{test}, {check} in {where}"

      if check in ("CheckMoves", "CheckAllLegalMoves"):
        if check == "CheckMoves":
          kind = KINDS[cargs[0].split("::")[1]]
          expected = [s[1:-1] for s in split_args(cargs[1][1:-1])]
          actual = [x["notation"] for x in pos.moves if x["kind"] == kind]
        else:
          expected = [s[1:-1] for s in split_args(cargs[1][1:-1])]
          for item in split_args(cargs[0][1:-1]):
            r, c, dists = split_args(item[1:-1])
            r, c = int(r), int(c)
            for d, dist in enumerate(split_args(dists[1:-1])):
              for k in range(1, int(dist) + 1):
                expected.append(f"({r},{c})-({r + DR[d] * k},{c + DC[d] * k})")
          actual = [x["notation"] for x in pos.moves]
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
          (known_failures if known(missing + extra) else failures).append(
              f"{label}: the engine lacks {missing}, has extra {extra}")

      elif check in ("CheckLegal", "CheckNotLegal"):
        arg = cargs[0]
        if arg.startswith('"'):
          r1, c1 = map(int, re.match(r'"\((\d+),(\d+)\)', arg).groups())
          try:
            action = pos.parse(arg[1:-1])
          except ValueError as e:
            failures.append(f"{label}: {e}")
            continue
        else:
          am = re.match(r"(LineAction|CaptureStepAction)\((.*)\)$", arg)
          values = [DIRECTIONS[v] if v in DIRECTIONS else int(v)
                    for v in split_args(am.group(2))]
          r1, c1 = values[:2]
          action = (line_action(*values) if am.group(1) == "LineAction"
                    else capture_step_action(*values))
        if pos.rows[r1][c1] != ("d" if pos.side == "dwarf" else "T"):
          trivial.append(f"{label}: {arg} starts on {pos.rows[r1][c1]!r}")
        if (action in {x["action"] for x in pos.moves}) != (check == "CheckLegal"):
          (known_failures if known([arg]) else failures).append(f"{label}: {arg}")

      elif check == "CheckReach":
        r, c = int(cargs[0]), int(cargs[1])
        reach = [int(v) for v in split_args(cargs[2][1:-1])]
        for d in range(8):
          got = sorted(x["k"] for x in pos.moves if (x["r1"], x["c1"]) == (r, c)
                       and x["kind"] in ("move", "step") and x["d"] == d)
          if got != list(range(1, reach[d] + 1)):
            failures.append(f"{label}: ({r},{c}) direction {d}: expected "
                            f"1..{reach[d]}, the engine {got}")

      elif check == "CheckAfter":
        move, after = cargs[0][1:-1], names[cargs[1]]
        want = cargs[2][1:-1] if cargs[2].startswith('"') else strings[cargs[2]]
        try:
          action = pos.parse(move)
        except ValueError as e:
          failures.append(f"{label}: {e}")
          continue
        hit = [x for x in pos.moves if x["action"] == action]
        if not hit:
          failures.append(f"{label}: {move} is not legal")
          continue
        board = ref.board_from(pos.diagram)
        board.apply_ply(hit[0]["ply"])
        if ref.diagram_of(board) != "\n".join(rows_of(after)):
          failures.append(f"{label}: {move} gives\n{ref.diagram_of(board)}")
        without_capture = 0 if hit[0]["ply"].captured else pos.without_capture + 1
        status = (f"to_move={'trolls' if pos.side == 'dwarf' else 'dwarfs'} "
                  f"turns={pos.turns + 1} turns_without_capture={without_capture}")
        if status != want:
          failures.append(f"{label}: {move} gives the status {status!r}")

  print("checked:", ", ".join(f"{n} {name}" for name, n in sorted(counts.items())))
  for title, items in (
      ("diagrams the reader must reject", rejected),
      ("checks not parsed (check them by hand)", unparsed),
      ("'must not be legal' checks that start on no piece of the side to move",
       trivial),
      ("differences explained by hexparrot's 6-square limit (check by hand)",
       known_failures),
      ("DISAGREEMENTS", failures)):
    print(f"\n{len(items)} {title}" + (":" if items else "."))
    for item in items:
      print("  " + item)
  sys.exit(1 if failures or rejected or unparsed or trivial else 0)


if __name__ == "__main__":
  main()
