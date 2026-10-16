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

"""Tests thud/experiments/ladder.py (the ratings that decide promotions, thud/PLAN.md, How we
evaluate networks) on matches made here, between players of known strengths: a dwarf
strength D and a troll strength T each, the anchor's 0, a battle with x as the dwarfs and
y as the trolls ending at D_x - T_y (the linear link) or 32 tanh((D_x - T_y) / 32).

  1. Without noise, the fit recovers every player's D, T and rating R = D + T exactly
     (within 1e-6), with either link, from a chain of matches (not every player meets
     the anchor).
  2. With noise (a standard deviation of 6 a battle), the 95% intervals (R ± 1.96 SE)
     cover the true ratings about 95% of the time (at least 85% over 20 fits of 5
     players).
  3. difference(p, q) is R_p - R_q, its standard error positive.
  4. player() names a network as the ladder does (the run's short name and step; a
     search unlike its training marked), and load() reads az_match's files (pair lines and
     a summary) into those players.
  Control: with one player's battles made 5 points better, its rating rises by about 10
  (5 a battle, both sides), so 1 can tell results apart.

  python3 thud/experiments/ladder_test.py
"""

import importlib.util
import json
import math
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ladder", os.path.join(HERE, "ladder.py"))
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)
failures = 0


def check(ok, what):
  global failures
  print(("PASS " if ok else "FAIL ") + what)
  failures += not ok


# The anchor X and four players, a chain X-A-B-C and A-D: B, C and D never meet the anchor.
TRUE = {"X": (0.0, 0.0), "A": (4.0, 3.0), "B": (-2.0, 5.0), "C": (7.0, -1.0), "D": (1.5, 2.5)}
MATCHES = [("X", "A"), ("A", "B"), ("B", "C"), ("A", "D"), ("C", "X")]


def make(link, noise, rng, pairs=40, bonus=None):
  expected = (lambda z: 32 * math.tanh(z / 32)) if link == "tanh" else (lambda z: z)
  bonus = bonus or {}
  out = []
  for a, b in MATCHES:
    rows = []
    for p in range(pairs):
      da, ta = TRUE[a]
      db, tb = TRUE[b]
      as_dwarfs = expected(da - tb) + rng.gauss(0, noise) + bonus.get(a, 0) - bonus.get(b, 0)
      b_as_dwarfs = expected(db - ta) + rng.gauss(0, noise) + bonus.get(b, 0) - bonus.get(a, 0)
      rows.append({"pair": p, "a_as_dwarfs": as_dwarfs, "a_as_trolls": -b_as_dwarfs})
    out.append({"name": f"{a}_vs_{b}", "a": a, "b": b, "pairs": rows,
                "margin": sum(r["a_as_dwarfs"] + r["a_as_trolls"] for r in rows) / pairs})
  return out


def main():
  rng = random.Random(1)

  # 1. Exact recovery.
  for link in ("linear", "tanh"):
    f = ladder.fit(make(link, 0, rng), "X", link=link)
    worst = max(max(abs(f.side(p, "D") - TRUE[p][0]), abs(f.side(p, "T") - TRUE[p][1]),
                    abs(f.rating(p)[0] - sum(TRUE[p]))) for p in TRUE)
    check(worst < 1e-6, f"1. {link} link, no noise: every D, T and R recovered (worst {worst:.1e})")

  # 2. Coverage with noise.
  covered = total = 0
  for k in range(20):
    f = ladder.fit(make("tanh", 6, random.Random(100 + k)), "X")
    for p in ("A", "B", "C", "D"):
      r, se = f.rating(p)[:2]
      covered += abs(r - sum(TRUE[p])) <= 1.96 * se
      total += 1
  check(covered / total >= 0.85,
        f"2. with noise: the 95% intervals cover the true ratings {covered} of {total} times")

  # 3. Differences.
  f = ladder.fit(make("tanh", 6, random.Random(7)), "X")
  d, se = f.difference("C", "B")
  check(abs(d - (f.rating("C")[0] - f.rating("B")[0])) < 1e-9 and se > 0,
        f"3. difference(C, B) = R_C - R_B ({d:.2f}), its standard error {se:.2f}")

  # Control: one player's battles 5 points better.
  base = ladder.fit(make("linear", 0, random.Random(3)), "X", link="linear")
  better = ladder.fit(make("linear", 0, random.Random(3), bonus={"B": 5}), "X", link="linear")
  rise = better.rating("B")[0] - base.rating("B")[0]
  check(abs(rise - 10) < 1e-6, f"control: B 5 points a battle better, its rating rises by {rise:.3f}")

  # 4. player() and load().
  check(ladder.player("/x/stage3b_E_7x:44", "sibling_mean_minus_reduction", 100) == "E44" and
        ladder.player("/x/stage3b_E_7x:44", "upstream", 100) == "E44 [upstream rule]" and
        ladder.player("/x/stage3b_E_7x:44", "sibling_mean_minus_reduction", 400) == "E44 @400" and
        ladder.player("/x/runM_K83_conv_planes:102", "sibling_mean_minus_reduction", 100,
                      "1.5,1") == "M102",
        "4. player(): E44, E44 [upstream rule], E44 @400, M102 (its own temperature)")
  tmp = tempfile.mkdtemp(prefix="ladder_test.")
  with open(os.path.join(tmp, "M102_vs_E44.jsonl"), "w") as out:
    for p in range(3):
      out.write(json.dumps({"pair": p, "a_as_dwarfs": -10 + p, "a_as_trolls": 20, "a_pair_margin": 10 + p}) + "\n")
    out.write(json.dumps({"summary": True, "a": "/x/runM_K83_conv_planes:102",
                          "b": "/x/stage3b_E_7x:44", "untried_a": "sibling_mean_minus_reduction",
                          "untried_b": "sibling_mean_minus_reduction", "temperature_a": "1.5,1",
                          "temperature_b": "", "sims_a": 100, "sims_b": 100,
                          "a_mean_pair_margin": 11.0, "ci95": [9, 13]}) + "\n")
  with open(os.path.join(tmp, "not_a_match.jsonl"), "w") as out:
    out.write(json.dumps({"summary": True, "net": "x"}) + "\n")
  m = ladder.load(tmp)
  check(len(m) == 1 and m[0]["a"] == "M102" and m[0]["b"] == "E44" and len(m[0]["pairs"]) == 3
        and m[0]["margin"] == 11.0,
        "4. load(): the match read (M102 against E44, 3 pairs), the file that is not a match skipped")

  print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
