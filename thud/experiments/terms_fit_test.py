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

"""Tests thud/experiments/terms_fit.py (the positional terms' weights fitted to our games'
results; thud/PLAN.md, *The teacher's finer evaluation*, gate 1) on data of known weights:

  1. Recovery: positions in 400 games whose final margin is material + 0.5 x term A
     - 0.25 x term B + a game's own noise (sd 3, shared by its positions) + a position's
     (sd 1): the fitted weights within 0.05 of the truth, and their 95% intervals cover
     the truth; term C, of no effect, gets an interval that covers 0 and is kept out.
  2. Clustering, with its control: the intervals resample whole games, so 4 copies of
     every position leave them as wide (within 25%); resampling positions one by one (the
     control) narrows them to about half — the false certainty the game resampling avoids.
  3. The held-out comparison: on held-out games the terms' model beats material alone,
     its interval above 0; with C alone (no effect) the interval covers 0.
  4. The replay: a self-play line of an actor's log and a match game, replayed by pyspiel,
     give positions whose final material is the recorded margin (the script's own check),
     and a wrong margin fails it.

  python3 thud/experiments/terms_fit_test.py
"""

import importlib.util
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("terms_fit", os.path.join(here, "terms_fit.py"))
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)

failures = []


def check(name, ok, detail=""):
  if not ok:
    failures.append(f"{name} {detail}")


def synthetic(games=400, per_game=50, copies=1, seed=0):
  rng = np.random.default_rng(seed)
  rows = []
  for g in range(games):
    shared = rng.normal(0, 3)
    for _ in range(per_game):
      material = rng.integers(-20, 21)
      a, b, c = rng.normal(5, 2), rng.normal(10, 4), rng.normal(0, 1)
      margin = material + 0.5 * a - 0.25 * b + shared + rng.normal(0, 1)
      for _ in range(copies):
        rows.append({"game": g, "material": material, "trolls_to_move": 0,
                     "terms": {"A": a, "B": b, "C": c}, "margin": margin})
  return rows


# 1. Recovery.
rows = synthetic()
result = tf.fit(rows, ["A", "B", "C"], boot=200, seed=1)
for name, truth in (("A", 0.5), ("B", -0.25), ("C", 0.0)):
  w = result["weights"][name]
  check(f"1: {name} weight", abs(w["weight"] - truth) < 0.05, str(w))
  check(f"1: {name} interval covers {truth}", w["low"] <= truth <= w["high"], str(w))
check("1: A and B kept, C not", result["kept"] == ["A", "B"], str(result["kept"]))

# 2. Clustering, and its control.
rows4 = synthetic(copies=4)
r4 = tf.fit(rows4, ["A", "B", "C"], boot=200, seed=1)
r4_flat = tf.fit(rows4, ["A", "B", "C"], boot=200, seed=1, cluster=False)
for name in ("A", "B"):
  width = lambda r: r["weights"][name]["high"] - r["weights"][name]["low"]
  check(f"2: {name} as wide with copies", abs(width(r4) / width(result) - 1) < 0.25,
        f"{width(r4):.4f} against {width(result):.4f}")
  check(f"2: {name} narrower by positions (control)", width(r4_flat) < 0.6 * width(r4),
        f"{width(r4_flat):.4f} against {width(r4):.4f}")

# 3. The held-out comparison.
gain = result["held_out"]
check("3: the terms beat material", gain["mse_gain"] > 0 and gain["low"] > 0, str(gain))
only_c = tf.fit(rows, ["C"], boot=200, seed=1)["held_out"]
check("3: C alone covers 0", only_c["low"] <= 0 <= only_c["high"], str(only_c))

# 4. The replay.
try:
  import pyspiel  # noqa: F401
  line = ("[2026-10-27 16:08:19.431] Game 1: Returns: 0.15625 -0.15625; Ending: "
          "no_capture_limit; Actions: (6,14)-(7,14) (6,6)-(5,6)")
  game = tf.parse_log_line(line)
  check("4: the log line parsed", game is not None and game["margin"] == 5 and
        game["actions"] == ["(6,14)-(7,14)", "(6,6)-(5,6)"], str(game))
  try:
    tf.positions(game)
    failures.append("4: a margin the final board does not give passed the replay's check")
  except tf.ReplayError:
    pass
  game["margin"] = 0  # The start's material: 32 - 4 x 8, after two quiet moves.
  ps = tf.positions(game)
  check("4: two positions, the mover's side", [p["trolls_to_move"] for p in ps] == [0, 1],
        str([p["trolls_to_move"] for p in ps]))
  check("4: the start's terms", ps[0]["terms"]["troll_cohesion"] == 12, str(ps[0]["terms"]))
  match = {"a_player": 1, "a_margin": 0, "actions": ["(6,14)-(7,14)"]}
  check("4: a match game's margin from the dwarfs' side",
        tf.match_game(match)["margin"] == 0 and tf.match_game(
            {"a_player": 1, "a_margin": 3, "actions": []})["margin"] == -3)
except ImportError:
  failures.append("4: pyspiel not importable (PYTHONPATH: CLAUDE.md)")

if failures:
  print("\n".join(failures))
  print(f"terms_fit_test: {len(failures)} FAILED")
  sys.exit(1)
print("terms_fit_test: all passed")
