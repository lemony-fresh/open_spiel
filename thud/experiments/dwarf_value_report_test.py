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

"""Tests thud/experiments/dwarf_value_report.py (measurement 4's ratios with intervals) on
records made here, of known values:

  1. Exact: every dwarf-to-move position worth 0.01, every troll-to-move one 0.02, every
     shift 0.002 — the ratio 0.5, net of the placebo 0.008 / 0.018, the intervals of zero
     width; a position without a shift (null) left out of the placebo's mean only.
  2. Coverage: games with an effect of their own shared by their positions (sd 0.004)
     and noise per position (sd 0.004) around worths 0.01 and 0.02: the 95% intervals of
     the ratio cover the true 0.5 in at least 85% of 40 data sets.
  3. Clustering, with its control: every position repeated 4 times within its game — no
     new information — leaves the interval as wide (within 25%); resampling positions
     one by one (cluster=False, the control) narrows it to about half, the error the
     game resampling avoids.
  4. Pairing: a network B equal to A but each dwarf-to-move worth 0.002 higher — the
     difference's interval excludes 0 and is narrower than each network's own interval
     (the same games resampled for both); A against an exact copy of itself, exactly 0.
  5. load() reads the probe's lines (a null shift and troll included); the command line
     prints a line per network and context it was measured in, and one per comparison
     where both networks were.

  python3 thud/experiments/dwarf_value_report_test.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("report", os.path.join(HERE, "dwarf_value_report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

failures = 0


def check(ok, what):
  global failures
  print(("PASS " if ok else "FAIL ") + what)
  failures += not ok


def records(net, rng, games=60, per_side=5, game_sd=0.004, noise_sd=0.004, repeat=1,
            dwarfs_extra=0.0, context="selfplay"):
  out = []
  for g in range(games):
    effect = {side: rng.normal(0, game_sd) for side in ("dwarfs", "trolls")}
    for side, worth in (("dwarfs", 0.01), ("trolls", 0.02)):
      for _ in range(per_side):
        d = worth + effect[side] + rng.normal(0, noise_sd)
        d += dwarfs_extra if side == "dwarfs" else 0
        for _ in range(repeat):
          out.append({"net": net, "kind": f"{context}_{side}_to_move", "game": g, "v": 0,
                      "dwarf": d, "shift": 0.0, "troll": 0.1})
  return out


def width(line, key="ratio"):
  return line[key][2] - line[key][1]


def main():
  # 1. Exact.
  exact = []
  for g in range(10):
    for side, worth in (("dwarfs", 0.01), ("trolls", 0.02)):
      exact.append({"net": "A", "kind": f"selfplay_{side}_to_move", "game": g, "v": 0,
                    "dwarf": worth, "shift": 0.002, "troll": None})
  exact.append({"net": "A", "kind": "selfplay_dwarfs_to_move", "game": 3, "v": 0,
                "dwarf": 0.01, "shift": None, "troll": None})
  line = report.summarize(exact, resamples=200)["nets"][0]
  ok = all(abs(x - 0.5) < 1e-12 for x in line["ratio"]) and \
       all(abs(x - 0.008 / 0.018) < 1e-12 for x in line["ratio_net"])
  check(ok and line["positions_dwarfs"] == 11,
        f"1. exact: ratio {line['ratio']}, net of the placebo {line['ratio_net'][0]:.6f} "
        f"(0.008 / 0.018), zero width; 11 dwarf-to-move positions, one without a shift")

  # 2. Coverage.
  rng = np.random.default_rng(3)
  covered = 0
  for _ in range(40):
    line = report.summarize(records("A", rng), resamples=1000,
                            seed=int(rng.integers(1 << 30)))["nets"][0]
    covered += line["ratio"][1] <= 0.5 <= line["ratio"][2]
  check(covered >= 34, f"2. coverage: the 95% intervals cover the true ratio 0.5 in "
                       f"{covered} of 40 data sets (at least 34)")

  # 3. Clustering and its control.
  rng = np.random.default_rng(4)
  base = records("A", rng, game_sd=0.006, noise_sd=0.002)
  repeated = [dict(r) for r in base for _ in range(4)]
  w = width(report.summarize(base, resamples=2000)["nets"][0])
  w_rep = width(report.summarize(repeated, resamples=2000)["nets"][0])
  w_naive = width(report.summarize(base, resamples=2000, cluster=False)["nets"][0])
  w_naive_rep = width(report.summarize(repeated, resamples=2000, cluster=False)["nets"][0])
  check(0.75 < w_rep / w < 1.33 and w_naive_rep / w_naive < 0.65,
        f"3. each position 4 times in its game: by games the interval stays "
        f"{w_rep / w:.2f} as wide; the control, by positions, narrows to "
        f"{w_naive_rep / w_naive:.2f}")

  # 4. Pairing.
  rng = np.random.default_rng(5)
  a = records("A", rng)
  b = [dict(r, net="B", dwarf=r["dwarf"] + (0.002 if "dwarfs" in r["kind"] else 0))
       for r in a]
  c = [dict(r, net="C") for r in a]
  result = report.summarize(a + b + c, compare=["B-A", "C-A"], resamples=2000)
  lines = {l["net"]: l for l in result["nets"]}
  diff = {l["compare"]: l for l in result["comparisons"]}
  d = diff["B-A"]["ratio_diff"]
  narrower = (d[2] - d[1]) < 0.5 * min(width(lines["A"]), width(lines["B"]))
  check(d[1] > 0 and narrower and diff["C-A"]["ratio_diff"] == [0.0, 0.0, 0.0],
        f"4. paired: B - A {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}] excludes 0 and is narrower "
        f"than A's own ({width(lines['A']):.3f}); a copy of A gives "
        f"{diff['C-A']['ratio_diff']}")

  # 5. load() and the command line.
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "each.jsonl")
    with open(path, "w") as f:
      f.write('{"net": "N1", "kind": "ab_games_trolls_to_move", "game": 7, "v": -0.588600, '
              '"dwarf": 0.017900, "shift": null, "troll": null}\n')
      for r in a + b:
        f.write(json.dumps(r) + "\n")
    loaded = report.load([path])
    out = subprocess.run([sys.executable, os.path.join(HERE, "dwarf_value_report.py"), path,
                          "--compare", "B-A", "--resamples", "200"],
                         capture_output=True, text=True)
    printed = [json.loads(l) for l in out.stdout.splitlines()]
    nets = [p for p in printed if "net" in p]
    comparisons = [p for p in printed if "compare" in p]
  check(len(loaded) == len(a + b) + 1 and loaded[0]["shift"] is None and out.returncode == 0
        and len(nets) == 3 and len(comparisons) == 1,
        f"5. load() reads {len(loaded)} records (a null shift kept as None); the command "
        f"line prints {len(nets)} network lines (A and B in self-play, N1 in the games "
        f"against alpha-beta) and {len(comparisons)} comparison (B - A, in self-play)")

  print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
