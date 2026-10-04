# Copyright 2026 The Thud-on-OpenSpiel authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compares the searches of az_reference_pilot.cc (thud/PLAN.md, How we evaluate
networks: the pilot for the validation set's reference).

For every pair of searches (reference@budget), per side to move: how often their
most visited moves agree, and the total variation distance of their visit
distributions (half the summed absolute differences), with 95% intervals over
positions; the correlation of their root values; and each search's mean seconds.
Stability is a reference against itself at a larger budget; agreement between
different references shows how much a target depends on one network.

  python3 thud/experiments/az_reference_pilot.py PILOT.jsonl [PAIR ...]

PAIR is A,B (e.g. A14@8000,mcts@100000); without pairs it prints every pair.
"""

import itertools
import json
import math
import statistics
import sys


def distribution(visits):
  total = sum(visits.values())
  return {int(a): n / total for a, n in visits.items()}


def top(dist):
  return max(sorted(dist), key=lambda a: dist[a])  # Ties: the smallest action.


def interval(values):
  m = statistics.mean(values)
  if len(values) < 2:
    return m, float("nan")
  return m, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def correlation(xs, ys):
  mx, my = statistics.mean(xs), statistics.mean(ys)
  sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
  sxx = sum((x - mx) ** 2 for x in xs)
  syy = sum((y - my) ** 2 for y in ys)
  return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main():
  rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
  names = list(rows[0]["searches"])
  pairs = ([tuple(p.split(",")) for p in sys.argv[2:]] if len(sys.argv) > 2
           else list(itertools.combinations(names, 2)))
  for side in ("dwarfs", "trolls"):
    side_rows = [r for r in rows if r["side"] == side]
    print(f"== {side}: {len(side_rows)} positions, median legal "
          f"{statistics.median(r['legal'] for r in side_rows)}")
    for name in names:
      secs = [r["searches"][name]["seconds"] for r in side_rows]
      visited = [len(r["searches"][name]["visits"]) for r in side_rows]
      print(f"   {name:16s} {statistics.mean(secs):7.2f} s a search, "
            f"median {statistics.median(visited)} moves visited")
    for a, b in pairs:
      agrees, distances, va, vb = [], [], [], []
      for r in side_rows:
        da = distribution(r["searches"][a]["visits"])
        db = distribution(r["searches"][b]["visits"])
        agrees.append(100.0 * (top(da) == top(db)))
        moves = set(da) | set(db)
        distances.append(0.5 * sum(abs(da.get(m, 0) - db.get(m, 0)) for m in moves))
        va.append(r["searches"][a]["value"])
        vb.append(r["searches"][b]["value"])
      ag, agi = interval(agrees)
      di, dii = interval(distances)
      print(f"   {a:16s} vs {b:16s} top move agrees {ag:5.1f}% ±{agi:4.1f}, "
            f"distance {di:.3f} ±{dii:.3f}, value correlation "
            f"{correlation(va, vb):+.2f}")


if __name__ == "__main__":
  main()
