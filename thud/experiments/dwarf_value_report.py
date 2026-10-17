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

"""Measurement 4's ratios with intervals (thud/DESIGN_dwarf_signal.md, measurement 4 and
E1's offline sweep), from az_dwarf_value_probe's records of every position (each=PATH).
Per network and context (self-play; the games against alpha-beta's dwarfs):

  worth_dwarfs  a far dwarf's worth to the value with the dwarfs to move (the mean of
                `dwarf` over the dwarf-to-move positions)
  worth_trolls  the same with the trolls to move
  ratio         worth_dwarfs / worth_trolls (the design's table)
  ratio_net     the same net of the placebo, the dwarf moved one square: (worth - the
                mean of `shift`) on each side

each with a 95% interval from a bootstrap that resamples whole games — the positions of
one game are not independent, and resampling them one by one would make the intervals too
narrow — and for each --compare A-B the differences between two networks, paired: the
same resampled games for both, since both were measured on the same positions.

  python3 thud/experiments/dwarf_value_report.py EACH.jsonl [...] [--compare A-B ...] \\
    [--resamples 2000] [--seed 1]

Network names must not contain "-" (it separates the two in --compare).

Prints one JSON line per network and context, then one per comparison and context.
"""

import argparse
import json
import sys

import numpy as np

CONTEXTS = ("selfplay", "ab_games")
MEASURES = ("worth_dwarfs", "worth_trolls", "ratio", "ratio_net")


def load(paths):
  """The records of every position, from az_dwarf_value_probe's each=PATH files."""
  records = []
  for path in paths:
    with open(path) as f:
      records.extend(json.loads(line) for line in f if line.strip())
  return records


def _units(records, cluster):
  """Each record's resampling unit: its game, or (cluster=False, for the tests' control)
  the position itself — the same position under every network."""
  ordinal, units = {}, []
  for r in records:
    if cluster:
      units.append(("game", r["game"]))
    else:
      key = (r["net"], r["kind"])
      ordinal[key] = ordinal.get(key, -1) + 1
      units.append((r["kind"], ordinal[key]))
  return units


def summarize(records, compare=(), resamples=2000, seed=1, cluster=True):
  """{"nets": [...], "comparisons": [...]}: per network and context the measures as
  [estimate, low, high]; per comparison A-B and context the differences A - B so."""
  rng = np.random.default_rng(seed)
  units = _units(records, cluster)
  out = {"nets": [], "comparisons": []}
  for context in CONTEXTS:
    chosen = [i for i, r in enumerate(records) if r["kind"].startswith(context + "_")]
    if not chosen:
      continue
    nets = sorted({records[i]["net"] for i in chosen})  # Those measured in this context.
    index = {u: k for k, u in enumerate(sorted({units[i] for i in chosen}))}
    n = len(index)
    # Per network, per unit: the sums and counts of dwarf and shift on each side.
    sums = {net: np.zeros((8, n)) for net in nets}
    for i in chosen:
      r, u = records[i], index[units[i]]
      side = 0 if r["kind"].endswith("dwarfs_to_move") else 4
      s = sums[r["net"]]
      s[side, u] += r["dwarf"]
      s[side + 1, u] += 1
      if r.get("shift") is not None:
        s[side + 2, u] += r["shift"]
        s[side + 3, u] += 1
    # Rows: the full data, then the resamples (counts of each unit drawn).
    weights = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=resamples)])

    def measures(net):
      t = weights @ sums[net].T  # (1 + resamples) x 8
      with np.errstate(divide="ignore", invalid="ignore"):
        wd, wt = t[:, 0] / t[:, 1], t[:, 4] / t[:, 5]
        sd, st = t[:, 2] / t[:, 3], t[:, 6] / t[:, 7]
        return {"worth_dwarfs": wd, "worth_trolls": wt, "ratio": wd / wt,
                "ratio_net": (wd - sd) / (wt - st)}

    def interval(x):
      rest = x[1:][np.isfinite(x[1:])]
      lo, hi = np.percentile(rest, [2.5, 97.5]) if len(rest) else (np.nan, np.nan)
      return [float(x[0]), float(lo), float(hi)]

    m = {net: measures(net) for net in nets}
    for net in nets:
      s = sums[net]
      out["nets"].append(dict(
          net=net, context=context, units=n, positions_dwarfs=int(s[1].sum()),
          positions_trolls=int(s[5].sum()),
          **{k: interval(m[net][k]) for k in MEASURES}))
    for pair in compare:
      a, b = pair.split("-", 1)
      if a not in m or b not in m:
        continue
      out["comparisons"].append(dict(
          compare=f"{a}-{b}", context=context,
          **{k + "_diff": interval(m[a][k] - m[b][k]) for k in MEASURES}))
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("each", nargs="+")
  parser.add_argument("--compare", action="append", default=[],
                      help="A-B: the differences A - B, paired (repeatable)")
  parser.add_argument("--resamples", type=int, default=2000)
  parser.add_argument("--seed", type=int, default=1)
  args = parser.parse_args()
  records = load(args.each)
  nets = {r["net"] for r in records}
  for pair in args.compare:
    for name in pair.split("-", 1):
      if name not in nets:
        sys.exit(f"--compare {pair}: no network {name} (have {sorted(nets)})")
  result = summarize(records, args.compare, args.resamples, args.seed)
  number = lambda x: round(x, 5) if np.isfinite(x) else None  # JSON has no NaN.
  for line in result["nets"] + result["comparisons"]:
    print(json.dumps({k: [number(x) for x in v] if isinstance(v, list) else v
                      for k, v in line.items()}))


if __name__ == "__main__":
  main()
