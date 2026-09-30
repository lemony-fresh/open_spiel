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

"""What more simulations gain each side, from two az_match runs on the same openings.

Do the dwarfs gain more from extra simulations than the trolls? (thud/PLAN.md Phase 6,
stage 3b: whether playout caps need separate settings per side.) Two matches of one
network against itself, with the same seed and opening length, so that pair i starts from
the same opening in both:

  UNEVEN    a with more simulations (sims_a), b with fewer (sims_b)
  BASELINE  both with the fewer (sims_a = sims_b = UNEVEN's sims_b)

Per opening i, in points (az_match's a_as_dwarfs and a_as_trolls are network A's margin
on that side):

  baseline dwarfs' margin  B_i   = (a_as_dwarfs_i - a_as_trolls_i) / 2 in BASELINE
  dwarfs' gain             G_d,i = UNEVEN's a_as_dwarfs_i - B_i
  trolls' gain             G_t,i = UNEVEN's a_as_trolls_i + B_i

A BASELINE pair is one game played twice when the searches are deterministic; the script
counts the pairs whose two battles differ (a control), and averaging them uses both. It
prints the means of G_d, G_t, G_d - G_t and G_d + G_t (= UNEVEN's pair margin) over the
openings, with 95% Student's t intervals.

  python3 thud/experiments/az_sims_gain.py UNEVEN.jsonl BASELINE.jsonl
  python3 thud/experiments/az_sims_gain.py --self-test
"""

import json
import math
import random
import sys

from scipy import stats


def rules(summary):
  """The two networks' rules for untried moves (one "untried" before 2026-09-29)."""
  return (summary.get("untried_a", summary.get("untried")),
          summary.get("untried_b", summary.get("untried")))


def read_pairs(path):
  """Pair index -> (a_as_dwarfs, a_as_trolls), and the summary line."""
  pairs, summary = {}, None
  with open(path) as f:
    for line in f:
      row = json.loads(line)
      if row.get("summary"):
        summary = row
      else:
        pairs[row["pair"]] = (row["a_as_dwarfs"], row["a_as_trolls"])
  return pairs, summary


def mean_interval(values):
  n = len(values)
  m = sum(values) / n
  if n < 2:
    return m, m, m
  sd = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
  half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
  return m, m - half, m + half


def gains(uneven, baseline):
  """Per opening: dwarfs' gain, trolls' gain; and the baseline pairs that differ."""
  common = sorted(set(uneven) & set(baseline))
  g_d, g_t, differing = [], [], 0
  for i in common:
    u_d, u_t = uneven[i]
    b_as_dwarfs, b_as_trolls = baseline[i]
    differing += b_as_dwarfs != -b_as_trolls
    b = (b_as_dwarfs - b_as_trolls) / 2
    g_d.append(u_d - b)
    g_t.append(u_t + b)
  return g_d, g_t, differing


def report(g_d, g_t, differing):
  print(f"openings: {len(g_d)}; baseline pairs whose two battles differ: {differing}")
  rows = [("dwarfs' gain", g_d), ("trolls' gain", g_t),
          ("dwarfs' minus trolls' gain", [d - t for d, t in zip(g_d, g_t)]),
          ("sum (uneven pair margin)", [d + t for d, t in zip(g_d, g_t)])]
  for name, values in rows:
    m, lo, hi = mean_interval(values)
    print(f"  {name:28s} {m:+6.2f} points  (95%: {lo:+6.2f} to {hi:+6.2f})")


def self_test():
  """Synthetic matches with known gains: dwarfs +3, trolls +1."""
  rng = random.Random(1)
  baseline, uneven = {}, {}
  for i in range(50):
    b = rng.randint(-20, 10)  # The dwarfs' margin at the fewer simulations.
    baseline[i] = (b, -b)  # Deterministic: the same game twice.
    uneven[i] = (b + 3, -b + 1)
  baseline[50], uneven[50] = (4, -2), (6, 0)  # Battles differ: B = 3.
  g_d, g_t, differing = gains(uneven, baseline)
  ok = (all(abs(x - 3) < 1e-9 for x in g_d[:50]) and
        all(abs(x - 1) < 1e-9 for x in g_t[:50]) and
        (g_d[50], g_t[50]) == (3, 3) and differing == 1)
  report(g_d, g_t, differing)
  print("self-test passed" if ok else "SELF-TEST FAILED")
  return 0 if ok else 1


def main(argv):
  if argv[1:] == ["--self-test"]:
    return self_test()
  if len(argv) != 3:
    print(__doc__)
    return 1
  uneven, u_summary = read_pairs(argv[1])
  baseline, b_summary = read_pairs(argv[2])
  for name, s in (("uneven", u_summary), ("baseline", b_summary)):
    if s:
      print(f"{name}: a={s['a']} sims_a={s.get('sims_a')} b={s['b']} "
            f"sims_b={s.get('sims_b')} untried={rules(s)} pairs={s['pairs']}")
  if (u_summary and b_summary and
      (u_summary["a"] != b_summary["a"] or rules(u_summary) != rules(b_summary)
       or b_summary.get("sims_a") != u_summary.get("sims_b")
       or b_summary.get("sims_b") != u_summary.get("sims_b"))):
    print("WARNING: the baseline is not the uneven match's network and rule at the "
          "fewer simulations on both sides")
  report(*gains(uneven, baseline))
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
