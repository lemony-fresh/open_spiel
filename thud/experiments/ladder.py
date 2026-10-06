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

"""Ratings from all our matches: the ladder (thud/PLAN.md, How we evaluate networks).

Reads every az_match output in a directory (one JSON line per pair, then a summary)
and fits one rating per player by least squares on every game's margin: a player has
a dwarf strength D and a troll strength T, and a battle with x as the dwarfs and y as
the trolls is expected to end at D_x - T_y (x's margin). The anchor, E44 by default,
is fixed at D = T = 0, so a player's rating R = D + T is its expected margin per pair
(both sides) against the anchor. Matches against the untrained network A0 are left
out by default: beating a near-random opponent does not predict strength between
strong networks (2026-10-03: with them in, the misfits were those matches — A14, which
beats our strong networks, beats A0 by less than broad searches do); they remain
per-side diagnostics.
A player is a network as it searched: the same network with another rule for untried
moves is another player; "as trained" is each run's own rule (upstream's for run A's
family, sibling_mean_minus_reduction since run C). Standard errors come from the
residual variance. The fit check compares each match's mean pair margin with the
difference of the two ratings — large misfits mean the margins do not add up
(non-transitive results, or a ceiling: margins against a weak anchor saturate, which
is why each network should play the anchors nearest its strength).

With --link tanh (the default) a battle's expected margin is 32 tanh((D_x - T_y) / 32)
instead: margins cannot pass ±32 (all of one side captured), and strong networks
against a weak one crowd below that ceiling, which a linear model reads as smaller
differences than they are. Ratings are then on that model's latent scale (equal to
margins for small differences). By default the table lists each network as trained;
--all adds the same networks searching with another rule (separate players, fitted
always). A player whose games were all against the anchor is marked *: such a rating
rests on beating a near-random opponent and need not transfer.

--anchor NAME fixes another player at 0; --exclude NAME,NAME drops every match of
those players (default: A0 and the ambiguous "latest" checkpoints; --exclude '' keeps
every match, with --anchor A0).

  python3 thud/experiments/ladder.py [MATCH_DIR] [--link tanh|linear] [--all]
      [--anchor NAME] [--exclude NAME,...] [--misfits N]
"""

import glob
import json
import math
import os
import sys

import numpy as np
from scipy import optimize

RUNS = {  # Run directory -> short name, and its own rule for untried moves.
    "stage1_sims100": ("A", "upstream"),
    "stage1_sims400": ("B", "upstream"),
    "stage1_sims100_buffer2x": ("A2x", "upstream"),
    "stage3a_sims100": ("C", "sibling_mean_minus_reduction"),
    "stage2_buffer4x": ("D", "sibling_mean_minus_reduction"),
    "stage3b_E_7x": ("E", "sibling_mean_minus_reduction"),
    "stage5_conv_C": ("C'", "sibling_mean_minus_reduction"),
    "stage5b_G_7x": ("G", "sibling_mean_minus_reduction"),
    "stage5b_Gaug_7x": ("G'", "sibling_mean_minus_reduction"),
    "stage5b_W_7x": ("W", "sibling_mean_minus_reduction"),
}
ANCHOR = "E44"  # The default; --anchor chooses another.
# By default the untrained network and the ambiguous "latest" checkpoints are left out.
EXCLUDED = "A0,A(latest),B(latest)"


def player(spec, rule, sims):
  run, step = spec.rstrip("/").split("/")[-1].rsplit(":", 1)
  short, own = RUNS.get(run, (run, "sibling_mean_minus_reduction"))
  rule = rule or "upstream"  # Matches before the rule existed searched as upstream.
  step = "(latest)" if step == "-1" else step
  name = f"{short}{step}" if len(short.rstrip("'")) == 1 else f"{short}:{step}"
  if rule != own:
    name += f" [{'upstream rule' if rule == 'upstream' else rule}]"
  if sims != 100:
    name += f" @{sims}"
  return name


def load(directory):
  matches = []
  for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
    lines = [json.loads(l) for l in open(path) if l.strip()]
    summaries = [l for l in lines if l.get("summary")]
    if not summaries or "a_mean_pair_margin" not in summaries[-1]:
      continue  # Not a match.
    s = summaries[-1]
    a = player(s["a"], s.get("untried_a", s.get("untried")), s.get("sims_a", s.get("sims", 100)))
    b = player(s["b"], s.get("untried_b", s.get("untried")), s.get("sims_b", s.get("sims", 100)))
    pairs = [l for l in lines if "pair" in l and "a_as_dwarfs" in l]
    matches.append({"name": os.path.basename(path)[:-6], "a": a, "b": b, "pairs": pairs,
                    "margin": s["a_mean_pair_margin"], "ci95": s.get("ci95")})
  return matches


class Fit:
  """The ladder's ratings for a set of matches (see fit())."""


def fit(matches, anchor, link="tanh"):
  """Fits every player's dwarf and troll strengths to the matches' games, `anchor` at 0.

  Returns a Fit with: players; rating(p) -> (R, its standard error, D, its standard
  error, T, its standard error); side(p, "D" or "T"); expected(z), a battle's expected
  margin for D_dwarfs - T_trolls; sigma2, the residual variance of a game; games and
  opponents per player; the number of games.
  """
  players = sorted({m["a"] for m in matches} | {m["b"] for m in matches})
  if anchor not in players:
    raise ValueError(f"no matches against the anchor {anchor}")
  free = [p for p in players if p != anchor]
  col = {}  # Unknowns: D and T of every player but the anchor.
  for p in free:
    col[("D", p)] = len(col)
    col[("T", p)] = len(col)
  rows, ys = [], []

  def add(dwarfs, trolls, margin):  # margin = D_dwarfs - T_trolls
    row = np.zeros(len(col))
    if dwarfs != anchor:
      row[col[("D", dwarfs)]] += 1
    if trolls != anchor:
      row[col[("T", trolls)]] -= 1
    rows.append(row)
    ys.append(margin)

  for m in matches:
    for p in m["pairs"]:
      add(m["a"], m["b"], p["a_as_dwarfs"])     # a as the dwarfs
      add(m["b"], m["a"], -p["a_as_trolls"])    # b as the dwarfs: b's margin
  X, y = np.array(rows), np.array(ys)
  coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
  result = Fit()
  result.undetermined = X.shape[1] - rank

  def expected(z):  # A battle's expected margin from D_dwarfs - T_trolls.
    return 32 * np.tanh(z / 32) if link == "tanh" else z

  if link == "tanh":
    solved = optimize.least_squares(lambda c: expected(X @ c) - y, np.clip(coef, -60, 60))
    coef = solved.x
    J = solved.jac
  else:
    J = X
  residual = y - expected(X @ coef)
  sigma2 = residual @ residual / max(1, len(y) - rank)
  cov = sigma2 * np.linalg.pinv(J.T @ J)

  def side(p, s):  # D or T of a player; the anchor's are 0.
    return 0.0 if p == anchor else coef[col[(s, p)]]

  def rating(p):
    if p == anchor:
      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    d, t = col[("D", p)], col[("T", p)]
    r = coef[d] + coef[t]
    se = math.sqrt(max(0.0, cov[d, d] + cov[t, t] + 2 * cov[d, t]))
    return r, se, coef[d], math.sqrt(max(0.0, cov[d, d])), coef[t], math.sqrt(max(0.0, cov[t, t]))

  games = {p: 0 for p in players}
  opponents = {p: set() for p in players}
  for m in matches:
    games[m["a"]] += 2 * len(m["pairs"])
    games[m["b"]] += 2 * len(m["pairs"])
    opponents[m["a"]].add(m["b"])
    opponents[m["b"]].add(m["a"])
  result.players, result.rating, result.side, result.expected = players, rating, side, expected
  result.sigma2, result.games, result.opponents, result.n_games = sigma2, games, opponents, len(y)
  return result


def main():
  argv = sys.argv[1:]
  options = {"--misfits": "10", "--link": "tanh", "--anchor": ANCHOR,
             "--exclude": EXCLUDED}
  for key in options:
    if key in argv:
      i = argv.index(key)
      options[key] = argv[i + 1]
      del argv[i:i + 2]
  misfits, link, anchor = int(options["--misfits"]), options["--link"], options["--anchor"]
  excluded = set(filter(None, options["--exclude"].split(",")))
  show_all = "--all" in argv
  argv = [a for a in argv if a != "--all"]
  directory = argv[0] if argv else os.path.expanduser("~/thud-runs/stage1_matches")
  matches = [m for m in load(directory)
             if m["a"] not in excluded and m["b"] not in excluded]
  try:
    f = fit(matches, anchor, link)
  except ValueError as e:
    sys.exit(str(e))
  if f.undetermined:
    print(f"warning: {f.undetermined} ratings not determined (players not connected)")
  print(f"{len(matches)} matches, {f.n_games} games, {len(f.players)} players; link {link}; "
        f"anchor {anchor} = 0; a game's residual standard deviation {math.sqrt(f.sigma2):.1f}")
  print(f"{'player':30s} {'rating':>14s} {'as dwarfs':>14s} {'as trolls':>14s} {'games':>6s}")
  for p in sorted(f.players, key=lambda q: -f.rating(q)[0]):
    if not show_all and "[" in p:
      continue
    r, rs, d, ds, t, ts = f.rating(p)
    mark = " *" if f.opponents[p] == {anchor} else ""
    print(f"{p + mark:30s} {r:+7.1f} ±{1.96 * rs:4.1f} {d:+7.1f} ±{1.96 * ds:4.1f} "
          f"{t:+7.1f} ±{1.96 * ts:4.1f} {f.games[p]:6d}")
  print(f"\nFit check, the {misfits} largest misfits (observed mean pair margin against "
        f"the ratings' difference):")
  checks = []
  for m in matches:
    a, b = m["a"], m["b"]
    predicted = (f.expected(f.side(a, "D") - f.side(b, "T"))
                 - f.expected(f.side(b, "D") - f.side(a, "T")))
    lo, hi = m["ci95"] if m["ci95"] else (float("nan"), float("nan"))
    checks.append((abs(m["margin"] - predicted), m, predicted, lo <= predicted <= hi))
  inside = sum(c[3] for c in checks)
  print(f"  {inside} of {len(checks)} matches have the predicted margin inside their 95% interval")
  for gap, m, predicted, ok in sorted(checks, key=lambda c: -c[0])[:misfits]:
    print(f"  {m['name']:42s} observed {m['margin']:+6.1f} predicted {predicted:+6.1f}"
          f"{'' if ok else '  (outside its interval)'}")


if __name__ == "__main__":
  main()
