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

"""The positional terms' weights, fitted to our own games' results (thud/PLAN.md, *The
teacher's finer evaluation*, gate 1): at every position of every game, the battle's final
margin (dwarfs' view) regressed on the material now, the side to move and the six terms of
teacher_terms.py (least squares). Every 5th game is held out; the weights' 95% intervals
resample whole games (a game's positions share its fate); a term is kept when its interval
excludes 0, and the held-out games say whether the terms predict the result better than
material alone (the mean squared error's gain, its interval resampling the held-out games).

Usage:
  terms_fit.py OUT_DIR --run RUN_DIR OFFSETS_JSON [--matches GAMES.jsonl ...]
    RUN_DIR/log-actor-*.txt read from OFFSETS_JSON's offsets to their ends (the self-play
    games; games against alpha-beta skipped); each GAMES.jsonl an az_match game file. Each
    set fitted on its own, on every position and on the first 100 plies (where captures
    still come: half of self-play's positions lie in the final quiet stretch, where the
    result is settled and no term can matter). The positions are kept in
    OUT_DIR/positions_SET.jsonl, the fits in OUT_DIR/fits.jsonl, a report on stdout.

The replay checks itself: each game's final board must give its recorded margin.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_terms  # noqa: E402

LINE = re.compile(r"Game \d+: Returns: (\S+) (\S+); Ending: (\w+); "
                  r"(?:Opponent: ([^;]+); )?Actions: (.*)$")
EARLY_PLIES = 100


class ReplayError(Exception):
  pass


def parse_log_line(line):
  """A self-play game of an actor's log line ({"margin", "actions"}), or None (another
  line, or a game against alpha-beta)."""
  m = LINE.search(line.rstrip("\n"))
  if not m or m.group(4):
    return None
  return {"margin": round(float(m.group(1)) * 32), "actions": m.group(5).split()}


def match_game(record):
  """An az_match game record as {"margin" (the dwarfs' view), "actions"}."""
  margin = record["a_margin"] if record["a_player"] == 0 else -record["a_margin"]
  return {"margin": margin, "actions": record["actions"]}


def _material(board):
  return sum(row.count("d") for row in board) - 4 * sum(row.count("T") for row in board)


_GAME = None


def positions(game):
  """The positions before each move of `game`, each {"ply", "trolls_to_move", "material",
  "terms", "margin"}; raises ReplayError if the final board does not give the margin."""
  global _GAME
  import pyspiel
  if _GAME is None:
    _GAME = pyspiel.load_game("thud")
  state = _GAME.new_initial_state()
  out = []
  for ply, action in enumerate(game["actions"]):
    board = teacher_terms.parse(str(state))
    out.append({"ply": ply, "trolls_to_move": state.current_player(),
                "material": _material(board), "terms": teacher_terms.terms(board),
                "margin": game["margin"]})
    state.apply_action(state.string_to_action(action))
  final = _material(teacher_terms.parse(str(state)))
  if final != game["margin"]:
    raise ReplayError(f"the final board's material {final}, the record's {game['margin']}")
  return out


def _design(rows, names):
  x = np.array([[1.0, r["material"], r["trolls_to_move"]] + [r["terms"][n] for n in names]
                for r in rows])
  y = np.array([r["margin"] for r in rows], dtype=float)
  return x, y


def _solve(x, y):
  return np.linalg.lstsq(x, y, rcond=None)[0]


def fit(rows, names, boot=200, seed=0, cluster=True):
  """The fit of `rows` (each with "game") on the terms `names`: the weights with their
  95% intervals (whole training games resampled; rows one by one if not `cluster`), the
  terms kept, and the held-out games' mean squared error, material alone against the
  terms."""
  rng = np.random.default_rng(seed)
  train = [r for r in rows if r["game"] % 5 != 0]
  held = [r for r in rows if r["game"] % 5 == 0]
  x, y = _design(train, names)
  coef = _solve(x, y)
  games = sorted({r["game"] for r in train})
  index = {}
  for i, r in enumerate(train):
    index.setdefault(r["game"], []).append(i)
  draws = []
  for _ in range(boot):
    if cluster:
      pick = np.concatenate([index[g] for g in rng.choice(games, len(games))])
    else:
      pick = rng.integers(0, len(train), len(train))
    draws.append(_solve(x[pick], y[pick]))
  draws = np.array(draws)
  low, high = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
  sd = x.std(axis=0)
  weights = {}
  for i, name in enumerate(["intercept", "material", "trolls_to_move"] + list(names)):
    weights[name] = {"weight": float(coef[i]), "low": float(low[i]), "high": float(high[i]),
                     "per_sd": float(coef[i] * sd[i])}
  kept = [n for n in names if weights[n]["low"] > 0 or weights[n]["high"] < 0]
  # Held out: material alone against the terms, both fitted on the training games.
  xh, yh = _design(held, names)
  base = _solve(x[:, :3], y)
  err_base = (yh - xh[:, :3] @ base) ** 2
  err_terms = (yh - xh @ coef) ** 2
  by_game = {}
  for r, a, b in zip(held, err_base, err_terms):
    s = by_game.setdefault(r["game"], [0.0, 0.0, 0])
    s[0] += a
    s[1] += b
    s[2] += 1
  sums = np.array(list(by_game.values()))
  gains = []
  for _ in range(boot):
    pick = sums[rng.integers(0, len(sums), len(sums))]
    gains.append((pick[:, 0].sum() - pick[:, 1].sum()) / pick[:, 2].sum())
  return {"weights": weights, "kept": kept, "positions": len(train), "games": len(games),
          "held_out": {"mse_material": float(err_base.mean()), "mse_terms": float(err_terms.mean()),
                       "mse_gain": float(err_base.mean() - err_terms.mean()),
                       "low": float(np.percentile(gains, 2.5)),
                       "high": float(np.percentile(gains, 97.5)),
                       "positions": len(held), "games": len(by_game)}}


def _report(title, result, names):
  print(f"\n== {title}: {result['games']} training games, {result['positions']} positions")
  print(f"   {'term':18} {'weight':>9} {'95% interval':>21} {'per sd':>8}")
  for name in ["material", "trolls_to_move"] + list(names):
    w = result["weights"][name]
    mark = " kept" if name in result["kept"] else ""
    print(f"   {name:18} {w['weight']:+9.4f} [{w['low']:+9.4f}, {w['high']:+9.4f}] "
          f"{w['per_sd']:+8.3f}{mark}")
  h = result["held_out"]
  print(f"   held out ({h['games']} games, {h['positions']} positions): mean squared error "
        f"{h['mse_material']:.3f} material alone, {h['mse_terms']:.3f} with the terms: gain "
        f"{h['mse_gain']:+.3f} [{h['low']:+.3f}, {h['high']:+.3f}]")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("out")
  parser.add_argument("--run", nargs=2, metavar=("RUN_DIR", "OFFSETS_JSON"))
  parser.add_argument("--matches", nargs="*", default=[])
  parser.add_argument("--boot", type=int, default=200)
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)
  sets = {}
  if args.run:
    run, offsets = args.run[0], json.load(open(args.run[1]))
    games = []
    for name, start in sorted(offsets.items()):
      with open(os.path.join(run, name), "rb") as f:
        f.seek(start)
        for raw in f:
          game = parse_log_line(raw.decode("utf-8", "replace"))
          if game:
            games.append(game)
    sets["selfplay"] = games
  if args.matches:
    sets["matches"] = [match_game(json.loads(line)) for path in args.matches
                       for line in open(path) if line.strip() and '"actions"' in line]
  names = list(teacher_terms.TERMS)
  with open(os.path.join(args.out, "fits.jsonl"), "w") as fits:
    for set_name, games in sets.items():
      rows = []
      with open(os.path.join(args.out, f"positions_{set_name}.jsonl"), "w") as kept:
        for g, game in enumerate(games):
          for p in positions(game):
            p["game"] = g
            rows.append(p)
            kept.write(json.dumps(p) + "\n")
      terms = np.array([[r["terms"][n] for n in names] for r in rows])
      corr = np.corrcoef(terms.T)
      print(f"\n{set_name}: {len(games)} games, {len(rows)} positions; the terms' "
            "correlations:")
      for i, n in enumerate(names):
        print(f"   {n:18} " + " ".join(f"{corr[i, j]:+.2f}" for j in range(len(names))))
      for part, subset in (("every position", rows),
                           (f"the first {EARLY_PLIES} plies",
                            [r for r in rows if r["ply"] < EARLY_PLIES])):
        result = fit(subset, names, boot=args.boot)
        _report(f"{set_name}, {part}", result, names)
        fits.write(json.dumps({"set": set_name, "part": part, **result}) + "\n")


if __name__ == "__main__":
  main()
