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

"""The game analyser (thud/PLAN.md, Instrumentation, item 6).

Replays a run's self-play games from the actors' logs (log-actor-*.txt: one line per
game with its returns, how it ended and its moves) and reports, per group of `--by`
steps, how the games were played: per side the captures — the dwarfs' hurls and the
trolls' capture steps and shoves with the dwarfs they took — and when each
side first captured; the dwarfs' final margin and how games ended; and how varied the
openings are (distinct sequences of the first 2, 4, 8 and 16 moves, as a share of the
games). A game belongs to the step whose new games it was: it was logged after the
learner's "Step: k-1" line and before its "Step: k" (log-learner.txt), so it was
played by checkpoint k-1, mostly — the trainer's own per-step statistics count the
same games.

Two limits of the logs: upstream logs only the first 20 actors (to limit open files),
so with more actors the logs are a sample; and until 2026-10-05 the trainer started the
actor logs anew on every resume, so older runs' logs hold only their latest segment
(since then they append, each segment opening with "actor-N started").

Every replay is checked: for a game that ended on the board (not cut off: self-play
ends some games early once the search's value passes the trainer's cutoff_value), the replayed final material must give the logged returns, and the
captures counted move by move must add up to the pieces missing at the end.

  python3 thud/experiments/analyse_games.py RUN_DIR [--by 4] [--jsonl FILE]
"""

import argparse
import collections
import datetime
import glob
import json
import os
import re

import numpy as np
import pyspiel

# Our trainer logs how a game ended since 2026-10-03; upstream's line has no "Ending:".
GAME_LINE = re.compile(r"^\[([^\]]+)\] Game (\d+): Returns: (\S+) (\S+);(?: Ending: ([^;]+);)? "
                       r"Actions: ?(.*)$")
STEP_LINE = re.compile(r"^\[([^\]]+)\] Step: (\d+)$")
MOVE = re.compile(r"^\((\d+),(\d+)\)-\((\d+),(\d+)\)(x?)$")
MAX_MARGIN, DWARFS, TROLLS, TROLL_VALUE = 32, 32, 8, 4
OPENINGS = (2, 4, 8, 16)


def timestamp(text):
  return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")


def pieces(state):
  board = str(state).split("\n")[:15]  # The last line ("to_move=dwarfs ...") is not board.
  return sum(row.count("d") for row in board), sum(row.count("T") for row in board)


def ending(state, final, limits):
  """How a replayed game ended, by the trainer's rules (GameEnding in thud/az/alpha_zero.cc)."""
  if not state.is_terminal():
    return "cutoff"
  counters = dict(f.split("=") for f in str(state).split("\n")[-1].split()[1:])
  if int(counters["turns"]) >= limits["max_turns"]:
    return "turn_limit"
  if int(counters["turns_without_capture"]) >= limits["max_turns_without_capture"]:
    return "no_capture_limit"
  return "dwarfs_gone" if final[0] == 0 else "trolls_gone" if final[1] == 0 else "no_legal_move"


def replay(game, moves):
  """One game's statistics, from its moves; checks each move is legal."""
  state = game.new_initial_state()
  stats = collections.Counter()
  first = {}
  dwarfs, trolls = DWARFS, TROLLS
  for turn, move in enumerate(moves):
    side = "dwarfs" if state.current_player() == 0 else "trolls"
    action = state.string_to_action(move)  # Fails if no legal move prints as `move`.
    r0, c0, r1, c1, capture = MOVE.match(move).groups()
    distance = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)))
    state.apply_action(action)
    stats[f"{side}_moves"] += 1
    if not capture:
      continue
    first.setdefault(side, turn)
    after_dwarfs, after_trolls = pieces(state)
    if side == "dwarfs":
      assert after_trolls == trolls - 1, f"turn {turn}: a hurl captured {trolls - after_trolls}"
      stats["hurls"] += 1
      stats["hurls_long"] += distance >= 2
    else:
      stats["capture_steps" if distance == 1 else "shoves"] += 1
      stats["dwarfs_lost"] += dwarfs - after_dwarfs
    dwarfs, trolls = after_dwarfs, after_trolls
  final = pieces(state)
  assert stats["hurls"] == TROLLS - final[1] and stats["dwarfs_lost"] == DWARFS - final[0], (
      "captures counted move by move do not add up to the pieces missing")
  return stats, first, final, state


def load_games(run_dir):
  steps = {}
  for line in open(os.path.join(run_dir, "log-learner.txt")):
    m = STEP_LINE.match(line.strip())
    if m:
      steps[int(m.group(2))] = timestamp(m.group(1))  # The latest segment's, if repeated.
  ends = sorted(steps.items())
  games = []
  for path in sorted(glob.glob(os.path.join(run_dir, "log-actor-*.txt"))):
    actor = int(re.search(r"log-actor-(\d+)", path).group(1))
    for line in open(path):
      m = GAME_LINE.match(line.strip())
      if not m:
        continue
      when = timestamp(m.group(1))
      step = next((k for k, t in ends if when <= t), ends[-1][0] + 1 if ends else 1)
      games.append({"actor": actor, "game": int(m.group(2)), "step": step,
                    "returns": (float(m.group(3)), float(m.group(4))),
                    "ending": m.group(5), "moves": m.group(6).split()})
  last_step = ends[-1][0] if ends else 0
  return games, last_step


def summarise(group, game):
  n = len(group)
  stats = [g["stats"] for g in group]
  played = [g for g in group if g["on_board"]]

  def mean(key):
    return float(np.mean([s[key] for s in stats]))

  def first(side):
    turns = [g["first"][side] for g in group if side in g["first"]]
    return (float(np.median(turns)) if turns else None), len(turns) / n

  out = {"steps": [min(g["step"] for g in group), max(g["step"] for g in group)],
         "games": n, "turns": mean("dwarfs_moves") + mean("trolls_moves"),
         "dwarfs_margin": (float(np.mean([g["margin"] for g in played])) if played else None),
         "dwarfs_margin_sd": (float(np.std([g["margin"] for g in played])) if played else None),
         "endings": dict(collections.Counter(g["ending"] for g in group)),
         "hurls": mean("hurls"), "hurls_long": mean("hurls_long"),
         "capture_steps": mean("capture_steps"), "shoves": mean("shoves"),
         "dwarfs_lost": mean("dwarfs_lost")}
  troll_captures = sum(s["capture_steps"] + s["shoves"] for s in stats)
  out["dwarfs_per_troll_capture"] = (sum(s["dwarfs_lost"] for s in stats) / troll_captures
                                     if troll_captures else None)
  out["first_hurl_turn"], out["games_with_hurl"] = first("dwarfs")
  out["first_troll_capture_turn"], out["games_with_troll_capture"] = first("trolls")
  out["distinct_openings"] = {k: len({tuple(g["moves"][:k]) for g in group}) / n
                              for k in OPENINGS}
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("run_dir")
  parser.add_argument("--by", type=int, default=4, help="steps per group")
  parser.add_argument("--jsonl", default=None, help="also write one JSON line per group")
  args = parser.parse_args()
  try:  # The game as the run played it (its limits decide how a game ended).
    game = pyspiel.load_game(json.load(open(os.path.join(args.run_dir, "config.json")))["game"])
  except OSError:
    game = pyspiel.load_game("thud")
  games, last_step = load_games(args.run_dir)
  checked = endings_checked = 0
  for g in games:
    g["stats"], g["first"], final, state = replay(game, g["moves"])
    g["margin"] = final[0] - TROLL_VALUE * final[1]
    g["on_board"] = state.is_terminal()  # Otherwise cut off.
    replayed = ending(state, final, game.get_parameters())
    assert g["ending"] in (None, replayed), (
        f"actor {g['actor']} game {g['game']}: logged {g['ending']}, replayed {replayed}")
    endings_checked += g["ending"] is not None
    g["ending"] = replayed
    if g["on_board"]:
      assert abs(g["returns"][0] - g["margin"] / MAX_MARGIN) < 1e-9, (
          f"actor {g['actor']} game {g['game']}: logged returns {g['returns']}, "
          f"replayed margin {g['margin']}")
      checked += 1
  print(f"{len(games)} games from {args.run_dir}'s actor logs, steps "
        f"{min(g['step'] for g in games)}-{max(g['step'] for g in games)} "
        f"(learner at step {last_step}; a later step is unfinished); {checked} ended on the "
        f"board and their replays reproduce the logged returns; {endings_checked} logged how "
        f"they ended, and their replays agree")
  groups = collections.defaultdict(list)
  for g in games:
    groups[(g["step"] - 1) // args.by].append(g)
  rows = [summarise(groups[k], game) for k in sorted(groups)]
  print("\nPer game, means: the dwarfs' margin (± sd, games ended on the board); captures "
        "per side; median turn of a side's first capture (share of games with one); "
        "distinct openings of 2/4/8/16 moves, as a share of the games")
  print(f"{'steps':>7s} {'games':>5s} {'turns':>5s} {'margin':>11s} | {'hurls':>5s} "
        f"{'long':>4s} | {'cap.steps':>9s} {'shoves':>6s} "
        f"{'dwarfs lost':>11s} {'per capture':>11s} | {'1st hurl':>12s} "
        f"{'1st troll cap.':>14s} | openings")
  for r in rows:
    def opt(x, fmt):
      return format(x, fmt) if x is not None else "-"
    print(f"{r['steps'][0]:>3d}-{r['steps'][1]:<3d} {r['games']:5d} {r['turns']:5.0f} "
          f"{opt(r['dwarfs_margin'], '+5.1f')} ±{opt(r['dwarfs_margin_sd'], '4.1f')} | "
          f"{r['hurls']:5.2f} {r['hurls_long']:4.2f} | "
          f"{r['capture_steps']:9.2f} {r['shoves']:6.2f} {r['dwarfs_lost']:11.2f} "
          f"{opt(r['dwarfs_per_troll_capture'], '11.2f')} | "
          f"{opt(r['first_hurl_turn'], '4.0f')} ({r['games_with_hurl']:4.0%}) "
          f"{opt(r['first_troll_capture_turn'], '6.0f')} ({r['games_with_troll_capture']:4.0%}) | "
          + "/".join(f"{r['distinct_openings'][k]:.0%}" for k in OPENINGS))
  for r in rows:
    print(f"  steps {r['steps'][0]}-{r['steps'][1]} endings: "
          + ", ".join(f"{e} {c}" for e, c in sorted(r["endings"].items(), key=lambda x: -x[1])))
  if args.jsonl:
    with open(args.jsonl, "w") as f:
      for r in rows:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
  main()
