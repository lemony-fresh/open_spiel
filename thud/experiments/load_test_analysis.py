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

"""Does a load beside a training run slow the trainer? (thud/PLAN.md, Instrumentation, 5.)

Reads the blocks of load_toggle.py (with the load running or paused, in pairs of one
each in random order) and the run's logs, and measures the trainer's self-play speed in
each block: the seconds per position of the games an actor played entirely inside the
block (the actors' logs time every game's end, and a game starts where the actor's
previous one ended). A game that overlaps the learner's learning phase is slower (~65%
for a full overlap, 2026-10-05), so each game's seconds per position is corrected for its
overlap, by a regression over all the games. Per pair: the relative difference of the
two blocks' means; over the pairs: its mean with a 95% interval (t). Also per block: the
trainer's and the load's CPU time (from load_toggle.py), and the learning phases'
durations by block.

A planted effect checks the analysis: --plant F multiplies the seconds per position of
the games in the "on" blocks by F; --schedule makes up blocks (pairs of BLOCK seconds,
random order) for logs that have none, e.g. an old run's. Pairs whose "on" block had
almost no load running (--min_load_cores) are left out:

  python3 thud/experiments/load_test_analysis.py RUN_DIR BLOCKS.jsonl [MORE_BLOCKS.jsonl]
  python3 thud/experiments/load_test_analysis.py RUN_DIR --schedule 1200 [--plant 1.05]
"""

import argparse
import datetime
import glob
import json
import os
import random
import re

import numpy as np
from scipy import stats

ACTOR_LINE = re.compile(r"^\[([^\]]+)\] (actor-\d+ started|Game \d+: .*Actions: ?(.*))$")
LEARNER_LINE = re.compile(r"^\[([^\]]+)\] (.*)$")


def timestamp(text):
  return datetime.datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def games_of(run_dir):
  """Every logged game as (start, end, positions); a game starts at the actor's previous end."""
  games = []
  for path in glob.glob(os.path.join(run_dir, "log-actor-*.txt")):
    previous = None
    for line in open(path):
      m = ACTOR_LINE.match(line.strip())
      if not m:
        continue
      t = timestamp(m.group(1))
      if "started" not in m.group(2) and previous is not None:
        games.append((previous, t, len(m.group(3).split())))
      previous = t
  return np.array(games)


def learning_phases(run_dir):
  """The learner's learning phases: from its "Step: k" line to "Checkpoint saved"."""
  phases, start = [], None
  for line in open(os.path.join(run_dir, "log-learner.txt")):
    m = LEARNER_LINE.match(line.strip())
    if not m:
      continue
    if m.group(2).startswith("Step: "):
      start = timestamp(m.group(1))
    elif m.group(2).startswith("Checkpoint saved") and start is not None:
      phases.append((start, timestamp(m.group(1))))
      start = None
  return phases


def overlap(a, b, phases):
  return sum(max(0.0, min(e, b) - max(s, a)) for s, e in phases)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("run_dir")
  parser.add_argument("blocks", nargs="*",
                      help="load_toggle.py's blocks (JSON lines); several files keep their pairs apart")
  parser.add_argument("--schedule", type=float, default=None,
                      help="make up pairs of blocks of this many seconds instead")
  parser.add_argument("--plant", type=float, default=1.0,
                      help="multiply the on blocks' seconds per position by this")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--min_load_cores", type=float, default=0.5,
                      help="skip pairs whose on block's load used fewer cores (none running)")
  args = parser.parse_args()
  games = games_of(args.run_dir)
  phases = learning_phases(args.run_dir)
  if args.schedule:
    rng = random.Random(args.seed)
    blocks, t, pair = [], games[:, 0].min() + 600, 0
    while t + 2 * args.schedule < games[:, 1].max():
      for load in rng.sample(["on", "off"], 2):
        blocks.append({"pair": pair, "load": load, "start": t, "end": t + args.schedule})
        t += args.schedule
      pair += 1
  else:
    blocks = []
    for n, path in enumerate(args.blocks):  # A file's pairs numbered apart from another's.
      for l in open(path):
        if l.strip():
          b = json.loads(l)
          b["pair"] += 1000 * n
          blocks.append(b)
    blocks = [b for b in blocks if "end" in b]  # Complete blocks only.

  seconds = (games[:, 1] - games[:, 0]) / games[:, 2]  # Per position.
  learn = np.array([overlap(s, e, phases) / (e - s) for s, e, _ in games])
  block_of = np.full(len(games), -1)
  for i, b in enumerate(blocks):
    block_of[(games[:, 0] >= b["start"]) & (games[:, 1] <= b["end"])] = i
  inside = block_of >= 0
  on = np.array([inside[g] and blocks[block_of[g]]["load"] == "on" for g in range(len(games))])
  seconds = np.where(on, seconds * args.plant, seconds)
  # The learning phases' cost, fitted with a block term each so the load cannot leak in.
  X = np.zeros((inside.sum(), len(blocks) + 1))
  X[np.arange(inside.sum()), block_of[inside]] = 1
  X[:, -1] = learn[inside]
  coef, *_ = np.linalg.lstsq(X, seconds[inside], rcond=None)
  corrected = seconds - coef[-1] * learn
  base = np.mean(corrected[inside])
  print(f"{len(games)} games, {inside.sum()} inside {len(blocks)} blocks; {base:.3f} s a position "
        f"without learning; a full overlap with learning adds {coef[-1] / base:.0%}")

  pairs = {}
  for i, b in enumerate(blocks):
    mine = block_of == i
    b["games"] = int(mine.sum())
    b["seconds"] = float(np.mean(corrected[mine])) if mine.any() else None
    pairs.setdefault(b["pair"], {})[b["load"]] = b
  diffs = []
  print(f"\n{'pair':>4s} {'on: games':>9s} {'s/pos':>6s} {'off: games':>10s} {'s/pos':>6s} "
        f"{'slower':>7s}   load CPU on/off (cores)  trainer CPU on/off (cores)")
  for p, both in sorted(pairs.items()):
    if set(both) != {"on", "off"} or not all(both[k]["games"] >= 5 for k in both):
      continue
    a, b = both["on"], both["off"]
    if "load_cpu_s" in a and a["load_cpu_s"] < args.min_load_cores * (a["end"] - a["start"]):
      continue  # No load to switch: e.g. the evaluation watcher was waiting for a checkpoint.
    d = a["seconds"] / b["seconds"] - 1
    diffs.append(d)

    def cores(x, key):
      return f"{x[key] / (x['end'] - x['start']):4.1f}" if key in x else "   -"
    print(f"{p:4d} {a['games']:9d} {a['seconds']:6.3f} {b['games']:10d} {b['seconds']:6.3f} "
          f"{d:+7.1%}   {cores(a, 'load_cpu_s')} / {cores(b, 'load_cpu_s')}"
          f"                {cores(a, 'trainer_cpu_s')} / {cores(b, 'trainer_cpu_s')}")
  d = np.array(diffs)
  if len(d) >= 2:
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nWith the load, self-play is {d.mean():+.1%} slower a position "
          f"(95% interval {d.mean() - half:+.1%} to {d.mean() + half:+.1%}; {len(d)} pairs"
          f"{'; planted ' + format(args.plant - 1, '+.1%') if args.plant != 1 else ''})")
  for load in ("on", "off"):
    durations = [e - s for s, e in phases
                 for b in blocks if b["load"] == load and s >= b["start"] and e <= b["end"]]
    if durations:
      print(f"learning phases inside {load} blocks: {len(durations)}, mean {np.mean(durations):.0f} s")


if __name__ == "__main__":
  main()
