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

"""The evaluation watcher (thud/PLAN.md, Instrumentation, item 5).

Beside a training run: every `--every` steps it plays the run's newest checkpoint, as
trained (the rule for untried moves from each run's config.json), against the
`--nearest` ladder anchors closest to its last rating, `--pairs` pairs each (az_match),
refits the ladder (ladder.py) and appends the checkpoint's rating — overall and
per side, against E44 — to RUN_DIR/watch.jsonl and to the progress log. A strength
curve during the run: a branch that falls behind (or drifts, as run W's trolls did) can
be stopped early. It only reports; stopping a run stays a decision.

It must not slow or endanger the run it watches. Its matches run in the idle scheduling
class (SCHED_IDLE), not at nice 19, which competes as an equal with our trainers (at
nice 19 themselves); the idle class still weighs 3 against nice 19's 15, so it gets
~1/6 of a contended core — whether that slows the trainer is load_test_analysis.py's
question (--nice runs the matches at nice 19 instead). And only with memory to spare:
its matches keep a cache of CACHE evaluations (az_match's default, 262,144, took a match
to 1.4 GB; a cache changes no game, only how often a position is evaluated again, and in
matches it answers only ~5% of the requests), which levels off at ~0.76 GB a match (20
threads; measured 2026-10-05); with MATCH_GB per match and RESERVE_GB besides available
they run at once, else one at a time, else it waits — the out-of-memory killer would
pick the largest process, the trainer (run G''s: ~9.9 GB).

It stops once the run's command.txt has a "finished" line and every due checkpoint is
evaluated (or at --until). Checkpoints already in watch.jsonl are not evaluated again,
so it can be restarted. Beside a run it evaluates only the newest due checkpoint (it
falls behind gracefully); --all evaluates every due one, oldest first — for a stopped or
finished run, after the fact (with --until for one that is not "finished").

  python3 thud/experiments/watch_run.py RUN_DIR [--every 4] [--pairs 20] [--nearest 2]
      [--start 0] [--until STEP] [--poll 60] [--guess RATING] [--nice] [--all]
      [--match_dir ~/thud-runs/stage1_matches]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder  # noqa: E402

RUNS = os.path.expanduser("~/thud-runs")
BIN = os.path.expanduser("~/thud-openspiel/build-shared/az_match")
CACHE, MATCH_GB, RESERVE_GB = 32768, 1.0, 1.5
# The ladder's anchors (thud/PLAN.md, How we evaluate networks), each as trained.
ANCHORS = {
    "G'44": f"{RUNS}/stage5b_Gaug_7x:44",
    "A14": f"{RUNS}/stage1_sims100:14",
    "E44": f"{RUNS}/stage3b_E_7x:44",
    "D29": f"{RUNS}/stage2_buffer4x:29",
    "C29": f"{RUNS}/stage3a_sims100:29",
    "C16": f"{RUNS}/stage3a_sims100:16",
}


def rule_of(spec):
  """The rule for untried moves the run of `spec` (RUN_DIR:STEP) trained with."""
  run_dir = spec.rsplit(":", 1)[0]
  try:
    config = json.load(open(os.path.join(run_dir, "config.json")))
  except OSError:
    return "upstream"
  return config.get("untried_move_value", "upstream")


def player_name(spec):
  return ladder.player(spec, rule_of(spec), 100)


def ratings(match_dir):
  matches = [m for m in ladder.load(match_dir)
             if m["a"] not in ladder.EXCLUDED.split(",")
             and m["b"] not in ladder.EXCLUDED.split(",")]
  return ladder.fit(matches, ladder.ANCHOR)


def log(match_dir, text):
  with open(os.path.join(match_dir, "progress.log"), "a") as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")


def checkpoints(run_dir):
  """The run's complete checkpoints: the trainer writes the network, then the optimizer."""
  steps = []
  for name in os.listdir(run_dir):
    if name.startswith("checkpoint-") and name.endswith("-optimizer.pt"):
      step = name[len("checkpoint-"):-len("-optimizer.pt")]
      if step.isdigit():
        steps.append(int(step))
  return sorted(steps)


def finished(run_dir):
  try:
    return any(l.startswith("finished") for l in open(os.path.join(run_dir, "command.txt")))
  except OSError:
    return False


def available_gb():
  for line in open("/proc/meminfo"):
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) / 2**20
  return 0.0


def run_matches(commands, paths, args):
  """Plays the matches, at once if memory allows (see the module's docstring)."""
  waited = False
  while available_gb() < MATCH_GB + RESERVE_GB:
    if not waited:
      log(args.match_dir, f"watch: waiting for memory ({available_gb():.1f} GB available)")
      waited = True
    time.sleep(args.poll)
  together = available_gb() >= len(commands) * MATCH_GB + RESERVE_GB
  priority = ["nice", "-n", "19"] if args.nice else ["chrt", "--idle", "0"]
  running = []
  for command, path in zip(commands, paths):
    out = open(path, "w")
    running.append((path, out, subprocess.Popen(
        priority + command, stdout=out, stderr=subprocess.DEVNULL,
        env=dict(os.environ, OMP_NUM_THREADS="1"))))
    if not together:
      running[-1][2].wait()
  for path, out, process in running:
    if process.wait() != 0:
      raise RuntimeError(f"az_match failed: {path}")
    out.close()


def evaluate(run_dir, step, args, last_rating):
  spec = f"{run_dir}:{step}"
  fitted = ratings(args.match_dir)
  anchors = sorted(ANCHORS, key=lambda a: abs(fitted.rating(a)[0] - last_rating)
                   if a in fitted.players else 1e9)[:args.nearest]
  run_name = os.path.basename(run_dir.rstrip("/"))
  # The matches at once, each with a thread per pair and OMP_NUM_THREADS=1: faster than
  # one after the other, whose last long pairs leave most threads idle (CLAUDE.md).
  paths = [os.path.join(args.match_dir,
                        f"watch_{run_name}_step{step}_vs_{a.replace(chr(39), 'aug')}.jsonl")
           for a in anchors]
  run_matches([[BIN, f"a={spec}", f"b={ANCHORS[a]}", f"untried_a={rule_of(spec)}",
                f"untried_b={rule_of(ANCHORS[a])}", f"pairs={args.pairs}",
                f"threads={args.pairs}", f"cache={CACHE}"] for a in anchors], paths, args)
  played = []
  for anchor, path in zip(anchors, paths):
    summary = [json.loads(l) for l in open(path) if '"summary"' in l][-1]
    played.append({"anchor": anchor, "margin": summary["a_mean_pair_margin"],
                   "ci95": summary["ci95"], "as_dwarfs": summary["a_mean_as_dwarfs"],
                   "as_trolls": summary["a_mean_as_trolls"]})
  fitted = ratings(args.match_dir)
  r, rs, d, ds, t, ts = fitted.rating(player_name(spec))
  entry = {"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "matches": played,
           "rating": round(r, 2), "rating_se": round(rs, 2), "as_dwarfs": round(d, 2),
           "as_trolls": round(t, 2), "against": ladder.ANCHOR}
  with open(os.path.join(run_dir, "watch.jsonl"), "a") as f:
    f.write(json.dumps(entry) + "\n")
  log(args.match_dir, f"watch {run_name} step {step}: rating {r:+.1f} ± {1.96 * rs:.1f} "
      f"against {ladder.ANCHOR} (as dwarfs {d:+.1f}, as trolls {t:+.1f}; "
      + ", ".join(f"vs {p['anchor']} {p['margin']:+.1f}" for p in played) + ")")
  return r


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("run_dir")
  parser.add_argument("--every", type=int, default=4)
  parser.add_argument("--pairs", type=int, default=20)
  parser.add_argument("--nearest", type=int, default=2)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--until", type=int, default=None)
  parser.add_argument("--poll", type=int, default=60)
  parser.add_argument("--all", action="store_true",
                      help="every due checkpoint, oldest first, not only the newest")
  parser.add_argument("--nice", action="store_true",
                      help="matches at nice 19, sharing the CPU, instead of idle priority")
  parser.add_argument("--guess", type=float, default=None,
                      help="the first checkpoint's expected rating (default: the weakest anchor's)")
  parser.add_argument("--match_dir", default=f"{RUNS}/stage1_matches")
  args = parser.parse_args()
  run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
  done, last_rating = set(), None
  watch = os.path.join(run_dir, "watch.jsonl")
  if os.path.exists(watch):
    for line in open(watch):
      if line.strip():
        entry = json.loads(line)
        done.add(entry["step"])
        last_rating = entry["rating"]
  if last_rating is None:  # Before any evaluation: a new run starts weak.
    fitted = ratings(args.match_dir)
    last_rating = (args.guess if args.guess is not None else
                   min(fitted.rating(a)[0] for a in ANCHORS if a in fitted.players))
  while True:
    due = [s for s in checkpoints(run_dir)
           if s >= args.start and s % args.every == 0 and s not in done
           and (args.until is None or s <= args.until)]
    if due:
      # The newest beside a run (the curve matters more than every point); with --all
      # each in turn.
      step = min(due) if args.all else max(due)
      last_rating = evaluate(run_dir, step, args, last_rating)
      done.update(s for s in due if s <= step)
      continue
    if finished(run_dir) or (args.until is not None and args.until in done):
      break
    time.sleep(args.poll)


if __name__ == "__main__":
  main()
