#!/bin/bash
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

# The trainer benchmark (thud/PLAN.md, the GPU readiness checklist: settings compared by the
# trainer's own step time, with the learner sharing the machine): for each setting, a copy
# of a run's state trained 2 steps on, the second timed.
#
#   thud/experiments/bench_trainer.sh RUN OUT SETTING...
#
# RUN: a run's folder or a snapshot of one, not running: its latest step S (learner.jsonl),
# the latest checkpoint (checkpoint--1.pt, which the trainer resumes from: S's), its
# optimizer, the buffer, vpnet.pb and config.json; checkpoint-S.pt too if there. OUT: a folder of
# its own, a subfolder per setting (s0, s1, ...). SETTING:
# ACTORS,BATCH,INFERENCE_THREADS,OMP_NUM_THREADS[,LEARNER_THREADS[,PRECISION]] — the
# learner's threads 0 (the default) share OMP_NUM_THREADS; the precision defaults to the
# run's. Everything else is the run's config.
#
# Prints a JSON line per setting, then a summary naming the fastest:
#   period          step S+2's time (learner.jsonl's time_rel, S+1 to S+2): the setting's speed
#   learning        S+2's learning, from "Learning on" to "Checkpoint saved" (log-learner.txt)
#   waited          how long the learner waited for S+2's positions after S+1's learning
#   games_waiting   finished games in the queue when S+2's collection began, against
#   games_taken     those S+2 took: more waiting than taken, the learner sets the pace;
#   bound           "learner" or "self-play", from those two
#   states_per_s    S+2's positions over the period; threads: learner.jsonl's counts
#   paused_s        the wall clock ahead of /proc/uptime over the run (a sleep or freeze)
#   ready_failures  samples, once a minute, at which BENCH_READY failed (below)
#   probe           BENCH_PROBE's samples, once a minute: their mean, least and count
#
# BENCH_READY: a command that must succeed before each setting starts (retried every 2
# minutes) and is sampled during it — e.g. a mains-power test on a laptop. BENCH_PROBE: a
# command printing one number, sampled once a minute — e.g. the CPU's performance limit
# (on this laptop Windows' "% Performance Limit": it throttles under sustained load, found
# 2026-10-19 at 40%), so that a setting measured on a slowed machine shows. A setting with
# its result (OUT/sN/result.json) is skipped, so a stopped benchmark resumes. Each setting's
# buffer and checkpoints are deleted after it (KEEP=1 keeps them); its logs stay. Runs the
# trainer at nice 19. BIN: the trainer's folder (default build-shared).

set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
bin=${BIN:-$repo/build-shared}
[ $# -ge 3 ] || { sed -n 16,50p "$0"; exit 2; }
run=$1 out=$2
shift 2
step=$(tail -1 "$run/learner.jsonl" | python3 -c 'import json,sys; print(json.load(sys.stdin)["step"])') ||
  { echo "no learner.jsonl in $run" >&2; exit 1; }
for f in config.json vpnet.pb replay_buffer.data checkpoint--1.pt checkpoint--1-optimizer.pt; do
  [ -f "$run/$f" ] || { echo "$run: no $f" >&2; exit 1; }
done
mkdir -p "$out"

ready() { [ -z "${BENCH_READY:-}" ] || bash -c "$BENCH_READY" > /dev/null 2>&1; }

k=0
for setting in "$@"; do
  folder=$out/s$k
  k=$((k + 1))
  if [ -f "$folder/result.json" ]; then cat "$folder/result.json"; continue; fi
  IFS=, read -r actors batch threads omp learner precision <<< "$setting"
  [ -n "$omp" ] || { echo "setting $setting: ACTORS,BATCH,INFERENCE_THREADS,OMP needed" >&2; exit 2; }
  rm -rf "${folder:?}"
  mkdir -p "$folder"
  for f in learner.jsonl replay_buffer.data vpnet.pb checkpoint--1.pt checkpoint--1-optimizer.pt \
           "checkpoint-$step.pt" "checkpoint-$step-optimizer.pt"; do
    [ -f "$run/$f" ] || continue  # Only checkpoint-S's may be missing (tested above).
    cp "$run/$f" "$folder/$f" && cmp -s "$run/$f" "$folder/$f" ||
      { echo "the copy of $f into $folder failed" >&2; exit 1; }
  done
  python3 - "$run/config.json" "$folder" "$((step + 2))" "$actors" "$batch" "$threads" \
    "${learner:-0}" "${precision:-}" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(path=sys.argv[2], max_steps=int(sys.argv[3]), actors=int(sys.argv[4]),
         inference_batch_size=int(sys.argv[5]), inference_threads=int(sys.argv[6]),
         learner_threads=int(sys.argv[7]))
if sys.argv[8]: c["inference_precision"] = sys.argv[8]
json.dump(c, open(sys.argv[2] + "/config.json", "w"), indent=2)
EOF
  until ready; do echo "$(date '+%F %T') waiting for BENCH_READY ($setting)" >&2; sleep 120; done
  (tick=0
   while true; do  # The pause detector; the readiness and probe samples, from the start.
     echo "$(date +%s) $(cut -d' ' -f1 /proc/uptime)" >> "$folder/clock.log"
     if [ $((tick++ % 6)) -eq 0 ]; then
       ready || echo x >> "$folder/not_ready.log"
       [ -z "${BENCH_PROBE:-}" ] || bash -c "$BENCH_PROBE" 2> /dev/null | tr -d '\r' |
         head -1 >> "$folder/probe.log"
     fi
     sleep 10
   done) &
  watcher=$!
  echo "OMP_NUM_THREADS=$omp nice -n 19 $bin/az_trainer $folder/config.json" > "$folder/command.txt"
  OMP_NUM_THREADS=$omp nice -n 19 "$bin/az_trainer" "$folder/config.json" > "$folder/trainer.log" 2>&1
  status=$?
  kill "$watcher" 2> /dev/null
  wait "$watcher" 2> /dev/null
  python3 - "$folder" "$step" "$setting" "$status" > "$folder/result.json.part" <<'EOF'
import json, os, re, sys
from datetime import datetime
folder, s, setting, status = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
rows = {d["step"]: d for d in map(json.loads, open(os.path.join(folder, "learner.jsonl")))}
r = {"setting": setting, "exit": status}
try:
  a, b = rows[s + 1], rows[s + 2]
  period = b["time_rel"] - a["time_rel"]
  # The learner's log lines of this segment, by time: the last step's collection,
  # learning and checkpoint, and the checkpoint before them.
  times = []
  for line in open(os.path.join(folder, "log-learner.txt")):
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] (.*)", line)
    if m: times.append((datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f"), m[2]))
  last = lambda prefix, before=None: max(
      i for i, (t, text) in enumerate(times) if text.startswith(prefix) and
      (before is None or i < before))
  saved = last("Checkpoint saved")
  learning = last("Learning on", saved)
  collected = last("Collected", learning)
  previous = last("Checkpoint saved", collected)
  seconds = lambda i, j: round((times[j][0] - times[i][0]).total_seconds(), 1)
  taken = b["total_trajectories"] - a["total_trajectories"]
  r.update(period=round(period, 1), learning=seconds(learning, saved),
           waited=seconds(previous, collected), games_waiting=b["queue_size"],
           games_taken=taken, bound="learner" if b["queue_size"] >= taken else "self-play",
           states_per_s=round((b["total_states"] - a["total_states"]) / period, 2),
           threads=b.get("threads"))
except (KeyError, ValueError) as e:  # A step or a log line missing: no timing.
  r["error"] = f"{type(e).__name__}: {e}"
clock = [tuple(map(float, l.split())) for l in open(os.path.join(folder, "clock.log"))]
r["paused_s"] = round((clock[-1][0] - clock[0][0]) - (clock[-1][1] - clock[0][1]), 1) if clock else None
try:
  r["ready_failures"] = sum(1 for _ in open(os.path.join(folder, "not_ready.log")))
except FileNotFoundError:
  r["ready_failures"] = 0
try:
  samples = [float(l) for l in open(os.path.join(folder, "probe.log")) if l.strip()]
  if samples:
    r["probe"] = {"mean": round(sum(samples) / len(samples), 2), "least": min(samples),
                  "samples": len(samples)}
except (FileNotFoundError, ValueError):
  pass
print(json.dumps(r))
EOF
  mv "$folder/result.json.part" "$folder/result.json"
  cat "$folder/result.json"
  [ -n "${KEEP:-}" ] || rm -f "$folder/replay_buffer.data" "$folder"/checkpoint-*.pt
done

python3 - "$out" "$k" <<'EOF'
import json, os, sys
results = [json.load(open(os.path.join(sys.argv[1], f"s{i}", "result.json")))
           for i in range(int(sys.argv[2]))]
timed = [(r["period"], i, r) for i, r in enumerate(results) if "period" in r]
summary = {"summary": True, "settings": len(results), "timed": len(timed)}
if timed:
  period, i, r = min(timed)
  summary.update(fastest=f"s{i}", setting=r["setting"], period=period,
                 clean=all(x["paused_s"] is not None and abs(x["paused_s"]) < 5 and
                           x["ready_failures"] == 0 for x in results))
print(json.dumps(summary))
EOF
