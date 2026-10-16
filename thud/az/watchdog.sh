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

# Keeps a training run going (thud/PLAN.md, the GPU readiness checklist: restarting a run
# by itself after a crash or a preemption). Runs the trainer; whenever it ends other than
# by finishing — a crash, a kill, the machine's out-of-memory killer — waits WAIT
# seconds and starts it again on the run's config.json, which resumes from the latest
# checkpoint and buffer (both saved whole: thud/az/vpnet.cc, thud/az/replay_buffer.h).
# Before the run's first step there is nothing to resume, and the first command runs
# again. Gives up after MAX restarts in a row that brought no new step.
#
# The trainer finishing (exit 0: max_steps reached, or stopped by SIGINT) ends the
# watchdog. A stop asked of the watchdog (SIGTERM or SIGINT) is passed to the trainer as
# SIGINT — it stops after its current step — with a hard kill after GRACE seconds, and the
# watchdog then ends without a restart. A machine that shuts down (a preempted spot
# instance) kills both; start the watchdog again when it is back (cron: @reboot), and
# the run resumes.
#
#   thud/az/watchdog.sh [--max-restarts MAX=5] [--wait WAIT=30] [--grace GRACE=120] \
#     RUN_DIR -- TRAINER [ARGS ...]
#
# TRAINER ARGS is the run's first command (flags for a new run, or RUN_DIR/config.json);
# restarts run TRAINER RUN_DIR/config.json. Each start and end is logged in
# RUN_DIR/watchdog.log; the trainer's output goes to RUN_DIR/trainer.log (appended).

set -uo pipefail
max_restarts=5 wait_s=30 grace=120
while [ $# -gt 0 ]; do
  case $1 in
    --max-restarts) max_restarts=$2; shift 2 ;;
    --wait) wait_s=$2; shift 2 ;;
    --grace) grace=$2; shift 2 ;;
    *) break ;;
  esac
done
[ $# -ge 3 ] && [ "$2" = "--" ] || {
  echo "Usage: $0 [--max-restarts N] [--wait S] [--grace S] RUN_DIR -- TRAINER [ARGS ...]" >&2
  exit 2
}
run=$1
shift 2
trainer=$1
first=("$@")
mkdir -p "$run"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$run/watchdog.log"; }

# The last step learner.jsonl records (the last line that parses), or 0.
last_step() {
  python3 - "$run/learner.jsonl" <<'EOF' 2>/dev/null || echo 0
import json, sys
step = 0
try:
  for line in open(sys.argv[1]):
    try:
      step = json.loads(line)["step"]
    except (ValueError, KeyError):
      pass
except FileNotFoundError:
  pass
print(step)
EOF
}

child=0 stopping=0
stop() {
  stopping=1
  log "stop asked: SIGINT to the trainer (pid $child), a hard kill after ${grace} s"
  if [ "$child" -gt 0 ]; then
    kill -INT "$child" 2>/dev/null
    for _ in $(seq "$grace"); do kill -0 "$child" 2>/dev/null || break; sleep 1; done
    kill -KILL "$child" 2>/dev/null
  fi
}
trap stop TERM INT

restarts=0
progress_at=$(last_step)
attempt=0
while :; do
  attempt=$((attempt + 1))
  if [ "$attempt" -eq 1 ] || [ "$(last_step)" -eq 0 ]; then
    command=("${first[@]}")
  else
    command=("$trainer" "$run/config.json")
  fi
  log "start $attempt (after step $(last_step)): ${command[*]}"
  "${command[@]}" >> "$run/trainer.log" 2>&1 &
  child=$!
  wait "$child"
  status=$?
  # A signal to the watchdog interrupts `wait`; wait for the trainer itself.
  while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
  now=$(last_step)
  log "end $attempt: exit $status, after step $now"
  if [ "$stopping" -eq 1 ]; then
    log "stopped as asked"
    exit 0
  fi
  if [ "$status" -eq 0 ]; then
    log "the trainer finished"
    exit 0
  fi
  if [ "$now" -gt "$progress_at" ]; then
    restarts=0
    progress_at=$now
  fi
  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$max_restarts" ]; then
    log "gave up: $max_restarts restarts in a row without a new step"
    exit 1
  fi
  log "restart $restarts of at most $max_restarts without progress, in ${wait_s} s"
  sleep "$wait_s" &
  wait $!
  [ "$stopping" -eq 1 ] && { log "stopped as asked"; exit 0; }
done
