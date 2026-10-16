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

# Checks the watchdog (thud/az/watchdog.sh) and the trainer's crash-safe saving on a
# small run (make_test_network's network at 32 x 2, so that a checkpoint takes long enough
# to write to be caught at it; 8 simulations, a buffer of 2,048, 8 steps):
#
#   1. Killed at the worst moments, the run still finishes: the trainer is killed (-9)
#      while it writes the buffer over an older one (step 2 on), and later while it
#      writes its latest checkpoint, after which a truncated record is appended to
#      learner.jsonl (a run killed while logging); a kill that came just after a write
#      is repeated at a later step, up to 3 times. The watchdog restarts it each time; it
#      resumes — the old buffer and checkpoint whole, the truncated line skipped and
#      ended, so the next record starts a line of its own — and reaches step 8 with every
#      step recorded once, 1 to 8.
#   2. The control: a half-written buffer is refused (the trainer on a copy of the run
#      with its buffer cut in half stops with "Cannot load the buffer") — what writing
#      in place, as upstream does, would have left at the first kill, and what
#      upstream's LoadBuffer would load without an error, half the positions lost
#      (thud/az/replay_buffer_check.cc, part 5).
#   3. A stop asked of the watchdog (SIGTERM) stops the trainer and the watchdog without
#      a restart.
#   4. A trainer that always fails: the watchdog gives up after its restarts.
#
#   thud/az/watchdog_check.sh [BIN_DIR]    # default build-checks/bin; KEEP=1 keeps the files

set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
bin=${1:-$repo/build-checks/bin}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/watchdog_check.XXXXXX")
trap '[ -n "${KEEP:-}" ] || rm -rf "$tmp"' EXIT
failures=0
export OMP_NUM_THREADS=2
watchdog=$repo/thud/az/watchdog.sh

check() {
  local what=$1
  shift
  if "$@"; then echo "PASS $what"; else echo "FAIL $what"; failures=$((failures + 1)); fi
}

new_run() {  # new_run DIR MAX_STEPS: the network, and the first command in $first
  "$bin/make_test_network" dir="$1" nn_model=resnet_conv_policy input_planes=1 nn_width=32 \
    nn_depth=2 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network"; exit 1; }
  first=("$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$1"
    --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=32 --nn_depth=2
    --actors=3 --evaluators=0 --inference_batch_size=2 --inference_threads=1
    --max_simulations=8 --replay_buffer_size=2048 --replay_buffer_reuse=4
    --train_batch_size=64 --learner_batches=2 --checkpoint_freq=1 --max_steps="$2")
}

# 1. Killed at the worst moments.
run=$tmp/run
new_run "$run" 8
"$watchdog" --wait 2 --grace 60 "$run" -- "${first[@]}" &
wd=$!
python3 - "$run" "$wd" > "$tmp/killer.log" 2>&1 <<'EOF'
import json, os, signal, subprocess, sys, time
run, wd = sys.argv[1], int(sys.argv[2])

def trainer():
  out = subprocess.run(["pgrep", "-P", str(wd), "-x", "az_trainer"], capture_output=True,
                       text=True).stdout.split()
  return int(out[0]) if out else None

def alive(pid):
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False

def last_step():
  step = 0
  try:
    for line in open(os.path.join(run, "learner.jsonl")):
      try: step = json.loads(line)["step"]
      except (ValueError, KeyError): pass
  except FileNotFoundError: pass
  return step

def kill_when(files, after_step, what, deadline=300):
  """Kills the trainer once one of `files` exists (a write under way), after a step."""
  start = time.time()
  while time.time() - start < deadline:
    pid = trainer()  # Looked up once per run of the trainer: the poll below is tight.
    if pid is None or last_step() < after_step:
      time.sleep(0.05)
      continue
    while alive(pid) and time.time() - start < deadline:
      if any(os.path.exists(f) for f in files):
        os.kill(pid, signal.SIGKILL)
        mid = any(os.path.exists(f) for f in files)
        print(f"killed {what}: mid-write {mid}, after step {last_step()}", flush=True)
        return mid
  print(f"no kill {what}: timed out", flush=True)
  return False

buffer = os.path.join(run, "replay_buffer.data.tmp")
a = kill_when([buffer], 1, "buffer")
latest = [os.path.join(run, f) for f in ("checkpoint--1.pt.tmp", "checkpoint--1-optimizer.pt.tmp")]
b = False
for attempt in range(3):  # A kill just after the rename is repeated at a later step.
  while trainer() is None:
    time.sleep(0.05)
  b = kill_when(latest, last_step() + 1, "checkpoint")
  if b:
    break
with open(os.path.join(run, "learner.jsonl"), "a") as f:
  f.write('{"step": 99, "total_traj')  # A record cut off by the kill.
print(f"appended a truncated record; kills mid-write: buffer {a}, checkpoint {b}", flush=True)
EOF
wait "$wd"
status=$?
cat "$tmp/killer.log" | sed 's/^/  killer: /'
check "1. the watchdog finished the run (exit $status)" [ "$status" -eq 0 ]
check "1. both kills came while a file was being written" \
  grep -q "kills mid-write: buffer True, checkpoint True" "$tmp/killer.log"
check "1. learner.jsonl: every step recorded once, 1 to 8, the fragment apart" \
  python3 - "$run/learner.jsonl" <<'EOF'
import json, sys
steps, bad = [], 0
for line in open(sys.argv[1]):
  try: steps.append(json.loads(line)["step"])
  except ValueError: bad += 1
ok = bad == 1 and steps == list(range(1, 9))
print("  steps", steps, "unparsed lines", bad)
sys.exit(0 if ok else 1)
EOF
check "1. the resume said it skipped the truncated record" \
  grep -q "skipping a line that does not parse" "$run/trainer.log"
starts=$(grep -c " start " "$run/watchdog.log")
kills=$(grep -c "^killed" "$tmp/killer.log")
check "1. the watchdog started the trainer once more for each of the $kills kills ($starts)" \
  [ "$starts" -eq $((kills + 1)) ]

# 2. The control: a half-written buffer.
cp -r "$run" "$tmp/control"
python3 - "$tmp/control" <<'EOF'
import json, os, sys
d = sys.argv[1]
c = json.load(open(d + "/config.json")); c["path"] = d; c["max_steps"] = 9
json.dump(c, open(d + "/config.json", "w"))
size = os.path.getsize(d + "/replay_buffer.data")
os.truncate(d + "/replay_buffer.data", size // 2)
EOF
timeout 300 "$bin/az_trainer" "$tmp/control/config.json" > "$tmp/control.log" 2>&1
control=$?
check "2. control: a half-written buffer is refused (exit $control)" \
  bash -c "[ $control -ne 0 ] && [ $control -ne 124 ] &&
           grep -q 'Cannot load the buffer' '$tmp/control.log'"

# 3. A stop asked of the watchdog.
run3=$tmp/stop
new_run "$run3" 50
"$watchdog" --wait 2 --grace 60 "$run3" -- "${first[@]}" &
wd=$!
timeout 300 bash -c "until [ -s '$run3/learner.jsonl' ]; do sleep 0.5; done"
kill -TERM "$wd"
timeout 120 bash -c "while kill -0 $wd 2>/dev/null; do sleep 0.5; done"
wait "$wd"
status=$?
check "3. SIGTERM: the watchdog ends (exit $status) without a restart, the trainer gone" \
  bash -c "[ $status -eq 0 ] && grep -q 'stopped as asked' '$run3/watchdog.log' &&
           [ \$(grep -c ' start ' '$run3/watchdog.log') -eq 1 ] &&
           ! pgrep -f '^$bin/az_trainer .*$run3' > /dev/null"

# 4. A trainer that always fails.
"$watchdog" --wait 0 --max-restarts 2 "$tmp/fails" -- false
status=$?
check "4. a trainer that always fails: the watchdog gives up (exit $status)" \
  bash -c "[ $status -eq 1 ] && grep -q 'gave up' '$tmp/fails/watchdog.log' &&
           [ \$(grep -c ' start ' '$tmp/fails/watchdog.log') -eq 3 ]"

if [ "$failures" -eq 0 ]; then echo "ALL PASSED"; else echo "$failures FAILED"; fi
[ "$failures" -eq 0 ]
