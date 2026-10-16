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

# Checks thud/transfer_run.sh between two local folders (the same commands as between
# machines, without ssh), on a small run of 2 steps (make_test_network's network):
#
#   1. A copy of the whole run: every file equal (the script's own SHA-256 check passes),
#      config.json's path set to the copy's folder.
#   2. The copy resumes where it is: the trainer on the copy's config.json trains step 3
#      into the copy; the original is untouched (its learner.jsonl, its buffer). The
#      control: the original's config.json still points at the original.
#   3. --state-only copies only the files to resume from (config.json, learner.jsonl,
#      vpnet.pb, the buffer, the latest checkpoint) — and the copy resumes.
#   4. A run whose trainer is running is refused (a dummy process named like one).
#   5. An interrupted copy goes on: a copy with a file cut short is completed by
#      running the same command again (rsync --partial), the checksums then equal.
#
#   thud/transfer_check.sh [BIN_DIR]    # default build-checks/bin

set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
bin=${1:-$repo/build-checks/bin}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/transfer_check.XXXXXX")
trap '[ -n "${KEEP:-}" ] || rm -rf "$tmp"' EXIT
failures=0
export OMP_NUM_THREADS=2
transfer=$repo/thud/transfer_run.sh

check() {
  local what=$1
  shift
  if "$@"; then echo "PASS $what"; else echo "FAIL $what"; failures=$((failures + 1)); fi
}
steps() { python3 -c "import json,sys; print(' '.join(str(json.loads(l)['step']) for l in open(sys.argv[1]) if l.strip()))" "$1"; }
config_path() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$1"; }
set_max_steps() {
  python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['max_steps']=int(sys.argv[2]); json.dump(c, open(sys.argv[1],'w'))" "$1" "$2"
}

run=$tmp/source
"$bin/make_test_network" dir="$run" nn_model=resnet_conv_policy input_planes=1 nn_width=8 \
  nn_depth=1 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network"; exit 1; }
"$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$run" \
  --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=8 --nn_depth=1 \
  --actors=3 --evaluators=0 --inference_batch_size=2 --inference_threads=1 \
  --max_simulations=8 --replay_buffer_size=2048 --replay_buffer_reuse=4 \
  --train_batch_size=64 --learner_batches=2 --checkpoint_freq=1 --max_steps=2 \
  > "$tmp/source.log" 2>&1 || { echo "FAIL the source run"; tail -3 "$tmp/source.log"; exit 1; }

# 1. A whole copy.
"$transfer" "$run" "$tmp/copy" > "$tmp/copy.log" 2>&1
status=$?
check "1. the copy (exit $status) passes its own checksum check" \
  bash -c "[ $status -eq 0 ] && grep -q 'files equal at both ends' '$tmp/copy.log'"
check "1. the copy's config.json points at the copy" [ "$(config_path "$tmp/copy/config.json")" = "$tmp/copy" ]

# 2. The copy resumes where it is.
before_steps=$(steps "$run/learner.jsonl")
before_buffer=$(sha256sum < "$run/replay_buffer.data")
set_max_steps "$tmp/copy/config.json" 3
"$bin/az_trainer" "$tmp/copy/config.json" > "$tmp/resume.log" 2>&1
check "2. the copy resumes and trains step 3 into itself" [ "$(steps "$tmp/copy/learner.jsonl")" = "1 2 3" ]
check "2. the original is untouched (its steps, its buffer)" \
  bash -c "[ '$(steps "$run/learner.jsonl")' = '$before_steps' ] &&
           [ \"\$(sha256sum < '$run/replay_buffer.data')\" = '$before_buffer' ]"
check "2. control: the original's config.json still points at the original" \
  [ "$(config_path "$run/config.json")" = "$run" ]

# 3. --state-only.
"$transfer" "$run" "$tmp/state" --state-only > "$tmp/state.log" 2>&1
files=$(cd "$tmp/state" && ls | sort | tr '\n' ' ')
check "3. --state-only copies the files to resume from: $files" \
  [ "$files" = "checkpoint--1-optimizer.pt checkpoint--1.pt config.json learner.jsonl replay_buffer.data vpnet.pb " ]
set_max_steps "$tmp/state/config.json" 3
"$bin/az_trainer" "$tmp/state/config.json" > "$tmp/state_resume.log" 2>&1
check "3. ... and the copy resumes (step 3)" [ "$(steps "$tmp/state/learner.jsonl")" = "1 2 3" ]

# 4. A running run is refused.
printf '#!/bin/bash\nsleep 30\n' > "$tmp/az_trainer"  # Ignores its arguments, as sleep would not.
chmod +x "$tmp/az_trainer"
"$tmp/az_trainer" "$run/config.json" &
dummy=$!
sleep 0.5
"$transfer" "$run" "$tmp/refused" > "$tmp/refused.log" 2>&1
status=$?
kill "$dummy" 2> /dev/null
check "4. a run whose trainer is running is refused (exit $status, nothing copied)" \
  bash -c "[ $status -ne 0 ] && grep -q 'A trainer is running' '$tmp/refused.log' &&
           [ ! -e '$tmp/refused/config.json' ]"

# 5. An interrupted copy goes on.
"$transfer" "$run" "$tmp/partial" > /dev/null 2>&1
size=$(stat -c %s "$tmp/partial/replay_buffer.data")
truncate -s $((size / 3)) "$tmp/partial/replay_buffer.data"
"$transfer" "$run" "$tmp/partial" > "$tmp/partial.log" 2>&1
status=$?
check "5. a copy with a file cut short is completed by the same command (exit $status)" \
  bash -c "[ $status -eq 0 ] && grep -q 'files equal at both ends' '$tmp/partial.log' &&
           cmp -s '$run/replay_buffer.data' '$tmp/partial/replay_buffer.data'"

if [ "$failures" -eq 0 ]; then echo "ALL PASSED"; else echo "$failures FAILED"; fi
[ "$failures" -eq 0 ]
