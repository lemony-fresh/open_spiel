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

# Moves a training run between machines (thud/PLAN.md, the GPU readiness checklist), with
# rsync: resumable (an interrupted copy goes on where it stopped), and checked — every
# file's SHA-256 compared at both ends. The run's config.json holds the run's own folder
# (`path`, where the resumed trainer writes), so the copy's is set to its new folder: a
# copy left pointing at the old path would write there, or fail. Refuses to copy a run
# whose trainer is running on this machine (its files would be from different steps).
#
#   thud/transfer_run.sh FROM TO [--state-only] [--force]
#
# FROM and TO are run folders, local (a path) or remote (HOST:PATH, through ssh; HOST as in
# ~/.ssh/config); one of them local. --state-only copies only what the trainer resumes
# from — config.json, learner.jsonl, vpnet.pb, replay_buffer.data and the latest checkpoint
# with its optimizer (checkpoint--1) — not the numbered checkpoints and the logs. Files
# being written (*.tmp) are never copied. --force copies a running run anyway.
#
#   thud/transfer_run.sh ~/thud-runs/runN_d1_cont gpu:/data/runs/runN_d1_cont --state-only
#   thud/transfer_run.sh gpu:/data/runs/runN_d1_cont ~/thud-runs/runN_d1_cont

set -uo pipefail
state_only=0 force=0
args=()
for a in "$@"; do
  case $a in
    --state-only) state_only=1 ;;
    --force) force=1 ;;
    *) args+=("$a") ;;
  esac
done
[ ${#args[@]} -eq 2 ] || { sed -n '17,32p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
from=${args[0]%/} to=${args[1]%/}

remote() { [[ $1 == *:* ]]; }
host() { echo "${1%%:*}"; }
dir() { if remote "$1"; then echo "${1#*:}"; else echo "$1"; fi; }
# Runs a shell command where the folder $1 is (on its host, or here).
at() {
  local where=$1
  shift
  if remote "$where"; then ssh "$(host "$where")" "$@"; else bash -c "$*"; fi
}

if remote "$from" && remote "$to"; then echo "One of FROM and TO must be local." >&2; exit 2; fi
at "$from" "test -f '$(dir "$from")/config.json'" ||
  { echo "$from has no config.json: not a run folder." >&2; exit 1; }
if ! remote "$from" && [ "$force" -eq 0 ] &&
    pgrep -f "az_trainer[^ ]* .*$(dir "$from")" > /dev/null; then
  echo "A trainer is running on $from: stop it first (or --force)." >&2
  exit 1
fi

filter=(--exclude='*.tmp')
if [ "$state_only" -eq 1 ]; then
  filter=(--include=config.json --include=learner.jsonl --include=vpnet.pb
          --include=replay_buffer.data --include='checkpoint--1.pt'
          --include='checkpoint--1-optimizer.pt' --exclude='*')
fi
at "$to" "mkdir -p '$(dir "$to")'" || exit 1
echo "Copying $from -> $to ..."
rsync -a --partial --info=progress2 "${filter[@]}" "$from/" "$to/" || {
  echo "rsync failed: run the same command again to go on where it stopped." >&2
  exit 1
}

# The copy's config.json points at its own folder.
at "$to" "python3 -c \"
import json, sys
p = '$(dir "$to")/config.json'
c = json.load(open(p))
old = c['path']; c['path'] = '$(dir "$to")'
json.dump(c, open(p, 'w'), indent=2)
print('config.json: path', old, '->', c['path'])
\"" || exit 1

# Every copied file's SHA-256, compared (config.json apart: its path differs by design).
list=$(cd /dev/null; at "$to" "cd '$(dir "$to")' && find . -type f ! -name config.json ! -name '*.tmp' | sort")
sums_from=$(at "$from" "cd '$(dir "$from")' && sha256sum $(echo $list)")
sums_to=$(at "$to" "cd '$(dir "$to")' && sha256sum $(echo $list)")
if [ "$sums_from" != "$sums_to" ]; then
  echo "CHECKSUMS DIFFER between $from and $to:" >&2
  diff <(echo "$sums_from") <(echo "$sums_to") | head >&2
  exit 1
fi
echo "Checked: $(echo "$list" | wc -w) files equal at both ends (SHA-256); config.json's path set."
