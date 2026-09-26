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

# The throughput measurements of thud/PLAN.md Phase 6, on Thud and, for comparison,
# connect_four and chess, with az_throughput (build it first:
# thud/experiments/build_az_program.sh az_throughput). About 55 minutes, most of it with
# all cores busy. Appends one JSON line per run to OUT:
#
#   thud/experiments/az_throughput.sh OUT.jsonl [SECTION ...]
#
#   1. The network alone, one position at a time and in batches of 64, one thread.
#   2. One search, one thread: the unit cost.
#   3. Scaling: 1 to 10 searches at once, one thread each, no batching.
#   4. Batched inference: searches hand positions to inference threads, which evaluate
#      them together; the searches themselves hardly use LibTorch.
#   5. The learner: one training step, on 1 and 10 threads.
#   6. Sustained load: 10 searches for 5 minutes, halves compared for throttling.
#   7. Section 3's Thud 64 x 4 points again, with longer windows: how noisy is it?

set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 OUT.jsonl [SECTION ...]   (default: all)" >&2
  exit 1
fi
out=$1
shift
sections=" ${*:-1 2 3 4 5 6 7} "
has() { [[ "$sections" == *" $1 "* ]]; }
bin=$(cd "$(dirname "$0")/../.." && pwd)/build-shared/az_throughput
sizes="32:2 64:4 128:6"
run() {  # OMP_NUM_THREADS, then az_throughput's arguments.
  local omp=$1
  shift
  echo "OMP_NUM_THREADS=$omp $*" >&2
  local line
  line=$(OMP_NUM_THREADS=$omp "$bin" "$@" 2>/dev/null | grep '^{' || true)
  if [ -z "$line" ]; then  # Record the failure and go on.
    line="{\"failed\": \"$*\"}"
  fi
  echo "${line/#\{/\{\"omp_num_threads\": $omp, \"section\": $section, }" >> "$out"
}

section=1  # The network alone.
if has 1; then
  for game in thud chess connect_four; do
    for size in $sizes; do
      run 1 infer game=$game width=${size%:*} depth=${size#*:}
    done
  done
fi
section=2  # One search, one thread.
if has 2; then
  for game in thud chess connect_four; do
    for size in $sizes; do
      run 1 search game=$game width=${size%:*} depth=${size#*:} sims=100 searchers=1 \
        seconds=20
    done
  done
fi
section=3  # Scaling without batching.
if has 3; then
  for size in $sizes; do
    for n in 2 4 6 8 10; do
      run 1 search game=thud width=${size%:*} depth=${size#*:} sims=100 searchers=$n \
        seconds=20
    done
  done
  for n in 2 4 6 8 10; do
    run 1 search game=chess width=64 depth=4 sims=100 searchers=$n seconds=20
  done
fi
section=4  # Batched inference: OMP_NUM_THREADS, searchers, batch, inference threads.
if has 4; then
  for size in $sizes; do
    for config in "2 16 16 2" "4 32 32 2" "2 32 32 4" "3 32 32 3" "5 32 32 2" \
                  "4 48 48 2"; do
      read -r omp searchers batch threads <<< "$config"
      run "$omp" search game=thud width=${size%:*} depth=${size#*:} sims=100 \
        searchers=$searchers batch=$batch inference_threads=$threads seconds=40 warmup=15
    done
  done
fi
section=5  # The learner.
if has 5; then
  for size in $sizes; do
    w=${size%:*} d=${size#*:}
    run 1 learn game=thud width=$w depth=$d batch=128 steps=3
    run 10 learn game=thud width=$w depth=$d batch=128 steps=3
    run 10 learn game=thud width=$w depth=$d batch=1024 steps=3
  done
fi
section=6  # Sustained load.
if has 6; then
  run 1 search game=thud width=64 depth=4 sims=100 searchers=10 seconds=300
fi
section=7  # Scaling without batching again, longer: how noisy is section 3?
if has 7; then
  for n in 1 2 4 10; do
    run 1 search game=thud width=64 depth=4 sims=100 searchers=$n seconds=30
  done
fi
