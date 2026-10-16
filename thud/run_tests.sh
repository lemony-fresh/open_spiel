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

# Runs our tests (thud/PLAN.md, *Tests*: what each covers, and the gaps): builds what is
# out of date — the game into build/ (thud_test, pyspiel) and build-shared/
# (libopen_spiel.so), the tests into build-tests/ (thud/tests.mk: without -DNDEBUG) —
# then runs the tests, a few at once, at the lowest CPU priority, and prints one line
# each with its time. Each test's output is kept in build-tests/last/NAME.log.
#
#   thud/run_tests.sh            # the quick tests (~20 s): what the pre-commit hook runs
#   thud/run_tests.sh --full     # every test (~4 min): what the pre-push hook runs
#   thud/run_tests.sh NAME ...   # only these (--list shows the names)
#   [--jobs N]                    # tests at once (default 4; 1 beside an experiment)
#
# Exits non-zero if a build or any test fails. The tests need no run folder: the ones
# that read a network get small untrained ones from make_test_network.

set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
# Hooks may run without the venv (CLAUDE.md, Shell environment): set it here.
export PATH="$repo/venv/bin:$PATH"
export PYTHONPATH="$repo:$repo/build/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2

mode=quick jobs=4 list=0
names=()
while [ $# -gt 0 ]; do
  case $1 in
    --quick) mode=quick ;;
    --full) mode=full ;;
    --jobs) jobs=$2; shift ;;
    --list) list=1 ;;
    --playthrough | --sanitized) names+=("$1") ;;  # Run as tests, below.
    -*) echo "Unknown option $1" >&2; exit 2 ;;
    *) names+=("$1") ;;
  esac
  shift
done

bin=$repo/build-tests/bin
logs=$repo/build-tests/last
nets=$repo/build-tests/nets

# NAME, quick or full, its time alone in seconds (measured 2026-10-15: the longest start
# first), and the command (run from the repo root). Quick: seconds each.
tests=(
  "thud_test|quick|2|build/games/thud_test"
  "crosscheck_tests|quick|0.2|python3 thud/experiments/crosscheck_tests.py"
  "alphabeta_training_test|quick|0.2|$bin/alphabeta_training_test"
  "augmentation_test|quick|0.2|$bin/augmentation_test"
  "grow_test|quick|2.4|$bin/grow_test"
  "growing_buffer_test|quick|0.2|$bin/growing_buffer_test"
  "instrumentation_test|quick|0.2|$bin/instrumentation_test"
  "search_settings_test|quick|0.3|$bin/search_settings_test"
  "teacher_mix_test|quick|0.3|$bin/teacher_mix_test"
  "learning_settings_test|quick|0.8|$bin/learning_settings_test"
  "playout_cap_test|quick|1.3|$bin/playout_cap_test"
  "regression_gates_test|quick|0.9|python3 thud/experiments/regression_gates_test.py"
  "ladder_test|quick|0.7|python3 thud/experiments/ladder_test.py"
  "position_edits_test|quick|1.5|$bin/position_edits_test"
  "features_test|quick|3|$bin/features_test"
  "replay_buffer_test|quick|0.5|$bin/replay_buffer_test $logs"
  "az_buffer_truncate|quick|0|$bin/az_buffer_truncate self_test"
  "pruning_test|quick|13|$bin/pruning_test $nets/peaked 0"
  "trainer_test|quick|26|thud/az/trainer_test.sh $bin"
  "playthrough|full|3.5|thud/run_tests.sh --playthrough"
  "sanitized|full|10|thud/run_tests.sh --sanitized"
  "identity_test|full|88|$bin/identity_test $nets/peaked 0 1"
  "untried_move_test|full|107|$bin/untried_move_test $nets/peaked 0 games=1"
  "input_planes_test|full|65|$bin/input_planes_test $logs/input_planes_scratch"
  "kickstart_test|full|25|$bin/kickstart_test"
  "conv_policy_test|full|6|$bin/conv_policy_test"
  "az_match_test|full|97|thud/experiments/az_match_test.sh $bin"
  "watchdog_test|full|90|thud/az/watchdog_test.sh $bin"
  "transfer_test|full|24|thud/transfer_test.sh $bin"
  "legal_policy_test|full|44|$bin/legal_policy_test"
)

# Two tests that are commands of their own, run as tests above.
case "${names[0]:-}" in
  --playthrough)
    # The playthrough regression baseline (CLAUDE.md), as upstream's playthrough_test.py
    # tests it: the moves recorded in the baseline replayed, every line compared (a new
    # playthrough would play other random moves). The control: a copy with one board line
    # changed must not replay equal.
    python3 - <<'EOF'
import sys, tempfile
from absl import flags
from open_spiel.python.algorithms import generate_playthrough as g
flags.FLAGS(["playthrough"])  # Its defaults, as upstream's test (absltest.main) parses them.
baseline = "open_spiel/integration_tests/playthroughs/thud.txt"
expected, actual = g.replay(baseline)
if expected != actual:
  for n, (e, a) in enumerate(zip(expected.split("\n"), actual.split("\n"))):
    if e != a:
      print(f"FAIL line {n + 1} differs:\n  baseline: {e}\n  now:      {a}")
      break
  sys.exit(1)
print("PASS the baseline's moves replay to the baseline, line for line")
text = open(baseline).read()
line = next(l for l in text.split("\n") if l.startswith("# ") and "d" in l and "." in l)
with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
  f.write(text.replace(line, line.replace("d", ".", 1), 1))
  f.flush()
  tampered, replayed = g.replay(f.name)
if tampered == replayed:
  sys.exit("FAIL control: a changed board line still replays equal")
print("PASS control: a copy with one board line changed does not")
EOF
    exit ;;
  --sanitized)
    # The tests without LibTorch under AddressSanitizer and UBSan; the control: the
    # binaries are instrumented (a sanitizer that is not there reports nothing).
    status=0
    for b in replay_buffer_test features_test az_buffer_truncate; do
      # grep -c reads all of nm's output (grep -q would stop early: nm's SIGPIPE fails
      # the pipe under pipefail).
      nm "$bin/sanitized/$b" | grep -c __asan_report_load > /dev/null ||
        { echo "FAIL control: $b is not instrumented"; status=1; }
    done
    mkdir -p "$logs/sanitized_scratch"
    "$bin/sanitized/replay_buffer_test" "$logs/sanitized_scratch" &&
      "$bin/sanitized/features_test" && "$bin/sanitized/az_buffer_truncate" self_test || status=1
    exit $status ;;
esac
if [ "$list" -eq 1 ]; then
  for c in "${tests[@]}"; do IFS='|' read -r n m t _ <<< "$c"; echo "$n ($m, ~$t s)"; done
  exit 0
fi

# The tests to run.
selected=()
for c in "${tests[@]}"; do
  IFS='|' read -r n m _ _ <<< "$c"
  if [ ${#names[@]} -gt 0 ]; then
    for want in "${names[@]}"; do [ "$want" = "$n" ] && selected+=("$c"); done
  elif [ "$mode" = full ] || [ "$m" = quick ]; then
    selected+=("$c")
  fi
done
[ ${#selected[@]} -gt 0 ] || { echo "No test named ${names[*]} (--list)" >&2; exit 2; }

start=$(date +%s)
# Compile jobs by the memory available, 1.5 GB each (a LibTorch source peaks at ~1.2 GB):
# beside a trainer -j10 would fill memory, and the out-of-memory killer picks the trainer.
available_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
make_jobs=$(( available_mb / 1536 ))
[ "$make_jobs" -gt 10 ] && make_jobs=10
[ "$make_jobs" -lt 1 ] && make_jobs=1
echo "Building (the game, build-shared/, build-tests/; $make_jobs jobs for ${available_mb} MB available) ..."
build_log=$repo/build-tests/build.log
mkdir -p "$repo/build-tests"
{
  nice -n 19 make -C build -j"$make_jobs" thud_test pyspiel &&
    nice -n 19 make -C build-shared -j"$make_jobs" open_spiel &&
    make -f thud/tests.mk -j"$make_jobs"
} > "$build_log" 2>&1 || { echo "BUILD FAILED (build-tests/build.log):"; tail -20 "$build_log"; exit 1; }

# The networks the tests read (made once; make_test_network is deterministic).
if [ ! -f "$nets/peaked/checkpoint-1.pt" ] || [ "$bin/make_test_network" -nt "$nets/peaked/vpnet.pb" ]; then
  rm -rf "$nets"
  mkdir -p "$nets"
  "$bin/make_test_network" dir="$nets/peaked" nn_model=resnet nn_width=8 nn_depth=1 \
    checkpoints=2 policy_scale=5 value_bias=-1 > /dev/null 2>&1 || { echo "make_test_network FAILED"; exit 1; }
fi

# Beside a running experiment (CLAUDE.md, session start: its processes): one test at a
# time in the idle scheduling class — even so it costs a trainer some speed (CLAUDE.md,
# Environment: ~25-35% for idle-class matches).
priority="nice -n 19"
experiments=$(ps -eo args | grep -E '[t]hud-runs/|[b]uild-shared/' | grep -v shell-snapshots)
if [ -n "$experiments" ]; then
  jobs=1
  priority="chrt --idle 0"
  echo "An experiment is running: the tests run one at a time, in the idle class."
fi

rm -rf "$logs"
mkdir -p "$logs"
run_one() {  # run_one NAME COMMAND: writes NAME.log, then NAME.result "exit seconds"
  local n=$1 cmd=$2 s e status
  s=$(date +%s.%N)
  $priority bash -c "$cmd" > "$logs/$n.log" 2>&1
  status=$?
  e=$(date +%s.%N)
  echo "$status $(echo "$e - $s" | bc)" > "$logs/$n.result"
}
# The longest first. Read through a command substitution, not a process substitution: bash
# lets `wait -n` return for a process substitution, which ran one test more at once than
# `jobs` (found 2026-10-15: two at once beside an experiment).
readarray -t order <<< "$(printf '%s\n' "${selected[@]}" | sort -t'|' -k3,3 -g -r)"
running=0
for c in "${order[@]}"; do
  IFS='|' read -r n m t cmd <<< "$c"
  run_one "$n" "$cmd" &
  running=$((running + 1))
  if [ "$running" -ge "$jobs" ]; then wait -n; running=$((running - 1)); fi
done
wait

failed=()
for c in "${selected[@]}"; do
  IFS='|' read -r n m _ _ <<< "$c"
  read -r status seconds < "$logs/$n.result"
  if [ "$status" -eq 0 ]; then
    printf "PASS %-26s %6.1f s\n" "$n" "$seconds"
  else
    printf "FAIL %-26s %6.1f s  exit %d, build-tests/last/%s.log:\n" "$n" "$seconds" "$status" "$n"
    grep -E "FAIL|Error|error|Assert|assert" "$logs/$n.log" | grep -v "^Warning" | head -5 | sed 's/^/       /'
    failed+=("$n")
  fi
done
echo "${#selected[@]} tests ($mode), $((${#selected[@]} - ${#failed[@]})) passed, ${#failed[@]} failed, $(( $(date +%s) - start )) s"
[ ${#failed[@]} -eq 0 ]
