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

# Runs our checks (thud/PLAN.md, *Tests*: what each covers, and the gaps): builds what is
# out of date — the game into build/ (thud_test, pyspiel) and build-shared/
# (libopen_spiel.so), the checks into build-checks/ (thud/checks.mk: without -DNDEBUG) —
# then runs the checks, a few at once, at the lowest CPU priority, and prints one line
# each with its time. Each check's output is kept in build-checks/last/NAME.log.
#
#   thud/run_checks.sh            # the quick checks (~20 s): what the pre-commit hook runs
#   thud/run_checks.sh --full     # every check (~3 min): what the pre-push hook runs
#   thud/run_checks.sh NAME ...   # only these (--list shows the names)
#   [--jobs N]                    # checks at once (default 4; 1 beside an experiment)
#
# Exits non-zero if a build or any check fails. The checks need no run folder: the ones
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
    --playthrough | --sanitized) names+=("$1") ;;  # Run as checks, below.
    -*) echo "Unknown option $1" >&2; exit 2 ;;
    *) names+=("$1") ;;
  esac
  shift
done

bin=$repo/build-checks/bin
logs=$repo/build-checks/last
nets=$repo/build-checks/nets

# NAME, quick or full, its time alone in seconds (measured 2026-10-15: the longest start
# first), and the command (run from the repo root). Quick: seconds each.
checks=(
  "thud_test|quick|2|build/games/thud_test"
  "crosscheck_tests|quick|0.2|python3 thud/experiments/crosscheck_tests.py"
  "alphabeta_training_check|quick|0.2|$bin/alphabeta_training_check"
  "augmentation_check|quick|0.2|$bin/augmentation_check"
  "grow_check|quick|2.4|$bin/grow_check"
  "growing_buffer_check|quick|0.2|$bin/growing_buffer_check"
  "instrumentation_check|quick|0.2|$bin/instrumentation_check"
  "search_settings_check|quick|0.3|$bin/search_settings_check"
  "teacher_mix_check|quick|0.3|$bin/teacher_mix_check"
  "features_check|quick|3|$bin/features_check"
  "replay_buffer_check|quick|0.5|$bin/replay_buffer_check $logs"
  "az_buffer_truncate|quick|0|$bin/az_buffer_truncate self_test"
  "pruning_check|quick|13|$bin/pruning_check $nets/peaked 0"
  "trainer_check|quick|26|thud/az/trainer_check.sh $bin"
  "playthrough|full|3.5|thud/run_checks.sh --playthrough"
  "sanitized|full|10|thud/run_checks.sh --sanitized"
  "identity_check|full|88|$bin/identity_check $nets/peaked 0 1"
  "untried_move_check|full|107|$bin/untried_move_check $nets/peaked 0 games=1"
  "input_planes_check|full|65|$bin/input_planes_check $logs/input_planes_scratch"
  "kickstart_check|full|131|$bin/kickstart_check"
  "conv_policy_check|full|148|$bin/conv_policy_check"
  "az_match_check|full|97|thud/experiments/az_match_check.sh $bin"
)

# Two checks that are commands of their own, run as checks above.
case "${names[0]:-}" in
  --playthrough)
    # The playthrough regression baseline (CLAUDE.md), as upstream's playthrough_test.py
    # checks it: the moves recorded in the baseline replayed, every line compared (a new
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
    # The checks without LibTorch under AddressSanitizer and UBSan; the control: the
    # binaries are instrumented (a sanitizer that is not there reports nothing).
    status=0
    for b in replay_buffer_check features_check az_buffer_truncate; do
      # grep -c reads all of nm's output (grep -q would stop early: nm's SIGPIPE fails
      # the pipe under pipefail).
      nm "$bin/sanitized/$b" | grep -c __asan_report_load > /dev/null ||
        { echo "FAIL control: $b is not instrumented"; status=1; }
    done
    mkdir -p "$logs/sanitized_scratch"
    "$bin/sanitized/replay_buffer_check" "$logs/sanitized_scratch" &&
      "$bin/sanitized/features_check" && "$bin/sanitized/az_buffer_truncate" self_test || status=1
    exit $status ;;
esac
if [ "$list" -eq 1 ]; then
  for c in "${checks[@]}"; do IFS='|' read -r n m t _ <<< "$c"; echo "$n ($m, ~$t s)"; done
  exit 0
fi

# The checks to run.
selected=()
for c in "${checks[@]}"; do
  IFS='|' read -r n m _ _ <<< "$c"
  if [ ${#names[@]} -gt 0 ]; then
    for want in "${names[@]}"; do [ "$want" = "$n" ] && selected+=("$c"); done
  elif [ "$mode" = full ] || [ "$m" = quick ]; then
    selected+=("$c")
  fi
done
[ ${#selected[@]} -gt 0 ] || { echo "No check named ${names[*]} (--list)" >&2; exit 2; }

start=$(date +%s)
echo "Building (the game, build-shared/, build-checks/) ..."
build_log=$repo/build-checks/build.log
mkdir -p "$repo/build-checks"
{
  nice -n 19 make -C build -j10 thud_test pyspiel &&
    nice -n 19 make -C build-shared -j10 open_spiel &&
    make -f thud/checks.mk -j10
} > "$build_log" 2>&1 || { echo "BUILD FAILED (build-checks/build.log):"; tail -20 "$build_log"; exit 1; }

# The networks the checks read (made once; make_test_network is deterministic).
if [ ! -f "$nets/peaked/checkpoint-1.pt" ] || [ "$bin/make_test_network" -nt "$nets/peaked/vpnet.pb" ]; then
  rm -rf "$nets"
  mkdir -p "$nets"
  "$bin/make_test_network" dir="$nets/peaked" nn_model=resnet nn_width=8 nn_depth=1 \
    checkpoints=2 policy_scale=5 value_bias=-1 > /dev/null 2>&1 || { echo "make_test_network FAILED"; exit 1; }
fi

# Beside a running experiment (CLAUDE.md, session start: its processes): one check at a
# time in the idle scheduling class — even so it costs a trainer some speed (CLAUDE.md,
# Environment: ~25-35% for idle-class matches).
priority="nice -n 19"
experiments=$(ps -eo args | grep -E '[t]hud-runs/|[b]uild-shared/' | grep -v shell-snapshots)
if [ -n "$experiments" ]; then
  jobs=1
  priority="chrt --idle 0"
  echo "An experiment is running: the checks run one at a time, in the idle class."
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
running=0
while IFS= read -r c; do
  IFS='|' read -r n m t cmd <<< "$c"
  run_one "$n" "$cmd" &
  running=$((running + 1))
  if [ "$running" -ge "$jobs" ]; then wait -n; running=$((running - 1)); fi
done < <(printf '%s\n' "${selected[@]}" | sort -t'|' -k3,3 -g -r)
wait

failed=()
for c in "${selected[@]}"; do
  IFS='|' read -r n m _ _ <<< "$c"
  read -r status seconds < "$logs/$n.result"
  if [ "$status" -eq 0 ]; then
    printf "PASS %-26s %6.1f s\n" "$n" "$seconds"
  else
    printf "FAIL %-26s %6.1f s  exit %d, build-checks/last/%s.log:\n" "$n" "$seconds" "$status" "$n"
    grep -E "FAIL|Error|error|Assert|assert" "$logs/$n.log" | grep -v "^Warning" | head -5 | sed 's/^/       /'
    failed+=("$n")
  fi
done
echo "${#selected[@]} checks ($mode), $((${#selected[@]} - ${#failed[@]})) passed, ${#failed[@]} failed, $(( $(date +%s) - start )) s"
[ ${#failed[@]} -eq 0 ]
