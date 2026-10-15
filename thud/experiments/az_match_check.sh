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

# Checks az_match end to end (thud/experiments/az_match.cc; thud/PLAN.md, the GPU readiness
# checklist), on two small untrained networks (make_test_network: the convolutional head
# with the input planes, width 8, depth 1, a peaked prior), 2 pairs at 8 simulations:
#
#   1. Reproducible: the same match twice plays the same battles (the games files equal).
#   2. The record: each pair's two battles start from the same opening, network A on each
#      side once; the summary's mean pair margin, pairs won / drawn / lost and per-side
#      means are those of the pair lines, and each pair's margin is its battles' sum.
#   3. Self-play's conditions, as documented, "the same battles whichever thread plays
#      them": noise=1,1 and sample_plies=6, each with 1 and 2 threads, play the same
#      battles. Controls: each differs from the match without it, and noise with seed 2
#      from seed 1.
#   4. devices=/cpu:0,/cpu:0 (two copies of each network) plays the default's battles;
#      the control that the option was taken: the summary records the two copies.
#   5. The teacher in A's prior (teacher_mix_a=0,0.5): reproducible, and (the control)
#      different from the match without it. Its arithmetic: teacher_mix_check.
#   6. Players without a network, as documented, the same battles whichever thread:
#      b=alphabeta:1 and b=mcts:20 with 1 and 2 threads; b=hexparrot:1 too if hexparrot
#      is set up (thud/experiments/perft_reference.py) — skipped, and said so, if not.
#
#   thud/experiments/az_match_check.sh [BIN_DIR]    # default build-checks/bin
#
# Exits non-zero on any failure.

set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
bin=${1:-$repo/build-checks/bin}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/az_match_check.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
failures=0

check() {  # check DESCRIPTION COMMAND...
  local what=$1
  shift
  if "$@"; then echo "PASS $what"; else echo "FAIL $what"; failures=$((failures + 1)); fi
}
# Battles are written as they finish, in another order with more threads: compared sorted.
same() { cmp -s <(sort "$tmp/$1.games") <(sort "$tmp/$2.games"); }
differ() { ! same "$@"; }

for net in a:1 b:5; do
  "$bin/make_test_network" dir="$tmp/${net%%:*}" nn_model=resnet_conv_policy input_planes=1 \
    nn_width=8 nn_depth=1 checkpoints=1 seed="${net##*:}" policy_scale=5 > /dev/null 2>&1 ||
    { echo "FAIL make_test_network"; exit 1; }
done

match() {  # match NAME ARGS...: A against B unless ARGS give b=
  local name=$1
  shift
  local b="b=$tmp/b:0"
  for arg in "$@"; do [[ $arg == b=* ]] && b=$arg; done
  OMP_NUM_THREADS=1 "$bin/az_match" a="$tmp/a:0" "$b" sims=8 pairs=2 opening=4 threads=1 \
    hexparrot_player="$repo/thud/experiments/hexparrot_player.py" \
    batch=1 untried=sibling_mean_minus_reduction "$@" games="$tmp/$name.games" \
    > "$tmp/$name.jsonl" 2> "$tmp/$name.err" || {
    echo "FAIL az_match $name exited non-zero:"; tail -3 "$tmp/$name.err"
    failures=$((failures + 1)); return 1
  }
  [ -s "$tmp/$name.games" ] ||
    { echo "FAIL az_match $name wrote no games"; failures=$((failures + 1)); return 1; }
}

match base && match base_again || exit 1
check "1. the same match twice plays the same battles" same base base_again

check "2. the record: openings shared within a pair, A on each side once, the summary's \
numbers those of the pair lines" python3 - "$tmp/base.jsonl" "$tmp/base.games" <<'EOF'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pairs = [l for l in lines if not l.get("summary")]
summary = [l for l in lines if l.get("summary")][0]
games = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
ok = len(pairs) == summary["pairs"] == 2 and len(games) == 4
by_pair = {}
for g in games:
  by_pair.setdefault(g["pair"], []).append(g)
for p, (x, y) in by_pair.items():
  ok &= x["actions"][:4] == y["actions"][:4] and {x["a_player"], y["a_player"]} == {0, 1}
margins = {p["pair"]: p["a_pair_margin"] for p in pairs}
for p, battles in by_pair.items():
  ok &= abs(sum(b["a_margin"] for b in battles) - margins[p]) < 1e-9
mean = sum(margins.values()) / len(margins)
ok &= abs(summary["a_mean_pair_margin"] - mean) < 0.006
ok &= summary["a_pairs_won"] == sum(m > 0 for m in margins.values())
ok &= summary["drawn"] == sum(m == 0 for m in margins.values())
ok &= summary["lost"] == sum(m < 0 for m in margins.values())
for side, player in (("dwarfs", 0), ("trolls", 1)):
  m = [g["a_margin"] for g in games if g["a_player"] == player]
  ok &= abs(summary[f"a_mean_as_{side}"] - sum(m) / len(m)) < 0.006
sys.exit(0 if ok else 1)
EOF

for opt in noise=1,1 sample_plies=6; do
  n=${opt%%=*}
  match "${n}_1" "$opt" && match "${n}_2" "$opt" threads=2 || continue
  check "3. $opt: 1 and 2 threads play the same battles" same "${n}_1" "${n}_2"
  check "3. control: $opt differs from the match without it" differ "${n}_1" base
done
match noise_seed2 noise=1,1 seed=2 &&
  check "3. control: noise with seed 2 differs from seed 1" differ noise_seed2 noise_1

match devices devices=/cpu:0,/cpu:0 && {
  check "4. devices=/cpu:0,/cpu:0 plays the default's battles" same devices base
  check "4. control: the summary records the two copies" \
    grep -q '"devices": "/cpu:0,/cpu:0"' "$tmp/devices.jsonl"
}

match teacher teacher_mix_a=0,0.5 && match teacher_again teacher_mix_a=0,0.5 && {
  check "5. teacher_mix_a=0,0.5: reproducible" same teacher teacher_again
  check "5. control: teacher_mix_a=0,0.5 differs from the match without it" differ teacher base
}

opponents="alphabeta:1 mcts:20"
if python3 -c "import sys; sys.path.insert(0, '$repo/thud/experiments'); import perft_reference as r; r.load_hexparrot(r.DEFAULT_HEXPARROT)" \
    > /dev/null 2>&1; then
  opponents="$opponents hexparrot:1"
else
  echo "SKIP 6. b=hexparrot:1 — hexparrot is not set up here (perft_reference.py)"
fi
for o in $opponents; do
  n=${o%%:*}
  match "${n}_1" "b=$o" && match "${n}_2" "b=$o" threads=2 &&
    check "6. b=$o: 1 and 2 threads play the same battles" same "${n}_1" "${n}_2"
done

if [ "$failures" -eq 0 ]; then echo "ALL PASSED"; else echo "$failures FAILED"; fi
[ "$failures" -eq 0 ]
