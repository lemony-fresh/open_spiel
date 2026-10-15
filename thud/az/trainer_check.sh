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

# Checks our trainer and the tools around it end to end (thud/az/; thud/PLAN.md, the GPU
# readiness checklist), on a small network (the convolutional head with the input
# planes, width 8, depth 1, from make_test_network) at 8 simulations, a buffer of 2,048:
#
#   1. A fresh run with every feature of ours on at once — two copies of the network
#      (--devices=/cpu:0,/cpu:0), batched inference, symmetry augmentation, policy target
#      pruning, two learner batches a step, games against alpha-beta (depth 1 for both
#      sides), kickstarting per player (0.7, 0.3; depth 1), root noise per player (off
#      at the dwarfs' roots), simulations and policy temperatures per player — runs 2
#      steps and exits 0; writes checkpoints 1, 2 and the latest, the buffer, and
#      learner.jsonl with steps 1 and 2 (losses finite, per side); config.json records
#      the settings; the actors log them.
#   2. Resuming (the trainer given the run's config.json, max_steps 3): step 3 follows
#      step 2 (not 1 again), the total games only grow, games against alpha-beta (half of
#      them) were played and counted, the actors' and the learner's logs are appended to,
#      not overwritten.
#   3. az_grow (8 x 1 -> 12 x 2, noise 0.01; its own checks of the function, the control
#      and the reload are fatal) on the latest network, then a run folder from it — the
#      config's new size, the buffer, learner.jsonl and the grown checkpoint as the
#      latest — resumed for a step: the trainer takes the grown network and learns on
#      (step 4, losses finite).
#   4. az_restructure as it was built for: a new policy head (the linear one, `resnet`)
#      on the run's trunk and value head, copied and frozen (freeze=copied), trained 20
#      batches on the run's buffer: exits 0 (its checks that the copy and the freeze keep
#      the values are fatal), writes the checkpoint and finite held-out losses. Not into
#      another width or depth: its copy check then fails by design, the trunk not copied
#      (2026-10-15; thud/PLAN.md, the checklist) — growth is az_grow's (3).
#   5. az_fresh_optimizer: the latest network's weights with a fresh optimizer at a new
#      learning rate (a folder from make_test_network, at 0.001, against the run's 0.0001;
#      its own checks of the outputs and the rate are fatal) exits 0.
#
#   thud/az/trainer_check.sh [BIN_DIR]    # default build-checks/bin
#
# KEEP=1 keeps its temporary folder (the runs and every program's log) for a look.
#
# Exits non-zero on any failure.

set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
bin=${1:-$repo/build-checks/bin}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/trainer_check.XXXXXX")
trap '[ -n "${KEEP:-}" ] || rm -rf "$tmp"' EXIT
run=$tmp/run
failures=0
export OMP_NUM_THREADS=2

check() {  # check DESCRIPTION COMMAND...
  local what=$1
  shift
  if "$@"; then echo "PASS $what"; else echo "FAIL $what"; failures=$((failures + 1)); fi
}
finish() {
  if [ "$failures" -eq 0 ]; then echo "ALL PASSED"; else echo "$failures FAILED"; fi
  [ "$failures" -eq 0 ]
  exit
}
# learner.jsonl's steps, one per line.
steps() { python3 -c "import json,sys; print(' '.join(str(json.loads(l)['step']) for l in open(sys.argv[1]) if l.strip()))" "$1"; }

"$bin/make_test_network" dir="$run" nn_model=resnet_conv_policy input_planes=1 nn_width=8 \
  nn_depth=1 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network"; exit 1; }

# 1. A fresh run.
"$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$run" \
  --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=8 --nn_depth=1 \
  --devices=/cpu:0,/cpu:0 --actors=3 --evaluators=0 --inference_batch_size=2 \
  --inference_threads=1 --max_simulations=8 --player_simulations=8,6 \
  --player_policy_temperature=1.5,1 --replay_buffer_size=2048 --replay_buffer_reuse=4 \
  --train_batch_size=64 --learner_batches=2 --checkpoint_freq=1 --max_steps=2 \
  --symmetry_augmentation --policy_target_pruning --alphabeta_share=0.5 \
  --alphabeta_depth=1,1 --kickstart_weight=0.3 --player_kickstart_weight=0.7,0.3 \
  --kickstart_depth=1 --player_policy_epsilon=0,0.25 --untried_move_value=sibling_mean_minus_reduction \
  > "$tmp/fresh.log" 2>&1
status=$?
check "1. a fresh run with every feature on exits 0" [ "$status" -eq 0 ]
[ "$status" -eq 0 ] || { tail -5 "$tmp/fresh.log"; finish; }
check "1. checkpoints 1, 2 and the latest, with their optimizers, and the buffer written" \
  bash -c "cd '$run' && ls checkpoint-1.pt checkpoint-2.pt checkpoint--1.pt \
    checkpoint-2-optimizer.pt checkpoint--1-optimizer.pt replay_buffer.data > /dev/null"
check "1. learner.jsonl has steps 1 and 2" [ "$(steps "$run/learner.jsonl")" = "1 2" ]
check "1. losses finite, per side" python3 - "$run/learner.jsonl" <<'EOF'
import json, math, sys
ok = True
for line in open(sys.argv[1]):
  d = json.loads(line)
  numbers = [d["loss"][k] for k in ("policy", "value", "l2reg", "sum")]
  for side in d["loss_by_side"].values():
    numbers += [v for v in side.values() if isinstance(v, (int, float))]
  ok &= len(d["loss_by_side"]) == 2 and all(math.isfinite(x) for x in numbers)
sys.exit(0 if ok else 1)
EOF
check "1. config.json records the settings" python3 - "$run/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
want = {"devices": "/cpu:0,/cpu:0", "learner_batches": 2, "symmetry_augmentation": True,
        "policy_target_pruning": True, "alphabeta_share": 0.5, "alphabeta_depth": "1,1",
        "player_kickstart_weight": "0.7,0.3", "player_policy_epsilon": "0,0.25",
        "player_simulations": "8,6", "nn_width": 8, "nn_depth": 1}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad: print(bad)
sys.exit(1 if bad else 0)
EOF
check "1. the actors log the kickstarting settings" \
  grep -q "Kickstarting: weight 0.3 (per player 0.7, 0.3), depth 1" "$run/log-actor-0.txt"

# 2. Resuming.
actor_lines=$(wc -l < "$run/log-actor-0.txt")
actor_first=$(head -1 "$run/log-actor-0.txt")
learner_lines=$(wc -l < "$run/log-learner.txt")
games_before=$(tail -1 "$run/learner.jsonl" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_trajectories'])")
python3 - "$run/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c["max_steps"] = 3
json.dump(c, open(sys.argv[1], "w"), indent=2)
EOF
"$bin/az_trainer" "$run/config.json" > "$tmp/resume.log" 2>&1
check "2. resuming exits 0" [ $? -eq 0 ]
check "2. step 3 follows step 2" [ "$(steps "$run/learner.jsonl")" = "1 2 3" ]
games_after=$(tail -1 "$run/learner.jsonl" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_trajectories'])")
check "2. the total games only grow ($games_before -> $games_after)" [ "$games_after" -gt "$games_before" ]
# Each game is against alpha-beta with probability 0.5: none in 11+ games is 1 in 2,000.
check "2. games against alpha-beta played and counted, over the 3 steps" \
  python3 - "$run/learner.jsonl" <<'EOF'
import json, sys
n = sum(d["vs_alphabeta"]["network_as_dwarfs"]["num"] + d["vs_alphabeta"]["network_as_trolls"]["num"]
        for d in map(json.loads, open(sys.argv[1])))
sys.exit(0 if n > 0 else 1)
EOF
appended() {
  [ "$(wc -l < "$run/log-actor-0.txt")" -gt "$actor_lines" ] &&
    [ "$(head -1 "$run/log-actor-0.txt")" = "$actor_first" ] &&
    [ "$(wc -l < "$run/log-learner.txt")" -gt "$learner_lines" ]
}
check "2. the actors' and the learner's logs are appended to" appended

# 3. Growth, and training on from the grown network.
grown=$tmp/grown
"$bin/az_grow" from="$run:-1" to="$grown" nn_width=12 nn_depth=2 noise=0.01 step=3 \
  positions=40 > "$tmp/grow.json" 2> "$tmp/grow.log"
check "3. az_grow 8 x 1 -> 12 x 2 exits 0 (its checks passed)" [ $? -eq 0 ]
if [ -f "$grown/checkpoint-3.pt" ]; then
  cp "$run/learner.jsonl" "$run/replay_buffer.data" "$grown/"
  cp "$grown/checkpoint-3.pt" "$grown/checkpoint--1.pt"
  cp "$grown/checkpoint-3-optimizer.pt" "$grown/checkpoint--1-optimizer.pt"
  python3 - "$run/config.json" "$grown" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(path=sys.argv[2], nn_width=12, nn_depth=2, max_steps=4, graph_def="vpnet.pb")
json.dump(c, open(sys.argv[2] + "/config.json", "w"), indent=2)
EOF
  "$bin/az_trainer" "$grown/config.json" > "$tmp/grown.log" 2>&1
  check "3. the trainer resumes from the grown network and exits 0" [ $? -eq 0 ]
  check "3. ... and learns step 4 with finite losses" python3 - "$grown/learner.jsonl" <<'EOF'
import json, math, sys
d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
sys.exit(0 if d["step"] == 4 and all(math.isfinite(d["loss"][k]) for k in ("policy", "value")) else 1)
EOF
fi

# 4. Distillation into a new structure.
"$bin/az_restructure" from="$run:-1" to="$tmp/restructured" buffer="$run/replay_buffer.data" \
  buffer_size=2048 nn_model=resnet input_planes=1 freeze=copied batch_size=64 eval_every=10 \
  max_batches=20 step=3 \
  > "$tmp/restructure.log" 2>&1
check "4. az_restructure, a new head on the frozen trunk, exits 0 (its checks passed)" [ $? -eq 0 ]
check "4. ... writes the checkpoint" [ -f "$tmp/restructured/checkpoint-3.pt" ]
check "4. ... with finite held-out losses" bash -c \
  "grep -qiE 'held' '$tmp/restructure.log' && ! grep -qiE '\bnan\b|\binf\b' '$tmp/restructure.log'"

# 5. A fresh optimizer.
mkdir -p "$tmp/fresh_opt"
"$bin/make_test_network" dir="$tmp/fresh_opt/net" nn_model=resnet_conv_policy input_planes=1 \
  nn_width=8 nn_depth=1 checkpoints=1 > /dev/null 2>&1
"$bin/az_fresh_optimizer" from="$run:-1" to="$tmp/fresh_opt/net" step=5 > "$tmp/fresh_opt.log" 2>&1
check "5. az_fresh_optimizer exits 0 (its checks passed)" [ $? -eq 0 ]

finish
