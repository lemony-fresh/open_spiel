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

# Tests our trainer and the tools around it end to end (thud/az/; thud/PLAN.md, the GPU
# readiness checklist), on a small network (the convolutional head with the input
# planes, width 8, depth 1, from make_test_network) at 8 simulations, a buffer of 2,048:
#
#   1. A fresh run with every feature of ours on at once — two copies of the network
#      (--devices=/cpu:0,/cpu:0) with upstream's --explicit_learning, batched inference, symmetry augmentation, policy target
#      pruning, two learner batches a step, games against alpha-beta (depth 1 for both
#      sides), kickstarting per player (0.7, 0.3; depth 1), root noise per player (off
#      at the dwarfs' roots), simulations and policy temperatures per player, a
#      learning-rate warm-up and schedule, a value-loss weight of 2, playout caps (half
#      the moves in full) — runs 2
#      steps and exits 0; writes checkpoints 1, 2 and the latest, the buffer, and
#      learner.jsonl with steps 1 and 2 (losses finite, per side); config.json records
#      the settings; the actors log them; every batch norm counted each learning batch in
#      training mode (learner.jsonl's batch_norms; the race's fix, thud/az/vpnet.h).
#   2. Resuming (the trainer given the run's config.json, max_steps 3): step 3 follows
#      step 2 (not 1 again), the total games only grow, games against alpha-beta (half of
#      them) were played and counted, each step's learning rate is the schedule's (the
#      network's 0.001 x 0.5, 1, 0.5) and its value-loss weight 2, the actors' and the
#      learner's logs are appended to, not overwritten.
#   3. az_grow (8 x 1 -> 12 x 2, noise 0.01; its own tests of the function, the control
#      and the reload are fatal) on the latest network, then a run folder from it — the
#      config's new size, the buffer, learner.jsonl and the grown checkpoint as the
#      latest — resumed for a step, on one device (the learner sharing the network with
#      inference: the lock alone keeps them apart): the trainer takes the grown network
#      and learns on (step 4, losses finite, every batch norm counting both batches).
#   4. az_restructure as it was built for: a new policy head (the linear one, `resnet`)
#      on the run's trunk and value head, copied and frozen (freeze=copied), trained 20
#      batches on the run's buffer: exits 0 (its tests that the copy and the freeze keep
#      the values are fatal), writes the checkpoint and finite held-out losses. Not into
#      another width or depth: its copy test then fails by design, the trunk not copied
#      (2026-10-15; thud/PLAN.md, the checklist) — growth is az_grow's (3).
#      And training on with another value-loss weight (2026-10-17, E1's offline sweep):
#      the run's own network (copy=all, freeze=none; its test that the copy computes the
#      same is fatal) at value-loss weight 4, a warm-up of 20 batches, keep=every, 20
#      batches: checkpoints 0, 10 and 20; the learning rate half at batch 10 and full at
#      20 (the warm-up); the trunk's gradient sizes positive and finite, with
#      weighted_value_to_policy 4 x value / policy (the weight is on the network that
#      learns; that it changes the learning is learning_settings_test's part 4); the
#      summary records the weight, keep=every and the last batch. With optimizer=source
#      (the run's own optimizer) it exits 0: its test that a step with it equals the
#      run's own step, and that a fresh optimizer's differs (the control), is fatal.
#   5. az_fresh_optimizer: the latest network's weights with a fresh optimizer at a new
#      learning rate (a folder from make_test_network, at 0.001, against the run's 0.0001;
#      its own tests of the outputs and the rate are fatal) exits 0.
#
#   6. az_batch_norm_convert (2026-10-17, the race's fix) on the run's network and buffer
#      (3 batches of 200): exits 0 — its tests are fatal: evaluation mode unchanged (1e-4;
#      found ~3e-8), training mode's losses within a fifth of the source's distance from
#      evaluation mode's (found 0.13-0.14 before, under 0.001 after, 3 seeds) — and writes
#      the checkpoint.
#
#   thud/az/trainer_test.sh [BIN_DIR]    # default build-tests/bin
#
# KEEP=1 keeps its temporary folder (the runs and every program's log) for a look.
#
# Exits non-zero on any failure.

set -uo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
bin=${1:-$repo/build-tests/bin}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/trainer_test.XXXXXX")
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
  --devices=/cpu:0,/cpu:0 --explicit_learning --actors=3 --evaluators=0 --inference_batch_size=2 \
  --inference_threads=1 --max_simulations=8 --player_simulations=8,6 \
  --player_policy_temperature=1.5,1 --replay_buffer_size=2048 --replay_buffer_reuse=4 \
  --train_batch_size=64 --learner_batches=2 --checkpoint_freq=1 --max_steps=2 \
  --symmetry_augmentation --policy_target_pruning --alphabeta_share=0.5 \
  --alphabeta_depth=1,1 --kickstart_weight=0.3 --player_kickstart_weight=0.7,0.3 \
  --kickstart_depth=1 --player_policy_epsilon=0,0.25 --untried_move_value=sibling_mean_minus_reduction \
  --lr_warmup_steps=2 --lr_schedule=3:0.5 --value_loss_weight=2 \
  --playout_cap_full_share=0.5 --playout_cap_fast_simulations=4 \
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
# Every batch norm counted each of a step's learning batches in training mode (the race's
# fix, thud/az/vpnet.h): learner.jsonl's batch_norms, from step FIRST on.
exact_counts() {
  python3 - "$1" "$2" <<'EOF'
import json, sys
d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = [(x["step"], x["batch_norms"]) for x in d if x["step"] >= int(sys.argv[2]) and
       not x["batch_norms"]["batches"] == x["batch_norms"]["fewest"] ==
           x["batch_norms"]["most"] == 2]
if bad: print(bad)
sys.exit(1 if bad else 0)
EOF
}
check "1. two copies, explicit learning: each batch norm counted both learning batches a step" \
  exact_counts "$run/learner.jsonl" 1
check "1. config.json records the settings" python3 - "$run/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
want = {"devices": "/cpu:0,/cpu:0", "explicit_learning": True, "learner_batches": 2,
        "symmetry_augmentation": True,
        "policy_target_pruning": True, "alphabeta_share": 0.5, "alphabeta_depth": "1,1",
        "player_kickstart_weight": "0.7,0.3", "player_policy_epsilon": "0,0.25",
        "player_simulations": "8,6", "nn_width": 8, "nn_depth": 1, "lr_warmup_steps": 2,
        "lr_schedule": "3:0.5", "value_loss_weight": 2, "playout_cap_full_share": 0.5,
        "playout_cap_fast_simulations": 4}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad: print(bad)
sys.exit(1 if bad else 0)
EOF
check "1. the actors log the kickstarting settings" \
  grep -q "Kickstarting: weight 0.3 (per player 0.7, 0.3), depth 1" "$run/log-actor-0.txt"
check "1. the actors log the playout caps" \
  grep -q "Playout caps: a full search on 0.5 of the moves, else 4 simulations" "$run/log-actor-0.txt"

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
# The network's rate is make_test_network's 0.001: a warm-up of 2 steps, then 0.5 from 3.
check "2. the learning rate by the schedule (5e-4, 1e-3, 5e-4) and the value-loss weight 2" \
  python3 - "$run/learner.jsonl" <<'EOF'
import json, sys
d = [json.loads(l) for l in open(sys.argv[1])]
rates = [x["learning_rate"] for x in d]
ok = all(abs(a - b) < 1e-12 for a, b in zip(rates, [5e-4, 1e-3, 5e-4])) and len(rates) == 3
ok &= all(x["value_loss_weight"] == 2 for x in d)
if not ok: print(rates)
sys.exit(0 if ok else 1)
EOF
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
check "2. ... and the resumed step's batch norms counted both its batches" \
  exact_counts "$run/learner.jsonl" 3

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
c.update(path=sys.argv[2], nn_width=12, nn_depth=2, max_steps=4, graph_def="vpnet.pb",
         devices="/cpu:0", explicit_learning=False)  # One device: the lock alone.
json.dump(c, open(sys.argv[2] + "/config.json", "w"), indent=2)
EOF
  "$bin/az_trainer" "$grown/config.json" > "$tmp/grown.log" 2>&1
  check "3. the trainer resumes from the grown network and exits 0" [ $? -eq 0 ]
  check "3. ... on one device, the learner sharing it: its batch norms counted both batches" \
    exact_counts "$grown/learner.jsonl" 4
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

# 4. ... and training on with another value-loss weight.
"$bin/az_restructure" from="$run:-1" to="$tmp/value_weight" buffer="$run/replay_buffer.data" \
  buffer_size=2048 nn_model=resnet_conv_policy input_planes=1 copy=all freeze=none \
  learning_rate=0.002 batch_size=64 eval_every=10 max_batches=20 value_loss_weight=4 \
  warmup_batches=20 keep=every > "$tmp/value_weight.jsonl" 2> "$tmp/value_weight.err"
check "4. az_restructure at value-loss weight 4, keep=every, exits 0 (its copy test passed)" \
  [ $? -eq 0 ]
check "4. ... keeps checkpoints 0, 10 and 20" bash -c "for b in 0 10 20; do \
  [ -f '$tmp/value_weight/checkpoint-'\$b.pt ] || exit 1; done"
check "4. ... warm-up, gradient sizes, the weight in the reports and the summary" \
  python3 - "$tmp/value_weight.jsonl" <<'EOF'
import json, math, sys
lines = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
reports = {d["batches"]: d for d in lines if "batches" in d}
summary = [d for d in lines if d.get("summary")][0]
ok = abs(reports[10]["learning_rate"] - 0.001) < 1e-9 and \
     abs(reports[20]["learning_rate"] - 0.002) < 1e-9
for b in (0, 10, 20):
  g = reports[b]["gradients"]
  ok = ok and all(math.isfinite(g[k]) and g[k] > 0 for k in ("policy", "value")) and \
       abs(g["weighted_value_to_policy"] - 4 * g["value"] / g["policy"]) <= \
       1e-3 * g["weighted_value_to_policy"] + 1e-4
ok = ok and summary["value_loss_weight"] == 4 and summary["keep"] == "every" and \
     summary["last_batches"] == 20 and summary["warmup_batches"] == 20
sys.exit(0 if ok else 1)
EOF

check "4. ... with optimizer=source, exits 0 (its test of the optimizer passed)" bash -c \
  "'$bin/az_restructure' from='$run:-1' to='$tmp/value_weight_source' \
  buffer='$run/replay_buffer.data' buffer_size=2048 nn_model=resnet_conv_policy \
  input_planes=1 copy=all freeze=none learning_rate=0.002 batch_size=64 eval_every=10 \
  max_batches=10 value_loss_weight=4 keep=every optimizer=source \
  > '$tmp/value_weight_source.jsonl' 2> '$tmp/value_weight_source.err' && \
  grep -q '\"optimizer\": \"source\"' '$tmp/value_weight_source.jsonl'"

# 5. A fresh optimizer.
mkdir -p "$tmp/fresh_opt"
"$bin/make_test_network" dir="$tmp/fresh_opt/net" nn_model=resnet_conv_policy input_planes=1 \
  nn_width=8 nn_depth=1 checkpoints=1 > /dev/null 2>&1
"$bin/az_fresh_optimizer" from="$run:-1" to="$tmp/fresh_opt/net" step=5 > "$tmp/fresh_opt.log" 2>&1
check "5. az_fresh_optimizer exits 0 (its checks passed)" [ $? -eq 0 ]

# 6. The batch norms converted.
"$bin/az_batch_norm_convert" from="$run:-1" to="$tmp/converted" \
  buffer="$run/replay_buffer.data" buffer_size=2048 batches=3 batch_size=200 \
  > "$tmp/converted.jsonl" 2> "$tmp/converted.err"
check "6. az_batch_norm_convert exits 0 (its tests passed) and writes the checkpoint" \
  bash -c "[ $? -eq 0 ] && [ -f '$tmp/converted/checkpoint--1.pt' ]"

finish
