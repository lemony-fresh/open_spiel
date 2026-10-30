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
#      sides), kickstarting per player (0.7, 0.3; depth 1; the teacher's ties shared by
#      the search, 2026-10-23), root noise per player (off
#      at the dwarfs' roots), simulations and policy temperatures per player, a
#      learning-rate warm-up and schedule, a value-loss weight of 2, playout caps (half
#      the moves in full), mixed-precision inference (2026-10-19), the learner's own
#      LibTorch thread count (3, against OMP_NUM_THREADS 2; 2026-10-19) — runs 2
#      steps and exits 0; writes checkpoints 1, 2 and the latest, the buffer, and
#      learner.jsonl with steps 1 and 2 (losses finite, per side); config.json records
#      the settings; the actors log them; every batch norm counted each learning batch in
#      training mode (learner.jsonl's batch_norms; the race's fix, thud/az/vpnet.h); each
#      step's thread counts are the learner's 3 and the inference thread's 2; per side the
#      prior's distance from the teacher (2026-10-27: a finite KL, a mass in [0, 1]).
#   2. Resuming (the trainer given the run's config.json, max_steps 3): step 3 follows
#      step 2 (not 1 again), the total games only grow, games against alpha-beta (half of
#      them) were played and counted, each step's learning rate is the schedule's (the
#      network's 0.001 x 0.5, 1, 0.5) and its value-loss weight 2, the learner's thread
#      count still 3 (from config.json), the actors' and the learner's logs are appended
#      to, not overwritten.
#   3. az_grow (8 x 1 -> 12 x 2, noise 0.01; its own tests of the function, the control
#      and the reload are fatal) on the latest network, then a run folder from it — the
#      config's new size, the buffer, learner.jsonl and the grown checkpoint as the
#      latest — resumed for a step, on one device (the learner sharing the network with
#      inference: the lock alone keeps them apart): the trainer takes the grown network
#      and learns on (step 4, losses finite, every batch norm counting both batches);
#      with learner_threads 0 its learner's thread count is OMP_NUM_THREADS's 2 (the
#      control for part 1's 3).
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
#   7. The capacity tests (2026-10-18; thud/DESIGN_dwarf_signal.md §10): az_dormant_probe on
#      the run's network and buffer exits 0 — its tests are fatal: the values after its
#      rectifiers equal Inference's, and a killed channel (the positive control) is found
#      dead — and reports the network and a fresh one; az_plasticity (the run's network,
#      fresh widths 8 and 2, random targets) exits 0 — its test that the wider fresh
#      network fits better is fatal; az_restructure copy=none (a fresh network trained on
#      the buffer) exits 0 and starts from outputs unlike the source's (the control that
#      nothing was copied).
#   8. The trainer benchmark (thud/experiments/bench_trainer.sh, 2026-10-19) on the run (3
#      steps), two settings — 3 actors, batches of 2, 1 inference thread, OMP_NUM_THREADS 1,
#      the learner's threads 3, then 0 (the control: OMP_NUM_THREADS's 1): exits 0, a line
#      per setting with step 5's period, learning, wait, games waiting and taken, the bound,
#      the thread counts as set (3 and 1; 1 and 1) and the probe's samples (a stand-in
#      printing 7: their mean 7); the summary names one of them the
#      fastest; each setting's buffer deleted, its logs kept; run again it reprints the same
#      results without training (resumed from the results).
#   9. az_teacher_vs_search (2026-10-25) on the run: its candidates' measures in range, the
#      uniform spread's agreement and breadth as computed, the visits' gap to themselves 0;
#      mixes= (2026-10-29): at the run's own weights a mix is the target exactly (the control:
#      the dwarfs' mix at 0.3 is not); gumbel=1: its three candidates in range, its move all
#      on one move; considered=1,4 gumbel_scale=0: the candidates named by the count, each
#      move one-hot (the control: the two counts' measures differ); since 2026-10-30
#      gumbel_sims=0,12 (the names with the budget; the control, the budgets' measures differ)
#      and targetN (the trainer's target at each extra budget of sims) in range; the
#      search's signals per position in range, their effective moves the summary's.
#  10. The carry-over and the stop between moves (2026-10-27): a fresh run of one step with a
#      long last learning step (300 batches) and short battles (10 turns without a capture
#      end one), so that games wait in the queue at its end:
#      the learner puts them into the buffer and writes the note (carried_games.txt, "1 STATES
#      GAMES", its log saying so; more than 0 games), and the actors' games in play end at the
#      stop (logged as abandoned). Resumed to step 2: the note read, step 2's record carrying
#      its counts, the buffer grown by exactly the carried states plus the collected ones, the
#      collected ones at least the step's 512 less the carried, the games counted once; the
#      note then the new end's (after step 2). The control: a stale note (for step 5) at the
#      resume to step 3 is ignored and removed — step 3 carries nothing and collects its 512.
#      Without a teacher, no distance from it is logged (the control for part 1's, below).
#  11. The completed-Q policy target (2026-10-29, --policy_target=completed_q; its arithmetic and
#      wiring: completed_q_test): a fresh run of one step with it exits 0 with finite losses,
#      config.json records it and the actors log it; the control, part 1's run (visits, the
#      default) logs no policy target.
#  12. The Gumbel search (2026-10-29, --gumbel_search; its rules and searches: gumbel_search_test):
#      a fresh run of one step with it (4 considered actions, 16 simulations, an evaluator
#      playing on PUCT) exits 0 with finite losses, config.json records it and the actors log
#      it; the control, part 1's run logs none.
#  13. An interrupt while the learner collects (2026-10-30: R7's trainer, interrupted at step
#      462's collection, hung — its actors abandon their games at a stop and push none, and the
#      learner waited on the empty queue for ever): a fresh run whose first step collects for
#      long (long battles, 16,384 positions) is interrupted once the actors have played a
#      game; it must exit by itself within 60 s, with status 0 and "Exiting cleanly". The
#      code before the fix hung here (the test's positive control, run 2026-10-30).
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
  --alphabeta_depth=1,1 --alphabeta_player=search,teacher --kickstart_weight=0.3 --player_kickstart_weight=0.7,0.3 \
  --kickstart_depth=1 --player_kickstart_depth=1,2 --kickstart_ties=search \
  --player_policy_epsilon=0,0.25 --untried_move_value=sibling_mean_minus_reduction \
  --lr_warmup_steps=2 --lr_schedule=3:0.5 --value_loss_weight=2 --inference_precision=mixed \
  --playout_cap_full_share=0.5 --playout_cap_fast_simulations=4 --learner_threads=3 \
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
        "playout_cap_fast_simulations": 4, "inference_precision": "mixed",
        "learner_threads": 3, "kickstart_ties": "search", "player_kickstart_depth": "1,2",
        "alphabeta_player": "search,teacher"}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad: print(bad)
sys.exit(1 if bad else 0)
EOF
# Each step's LibTorch thread counts from step FIRST on (learner.jsonl's threads): the
# learner's LEARNER and every inference thread's INFERENCE, at least one reporting.
thread_counts() {
  python3 - "$1" "$2" "$3" "$4" <<'EOF'
import json, sys
d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
first, learner, inference = map(int, sys.argv[2:])
bad = [(x["step"], x["threads"]) for x in d if x["step"] >= first and not (
       x["threads"]["learner"] == learner and x["threads"]["inference"] and
       all(n == inference for n in x["threads"]["inference"]))]
if bad: print(bad)
sys.exit(1 if bad else 0)
EOF
}
check "1. the learner's thread count 3, the inference thread's OMP_NUM_THREADS 2" \
  thread_counts "$run/learner.jsonl" 1 3 2
check "1. the prior's distance from the teacher logged per side (2026-10-27): a finite KL, a mass in [0, 1]" \
  python3 - "$run/learner.jsonl" <<'EOF'
import json, math, sys
# A side with no self-play positions in a step (all its games against alpha-beta, possible in
# these short steps) has no statistics there: every side present must carry them, and some
# step must have both sides (2026-10-29: the check failed once on a step without one).
ok, both = True, False
# 2026-10-30: it failed a third time, its log then showing why — a step whose side entries both
# held 0 positions (all its games against alpha-beta): "present" means positions to measure.
for d in map(json.loads, open(sys.argv[1])):
  present = [side for side in ("dwarfs", "trolls")
             if d["by_side"].get(side, {}).get("positions", 0) > 0]
  both |= len(present) == 2
  for side in present:
    s = d["by_side"][side]
    kl, mass = s.get("mean_kl_teacher_prior"), s.get("mean_prior_on_teacher_best")
    good = kl is not None and math.isfinite(kl) and kl >= 0 and mass is not None and 0 <= mass <= 1
    if not good:  # Its cause, for the log (2026-10-29: failed twice under load, not reproduced).
      print(f"step {d['step']} {side}: kl {kl!r} mass {mass!r} positions {s.get('positions')}")
    ok &= good
if not both:
  print("no step with both sides' self-play positions:",
        [(d["step"], sorted(d["by_side"])) for d in map(json.loads, open(sys.argv[1]))])
sys.exit(0 if ok and both else 1)
EOF
check "1. the actors log the kickstarting settings" \
  grep -q "Kickstarting: weight 0.3 (per player 0.7, 0.3), depth 1 (per player 1, 2), temperature 1, ties search" "$run/log-actor-0.txt"
check "1. the actors log the training opponent per side, the teacher as the trolls" \
  grep -q "Training against alpha-beta: share 0.5; as the dwarfs the search at depth 1, as the trolls the teacher at depth 1" "$run/log-actor-0.txt"
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
check "2. ... and its learner's thread count still 3 (from config.json)" \
  thread_counts "$run/learner.jsonl" 3 3 2

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
         devices="/cpu:0", explicit_learning=False,  # One device: the lock alone.
         learner_threads=0)  # The control for part 1's thread counts.
json.dump(c, open(sys.argv[2] + "/config.json", "w"), indent=2)
EOF
  "$bin/az_trainer" "$grown/config.json" > "$tmp/grown.log" 2>&1
  check "3. the trainer resumes from the grown network and exits 0" [ $? -eq 0 ]
  check "3. ... on one device, the learner sharing it: its batch norms counted both batches" \
    exact_counts "$grown/learner.jsonl" 4
  check "3. ... with learner_threads 0 the learner's thread count is OMP_NUM_THREADS's 2" \
    thread_counts "$grown/learner.jsonl" 4 2 2
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

# 7. The capacity tests.
"$bin/az_dormant_probe" nets="run=$run:-1" buffer="$run/replay_buffer.data" buffer_size=2048 \
  positions=256 batch=64 > "$tmp/dormant.jsonl" 2> "$tmp/dormant.err"
check "7. az_dormant_probe exits 0 (its tests passed) and reports the network and a fresh one" \
  bash -c "[ $? -eq 0 ] && [ \$(grep -c '\"summary\": true' '$tmp/dormant.jsonl') -eq 2 ]"
"$bin/az_plasticity" nets="run=$run:-1,f8=fresh:8,f2=fresh:2" buffer="$run/replay_buffer.data" \
  buffer_size=2048 scratch="$tmp/plasticity" positions=64 batch=32 batches=40 report=20 \
  learning_rate=0.01 > "$tmp/plasticity.jsonl" 2> "$tmp/plasticity.err"
check "7. az_plasticity exits 0 (the wider fresh network fits better) and reports all three" \
  bash -c "[ $? -eq 0 ] && [ \$(grep -c '\"summary\": true' '$tmp/plasticity.jsonl') -eq 3 ]"
"$bin/az_restructure" from="$run:-1" to="$tmp/reset" buffer="$run/replay_buffer.data" \
  buffer_size=2048 nn_model=resnet_conv_policy input_planes=1 copy=none freeze=none \
  batch_size=64 eval_every=10 max_batches=10 > "$tmp/reset.jsonl" 2> "$tmp/reset.err"
check "7. az_restructure copy=none exits 0 from outputs unlike the source's" \
  bash -c "[ $? -eq 0 ] && grep -q 'copy=none: a fresh network' '$tmp/reset.err' && \
    python3 -c \"import re,sys; d=float(re.search(r'values differ from the source.s by (\\S+)', open(sys.argv[1]).read()).group(1)); sys.exit(0 if d > 1e-3 else 1)\" '$tmp/reset.err'"

# 8. The trainer benchmark.
BIN=$bin BENCH_PROBE='echo 7' "$repo/thud/experiments/bench_trainer.sh" "$run" "$tmp/bench" \
  3,2,1,1,3 3,2,1,1,0 > "$tmp/bench.jsonl" 2> "$tmp/bench.err"
check "8. bench_trainer.sh exits 0" [ $? -eq 0 ]
check "8. ... a line per setting, timed, the thread counts as set; the summary names one" \
  python3 - "$tmp/bench.jsonl" <<'EOF'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1])]
ok = len(lines) == 3
for r, (learner, inference) in zip(lines[:2], ((3, 1), (1, 1))):
  ok &= (r.get("exit") == 0 and r.get("period", 0) > 0 and r.get("learning", 0) > 0 and
         r.get("waited", -1) >= 0 and r.get("games_taken", 0) > 0 and
         r.get("games_waiting", -1) >= 0 and r.get("bound") in ("learner", "self-play") and
         r["threads"]["learner"] == learner and r["threads"]["inference"] and
         all(n == inference for n in r["threads"]["inference"]) and
         abs(r.get("paused_s", 99)) < 5 and r.get("probe", {}).get("mean") == 7)
ok &= lines[2].get("summary") is True and lines[2].get("timed") == 2 and \
      lines[2].get("fastest") in ("s0", "s1")
if not ok: print(lines)
sys.exit(0 if ok else 1)
EOF
check "8. ... each setting's buffer deleted, its logs kept" bash -c \
  "[ ! -e '$tmp/bench/s0/replay_buffer.data' ] && [ -s '$tmp/bench/s1/log-learner.txt' ]"
BIN=$bin "$repo/thud/experiments/bench_trainer.sh" "$run" "$tmp/bench" 3,2,1,1,3 3,2,1,1,0 \
  > "$tmp/bench_again.jsonl" 2>&1
check "8. ... run again, the same results, nothing trained" bash -c \
  "cmp -s '$tmp/bench.jsonl' '$tmp/bench_again.jsonl' && [ ! -e '$tmp/bench/s0/replay_buffer.data' ]"

# 9. The teacher against the search (az_teacher_vs_search, 2026-10-24): on the run's own
# games, with its settings (the teacher at depths 1 and 2, weights 0.7 and 0.3), the
# trainer's own budget and 24 simulations against references of 16 and 32.
"$bin/az_teacher_vs_search" run="$run" step=-1 sims=24 refs=16,32 last=2 max=3 threads=2 batch=2 mixes=0.3,0.7 gumbel=1 \
  inference_threads=1 > "$tmp/teacher_vs_search.jsonl" 2> "$tmp/teacher_vs_search.err"
check "9. az_teacher_vs_search exits 0 (its self-test passed), with the run's teacher" \
  bash -c "[ $? -eq 0 ] && grep -q \"the teacher's depths 1, 2, weights 0.7, 0.3\" '$tmp/teacher_vs_search.err'"
check "9. ... every position each candidate against both references, the uniform one agreeing 1/legal (ties shared); the summaries per side, candidate and reference, the uniform one spread over every legal move, and the references' stability" \
  python3 - "$tmp/teacher_vs_search.jsonl" <<'EOF'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1])]
rows = [d for d in lines if "position" in d]
summaries = [d for d in lines if d.get("summary") and not d.get("stability")]
stability = [d for d in lines if d.get("stability")]
ok = len(rows) >= 4 and {d["side"] for d in rows} == {"dwarfs", "trolls"}
for d in rows:
  for c in ("visits", "visits24", "completed_q", "teacher", "target", "cq_target", "uniform",
            "gumbel", "gumbel_target", "gumbel_move"):
    for r in (16, 32):
      x = d[f"{c}_ref{r}"]
      ok &= all(0 <= x[k] <= 1 + 1e-9 for k in ("agrees", "mass_on_best", "distance"))
    ok &= abs(d[f"uniform_ref16"]["agrees"] - 1 / d["legal"]) < 1e-3
  for r in (16, 32):  # The move: all its probability on one move.
    x = d[f"gumbel_move_ref{r}"]
    ok &= x["agrees"] in (0, 1) and x["mass_on_best"] == x["agrees"]
ok &= len(summaries) == 2 * 13 * 2 and len(stability) == 2
ok &= all(abs(s["effective_moves"][0] - 1) < 1e-9 for s in summaries
          if s["candidate"] == "gumbel_move")
ok &= all(0 <= d[f"target24_ref{r}"]["distance"] <= 1 + 1e-9 for d in rows for r in (16, 32))
# The search's signals (2026-10-30): in range, and their visits' effective moves those the
# summary reports for the visits (the same root, computed apart).
for d in rows:
  s = d["signals"]
  ok &= 0 <= s["top2_gap"] <= 1 and s["unstable"] in (0, 1) and s["surprise"] >= -1e-9
  ok &= 1 - 1e-9 <= s["moves"] <= d["legal"] + 1e-6 and s["teacher_gap"] >= 0
for side in ("dwarfs", "trolls"):
  m = [d["signals"]["moves"] for d in rows if d["side"] == side]
  v = [s for s in summaries if s["side"] == side and s["candidate"] == "visits" and s["ref"] == 16]
  ok &= abs(sum(m) / len(m) - v[0]["effective_moves"][0]) < 2e-3
# mixes= (2026-10-29): at the run's own weights (0.7 for the dwarfs, 0.3 for the trolls) a mix
# is the target exactly; at the other side's weight it is not.
def row(side, c, r):
  return [s for s in summaries if s["side"] == side and s["candidate"] == c and s["ref"] == r][0]
for r in (16, 32):
  ok &= row("dwarfs", "mix0.70", r)["distance"] == row("dwarfs", "target", r)["distance"]
  ok &= row("trolls", "mix0.30", r)["distance"] == row("trolls", "target", r)["distance"]
ok &= row("dwarfs", "mix0.30", 16)["distance"] != row("dwarfs", "target", 16)["distance"]
ok &= all(s["minus_visits"]["distance"][0] == 0 for s in summaries if s["candidate"] == "visits")
ok &= all(s["effective_moves"][0] >= 1 - 1e-9 for s in summaries)
legal = {side: [d["legal"] for d in rows if d["side"] == side] for side in ("dwarfs", "trolls")}
ok &= all(abs(s["effective_moves"][0] - sum(legal[s["side"]]) / len(legal[s["side"]])) < 1e-3
          for s in summaries if s["candidate"] == "uniform")
if not ok: print(lines[:3], summaries[:2])
sys.exit(0 if ok else 1)
EOF
"$bin/az_teacher_vs_search" run="$run" step=-1 refs=16 last=2 max=3 threads=2 batch=2 gumbel=1 \
  considered=1,4 gumbel_scale=0 gumbel_sims=0,12 inference_threads=1 > "$tmp/considered.jsonl" \
  2> "$tmp/considered.err"
check "9. considered=1,4 gumbel_scale=0 gumbel_sims=0,12: exits 0, the candidates named by the count and the budget, each move one-hot; the controls: the counts' measures differ, and the budgets'" \
  python3 - "$tmp/considered.jsonl" $? <<'EOF'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1])]
rows = [d for d in lines if "position" in d]
summaries = [d for d in lines if d.get("summary") and not d.get("stability")]
names = {s["candidate"] for s in summaries}
ok = sys.argv[2] == "0" and len(rows) >= 4 and len(summaries) == 2 * 18
gumbels = [f"gumbel{m}{n}" for m in (1, 4) for n in ("", "_s12")]
ok &= {g + s for g in gumbels for s in ("", "_target", "_move")} <= names
ok &= "gumbel" not in names
for d in rows:
  for g in gumbels:
    x = d[f"{g}_move_ref16"]
    ok &= x["agrees"] in (0, 1) and x["mass_on_best"] == x["agrees"]
ok &= any(d["gumbel1_ref16"] != d["gumbel4_ref16"] for d in rows)
ok &= any(d["gumbel4_ref16"] != d["gumbel4_s12_ref16"] for d in rows)
if not ok: print(sorted(names), rows[:1])
sys.exit(0 if ok else 1)
EOF

# 10. The carry-over and the stop between moves.
carry=$tmp/carry
"$bin/make_test_network" dir="$carry" nn_model=resnet_conv_policy input_planes=1 nn_width=8 \
  nn_depth=1 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network (10)"; exit 1; }
carry_run() {  # carry_run MAX_STEPS LOG: the run, fresh or resumed
  if [ -f "$carry/config.json" ]; then
    python3 - "$carry/config.json" "$1" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); c["max_steps"] = int(sys.argv[2])
json.dump(c, open(sys.argv[1], "w"), indent=2)
EOF
    "$bin/az_trainer" "$carry/config.json" > "$2" 2>&1
  else
    # Short battles (10 turns without a capture end one), so that the actors finish games
    # while the last step learns (300 batches): with the default game a battle outlasted it.
    "$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$carry" \
      --game="thud(max_turns_without_capture=10)" \
      --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=8 --nn_depth=1 \
      --actors=3 --evaluators=0 --inference_batch_size=2 --inference_threads=1 \
      --max_simulations=8 --replay_buffer_size=2048 --replay_buffer_reuse=4 \
      --train_batch_size=64 --learner_batches=300 --checkpoint_freq=1 --max_steps="$1" \
      > "$2" 2>&1
  fi
}
carry_run 1 "$tmp/carry1.log"
check "10. a run of one step with a long learning step exits 0" [ $? -eq 0 ]
note1=$(cat "$carry/carried_games.txt" 2>/dev/null)
check "10. the learner carried the waiting games: the note ($note1) and its log line" \
  python3 - "$carry" "$note1" <<'EOF'
import re, sys
after, states, games = map(int, sys.argv[2].split())
log = open(sys.argv[1] + "/log-learner.txt").read()
ok = after == 1 and games > 0 and states > 0
ok &= f"Carried {games} games ({states} states) into the buffer after step 1" in log
sys.exit(0 if ok else 1)
EOF
check "10. the actors' games in play ended at the stop (logged as abandoned)" \
  bash -c "cat '$carry'/log-actor-*.txt | grep -q 'abandoned at a stop'"
carry_run 2 "$tmp/carry2.log"
check "10. resumed to step 2: exits 0" [ $? -eq 0 ]
check "10. ... step 2 takes the carried games: its record, the buffer, the collection, the count" \
  python3 - "$carry" "$note1" <<'EOF'
import json, re, sys
d = {r["step"]: r for r in map(json.loads, open(sys.argv[1] + "/learner.jsonl"))}
after, states, games = map(int, sys.argv[2].split())
log = open(sys.argv[1] + "/log-learner.txt").read()
collected = [(int(s), int(g)) for s, g in re.findall(r"Collected +(\d+) states from +(\d+) games", log)]
c_states, c_games = collected[1]  # Step 2's, the resumed run's first.
ok = f"The last run carried {games} games ({states} states)" in log
ok &= d[2]["carried_games"] == games and d[2]["carried_states"] == states
ok &= d[1]["carried_games"] == 0 and d[1]["carried_states"] == 0
ok &= d[2]["total_states"] - d[1]["total_states"] == states + c_states
ok &= c_states + states >= 512 and collected[0][0] >= 512
ok &= d[2]["total_trajectories"] - d[1]["total_trajectories"] == games + c_games
if not ok: print(d[1]["total_states"], d[2]["total_states"], states, collected, games)
sys.exit(0 if ok else 1)
EOF
check "10. ... and the note is the new end's (after step 2), if games waited" \
  bash -c "[ ! -f '$carry/carried_games.txt' ] || [ \"\$(cut -d' ' -f1 '$carry/carried_games.txt')\" = 2 ]"
echo "5 100 3" > "$carry/carried_games.txt"  # The control: a stale note.
carry_run 3 "$tmp/carry3.log"
check "10. ... without a teacher, no teacher distance logged (the control for part 1's)" \
  python3 - "$carry/learner.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
sys.exit(0 if all("mean_kl_teacher_prior" not in d["by_side"][s] for d in rows
                  for s in ("dwarfs", "trolls")) and "mean_kl_visits_prior" in rows[0]["by_side"]["dwarfs"] else 1)
EOF
check "10. the control: a stale note is ignored and removed, step 3 collects its 512" \
  python3 - "$carry" <<'EOF'
import json, re, sys
d = {r["step"]: r for r in map(json.loads, open(sys.argv[1] + "/learner.jsonl"))}
log = open(sys.argv[1] + "/log-learner.txt").read()
collected = [int(s) for s in re.findall(r"Collected +(\d+) states", log)]
ok = "is for step 5, not 2: ignored and removed" in log
ok &= d[3]["carried_games"] == 0 and d[3]["carried_states"] == 0 and collected[2] >= 512
sys.exit(0 if ok else 1)
EOF

# 11. The completed-Q policy target.
cq=$tmp/completed_q
"$bin/make_test_network" dir="$cq" nn_model=resnet_conv_policy input_planes=1 nn_width=8 \
  nn_depth=1 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network (11)"; exit 1; }
"$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$cq" \
  --game="thud(max_turns_without_capture=10)" \
  --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=8 --nn_depth=1 \
  --actors=2 --evaluators=0 --inference_batch_size=2 --inference_threads=1 \
  --max_simulations=8 --replay_buffer_size=2048 --replay_buffer_reuse=4 \
  --train_batch_size=64 --learner_batches=2 --checkpoint_freq=1 --max_steps=1 \
  --policy_target=completed_q > "$tmp/completed_q.log" 2>&1
check "11. a run with --policy_target=completed_q exits 0" [ $? -eq 0 ]
check "11. ... finite losses, config.json records it, the actors log it" python3 - "$cq" <<'EOF'
import glob, json, math, sys
d = [json.loads(l) for l in open(sys.argv[1] + "/learner.jsonl")]
ok = len(d) == 1 and all(math.isfinite(d[0]["loss"][k]) for k in ("policy", "value"))
ok &= json.load(open(sys.argv[1] + "/config.json"))["policy_target"] == "completed_q"
ok &= all("Policy target: completed_q" in open(f).read() for f in glob.glob(sys.argv[1] + "/log-actor-*.txt"))
sys.exit(0 if ok else 1)
EOF
check "11. control: part 1's run (visits, the default) logs no policy target" \
  bash -c "! grep -q 'Policy target' '$run'/log-actor-*.txt"

# 12. The Gumbel search (2026-10-29, --gumbel_search; its rules and searches: gumbel_search_test).
gs=$tmp/gumbel
"$bin/make_test_network" dir="$gs" nn_model=resnet_conv_policy input_planes=1 nn_width=8 \
  nn_depth=1 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network (12)"; exit 1; }
"$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$gs" \
  --game="thud(max_turns_without_capture=10)" \
  --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=8 --nn_depth=1 \
  --actors=2 --evaluators=1 --inference_batch_size=2 --inference_threads=1 \
  --max_simulations=16 --replay_buffer_size=2048 --replay_buffer_reuse=4 \
  --train_batch_size=64 --learner_batches=2 --checkpoint_freq=1 --max_steps=1 \
  --gumbel_search --gumbel_considered=4 > "$tmp/gumbel.log" 2>&1
check "12. a run with --gumbel_search exits 0" [ $? -eq 0 ]
check "12. ... finite losses, config.json records it, the actors log it" python3 - "$gs" <<'EOF'
import glob, json, math, sys
d = [json.loads(l) for l in open(sys.argv[1] + "/learner.jsonl")]
ok = len(d) == 1 and all(math.isfinite(d[0]["loss"][k]) for k in ("policy", "value"))
c = json.load(open(sys.argv[1] + "/config.json"))
ok &= c["gumbel_search"] is True and c["gumbel_considered"] == 4
ok &= all("Gumbel search: 4 considered actions" in open(f).read()
          for f in glob.glob(sys.argv[1] + "/log-actor-*.txt"))
sys.exit(0 if ok else 1)
EOF
check "12. control: part 1's run (PUCT, the default) logs no Gumbel search" \
  bash -c "! grep -q 'Gumbel search' '$run'/log-actor-*.txt"

# 13. An interrupt while the learner collects.
sig=$tmp/sigint
"$bin/make_test_network" dir="$sig" nn_model=resnet_conv_policy input_planes=1 nn_width=8 \
  nn_depth=1 checkpoints=1 > /dev/null 2>&1 || { echo "FAIL make_test_network (13)"; exit 1; }
"$bin/az_trainer" --flagfile="$repo/thud/experiments/az_thud.flags" --path="$sig" \
  --graph_def=vpnet.pb --nn_model=resnet_conv_policy --nn_width=8 --nn_depth=1 \
  --actors=3 --evaluators=0 --inference_batch_size=2 --inference_threads=1 \
  --max_simulations=8 --replay_buffer_size=65536 --replay_buffer_reuse=4 \
  --train_batch_size=64 --checkpoint_freq=1 --max_steps=5 > "$tmp/sigint.log" 2>&1 &
sig_pid=$!
for _ in $(seq 120); do  # Until an actor has played a game (at most 60 s).
  grep -qs 'Actions: ' "$sig"/log-actor-*.txt && break
  sleep 0.5
done
kill -INT "$sig_pid"
sig_ended=0
for _ in $(seq 120); do
  kill -0 "$sig_pid" 2> /dev/null || { sig_ended=1; break; }
  sleep 0.5
done
if [ "$sig_ended" = 1 ]; then
  wait "$sig_pid"
  sig_status=$?
else
  kill -9 "$sig_pid" 2> /dev/null
  wait "$sig_pid" 2> /dev/null
  sig_status=hung
fi
check "13. interrupted while collecting, it exits by itself within 60 s, status 0 (it: $sig_status)" \
  bash -c "[ '$sig_status' = 0 ] && grep -q 'Exiting cleanly' '$tmp/sigint.log'"

finish
