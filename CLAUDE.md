# CLAUDE.md — Thud on OpenSpiel

This is a **fork of `google-deepmind/open_spiel`**. Our purpose here is to implement the
board game **Thud** as a C++ game inside OpenSpiel, and then build an AI for it.

Our work lives in exactly two places:

- `open_spiel/games/thud/` — the game implementation (`thud.h`, `thud.cc`, `thud_test.cc`)
- `thud/` — our roadmap, session log, rules spec, and experiment scripts

Everything else in this tree is upstream OpenSpiel code.

**Between us** (agreed 2026-10-01): in interactive sessions the user and Claude call each
other "bro", and Claude sprinkles it into many sentences, at the start or the end, to
build a connection — the user asked for it more often than Claude first did: several
times in a reply, in short replies too, though not in every sentence. In the docs
(`thud/`, commit messages) the user stays "the user" or "they".

**Important information goes at the end of a reply, never in its middle** (the user,
2026-10-15): results, decisions needed, questions and anything the user must act on
belong in the final message of a turn, after the work, not in the text between tool
calls, where long output buries them.

**Answer as long as needed, no longer** (the user, 2026-10-16, to save tokens): a
confirmation can be just a confirmation, and a simple question gets a simple answer;
detail where the question or the result calls for it.

---

## Session start

Before doing anything else, read:

1. `thud/PROGRESS.md` — the `## Current status` block at the top says where we actually are
   and what the next step is.
2. `thud/PLAN.md` — the roadmap and which phase we are in.
3. `thud/THUD_RULES.md` — the authoritative ruleset and action encoding. **Implement what
   this file says**, not what you remember about Thud. The published rules are ambiguous in
   several places; its section 9 records which reading we chose and why. All of those
   points were settled with the user on 2026-09-22 — if one ever needs changing, raise it
   with the user rather than deviating in code.
4. **Look for experiments still running** from an earlier session. Every long run is a
   script in `~/thud-runs/` or a program in `build-shared/`, so this lists them all:

   ```bash
   ps -eo pid,etime,args | grep -E '[t]hud-runs/|[b]uild-shared/' | grep -v shell-snapshots
   ```

   (the brackets stop `grep` from matching itself, `shell-snapshots` drops Claude Code's
   own shell; checked 2026-09-30 with a dummy process). Then read the end of
   `~/thud-runs/stage1_matches/progress.log`, and compare it with the status block. A run
   started detached (Environment, below) outlives sessions; one that was not may have
   died with its session — check its files (the trainer's `learner.jsonl`, checkpoints,
   `command.txt`) before restarting anything, and never start a second copy of a run
   that is still alive.

## Session end

This project runs across many sessions, so the log is what keeps it coherent. At the end of
each working session, without being asked:

1. Append a dated entry to `thud/PROGRESS.md` under `## Session log` covering: what changed,
   what decisions were made (and why), what is half-finished, and what the next step is.
2. Overwrite the `## Current status` block at the top of `thud/PROGRESS.md` so it reflects
   reality.
3. If the roadmap itself changed — scope, phase order, a deferred decision now resolved —
   update `thud/PLAN.md` too. Do not record roadmap changes only in the session log.
4. **Check the docs for consistency** — also before every commit that touches them (user,
   2026-09-27: "I always want consistency checks"): statements superseded by newer
   results, stale status lines, next steps already done, changed cross-references (stage
   numbers, file names), numbers that differ between places. Fix superseded statements in
   place in `thud/PLAN.md` and the status block; the session log stays chronological and
   is corrected by later entries.

Prefer concrete detail over summary. "Implemented hurl generation; blocked on whether a hurl
onto an empty square is legal" is useful. "Worked on move generation" is not.

---

## Environment

These facts were established by measurement and cost real time to work out. Do not
re-derive them — on this machine. On any other (a cloud GPU), the machine-dependent
settings below and in the plan (buffer, network and batch sizes, threads, cadence) are
reasoned about or benchmarked again first (user, 2026-10-03; `thud/PLAN.md`, *Cloud GPU
options*).

- **Development happens in WSL2 / Ubuntu on ARM64.** The host is a Snapdragon X, 10 cores,
  ~31.6 GB RAM — but **WSL itself sees 15 GB RAM + 4 GB swap** (its default is half the
  host). Since 2026-10-07 `C:\Users\waech\.wslconfig` sets `swap=8GB` (a safety net,
  not working memory: a job that needs swap crawls; in effect from the next `wsl
  --shutdown`); `memory=20GB` is an option for later (the user: Windows would keep ~11.6
  GB at worst). That is ample for the build: full `make -j10` takes ~4m40s with a 1.2 GB peak per
  compiler process; full `ctest -j10` takes ~90 s.
- **Do not develop on the Windows side.** The Windows Python is an x64 build running under
  emulation (its pip tags are `win_amd64`), and no `win_arm64` OpenSpiel wheels exist.
- **Keep the repo on the WSL native filesystem** (`~/thud-openspiel`), never under
  `/mnt/c/...`. Cross-filesystem I/O is drastically slower and makes every build painful.
- **There is no CUDA GPU.** The integrated GPU is a Qualcomm Adreno X1-85. Any serious
  neural-network training will need either cloud compute or a reduced scale.
- **LibTorch for aarch64 comes from PyTorch's pip wheel.** PyTorch publishes no standalone
  LibTorch download for aarch64 Linux, but the wheel `torch==2.10.0` (cp314,
  `manylinux_2_28_aarch64`) ships LibTorch complete — libraries, C++ headers and
  `TorchConfig.cmake` — built with the C++11 ABI, like our clang build (checked
  2026-09-25). OpenSpiel's C++ AlphaZero (`OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON`) builds
  that way, unpatched, in its own folder `build-torch/` (below), and passes its tests and
  the tic-tac-toe control: `thud/PLAN.md` Phase 6.
- **Set `OMP_NUM_THREADS` for LibTorch programs to match the work split.** LibTorch
  gives every caller 10 OpenMP threads, so several callers oversubscribe the 10 cores:
  three searches at once ran at 16 simulations/s each by default, 128 each with one
  thread (2026-09-25). Without batched inference every AlphaZero actor calls the network
  itself: use `OMP_NUM_THREADS=1`. With batched inference, much faster for Thud's small
  networks, only the inference threads call it: 2 inference threads with
  `OMP_NUM_THREADS=4` was best for every network size tried, and matched in the trainer
  (`thud/PLAN.md` Phase 6, throughput final report, 2026-09-26). For matches
  (`az_match`, one evaluator per network; games are identical whatever these settings):
  a match of 64+ pairs alone with `threads=64 batch=32` and `OMP_NUM_THREADS=4` (1,923
  simulations/s for 64 x 4); smaller matches up to 3 at once with `OMP_NUM_THREADS=1`
  and threads = pairs (1,720) — against ~1,050 for one match at a time with its default
  16 threads, whose last long pairs leave most threads idle (2026-09-29). With the
  50-plane network (M102) one 40-pair match alone ran ~680 simulations/s, three at once
  ~1,500 together, six at once no more (~1,400-1,700; 2026-10-13); the teacher mixed into
  a prior (`teacher_mix_a`) costs ~1.5x.
- **Jobs beside a training run: the idle scheduling class, and mind the memory.**
  `nice -n 19` competes as an equal with our trainers, which run at nice 19 themselves;
  `chrt --idle 0 CMD` (SCHED_IDLE, allowed unprivileged here) weighs 3 against a nice-19
  task's 15 (the kernel's `WEIGHT_IDLEPRIO` and `sched_prio_to_weight`), so it still gets
  ~1/6 of a contended core — beside run G''s trainer two idle-class matches cut the
  trainer's CPU from ~7.4-8.0 to ~6.6-7.0 cores. **And even idle-class matches cost the
  trainer ~25-35% of its self-play speed** (the load test, 2026-10-05, 5 pairs of
  20-minute blocks; `thud/PLAN.md`, *Instrumentation*): mostly efficiency — shared caches
  and memory bandwidth — not CPU share, so no priority setting fixes it. Run evaluations
  between training runs, not beside them. Memory, measured
  2026-10-05 at the plateau, not at the start: a 7x trainer ~9.9 GB of WSL's 15 (7.4 GB
  in its first hour, its cache still filling); an `az_match` 1.4 GB with its default
  cache of 262,144 evaluations, 0.76 GB with `cache=32768` (20 threads; the cache
  changes no game). The out-of-memory killer would pick the largest process, the
  trainer — check `free -m` (available) first, and again once the job's caches are full.
  Searches grow with their simulations too: `az_target_quality` with 32 threads and a
  16,000-simulation reference held 8.0 GB, and a 40-pair `az_match` (40 threads, the
  default cache) up to 3.6 GB — together with a second match they filled RAM and swap on
  2026-10-07 (the filter was stopped). Run such jobs alone, or with fewer threads.
  OpenSpiel's MCTS with random rollouts (`az_match b=mcts:SIMS`) holds up to 1.03 GB of
  tree per thread at 100,000 simulations (the opening's dwarf move, at its 1,000 MB cap;
  230 MB a few moves in) but only ~12 MB at 10,000 (peak RSS, the opening; measured
  2026-10-07) — deeper searches expand the dwarf nodes, each with ~650 children.
- **Sleep freezes WSL**: nothing runs, WSL's clocks stop, and the wall clock jumps on
  resume. The laptop used to sleep after 5 minutes without input even on mains power; on
  2026-09-26 the user set sleep to never while plugged in (on battery it still sleeps
  after 3 minutes). Before long runs, check from WSL with `powercfg.exe /query
  SCHEME_CURRENT SUB_SLEEP` ("Sleep after", AC index 0 = never) and keep it plugged in
  with the lid open — even with sleep off it paused ~18 minutes on 2026-09-26, most likely
  from the lid (its setting cannot be read with `powercfg` here). The setting is Settings >
  System > Power & battery, or `powercfg /change standby-timeout-ac 0`. Long runs keep a
  pause detector (wall clock against `/proc/uptime` every 10 s) to prove they ran
  unpaused. Time measurements with a
  monotonic clock survive a pause; wall-clock times (the AlphaZero trainer's logs, `date`
  differences) do not. A pause shows as `/proc/uptime` falling behind the wall clock.
  **A sleep can also end WSL altogether**: on 2026-10-01 the laptop slept on battery from
  ~15:12 to 16:43 (Windows logged nothing in between, then "Wake from sleep detected"),
  and WSL restarted on waking — every process ended, detached ones too (2026-09-30 it had
  survived a 3-hour battery freeze). So check with `powershell.exe -NoProfile -Command
  "(Get-CimInstance Win32_Battery).BatteryStatus"` (2 = mains, 1 = battery) before and
  during long runs; `uptime -s` shows when WSL last started.
- **Background jobs die with the Claude Code session that started them.** On 2026-09-30
  run D's whole process tree stopped at 10:54, minutes after a new session replaced the
  old one following a battery freeze (no crash, no out-of-memory kill; the harness marked
  the job "[killed]"). Start any long run detached, in its own session: `setsid nohup
  SCRIPT > LOG 2>&1 < /dev/null &`; watch it through its files (e.g.
  `~/thud-runs/stage1_matches/progress.log`) or a separate waiter whose death is harmless.
  Keep long runs as scripts in `~/thud-runs/` calling programs in `build-shared/`, so
  that the session-start listing (above) finds them. A waiter must not match itself:
  `pgrep -f NAME` also matches a shell whose command line contains NAME — write the
  pattern with brackets (`pgrep -f 'step2_buffer4x_resum[e].sh'`). Brackets do not
  stop it matching *another* shell that merely mentions the name, and the Claude Code
  shell that wrote or launched a script holds the whole text in its command line and can
  outlive the launch: on 2026-10-02 `head_check_augment.sh` waited on `head_check.sh`,
  matched the shell that had created it 3.5 hours earlier, and stalled the queue until
  that shell was ended. Anchor waits to the script's own command line instead: `pgrep -f
  '^/bin/bash [^ ]*/head_check[.]sh'` (checked: a dummy shell mentioning the name, and
  Claude Code's own, do not match; the running script does). **Never `pkill -f` a
  pattern — kill by PID.** The same matching kills more than meant: on 2026-10-07
  `pkill -f 'C16_vs_hexparrot3_games.jsonl'`, aimed at a test's stand-in, also killed
  the real match with that file in its command line (at 9 of 20 pairs) and the Claude
  Code shell running the command (exit 144, which also left the command's later steps
  undone).
- **The venv has OpenSpiel's pinned JAX set** (`jax==0.9.0.1`, `flax==0.12.3`, ... from
  `open_spiel/scripts/python_extra_deps.sh`), which pins numpy to 2.3.5: flax 0.12.3
  requires numpy below 2.4. The JAX CPU build warns "An NVIDIA GPU may be present"; there
  is none, and it falls back to the CPU as it should.
- OpenSpiel does publish `manylinux_2_28_aarch64` wheels (cp311–cp314). If the C++ build
  breaks, `pip install open_spiel` is a usable fallback to keep Python-side work moving.

## Shell environment — one-time setup, then nothing per session

**Claude Code's Bash tool does not persist shell state between commands.** The working
directory carries over, but exported variables and `source venv/bin/activate` do not. So an
activated venv cannot be relied on across tool calls.

Put the environment in `~/.bashrc` instead, where every new shell inherits it:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/thud-openspiel
export PYTHONPATH=$PYTHONPATH:$HOME/thud-openspiel/build/python
source $HOME/thud-openspiel/venv/bin/activate
```

Do this once. Afterwards there is nothing to do at the start of a session. If a build command
fails with a missing module or a missing `pyspiel`, check `~/.bashrc` first rather than
prefixing activation onto the command.

## Build and test

From the repo root:

```bash
# one-time setup
./install.sh
python3 -m pip install --upgrade pip setuptools
python3 -m pip install -r requirements.txt

# build
mkdir -p build && cd build
CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
make -j10

# test
ctest -j10                 # everything
ctest -R thud              # just ours
./examples/example --game=thud
```

`build/` is OpenSpiel's default build type, "Testing" (`-O2`, all runtime assertions on). **For
speed measurements use a Release build** (`-O3`, `SPIEL_DCHECK` off) in its own folder, which
git ignores:

```bash
mkdir -p build-release && cd build-release
BUILD_TYPE=Release CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) \
  -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
make -j10 benchmark_game mcts_example   # ~2.5 min; add targets as needed
```

OpenSpiel's C++ AlphaZero needs LibTorch (from the pip wheel, above) and libnop
(`open_spiel/libnop/libnop/`, cloned by `install.sh`; git ignores it). It has its own
Release folder, `build-torch/`, which builds only the C++ targets: `pyspiel` keeps
coming from `build/`, and LibTorch's bundled pybind11 was reported to clash with ours in
a `pyspiel` build (OpenSpiel issue #966; not tried with 2.10):

```bash
mkdir -p build-torch && cd build-torch
BUILD_TYPE=Release OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON OPEN_SPIEL_BUILD_WITH_LIBNOP=ON \
  CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') \
  ../open_spiel
make -j10 torch_integration_test torch_model_test torch_vpnet_test \
  alpha_zero_torch_example alpha_zero_torch_game_example   # ~3.5 min
```

cmake warns "static library kineto_LIBRARY-NOTFOUND not found" (a profiling library
the wheel does not ship); the build does not need it. Our AlphaZero defaults for Thud
are in `thud/experiments/az_thud.flags` (`--flagfile`).

Our own C++ programs that use the AlphaZero model link against OpenSpiel built as a
shared library, as `docs/library.md` describes, in `build-shared/`, so no upstream CMake
file changes: `build-torch/`'s configure plus `BUILD_SHARED_LIB=ON`, then `make -j10
open_spiel` (~2.5 min). The commands, and the compiler flags copied from CMake, are in
`thud/experiments/build_az_program.sh`, which builds the programs on upstream's
AlphaZero (`az_layout_probe`, `az_throughput`, `az_reuse`, `az_buffer_stats`). Programs on
**our copy** (`thud/az/`: the trainer `az_trainer` — upstream's example plus
`--untried_move_value`, `--untried_move_reduction`, `--learner_batches`,
`--nn_model=resnet_conv_policy`, `--symmetry_augmentation`, `--policy_target_pruning`
and `--replay_buffer_start_size` (2026-10-04), `--player_policy_temperature` and
`--player_simulations` (2026-10-06), `--alphabeta_share` and `--alphabeta_depth`
(2026-10-07; the depth per side since 2026-10-08: a share of the positions from games against OpenSpiel's alpha-beta,
`thud/az/alphabeta_player.h`, only the network's positions recorded, reported apart
as `vs_alphabeta`), `--kickstart_weight`, `--kickstart_depth` and
`--kickstart_temperature` (2026-10-08: each recorded position's policy target mixed with
a softmax of alpha-beta's move scores, a teacher; actors log the settings),
`--player_kickstart_weight` and `--player_policy_alpha` / `--player_policy_epsilon`
(2026-10-13: the teacher's weight and the root noise per player to move; empty: as
before; built as `az_trainer_d1`), `--lr_warmup_steps`, `--lr_warmup_start`,
`--lr_schedule=STEP:FACTOR,...` and `--value_loss_weight` (2026-10-15: a learning-rate
schedule on the network's own rate, logged per step, and the value loss's weight; off by
default), `--playout_cap_full_share` and `--playout_cap_fast_simulations` (2026-10-15:
KataGo's playout cap randomisation; off by default); since 2026-10-15 crash-safe saving (the buffer and the checkpoints written under
a temporary name and renamed, a truncated last record skipped on resume, a cut-off buffer
refused) and **the legal policy by default** (`VPNetModel::SetLegalPolicy`: the priors
from the legal moves' logits alone, equal to upstream's mask-and-softmax over all 19,800
actions to ~1e-7; `SetLegalPolicy(false)` is upstream's, bit for bit); since 2026-10-05 a replay buffer that moves
positions into their slots and compacts itself after loading (`thud/az/replay_buffer.h`)
and actor and evaluator logs that append; and since 2026-10-03
logging per-side losses, self-play margins and endings and search statistics —
`identity_test` (built with upstream's `model`, `vpnet` and `vpevaluator` sources as
extra arguments, as its header says), `untried_move_test`, `conv_policy_test`,
`augmentation_test`, `instrumentation_test`, `pruning_test`, `growing_buffer_test`,
`search_settings_test`, `alphabeta_training_test`, `kickstart_test`, `input_planes_test`
(the 50 input planes inside the network; `features_test` tests the planes themselves),
`grow_test` (Net2Net, on small networks), `make_test_network` (small untrained networks for the tests), and from
`thud/experiments/`
`az_match` (since 2026-10-07 also against players without a network:
`b=hexparrot:N`, through `hexparrot_player.py`, `b=alphabeta:N`, `b=mcts:N`; since
2026-10-13 self-play's root noise per side and sampled plies, `noise=`, `sample_plies=`,
and the teacher in one network's prior, `teacher_mix_a=`), `az_forgetting`, `az_merge_buffers`, `az_target_quality`, `az_head_probe`,
`az_reference_pilot`, `az_noise_probe`, `az_search_cost`, `az_blunder_probe`,
`az_capture_targets`, `az_teacher_stats`, `az_feature_probe`, `az_long_capture_probe`,
`az_input_plane_trace`, `az_risk_probe`, `az_dwarf_value_probe` and `teacher_mix_test`;
`az_restructure`, which builds a network of a new structure from a trained one and
its replay buffer, `az_grow`, which grows a network wider or deeper computing the same
(Net2Net, 2026-10-15), and `az_fresh_optimizer`, a network's weights with a fresh optimizer;
`az_match devices=`, `az_restructure device=` and `az_grow device=` choose where the
networks run)
are built by
**`thud/build_programs.sh`** (2026-10-17: `thud/programs.mk`, incremental, a second when
nothing changed; it refuses while one of them runs, since a run's next segment would start
the new binary): **every queue in `~/thud-runs/` runs it before it starts**, and
`run_tests.sh --full` (so the pre-push hook) does too. One-offs can still use
`thud/az/build.sh MAIN.cc`, which also links Abseil's static libraries (the flag parsing
`libopen_spiel.so` does not re-export) and runs at `nice 19`. The tests build in
`build-tests/` (below).

**Now that JAX and PyTorch are in the venv, the next cmake run in `build/` adds their
Python tests** (OpenSpiel detects both when `OPEN_SPIEL_ENABLE_JAX`/`_PYTORCH` are
unset, `open_spiel/python/CMakeLists.txt:172-190`; `build-torch/`'s configure did), so
`ctest` will then run more than the 285 tests recorded so far.

Playthrough regression baseline (our most valuable correctness tool — it catches accidental
rule changes): `open_spiel/integration_tests/playthroughs/thud.txt`, one random game
recorded by `./open_spiel/scripts/generate_new_playthrough.sh thud` (once, 2026-09-25).
The test replays **the moves recorded in it** and compares every line, as upstream's
`playthrough_test.py` does (`docs/developer_guide.md`, step 11) — run it after any change
to move generation, the action encoding, the text formats or the observation:

```bash
thud/run_tests.sh playthrough       # ours alone; ctest's playthrough_test does every game's
```

Running `generate_new_playthrough.sh` again is not the test: it plays new random moves
(its seed is drawn afresh), so its diff is never empty (found 2026-10-15; this file said
otherwise until then). A difference in the replay means behaviour changed; if the change
was intended, update the baseline from its own moves and read the diff:

```bash
python3 open_spiel/python/examples/playthrough.py \
  --update_path open_spiel/integration_tests/playthroughs/thud.txt && git diff --stat
```

(Upstream's `regenerate_playthroughs.sh` would rewrite every game's baseline: never run it
here.)

**Our tests, in one command** (2026-10-15): `thud/run_tests.sh` builds what is out of
date — the game in `build/` and `build-shared/`, the tests and the programs the
end-to-end tests run in `build-tests/` (`thud/tests.mk`: incremental, **without
`-DNDEBUG`**, so these differ from `build.sh`'s binaries) — and runs the quick tests
(~20 s), or with `--full` every test (~4 min: `thud_test`, the hexparrot cross-check,
the playthrough diff — against a temporary file, never over the baseline — the
sanitized tests, every test program, `thud/az/trainer_test.sh` and
`thud/experiments/az_match_test.sh`); `--list` names them, `NAME ...` runs only those;
each test's output is in `build-tests/last/NAME.log`. None needs a run folder. **The
git hooks run them**: `.githooks/pre-commit` the quick tests when a commit changes code
(docs alone go through at once), `.githooks/pre-push` every test when the pushed commits
change code, after its upstream guard (`core.hooksPath` is set in this clone). What each
test covers, and what no test covers yet: `thud/PLAN.md`, *Tests*.

**For another machine** (2026-10-15; `thud/PLAN.md`, the GPU readiness checklist):
`thud/setup_machine.sh` sets a fresh clone up — packages, the venv with torch, the
builds, our programs, hexparrot, every test, and with a GPU a CUDA smoke test (rerun it
after a failure: done steps are skipped); `thud/transfer_run.sh FROM TO [--state-only]`
moves a run folder (rsync, checksums at both ends, the copy's `config.json` path set to
its folder); `thud/az/watchdog.sh RUN_DIR -- TRAINER ARGS` keeps a run going through
crashes and kills. `az_search_cost parts=2 net=RUN:STEP` with lists of
`batched_threads`, `batch` and `inference_threads` is the throughput sweep, and
`thud/experiments/regression_gates.py` the regression gates (from a candidate's and the
anchor's matches and probes).

---

## Conventions

**Stay out of upstream code.** Do not edit anything under `open_spiel/` except
`open_spiel/games/thud/`, these two registration points:

- `open_spiel/games/CMakeLists.txt` — add the Thud sources and test target
- `open_spiel/python/tests/pyspiel_test.py` — add the `thud` short name

and the generated playthrough baseline `open_spiel/integration_tests/playthroughs/thud.txt`
(a new file, written only by OpenSpiel's playthrough tools — `generate_new_playthrough.sh`,
`playthrough.py --update_path` on this one file — never by hand).

Keeping the diff that small is what lets us rebase onto upstream without pain.

**Branches.** Work on `thud`. Leave `master` tracking upstream so rebasing stays easy.

**Style.** Follow the Google C++ style guide and match the surrounding OpenSpiel code. Mirror
`open_spiel/games/tic_tac_toe/` for the surrounding boilerplate and `open_spiel/games/chess/`
for the from-square × direction × distance action layout. The implementation must be very
fast yet readable — see `thud/PLAN.md` Phase 3.

**Suppress diagnostics narrowly** (user, 2026-10-05): never a blanket switch such as
`-Wno-everything` for code we build. Suppress one named diagnostic, at the one place
where we are certain it is safe — `#pragma clang diagnostic push` / `ignored "-W..."` /
`pop` around the line or the `#include` that triggers it — with a comment saying why it
is safe (example: `thud/az/replay_buffer_test.cc`, libnop's
`-Wmissing-template-arg-list-after-template-kw`). Upstream's CMake passes
`-Wno-everything` (`open_spiel/CMakeLists.txt`), and our `thud/az/build.sh` and
`thud/experiments/build_az_program.sh` copied it: to be narrowed the same way, after
seeing what fires without it. More generally, **keep the safety flags** (user,
2026-10-05: "I would like to keep the code safe"): never weaken a warning, assertion or sanitizer flag to make something build; tests build without `-DNDEBUG` (so
`assert` and `SPIEL_DCHECK` run), and changes to memory handling in our copies (moves,
pointers, lifetimes) are also run under `-fsanitize=address,undefined`. Such a build
needs `-DNDEBUG_SANITIZER` (2026-10-05): with AddressSanitizer on, abseil adds fields to
its hash tables unless that is defined, so sanitized code and the uninstrumented abseil
in `libopen_spiel.so` disagree on their layout and the sanitizers report false errors
(a null load in an `LRUCache`, a new-delete size mismatch in the flag registry).

**Decide from evidence, not preference.** When choosing between approaches — a layout, a
convention, an encoding — check what the upstream project actually documents and what other
implementations actually do, and cite it. "The developer guide documents only this workflow"
settles a question; an unsourced recommendation invites a round trip. This project has
already been bitten the other way: the rules were pinned down by reading three existing Thud
implementations, which overturned two conclusions reached by reasoning alone.

## Licensing — these are obligations, not preferences

OpenSpiel is Apache License 2.0, and we keep Apache 2.0 for our own files too.

- **Every new file** we add carries an Apache 2.0 header with our own copyright line.
- **Every upstream file we modify** must carry a prominent notice stating that we changed it.
  Apache 2.0 section 4(b) requires this. Currently that means `open_spiel/games/CMakeLists.txt`
  and `open_spiel/python/tests/pyspiel_test.py` — and our **copies** of upstream files in
  `thud/az/` (OpenSpiel's C++ AlphaZero and MCTS, our namespace `open_spiel::thud_az`), which
  keep their original headers and get a change notice from `thud/az/import_from_upstream.py`.
  Rerun that script only to re-import from upstream: it overwrites the copies, and with them
  any change we made later.
- **Never** remove or alter existing copyright notices, and never touch `LICENSE`.

Note that Thud itself is a commercially published game. Upstreaming our implementation to
OpenSpiel may not be possible — their contributing guide flags that copyrighted games can
need legal approval. Assume this fork is the final home unless that question gets resolved.

**`thud/THUD_RULES.md` quotes the official rules verbatim and must never be upstreamed.**
Game mechanics are not copyrightable but their expression is, so anything that could go
upstream — header comments, a `docs/games.md` entry — must be written from scratch in our own
words. `thud/PLAN.md` has the full pre-PR checklist.

Never upstream `CLAUDE.md` or anything under `thud/`. A `.githooks/pre-push` hook enforces
this for pushes aimed at `deepmind/open_spiel`, but it only runs if this clone has been set
up with `git config core.hooksPath .githooks`, and `--no-verify` bypasses it. **Do not reach
for `--no-verify` if that hook fires** — it is telling you the branch is wrong, not that the
hook is.
