# Progress

## Current status

**Phase:** 0–5 **done** — Thud is implemented, passes all its tests, has its integration
baselines, and is benchmarked (session 5, 2026-09-25); **Phase 6 in progress**:
AlphaZero-style learning with **OpenSpiel's C++ AlphaZero**, decided with the user
2026-09-25 (`PLAN.md` Phase 6). **It builds and runs here**: LibTorch 2.10 from PyTorch's
aarch64 pip wheel, in `build-torch/`, no change to upstream code; OpenSpiel's LibTorch
tests pass, it learns tic-tac-toe like the Python control, a Thud smoke test runs end
to end, and on Thud's board it learns like the Python model with the layout fixed (the
C++ against Python layout control). Run it with batched inference (32 actors, batches of
32, 2 inference threads, `OMP_NUM_THREADS=4`: 2.7-7x faster self-play for the small
networks) and our defaults in `thud/experiments/az_thud.flags` (resignation off, root
noise α 0.1). OpenSpiel's Python AlphaZero also runs here but is not our route (too slow,
and a layout bug). **Stage 1 of the Phase 6 roadmap is done** (2026-09-27): learning works
on unmodified OpenSpiel — run A's network (64 x 4, 100 simulations) rose steadily against
its untrained start to step 14 — then the dwarfs' play collapsed in its last two steps;
100 simulations beat 400 at equal machine time; the dwarfs' searches are pure breadth
(untried moves count as even games). **Stage 2 is done** — our own copy of the C++
AlphaZero in `thud/az/`, proven identical to upstream — and so is **stage 3a**: valuing
untried moves at their visited siblings' mean minus a reduction (now our default) kept the
dwarfs' searches narrow, stopped run A's collapse and made the dwarf play far stronger
(run C, 2026-09-28) — through step 16; continued to step 29 and evaluated as trained, no
collapse but a drift: it beats step 16, loses to A step 14, and its dwarfs' move
probabilities narrowed (forgetting, by the quick check; Step 2 tests a longer memory).
**Step 2 is done** (2026-09-30): run D, run C branched at step 15 with a 4x memory, keeps
its dwarfs' move probabilities from narrowing and beats C29 (+7.3) and C16 (+8.4), but
still loses to A14 (−5.4) and its dwarfs still slip against the anchor — forgetting
explains part of the decline. **Run E is done** (2026-10-01, from 13:17, detached; stopped when the laptop slept on
battery at ~15:12 and WSL restarted on waking; resumed from step 32 at 17:54, finished at
step 44 at 21:59):
run D continued from step 29 with a 7x memory (458,752 positions) and no playout caps,
6 hours of machine time — the control arm for stage 3b's playout caps (run F later, from
the same state, same machine time). Playout caps would cut the recorded positions per
hour ~7x (F: ~2 learning steps in 6 hours against E's ~14), so first **the cheap hint**:
`az_target_quality` (how much closer 400-simulation targets come to a 2,000-simulation
reference than 100-simulation ones), done 22:22: **moderately better** — the dwarfs'
top move agrees with the reference 57.4% instead of 51.8% (+5.6 points), the distance
shrinks ~12% (trolls ~19%); the control (1,000 simulations) closer still. Not worth 7x
less data here: the recommendation (for the user to decide) is no run F on this CPU,
playout caps on the GPU (`PLAN.md`). The 2,000-simulation reference is stable enough
(2026-10-02, the user asked: against 4,000 the gain from 100 to 400 is the same). **Built 2026-10-02, 00:45-03:00** (each behind a
switch, off by default, tested with controls; committed 2026-10-02 in three commits): the
**convolutional policy head** (stage 5, `--nn_model=resnet_conv_policy`, 420,988 weights
at 64 x 4 against 9,244,626), **symmetry augmentation** (5b, `--symmetry_augmentation`;
the board has **8** symmetries, not 16) and the trainer's **overfitting check** (new
against trained positions' loss before each learning step). **Run E's evaluation**
(00:45-03:27, `PLAN.md`, *First training runs*): **E44 beats D29 by +9.7** (33 of 40
pairs) and the anchor by +37.7, the best yet; its dwarfs recovered against the anchor
(+8.7, D29 +3.2, p = 0.03); no forgetting of the dwarfs' moves; **it still loses to A14,
−3.5** (−5.7 to −1.3; D29 −5.4), what is left of A14's edge being on the dwarf side.
**The head check** (04:02-05:16, `PLAN.md`, *Changes to the search and trainer*): the
**new head learns the dwarfs' hurls on held-out positions — 98.8-99.4% of their policy
mass against the linear head's 75-78%** (which reproduces the layout control exactly) —
without overfitting (the linear head: 94-96% on its training positions), and is 3x
closer to symmetric; ~15% slower per full batch of 32, faster per small one, 12% slower
over run C'. **Symmetry augmentation** (05:49-06:45, the 2 x 2 at 64
x 4): a strong regulariser for the linear head (held-out hurl mass 75.0% → 86.8%, its
overfitting gone), little for the new head (98.8% → 99.1%). **Run C'** (`stage5_conv.sh`,
06:50-13:38 after the full test suite passed, 285 of 285; run C's settings with the new
head, from scratch to step 16): no overfitting, but **its dwarfs are much weaker than
C's at equal steps** — against the anchor −10.3 at step 16 (C16 +9.5, p < 0.001), the
trolls equal; in its self-play the trolls won every game; **C'16 loses to C16
head-to-head, −8.5 a pair** (−12.6 to −4.3, 30 of 40 pairs). Its trolls learnt faster
(+27.8 against the anchor at step 8, C +15.8), its dwarfs much worse. So the harness's
win did not carry over to self-play, in one run each: stage 5 fails by the roadmap's
rule, and the head is not adopted for now (the user: revisit it later). Nothing is
running. **Decided (user, 2026-10-02): the next run is a fresh 7x run from step 1 with
the old head**, probably with symmetry augmentation — open: whether to add augmentation
with a control run without it (one change at a time; `PLAN.md` decision log). **Next:**
settle that and start it; still open: playout caps (recommended: on the GPU, no run F
here) and the analysis of run C''s saved data (the dwarfs' search breadth, its dwarf
policy on C's archives; `PLAN.md`).

**Where we stand:**

- `open_spiel/games/thud/thud.cc` implements the game as designed in `PLAN.md` Phase 3
  (padded 17x17 grid, one function per move type, `IsTerminal` cached and decided without
  generating moves). `thud_test.cc` has **49 test functions, all passing**; the first 46
  passed without any test being changed, the 47th runs OpenSpiel's generic tests, and
  the 48th and 49th (2026-10-02) test the policy-plane map and the symmetry helpers
  (`PLAN.md` Phase 6, stages 5 and 5b).
  `ctest -R thud` takes about 3 s. A planted wrap-around bug is caught by 21 tests, and
  removing either end-of-battle limit by 2.
- **The full OpenSpiel suite passes 285 of 285** (0 build warnings), including the upstream
  Python tests that run every game and `playthrough_test` against Thud's new baseline,
  `open_spiel/integration_tests/playthroughs/thud.txt`. **Regenerate the playthrough and
  read its diff after any change to the rules, the encoding, the text formats or the
  observation** (`./open_spiel/scripts/generate_new_playthrough.sh thud`).
- The design decisions (`PLAN.md` Phase 3 list, each with its evidence): position text with
  the cut-off corners written `-`, `PositionFromText()` rejecting malformed text and
  impossible positions; every capture written `(r,c)-(r,c)x`; a 6-plane observation; the
  move history as information state; a padded 17x17 grid; `Cell::kOffBoard`; states built
  from a `Position`; games started from a diagram saved as their position text, then their
  moves.
- **Rule for failing tests (user, 2026-09-25):** a test that fails and seems to need
  changing is never just edited to pass — record it, and review each such change with the
  user. None was needed in session 5.
- **After changing any test, run `thud/experiments/crosscheck_tests.py`**: it checks every
  hand-written move expectation (110 checks) and every test diagram against hexparrot's
  engine and the reading rules. Four scripts in `thud/experiments/` (`perft_reference.py`,
  `crosscheck_tests.py`, `move_kinds_sim.py`, `limits_sim.py`) run hexparrot/thudgame;
  `random_endings.py` and `mcts_games.py` need only pyspiel; `az_layout_check.py` and
  `az_speed.py` need pyspiel and the JAX set. The C++ programs — `az_layout_check.cc`
  (the layout control), `az_throughput.cc` (with `az_throughput.sh`), `az_reuse.cc` and
  `az_buffer_stats.cc` (search breadth from a saved replay buffer) — are built by
  `build_az_program.sh PROGRAM` against `build-shared/libopen_spiel.so`; `az_match.cc`
  (head-to-head matches, searching with our copy; `sims_a`/`sims_b` for uneven matches)
  and `az_head_check.cc` (the layout check on our copy, for the policy head and
  augmentation) by `thud/az/build.sh`, and `az_sims_gain.py` analyses an uneven match
  against its baseline; `az_thud.flags` holds our trainer defaults. Training
  runs, their scripts and match results live outside the repo in `~/thud-runs/`.
  hexparrot runs from a clone
  outside the repo, by default
  `~/.local/share/thud-openspiel/hexparrot_thudgame` (under `$XDG_DATA_HOME` if set), which
  **exists on this machine** at the pinned commit 7b171108 (2026-09-25). If it is missing,
  every script stops with a full explanation: what hexparrot is, which scripts need it,
  where it goes, and the exact commands to restore and check it.

**Next steps, in order (`PLAN.md` Phase 6).** The order of all Phase 6 work, what each
stage must show first, every decision so far and what waits for later are in **`PLAN.md`
Phase 6, *Phase 6 roadmap and decision log*** — read it before choosing what to do.

1. **Run C is done** (2026-09-27 to 2026-09-29, 12 hours of machine time over three
   sessions: stopped once for the user's evening, paused ~2 hours once on battery;
   resuming works in our copy). Final network step 29; `buffer_stats.jsonl`: the dwarfs'
   searches narrowed to a median 11 moves visited, the most visited holding 0.30 (the
   ~30% mark for tree reuse, one reading); in self-play the dwarfs did worse (mean return
   target −0.13 at step 25, −0.40 at step 29). Matches searching with upstream's rule:
   against C step 16 +1.1 (even); against A step 14 −11.5 (95 of 100 pairs lost; as
   dwarfs −20.0, step 16 −2.4); against the anchor +21.2 (step 16 +46.4; as dwarfs −4.5,
   step 16 +21.6). But under upstream's rule neither C network plays the dwarfs as
   trained (`untried_move_check`: their searches spread over 99 and 76 moves, so the
   move is picked by one evaluation each). Details: `PLAN.md`, *First training runs*.
2. **Verify the untried-move rule and solve the dwarf collapse** (user, 2026-09-29).
   **Step 1 is done** (every network as trained, `~/thud-runs/step1_as_trained.sh`,
   19:21-23:25; table and reading in `PLAN.md`, *First training runs*): no collapse like
   run A's (C's dwarfs never below −0.6 against the anchor, A step 16's −24.9), but a
   drift from step 16 on — step 29 beats step 16 (+4.1, +0.9 to +7.3), yet its dwarfs
   lost 10.1 points against the anchor (p = 0.001) and it loses to A step 14 (−8.5, −10.9
   to −6.2, 50 of 60 pairs; as dwarfs −16.8), which step 16 had been even with.
   Specialising on its own play: by the plan's rule, **Step 2 is indicated — for the user
   to decide.** Also found: C's dwarfs play better with a broad search that picks by one
   evaluation per move (+21.6 against the anchor at step 16) than with their policy's
   focused one (+9.5) — their move probabilities look like the weak point.
   **Step 2 is done** (2026-09-30; results in `PLAN.md`, *First training runs*). The quick
   forgetting check (`thud/experiments/az_forgetting.cc`) had found C's dwarfs' move
   probabilities drifting, so run D branched C at step 15 with option C (a buffer of
   262,144 pre-filled from C's archives by `az_merge_buffers`, C's cadence, C's 64
   batches per step with the new `learner_batches`), to step 29 (`~/thud-runs/
   step2_buffer4x.sh`; frozen 07:40-10:48 by a battery freeze, stopped at 10:54 with
   the Claude Code session that had started it, resumed detached at 14:43 with
   `step2_buffer4x_resume.sh`). Results, each as trained: **D29 beats C29 by +7.3 and
   C16 by +8.4**; against A14 −5.4 (C29: −8.5); its dwarfs against the anchor +11.7,
   +8.1, +3.2 at steps 20, 24, 29 (C: +3.2, +0.9, −0.6); D's dwarf policy keeps fitting old
   positions at C16's level (C29's narrowing gone). Forgetting explains part: the longer
   memory removes the policy's narrowing and makes the strongest network so far, but the
   dwarfs still slip against the anchor and the gap to A14 remains. **Adopted** (user,
   2026-10-01): a 7x memory as the baseline (its values to revisit later, `PLAN.md`,
   *Settings to determine*, 7); run E, D continued with it to step 44, beats D29 by +9.7
   and its dwarfs no longer slip, but it still loses to A14 (−3.5; above).
3. **Proposed step back** (Claude, 2026-09-29; for the user to decide): on this CPU only
   "does it work" questions — correctness, learning against fixed opponents, no
   collapse; "which setting is best" waits for the GPU, since one run per setting cannot
   tell a better setting from a luckier run. Before the GPU: Steps 1-2, playout caps
   (3b) with equal settings per side and a no-worse run, the convolutional policy head
   (5), a GPU throughput test; the settings (4), network growth and the buffer then on the
   GPU. **Tree reuse (3c) is postponed** (user, 2026-09-29): why, how to build it (root
   noise re-applied, Leela Zero's reset) and how to test it are in `PLAN.md`.
4. **Stage 3b: playout cap randomisation** in our copy, after Step 1. A proposal is
   written up (`PLAN.md`, *Changes to the search and trainer*, playout caps: `p` 0.25
   for both sides, `N` 400, `n` 100, quick searches unrecorded and without root noise,
   a switch off by default, judged against run C at equal machine time). Since measured
   (2026-10-01/02): here it costs 7x fewer training positions an hour for moderately
   better targets (stable against a 4,000-simulation reference) — **recommended: no run
   F on this CPU, playout caps on the GPU; for the user to decide.**
   **Parked:** the uneven match (`~/thud-runs/night_2026-09-28.sh`, ~8 hours; per-side
   settings become a setting to tune later); the 400-simulation match (dropped for now);
   replicates of runs A and C (skipped); the backlog match with the new rule in both
   searches (replaced by Step 1's C29 against A14, each as trained).
5. **If we ever fall back to the Python route** (user, 2026-09-25): first re-verify the
   layout-bug findings (`PLAN.md` Phase 6), then fix it in **one concise PR with
   experiments verifying correctness, for example on tic-tac-toe, chess and Thud**; the
   uncompiled inference needs its own fix.

Earlier conclusions that still stand (`PLAN.md` Phase 5): no further optimisation of the
game logic for now — the network, not the move generator, is the cost of AlphaZero-style
learning. MCTS against MCTS (dwarfs win 7 of 10 at 1,000 simulations, 19 of 20 at 5,000)
measures the searcher, not Thud's balance; OpenSpiel's AlphaZero evaluates itself against
that searcher, so read its evaluation per side.

**Where we are:** The repo is at `~/thud-openspiel` (WSL2, Ubuntu 26.04.1, aarch64) on branch
`thud`, pushed to `origin`, with an `upstream` remote and the pre-push hook installed. OpenSpiel
builds natively on ARM64: clang 21.1.8, cmake 4.2.3, Python 3.14.4 venv; `make -j10` takes
4m41s with 0 warnings; `ctest -j10` passes 285 of 285 in about 2 minutes;
`./examples/example --game=tic_tac_toe` runs. The `manylinux` wheel fallback was not
needed. `build/` is OpenSpiel's default "Testing" build (`-O2`, all checks on); speed
measurements use a Release build in `build-release/` (`BUILD_TYPE=Release`, only
`benchmark_game` and `mcts_example` built — commands in the session-5 log), and the C++
AlphaZero a Release build with LibTorch in `build-torch/` (commands in `CLAUDE.md`). The
venv now has the JAX set and `torch==2.10.0`, so **the next cmake run in `build/` adds
their Python tests**, and the 285 will grow.

**Decided 2026-09-22 (all by the user; reasoning in the session-2 log):**

1. **Lone dwarf hurl — allowed.** The official rules say so explicitly.
2. **Troll captures — all or none; declining allowed.** No removal phase.
3. **Shoves travel 2..N squares** (a one-square shove equals a capturing step).
4. **One action per turn, numbered from-square × direction × distance** as in OpenSpiel's
   `chess`, plus separate "troll steps and captures all" IDs: 19,800 actions
   (`THUD_RULES.md` §8).
5. **Implementation must be very fast yet readable** (user requirement, now in `PLAN.md`
   Phase 3).
6. **End-of-battle limits: 200-turn no-capture cap, 800-turn hard cap** (user, chosen by
   measurement over 50/400, 100/400, 100/800 and 200/400); to be re-evaluated once our engine
   plays strongly (`PLAN.md`, *Deferred decisions*). Termination otherwise as before: the
   battle also ends when the player to move has no legal move.

**Decided 2026-09-23 and 2026-09-24 (session 3, all by the user; reasoning in the session-3
log and in `PLAN.md` Phase 3):**

7. **Tests first, then a review of them, then a review of the design, then code.** The tests
   include whole-position checks: the opening's complete legal set, perft against
   hexparrot, the board's 8 symmetries, and invariants in every position of random games.
8. **Position text kept**; reading rejects malformed text and impossible positions, and
   `PositionFromText()` reports that by returning nullopt, as chess's `BoardFromFEN` does.
9. **One notation for every capture:** `(r,c)-(r,c)x`; `(r,c)-(r,c)` captures nothing.
10. **Observation: 6 planes** — dwarfs, trolls, empty, trolls to move, and the two
    counters. The Thudstone is a hole like the cut-off corners, 0 in every board plane.
11. **Information state: the move history**, as in tic-tac-toe, chess and Go.

**Decided 2026-09-24 (session 4, all by the user; reasoning in the session-4 log and in
`PLAN.md` Phase 3):**

12. **Board storage: a padded 17x17 grid**, stepped by fixed offsets — for readability and
    safety, not speed. Tracking lines of pieces incrementally, and `UndoAction`, are
    postponed as Phase 5 candidates.
13. **The cut-off corners are written `-`, not `#`**, so that no line of the position text
    looks like a comment in OpenSpiel's saved-game files.
14. **The `thud.h` interface:** `Cell::kOffBoard` for the grid's non-squares; states built
    from a `Position`; a game started from a diagram saves its position text, then its
    moves.

**Verified 2026-09-23:** the `~/.bashrc` lines from `CLAUDE.md` do reach Claude Code's Bash
tool — a new session's shell snapshot has `PYTHONPATH` and the venv (`which python3` →
`venv/bin/python3`). Session 2's worry was unfounded.

**No outstanding chores.** (Session 2's clean-up is complete: `C:\Users\waech\thud-init\` is
deleted and the Windows-side project memory redirects to this repo.)

---

## Session log

### 2026-09-21 — Session 1: research and scaffolding

No code written. This session was environment research and project setup, and most of its
value is in the facts recorded below and in `CLAUDE.md`.

**Decisions made**

- **C++ game, not Python.** Driven by the expectation that move generation bottlenecks
  training. Thud's branching factor is large (32 dwarfs moving like chess queens) and games
  are long.
- **Develop in WSL2 / Ubuntu on ARM64, not on Windows.** See the hardware findings below.
- **Fork OpenSpiel and work in-tree**, rather than a separate repo with OpenSpiel as a
  submodule or an out-of-tree C++ target. The deciding factor was testing: OpenSpiel's
  playthrough regression system (`generate_new_playthrough.sh`) and its `ctest` targets only
  exist in-tree, and for a game with rules as fiddly as Thud's those are the most valuable
  correctness tools available. `docs/library.md` claims external game registration is
  possible but documents no way to actually do it.
- **`CLAUDE.md` sits at the repo root**, not in `thud/`, despite the preference for keeping
  the fork's top level clean. Claude Code loads `CLAUDE.md` from the working directory and
  every *parent* at launch, and a subdirectory's file only on demand when it reads files
  there. Since sessions run from the repo root, a `thud/CLAUDE.md` would not load at session
  start. Everything else lives under `thud/`.
- **Apache 2.0 throughout**, including our own new files. Apache 2.0 would permit licensing
  our additions differently, but mixing licenses in one tree creates ambiguity for no gain
  and would foreclose upstreaming.
- **Scope: a correct game first.** The AI approach and the compute plan are explicitly
  deferred until Phase 5 benchmarks exist.

**Hardware and environment findings (measured — do not re-derive)**

- Host is a Snapdragon X, **ARM64**, 10 cores, 31.6 GB RAM, ~328 GB free.
- The Windows Python is an **x64 build under emulation** — its pip tags are `win_amd64`
  despite `platform.machine()` reporting `ARM64`. No `win_arm64` OpenSpiel wheels exist.
  This is why Windows-native development was rejected.
- OpenSpiel publishes **`manylinux_2_28_aarch64`** wheels for cp311–cp314, so WSL gives a
  native ARM64 OpenSpiel. Useful as a fallback if the C++ source build misbehaves.
- **No CUDA GPU** (Qualcomm Adreno X1-85), and there is no official prebuilt LibTorch for
  aarch64 Linux — only PyTorch aarch64 wheels which bundle libtorch. So OpenSpiel's C++
  AlphaZero (`OPEN_SPIEL_BUILD_WITH_LIBTORCH`) is high-risk here and is off for now.
  Serious neural-network training will need cloud compute or a reduced scale.
- `gh` (GitHub CLI) is **not installed**, which is why the fork is created via the web UI.

**Prior art found** — worth cross-checking rules against, none of it in OpenSpiel:
`github.com/dstu/thud` (has MCTS), `github.com/hexparrot/thudgame`,
`github.com/willwagner602/Thud`, `github.com/Saucistophe/Thud`,
`github.com/THFlowers/Thud-CLI` (MCTS). Thud is genuinely new work for OpenSpiel.

**Rules research (Phase 1) — done, written up in `THUD_RULES.md`.** A first attempt failed
to an API outage and was redone. The official PDF at
`tesera.ru/images/items/1543265/THUD_RULES.pdf` returns HTTP 403 and Wikipedia carries no
ruleset, but the official rule text is reproduced on BoardGameGeek and in the Tabletop
Simulator workshop edition, which together resolved most of the open questions:

- Troll captures are **optional and a chosen subset** — "any (all) ... may be captured",
  "capturing is not compulsory". This was the question most likely to cause rework, and the
  answer is the expensive one for the action space.
- A **shove is legal only if it captures at least one dwarf**.
- Setup is exact and arithmetically confirmed: dwarfs occupy all 36 perimeter squares except
  the 4 in line with the Thudstone, giving 32; trolls occupy the 8 squares around it.
- Dwarfs move first; a dwarf never captures by moving; a hurl may only land on a troll.
- The official end condition is **agreement-based** and therefore not implementable —
  `THUD_RULES.md` section 8.4 specifies a concrete surrogate.

Note that `wiki.lspace.org/Thud` is unreliable on the rules (it implies trolls can hurl);
prefer the BGG text. `tafl.cyningstan.com` refused connections.

**Cross-checked the remaining ambiguities against three implementations** —
`dstu/thud` (Rust, MCTS), `THFlowers/Thud-CLI` (Java, MCTS), `hexparrot/thudgame` (Python).
This settled two of four: a lone troll may not shove (unanimous), and termination should be a
no-progress cap plus a stalemate check (`hexparrot`'s 400-ply cutoff is annotated "self-play
only"; `dstu` implements the official agreement mechanism as proposal/accept/decline, which
is faithful but does not survive self-play). The remaining two were then resolved as well:

- **Lone dwarf hurl — not allowed.** Decided on a textual argument rather than the 2-to-1
  implementation split: the official rules phrase hurl and shove *identically* ("anywhere
  there is a straight line of adjacent trolls/dwarfs ... they may shove/hurl"), so identical
  phrasing must be read identically, and shove's `N >= 2` is unanimous. `THFlowers` is simply
  wrong here.
- **Troll captures — a separate dwarf-removal phase**, the faithful reading.

Detail and code quotes are in `THUD_RULES.md` section 8.

Caveat for whoever revisits this: the `dstu/thud` reading of troll captures is **not
settled**. Its action type carries a capture list, which initially looked like a player
choice, but generation appears to compute the captured set itself — which reads more like
capture-all. Read `board.rs` properly before citing it either way.

**Two findings that changed the design:**

- **`open_spiel/games/amazons/` is a better template than `tic_tac_toe/`.** The developer
  guide points at tic-tac-toe, but that is one action per turn. Amazons implements a genuine
  multi-phase turn (`enum MoveState { amazon_select, destination_select, shot_select }`,
  with `current_player_` flipped only in the last phase), which is the same shape as Thud's
  move-then-select-captures turn. Nothing in OpenSpiel requires players to alternate.
- **That reopens the action encoding as a real trade-off.** Decomposing the turn into
  sequential single-square choices, Amazons-style, collapses `NumDistinctActions` from
  ~27,225 (from-to pairs) to roughly **166**, which is far friendlier for a policy network,
  at the cost of 2–3x tree depth. Currently leaning that way; see `PLAN.md` Phase 2.

**Added a pre-PR upstreaming checklist to `PLAN.md`**, prompted by the observation that
`THUD_RULES.md` quotes copyrighted rule text and therefore can never be upstreamed. Several
items on that list are cheap to honour as we go and expensive to retrofit — chiefly keeping
our files out of `open_spiel/` and keeping `master` clean.

**Built three layers of protection against accidentally upstreaming private files.** Worth
recording that **no GitHub feature can do this** — a PR is just a diff between branches, and
nothing in a file's contents can veto it. (`.gitattributes export-ignore` does not help; it
only affects `git archive`.) So: structural (build the PR branch so those files were never on
it), mechanical (`.githooks/pre-push`), and advisory (uppercase banner in `THUD_RULES.md`
plus the rule in `CLAUDE.md`). The advisory layer probably matters most in practice, since the
agent preparing a PR reads those files.

The hook was tested before being trusted, against three cases: upstream HTTPS URL (refused),
upstream SSH URL (refused), and our own fork's URL (allowed). That last case is the trap — a
fork's URL also contains `open_spiel`, so a naive match would block every push to our own
repo; the hook matches on the owner instead. A buggy pre-push hook fails *silently open*,
which is worse than having none, so re-test it if it is ever edited.

**Also produced this session:**

- `VSCODE_WSL_SETUP.md` — human-facing one-time setup for running Claude Code in VS Code
  against WSL. Anthropic's VS Code extension docs never mention WSL, so this follows
  Microsoft's remote model. Two corrections worth keeping: the extension does **not** need
  the `claude` CLI installed (self-contained; the CLI is only a fallback for editors that
  cannot install extensions), and it must be installed **into the WSL remote**, not on the
  Windows side, or it silently operates on the wrong filesystem.
- **A shell-state gotcha now recorded in `CLAUDE.md`:** Claude Code's Bash tool does not
  persist shell state between commands. The working directory carries over; exported
  variables and `source venv/bin/activate` do not. So the Python environment must live in
  `~/.bashrc`, not in a per-session activation.
- A cleanup step (`START_HERE.md` step 6) for the now-redundant Windows-side files, gated on
  the repo being committed, pushed and built. Note the dependency: the Windows-side memory
  points at `thud-init/START_HERE.md`, so deleting that folder and updating the memory must
  happen together.

**Next step:** as recorded in `## Current status` above.

### 2026-09-22 — Session 2: Phase 0 in WSL, and a prior-art re-check that reopened a rule

**What changed**

- Added the `upstream` remote; created `thud` from `master` (fork `master` was identical to
  upstream at `48401890`). Copied the scaffolding into place, committed it (`9b64ed77`) and
  pushed `thud` to `origin`. `master` is untouched.
- Installed the pre-push hook (`core.hooksPath = .githooks`) and re-tested it under Linux,
  where `/bin/sh` is `dash` rather than the Git Bash it was first tested with. Simulated
  inputs: upstream HTTPS new branch → refused; upstream SSH existing branch → refused;
  upstream clean `master` → allowed; own fork → allowed.
- `install.sh` (cloned abseil, pybind11, pybind11_abseil, pybind11_json, json, dds), venv
  with `requirements.txt`, CMake + `make -j10`, `ctest -j10`: see `## Current status`.
- Appended the `CLAUDE.md` environment lines to `~/.bashrc` (they work in interactive
  shells; whether they reach Claude Code's Bash tool is unverified, see status).
- Windows side: updated `project_thud_openspiel.md` to redirect to this repo and dropped its
  hardware findings; left `MEMORY.md` and the feedback memory alone as instructed. Deleted
  `C:\Users\waech\thud-init\` with the user's explicit approval, including
  `VSCODE_WSL_SETUP.md`, which existed only there. The launch command it carried,
  `code --remote wsl+Ubuntu /home/waech/thud-openspiel`, survives in the Windows memory.
- Copied the general "evidence over assertion" feedback memory into the Linux-side memory,
  which started empty.

**Environment findings (measured)**

- WSL sees **15 GB RAM and 4 GB swap**, not the host's 31.6 GB (WSL defaults to half).
  Peak RSS of any one compiler process during the full build was 1.2 GB, so `-j10` is safe.
- A fresh WSL home has no git identity, no GitHub credentials, and no build toolchain
  (no `clang`, `cmake`, `make`), and `sudo` needs the user's password — so the user had to run
  the `apt-get install` (install.sh's list plus `make` and `gh`), `git config --global`, and
  `gh auth login` themselves. Git pushes now go through `gh auth git-credential`.
- `generate_new_playthrough.sh` calls `python`, not `python3`, so it needs the venv active.
- It also writes a new file under `open_spiel/integration_tests/playthroughs/`, which
  `CLAUDE.md`'s "stay out of upstream code" list does not mention (PLAN's PR checklist does).

**Prior-art re-check — THUD_RULES §8.1–§8.3 misattributed evidence.** Shallow clones of all
three engines were read in full along every hurl/shove path (generator, validator, the
authors' tests) and hexparrot was also executed. Findings:

| | dstu/thud (Rust) | hexparrot/thudgame (Python) | THFlowers/Thud-CLI (Java) |
|---|---|---|---|
| Lone dwarf hurls onto adjacent troll | **allowed** | **allowed** | **allowed** |
| Hurl distance | `1..N` | `1..N` | `1..N` |
| Lone troll captures after a 1-square step | yes — as a distance-1 `Shove` | yes — "a shove of length 1" | yes — `M` move, then optional `R` remove turn |
| Lone troll shoves 2+ squares | no | no | no |
| Which dwarfs a troll captures | all on `Shove`; a plain `Move` never captures | all (engine/AI); GUI "Compulsory Capturing" off → chosen subset, shove needs ≥1 | chosen subset (human CLI); AI offers only all-or-nothing |

Citations:
- **dstu** `thud_game/src/board.rs:152-171` (`hurl_action_from`) and `:175-202`
  (`shove_actions_from`) zip the forward ray (after `.skip(1)`) with the reverse ray
  **without** a skip. `Ray::new(c, d)` yields `c` itself first (`board.rs:99-105`), so the
  first "previous" square checked is the moving piece itself — which is why a lone piece
  passes. The session-1 reading ("the square *behind* the dwarf must hold a dwarf") missed
  this. The author's reference property tests (`board.rs:609-648`, `:679-734`) start the
  line at `Some(*start)` and agree, but are disabled (`// #[quickcheck]`). The **active**
  test `troll_can_move_and_shove` (`actions.rs:507-577`) requires one-square shoves by the
  isolated trolls at (2,4), (6,7) and (6,9). No active test covers hurls.
  `State::role_actions` (`state.rs:42-51`) passes the board's actions through unfiltered.
- **hexparrot** `thud/gameboard.py:292-320` (`is_valid_cap_normal`): `get_range` is
  inclusive at both ends, so for an adjacent target the two `pop`s leave `seq` empty and it
  returns `[dest]` (dwarf) or all adjacent dwarfs (troll) without inspecting the square
  behind. Active tests `tests/test_gameboard.py:256` "A lone dwarf throws itself one square
  onto an adjacent troll" and `:271` "a one-square troll move that lands beside a dwarf is a
  shove of length 1". Executed `find_caps` myself: lone dwarf D8 → troll C8 generated; lone
  dwarf two squares away not generated; two-dwarf line two squares generated; lone troll
  one-step captures generated; lone troll two-square shove not generated. GUI
  non-compulsory capture logic: `gui.py:437-530`.
- **THFlowers** `src/thud/Player.java:298-332` (`distanceAttackCheck`) counts the moving
  piece in `numInLine` and throws only for `numInLine == 1 && turn == TROLL`. The generator
  used by its MCTS, `getPossibleTrollPieceMoves` (`:350-407`), emits shoves only for
  `steps >= 2`. One-square captures go through `M` plus the remove turn (`mustRemove()`,
  `:168-191`: optional after `M`, mandatory after `S`). The AI's removal options are
  all-or-nothing (`PossibleMoves.java:33-43`, "All or nothing options to simplify AI").

Consequences:
- §8.2's vote is actually **3–0 for allowing** the lone-dwarf hurl, not 2–1 against.
- §8.2's textual argument rested on "shove's `N >= 2` is unanimous", which is false: dstu
  and hexparrot both call a lone troll's capturing step a shove of length 1. And for trolls
  the question has **no effect on outcomes** (next point), so it cannot anchor the hurl
  reading. Two of the three engines in fact read hurl and shove identically, with a lone
  piece counting as a line of one.
- §8.1's "hexparrot captures all" is true only of its engine/AI; its GUI implements our
  settled model (a) as an option. dstu is now clear: capture-all on shove, none on move.
  Model (a) itself was chosen on the rules text, not the vote, so it is unaffected — but
  note that **no existing Thud AI searches capture subsets**; all three offer all-or-none.

**Design finding — no "move type" component is needed in either encoding.** A distance-1
shove (troll moves one square, then removes a non-empty set S of adjacent dwarfs) yields
exactly the same position as a one-square troll *move* followed by removing S, and the move
is strictly more permissive (it also allows removing nothing). So distance-1 shoves can be
dropped without losing any reachable outcome: a troll moving to an adjacent square is a move
(removal optional), one moving 2+ squares is a shove (≥1 removal required). THFlowers' MCTS
generator does exactly this (`steps >= 2`). Dwarf move vs hurl already separates by landing
square (empty vs troll). This corrects PLAN.md Phase 2's claim that the shallow encoding
needs an explicit move-type component.

**Other Phase 2 inputs**
- Opening branching factor (counted with hexparrot's engine): dwarfs have **656** legal
  moves (all 32 dwarfs, 18–24 destinations each); trolls have 32.
- `amazons` and `checkers` both expose only the board in `ObservationTensor` — no side to
  move, no sub-phase, no selected piece — so their observations are not Markov mid-turn.
  Thud's tensor needs those planes (plus the must-capture flag after a shove). Copy
  Amazons' state machine, not its observation.
- `checkers` is a second in-tree precedent for a player acting several times in a row (a
  multi-jump keeps `current_player_`, `checkers.cc:323`), and for a no-capture draw counter
  (`kMaxMovesWithoutCapture = 40`, `checkers.h:61`).
- Whatever the encoding, count the no-capture cap and the hard cap in **turns**, not
  OpenSpiel actions, so the encoding choice cannot change the game; derive `MaxGameLength`
  (in actions) from the turn cap.

**Addendum (same session) — official rules text found; encoding decided.**

- **The official rules are on the archived official site.** Two copies of the same
  Classic Thud text, marked "Copyright 2001/2005 Terry Pratchett, Trevor Truran and Bernard
  Pearson": `https://web.archive.org/web/20071113030431/http://shop.thudgame.com/rules`
  (page 1; page 2 at `.../web/20071103213932/http://www.thudgame.com/rules2`) and
  `https://web.archive.org/web/20060912052229/http://game.thudgame.com/rules2.html`. The 2002
  site FAQ says the printed rulebook was deliberately never put online, so this is the
  earliest official text reachable. Key sentences:
  - *"Capturing is not compulsory."*
  - *"A troll captures one or more dwarfs by moving to a square next to it (them)."*
  - Shove: *"the front troll is shoved in the back by the rest in the line … moves directly
    forward as many squares (or fewer) as there were trolls in the line. A troll can only be
    shoved to make a capture."*
  - Hurl: *"4 dwarfs can attack a troll if there are 0, 1, 2, or 3 empty squares between the
    front dwarf and the troll"* — i.e. distance `1..N`.
  - *"(Note: 1 lone dwarf can form a line of 1 by moving to a square adjacent to a troll then
    hurling himself and capture a troll in this way. …)"* — **settles the lone-dwarf hurl:
    allowed** (pending the user's confirmation).
  - Nothing about choosing *which* adjacent dwarfs to capture. The "any (all) … may" wording
    quoted in `THUD_RULES.md` is not on the site; the user points to the tesera.ru rulebook
    PDF as its source (HTTP 403 to every fetcher tried this session).
- **THFlowers' "remove turn" is not a separate game turn.** `PlayState.alternateTurn()` does
  nothing while `removeTurn` is set, so the troll player enters `R …` immediately and the
  dwarfs do not move in between — it is an input stage of the same troll turn, costing no
  tempo. So for a lone troll stepping next to dwarfs, all three engines' AIs offer the same
  two outcomes: capture none or capture all.
- **Encoding decided by the user: one action per move, `(from, to)`.**
- **Troll captures: user leaning to all-or-none**, awaiting confirmation. Consequences if
  confirmed: no removal phase and no removal-order question; a shove always captures all
  adjacent dwarfs (it must capture ≥1, so "none" is not an option); a one-square troll step
  next to dwarfs has two outcomes and so needs two action IDs. Precedents: dstu uses distinct
  `Move`/`Shove` variants and hexparrot distinct plies (a separate action per variant);
  THFlowers uses a follow-up `R all`/`R none` stage.
- **Dwarf hurls have no equivalent issue:** a hurl always captures exactly the one troll it
  lands on (no choice of which or how many, and no "decline" — the rules allow a hurl only
  if it captures), and `(from, to)` determines it uniquely.

**Addendum 2 (same session) — all decisions made; spec rewritten.**

- **Lone dwarf hurl: allowed** (user, on the official note quoted above).
- **Troll captures: all or none, declining allowed** (user chose this over compulsory
  capture and over a game parameter). The user asked whether declining ever makes sense:
  with lone hurls allowed, a declined dwarf stands next to the troll and can hurl it on the
  next turn (1 dwarf for a 4-point troll), and declining never changes where the troll
  stands, so it is almost always a blunder. The only counter-case found is second-order (the
  captured dwarf was blocking a dwarf line aimed at a *different* troll), and even there
  capturing is one dwarf ahead. Kept anyway because the official rules say capturing is not
  compulsory, and dstu's and THFlowers' MCTS keep the option too. hexparrot's engine can
  decline, but its server and default GUI capture compulsorily and its AI always captures
  when it can.
- **Capture vs. decline encoding: separate action IDs** for "step and capture all" (user;
  dstu and hexparrot precedent), not a follow-up decision (THFlowers). Reasons given: every
  turn stays one action, so no mid-turn state has to appear in the observation; and grouping
  a strong capture and a usually-bad decline under one tree edge would briefly drag the
  capture's value estimate down under UCT averaging. That second point is our own reasoning.
  The MCTS literature calls such bundling "move groups" (Childs, Brodeur & Kocsis, IEEE CIG
  2008), but only that paper's abstract could be read.
- **Numbering: from-square × direction × distance** (user), as in OpenSpiel's `chess`;
  `chinese_checkers`, `checkers`, `breakthrough` and `clobber` also use square × direction.
  Only `nine_mens_morris` (24 points) uses raw from×to pairs. Both layouts contain the same
  6,300 geometrically possible moves; raw pairs would need 27,225 IDs (23% usable), this
  layout needs 18,480 (34%) plus 1,320 capture IDs. MCTS never sees the numbering. A
  **correction** was made in conversation: OpenSpiel's stock AlphaZero ends in a fully
  connected layer in both implementations (`python/algorithms/alpha_zero/model_nnx.py`
  `PolicyHead`; `algorithms/alpha_zero_torch/model.cc` `policy_linear_`), so there the
  layout only changes the size of the last layer. Board-shaped output planes would need a
  custom convolutional policy head.
- **Section numbers changed:** the old §8 ("Ambiguities and our decisions", items 8.1–8.6)
  is now §9, and §8 is now the action encoding. Log entries above this line use the **old**
  numbering.
- **`THUD_RULES.md` rewritten** at the user's request to be sharp and concise: sections 1–8
  are the specification, with explicit Allowed / Not allowed lists per move type, the end
  conditions, scoring, and the full action encoding (§8, including which action is which
  move). Section 9 gives the reasoning in one short paragraph per decision, with sources. The
  session-1 history and per-implementation evidence were removed from it; they live in this
  log. Verified by script: 165 squares; row-major indices as stated; maximum straight-line
  distance 14; the 36-square perimeter and the 32-dwarf / 8-troll setup match hexparrot's
  actual starting position square for square.
- **`PLAN.md`:** Phase 1 summarised as settled; Phase 2 records the encoding; Phase 3 gains
  the user's "very fast yet readable" requirement and switches the template from `amazons`
  (no longer needed with one action per turn) to `tic_tac_toe` for the boilerplate and
  `chess` for the action layout; Phase 4's test table now matches the rules (it previously
  required a lone-dwarf hurl to be *illegal* and tested a removal phase that no longer
  exists).
- The `amazons`/`checkers` observation lesson above still applies in its general form: the
  observation must include the side to move. There is no mid-turn state any more.

**Measurement — how the end-of-battle limits affect game length** (asked by the user; run
with hexparrot's engine, whose rules match ours closely enough for length statistics):

- *hexparrot's heuristic AI, 50 self-play games:* all ended in a rout (one side wiped
  out) after a median of 73 turns (max 109); the longest stretch without a capture was 21
  turns. No limit ever mattered.
- *Uniformly random play, 1,000 games* (the regime of plain-MCTS rollouts and early
  training), run on 10 processes in 148 s: every game ended in a rout if left alone —
  median 334 turns, mean 343, max 794. Turns actually played under each limit pair:

  | No-capture cap | Hard cap | Mean turns | vs 50/400 | Games cut short |
  |---|---|---|---|---|
  | 50 | 400 | 294 | 1.00x | 57% |
  | 100 | 400 | 327 | 1.11x | 25% |
  | 50 | 800 | 295 | 1.00x | 56% |
  | 100 | 800 | 333 | 1.13x | 16% |
  | 200 | 800 | 342 | 1.16x | 1.5% |
  | none | none | 343 | 1.17x | 0% |

  So in this regime even removing both limits costs only +17% turns, and the hard cap
  barely matters — the no-capture cap is the lever. The user asked about 100/400 as a
  speed-up for early training: 327 mean turns, **−4.4% vs 200/800**, but 25% of games cut
  short (10% by the no-capture cap, 15% by the 400-turn cap) vs 1.5%. The shortest pair
  tested, 50/400, saves 13.9%. Precedents: OpenSpiel `checkers` ends
  after 40 plies without a capture; chess's 50-move rule is 100 plies.
- *hexparrot's heuristic AI at lookahead 3 (its GUI/console default; in its code lookahead
  only affects the dwarf side), 3,000 self-play games* on 10 processes in 755 s, 2,997
  distinct: **every game ended in a rout** (dwarfs won 85%, trolls 15%); median 79 turns,
  99th percentile 120, max 152; longest no-capture stretch median 10, 99th percentile 21,
  max 41. Under both 100/800 and 200/800: mean 80.1 turns, **0 games cut short**. So this
  proxy cannot tell the two apart — and its games never stall, unlike the official rules'
  description of real games ending by agreement once captures dry up. It is the best proxy
  available, but a weak one for strong play.
- *Not measured:* strong play. The official strategy text says that once the dwarfs form a
  block, "it becomes difficult for the trolls to make any more gains" — so strong games may
  stall, and then **every** game runs the full no-capture cap at its end, making each
  +1 of the cap cost +1 turn per game.
- Reproduce with `thud/experiments/limits_sim.py` (`--player random --games 1000` and
  `--player ai --games 3000`; setup in its docstring). Seeds are fixed; checked on 100 seeds
  of each mode against the original runs, with identical results. The raw per-game data
  (3.4 MB) was not kept.
- The user asked for a plan note to re-evaluate the limits once our engine is strong; it is
  in `PLAN.md` under *Deferred decisions*.
- Also evaluated on the saved data, 200/400: 0 of the 3,000 AI games cut short, but 199 of
  1,000 random games (198 by the 400-turn cap, since 199 random games run past 400 turns);
  mean 331 turns, −3% vs 200/800. With a 400-turn cap, any game whose last capture falls
  after turn 200 ends at the turn cap instead of the no-capture cap.
- **Decided (user): 200/800**, now in `THUD_RULES.md` §6, with the evidence summarised in
  §9.4. It cuts the fewest early (random-like) games short for 3–16% more turns than the
  shorter pairs, and no pair tested ever cut an AI game. The observation must include both
  counters (now in `PLAN.md` Phase 3, following `chess`'s 50-move counter).

**Session close.** The user read `THUD_RULES.md` and asked two questions that led to
clarifications in §8: a shove that would capture nothing is simply illegal (the shove row now
says so), and the first table's Meaning column now names which move types each ID range
covers. `PLAN.md` Phases 1–2 were rewritten to record what was done and the alternatives
rejected; Phase 3 lists the two game parameters; Phase 5 adds checking the limit
measurements against our own engine. `CLAUDE.md` now lists the generated playthrough as an
allowed file under `open_spiel/`. The limit-measurement script was saved as
`thud/experiments/limits_sim.py`. Everything was committed and pushed to `origin/thud`.

**After the commit:** the user proposed **test-driven development** (recommended: the spec is
already allowed / not-allowed lists, and tests written from it cannot inherit the code's
misunderstandings) and for **border corner cases**: moves near horizontal, vertical and
diagonal borders, and hurls/shoves away from, towards and across a border. Both are in
`PLAN.md` Phases 3–4, extended with lines ending at a border, captures at border squares, the
Thudstone and the exact end-condition boundaries. Checked by script: the octagon is convex
along all 8 directions (no straight path leaves the board and re-enters, so "across the
border" always means the path leaves the board); the longest move is 14 squares
orthogonally but only 9 diagonally.

The user then raised **row/column wrap-around**: with a flat board array, a long move can run
off the end of one row and continue on the next (east from `(5,14)` lands on `(6,0)`), and in
rows and columns 5–9 both ends are playable, so an off-board mask does not catch it (three
such examples verified by script). Added as a corner case, with a Phase 3 design rule: build
the ray tables from `(row, col)` coordinates, never by index arithmetic. The user also asked
how test-first is reflected in the plan — it was only a paragraph, while the phase structure
still said "Phase 3 implement, Phase 4 tests". Restructured: **Phase 3 is now "Implement
test-first"** and contains the unit-test table and corner cases; **Phase 4 is "Integration
tests and baselines"** (`RandomSimTest`, pyspiel registration, playthrough, cross-checks).

**Next step:** as recorded in `## Current status` — Phase 3.

### 2026-09-23 to 2026-09-24 — Session 3: tests first, their review, and half the design review

**Ground rules from the user.** Write all tests first and pause for a review before
implementing anything they test. Modify OpenSpiel itself only for registration or a clear
bug. An independent Python model of the rules, planned as a cross-check, was dropped when
the user asked why it was needed: the tests derive their expectations from `THUD_RULES.md`
by hand, and perft adds an independent engine.

**Written.** `thud.h` (the API, with a header comment in our own words: rules summary,
parameters, position text format, move notation), `thud.cc` (game type, parameters,
registration, observer, and a `NotImplemented()` stub for every function), `thud_test.cc`,
and the two registration points with their Apache §4(b) notices. Build notes that cost time:

- `SPIEL_CHECK_OP` declares locals `x` and `y`, so a test variable named `x` inside a check
  fails to compile ("cannot appear in its own initializer"). `SPIEL_CHECK_TRUE_WSI` needs a
  `Game`; the tests use their own `Require()` instead.
- A new test target needs `cmake .` in `build/` before `make thud_test`.
- `clang-format` is not installed. Check the 80-column limit with
  `LC_ALL=C.UTF-8 grep -nP '^.{81,}$'` — `awk` counts bytes and overcounts `×` and `−`.
- Python 3.14 starts `multiprocessing` workers with `forkserver`, so the experiment scripts
  pass hexparrot's path to each worker through a pool initializer.

**Completeness checks, added at the user's request before the review:** the opening's
complete legal set for both sides (656 dwarf moves, 32 troll steps); **perft** on five
positions against hexparrot/thudgame's engine (commit 7b171108), which ran from a
temporary clone and is not a dependency — only its numbers are in the test; and the board's
**8 symmetries**, on fixed positions and every position of 10 random games.

**The test review, in 8 chunks, with the user.** Additions, by chunk:

- Hurls: one dwarf ending several lines hurls along each, with `N` counted per direction
  (`TestHurlSeveralLines`).
- Dwarf moves: a move next to a troll captures nothing (`TestApplyDwarfMove`).
- Shoves: the same for a troll ending several lines (`TestShoveSeveralLines`). The user
  asked whether a shove must land next to a dwarf: yes, §5b.
- Troll steps: a troll next to the landing square survives both a capture-all and a
  declined capture; every capture apply test starts from non-zero counters, so the reset to
  0 is visible; and a new random-play test of captures and counters after every move.
- End of battle: a `CheckOver` helper — a finished battle has no player to move, no legal
  action, and the returns of its margin — used by every end test, the turn limits included;
  and capturing the last opposing piece ends the battle at once (a hurl of the last troll; a
  capturing step taking the last two dwarfs). **The user asked whether the dwarfs can run out
  of moves while they have dwarfs: no.** A dwarf next to an empty square can move there, one
  next to a troll can hurl 1 square onto it; if every dwarf were stuck, every square next to
  a dwarf would hold a dwarf or the Thudstone, so dwarfs would fill all 164 other squares.
  The trolls can be stuck with pieces left. The proof is a comment above
  `TestEndNoLegalMove`, and it gives the implementation a cheap end check (`PLAN.md`
  Phase 3).
- Whole positions: `TestShoveMaxDistance` (7 squares along a full row; perft cannot cover it,
  because hexparrot stops hurls and shoves at 6 — evidence that this bug occurs in practice).
  The random-play test became `TestRandomPlay`: in every position the battle is over exactly
  when an ending of §6 holds (decided from the pieces with the proof above) and the players
  alternate; each game ends through `CheckOver` with the margin counted on the board; all
  five kinds of move must have been played; and a `RandomAction` helper fails cleanly on an
  empty legal-action list instead of indexing into it.
- **The user challenged "`TestSymmetry` needs no expected values":** right — a mirroring
  function that did nothing would make both comparisons pass trivially. The test now checks
  its own mirroring code first: each symmetry maps the 165 squares one-to-one onto the
  board, the 8 send (0,5) to 8 different squares, and one quarter turn matches a
  hand-drawn position, move and capturing step (recomputed independently in Python).
- Random games use fixed seeds (20260923, 20260924), as upstream does (`chess_test.cc:339`
  seeds `rng(23)`; `basic_tests.cc` uses the default seed). Whether ten games are enough to
  play every kind of move was measured, not assumed — see the scripts below.
- Also: 9 lines over 80 columns wrapped, and `TestPerft`'s comment corrected — 7-square
  hurls and shoves fit along full columns as well as rows.

**Experiment scripts** (user: keep them, note that they must not go upstream). Both are in
`thud/experiments/`, set up like `limits_sim.py`, and re-run from there with identical
results:

- `perft_reference.py`, the source of `TestPerft`'s counts (about 21 s on 10 processes):
  opening 656 / 22,624 / 14,142,624; opening with the trolls to move 32 / 20,360; tangled
  380 (376 moves, 4 hurls) / 24,359; tangled with the trolls to move 64 (48 steps, 11
  capturing steps, 5 shoves) / 24,155; midgame 40 (32, 7, 1) / 20,141 / 849,507. The
  midgame is hexparrot's AI playing itself (lookahead 3), seed 0, after 17 turns. The
  longest line of pieces seen in any position was 6, so hexparrot's 6-square cap never
  mattered.
- `move_kinds_sim.py`, the evidence behind `TestRandomPlay`'s coverage check: 1,000 random
  games under our 200/800 limits (seeds 0–999, about 160 s) last 342 turns on average.
  Dwarf moves, troll steps and capturing steps occur in every game (169.6, 143.4 and 25.5
  per game); shoves in 89.1% (2.1 per game) and hurls in only 77.8% (1.4 per game). So ten
  games miss a kind with a chance of about 3 in 10 million, and none of the 100 blocks of
  ten consecutive games did.

`PLAN.md`'s pre-PR checklist now says that `thud/experiments/` stays out like the rest of
`thud/` (the push hook already refuses all of `thud/`), and that the upstream-bound files'
33 comments pointing into `thud/` must be rewritten to stand alone, with a `grep` that
lists them.

**Design review, first half (2026-09-24).** Each decision is recorded with its evidence in
`PLAN.md` Phase 3; in brief:

- **Position text: kept.** Chess prints FEN and reads it back (`chess.cc:381`); nine
  upstream games read positions from text; dstu prints the same unlabelled `d`/`T`/`O`
  rows. Reading must reject malformed text and impossible positions (more than 32 dwarfs
  or 8 trolls, `#` anywhere but exactly the cut-off corners, anything but the Thudstone on
  (7,7)). The agreed test mechanism — an error handler that throws — was **withdrawn** on
  finding `pyspiel.cc:829`: "When used from C++, OpenSpiel will never raise exceptions";
  Google style bans them too. Instead, as chess's `BoardFromFEN` returns nullopt for a bad
  FEN (`chess_board.h:265`, checked by `ChessState`'s constructor, `chess.cc:112`),
  `PositionFromText()` returns nullopt and `TestRejectedPositions` checks it directly: 12
  small edits of text that reads fine (8 changed or deleted characters, a deleted row, 3
  bad status lines), and controls proving that the unedited text reads correctly.
- **Notation: one form for every capture, `(r,c)-(r,c)x`** (the user's proposal).
  hexparrot's notation also marks every capture after the move (`TF7-D5xC4`), and
  OpenSpiel's `StringToAction` matches against the legal moves of the position, so the
  strings stay unique. A one-square capture is now a hurl or a capturing step depending on
  the piece: the tests' parser looks the piece up, and its position-free version refuses
  one-square captures. 35 hurl names were rewritten; all 24 one-square captures in the tests
  were checked by hand, and none changed meaning.
- **Observation: 6 planes** — dwarfs, trolls, empty, trolls to move, turns without a
  capture ÷ its limit, turns ÷ its limit — planes first, the same for both players, as in
  chess (`chess.cc:406`). The proposed Thudstone plane was dropped after the user asked why
  a stone that never moves needs one: by the rules it is simply a hole, and holes are 0 in
  every board plane, as Havannah and Y leave their off-board cells. The user then asked
  whether more planes could go, and to read the papers. AlphaGo (2016) had explicit
  "Stone colour: Player stone/opponent stone/empty" planes plus constant "Ones" and "Zeros"
  planes; AlphaGo Zero and AlphaZero dropped the empty plane ("0 if the intersection is
  empty, contains an opponent stone, or if t < 0") — possible only because Go and chess
  boards have no holes. On our grid, 61 squares are holes, so the empty plane stays: it is
  the only thing telling an empty square from a hole. `thud.h` explains this in a table
  (the user asked for that comment). AlphaZero also keeps a total-move-count plane for the
  same reason as our turns plane: games "exceeding a maximum number of steps … were
  terminated and assigned a drawn outcome".
- **Information state: the move history**, as `tic_tac_toe.cc:232`, `amazons.cc:359`,
  `chess.cc:397`, `checkers.cc:499` and `go.cc:129` do (Breakthrough provides none);
  `TestInformationState` covers it.

**Noted for later:** Part 4 must decide how a game started from a diagram is saved and
restored — chess writes `FEN: …` before its moves (`ChessState::Serialize`,
`ChessGame::DeserializeState`). And before training with OpenSpiel's Python AlphaZero, check
its input layout: it reshapes observations to the game's planes-first shape
(`model_linen.py:209`) and feeds flax's `nn.Conv`, which expects channels last (unverified;
in `PLAN.md`'s deferred decisions).

**Half-finished:** the design review's Parts 3 (board storage) and 4 (`thud.h` API); no
implementation yet; **nothing from this session is committed** — the user will review the
docs and commit and push next session.

**Next step:** as recorded in `## Current status`.

### 2026-09-24 — Session 4: board storage, a test audit, the `thud.h` interface

**Committed session 3** after the user read the docs: `e57b9ea8`, pushed to `origin/thud`.

**Design review, Part 3 — board storage: a padded 17x17 grid** (decision and evidence in
`PLAN.md` Phase 3). The user's questions shaped it:

- *What would a table of precomputed paths (the 165-square proposal) actually contain?*
  Positions only — the board's shape, computed once, never updated; the pieces live in a
  separate array. Nothing tracks lines of pieces.
- *Isn't a lookup slower than adding 16, 15, 14 or 1?* Yes: a small table stays in the
  fastest cache, but each step waits for its lookup, while an add takes one cycle. Fixed
  offsets need a rectangular grid, though — the actions' 165 numbering has rows of 5 to 15
  squares — and on an unpadded 15-wide grid they wrap (east of (5,14) is index +1 = (6,0)).
- *Isn't a bounds check on (row, col) as cheap as padding?* Yes. Padding doesn't add a
  lookup either — each step reads the cell anyway, and a padded cell answers "off the
  board" in that same read — so the choice is readability and safety: one condition stops
  every walk. Chess checks `InBoardArea` per step, Go pads (`go_board.h:50`), dstu keeps
  165 cells with neighbour tables, hexparrot pads.
- *Would dropping the padding help caching across cores?* No: boards are 165–289 bytes,
  OpenSpiel's MCTS keeps no boards in its tree (`mcts.h:114`) and copies one state per
  simulation, and each state's move history alone is 16 bytes per turn.
- *Track lines of dwarfs and trolls incrementally?* Postponed as premature optimisation; in
  `PLAN.md` Phase 5 as a candidate, checked against the simple implementation.

**A test audit, at the user's request.** A scratch script — kept as
`thud/experiments/crosscheck_tests.py` — reads every position and move check out of
`thud_test.cc` and recomputes it with hexparrot's engine: 27 exact move sets, 59 legal /
illegal checks, 12 reach tables, 2 complete legal sets and 9 after-move positions. All
agreed except two 7-square moves hexparrot cannot generate (its 6-square cap), confirmed by
hand. No "must not be legal" check started on the wrong side's piece, and the opening
diagram equals hexparrot's built-in start. Found and fixed:

- **Two diagrams with 12 trolls**, which the previous day's reading rules reject:
  `TestShoveBlockedAndThudstone`'s position (split into a Thudstone position with 7
  trolls and a blocked-lines position with 5, which also gained a "no shoves at all"
  check) and `TestDiagramRoundTrip`'s copy of it (now the 7-troll layout). The script now
  checks every diagram against the reading rules.
- Additions: the observation (every plane) and both strings checked in every position of
  the random games, for both players, including after the battle ends (`thud.h` now says
  the trolls-to-move plane then shows the side that would have moved);
  `TestInitialPosition` checks the dwarfs are to move and the battle isn't over; reading
  `turns_without_capture=200` gives a finished battle; and `"(7,4)-(4,4)"`'s comment now
  says it would be a hurl with N = 1, not a move.
- The script itself was checked with planted mistakes (a wrong expected shove, a ninth
  troll): it caught both and exited with status 1.

**Design review, Part 4 — the `thud.h` interface** (details in `PLAN.md`):

- `Cell::kOffBoard` for the grid's non-squares, as Go keeps `GoColor::kGuard`
  (`go_board.h:30`); states are built from a `Position`, with `NewInitialState(text)`
  reading through `PositionFromText()`.
- **Saving a game started from a diagram.** OpenSpiel's saved-game reader drops every line
  starting with `#` as a comment (`spiel.cc`), and 10 of the 15 board rows started with
  `#`. OpenSpiel's own starting-state mechanism (`starting_state_str_`, used by
  tic-tac-toe, connect four and catch with JSON) was examined and rejected:
  `State::StartingState()` always parses that string as JSON (`spiel.h:1257`) and is
  exposed to Python as `starting_state()`. So, as chess does with its FEN, `Serialize()`
  writes the starting position's text, then the moves; a game from the opening keeps the
  default. **The user proposed replacing `#` with another character; `-` was chosen**
  (light, not a piece letter, no special meaning; a space would be stripped by the reader,
  and `_` is dstu's empty square).
- The sweep: all 52 diagrams in `thud_test.cc` (3,120 characters, exactly 52 × 60
  cut-off cells), two test code lines, `thud.h`'s format comment and reading rule,
  `perft_reference.py`'s two diagrams and its diagram writer, `crosscheck_tests.py`'s
  reading rules, and `PLAN.md`. Session 3's log above still says `#`, as the record of
  that day's decision. Verified: no board `#` left in any file; all 110 cross-checks pass
  and no diagram breaks the rules; `perft_reference.py` reproduces every count, and its
  printed midgame equals `kMidgame`; a planted `#` is caught. New: `TestSerialize`, which
  pins the saved text and restores both kinds of game directly and through
  `SerializeGameAndState` / `DeserializeGameAndState`, and a rejection case for `#`.
- Not added: reading out a whole `Position`, and `UndoAction` (useful only to OpenSpiel's
  alpha-beta search, `use_undo` in `minimax.cc`; a Phase 5 candidate).

**State at the end:** 46 test functions; `make thud_test` builds with no warnings and stops
at the first stub; the design review is complete.

**Next step:** as recorded in `## Current status` — the implementation.

### 2026-09-25 — Session 5: the implementation

**Instructions from the user.** Work through all move types in one go, without pausing for
review (the plan's pause note is replaced). Check that all tests pass. **A test that fails
and seems to need changing must not simply be edited to pass:** note each such change and
review it with the user, so the changed test is truly correct rather than bent to match a
broken implementation. Now in `PLAN.md` Phase 3.

**Implemented `thud.cc`** as designed:

- Geometry computed once from `(row, col)`: square numbers and grid cells. The padded grid
  holds `Cell::kOffBoard` on its border and the cut-off corners, and each direction is a
  fixed grid offset (`kRowStep * 17 + kColStep`).
- One function per move type: `AddDwarfMoves` (walk empty squares; a hurl if the first
  non-empty square holds a troll within the length of the line behind the dwarf) and
  `AddTrollMoves` (a step to each empty neighbour; a capture step if dwarfs are next to it;
  shoves of 2..N squares along an empty path to a square next to a dwarf). `LegalActions`
  visits squares and directions in increasing order and appends the capture steps last,
  so its result is sorted without sorting.
- `IsTerminal` is cached in `terminal_`, recomputed after every move: a limit reached, or
  the side to move stuck — decided locally (a dwarf can move while it has an empty or troll
  neighbour, a troll while it has an empty neighbour), independent of the proof the
  random-play test uses. Piece counts are kept for the returns.
- `PositionFromText` implements every reading rule in `thud.h`, printing its reason on
  rejection; the opening is parsed once from a text constant. `Serialize` /
  `DeserializeState` save a diagram-started game as its position text, then its moves.
- `SetPosition` also checks a `Position` built without the reader (pieces within 32 and 8,
  the Thudstone only on (7,7), no `kOffBoard` squares), so returns stay in [-1, 1].
- `ThudGame` checks that both limits are positive. `Position::board` is now
  value-initialised.

**Results.** All 46 test functions passed on the first run (about 2 s), **with no change
to any test** — `thud_test.cc` is byte-identical to its committed version. The 13
rejection cases each printed the reader's reason. The **wrap-around check**, with the
files backed up: a 15-wide row stride (the plan's flat-array bug) plus a temporary
per-test harness showed 21 failing tests — all four wrap-around tests, the hurl, move,
shove and step tests near row ends, perft, symmetry and random play; both files were then
restored and compared byte for byte. The **full build** has 0 warnings, and **`ctest`
passes 284 of 285**: `api_test` and `games_sim_test` now play Thud and pass; the one
failure is `playthrough_test`, because Thud has no playthrough yet.

**The end-of-battle limits, checked the same way** (user's request, after committing the
implementation as `8a4e6ed5`), each removed on its own under the per-test harness:

- Without the no-capture limit, 2 tests fail: `TestEndNoCaptureLimit` and
  `TestEndDefaultNoCaptureLimit`. `TestRandomPlay` does not notice: none of its 10 random
  games goes 200 turns without a capture.
- Without the turn limit, 2 tests fail: `TestEndTurnLimit`, and `TestReturns`, whose
  official scoring example is a finished battle only because it is read with `turns=800`.

Both files were then restored from the commit, and all tests pass again.

**Phase 4, the same day.**

- **Playthrough:** `generate_new_playthrough.sh thud` wrote
  `open_spiel/integration_tests/playthroughs/thud.txt` (890 KB, 3 s): one seeded random game
  of 308 turns in which the trolls take the last dwarf, leaving 6 trolls — returns -0.75 /
  +0.75, current player -4 (terminal). The playthrough format prefixes state lines with
  `# `, so our rows show as `# -----dd.dd-----`. The full suite then passed 285 of 285.
- **`TestOpenSpielGenericTests`** (a new, 47th test function, added for the user's review):
  `LoadGameTest`, `NoChanceOutcomesTest`, and `RandomSimTest` with 10 games from the opening
  and 10 from a position read from text (`RandomSimTestWithSpecificInitialState`, which
  exercises saving diagram-started games). It passes; `thud_test` now takes 2.8 s. No
  `RandomSimTestWithUndo`, as Thud has no `UndoAction`.
- **The scripts' hexparrot clone:** re-running `crosscheck_tests.py` after the test change
  failed with "No module named 'thud.bitboard'" — the temporary folder, and with it the
  clone, had been cleared between sessions, and without it `import thud` finds this repo's
  own `thud/` folder (the repo is on `PYTHONPATH`). `load_hexparrot` now checks for the
  clone and prints how to make one. With a fresh clone: all 110 checks agree.

**The user's three questions, afterwards:**

- **Is `UndoAction` needed?** No. OpenSpiel's MCTS, in C++ and Python (AlphaZero builds
  on the latter), never replays from scratch: it copies the root state once per simulation
  (`mcts.cc:182`, `mcts.py:329`), so undo would replace one copy by many undos. Only
  alpha-beta's optional `use_undo` benefits; `PLAN.md`'s Phase 5 entry now says so.
- **Where the hexparrot clone lives:** moved from the proposed `~/hexparrot_thudgame` to
  `~/.cache/thud-openspiel/hexparrot_thudgame` (the XDG cache directory: a re-creatable
  download), now cloned there at the pinned commit. The location is defined once, in
  `perft_reference.py` (`DEFAULT_HEXPARROT`, honouring `$XDG_CACHE_HOME`), and all four
  scripts use it — including session 2's `limits_sim.py`, which added the path itself and
  now also checks for the clone first. `move_kinds_sim.py` and `limits_sim.py` check in the
  main process, since a worker that exits at start-up can hang a process pool. Verified:
  all four scripts run with no `--hexparrot`, perft reproduces every count, and a missing
  clone fails at once with the instructions.
- **Then moved again, to `~/.local/share/thud-openspiel/hexparrot_thudgame`** after the user
  asked whether `~/.cache` gets deleted. Nothing on this machine cleans it (no tmpfiles
  rule; the one cleanup timer is for `~/.launchpadlib`), but the XDG specification defines
  `$XDG_CACHE_HOME` for "user-specific non-essential data files" — what people and cleanup
  tools delete to free space — while `$XDG_DATA_HOME` (default `~/.local/share`) holds
  "user-specific data files". `DEFAULT_HEXPARROT` now honours `$XDG_DATA_HOME`. At the
  user's request the missing-clone message now says everything a future session needs:
  what hexparrot is and which scripts use it, that nothing in the build needs it, that it
  must live outside the repo, the default location, the pinned commit, the two restore
  commands, and how to check the result.
- **Could a faster move generator change the order and break tests?** No, as long as it is
  correct: OpenSpiel requires legal actions in ascending order (`spiel.h`), checked in
  every state by `RandomSimTest` and by the tests' `LegalMoves`, so every correct generator
  returns the same list. No explicit sort is needed now — ours generates that order by
  construction; `PLAN.md` records the requirement for future optimisations.

**Phase 5, first measurements, the same day.** No instrumentation was needed: OpenSpiel's
own programs time the game through its public API (the user's question about compiled-out
instrumentation, and the pattern agreed for later, are in `PLAN.md` Phase 5).

- **A Release build** in `build-release/` (ignored by git's `build*/`), only the two
  programs needed: `BUILD_TYPE=Release CXX=clang++ cmake -DPython3_EXECUTABLE=$(which
  python3) -DCMAKE_CXX_COMPILER=clang++ ../open_spiel && make -j10 benchmark_game
  mcts_example` — 2 min 34 s, 0 warnings. `BUILD_TYPE` is read from the environment
  (`open_spiel/CMakeLists.txt`); the default, "Testing", compiles at `-O2` with all checks,
  Release at `-O3` with `SPIEL_DCHECK` off.
- **Random games**, `./examples/benchmark_game --game=G --sims=N --attempts=5`, median of
  rounds 2-5, one core (every move computes the observation, lists the legal actions and
  applies one):

  | Game | Moves/s | Games/s | Moves per game |
  |---|---|---|---|
  | **thud** | **905,480** | **2,636.9** | 343 |
  | chess | 234,838 | 689.4 | 341 |
  | go (19x19) | 478,894 | 828.5 | 578 |
  | amazons | 3,107,885 | 14,926.4 | 208 |
  | breakthrough | 1,280,232 | 19,843.8 | 65 |
  | checkers | 338,711 | 4,966.1 | 68 |
  | clobber | 1,754,625 | 92,658.4 | 19 |
  | hex | 2,177,295 | 20,212.2 | 108 |

  Thud is about 3.9 times faster per move than chess, at almost the same random-game
  length — the user had expected them to be similar. Why: chess must discard every
  pseudo-legal move that leaves its king in check and handles castling, en passant,
  promotion and repeated positions, while Thud's moves are plain walks along lines with
  nothing to discard; the observations are of similar size (1,280 vs 1,350 numbers).
  Amazons' "moves" are thirds of a turn (queen, destination, arrow), so its numbers are not
  comparable. OpenSpiel's docs publish no such figures.
- **Release vs Testing:** Thud 905,480 vs 925,063 moves/s (no gain); chess 234,838 vs
  205,811 and Go 478,894 vs 413,399 (about +15%). So Thud's hot path is probably not
  compute-bound — a guess (for example, the fresh `std::vector` that every `LegalActions()`
  returns) that only a profiler can confirm.
- **MCTS**, `./examples/mcts_example --game=G --player1=mcts --player2=random
  --max_simulations=1000 --rollout_count=1 --verbose=true --seed=1`, the "sims/s" that
  `MCTSBot` reports for its first 10 moves: **Thud median 3,546 sims/s** (2,861-3,581),
  chess 757 (699-795).
- **How random games end**, new `thud/experiments/random_endings.py` (pyspiel, our
  implementation, 1,000 games, seeds 0-999, 2.6 s): 98.6% routs, 1.4% the no-capture limit,
  no turn-limit or stuck endings; length median 330, mean 343, max 668; longest stretch
  without a capture median 53, max 200. The trolls win every random game (mean margin
  -26.6), and every game's returns equal its margin / 32. Session 2's hexparrot figures
  were 1.5% cut short, median 334, mean 343 — the proxy the limits were chosen on held.

- **What this means for the deferred decisions** (input, nothing decided): classical MCTS
  is cheap here — a 10,000-simulation search per move takes about 3 s on one core, about
  0.3 s over 10 cores, before any optimisation. For AlphaZero-style learning each
  simulation costs a network evaluation rather than a random game, so the network, not the
  game, dominates — which shifts the weight to the CPU-vs-cloud-GPU decision, as this
  machine has no CUDA GPU. Recorded in `PLAN.md` Phase 5 and its deferred-decisions table.
- **The user asked whether that still holds with thousands of simulations per move.**
  Yes, per simulation — each one is one network evaluation plus one legal-move list and a
  few moves — but the claim was too sweeping, and was corrected in `PLAN.md`: a small
  network on a fast GPU with big batches can get near microseconds per position, and then
  the Python side becomes the bottleneck, still not the move generator. Measured through
  pyspiel (random games, seed 0): 294,930 moves/s for legal moves + apply (3.4 µs a move),
  18,819 moves/s once each move also fetches the observation into numpy (about 50 µs for
  the observation), and 10.9 µs for one `legal_actions()` call in the opening (656 moves) —
  against about 1.1 µs for a whole move in C++. OpenSpiel's Python AlphaZero fetches
  observations with `state.observation_tensor()` (`alpha_zero.py:246`). The AlphaZero paper
  confirms where the compute goes: "5,000 first-generation TPUs to generate self-play
  games and 64 second-generation TPUs to train the neural networks".
- **Concluded with the user: no further optimisation of the game logic for now** — in the
  status block and `PLAN.md` Phase 5, to revisit only if classical MCTS with random
  rollouts is chosen and proves too slow. `CLAUDE.md`'s build section now says that
  `build/` is the "Testing" build type and how to make the Release build for speed
  measurements.
- **MCTS against MCTS, at the user's request** ("is classical MCTS already in place? Then
  see whether the win ratio moves towards the dwarfs"): plain MCTS with random rollouts is
  entirely OpenSpiel's (`MCTSBot`, played by `mcts_example`); only MCTS with an evaluation
  function would need a Thud heuristic of our own. 10 games, one per core,
  `./examples/mcts_example --game=thud --player1=mcts --player2=mcts
  --max_simulations=1000 --rollout_count=1 --num_games=1 --seed=N` for N = 1..10 (Release
  build), 53 s in all. **The dwarfs won 7 of 10** (dwarf margins +5, +11, +6, +6, +10, +9,
  +11; the trolls' wins -4, -4, -8; mean +4.2), where random play gives the trolls every
  game. Every game ended in a rout — 7 with no trolls left, 3 with no dwarfs — in 137-336
  turns (random games: median 330); neither limit came into play.
- **Again at 5,000 simulations per move, 20 games** (user's request): the same command with
  `--max_simulations=5000`, seeds 1-20, 10 at a time through `xargs -P 10`, 400 s in all,
  91-292 s a game, 1.39 s a move (about 3,600 simulations/s). **The dwarfs won 19 of 20**
  (95% Wilson interval 76-99%, against 40-89% for the 7 of 10), mean dwarf margin +17.7
  (margins -8, +5, +7, +9, +12, +16, +17, +19 ×3, +20 ×2, +22 ×2, +24 ×3, +26, +27, +30).
  Games got much shorter: median 91 turns (47-248), against 246 at 1,000. 19 ended with no
  troll left, the dwarfs keeping a median 19.5 of their 32; the one troll win (seed 5) left
  2 trolls. Again only routs.
- **Why the dwarfs gain so much from more search** — measured by replaying all 30 games
  through pyspiel (new `thud/experiments/mcts_games.py`, which also produces the figures
  above from the logs; its docstring has the exact command, checked by running it at 10
  simulations):
  - Width: the dwarfs had a median 328 (1,000 sims) and 469 (5,000) legal moves a turn,
    the trolls 39 and 26. OpenSpiel's UCT gives an unvisited move infinite value
    (`mcts.cc:95`), so a search tries every move once before any twice: about 3 visits per
    dwarf move at 1,000 simulations, about 11 at 5,000. With a hurl on the board, the
    dwarfs took it 14% of the time at 1,000 and 57% at 5,000; they had one on 46% and 24%
    of their turns.
  - Rollouts: when a capture is available, it is a median 0.2-0.4% of the dwarfs' moves
    but about 20% of the trolls'. A random rollout hardly ever plays a hurl, so a troll
    left in a line of dwarfs is punished only through the tree, where the trolls' search
    must reach the one hurl among the dwarfs' ~450 replies; a dwarf left next to trolls is
    punished by the rollouts too. The trolls took an available capture only 26% and 28%
    of the time — probably because one dwarf is 1/32 of the margin, small next to rollout
    noise (a guess, not measured).
  - Conclusion recorded in `PLAN.md` Phase 5: these runs measure the searcher, not the
    balance of Thud (the result follows the search budget; the official match swaps sides
    anyway, `THUD_RULES.md` §9.5). Width limits plain UCT more than speed does, which an
    evaluation function alone would not fix; AlphaZero's PUCT gives unvisited moves their
    policy prior instead of infinity (`mcts.cc:103-111`; Python AlphaZero uses PUCT,
    `alpha_zero.py:201`). And OpenSpiel's AlphaZero evaluates itself against plain UCT
    with random rollouts, alternating sides (`alpha_zero.py:338-360`), so its evaluation
    must be read per side.
- **AlphaZero route: prepared, not started.** The user's direction (2026-09-25): if
  classical MCTS was not already in place, go the AlphaZero learning route, on this
  machine's CPU for now and on a cloud GPU once everything works. MCTS turned out to be in
  place, and the user asked for the 5,000-simulation run next, so the route is still to be
  confirmed. Groundwork: the venv has no JAX (`import jax` fails). For Python above 3.13
  OpenSpiel pins `jax==0.9.0.1 jaxlib==0.9.0.1 dm-haiku==0.0.16 optax==0.2.7 chex==0.1.91
  rlax==0.1.8 distrax==0.1.7 flax==0.12.3` (`open_spiel/scripts/python_extra_deps.sh:69-71`).
  Installing them waits for the user's go-ahead.

- **The AlphaZero route, steps 1-2** (the user confirmed it, and asked for a reminder of
  the steps: install JAX; control run on tic-tac-toe; settle the input-layout question;
  Thud smoke test; throughput on this CPU; a small real run; cloud GPU):
  - Installed OpenSpiel's pinned JAX set with
    `python3 -m pip install --upgrade jax==0.9.0.1 jaxlib==0.9.0.1 dm-haiku==0.0.16
    optax==0.2.7 chex==0.1.91 rlax==0.1.8 distrax==0.1.7 flax==0.12.3`, as OpenSpiel's CI
    does (`ci_script.sh:38`). A dry run first showed it would take numpy from 2.5.3 to
    2.3.5; the wheels' metadata shows why: `flax-0.12.3` requires `numpy<2.4.0` (jax needs
    `>=2.0`, scipy 1.18.1 `>=2.0.0,<2.8`, OpenSpiel's `requirements.txt` `>=1.21.5`).
    `pip check` is clean. JAX runs on the CPU (it warns about "an NVIDIA GPU", which does
    not exist, and falls back); pyspiel works with numpy 2.3.5 (Thud's initial observation
    reshapes to planes summing to 32 dwarfs, 8 trolls, 124 empty), and
    `ctest -R "pyspiel_test|playthrough|thud"` passes.
  - Control: OpenSpiel's `model_test.py` (6 passed, 2 skipped by upstream design: "Too
    slow for the CI", "May save to the disk") and `evaluator_test.py` (4 passed). The
    example, `python3 alpha_zero.py --path DIR` (tic-tac-toe, resnet 256 x 2, 2 actors, 2
    evaluators), ran until my 30-minute `timeout`: 17 of its 26 learning steps (each waits
    for 2,048 new states, `alpha_zero.py:389`). It learns: policy loss 1.53 -> 1.03,
    value loss 0.23 -> 0.03; self-play draws 42% -> 80%; against MCTS at 40 to 40,000
    simulations, from -0.44..-0.5 at every level to +0.42, +0.20, +0.10 at the three
    weakest and draws at the rest.
- **The input-layout question, settled: a real bug in OpenSpiel's Python AlphaZero.** The
  user challenged it ("wouldn't someone else have found it already? double- and
  triple-check"), so each way it could be wrong was checked:
  - A transpose somewhere on the way? None: `state.observation_tensor()`
    (`alpha_zero.py:246`) -> replay buffer (`:455`) -> reshape to
    `game.observation_tensor_shape()` (`:589`, `:144`; `model_linen.py:209`) -> `nn.Conv`.
    Built for Thud, the model's first kernel is `(3, 3, 15, 8)`: 15 input channels, the
    board's columns; the same `nn.Conv` on a `(15, 15, 6)` control input has 6. One dwarf
    at (7, 3) reaches output positions (plane 0-1, row 6-8): the windows slide over planes.
  - Planes last intended? No: OpenSpiel's API reference says planes first
    (`docs/api_reference/game_observation_tensor_shape.md:28`), so are the games its
    AlphaZero docs use (`tic_tac_toe.h:155`, `connect_four.h:198`), the docs call the
    resnet the AlphaGo Zero one (`docs/alpha_zero.md:55`), and the C++ AlphaZero reads
    planes first (`alpha_zero_torch/model.cc:39`, `:82`).
  - Known or intended? The TensorFlow model of 2020-03-02 (`6393dd33`) transposed
    planes-first input and the tic-tac-toe example asked for it
    (`data_format="channels_first"`); the rewrite `bcdb0b44` (2020-03-23) dropped the
    transpose (`tfkl.Reshape(input_shape)` into `tfkl.Conv2D(padding="same")`, Keras
    channels-last), and every version since, including the 2025 flax port (`model_nnx.py`
    too), has it. `gh search issues`/`prs` on google-deepmind/open_spiel for "channels
    first/last", "data_format", "NHWC", "permute_dimensions", "Conv2D alpha", "alpha zero
    observation shape" and more found nothing related.
  - Does it matter? New `thud/experiments/az_layout_check.py`: OpenSpiel's own model and
    update step, unmodified, trained on positions of random games (400 games for training,
    100 others for testing, every 7th turn so both sides appear — every 8th sampled only
    dwarf turns, caught by the per-side report). 6 runs in parallel, 545 s. Results in
    `PLAN.md` Phase 6: on the dwarfs' "is a hurl available?" 79.0% as is vs 98.0% planes
    last (baseline 66.9%); trolls 91.2% vs 99.4% (baseline 91.8%); dwarfs' policy mass on
    hurls 13.7% vs 68.1% (uniform 1.2%). Worst planes-last run beats best as-is run on
    every discriminating measure; planes last takes 1.7 times as long per step.
- **The user asked: is the C++ AlphaZero faster, did nobody serious use the Python one, why
  Python?** Python was the route only because `CLAUDE.md` said no LibTorch exists for
  aarch64. Findings:
  - OpenSpiel's docs recommend C++ for speed (`docs/alpha_zero.md:37-43`, quoted in
    `PLAN.md` Phase 6); its README warns it is an unmaintained user contribution with a
    pybind11 problem (issue #966).
  - New `thud/experiments/az_speed.py` (57 s): the Python search alone does 7,100-8,500
    simulations/s in this run (12,900 in an earlier one) at 656 and 253 dwarf moves; with
    AlphaZero's network evaluator, 12-24 simulations/s for resnets 32 x 2, 64 x 4 and
    128 x 6. The network per position: 26.9 / 41.2 / 52.3 ms as upstream calls it
    (`Model.inference`, never compiled), 0.68 / 1.50 / 3.59 ms compiled once, 0.07 / 0.18
    / 0.68 ms in compiled batches of 64.
  - LibTorch: `pip install --dry-run torch==2.10.0` offers
    `torch-2.10.0-cp314-cp314-manylinux_2_28_aarch64.whl`; downloaded (146 MB), it
    contains `TorchConfig.cmake`, `libtorch.so`, `libtorch_cpu.so`, `libc10.so` and the
    C++ API headers (9,788 header files); `torch/utils/__init__.py` defines
    `cmake_prefix_path` as `share/cmake`; `nm -D libc10.so` shows 240 `__cxx11` symbols.
    `CLAUDE.md` corrected.
- **Decided with the user: the C++ route, with a generous time limit** (`PLAN.md` Phase 6),
  and a standing note for the Python route: re-verify the layout findings, then one concise
  PR with experiments (tic-tac-toe, chess, Thud). Phase 5 marked done; the deferred
  decisions table updated.

- **The C++ AlphaZero builds, unpatched, and passes its tests** (commands now in
  `CLAUDE.md`, *Build and test*):
  - `torch==2.10.0` installed in the venv from the wheel downloaded for the inspection
    (`python3 -m pip install --find-links DIR torch==2.10.0`; it reports `2.10.0+cpu`;
    `pip check` clean). libnop cloned as `install.sh` does (its `master`, then
    `35e800d8`).
  - `build-torch/` configured Release with LibTorch and libnop, `CMAKE_PREFIX_PATH` from
    `torch.utils.cmake_prefix_path`. cmake found `libtorch.so` in the venv, warned
    "static library kineto_LIBRARY-NOTFOUND not found" (unneeded), and detected JAX
    0.9.0.1 and PyTorch for the Python tests — so the next cmake run in `build/` will add
    those tests too.
  - `make -j10` of the three LibTorch tests and both AlphaZero examples: 215 s, 0 errors,
    0 warnings, no change to upstream code. Issue #966, read beforehand, reports two
    problems, neither of which applies: link errors from an old-ABI LibTorch (ours is
    C++11-ABI), and a pybind11 clash when building `pyspiel` (not built in `build-torch/`).
  - `torch_integration_test` (prints nothing on success), `torch_model_test`, and
    `torch_vpnet_test` (97 s: trains the resnet on all 4,520 tic-tac-toe states and
    requires both losses below 0.1) pass. PyTorch 2.10 warns once that the `uint8`
    legal-moves mask is deprecated (`vpnet.cc:183`, `model.cc:210`).
- **Tic-tac-toe control passes** (`PLAN.md` Phase 6, with the table): the Python example's
  settings, 26 steps, 21 minutes; losses fall, self-play draws 41% → 83%, beats MCTS at
  40-400 simulations and draws the rest. 48 s a step against Python's 106 s, but
  training-bound. **The user asked whether the Python control was a corrected version:
  no** — upstream's, layout bug included, which on a 3x3 board probably costs little (not
  measured); the comparison shows only that the C++ pipeline learns. Clarified in
  `PLAN.md`.
- **The user asked for a fair C++ against Python comparison on Thud, with exactly the
  Python runs' parameters.** The Python Thud numbers come from a supervised test with no
  search (`az_layout_check.py`), so the fair C++ run is the same task with the C++ model;
  recorded as a Phase 6 item, with what cannot be matched without changing upstream code
  (initialisation; value loss halved in Python; different L2 terms).
- **Thud smoke test passes** (resnet 32 x 2, 50 simulations, 2 steps): games finish
  (110-296 moves), the learner trains, checkpoints are written, the evaluator plays, clean
  exit. The trolls won everything, whichever side AlphaZero took. It ran at 0.8 states/s.
- **Why so slow: LibTorch's threads oversubscribe the CPU.** Its OpenMP backend gives each
  caller 10 threads, and every actor and evaluator calls the network itself. A first
  measurement failed: `alpha_zero_torch_example --verbose` logs nothing per move, as the
  flag is never passed on (`alpha_zero.cc:112`, `:148`). Redone with
  `alpha_zero_torch_game_example --verbose`, which prints each search's speed: 1 search
  244 simulations/s by default, 163 with `OMP_NUM_THREADS=1`; 3 at once 16 each by
  default, 128 each with one thread. In the trainer: 0.8 → 3.0 and 6.4 states/s.
  `CLAUDE.md` now says to run LibTorch programs with `OMP_NUM_THREADS=1`.
- **Thud's returns are margins, and that suits AlphaZero.** I first called them win/draw/
  loss, which was wrong: they are the final margin over 32. The user explained the
  objective — a match is two battles with the sides swapped, won on the combined margin.
  Checked in the code: the value head learns the expected margin (real-valued target,
  squared error, `tanh`); only monitoring statistics count by sign. Recorded in
  `PLAN.md` Phase 6, *Margins as the value target*.
- **Match-aware play deferred** (user: the single-battle agent first). Maximising the
  expected margin maximises the expected match total, but not the chance of winning the
  match, where the second battle's play depends on the carried margin. The network would
  need the battle number and the carried margin as input; designs in *Deferred
  decisions*. KataGo's komi input and score utility verified in its paper.
- **The search: the user corrected me on `uct_c`.** I said margins might need a lower
  `uct_c`; the user pointed out that it works multiplicatively — right: it absorbs the
  values' scale. What it does not absorb is their level: OpenSpiel counts an untried move
  as 0, an even game (`mcts.cc:108`). From the formula, the side that is ahead then
  follows its prior below the root, and the side that is losing tries a new move with
  almost every simulation while its prior is flat. **Verified: AlphaZero counts untried
  moves as losses** (its pseudocode; Leela Chess Zero's blog; MuZero's official
  pseudocode likewise), so following the prior is AlphaZero's design, and the losing
  side's burst of breadth is OpenSpiel's own, mostly early. KataGo uses the parent's value
  minus a reduction. **User: measure first** (small run); a fix would be an upstream
  option, raised with the user first.
- **Settings to determine** (user asked): a list in `PLAN.md` Phase 6. Verified
  AlphaZero's root-noise rule (α inversely proportional to the move count; 0.3, 0.15, 0.03).
  Thud's sides would want about 0.025 and 0.3, but OpenSpiel uses one α: **α = 0.1 chosen
  by the user**, testing 0.03 and 0.3 recorded as follow-up. **Resignation off** (user
  agreed): on by default in upstream (80% of self-play games), off completely with
  `--cutoff_probability=0` (`alpha_zero.cc:202-203`).
- **New `thud/experiments/az_thud.flags`** (user: off by default, without upstream
  changes): our defaults, loaded with Abseil's `--flagfile`, each with its reasoning in a
  comment. Tested: the file's values reach `config.json`, the last value of a flag wins,
  `#` comments are ignored.
- Evaluation, from the code: OpenSpiel does not record which side AlphaZero played, but
  each evaluator log line pairs the game's returns with AlphaZero's, so the side can be
  recovered whenever the margin is not 0. Its default 7 levels reach 300,000 simulations a
  move against MCTS with random rollouts — use fewer on Thud.
- `CLAUDE.md` updated: LibTorch works from the wheel; the `OMP_NUM_THREADS=1` rule; the
  `build-torch/` commands and the kineto warning; the coming JAX and PyTorch tests.
- Committed and pushed after the user's review: a9c96748.

- **The C++ against Python layout control on Thud passes: the C++ model reads the board
  correctly** (`PLAN.md` Phase 6, with the table). Built as OpenSpiel's `docs/library.md`
  describes, with no CMake change (the user approved the route): `build-shared/`
  configured like `build-torch/` plus `BUILD_SHARED_LIB=ON`, `make open_spiel` in 136 s,
  0 warnings; `thud/experiments/build_az_layout_check.sh` (since renamed
  `build_az_program.sh`) compiles the new `az_layout_check.cc` with upstream's
  `model.cc` and `vpnet.cc` (the shared library
  leaves the LibTorch model out) using CMake's flags for them (33 s; it first missed
  `open_spiel/json/include`). `az_layout_check.py --export` (6 s) writes the games,
  checksums and batch orders. My refactor of its `positions()` was checked against the
  committed version: all 24,852 positions identical; the earlier Python logs, still in the
  scratchpad, show the same 20,036 + 4,816. The checksums first compared a float32 sum
  as an integer, inexact for the two fractional planes (1,022,859 against 1,022,834.531
  in float64): now float64 sums, compared to within 0.01; a tampered count and a tampered
  plane sum both stop the program. Seeds 1-3 in parallel with 3 threads each, 946 s. At
  step 1,500: dwarfs' hurl question 97.9% (Python planes last 98.0%, as is 79.0%),
  trolls' capture question 99.0% (99.4%, 91.2%), dwarfs' policy mass on hurls 77.9%
  (68.1%, 13.7%).

- **Throughput, mostly measured** (`PLAN.md` Phase 6, with the tables; the user put the
  laptop on mains power and "best performance"). The user asked for comparison numbers:
  the AlphaZero paper's (80,000 positions/s in chess on TPUs; 5,000 TPUs for self-play,
  800 simulations a move, 44 million chess games in 9 hours; checked in the paper) only
  show the scale, so chess and Connect Four were measured here alongside Thud. New
  `thud/experiments/az_throughput.cc` (modes search, learn, infer) and
  `az_throughput.sh` (sections 1-7); `build_az_layout_check.sh` became
  `build_az_program.sh PROGRAM`, which also compiles upstream's `vpevaluator.cc`. The
  first run, 60 measurements in 37 minutes, all succeeded. Findings: Thud's search is
  network-bound; a large fixed cost per network call (plausibly the 35.6 MB policy head)
  makes batching essential; batched inference gave 1,600 simulations/s on 64 x 4 (2.8x
  unbatched); no throttling. Two parts were not reliable (the 128 x 6 batched runs too
  short; Thud's unbatched scaling erratic), so a follow-up of sections 4 and 7 started,
  and was stopped when the user needed the laptop. `pkill -f` on a pattern that also
  occurs in its own command line killed my shell: use `pgrep -x`/`pkill -x` by name.

- **2026-09-26: throughput finished** (the user asked for the follow-up). Sections 4 and
  7 rerun, 22 runs, 82 minutes of wall time against my estimate of 20 — I first blamed
  searches overrunning their windows, but it was probably mostly the laptop sleeping (see
  below) — none failed. Batched inference with 32 searchers, batches of 32 and 2
  inference threads at `OMP_NUM_THREADS=4` is best for every size: 4,245 simulations/s
  on 32 x 2 (~7x unbatched), 1,690 on 64 x 4 (2.7x), ~350 on 128 x 6 (no gain: its
  convolutions dominate). All 10 cores on inference halves it. The unbatched scaling,
  repeated with longer windows, is smooth (Thud 64 x 4: 140, 267, 427, 620 for 1, 2, 4,
  10 searches); the first run's dips did not reproduce. At 100 simulations a move and
  ~250 moves a game: ~600 games an hour at 32 x 2, ~240 at 64 x 4, ~50 at 128 x 6 —
  this CPU suits small networks and small runs. `CLAUDE.md` and the flagfile's comment
  now say how to set `OMP_NUM_THREADS` with and without batching.

- **Trainer check and control** (the user asked for both, then a summary): 64 x 4, 100
  simulations, no evaluators, 15 minutes each — batched (32 actors, batches of 32, 2
  inference threads, `OMP_NUM_THREADS=4`) and unbatched (10 actors, one thread). Speed
  read from the actors' logs, as the learner only sees whole games in bursts (my first
  plan, reading it from the learner's steps, would have measured nothing for 8 minutes;
  I told the user it would take ~30 minutes, not 18). **The laptop slept during both**:
  `/proc/uptime` was ~11 hours behind the wall clock, the batched run took 55 minutes of
  wall time for 15 of its own, and `powercfg` shows why — on mains power it sleeps after
  5 minutes without input (3 on battery; hibernation never on mains). The pause-free
  games still agree with the benchmark: batched ~1,760 simulations/s (6 games, 0.536-0.566
  moves/s per actor; benchmark 1,690), unbatched ~640 (1 game; benchmark 620). Also: the
  learner needed ~3.5-4 s per 1,024 positions alongside the actors; the batched trainer
  kept only half to two thirds of the CPU busy; stopping the trainer waits for all
  running games, so both runs were killed after my 60 s grace.
- **At the user's request, one consolidated analysis** of all the throughput work, by
  topic rather than by day: `PLAN.md` Phase 6, throughput — how it was measured, where
  the time goes, using all cores, the learner, the trainer, what it means per hour, the
  pitfalls met, and what is still open. `CLAUDE.md` now notes the sleep timer.

- **The clean trainer check** (the user turned sleep off on mains power, verified with
  `powercfg`: sleep and hibernation never while plugged in, the laptop on mains power):
  three 30-minute runs with a pause detector (wall clock against `/proc/uptime` every 10
  s: 539 samples over 90 minutes, no pause, largest drift 0.04 s). 64 x 4 batched: 1,707
  simulations/s (benchmark 1,690); 64 x 4 unbatched: 513 (benchmark 620; the learner's
  single thread takes a core); 32 x 2 batched: 4,259 (benchmark 4,245). The benchmark
  carries over, learner and all. At the user's request, all the throughput work is now
  one final report in `PLAN.md` Phase 6 (*Throughput on this CPU — final report*).

- **Decisions with the user after the throughput report** (all in `PLAN.md` Phase 6, now
  with a **roadmap and decision log** at its top, made because the user was losing track):
  64 x 4 to start (32 x 2's convolutions see only 11 x 11 squares); first runs 100 against
  400 simulations at equal machine time, decided head-to-head in pairs of battles with the
  sides swapped; our own copy of the C++ AlphaZero and its MCTS, for changes inside the
  search and trainer; the order: baseline learning on unmodified OpenSpiel first, then the
  copy proven identical, then one change at a time (playout caps with per-side settings,
  tree reuse with one shared tree, the value of untried moves), settings, the
  convolutional policy head later (stage 5: the one change that makes our networks
  unloadable by unmodified OpenSpiel — a Thud game is far easier to get accepted upstream
  than a new network type), then a cloud GPU. Replay buffer stays at the default; notes
  for later on growing the network (KataGo's recipe works without upstream changes) and
  on when to enlarge the buffer. Evidence checked on the way: AlphaZero's network (19
  residual blocks of 256 filters), KataGo's growth and playout caps (1.37x), the Gumbel
  paper's warning about unvisited root moves, AlphaGo Zero's tree reuse, AlphaGo's
  one-position-per-game value data.
- **Tree-reuse potential** (new `thud/experiments/az_reuse.cc`, 45 minutes): a side's own
  tree two plies later would inherit ~0% — only a tree shared by both sides is worth it.
  Its shared-tree numbers are not usable for self-play: to cover all phases quickly it
  started games after up to 200 random moves, and there the searches behave unlike
  self-play's (the dwarfs concentrated their visits). Default now no prefix (source
  changed; `build-shared/az_reuse` still to rebuild).
- **The dwarfs' searches are pure breadth at 100 simulations** (new
  `thud/experiments/az_buffer_stats.cc`, reading the trainer's saved replay buffer; the
  clean 64 x 4 run): a median 99 of 100 simulations on different moves, the most visited
  move 1% of the visits; the trolls' searches 5 moves, 44%. As the untried-move analysis
  predicted. The same numbers give the real tree-reuse potential: the dwarfs would
  inherit a median 44% of the trolls' visits, the trolls ~1% of the dwarfs'.
- **Match program** (new `thud/experiments/az_match.cc`): pairs of battles from the same
  random opening with sides swapped, deterministic searches, one batched evaluator per
  network. Controls pass: a network against itself gives exactly 0 for every pair; two
  different networks do not (after 11 learning steps against its own start, 6 pairs at 10
  simulations: −6.3 points a pair, too little to mean anything).
- `LoadBuffer` refuses a replay buffer saved with another maximum size, so enlarging the
  buffer on resume needs the file rewritten first (noted in `PLAN.md`).
- **The laptop paused ~18 minutes between 15:56 and 19:02** although sleep on mains power
  is off (`/proc/uptime` against the wall clock); the lid action cannot be read with
  `powercfg` here — closing the lid is the likely cause. Runs just pause and resume; the
  pause detectors record it.
- **Stage 1, run A started 2026-09-26 19:49** (the user away for 4-6 hours, asking for
  easy experiments): `~/thud-runs/stage1_sims100/` (outside the repo), 64 x 4, 100
  simulations, batched, our flagfile, one evaluator at 3 levels, every checkpoint kept,
  6 hours of machine time (`timeout` counts only running time), pause detector
  (`clock.log`) and a watcher recording the buffer statistics after every learning step
  (`buffer_stats.jsonl`). Script: `~/thud-runs/stage1_sims100.sh`.

- **Overnight 2026-09-26/27** (the user asleep, long experiments allowed until ~9:00):
  run A finished at 01:55 (16 learning steps, no pause). It learned — its final network
  beat its untrained start by +3.9 points a pair (95% interval +1.5 to +6.3), its
  evaluation against MCTS improved in every cell — but then **regressed**: the step-8
  network beat the final one by 5.1 points a pair (+1.8 to +8.4), in step with swings in
  the self-play results. Details in `PLAN.md` Phase 6, *First training runs*. My first
  evaluation-by-side count put 30 of 33 games on the trolls' side: the log rounds
  AlphaZero's return to 2 decimals, so exact matching failed; with a tolerance the sides
  split 34/34. Run B (400 simulations) started at 02:30 in `~/thud-runs/stage1_sims400/`.
  Because the networks rise and fall, the plan after run B changed (I stopped
  `stage1_overnight.sh` once run B was running; `stage1_after_b.sh` took over): A final
  against B final, then B final against the common anchor (run A's untrained start), then
  A step 8 against it; the 400-simulation match moved to later. Progress in
  `~/thud-runs/stage1_matches/progress.log`.

- **Run B finished at 08:30** (4 learning steps, no pause), the matches at 09:17
  (`A_step8_vs_A_start` skipped: it was past 09:10). **At equal machine time 100
  simulations beat 400, early in training**: A final beat B final by +2.9 points a pair
  (95% interval +1.5 to +4.2; 15 pairs won, 14 drawn, 1 lost); against the anchor A final
  +3.9 (+1.5 to +6.3), B final +1.9 (−4.2 to +8.1). Caveats in `PLAN.md` Phase 6: matches
  at 100 simulations, run B learned only 4 times, 6 hours is early. The dwarfs lost by
  19-32 points in every match; the dwarfs' searches were broad at 400 simulations too
  (220 of 400 on different moves) once the network judged them lost.

- **2026-09-27, with the user in the morning** (the user had lost track of some terms, so
  they were explained again: stage 1's gate, the buffer-size trigger, the untried-move
  fix, match noise). Decided: 100 simulations for now (revisit with playout caps); the
  untried-move fix first in stage 3, as urgent (user: as the trolls improve, the dwarfs'
  targets stay flat); the replay buffer to be increased (user: also needed to grow the
  network and train other heads later) — tested 2x first as run A2 — and every position
  archived from A2 on. The user noted that newer networks losing is not unusual (seen in
  AlphaZero for Go); AlphaGo Zero indeed gated new networks ("a margin of 55%"),
  AlphaZero did not. To fit the user's 4.5 hours away, the learning curve was cut to
  steps 8, 12, 14 and run A2 to 4.5 hours, compared with A at equal positions.
- **Run A's learning curve** (`stage1_curve_then_a2.sh`, 10:50-12:17): against the anchor
  step 8 +18.9, step 12 +23.1, step 14 +26.1 (each 20 of 20 pairs), step 16 +3.9. I first
  suspected the "final" checkpoint (`-1`) was not the last one; the code saves it at every
  step (`alpha_zero.cc:428-432`) and its file time equals step 16's, so the measurements
  stand: **learning works, then the dwarfs' play collapsed at steps 15-16** (as dwarfs +0.1
  at step 14, −24.9 at 16), likely the loop of the dwarfs' breadth searches. The
  morning's "regressed after about step 8" was wrong: it fell only at the end. The user
  confirmed: stage 1's gate is met; the collapse is stage 3a's first problem. Run A2
  started at 12:17. The user asked whether to give the dwarfs more simulations: yes, but
  after 3a (until then extra simulations only widen a losing side's search), within 3b,
  with the self-play tilt it causes measured (`PLAN.md` roadmap, 3b).
- **Consistency check of the docs** (user: always wanted; now in `CLAUDE.md`, *Session
  end*, and in memory): fixed superseded statements in `PLAN.md` (stage 1 marked done,
  "regressed after about step 8", "gate only half met", "measure it in the small run",
  the simulations setting, the untried-move entry, the settings summary, the breadth
  paragraph) and the status block here (stage 1's result, the experiment programs, the
  next steps, whose numbering had jumped).

- **Committed and pushed** the stage-1 work after the consistency check: a40f2a0b.
- **Stage 2 started** (the user: without taking much CPU from run A2): new `thud/az/`.
  `import_from_upstream.py` copies OpenSpiel's C++ AlphaZero (trainer, network,
  evaluator, device manager, the example's `main` as `az_trainer.cc`) and its MCTS from
  upstream commit 540bba6e, renames the namespace to `open_spiel::thud_az`, points the
  includes at the copies, renames the header guards and adds a change notice; its own
  assertions caught my notice naming the upstream path (fixed by checking first).
  `build.sh` builds on the shared-library route at `nice 19`; the first link failed —
  `libopen_spiel.so` does not re-export Abseil's flag parsing — fixed by linking
  Abseil's static libraries after it as one group. New `identity_check.cc` compares
  upstream's code and ours on run A's step 14 (step 8 as the control), single-threaded:
  equal network outputs on 50 positions, equal searches without and with root noise (50
  of 50), a checkpoint of ours loads in upstream's code with equal outputs, one learning
  step gives equal losses and outputs; the controls (another network, another noise
  seed, before against after learning) all differ. 380 s at `nice 19`. The user asked
  whether the tic-tac-toe control is the only remaining guard: the gate had four items,
  and the question showed two gaps in my checks, both closed: `import_from_upstream.py
  --check` proves the copies are textually a fresh import (a one-character change is
  caught), covering code no test reaches; and the identity check now also compares the
  evaluation games' search (UCT, random rollouts, the solver; 50 of 50 equal, another
  seed changes all 50). Remaining: the tic-tac-toe control through `az_trainer`, queued
  after run A2 (`~/thud-runs/stage2_ttt_control.sh`). Not yet committed.

- **Run A2 finished at 16:47; the tic-tac-toe control through our copy (16:47-17:06)
  learned as upstream's did — stage 2 is done** (numbers in `PLAN.md`'s roadmap).

- **Stage 3a, with the user in the evening.** Decided: three rules for untried moves
  behind a flag, defaulting to the siblings' mean minus 0.2 × √(their prior mass), at
  every node; test against upstream too if quick (it was: 6 minutes). Implemented in
  `thud/az/mcts.{h,cc}` (the rule, `UntriedValue`, a `PUCTValue` overload; upstream's
  `PUCTValue` kept), `alpha_zero.{h,cc}` (config, `config.json` — a run from before the
  setting resumes with upstream's rule) and `az_trainer.cc` (flags). `--check` now lists
  exactly those five files. New `untried_move_check.cc`: the formula on a hand-built node
  passes; on self-play positions of run A's step 14 the dwarfs' searches visit 99 moves
  (upstream), 31 (default), 1 (`loss`, no root noise). My first version sampled every 8th
  move and so only dwarf positions (the dwarfs move on even moves — the same mistake as
  in the first layout check); now every 7th, with a control that both sides appear.
  `identity_check.cc` passes with `upstream` explicit, plus a control that the default
  changes the searches. `az_match.cc` moved onto our copy with `untried=`; with
  upstream's rule it reproduced 4 of 4 earlier pairs exactly. Two bugs caught in the
  night script before it ran: the pause-detector helper would have hung its `$(...)`,
  and `$?` after `$(date)` again. Run C started at 20:31.
- **Committed in two parts** (the user asked to commit): stage 2 as the pure import
  with its scripts and the stage-2 identity check (4acaee09), then stage 3a on top, so
  `git diff 4acaee09 -- thud/az` shows exactly what stage 3a changed in upstream's
  code (6 files, +136 −9).

- **The night of 2026-09-27/28** (`~/thud-runs/night_2026-09-27.sh`, no pause in either
  run): **run C** (the new rule, run A's settings) 20:31-02:32, 16 learning steps, 352,553
  positions. Its dwarf searches visited a median 19 moves at every step (run A: 95-99).
  Against the anchor: step 8 +10.4, 12 +43.5, 14 +40.1, 16 +46.4 — no collapse — and at
  step 16 +21.6 as dwarfs (run A's step 16: −24.9). Head-to-head C step 16 against A step
  16 +20.9 a pair (20 of 20 pairs); against A's best (step 14) −1.2 (−6.6 to +4.1) with
  upstream's rule in the search, +3.9 (+0.1 to +7.6) with the new one. **Run A2 resumed**
  04:35-07:16 with upstream's trainer: continued at step 6 with its model and full buffer
  — resuming works — to step 8; against the anchor step 5 +9.4, step 8 +16.6 (run A at
  equal positions: ~+19 to +23 before its collapse, +3.9 after). Conclusions and caveats
  (one run each; the anchor stops ranking strong networks) in `PLAN.md`. My results loop
  missed A2's step 8 (a name list); read from its file.

- **The user asked how sure we are, whether a longer run is needed, about the buffer and
  about tree reuse (2026-09-28 morning).** Match intervals had used the normal factor
  1.96; with 20 pairs Student's t (2.09) is right: C step 16 against A's best with the new
  rule is +3.9 (−0.2 to +7.9), not significant (reported as +0.1 to +7.6). `az_match` now
  uses t. Certain: the rule narrows the dwarfs' searches, and C's final far outplays A's
  final; not shown: C beats A's best, nor — with one run per rule — that the rule trains
  better. Proposed: resume run C ~6 hours (stability, and resuming in our copy). Buffer:
  left as is; retest when learning stalls and before training a bigger network from a
  smaller one's games (user). Tree reuse: with the new rule the move played holds ~13-17%
  of the visits for both sides (run C's buffer statistics) — roughly 15% more
  simulations per search, modest.

- **Later that morning**: the user asked how long C against A's best takes (20 pairs:
  12 minutes with upstream's rule in the search, 27 with the new one; ~100 pairs to see
  a 2-3 point difference), agreed to resume run C with most cores while working, asked
  for more buffer triggers — with caveats, since some happen in healthy training (a newer
  network losing to an older one): now in `PLAN.md` as event triggers and signal
  triggers, the latter counting only when pronounced and persistent against a fixed
  reference. Tree reuse: the user expects its gain to grow as searches focus, and noted
  that with per-side budgets the dwarfs' extra simulations also serve the trolls, and
  with equal budgets the more focused side helps the other — all in `PLAN.md`. Its
  priority stays after 3b (~15% more simulations today against playout caps' 1.37x; the
  larger change), to move up if the share passes ~30% or when 3b is done. Run C resumed
  at 12:27.

- **Afternoon**: resuming works in our copy (step 17 at 12:53 continued from 352,553
  positions with the full buffer); KataGo's quick searches turn root noise off (checked
  at the source); a proposal for stage 3b written into `PLAN.md` (f2833b5d).

- **Evening (2026-09-28).** Run C's continuation paused ~2 hours (15:04-17:01, pause
  detector: the laptop was on battery, 72%), so training ends ~20:25; its 6 hours are
  machine time. By step 23 the dwarfs' searches visit a median 17 of 156 legal moves, the
  trolls' 14 of 18-22, and the most visited move holds 18-20% for both (13-17% through
  step 16). The user asked whether, with the dwarfs' searches now about as focused as the
  trolls', per-side simulation numbers are still needed. Mostly not for the original
  reason (upstream's rule spread the dwarfs' searches over every move), but focus is not
  search quality: the dwarfs' searches cover ~11% of their legal moves, the trolls'
  ~64%, so a good dwarf move the network overlooks is rarely found. Recommended, and put
  into `PLAN.md` at the user's request: settings per side in the code, equal to start
  (`p` 0.25: ~1.75x run C's simulations, not 2.1x), and an **uneven match** to decide —
  C's final network at 400 simulations against itself at 100, then 100 against 100 on
  the same openings. `az_match` got `sims_a` and `sims_b` (default: `sims`);
  `thud/experiments/az_sims_gain.py` (new, with a self-test) turns the two matches into
  each side's gain with t intervals. The night script `~/thud-runs/night_2026-09-28.sh`
  plays both after run C (100 openings, ~8 hours). Controls: with the default and with
  explicit equal settings the new `az_match` plays exactly the old binary's games (2
  pairs); the 400/100 match mirrored when the networks swap is still to run. The user
  asked to rename `sims_a`/`sims_b` to `sims_dwarfs`/`sims_trolls`; per-side budgets
  would need a third match (with one network on both sides, a pair's second battle
  replays its first, so 400/100 and 100/400 become separate matches: ~13.5 hours instead
  of ~8), and the user withdrew the request: the budgets stay per network. At 18:28 the
  user needed the computer for the evening: run C stopped after step 25 (4:04 of its 6
  hours of machine time used; step 26's self-play lost), and everything waits for the
  user's green light — `~/thud-runs/stage3a_resume2.sh` then trains the remaining 6,960
  s, plays run C's matches and the uneven match. The user: experiments
  whose timing does not matter may run past 9am (saved as a memory; equal-time
  comparisons still should not). Corrected in `PLAN.md`: the dwarfs have 4-9x the
  trolls' legal moves in run C's self-play, not ~10x; ~300 moves was stale; the trolls'
  floor of ~100 simulations was upstream's rule's. In the status block: `az_match` is
  built by `thud/az/build.sh`, and resuming in our copy is done.

- **2026-09-29 morning.** The user forgot the green light, so nothing ran overnight;
  asked which experiments were planned and how long each takes (the check ~20 minutes,
  training ~2 hours, run C's matches ~2 hours, the uneven match ~8 hours), then for the
  check and the training only. `stage3a_resume2.sh` now trains only, at
  `OMP_NUM_THREADS=3` while the user works; run C's matches and the uneven match moved
  to `stage3a_matches2.sh`, to start on the user's word. The last control passed (C
  step 16 at 400 against A step 14 at 100, 2 pairs, and the same with the networks
  swapped: identical games mirrored, margins and lengths; they differ from the 100/100
  games); training started at 08:12 and ended at 10:08 after step 29, unpaused. Its
  final buffer (steps 28-30) is archived too. Asked which experiments were left, the
  user chose run C's matches now (started 10:25), the uneven match later, and dropped
  the 400-simulation match (A final against B final searching at 400) for the time
  being — playout caps revisit the budget anyway. C step 29 against A step 14 lost
  −11.5 a pair (95 of 100 pairs; its dwarfs −20.0 against step 16's −2.4); the user
  put the same match with the new rule in both searches into the backlog, and asked
  whether dwarf play is simply harder to learn (the trolls start in lines of three and
  can shove at once; the dwarfs must line up first). It fits the trolls' faster start
  and the self-play results, not a decline against a fixed opponent; supporting: the
  dwarfs' 4-9x moves, run A's trolls focusing first, plain MCTS dwarfs needing deep
  searches. The other matches (done 13:54): against C step 16 +1.1 (even), against the
  anchor +21.2 (step 16: +46.4; as dwarfs −4.5 against +21.6). Its dwarfs beat step
  16's trolls but lose to the upstream-trained networks — not transitive; forgetting or
  the evaluation's rule, unresolved (`PLAN.md`, *First training runs*). The user asked
  for the cheap check: under upstream's rule the dwarfs' searches spread for both step
  16 (a median 99 moves) and step 29 (76), so the drop is not step 29 spreading where 16
  did not; with every move visited about once, both pick the dwarfs' move by a single
  evaluation (`CompareFinal`), so these matches test the value head, not the trained
  play. Its control "upstream's rule spreads the dwarfs' searches (90+)" fails for step
  29 (76): written for run A, a result here, not a fault.

- **Afternoon and evening: a step back.** The user felt we were getting lost in small
  experiments. Listed: six defined experiments and five stages, more than 10 nights of
  machine time. Proposed (open): on this CPU only "does it work" questions, "which
  setting is best" on the GPU. Inventory: beyond the game and OpenSpiel's training, one
  change to the AI (the untried-move rule, ~110 lines; 7 of the 12 copied files still
  byte-identical to upstream) plus two settings; everything else measures. The user,
  mostly convinced the rule works, wants the dwarf collapse solved: the plan is Step 1
  (every network as trained) and Step 2 (the buffer) only if needed; the uneven match
  parked. `az_match` got `untried_a`/`untried_b` and `progress=` (controls passed:
  default games identical with and without the reports, mixed rules mirror when the
  networks swap, the rule takes effect). Tuning, 18:11-19:00, other programs ~1%: 1,315
  → 1,923 simulations/s with 64 threads and batch 32; 1,720 for 3 matches at once with
  OMP 1; whole matches had averaged ~1,050, their tails idle. `step1_as_trained.sh` is
  ready (~3.5 hours, the decisive answer after ~1.3), waiting for the user's word.

- **Evening: tree reuse re-read and postponed.** Step 1 started at 19:21. The user asked
  about working on tree reuse in parallel, then asked for the sources to be re-read:
  AlphaGo Zero reused the tree in self-play (Methods, *Play*); AlphaZero's pseudocode,
  KataGo (its self-play clears the tree "to make sure root noise is effective") and
  Leela Chess Zero do not; Leela Zero has an open issue where reused visits swamp the
  noise, Leela Chess Zero a closed one where a reused root got no noise at all. Checked
  in our code on the user's questions: the evaluation cache (keyed by the position with
  both counters, 262,144 entries, cleared every learning step) is hit 52-53%, much of it
  a node's second request; a larger cache would gain little. The trainer's 10 MB tree
  limit prunes rather than stops and matters only above ~850 simulations. The user
  postponed tree reuse; the plan records why, how to build it and how to test it.

- **Night: Step 1's results and the cache counters.** Step 1 finished at 23:25 (results
  in the status block and `PLAN.md`): no collapse, but C drifted from step 16 on — it
  beats step 16, loses to A step 14 (−8.5) and its dwarfs lost 10 points against the
  anchor; Step 2 indicated. On the user's request the evaluator (our copy) now counts
  value and move-probability requests with their cache hits, the trainer logs them, and
  `az_match` has `cache=` and `share=`. Controls: default games unchanged, the cache off
  gives the same games and no hits, a shared evaluator the same games as two; C step 29
  against itself: 39% of value requests cached with two evaluators, 69% with one shared.
  A first build of the identity check lacked upstream's sources (the header has the
  command); a server-side outage of the command check stalled the shell for a while.
  After Step 1: the identity check, then `~/thud-runs/tune_cache.sh` (the cache on, off
  and 4x, ~35 minutes).

- **After midnight: the cache comparison.** The identity check passed (12 of 12). The
  cache makes self-play-like search 2.6-2.7x faster (3,275-3,538 against 1,301
  simulations/s without it); a shared evaluator beats two separate ones (2,105). A 4x
  cache first seemed +42%, but that was an artifact — a network playing itself replays
  each pair's first battle, and the larger cache remembered it; before any replay it
  gave no gain. Details in `PLAN.md`, *First training runs*, the match program. With the
  counters, `import_from_upstream.py --check` lists 5 of 12 copies as a fresh import
  (differing: mcts, alpha_zero, vpevaluator — `.h` and `.cc` each — and az_trainer.cc).

- **2026-09-30, early morning: the forgetting check and Step 2.** The user explained
  how the buffer would be enlarged (options A, B and C explained: upstream ties the
  learning cadence and the training per step to the buffer size), asked for the quick
  check and gave the green light for option C if warranted; its new setting is to be revisited later (in the
  plan). `az_forgetting` (C's networks on C's archives, 3.5 minutes): values sound, the
  dwarfs' move probabilities drifting, mostly between steps 24 and 29 — warranted. Our
  copy's trainer got `learner_batches` (0 = upstream; it logs the batches per step);
  `az_merge_buffers` built run D's 262,144-position buffer from C's archives of steps
  4-15 and checked it (all distinct, exactly the archived positions). The archives miss
  ~0.6% of positions (copied a few hundred late each time). Run D started at 02:57.

- **2026-09-30 morning.** The laptop, unplugged by accident, ran out of battery: Windows
  suspended it (WSL up since 2026-09-26, every process of run D alive), the pause
  detector shows one freeze of 11,282 s (07:40-10:48), and run D continued at step 25.
  The trainer's new request counts corrected the cache argument: in self-play only
  13-22% of value requests hit (the noise-free match: 40-45%), about the share tree reuse
  would inherit — so each search spends ~a fifth of its simulations re-traversing what
  the previous one explored; the cache spares the network calls, not the simulations. My
  earlier claim that the cache recovers most of reuse's saving was wrong; corrected in
  `PLAN.md` (the tree-reuse entry, its roadmap row, the match program), with a new
  trigger to revisit reuse (the dwarfs' most visited move above ~30%). The user asked
  whether this makes reuse more promising: somewhat (its gain is the whole inherited
  share, and grows as the policy sharpens), but the value of 20-30% more simulations is
  unmeasured and the risk to root noise — the dwarfs' exploration — unchanged, so it
  stays postponed.

- **2026-09-30 afternoon.** Run D had not advanced since 10:54: its process tree died
  with the old Claude Code session (the job's output reads "[killed]"; no out-of-memory
  kill). Resumed from step 25 at 14:43 with `setsid nohup`, so a session change can no
  longer stop it; the lesson is in `CLAUDE.md` (Environment).

- **2026-09-30 evening: Step 2's results.** Run D finished training at ~16:40 and its
  evaluation at 19:44, ahead of the estimate: D29 beats C29 (+7.3) and C16 (+8.4), loses
  to A14 (−5.4, C29 −8.5); its dwarfs against the anchor +11.7 / +8.1 / +3.2 at steps 20,
  24, 29; the forgetting check shows no policy narrowing (D29's dwarf loss on old
  archives at C16's level). Forgetting was part of the cause.

- **2026-09-30 late evening and 2026-10-01 morning.** The user asked whether anything
  speaks against an even bigger buffer: not the training time (with `learner_batches`,
  64 batches whatever the size; saving 2.5 GB took ~16 s a step), but memory (WSL's 15
  GB: ~7x at most with the buffer watcher's copy) and staleness; KataGo grows its window
  sublinearly with the data (250,000 to 22 million after ~225 million positions; by its
  rule ~4.3x our former 65,536 at C's step 15, ~5.9x at step 29, ~7.5x after a million),
  so run C's forgetting with a quarter of KataGo's smallest window is unsurprising. Sizes
  need not be powers of two (6x = 393,216 with reuse 18, 7x = 458,752 with 21). Asked
  about 7x: slower uptake of fresh games (the newest step's share of each update 4.8%,
  at 4x 8.3%), slightly stale until ~900,000 positions, ~11.6 GB peak (no full `make
  -j10` meanwhile), pre-filling from run D needs `az_merge_buffers` to skip duplicates
  (D's saved buffers overlap), and it must not change together with playout caps in one
  comparison. In `CLAUDE.md`: a session-start check for running experiments (tested with
  a dummy process), and the user and Claude call each other "bro" in sessions (the docs
  keep "the user").

- **2026-10-01.** The user agreed to a 7x memory as the common baseline of the playout-caps
  comparison, rather than testing 7x against 4x (a tuning question for the GPU).
  `az_merge_buffers` now skips positions already added (controls: the same archive twice
  skips the whole second copy, plus 45 exact repeats inside it; D's archive 27 and final
  buffer overlap by 218,061 positions). Run E's 7x buffer: the newest 458,752 positions
  of D's lineage (C's steps 7-15, D's 16-29), all distinct; kept as
  `replay_buffer_start.data` for run F. Run E started at 13:17, detached, the trainer at
  normal priority (other work at nice 19, a load monitor logging the CPU the rest takes),
  for 6 hours of machine time (`~/thud-runs/stage3b_E.sh`).

- **2026-10-01 afternoon and evening.** Playout caps would cut the recorded positions per
  hour ~7x (1.75x costlier moves, a quarter of them recorded), so run F would get ~2
  learning steps in 6 hours against E's ~14; the user chose the cheap hint first,
  `az_target_quality` (smoke-tested; it refuses an even sampling step, which would
  sample only the dwarfs). The laptop slept on battery at ~15:12 and WSL restarted on
  waking (Windows: no events until "Wake from sleep detected" at 16:43), ending run E
  after step 32 and the queued script; not memory (a memory kill ends single processes),
  but the 7x buffer plus the watcher's copy came to ~12 GB of 15, so E resumed at 17:54
  without the watcher. In `CLAUDE.md`: a sleep can end WSL altogether — check the power
  source before and during long runs. The target-quality script now waits for any part
  of run E (it would have recognised only the first script's name).

- **2026-10-01 night.** Run E finished at step 44 (21:59, no pause since the resume); the
  target-quality run (22:00-22:22, 864 positions) found 400-simulation targets moderately
  better than 100-simulation ones (dwarfs: top move agreeing with a 2,000-simulation
  reference 57.4% against 51.8%, distance −0.061; trolls −0.053), the 1,000-simulation
  control closer still — not worth 7x fewer training positions an hour here.

- **2026-10-02, 00:45-03:00: the convolutional policy head, symmetry augmentation, an
  overfitting check.** Run E's evaluation started at 00:45 (detached,
  `stage3b_E_eval.sh`); the target-quality check with references of 2,000 and 4,000
  simulations is queued after it (`target_quality2.sh`; the user asked whether 2,000 is
  itself a stable reference). Meanwhile, at the user's request (2026-10-01), all in our
  copy or in the game, each behind a switch that is off by default, each with tests and
  controls:
  - **Game helpers** in `thud.h`: `PolicyPlaneIndex` (each action's entry in 120 planes
    over the board) and the board's symmetries (`SymmetricCoord`, `SymmetricDirection`,
    `SymmetricAction`, `SymmetricObservation`). The user asked whether an octagon has 16
    symmetries: a regular one does, but Thud's has edges of alternately 5 and 4 squares
    and keeps 8 (a 45-degree turn would not take squares to squares) — recorded in
    `PLAN.md`, to keep in mind in every change. Two new tests in `thud_test.cc`
    (`TestPolicyPlaneIndex`, `TestSymmetryHelpers`, exhaustive over all 19,800 actions;
    now 49 test functions, all passing); four planted errors (two mirror matrices
    swapped, a plane off by one, the observation not permuted, directions not turned)
    each caught; `crosscheck_tests.py` unchanged (0 disagreements). No existing test
    changed. `build-shared/libopen_spiel.so` relinked with them at 01:31 (GNU ld writes
    a new file, so the running matches kept their copy; the next match started at
    01:33 with the new one, which only adds functions).
  - **The convolutional policy head**, `--nn_model=resnet_conv_policy`, built as
    AlphaZero's and Leela Chess Zero's (sources checked: the *Science* paper's
    supplementary materials, lczero-training's `tfprocess.py`): 420,988 weights at 64 x 4
    against 9,244,626. `conv_policy_check` passes (layout 39,600 of 39,600 with a
    swapped-axes control, masking, gradients, both heads fit a fixed batch, checkpoints
    with a control); `identity_check` still 12 of 12; the trainer runs with it.
  - **Symmetry augmentation**, `--symmetry_augmentation`: `SymmetricTrainInputs` in
    `vpnet`, applied to each sampled position in the learner. `augmentation_check`
    passes against positions mirrored independently as text (1,608 of 1,608), with a
    control, and a planted error in the policy mapping is caught.
  - **Overfitting check** (the user's "3a"): the trainer logs, before each learning
    step, the losses on 2,048 new positions and 2,048 trained ones (`VPNetModel::Loss`,
    `Learn`'s code without the step; `identity_check` passes after that refactoring).
    Control: a tiny buffer trained hard shows new 7.12 against trained 3.76; an ordinary
    short run 4.53 against 4.66.
  - **The head check** (`thud/experiments/az_head_check.cc`): the layout check ported to
    our copy, evaluating held-out and training positions, with `augment=1` and a measure
    of how differently a network answers mirror images. The export was regenerated in
    `~/thud-runs/head_check/data` (same checksums as in September: 20,036 + 4,816
    positions). Training is deterministic (a repeat gave identical numbers), so the
    linear head at 32 x 2 must reproduce the layout control's 77.9%. Queued, detached:
    `head_check.sh` waits for E's evaluation and the target-quality check, then times
    both heads and trains both at 32 x 2 and 64 x 4, seeds 1-3; `head_check_augment.sh`
    then the 64 x 4 runs with augmentation. Estimated 05:00-07:30.
  - Not rebuilt until the night's runs are done: `az_match`, `az_forgetting`,
    `az_target_quality` (they compile our copy in; their resnet path is unchanged). A
    scratch build of `az_match` on the new copy plays conv-head networks; today's
    binary stops with "Unknown nn_model: resnet_conv_policy" (the control).
  - **03:30, for the morning** (the user sleeps until ~11:00 and asked for meaningful
    experiments to fill the machine): run E's evaluation ended at 03:27, so the queue
    above would leave the machine idle from roughly 08:00-09:00. Queued after it, run
    C' (`~/thud-runs/stage5_conv.sh`): stage 5's self-play test, one variable against
    run C. Equal steps rather than equal time, so the user's daytime use cannot spoil
    it (the learner trains on every 21,845 new positions whatever the speed; the new
    head is faster, so this understates it). Not chosen: a fresh 7x run from step 1
    (waits for E's results and the user), longer harness runs (the self-play run
    answers the overfitting question where it matters, through the trainer's new
    log).

- **2026-10-02, 05:50-06:40, while the user sleeps** (they asked for each result to be
  analysed as it lands, without a reminder; at 03:50 they asked whether that survives a
  compaction: knowing does, acting needs a trigger set beforehand — a background waiter,
  limited to 2 hours, then an hourly scheduled check, both living only as long as the
  Claude Code session):
  - **A stalled queue, fixed**: the reference check ended at 04:02, the head check at
    05:16, but `head_check_augment.sh` kept waiting — its `pgrep -f 'head_check[.]sh'`
    matched the Claude Code shell that had written and launched it at 02:06, still alive
    and holding the script's text in its command line. Ended that shell (the script runs
    in its own session under `nohup`, so it was unaffected); the augmented runs started
    at 05:49, 33 minutes late. `stage5_conv.sh` waits on the same pattern and would have
    stalled too. In `CLAUDE.md`: anchor waits to the script's own command line, checked
    with a dummy shell as the control. The running scripts were not edited (bash reads a
    script as it runs).
  - **Run E's evaluation** (`PLAN.md`, *First training runs*): E44 beats D29 by +9.7
    (+6.6 to +12.9, 33 of 40 pairs) and the anchor by +37.7 (dwarfs +8.7 against D29's
    +3.2, p = 0.03; trolls +29.0); E36 and E40 against the anchor +30.6 and +31.1, so
    the dwarfs' recovery came late; against A14 −3.5 (−5.7 to −1.3; D29 −5.4, the
    difference not significant). The forgetting check finds no narrowing (archive 18:
    4.96, D29 4.89, C29 6.57), a slow rise of ~0.2 on archives E no longer holds.
  - **The reference check** (`PLAN.md`, playout caps): 2,000 and 4,000 simulations agree
    on the dwarfs' top move in 88% of positions, at a quarter of 100 simulations'
    distance; against 4,000 the gain from 100 to 400 is the same (+6.2 points, distance
    −13%). The recommendation stands, no longer provisional.
  - **The head check** (`PLAN.md`, the convolutional head): the linear head at 32 x 2
    reproduces the layout control exactly (77.9%, 75.8-81.3); the new head reaches
    98.8-99.4% hurl mass on held-out positions (linear 75-78%), no train-test gap (the
    linear head 94-96% on training positions, a loss gap of 0.26), policy 3x closer to
    symmetric. **Correction to the 03:30 entry:** the new head is not faster in the
    trainer — 18.9 against 17.0 ms per batch of 32 and 514 against 448 ms per learning
    step at 64 x 4, OMP 4; only single positions are faster (1.35 against 3.06 ms). Run
    C' at equal steps is therefore not flattered.
  - Uncommitted, for the user: these doc updates (`PLAN.md`, this file, `CLAUDE.md`'s
    pgrep note) on top of the night's code.

- **2026-10-02, 07:30, the hourly check:**
  - **Symmetry augmentation on the harness** (`PLAN.md`, *Changes to the search and
    trainer*, the 2 x 2 at 64 x 4, seeds 1-3): for the linear head held-out hurl mass
    75.0% → 86.8% and the train-test loss gap 0.26 → 0.03; for the new head 98.8% →
    99.1%, its policy 20% closer to symmetric. No cost in time.
  - **Stage 5's gate passed** at 06:46 (98.8% against 75.0%); `make` in `build/` without
    warnings, `ctest` 285 of 285 (84 s); `az_match`, `az_forgetting`,
    `az_target_quality` rebuilt; **run C' started at 06:50**.
  - **Correction to the 05:50 entry:** "~10-15% slower in the trainer" assumed full
    batches of 32. The trainer's batches average far less (run C''s step 1: 8.6
    positions, run C's 11.8), where the new head is faster (one position 1.35 against
    3.06 ms), and C''s step 1 made 14.0 positions a second against C's 12.1. Run C''s
    later steps will tell.

- **2026-10-02, 14:30, the hourly check:** run C' trained to step 16 (06:50-13:38, exit
  0, no pause; one battery reading at 13:28 showed 1, the next 2). **Its dwarfs are much
  weaker than C's** at equal steps (`PLAN.md`, the convolutional head, run C'): against
  the anchor −6.9 and −10.3 at steps 12 and 16 (C +2.4 and +9.5), the trolls equal;
  the trainer's evaluator agrees, so not an `az_match` problem; in self-play the trolls
  won every game. No overfitting by the new check. 12% slower than C over the run — the
  07:30 entry's step-1 reading (faster) did not hold. The head-to-head with C16 and C'8
  against the anchor still running.

- **2026-10-02, 15:30, the last hourly check:** run C''s evaluation ended at 14:50.
  **C'16 loses to C16 head-to-head, −8.5** (−12.6 to −4.3, 30 of 40 pairs; its dwarfs
  −16.4 against C's trolls, C's −7.9 against its trolls); C'8 against the anchor +10.4,
  its trolls +27.8 (C8 +15.8), its dwarfs −17.4 (C8 +4.1). The new head learnt the
  trolls faster and the dwarfs much worse; stage 5 fails in this run, the head is not
  adopted for now (`PLAN.md`, decision log, for the user to confirm). A possible
  mechanism, unchecked: stage 3a's problem — strong trolls make the dwarfs' searches
  judge every move lost. The hourly check is deleted: the night's queue is done, and
  the machine is idle.

- **2026-10-02, ~16:00, with the user:** they decided the next run — a fresh 7x run
  from step 1 with the old head, probably with symmetry augmentation (whether with a
  control run without it is still to settle) — and that the convolutional head is to
  be revisited later. Committed the night's work in three commits (game helpers, tests
  and the convolutional head; symmetry augmentation; the overfitting check, the
  target-quality references and these docs); each intermediate state compiles (checked
  with `-fsyntax-only`; control: `augmentation_check` fails against the first). Also
  fixed in `PLAN.md`: roadmap row 1 still said run A2 was running (done 2026-09-28).

**Next step:** as recorded in `## Current status` — settle augmentation for the fresh 7x
run and start it; playout caps (recommended: on the GPU, no run F here) and the analysis
of run C''s saved data stay open.
