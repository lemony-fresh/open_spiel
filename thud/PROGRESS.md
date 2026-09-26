# Progress

## Current status

**Phase:** 0–5 **done** — Thud is implemented, passes all its tests, has its integration
baselines, and is benchmarked (session 5, 2026-09-25); **Phase 6 in progress**:
AlphaZero-style learning with **OpenSpiel's C++ AlphaZero**, decided with the user
2026-09-25 (`PLAN.md` Phase 6). **It builds and runs here**: LibTorch 2.10 from PyTorch's
aarch64 pip wheel, in `build-torch/`, no change to upstream code; OpenSpiel's LibTorch
tests pass, it learns tic-tac-toe like the Python control, a Thud smoke test runs end
to end, and on Thud's board it learns like the Python model with the layout fixed (the
C++ against Python layout control). Run it with `OMP_NUM_THREADS=1` (self-play up to 8x faster) and our defaults in
`thud/experiments/az_thud.flags` (resignation off, root noise α 0.1). OpenSpiel's Python
AlphaZero also runs here but is not our route (too slow, and a layout bug).

**Where we stand:**

- `open_spiel/games/thud/thud.cc` implements the game as designed in `PLAN.md` Phase 3
  (padded 17x17 grid, one function per move type, `IsTerminal` cached and decided without
  generating moves). `thud_test.cc` has **47 test functions, all passing**; the first 46
  passed without any test being changed, and the 47th runs OpenSpiel's generic tests.
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
  `az_speed.py` need pyspiel and the JAX set; `az_layout_check.cc` is their C++
  counterpart, built by `build_az_layout_check.sh` against `build-shared/libopen_spiel.so`.
  hexparrot runs from a clone
  outside the repo, by default
  `~/.local/share/thud-openspiel/hexparrot_thudgame` (under `$XDG_DATA_HOME` if set), which
  **exists on this machine** at the pinned commit 7b171108 (2026-09-25). If it is missing,
  every script stops with a full explanation: what hexparrot is, which scripts need it,
  where it goes, and the exact commands to restore and check it.

**Next steps, in order (`PLAN.md` Phase 6):**

1. **Throughput on this CPU**, always with `OMP_NUM_THREADS=1`: simulations/s and
   games/hour for a few network sizes, and batched inference (`--inference_batch_size`,
   `--inference_threads`). Decides when to move to the cloud.
2. **A small training run** with `thud/experiments/az_thud.flags`: read the evaluation
   per side, as the share of games won and the mean margin (the side is recovered from the
   evaluator logs); measure how widely both sides' searches spread their visits and whether
   the dwarfs' policy sharpens (the untried-move question — measure first, user); use few
   evaluation levels (the default 7 reach 300,000 simulations a move).
3. Then settings (`PLAN.md` Phase 6, *Settings to determine empirically*: `uct_c`,
   simulations and network size first; then α 0.03 and 0.3 against 0.1), then a cloud GPU.
4. **If we ever fall back to the Python route** (user, 2026-09-25): first re-verify the
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
  0 warnings; `thud/experiments/build_az_layout_check.sh` compiles the new
  `az_layout_check.cc` with upstream's `model.cc` and `vpnet.cc` (the shared library
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

**Next step:** as recorded in `## Current status` — throughput on this CPU.
