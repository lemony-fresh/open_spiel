# Progress

## Current status

**Phase:** 0 (environment), 1 (rules) and 2 (action encoding) are **done** as of 2026-09-22.
The user read and approved `THUD_RULES.md`, and session 2's documentation changes are
committed and pushed to `origin/thud`. **Next: Phase 3** — implement the game.

**Next step, concretely:** first run the `~/.bashrc` check below. Then create
`open_spiel/games/thud/` (`thud.h`, `thud.cc`, `thud_test.cc`, each with our Apache 2.0
header), register it in `open_spiel/games/CMakeLists.txt` (adding the Apache §4(b)
"modified" notice), and work **test-first** as `PLAN.md` Phase 3 lays out: foundation
(board, setup, positions from ASCII diagrams, action encoding with a round-trip test), then
per move type — hurl, dwarf move, shove, troll step — write its tests from `THUD_RULES.md`
and the unit-test table and corner cases in `PLAN.md` Phase 3 (borders, wrap-around), see
them fail, implement, see them pass. Keep the code very fast yet readable. The user may want to review tests before
the implementation; ask where to pause if they have not said.

**Where we are:** The repo is at `~/thud-openspiel` (WSL2, Ubuntu 26.04.1, aarch64) on branch
`thud`, pushed to `origin`, with an `upstream` remote and the pre-push hook installed. OpenSpiel
builds natively on ARM64: clang 21.1.8, cmake 4.2.3, Python 3.14.4 venv; `make -j10` takes
4m41s with 0 warnings; `ctest -j10` passes 284/284 in 91 s; `./examples/example
--game=tic_tac_toe` runs. The `manylinux` wheel fallback was not needed.

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

**Unverified — check at the start of the next session:** whether the `~/.bashrc` lines from
`CLAUDE.md` reach Claude Code's Bash tool. Its shell snapshot carried only `PATH` among
exported variables and no `.bashrc` aliases, so `PYTHONPATH` probably does **not** arrive;
the venv (a `PATH` change) might. Run `echo $PYTHONPATH; which python3` first thing. Until
then, use `~/thud-openspiel/venv/bin/python3` explicitly. `ctest` is unaffected — CMake sets
`PYTHONPATH` for the Python tests itself.

**No outstanding chores.** `C:\Users\waech\thud-init\` was deleted (with the user's approval)
and the Windows-side project memory now redirects to this repo.

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
