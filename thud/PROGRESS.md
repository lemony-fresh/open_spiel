# Progress

## Current status

**Phase:** 0 (environment) — in progress, straddling a reboot.

**Where we are:** Project scaffolding complete. The fork exists at
`https://github.com/lemony-fresh/open_spiel`. WSL2 is installed on the host but requires a
reboot to take effect, so no Linux-side work has happened yet and the repo has not been
cloned. Session 1 ended here, before the reboot.

**Next step:** Complete Phase 0 in WSL — clone the fork to `~/thud-openspiel`, create the
`thud` branch, move these files into place, then run `install.sh` and the CMake build. The
gate is `ctest -j10` passing and `./examples/example --game=tic_tac_toe` running.

**Phase 1 is done ahead of schedule:** `THUD_RULES.md` is written, sourced, and
cross-checked against three existing implementations. **All four undetermined points are now
settled** — lone troll shove (not allowed), lone dwarf hurl (not allowed), termination
(no-progress cap plus stalemate check), and troll captures (**a separate dwarf-removal phase**
with the same player still to act, modelled on `open_spiel/games/amazons/`).

Nothing about the rules is open. Phase 2's remaining decision is the action encoding —
shallow from-to pairs (~27,000 actions) versus Amazons-style sequential selection (~166,
deeper tree) — which should be settled against Phase 5 throughput.

**Nothing is blocked** other than by the reboot.

**Outstanding chore, carried until done:** the Windows-side staging folder
`C:\Users\waech\thud-init\` and the Windows-side memory at
`C:\Users\waech\.claude\projects\C--Users-waech\memory\` both go stale once the repo is
live. `START_HERE.md` step 6 has the detail. It is gated on the repo being committed, pushed
and built, so it will likely outlive the session that does the clone — **keep re-listing it
here until it is actually done.** The folder deletion and the memory update must happen
together, since that memory points into the folder.

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
