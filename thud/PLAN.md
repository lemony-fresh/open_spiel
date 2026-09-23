# Thud on OpenSpiel — Roadmap

The roadmap and the reasoning behind it. Changes when scope changes — not a session log
(that is `PROGRESS.md`).

## Goal

Implement **Thud** (Discworld board game by Trevor Truran: 32 dwarfs vs 8 trolls on an
octagonal board) as a C++ game in OpenSpiel, then build an AI for it.

## Two framing decisions

1. **C++, not Python.** Move generation is expected to be the training bottleneck, and Thud
   has a large branching factor (32 dwarfs each moving like a chess queen) and long games.
   A Python game implementation would cap throughput early.
2. **A correct game first; the AI approach is deliberately deferred.** Classical MCTS versus
   AlphaZero-style learning is not decidable until we can measure real throughput on this
   hardware. See *Deferred decisions*.

---

## Phases

### Phase 0 — Environment

- [ ] Fork `google-deepmind/open_spiel` (via GitHub web UI)
- [ ] Clone to `~/thud-openspiel` on the **WSL native filesystem**; create branch `thud`
- [ ] Place `CLAUDE.md` (root) and `thud/` files; commit and push from Linux
- [ ] `./install.sh`, venv, `pip install -r requirements.txt`
- [ ] CMake build with clang, `make -j10`
- [ ] Gate: `ctest -j10` passes and `./examples/example --game=tic_tac_toe` runs

**Risk:** OpenSpiel's ARM64 Linux source build is far less exercised than x86_64. If abseil
or another dependency fails, fall back to the `manylinux aarch64` wheel to keep Python-side
work unblocked while debugging the C++ build separately. Keep
`OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF`.

### Phase 1 — Ruleset spec, before any code — **done**

`thud/THUD_RULES.md` is written and sourced: board geometry, exact setup, all four move
types, scoring, and termination.

What it settled: troll captures are **optional and a chosen subset**; a **shove must capture
at least one dwarf**; dwarfs move first; a dwarf never captures by moving and a hurl may only
land on a troll; setup is exact (32 dwarfs on all perimeter squares but the 4 in line with
the Thudstone, 8 trolls ringing it).

Section 8 of `THUD_RULES.md` records the four points the published rules leave undetermined,
each cross-checked against three existing implementations (`dstu/thud`, `THFlowers/Thud-CLI`,
`hexparrot/thudgame`). Two are now settled:

- **Lone troll shove (`N = 1`) — settled: not allowed.** All three implementations agree.
- **Termination — settled: no-progress cap plus a stalemate check**, following
  `hexparrot/thudgame` (which annotates its 400-ply cutoff "self-play only"). `dstu/thud`
  models the official agreement mechanism faithfully, but it does not survive self-play,
  because the trailing player never accepts.

- **Lone dwarf hurl (`N = 1`) — settled: not allowed.** The official rules phrase hurl and
  shove *identically* ("anywhere there is a straight line of adjacent trolls/dwarfs ... they
  may shove/hurl"), so they must be read identically, and shove's `N >= 2` is unanimous.
  `THFlowers/Thud-CLI` allows `N = 1` and is simply wrong here.

- **Troll captures — settled: a separate dwarf-removal phase.** Implementations were split
  (`THFlowers` does it faithfully as a remove phase, `hexparrot` captures all, `dstu` unclear
  on a first reading), so this was decided on the faithful reading. After a troll move or
  shove resolves, the state enters a removal phase with the **same player** still to act:
  legal actions are "remove the dwarf at square X" for each adjacent dwarf, plus "done".
  "Done" is available immediately after a move (capturing is not compulsory) but only after
  at least one removal following a shove (a shove must capture).

**All four rules questions are now settled.** Where a rule ever becomes contested again,
prefer an **OpenSpiel game parameter** over a hard-coded choice, so it becomes an experiment
rather than an argument.

### Phase 2 — Board and action encoding

- **Board:** 15x15 with a 15-square triangle removed from each corner gives 165 squares, one
  of which is the central Thudstone (never moved onto or through). Represent as a flat 15x15
  array with an off-board sentinel — simpler and faster than a packed 165-entry list, and it
  mirrors how `chess` does it.
- **Actions — the central trade-off is shallow-and-wide versus deep-and-narrow.**
  - *Shallow:* every Thud move is a `(from, to)` pair, so one action per move. 165x165 =
    27,225 actions — tractable for MCTS (Go is 362, chess ~4,672) but a large policy head.
  - *Deep:* decompose the turn into sequential single-square choices, as
    `open_spiel/games/amazons/` does — select piece, then select destination, then (for
    trolls) select captures. `NumDistinctActions` collapses to roughly **166**, which is far
    friendlier for a policy network, at the cost of 2–3x tree depth per turn.

  The deep encoding also makes the troll capture phase fall out naturally rather than being
  bolted on, and it shares the state machine we need for captures anyway. **Leaning deep**,
  but decide with Phase 5 throughput in mind, since depth is what MCTS pays for.

  - **Disambiguation matters in the shallow encoding.** Dwarf *move* vs *hurl* separate
    cleanly by target occupancy (a hurl lands on a troll, a move lands on an empty square).
    Troll *move* vs *shove* at distance 1 do **not** — both land on an empty adjacent square
    but differ in legality and effect. The shallow encoding therefore needs an explicit
    move-type component or disjoint action ranges. The deep encoding sidesteps this by
    selecting the move type as its own decision.
- **Utility:** dwarf = 1 point, troll = 4 points, giving 32 vs 32 — a naturally balanced
  game. Score-difference utility normalised to [-1, 1] gives a richer learning signal than
  win/loss. Check this against the two-round swap-sides match structure before committing.

### Phase 3 — Implement `thud.h` / `thud.cc`

**Mirror `open_spiel/games/amazons/`, not `tic_tac_toe/`.** The developer guide points at
tic-tac-toe, but that is a single-action-per-turn game. Amazons has a genuine multi-phase
turn — move a piece, then shoot an arrow — implemented as
`enum MoveState { amazon_select, destination_select, shot_select }` with `current_player_`
flipped only in the last phase. Thud's turn structure (choose move type, move, then select
captures) is the same shape, so Amazons is the closer template. Keep tic-tac-toe as the
reference for the surrounding boilerplate only.

- `ThudGame`: `NumDistinctActions`, `NumPlayers`, `Min/MaxUtility`, `ObservationTensorShape`,
  `MaxGameLength`
- `ThudState`: `CurrentPlayer`, `LegalActions`, `DoApplyAction`, `IsTerminal`, `Returns`,
  `ObservationTensor`, `ObservationString`, `ToString`, `Clone`, `ActionToString`

Build move generation in the order **hurl, dwarf move, shove, troll move**, testing each
before starting the next. Writing all four and then debugging them together is the obvious
trap here.

### Phase 4 — Tests

- **Unit-test every move type both ways: positive and negative.** For each of the five
  decision types, assert that every *allowed* move is generated, and — equally important —
  that every *forbidden* move is absent from `LegalActions()`. A move generator that is
  merely permissive passes any positive-only test suite, so the negative cases are what
  actually pin the rules down. Derive them clause by clause from `THUD_RULES.md`:

  | Decision | Positive | Negative — must NOT be legal |
  |---|---|---|
  | Dwarf move | 1..k squares, all 8 directions, to empty squares | through or onto any piece; through or onto the Thudstone; onto a troll (a dwarf never captures by moving) |
  | Dwarf hurl | line of `N >= 2`, distance `1..N`, landing on a troll | lone dwarf (`N = 1`); distance `> N`; blocked path; landing on an empty square; landing on a dwarf; hurling the *rear* dwarf |
  | Troll move | exactly 1 square, all 8 directions, to an empty square | more than 1 square; onto any piece; onto the Thudstone |
  | Troll shove | line of `N >= 2`, distance `1..N`, lands empty, captures `>= 1` | lone troll (`N = 1`); distance `> N`; blocked path; landing on a piece; **landing where no dwarf is adjacent** |
  | Dwarf removal | removing any dwarf adjacent to the destination; "done" | removing a non-adjacent dwarf; removing a troll; "done" immediately after a *shove* before any removal (a shove must capture) |

  Cover the board geometry explicitly too: moves off the octagon's diagonal edges, lines that
  run into a cut corner, and maximum-length lines along a full 15-square rank.

- `thud_test.cc` with OpenSpiel's `RandomSimTest` for crash-freedom and invariant checking.
- Register the short name in `open_spiel/python/tests/pyspiel_test.py`.
- Generate the playthrough baseline: `./open_spiel/scripts/generate_new_playthrough.sh thud`.
- Cross-check a handful of positions against an existing implementation — candidates:
  `github.com/dstu/thud` (Rust, has MCTS), `github.com/hexparrot/thudgame` (Python),
  `github.com/THFlowers/Thud-CLI` (Java, MCTS).

### Phase 5 — Benchmark, then decide

Measure legal-move generations/sec, random rollouts/sec, and MCTS simulations/sec. Record the
numbers in `PROGRESS.md`. These unblock both deferred decisions below.

---

## Upstreaming to OpenSpiel — pre-PR checklist

Not a near-term goal, and it may be blocked outright (see step 1). Recorded now because
several steps are cheap to honour as we go and expensive to retrofit — above all keeping our
files out of `open_spiel/` and keeping `master` clean.

**1. Ask before polishing.** OpenSpiel's contributing guide says to contact the maintainers
*before* writing a large piece of code, and they keep a "Call for New Games" issue. Raise the
copyright question in that first contact: their guide explicitly warns that "some games may
have copyrights which might require legal approval", and Thud is commercially published by
Trevor Truran / The Cunning Artificer. A "no" here makes every later step moot, so ask early.

**2. Strip copyrighted text.** Game *mechanics* are not copyrightable, but the *expression*
of them is, and `thud/THUD_RULES.md` quotes the official rules verbatim in places.

Three layers guard this, because **no GitHub feature can block a file from being
upstreamed** — a PR is just a diff between two branches, and nothing in a file's contents can
veto it:

- *Structural (strongest).* Build the PR branch so our files were never on it. Nothing to
  strip means nothing to forget.
- *Mechanical.* `.githooks/pre-push` refuses any push whose target remote URL is
  `deepmind/open_spiel` when `CLAUDE.md` or anything under `thud/` is in the diff. Pushes to
  our own fork are unaffected. Requires `git config core.hooksPath .githooks` per clone, and
  is bypassable with `--no-verify` — a backstop, not a guarantee.
- *Advisory.* An uppercase DO-NOT-UPSTREAM banner at the top of `thud/THUD_RULES.md`, plus
  the rule in `CLAUDE.md`. Realistically this is the layer that matters most, since the agent
  preparing a PR reads those files.

Note that `.gitattributes export-ignore` does **not** help here: it only affects
`git archive` output, not pushes or pull requests.

- `thud/THUD_RULES.md` must **not** be upstreamed at all.
- Any rules description that does go upstream — header comments, the `docs/games.md` entry —
  must be written from scratch in our own words, with no quoted phrasing.
- The name "Thud" and its Discworld associations are plausibly trademarked. Flag this to the
  maintainers; it may require a rename or explicit permission.

**3. The PR branch contains only these files.** Nothing from `CLAUDE.md` or `thud/`.

- `open_spiel/games/thud/{thud.h,thud.cc,thud_test.cc}`
- `open_spiel/games/CMakeLists.txt` — game sources plus the test target
- `open_spiel/python/tests/pyspiel_test.py` — the `thud` short name
- `open_spiel/integration_tests/playthroughs/thud.txt` — generated, not hand-written
- `docs/games.md` — one table row: status badge, name, players, deterministic, perfect-info,
  description

**4. Remove our Apache 2.0 "modified file" notices** from `games/CMakeLists.txt` and
`pyspiel_test.py`. Those exist because *we* distribute a modified fork (Apache 2.0 §4(b));
once the changes are upstream they are no longer third-party modifications and the notices
are just noise in someone else's file.

**5. Style and hygiene.** `pip install pre-commit && pre-commit install`; clang-format for
C++; follow the Google C++ style guide. Commit from Linux so line endings stay LF — OpenSpiel
explicitly calls out CRLF contamination from Windows contributors.

**6. All tests green.** Full `ctest`, `pyspiel_test.py`, and the playthrough must regenerate
byte-identically.

**7. Branch hygiene.** Rebase onto current upstream `master` and open the PR from a branch
carrying only the upstreamable commits. This is the whole reason `master` stays clean and our
diff inside `open_spiel/` stays to two registration lines.

**8. Expect a wait.** PRs are merged in batches, roughly every two weeks.

## Deferred decisions

Recorded here so they are decided deliberately rather than by accident.

| Decision | Blocked on |
|---|---|
| Classical MCTS + evaluation function vs AlphaZero-style learning | Phase 5 benchmarks |
| Local CPU training vs rented cloud GPU | Phase 5 benchmarks |
| `OPEN_SPIEL_BUILD_WITH_LIBTORCH` on aarch64 (needed for C++ AlphaZero) | Only if we commit to C++ AlphaZero |
| Whether to attempt upstreaming to OpenSpiel | Thud is commercially published; copyright question unresolved |
