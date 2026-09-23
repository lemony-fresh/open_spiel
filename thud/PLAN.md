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

### Phase 0 — Environment — **done** 2026-09-22

- [x] Fork `google-deepmind/open_spiel` (via GitHub web UI)
- [x] Clone to `~/thud-openspiel` on the **WSL native filesystem**; create branch `thud`
- [x] Place `CLAUDE.md` (root) and `thud/` files; commit and push from Linux
- [x] `./install.sh`, venv, `pip install -r requirements.txt`
- [x] CMake build with clang, `make -j10` (4m41s, 0 warnings, native ARM64 — no fallback needed)
- [x] Gate: `ctest -j10` passes (284/284) and `./examples/example --game=tic_tac_toe` runs

**Risk:** OpenSpiel's ARM64 Linux source build is far less exercised than x86_64. If abseil
or another dependency fails, fall back to the `manylinux aarch64` wheel to keep Python-side
work unblocked while debugging the C++ build separately. Keep
`OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF`.

### Phase 1 — Ruleset spec, before any code — **done** 2026-09-22

`thud/THUD_RULES.md` is the specification: board, setup, every move type as explicit
allowed / not-allowed lists, end conditions, scoring and the action encoding (§8), with the
reasoning in §9.

- [x] Draft the spec from the published rules as reproduced on BoardGameGeek and in the
  Tabletop Simulator edition, cross-checked against three implementations: `dstu/thud`,
  `THFlowers/Thud-CLI`, `hexparrot/thudgame` (session 1).
- [x] Find the primary source: the official website's rules text, archived on the Wayback
  Machine (session 2). It settles the lone-dwarf hurl outright.
- [x] Re-verify the implementation evidence by reading every hurl and shove code path and
  running hexparrot's engine (session 2). Session 1 had misread two of the three engines,
  and two of its decisions rested on that; `PROGRESS.md`, session 2, has the corrected
  evidence with line citations.
- [x] Settle every open rule with the user (session 2) — list below.
- [x] Rewrite the spec to be sharp and concise; verify the board indexing, the action ranges
  and the starting position by script (the setup matches hexparrot's square for square).
- [x] Choose the end-of-battle limits by measurement, on 1,000 random and 3,000 AI games.

Settled:

- **Lone dwarf hurl — allowed**, as the official rules state explicitly.
- **Troll captures — all or none, declining allowed.** No separate removal phase.
- **Shoves travel 2..N squares and must capture** (official); a one-square shove equals a
  capturing step, so it is dropped.
- **End of battle** — no legal move, 200 turns without a capture, or 800 turns. The official
  end by agreement does not survive self-play. The limits are to be re-evaluated once our
  engine is strong (*Deferred decisions*).
- **One battle per OpenSpiel game**, returns `±m / 32`.

Where a rule ever becomes contested again, prefer an **OpenSpiel game parameter** over a
hard-coded choice, so it becomes an experiment rather than an argument.

### Phase 2 — Board and action encoding — **done** 2026-09-22 (design; built in Phase 3)

- **Actions — decided with the user, specified in `THUD_RULES.md` §8:** one action per
  turn, numbered from-square × direction × distance as in OpenSpiel's `chess` (18,480 IDs),
  plus 1,320 "troll step capturing all" IDs, for 19,800 in total. Every turn is exactly one
  action, so no multi-phase turn state machine is needed.
- **Alternatives considered and rejected** (reasoning in `PROGRESS.md`, session 2):
  - *Amazons-style "pick the piece, then the destination"* (~166 actions): a tiny policy
    output, but an extra tree level every turn and mid-turn states that the observation
    would have to represent. Session 1 had leaned this way.
  - *A follow-up capture/decline decision after a troll step* (THFlowers' style): the same
    outcomes, but it groups a strong capture with a usually-bad decline under one tree edge,
    and it adds mid-turn states.
  - *Raw from×to pairs* (27,225 IDs, 23% ever usable): the same moves as the chosen layout,
    in more IDs; the only in-tree precedent is `nine_mens_morris`.
  - No move-type component is needed in any of these: a one-square shove equals a capturing
    step, so an adjacent destination is a step and 2+ squares is a shove.
- **Board:** a 15x15 grid with 165 playable squares, one of them the Thudstone. Store it as
  a flat 15x15 array with a sentinel value on the cut corners. `chess` also uses a flat
  array (`std::array<Piece, k2dMaxBoardSize> board_`, 8×8, in `chess_board.h`), but bounds-checks with
  `InBoardArea()` instead of a sentinel. Padding the array so that rays never need bounds
  checks is a Phase 3 performance choice. The action index uses the packed 165-square
  numbering (§8), so the two need a precomputed mapping.
- **Utility:** dwarf = 1 point, troll = 4 points, giving 32 vs 32 — a naturally balanced
  game. Score-difference utility normalised to [-1, 1] gives a richer learning signal than
  win/loss. Settled: one battle per game, returns `±m / 32` (`THUD_RULES.md` §7, §9.5).

### Phase 3 — Implement `thud.h` / `thud.cc`

**Performance and readability are both requirements.** Move generation is expected to be
the training bottleneck, so the implementation must be **very fast** — but it must stay
**readable**, because the rules are fiddly and correctness comes first. In practice: fast,
simple data layouts (a flat padded board array, precomputed per-square ray tables, no
allocation in the hot path) rather than clever tricks that obscure the rules; one small,
clearly named function per move type that reads like its section of `THUD_RULES.md`; and
optimise from Phase 5 measurements, not guesses. A speed-up that makes a rule hard to
check against the spec is not worth it.

**Template: `tic_tac_toe/` for the boilerplate, `chess/` for the action encoding.** With one
action per turn (decided in Phase 2), Thud no longer needs Amazons' multi-phase turn state
machine; `current_player_` simply flips after every action. `chess` is the in-tree example
of a from-square × direction × distance action layout.

- `ThudGame`: `NumDistinctActions`, `NumPlayers`, `Min/MaxUtility`, `ObservationTensorShape`,
  `MaxGameLength`; game parameters `max_turns_without_capture` (default 200) and
  `max_turns` (default 800) in `parameter_specification`
- `ThudState`: `CurrentPlayer`, `LegalActions`, `DoApplyAction`, `IsTerminal`, `Returns`,
  `ObservationTensor`, `ObservationString`, `ToString`, `Clone`, `ActionToString`
- **The observation must be Markov, including the end-of-battle limits:** besides the pieces
  and the side to move, `ObservationTensor` must carry the no-capture counter and the turn
  count (each scaled by its limit), because both limits change the outcome. OpenSpiel's
  `chess` does the same for its 50-move counter (`AddScalarPlane(IrreversibleMoveCounter(),
  0, 101, ...)`, `chess.cc`). Without them, a position looks identical one turn before and one
  turn after the cap ends the game.

Build move generation in the order **hurl, dwarf move, shove, troll step**, testing each
before starting the next. Writing all four and then debugging them together is the obvious
trap here.

### Phase 4 — Tests

- **Unit-test every move type both ways: positive and negative.** For each of the five
  move types, assert that every *allowed* move is generated, and — equally important —
  that every *forbidden* move is absent from `LegalActions()`. A move generator that is
  merely permissive passes any positive-only test suite, so the negative cases are what
  actually pin the rules down. Derive them clause by clause from `THUD_RULES.md`:

  | Move type | Positive | Negative — must NOT be legal |
  |---|---|---|
  | Dwarf move | 1..k squares, all 8 directions, to empty squares | through or onto any piece; through or onto the Thudstone; onto a troll (a dwarf never captures by moving) |
  | Dwarf hurl | line of `N >= 1`, distance `1..N`, landing on a troll — including a **lone dwarf onto an adjacent troll** | lone dwarf at distance 2; distance `> N`; blocked path; landing on an empty square, a dwarf or the Thudstone; hurling the *rear* dwarf |
  | Troll step, captures none | exactly 1 square, all 8 directions, to an empty square | more than 1 square; onto any piece; onto the Thudstone |
  | Troll step, captures all | 1 square to an empty square with ≥ 1 adjacent dwarf; **every** adjacent dwarf removed | landing where no dwarf is adjacent; any dwarf left adjacent afterwards |
  | Troll shove | line of `N >= 2`, distance `2..N`, lands empty, ≥ 1 adjacent dwarf, **all** adjacent dwarfs removed | distance 1 (that is a step); lone troll; distance `> N`; blocked path; landing on a piece; **landing where no dwarf is adjacent** |

  Cover the board geometry explicitly too: moves off the octagon's diagonal edges, lines that
  run into a cut corner, and maximum-length lines along a full 15-square rank. Test the
  action encoding round trip (`THUD_RULES.md` §8) and the end conditions (§6) directly.

- `thud_test.cc` with OpenSpiel's `RandomSimTest` for crash-freedom and invariant checking.
- Register the short name in `open_spiel/python/tests/pyspiel_test.py`.
- Generate the playthrough baseline: `./open_spiel/scripts/generate_new_playthrough.sh thud`.
- Cross-check a handful of positions against an existing implementation — candidates:
  `github.com/dstu/thud` (Rust, has MCTS), `github.com/hexparrot/thudgame` (Python),
  `github.com/THFlowers/Thud-CLI` (Java, MCTS).

### Phase 5 — Benchmark, then decide

Measure legal-move generations/sec, random rollouts/sec, and MCTS simulations/sec. Record the
numbers in `PROGRESS.md`. These unblock both deferred decisions below. Also record how long
random rollouts run and how they end under the 200/800 limits, and compare with the
hexparrot-based measurements in `PROGRESS.md`, session 2, to confirm that proxy held.

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
diff to existing files inside `open_spiel/` stays at the two registration points.

**8. Expect a wait.** PRs are merged in batches, roughly every two weeks.

## Deferred decisions

Recorded here so they are decided deliberately rather than by accident.

| Decision | Blocked on |
|---|---|
| Classical MCTS + evaluation function vs AlphaZero-style learning | Phase 5 benchmarks |
| Local CPU training vs rented cloud GPU | Phase 5 benchmarks |
| `OPEN_SPIEL_BUILD_WITH_LIBTORCH` on aarch64 (needed for C++ AlphaZero) | Only if we commit to C++ AlphaZero |
| Whether to attempt upstreaming to OpenSpiel | Thud is commercially published; copyright question unresolved |
| **End-of-battle limits** (`THUD_RULES.md` §6: no-capture cap and hard turn cap) — **re-evaluate once our engine plays strongly** | A strong engine of our own. The current defaults rest on proxies only (random play and hexparrot's heuristic AI; see `PROGRESS.md`, session 2), and strong play may stall in ways the proxies never do. Re-measure on our engine's self-play: how games end (rout, no legal move, no-capture cap, hard cap), the longest no-capture stretches, and the "dead tail" after the last capture. Lower the no-capture cap if games regularly sit out the whole cap; raise it if it cuts off real manoeuvring. Both are game parameters, so no code change is needed. |
