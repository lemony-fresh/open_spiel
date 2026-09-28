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
   hardware. See *Deferred decisions*. **Decided 2026-09-25, after Phase 5: AlphaZero-style
   learning, with OpenSpiel's C++ AlphaZero** (Phase 6).

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

### Phase 3 — Implement test-first: `thud.h`, `thud.cc`, `thud_test.cc` — **done** 2026-09-25

All 46 test functions pass, with no change to any test; a planted wrap-around bug (a
15-wide row stride) is caught by 21 of them, including all four wrap-around tests.

**Work test-first (proposed by the user, 2026-09-22).** `THUD_RULES.md` is already written as
allowed / not-allowed lists, so turn it into tests *before* the code that satisfies them —
tests written from the spec cannot inherit the implementation's misunderstandings. Order:

1. **Foundation:** board, setup, and constructing a position from an ASCII board diagram (as
   the dstu and hexparrot tests do); the action encoding and `ActionToString`, with an
   encoding round-trip test. Tests need these to set up positions and name moves.
2. **Per move type, in the order hurl, dwarf move, shove, troll step:** write its tests from
   the unit-test table and corner cases below, see them fail, implement, see them pass —
   then move on. Writing all four generators and debugging them together is the obvious
   trap here.
3. **End conditions and scoring**, the same way.

The user reviewed the tests before any implementation (sessions 3–4), so the implementation
works through all move types in one go, without pausing for review (user, 2026-09-25).
**A test that fails and seems to need changing is never just edited to pass:** record each
such change with its reason, and go over every one with the user afterwards, to make sure
the changed test is truly correct rather than bent to match a broken implementation.

**After the test review, revisit the design decisions — before implementing anything (user,
2026-09-23).** All tests were written first and reviewed with the user chunk by chunk
(session 3, finished 2026-09-24). Now go back over the decisions the tests and `thud.h` lock
in, since changing them after the implementation exists means rewriting both. **Status
2026-09-24: all six are decided; implementation is next.**

- the position text format — `-` `.` `d` `T` `O` rows plus the status line
  `to_move=… turns=… turns_without_capture=…` — used by `ToString()` and for reading
  positions. Every test position is written in it; `TestDiagramRoundTrip` pins the details.
  **Decided 2026-09-24: kept**, one format both ways like chess's FEN, counters included.
  The cut-off corners were `#` until the `thud.h` API review the same day (below) changed
  them to `-`: OpenSpiel's saved-game files drop every line starting with `#` as a comment
  (`DeserializeGameAndState`, `spiel.cc`), and 10 of the 15 rows started with `#`.
  Reading must also reject malformed text and impossible positions (listed in `thud.h`).
  Following chess's `BoardFromFEN`, `PositionFromText()` returns nullopt for them and
  `NewInitialState()` turns that into the usual fatal error; `TestRejectedPositions` checks
  the function directly. Not a throwing error handler: OpenSpiel's C++ never throws
  (`pyspiel.cc:829`), and Google style bans exceptions;
- the `ActionToString` notation. The tests name every move in it; `TestActionToString` pins
  it. **Decided 2026-09-24:** `(r,c)-(r,c)` for a move that captures nothing and
  `(r,c)-(r,c)x` for every capture — hurl, capturing step or shove — as hexparrot marks
  every capture after the move. So a one-square capture is a hurl or a capturing step
  depending on the piece; the tests' parser looks it up in the position;
- the observation tensor, proposed as 7 planes of 15x15 (dwarfs, trolls, empty, Thudstone,
  trolls-to-move, turns-without-capture fraction, turns fraction), the same for both
  players. `TestObservation` pins all of it. **Decided 2026-09-24: 6 planes, the Thudstone
  plane dropped.** By the rules the stone is just a hole, like a cut-off corner, and holes
  are 0 in every board plane, as Havannah and Y leave their off-board cells. The empty
  plane stays: it is what tells an empty square from a hole. AlphaGo Zero and AlphaZero
  drop it only because Go and chess boards have no holes. The rest is as in chess (and
  AlphaZero): a side-to-move plane and counters scaled into 0-1, planes first;
- `InformationStateString`: `thud.cc` declares it provided, but it is only a stub and no test
  covers it. Upstream perfect-information games either return the move history,
  `HistoryString()` (`tic_tac_toe.cc:232`, `amazons.cc:359`, `chess.cc:397`,
  `checkers.cc:499`, `go.cc:129`), or do not provide it (`breakthrough.cc:52`).
  **Decided 2026-09-24:** the move history, like those five; `TestInformationState`
  covers it;
- the board storage, proposed as a 165-square array numbered like the actions, with move
  paths precomputed from `(row, col)`. It is internal, so no test depends on it.
  **Decided 2026-09-24: a padded grid** — the 15x15 grid inside a one-cell border, 17x17
  cells, with the border and the cut-off corners holding a "not a square" value. A line is
  walked by adding a fixed offset per direction (±1, ±16, ±17, ±18), and one condition —
  "is the next cell empty?" — stops every walk, whatever is in the way: a piece, the
  Thudstone, a cut-off corner or the edge. OpenSpiel's Go pads its board the same way
  (`go_board.h:50`); hexparrot does too. Chosen for readability and safety, not speed: a
  precomputed neighbour table (dstu's design, 165 cells) costs a dependent lookup per
  step, and an unpadded grid needs a row/column bounds check per step (as chess's
  `InBoardArea` does), whose omission is exactly the wrap-around bug; the speed and
  memory differences are a cycle or so per step and ~100 bytes per state (a state's move
  history alone is 16 bytes per turn). Translating between the actions' 165 numbers and
  grid cells happens once per piece or applied move, never per step. How `thud.h`'s
  private board member and the "not a square" value are declared is part of the API
  item below;
- the public API of `thud.h`: names, and what it exposes for tests and tools (`CellAt`,
  `TurnsPlayed`, `TurnsWithoutCapture`, the encoding helpers, and `Position` with
  `PositionFromText()`, added 2026-09-24). **Decided 2026-09-24:**
  - **The grid's "not a square" value is `Cell::kOffBoard`** in the public `Cell` type,
    documented as never returned by `CellAt()` nor found in a `Position` — as OpenSpiel's
    Go keeps `GoColor::kGuard` for its padding (`go_board.h:30`).
  - **A state is built from a `Position`**, not from text: `NewInitialState(text)` reads
    the text with `PositionFromText()`, stops with a fatal error on nullopt, and builds the
    state. The state remembers a starting `Position` other than the opening.
  - **Saving a game started from a diagram**, as chess saves its starting FEN
    (`ChessState::Serialize`, `ChessGame::DeserializeState`): `Serialize()` writes that
    position in the text format, then the actions one per line; a game from the opening
    keeps OpenSpiel's default of actions only; `DeserializeState()` tells them apart by the
    first character. That needed the `#` → `-` change above. OpenSpiel's own mechanism
    (`starting_state_str_`, written as `starting_state=…`) was rejected: it stores one line,
    and `State::StartingState()`, exposed to Python as `starting_state()`, always parses
    it as JSON (`spiel.h:1257`), so a Thud position there would make that call fail.
    `TestSerialize` pins the saved text and restores both kinds of game, directly and
    through `SerializeGameAndState` / `DeserializeGameAndState`.
  - Not added: a way to read out the whole `Position` (tools can use `CellAt` until
    something needs more) and `UndoAction` (a Phase 5 candidate).

**Performance and readability are both requirements.** Move generation is expected to be
the training bottleneck, so the implementation must be **very fast** — but it must stay
**readable**, because the rules are fiddly and correctness comes first. In practice: fast,
simple data layouts (the padded grid decided above, no allocation in the hot path) rather
than clever tricks that obscure the rules; one small, clearly named function per move type
that reads like its section of `THUD_RULES.md`; and optimise from Phase 5 measurements, not
guesses. A speed-up that makes a rule hard to check against the spec is not worth it.
**Step by fixed offsets only on the padded grid, whose border stops every line — never on
an unpadded flat array**, where a line can wrap from one row or column into the next (see
the wrap-around corner case below). **`IsTerminal` needs
no move generation:** the dwarfs have a legal move exactly while a dwarf is left, and the
trolls exactly while some troll has an empty square next to it (proof in the comment above
`TestEndNoLegalMove`; `TestRandomPlay` checks the same condition in every position).

**Template: `tic_tac_toe/` for the boilerplate, `chess/` for the action encoding.** With one
action per turn (decided in Phase 2), Thud no longer needs Amazons' multi-phase turn state
machine; `current_player_` simply flips after every action. `chess` is the in-tree example
of a from-square × direction × distance action layout.

- `ThudGame`: `NumDistinctActions`, `NumPlayers`, `Min/MaxUtility`, `ObservationTensorShape`,
  `MaxGameLength`; game parameters `max_turns_without_capture` (default 200) and
  `max_turns` (default 800) in `parameter_specification`
- `ThudState`: `CurrentPlayer`, `LegalActions`, `DoApplyAction`, `IsTerminal`, `Returns`,
  `ObservationTensor`, `ObservationString`, `InformationStateString`, `ToString`, `Clone`,
  `ActionToString`
- **The observation must be Markov, including the end-of-battle limits:** besides the pieces
  and the side to move, `ObservationTensor` must carry the no-capture counter and the turn
  count (each scaled by its limit), because both limits change the outcome. OpenSpiel's
  `chess` does the same for its 50-move counter (`AddScalarPlane(IrreversibleMoveCounter(),
  0, 101, ...)`, `chess.cc`). Without them, a position looks identical one turn before and one
  turn after the cap ends the game.

**Unit tests — written before the code they test** (`thud_test.cc`):

- **Test every move type both ways: positive and negative.** For each of the five
  move types, assert that every *allowed* move is generated, and — equally important —
  that every *forbidden* move is absent from `LegalActions()`. A move generator that is
  merely permissive passes any positive-only test suite, so the negative cases are what
  actually pin the rules down. Derive them clause by clause from `THUD_RULES.md`:

  | Move type | Positive | Negative — must NOT be legal |
  |---|---|---|
  | Dwarf move | 1..k squares, all 8 directions, to empty squares | through or onto any piece; through or onto the Thudstone; onto a troll (a dwarf never captures by moving) |
  | Dwarf hurl | line of `N >= 1`, distance `1..N`, landing on a troll — including a **lone dwarf onto an adjacent troll**, and **one dwarf heading several lines, hurling along each** | lone dwarf at distance 2; distance `> N`; blocked path; landing on an empty square, a dwarf or the Thudstone; hurling the *rear* dwarf; a distance that only a longer line in **another direction** would allow (`N` is counted per direction) |
  | Troll step, captures none | exactly 1 square, all 8 directions, to an empty square | more than 1 square; onto any piece; onto the Thudstone |
  | Troll step, captures all | 1 square to an empty square with ≥ 1 adjacent dwarf; **every** adjacent dwarf removed | landing where no dwarf is adjacent; any dwarf left adjacent afterwards |
  | Troll shove | line of `N >= 2`, distance `2..N`, lands empty, ≥ 1 adjacent dwarf, **all** adjacent dwarfs removed — including **one troll ending several lines, shoved along each** | distance 1 (that is a step); lone troll; distance `> N`; blocked path; landing on a piece; **landing where no dwarf is adjacent**; a distance that only a longer line in **another direction** would allow (`N` is counted per direction) |

- **Corner cases — cover each explicitly:**
  - **No wrap-around between rows or columns** (raised by the user). A board stored as a flat
    array can let a long move run off the end of one row and continue on the next: east from
    `(5,14)` is index +1, which is `(6,0)`; north-east from `(6,14)` is index −14, also
    `(6,0)`; in a column-major layout, south from `(14,5)` continues at `(0,6)`. In rows 5–9
    and columns 5–9 both ends are playable squares, so a mask of off-board squares does
    **not** catch it. Test long horizontal, vertical and diagonal moves, hurls and shoves
    that end on, or would continue past, the last square of a full-length row or column, in
    both directions, and assert that no legal move crosses from one row or column into
    another.
  - **Borders of all three kinds:** horizontal, vertical, and the diagonal edges of the cut
    corners. For every move type, test moves along a border and next to it; for hurls and
    shoves, test moving **away from** a border, **towards** it (e.g. hurling onto a troll on
    an edge square) and **across** it — always illegal. The octagon is convex along all 8
    directions (checked by script), so a path that leaves the board never re-enters: "across"
    simply means the path leaves the board, and such action IDs are never legal.
  - **Lines that end at a border:** `N` counts pieces only up to the edge.
  - **Captures around a landing square on a border**, which has fewer than 8 neighbours.
  - **The Thudstone:** it blocks paths, breaks a line, cannot be landed on or captured.
  - **Maximum lengths:** 14-square moves along a full row or column, and 9-square moves along
    the longest diagonals (the cut corners shorten every diagonal; checked by script). Hurls
    and shoves: 7 squares along a full row or column, the longest possible — perft cannot
    cover these, because its reference engine stops at 6.
  - **Exact boundaries of the end conditions** (§6): the battle ends after exactly 200 turns
    without a capture, not 199; after exactly 800 turns; when the player to move has no
    legal move, including when a side has no pieces left — and so at once when a capture
    takes the last opposing piece. However it ends, a finished battle has no player to move
    and no legal action.
- Test the action encoding round trip (`THUD_RULES.md` §8) directly.
- **Whole positions** (added during the user's test review, 2026-09-23):
  - the opening's **complete** legal set, every action of every kind, with both sides to
    move;
  - **perft** — the number of move sequences of 1-3 turns — on five positions (opening,
    a constructed tangled position, a midgame from a real game), against reference counts
    from hexparrot/thudgame's independent engine, as OpenSpiel's `chess` test does with
    published counts;
  - the board's **8 symmetries**: in fixed and random-play positions, the legal moves of a
    rotated or mirrored position are the rotated or mirrored legal moves, and applying a
    move commutes with the symmetry;
  - after **every move of random games**: the turn count rises by one, plain moves and
    steps capture nothing and add one turn without a capture, a hurl removes exactly one
    troll and a capturing step or shove exactly the dwarfs next to its landing square,
    each resetting that count. In **every position** of those games, the players take turns
    and the battle is over exactly when an ending of §6 holds, with the returns of the
    pieces left; and both players' observation (every plane) and strings match the
    position — also after the battle has ended, as OpenSpiel's generic tests require. The
    games must play all five kinds of move, or the test fails.
- **Every hand-written move expectation is cross-checked** against hexparrot's independent
  engine by `thud/experiments/crosscheck_tests.py` (added 2026-09-24): exact move sets,
  legal and illegal moves, reach tables and the position after a move, plus a check that
  every test diagram is a position the reader accepts. **Re-run it after changing any
  test.** Its first run found two diagrams with 12 trolls, which the reading rules decided
  that day reject.

### Phase 4 — Integration tests and baselines — **done** 2026-09-25

The full OpenSpiel suite passes 285 of 285.

- The playthrough baseline, `open_spiel/integration_tests/playthroughs/thud.txt` (890 KB),
  from `./open_spiel/scripts/generate_new_playthrough.sh thud`: one seeded random game of
  308 turns, which the trolls win by taking the last dwarf (returns -0.75 / +0.75).
  **Regenerate it and read the diff after any change to the rules, the encoding, the text
  formats or the observation**; an unexpected diff means behaviour changed.
- `TestOpenSpielGenericTests` in `thud_test.cc`: `LoadGameTest`, `NoChanceOutcomesTest`,
  and `RandomSimTest` with 10 games from the opening and 10 from a position read from text
  (`RandomSimTestWithSpecificInitialState`), as chess's test runs them. Not
  `RandomSimTestWithUndo`: Thud has no `UndoAction` (a Phase 5 candidate).
- The short name in `open_spiel/python/tests/pyspiel_test.py` — done in Phase 3.
- Cross-checks against an existing implementation — done in Phase 3 against
  hexparrot/thudgame: perft counts on five positions (`perft_reference.py`) and every
  hand-written move expectation (`crosscheck_tests.py`).

### Phase 5 — Benchmark, then decide — **done** 2026-09-25

Measure legal-move generations/sec, random rollouts/sec, and MCTS simulations/sec. Record the
numbers in `PROGRESS.md`. These unblock both deferred decisions below. Also record how long
random rollouts run and how they end under the 200/800 limits, and compare with the
hexparrot-based measurements in `PROGRESS.md`, session 2, to confirm that proxy held.

**First measurements, 2026-09-25** (details and commands in `PROGRESS.md`, session 5), one
core, Release build (`build-release/`, `BUILD_TYPE=Release`):

- Random games with OpenSpiel's `benchmark_game` (every move: observation, legal actions,
  apply): **905,000 moves/s, 2,640 games/s** — chess 235,000 moves/s and 690 games/s at
  the same game length (341 moves), Go 479,000 moves/s.
- MCTS with random rollouts (`mcts_example`, 1,000 simulations per move): **about 3,500
  simulations/s** — chess about 760.
- **MCTS against MCTS** (OpenSpiel's `MCTSBot` with random rollouts, which `mcts_example`
  plays; summarised by `thud/experiments/mcts_games.py`). Plain MCTS needs nothing new;
  only "MCTS with an evaluation function" would need a Thud heuristic of our own. Random
  against random, the trolls win every game (mean dwarf margin -26.6). With MCTS:

  | Simulations per move | Games (seeds) | Dwarfs win (95% interval) | Mean dwarf margin | Turns, median (range) | Time on 10 cores |
  |---|---|---|---|---|---|
  | 1,000 | 10 (1-10) | 7 (40-89%) | +4.2 | 246 (137-336) | 53 s |
  | 5,000 | 20 (1-20) | **19** (76-99%) | **+17.7** | **91** (47-248) | 400 s, 1.4 s a move |

  Every game ends in a rout; neither limit comes into play. At 5,000 simulations 19 games
  end with no troll left, the dwarfs keeping a median 19.5 of their 32.
- **Why more search helps the dwarfs so much** (measured by replaying those games): the
  dwarfs have a median 330-470 legal moves a turn, the trolls 26-39, and OpenSpiel's MCTS
  tries every move once before any move twice (an unvisited move's UCT value is infinite,
  `mcts.cc:95`). At 1,000 simulations the dwarfs' search gives each move about 3 visits and
  took an available hurl 14% of the time; at 5,000, about 11 visits and 57%. The trolls'
  search has a blind spot that more simulations barely reach: when a hurl is on the board,
  hurls are about 0.2-0.4% of the dwarfs' moves, so a random rollout almost never plays it,
  while a troll capture is about 20% of the trolls' moves. Rollouts therefore punish a
  dwarf left open to trolls but hardly ever a troll left in a line of dwarfs; the trolls'
  search sees that threat only where its tree reaches the dwarfs' ~450 replies.
- **So these runs measure the searcher, not the balance of Thud** — the result follows
  the search budget: the trolls win every random game, the dwarfs 7 of 10 at 1,000
  simulations and 19 of 20 at 5,000. The official match (two battles, sides swapped, combined margin;
  `THUD_RULES.md` §9.5) does not need a balanced battle anyway.
- Random games' endings (`thud/experiments/random_endings.py`, our implementation): 98.6%
  routs, 1.4% the no-capture limit, none the turn limit; median 330 turns, mean 343 —
  matching hexparrot's 1.5% and 334 / 343, so the proxy held.
- The Release build is no faster than the Testing build for Thud (chess and Go gain about
  15%), which hints that the hot path is not compute-bound — for example the fresh
  `std::vector` each `LegalActions()` returns. Only a profiler can tell; see the
  optimisation candidates below.
- **Why Thud is about 4 times faster than chess per move** (the user expected them to be
  similar): chess must discard every pseudo-legal move that leaves its king in check, and
  handles castling, en passant, promotion and repeated positions; Thud's moves are plain
  walks along lines with nothing to discard afterwards. The observations are of similar
  size (chess 1,280 numbers, Thud 1,350), so they do not explain the difference.

**What the numbers mean for the deferred decisions** (the input for deciding them with
the user; decided 2026-09-25, see Phase 6):

- **Classical MCTS is cheap here:** a 10,000-simulation search per move takes about 3 s on
  one core, about 0.3 s spread over the 10 cores — before any optimisation.
- **But width, not only speed, limits it** (MCTS against MCTS above): at a dwarf node,
  plain UCT spends its first ~450 simulations trying every move once, and random rollouts
  hardly ever find a hurl. An evaluation function would replace the rollouts, not the
  width; that also needs move priors or pruning. AlphaZero's search has priors built in:
  under PUCT an unvisited move is worth its policy prior, not infinity (`mcts.cc:103-111`),
  and OpenSpiel's Python AlphaZero uses PUCT (`alpha_zero.py:201`).
- **Read AlphaZero's evaluation per side.** OpenSpiel's Python AlphaZero measures progress
  against exactly this searcher — plain UCT with random rollouts, alternating sides, at
  growing simulation counts (`alpha_zero.py:338-360`). Since that opponent's strength
  differs so much between dwarfs and trolls, an average over both sides would hide which
  side the network has learned.
- **For AlphaZero-style learning the move generator is not the bottleneck** — checked after
  the user asked whether thousands of simulations per move would make it one. Each
  simulation costs one network evaluation, one legal-move list and a few moves on a copied
  state, whatever the number of simulations, so the comparison is per simulation:
  - Our C++ does a whole move, observation included, in about 1 µs. A network evaluation
    on a 15x15 board costs about 0.2 GFLOP for a small net (6 residual blocks of 64
    channels) and about 10 GFLOP at AlphaZero's size (20 blocks of 256): milliseconds on a
    CPU core. AlphaZero itself spent its compute on self-play, not on training: "5,000
    first-generation TPUs to generate self-play games and 64 second-generation TPUs to
    train the neural networks" (arXiv 1712.01815).
  - Only a small net on a fast GPU, evaluating big batches, gets near microseconds per
    position. Then the next bottleneck is the Python side, still not our move generator:
    measured through pyspiel, fetching one observation into numpy takes about 50 µs, and a
    legal-move list about 11 µs in the opening, against about 1 µs for the move in C++ —
    and OpenSpiel's Python AlphaZero fetches observations that way
    (`state.observation_tensor()`, `alpha_zero.py:246`). The remedies there are a C++
    search (OpenSpiel's C++ AlphaZero, which needs LibTorch) or filling numpy buffers
    directly, not a faster move generator.
  - So the decision that matters is where the network runs; this machine has no CUDA GPU
    (`CLAUDE.md`). For classical MCTS with random rollouts, by contrast, the game's speed
    is the cost: each simulation plays a whole random game.

**Instrumentation, if we ever need it** (agreed with the user, 2026-09-25): these
measurements need none, since OpenSpiel's tools time the game through its public API.
Counters inside the code, if a question needs them, go under a compile-time switch that is
off by default — `if constexpr (kInstrumentation)`, with `kInstrumentation` set from
`-DTHUD_INSTRUMENTATION` in a separate build folder — as OpenSpiel's `SPIEL_DCHECK` checks
are compiled out of Release builds. `if constexpr` keeps the code compiled and type-checked
while it is off, so it cannot rot. A template parameter would be awkward: OpenSpiel creates
every Thud state through one registered factory. For "where does the time go", a sampling
profiler needs no code at all; `perf` and `valgrind` are not installed in this WSL (the user
can install them), `gprof` is but needs a `-pg` build.

**Status 2026-09-25: no further optimisation of the game logic for now** (the user's
conclusion from the measurements above). For AlphaZero-style learning the move generator
is not the bottleneck — the network is, and with a small network on a GPU, the Python side
— so a faster generator would not show in training time. Revisit only if we choose
classical MCTS with random rollouts, where each simulation plays a whole random game, and
its ~3,500 simulations/s per core prove too slow; then profile first.

**Optimisation candidates — only where the measurements point, never before** (user,
2026-09-24). Each is checked against the simple Phase 3 implementation, which stays as the
reference: the same perft counts, and identical legal moves in every position of many random
games. **Any faster move generator must still return the legal actions in ascending order**
(`spiel.h`: "The actions should be returned in ascending order"; checked in every state by
`RandomSimTest`'s `CheckLegalActionsAreSorted` and by the tests' `LegalMoves` helper).
Since the order is defined by the action numbers, every correct generator returns the same
list, however it finds the moves — so the tests' random games and the playthrough, which
pick from that list, stay the same. The current generator produces that order by
construction; a generator that finds moves in another order must sort them.

- **Track lines of dwarfs and trolls incrementally** instead of recounting a line behind a
  piece whenever moves are generated. Postponed as premature: lines are short, hurls and
  shoves are a small share of all moves (most work is walking empty squares for plain
  moves), and every move and capture would have to update lines along up to four axes —
  error-prone, and extra state to copy with every simulation.
- **`UndoAction`**, only if we search with OpenSpiel's alpha-beta: it can undo moves
  instead of copying the state at every node (`use_undo` in `algorithms/minimax.cc`).
  **Not for MCTS** (checked 2026-09-25): OpenSpiel's MCTS, in C++ and in Python (which its
  AlphaZero builds on), never replays a game from scratch — it copies the root state once
  per simulation and plays forward on the copy (`mcts.cc:182`, `mcts.py:329`). Undo would
  replace that one copy by undoing every move of the simulation, rollouts included, which
  is more work, not less. Undoing a capture means restoring up to 8 dwarfs, so each move
  would have to remember what it captured.

### Phase 6 — AlphaZero-style learning — **in progress** (from 2026-09-25)

**Decided with the user, 2026-09-25:** AlphaZero-style learning, not classical MCTS with an
evaluation function (Phase 5: the dwarfs' ~450 moves a turn limit plain UCT more than speed
does, and AlphaZero's policy priors address that); on this machine's CPU until everything
works, then a cloud GPU; and **OpenSpiel's C++ AlphaZero** (`open_spiel/algorithms/alpha_zero_torch/`,
built on LibTorch), not its Python one. Details and commands: `PROGRESS.md`, session 5.

#### Phase 6 roadmap and decision log

**This is the one place to look first**: the order of the work, what each stage must show
before the next begins, what is decided, and what waits for later. The details and the
evidence are in the sections below; update this list whenever a decision is made.

**The rule for the order** (user, 2026-09-26): see learning work on unmodified OpenSpiel
first; then change one thing at a time, each behind a switch whose default is upstream's
behaviour, each with its own tests, and each checked against the version before it at
equal machine time. Never stack changes that have not been shown to work.

| Stage | What | Must show before the next stage |
|---|---|---|
| 0 — done | C++ AlphaZero builds (LibTorch from the pip wheel); tic-tac-toe control; Thud smoke test; C++ against Python layout control; throughput | — |
| 1 — done 2026-09-27 | **Baseline learning with unmodified OpenSpiel**: 64 x 4, 100 against 400 simulations at equal machine time, with a **match program** (head-to-head, pairs of battles with sides swapped); with unchanged code, the **tree-reuse potential** and the dwarfs' **visit spread**. Result: learning works through step 14, then the dwarfs' play collapsed; 100 beat 400; the dwarfs' searches are pure breadth. Still running: run A2 (2x buffer) | Learning works: later networks beat earlier ones head-to-head, the evaluation per side improves. And which of 100 or 400 is better |
| 2 — done 2026-09-27 | **Our own copy** of the C++ AlphaZero and its MCTS in `thud/az/`, changed in nothing but its namespace (`open_spiel::thud_az`; parameter names identical) — made by `import_from_upstream.py` from upstream commit 540bba6e, built by `thud/az/build.sh`. Passed: (1) **textual identity** — `import_from_upstream.py --check`: all 12 copies are exactly a fresh import (a one-character change is caught), which covers code no test reaches (the batching queue, the trainer's threads, resuming); (2) **`identity_check.cc`** against upstream's code: equal network outputs, equal searches without and with root noise and with random rollouts and the solver (50 of 50 positions each), a checkpoint of ours loads in upstream's code with equal outputs, an equal learning step; every control differs — which covers the layout check; (3) the **tic-tac-toe control** through `az_trainer` (session 5's settings, 19 minutes): losses 1.55 / 0.50 → 1.27 / 0.07 → 0.96 / 0.04 at steps 1, 17, 26 (upstream's copy: 1.56 / 0.47 → 1.27 / 0.07 → 0.95 / 0.05), self-play draws 38% → 86% (upstream 41% → 83%), against MCTS at 40 / 126 / 400 simulations +0.36 / +0.26 / +0.12 at the end (upstream +0.34 / +0.38 / +0.06), draws against stronger MCTS. **Resuming works** in upstream's trainer (run A2) and in our copy (run C resumed 2026-09-28: step 17 continued from 352,553 positions with the full buffer and the latest model reloaded) | Textual identity; identical outputs, searches, training step; upstream loads our checkpoint; the tic-tac-toe control learns as upstream's did |
| 3a — done 2026-09-28 | **Value of untried moves** — urgent (user): the dwarfs' searches spread over nearly every move once the network judges them lost, so their policy targets stay flat, and the better the trolls get, the worse it becomes. **Implemented** in our copy (`thud/az/mcts.{h,cc}`, passed through the trainer as `--untried_move_value` and `--untried_move_reduction`, saved in `config.json`): three rules — `upstream` (0), `sibling_mean_minus_reduction` (the visit-weighted mean of the visited siblings minus 0.2 × √(their prior mass), KataGo's rule with the siblings' mean for the parent's value), `loss` (AlphaZero's) — at every node, **defaulting to sibling_mean_minus_reduction** (user, 2026-09-27; an exception to the upstream-default rule above). Tests: `untried_move_check.cc` (the formula on a hand-built node; on 85 self-play positions of run A's step 14 at 100 simulations the dwarfs' searches visit a median 99 moves with upstream's rule, 31 with the default, 1 with `loss`, which without root noise never leaves its first move; the trolls' 17, 23, 1); `identity_check.cc` still passes with `upstream`, and the default changes 50 of 50 searches. **Run C** (run A's settings with the default rule, 6 hours, 2026-09-27 20:31 to 2026-09-28 02:32): **no collapse, and much stronger dwarf play** — against the anchor step 8 +10.4, 12 +43.5, 14 +40.1, 16 +46.4 (run A: +18.9, +23.1, +26.1, +3.9); at step 16 as dwarfs +21.6 (A: −24.9); the dwarfs' searches visited a median 19 moves at every step (A: 95-99). Head-to-head at equal positions, C step 16 against A step 16: **+20.9** a pair (+17.3 to +24.4, 20 of 20 pairs); against A's best, step 14: −1.2 (−6.9 to +4.4) searching with upstream's rule, +3.9 (−0.2 to +7.9) with the new one — neither significant. Details in *First training runs* | Beats run A head-to-head at equal positions, and the dwarfs' searches narrow |
| 3b — next | **Playout cap randomisation**, with separate settings per side so the dwarfs get deep searches more often — or budgets scaled with the number of legal moves (user asked about more simulations for the dwarfs, 2026-09-27: after 3a, since until untried moves are valued differently extra simulations only widen a losing side's search; it tilts self-play towards the dwarfs, a bias to measure; the trolls' floor is ~100 simulations, as at 50 their ~60 moves would also get under one visit each; matches then fix the per-side budgets) | Beats the previous best setting head-to-head at equal time |
| 3c | **Tree reuse, one tree shared by both sides** — only if the reusable share stays large (own-tree reuse ~0. Shared-tree share = the most visited move's share, from the buffer statistics: with upstream's rule the dwarfs' 1-3%, the trolls' 57-77% while they won easily, then 17-20% (run A); **with the new rule ~13-14% for the dwarfs and ~13-17% for the trolls at 100 simulations (run C) — roughly 15% more simulations per search, modest against playout caps' 1.37x**; re-read as the policy sharpens, since the watcher records it every step). **Priority** (user asked, 2026-09-28): kept after 3b — today it buys ~15% more simulations a search, playout caps 1.37x in KataGo's measurement, and it is the larger change (a tree kept across moves and shared by both sides, root noise re-applied, a larger memory limit than the trainer's 10 MB); 3b's budget semantics (new simulations, or topping up to N visits) are chosen with reuse in mind. **Move it up** if the shared-tree share rises above ~30% for either side in the watcher's statistics, or when 3b is done | Same searches with reuse off; beats 3b's setting head-to-head |
| 4 | Settings: `uct_c`, root noise α (0.03 and 0.3 against 0.1), temperature drop | Each change beats the previous setting head-to-head |
| 5 | **Convolutional policy head** (user, 2026-09-26: later in the roadmap) — first on the layout-check harness, with an exhaustive test of the action-to-plane map. It is the one planned change that makes our networks unloadable by unmodified OpenSpiel | Faster policy learning on the harness; then no worse in self-play head-to-head |
| 6 | Cloud GPU, longer runs | Throughput measured there first |

**Later, when a trigger fires** (details in *Notes for later*):

- **Before publishing a network**: check that OpenSpiel's default LibTorch (2.3.0,
  `global_variables.sh:89`) loads checkpoints saved with our 2.10, and remember that any
  user needs our Thud implementation (this fork) — Thud is not upstream.

- **Grow the network** (user: "will almost certainly be needed") — when improvement
  flattens. KataGo's recipe, possible without changing upstream code.
- **The replay buffer** — tested 2x as run A2 (2026-09-27/28: slower learning, no collapse
  by 351,000 positions, weak evidence); left at the default for now (user, 2026-09-28).
  Retest on the triggers in *Notes for later*: events (before growing the network, faster
  self-play, playout caps, a change of reuse or learning rate) and — only when pronounced
  and persistent against a fixed reference — signals (stalled learning, a newer network
  losing, swinging results, a growing gap between fresh and training loss, game length).
  Every self-play position is archived.
- **Match-aware play** — after a strong single-battle agent (*Deferred decisions*).
- **End-of-battle limits** — re-measure once the engine plays strongly (*Deferred
  decisions*).
- **Gumbel AlphaZero** — only if few-simulation search remains the bottleneck after 3b
  and 3c; a large change.

**Decided** (details and reasons in the sections named):

| Date | Decision | By | Where |
|---|---|---|---|
| 2026-09-25 | AlphaZero-style learning with OpenSpiel's C++ AlphaZero; CPU first, then a cloud GPU | user | this phase |
| 2026-09-25 | Single battle, margins as returns; match-aware play later | user | *Deferred decisions* |
| 2026-09-25 | Resignation off | user | *Settings*, 5 |
| 2026-09-26 | Root noise α = 0.1; 0.03 and 0.3 to test later | user | *Settings*, 2 |
| 2026-09-26 | Defaults in `thud/experiments/az_thud.flags`, no upstream change | user | *Settings* |
| 2026-09-26 | Batched inference (32 actors, batches of 32, 2 inference threads, `OMP_NUM_THREADS=4`) | measured | *Throughput* |
| 2026-09-26 | Network 64 x 4 to start; grow later | user | *Settings*, 6 |
| 2026-09-26 | First runs: 100 against 400 simulations at equal time, decided head-to-head | user | *First training runs* |
| 2026-09-27 | Compare networks against a common anchor (run A's untrained start) as well, since they rise and fall | Claude, overnight | *First training runs* |
| 2026-09-27 | 100 simulations for now; revisit with playout caps, not assumed good enough later | user | *First training runs* |
| 2026-09-27 | The value of untried moves first in stage 3, as urgent | user | roadmap |
| 2026-09-27 | Increase the replay buffer: test 2x first as run A2 (on unmodified OpenSpiel); archive every self-play position from now on, for growing the network and training other heads | user (archive: Claude's proposal) | *Notes for later* |
| 2026-09-27 | Before stage 2: run A's learning curve (each checkpoint against the anchor) and the 400-simulation match, while the user is away | user | *First training runs* |
| 2026-09-27 | Stage 1's gate met (learning works through step 14); the dwarfs' collapse at steps 15-16 is the first problem for stage 3a | user confirmed | *First training runs* |
| 2026-09-27 | Stage 2 done: our copy in `thud/az/`, proven identical (textual check, identity check, tic-tac-toe control) | measured | roadmap |
| 2026-09-27 | Untried moves: three rules (upstream, siblings' mean minus reduction, loss) behind a flag, **default siblings' mean minus 0.2 × √(prior mass)**, applied at every node; test identity with upstream's rule too | user | roadmap, 3a |
| 2026-09-27 | Matches search with our copy (`az_match`, `untried=`, default upstream's rule, which reproduces the earlier matches exactly) | Claude | *First training runs* |
| 2026-09-28 | Stage 3a done: the siblings' mean minus a reduction stops run A's collapse and strengthens the dwarfs (run C) | measured, for the user to confirm | roadmap, 3a |
| 2026-09-28 | Replay buffer: leave it for now; retest when learning stalls and before training a bigger network from a smaller one's games | user | *Notes for later* |
| 2026-09-28 | Match intervals use Student's t (1.96 before: slightly too narrow) | Claude | *First training runs* |
| 2026-09-26 | Replay buffer: default 65,536 positions for now | user | *Notes for later* |
| 2026-09-26 | Untried moves: measure first | user | *Margins as the value target* |
| 2026-09-26 | Our own copy of the C++ AlphaZero and its MCTS, for the changes of stage 3 | user | *Deferred decisions* |
| 2026-09-26 | Convolutional head: later in the roadmap (stage 5), first on the layout-check harness — a Thud game is far easier to get accepted upstream than a new network type, so upstream compatibility matters | user | *Changes to the search and trainer* |

**Considered and set aside** (with the reason): the Python AlphaZero (a layout bug, and
slow; fallback note below); a 32 x 2 network (its convolutions see 11 x 11 squares);
two tree levels, move then capture (7% fewer head weights, an extra network call);
keeping every n-th position only (wastes two thirds of the searches; a large buffer is
the usual remedy); a 3 GB buffer now (network updates every 70 minutes to 4.7 hours);
resignation (fits margins badly).

**Why C++, not Python:**

- OpenSpiel's own docs: "The Python implementation uses one process per actor/evaluator,
  doesn't support batching for inference [...]. The C++ implementation, by contrast, uses
  threads, a shared cache, supports batched inference, and can do both inference and
  training on GPUs. As such the C++ implementation can take advantage of additional
  hardware and can train significantly faster." (`docs/alpha_zero.md:37-43`)
- Measured on Thud (`thud/experiments/az_speed.py`): OpenSpiel's Python AlphaZero search
  manages **12-24 simulations/s per process**, whatever the network's size, because its
  network costs 27-52 ms per position: upstream's `Model.inference` is never compiled — its
  `@jax.jit` is commented out as "much slower" (`model_linen.py:451`), plausibly because it
  wraps a function defined anew on every call, which recompiles. The Python search alone
  does 7,000-13,000 simulations/s; compiled once, the network takes 0.7-3.6 ms per position
  (resnets 32 x 2 to 128 x 6), and in compiled batches of 64, 0.07-0.68 ms.
- The Python models also convolve over the wrong axes (below); the C++ model reads the
  planes first, as OpenSpiel's observations are laid out (`alpha_zero_torch/model.cc:39`,
  `:82`).
- A GPU needs batched inference, which only the C++ version has — and the cloud phase is
  the one that matters for training.

**LibTorch on this ARM64 machine** (checked 2026-09-25; corrects the earlier assumption in
`CLAUDE.md`): PyTorch publishes no standalone LibTorch download for aarch64 Linux, but its
pip wheel `torch-2.10.0-cp314-cp314-manylinux_2_28_aarch64` (146 MB, the version OpenSpiel
pins for Python above 3.13, `python_extra_deps.sh:70`) contains LibTorch complete:
`libtorch.so`, `libtorch_cpu.so`, `libc10.so`, the C++ API headers, and
`share/cmake/Torch/TorchConfig.cmake` (`torch.utils.cmake_prefix_path`). It is built with
the C++11 string ABI (240 `__cxx11` symbols in `libc10.so`), as our clang build is.
OpenSpiel's CMake finds Torch with `find_package(Torch)` after appending its own download
folder to `CMAKE_PREFIX_PATH` (`open_spiel/CMakeLists.txt:255-258`), so the wheel's folder
can be passed in without editing upstream code. **Tried 2026-09-25, and it works**
unchanged: the C++ AlphaZero, its example programs and OpenSpiel's three LibTorch tests
build with LibTorch 2.10 from the wheel in 215 s, with no errors, no warnings and no patch
to upstream code (commands: `CLAUDE.md`, *Build and test*).

**Risks of the C++ route**, from its README: it "was a user contribution [...] and is not
regularly tested nor maintained by the core team [...] it may not build or work as
originally intended", and there are "known problems with the C++ PyTorch: inteferences
with pybind11 versions", with a workaround (OpenSpiel issue #966). OpenSpiel pins LibTorch
2.3.0 (`global_variables.sh:89`), while Python 3.14 has torch 2.10 and later only, so
LibTorch API changes may need small patches to upstream code — each with a §4(b) change
notice and listed in `CLAUDE.md`. **Neither risk has materialised** (2026-09-25): 2.10 needed
no patch, and `build-torch/` builds only the C++ targets we need; issue #966's pybind11
clash hits the Python bindings (`pyspiel`) in a LibTorch build, which we do not build
there — `pyspiel` keeps coming from `build/`. A full `make` in `build-torch/` was not tried.
One harmless runtime warning: PyTorch 2.10 deprecates the `uint8` legal-moves mask
(`kByte`, `vpnet.cc:183`, `:256`) that `torch::where` receives (`model.cc:210`, `:259`):
"where received a uint8 condition tensor".

**Time limit** (user: "a generous limit"): about two working sessions to get the C++
AlphaZero building on this machine and passing the tic-tac-toe control (step 4). Stop
earlier and ask the user if it needs LibTorch built from source, or more than small,
mechanical API-compatibility patches to upstream code. **Met 2026-09-25, within the
first session, with no patch at all.**

- [x] OpenSpiel's pinned JAX set in the venv (2026-09-25): needed by the Python route and
  its tests; numpy went from 2.5.3 to 2.3.5, because flax 0.12.3 requires numpy below 2.4.
- [x] Python AlphaZero control (2026-09-25): its model and evaluator tests pass, and its
  example trains on tic-tac-toe.
- [x] Install `torch==2.10.0` in the venv; build OpenSpiel with LibTorch and libnop in a
  separate `build-torch/` (Release), leaving `build/` and `build-release/` untouched
  (2026-09-25; libnop at `35e800d8`, as `install.sh` clones it).
- [x] Control (2026-09-25): OpenSpiel's LibTorch tests pass (`torch_integration_test`,
  `torch_model_test`, and `torch_vpnet_test`, which trains the resnet on all 4,520
  tic-tac-toe states and requires both losses below 0.1). Then `alpha_zero_torch_example`
  on tic-tac-toe with the Python example's settings (resnet 256 x 2, 40 simulations, 2,048
  new states per step, batches of 128, 2 actors, 2 evaluators; resignation off, as Python
  has none), 26 steps. It learns, much as the Python control did. That control was
  upstream's Python AlphaZero **unmodified, layout bug included**, which on tic-tac-toe
  probably costs little (not measured): the observation is (3, 3, 3)
  (`tic_tac_toe.h:154-155`), so the misread convolution still slides over a 3 x 3 grid with
  3 channels, and the centre window covers the whole board. So this shows that the C++
  pipeline learns. It neither measures the layout bug nor shows that the two separate
  implementations are equal:

  | | C++, step 1 → 17 → 26 | Python, step 1 → 17 |
  |---|---|---|
  | Policy / value loss | 1.56 / 0.47 → 1.27 / 0.07 → 0.95 / 0.05 | 1.53 / 0.23 → 1.03 / 0.03 |
  | Self-play draws | 41% → 51% → 83% | 42% → 80% |
  | Against MCTS, 40 / 126 / 400 simulations | −0.62 / −0.78 / −1.0 → +0.32 / +0.30 / +0.06 → +0.34 / +0.38 / +0.06 | about −0.5 each → +0.42 / +0.20 / +0.10 |
  | Against MCTS, 1,265 to 40,000 simulations | −1.0 → draws → draws | about −0.5 → draws |

  Against MCTS: mean result of the last 50 games per level, +1 a win. 21 minutes, **48 s
  per step against Python's 106 s** — training-bound (49 s per step for 64 batches, while
  the actors fill their queue in parallel), so this says little about Thud, where the
  search dominates: that is the throughput step.
- [x] Thud smoke test (2026-09-25): resnet 32 x 2, 50 simulations, 512 new states per
  step, 2 steps, 2 actors, 1 evaluator against MCTS with 50 simulations, resignation off.
  Everything works: games finish (self-play 114-230 moves, evaluation 110-296, far from
  the 800-turn limit),
  the learner trains (value loss 0.82 → 0.42), checkpoints and the replay buffer are
  written, the evaluator plays, and it exits cleanly. The trolls won all 10 self-play
  games by the full margin, and all 6 evaluation games whichever side AlphaZero took, so
  its average against MCTS was 0 — at this strength the side decides, which is why the
  evaluation is read per side. It was slow, 0.8 states/s (about 20 simulations/s per
  actor): see the throughput step.
- [x] **Thud layout control, C++ against Python** (user, 2026-09-25: compare fairly, with
  exactly the Python runs' parameters) — **passes, 2026-09-25: the C++ model reads Thud's
  board correctly.** `az_layout_check.py`'s supervised task, trained with upstream's C++
  model, unmodified (`VPNetModel::Learn`), on exactly the Python runs' data and settings:
  `az_layout_check.py --export` writes the random games (0-399 for training,
  10,000-10,099 for testing), checksums of the positions taken from them (every 7th turn),
  and each seed's batch order; the new `az_layout_check.cc` replays the games with Thud's
  C++ code, stops unless its 20,036 + 4,816 positions reproduce the checksums (a
  tampered count and a tampered plane sum were both caught), and trains on the same
  batches in the same order — resnet 32 x 2, learning rate 1e-3, weight decay 1e-4,
  1,500 steps of 128, seeds 1-3. Built as OpenSpiel's `docs/library.md` describes, with no
  CMake change: `build-shared/libopen_spiel.so` plus upstream's `model.cc` and `vpnet.cc`
  compiled alongside (`thud/experiments/build_az_program.sh az_layout_check`). On the
  held-out positions at step 1,500, mean of the seeds (range):

  | Task | Baseline | Python as is | Python planes last | **C++** |
  |---|---|---|---|---|
  | Dwarfs: is a hurl available? (value head) | 66.9% | 79.0% (77.4-81.0) | 98.0% (97.9-98.2) | **97.9%** (97.8-97.9) |
  | Trolls: is a capture available? (value head) | 91.8% | 91.2% (90.2-92.2) | 99.4% (99.3-99.5) | **99.0%** (98.0-99.5) |
  | Dwarfs: policy mass on hurls | 1.2% | 13.7% (9.5-16.0) | 68.1% (63.8-71.6) | **77.9%** (75.8-81.3) |

  The value heads match planes last step by step (the dwarfs' question at 97.4% by step
  500 in both). The C++ policy learns the hurls faster (47% at step 500, against 10% for
  planes last); the differences that cannot be matched without changing upstream code
  may explain it: the initialisation; the value loss (C++ `MSELoss`, `model.cc:353`;
  Python `optax.l2_loss`, which halves it, `model_linen.py:373`); the L2 term (C++
  `weight_decay * sum(w^2) / 2` over all but the biases, batch-norm scales included,
  `model.cc:356-373`; Python `weight_decay * sum(w^2)` without biases and batch norm,
  `model_linen.py:316-322`). Both runs dip on the trolls' question at step 1,250 (C++
  95.0%, planes last 97.7%), plausibly from the shared batch order. The trolls' policy
  scores about 99% everywhere, uninformative as before. Wall times are not comparable
  (different parallelism). There is no Python AlphaZero self-play run on Thud to compare
  against (too slow, above); for self-play the like-for-like comparison is the
  tic-tac-toe control.
- [x] **Throughput on this CPU — final report** (measured 2026-09-25 and -26): the
  numbers that decide the network size, the simulations and when to move to the cloud.

  **Bottom line.** Thud's search is network-bound, and batched inference is what makes
  this CPU usable: in the trainer itself, with the learner running, 100 simulations a
  move, a 32 x 2 network makes ~4,260 simulations/s (~550 games an hour), 64 x 4 ~1,710
  (~220 games an hour); a 128 x 6 network manages ~350 however it is run (~45 games an
  hour, benchmark only). This CPU suits small networks and small runs; anything larger
  belongs on a GPU.

  **How it was measured.**
  - The benchmark `thud/experiments/az_throughput.cc`, run by `az_throughput.sh`
    (sections 1-7), uses upstream's own pieces, unmodified: `MCTSBot` with PUCT as the
    actors use it, `VPNetEvaluator` with its shared cache (2^18 entries) and optional
    batching, and `VPNetModel`, freshly initialised (the speed does not depend on the
    weights). Each searcher plays its own random games and searches every third
    position, so no position repeats; 100 simulations a search. Rates are timed with a
    monotonic clock after a warm-up and split into two halves; that clock stops while the
    machine sleeps, so sleep cannot distort them.
  - The trainer (`alpha_zero_torch_example`, our flagfile, no evaluators, 8,192-position
    buffer used 4 times, learning every 2,048 new positions) was measured from its actors'
    logs, which record when each actor starts and when each game ends, with all its
    moves: moves per second per actor, averaged, times the number of actors. Not from the
    learner's steps, which see only whole games, in bursts. The trainer logs wall-clock
    time, so these runs need a machine that does not sleep: the final runs had sleep off
    and a pause detector alongside (wall clock against WSL's uptime every 10 s: no pause
    in 90 minutes).
  - Mains power, Windows power mode "best performance", 10 cores in WSL. Chess and
    Connect Four were measured alongside, as no published figures compare (the AlphaZero
    paper's are on TPUs with far larger networks).

  **Where the time goes: the network, and for Thud a large fixed cost per call.**

  | Network | Network alone, 1 thread, ms per position (one at a time / in batches of 64): Thud; chess; Connect Four | One search, 1 thread, simulations/s: Thud; chess; Connect Four |
  |---|---|---|
  | 32 x 2 | 5.5 / 0.63; 0.66 / 0.12; 0.18 / 0.06 | 180; 1,585; 10,400 |
  | 64 x 4 | 7.4 / 2.1; 1.2 / 0.51; 0.67 / 0.29 | 140; 835; 2,555 |
  | 128 x 6 | 14.9 / 9.3; 3.9 / 2.7; 3.1 / 1.6 | 65; 260; 450 |

  - Thud's search is almost all network time: 180 simulations/s is 5.6 ms each, one
    network call at a 15% cache-hit rate. The tree costs little despite ~200 legal moves.
  - One position at a time, Thud costs 8x chess with the small network but 4x with the
    large one; in batches of 64 the large network's ratio falls to 3.5x, the board-size
    ratio (225 against 64 squares). That fits a large fixed cost per call — plausibly
    the policy head's 450 x 19,800 weights, 35.6 MB read for every call — which batching
    spreads over the batch.
  - These are single-thread times, not comparable with `az_speed.py`'s Python figures
    (JAX used all cores).

  **Using all 10 cores.**
  - LibTorch's OpenMP backend gives every caller 10 threads, so several callers
    oversubscribe the cores: three searches at once ran at 16 simulations/s each by
    default and 128 each with `OMP_NUM_THREADS=1` (the smoke test's network, 50
    simulations, `alpha_zero_torch_game_example --verbose`).
  - Without batching, one thread per search: chess 64 x 4 grows from 835 to 4,816
    simulations/s at 10 searches (5.8x), Thud 64 x 4 from 140 to 620 (4.4x; 140, 267,
    427, 620 for 1, 2, 4, 10). Single 20 s runs showed dips (2 searches slower than 1)
    that 30 s reruns did not reproduce; their cause is unknown.
  - **Batched inference**: searchers hand their positions to inference threads, which
    evaluate up to a batch at once (waiting at most 1 ms for more,
    `vpevaluator.cc:127-134`). One configuration was best for all three sizes: 32
    searchers, batches of 32, 2 inference threads with `OMP_NUM_THREADS=4` — about 8
    cores on the network, 2 for the searches' tree work:

    | Network | Best without batching (10 searches) | Batched, 32 searchers, 2 x 4 threads | Gain |
    |---|---|---|---|
    | 32 x 2 | ~600 | 4,245 simulations/s | ~7x |
    | 64 x 4 | 620 | 1,690 | 2.7x |
    | 128 x 6 | 355 | ~350 (330-372 in every configuration) | none |

    All 10 cores on inference (2 x 5) halves it; 4 x 2 and 3 x 3 are slower; 48
    searchers add nothing steady. The large network gains nothing: batches cut its cost
    per position only from 14.9 to 9.3 ms, as its convolutions dominate.
  - No throttling: 10 searches for 5 minutes gave 569 and 592 simulations/s in the two
    halves.

  **In the trainer** (30 minutes each, 2026-09-26, no pauses):

  | Configuration | Games finished (20 logged actors) | Moves/s per actor (range) | Simulations/s | Benchmark |
  |---|---|---|---|---|
  | 64 x 4, batched: 32 actors, batches of 32, 2 x 4 threads | 59 | 0.534 (0.514-0.565) | 1,707 | 1,690 |
  | 64 x 4, unbatched: 10 actors, 1 thread each | 28 | 0.513 (0.502-0.530) | 513 | 620 |
  | 32 x 2, batched, as above | 154 | 1.331 (1.317-1.349) | 4,259 | 4,245 |

  - The batched benchmark carries over exactly, learner and all; unbatched loses ~17% to
    the learner, whose single thread takes a core (its steps took 51-139 s). In the
    trainer, batching gains 3.3x on 64 x 4.
  - The learner, per 1,024 positions: alone on 10 idle threads 0.54 s (32 x 2), 1.9 s
    (64 x 4), 14.9 s (128 x 6); in the batched trainer, beside the actors, ~0.9 s (32 x 2)
    and ~4 s (64 x 4) — its steps (up to 8 batches) took ~7 and ~32 s, against a new step
    every ~50 and ~120 s.
  - Inference batches averaged 24-29 of 32 (64 x 4) and 28-31 (32 x 2); cache hits
    12-28%. Self-play games ran 103-584 moves (medians 268-284).
  - One CPU reading during an earlier batched run showed only about half to two thirds of
    the CPU busy, yet the rate matched the benchmark: there may be headroom, not yet
    explored.
  - Stopping the trainer with SIGINT waits for every running game to finish, which takes
    minutes; killing it loses nothing but the games in progress.

  **What it means**, from the trainer's rates and ~275 moves a game:

  | Network | Positions per hour | Games per hour | Games per day |
  |---|---|---|---|
  | 32 x 2 | ~153,000 | ~550 | ~13,000 |
  | 64 x 4 | ~61,000 | ~220 | ~5,300 |
  | 128 x 6 (benchmark) | ~12,600 | ~45 | ~1,100 |

  AlphaZero's 800 simulations a move would cut these eightfold; game lengths will change
  as play improves.

  **Pitfalls met**, worth remembering:
  - **Sleep freezes WSL.** This laptop was set to sleep after 5 minutes without input
    even on mains power (the user set it to never on 2026-09-26). WSL's clocks stop
    meanwhile and the wall clock jumps on resume. Monotonic-clock measurements survive;
    wall-clock ones do not — a first trainer check lost ~39 and ~9 minutes to sleep, and
    only its pause-free games could be used (they agreed with the final numbers), and the
    benchmark follow-up's 82 minutes of wall time were probably mostly sleep, not the
    overrunning searches I first blamed. Detect pauses by comparing `/proc/uptime` with
    the wall clock.
  - Single short runs are noisy; slow configurations need windows long enough for several
    searches per searcher (the first 128 x 6 batched runs were not).
  - `alpha_zero_torch_example --verbose` prints nothing: the flag is never passed on.
  - `pkill -f`/`pgrep -f` with a pattern that also occurs in their own command line match
    themselves; use `-x` with the process name.

  **Still open:** whether the batched trainer's idle CPU can be used (more actors or
  inference threads, measured in the trainer); 128 x 6 was not run in the trainer.
- [ ] **First training runs on this CPU: 100 against 400 simulations a move** (user,
  2026-09-26), 64 x 4 (user), batched inference, our flagfile, equal machine time for
  both (for example 6 hours each: ~1,300 against ~330 games). Decided by **head-to-head
  matches** between the two final networks, in Thud's match format (pairs of battles with
  the sides swapped, margins summed), both searching with the same budget, ~50 matches to
  start, more until the interval on the summed margin excludes 0 — with
  `thud/experiments/az_match.cc` (written 2026-09-26; OpenSpiel's game example loads one
  network only; since 2026-09-27 built on our copy, `thud/az/build.sh
  thud/experiments/az_match.cc`, with `untried=` for the search's rule — its default,
  upstream's, reproduced 4 of 4 earlier pairs exactly, margins and game lengths): both battles of a pair start from the same few random moves, searches
  are deterministic, each network has its own batched evaluator. Controls: a network
  against itself gives exactly 0 for every pair (each pair's second battle replays the
  first with the roles swapped: equal lengths, opposite margins); two different networks
  give non-zero pairs (the clean run's network after 11 steps against its own start, 6
  pairs at 10 simulations: −6.3 points a pair, 95% interval −11.0 to −1.6 — too little to
  conclude anything).
  As diagnostics: the evaluation against MCTS **per side** (Phase 5: that opponent's
  strength differs greatly between dwarfs and trolls), as the share of games won and the
  mean margin; how widely both sides' searches spread their visits, and whether the
  dwarfs' policy sharpens (*Margins as the value target*, below), read from the saved
  replay buffers; the loss curves. OpenSpiel's evaluation does not record which side AlphaZero played
  (`EvalResults` averages per level only, `alpha_zero.cc:214-257`), but each evaluator
  log line pairs the game's returns (`Game N: Returns: r0 r1`) with AlphaZero's own
  (`AZ: a`), so its side is the player whose return is `a` — whenever the margin is not
  0; at 0 both sides scored 0 anyway.
  **Run A (100 simulations), 2026-09-26 19:49 to 2026-09-27 01:55** (`~/thud-runs/
  stage1_sims100/`): 6 hours of machine time without a pause, 16 learning steps
  (21,845 new positions each), 365,611 positions (~16.9 a second), 17 checkpoints.
  - Policy loss 4.78 → 4.09; value loss 0.03-0.32, rising while the games were close.
  - Self-play results swung: the trolls won 494 of 496 games in steps 1-6, the dwarfs
    137 of 256 in steps 7-9, the trolls 645 of 777 in steps 10-16; games got shorter
    late (~300 → 143 moves).
  - Searches (`buffer_stats.jsonl`): the dwarfs' stayed pure breadth throughout (95-99
    of 100 simulations on different moves); the trolls' narrowed to 2-3 moves (most
    visited ~0.7) while they won easily, and widened to 15-22 moves (~0.18) once games
    were close — the untried-move rule reacting to values near 0.
  - Evaluation against fixed MCTS (68 games, 5-6 per cell), mean margin for AlphaZero,
    first half → second half: as dwarfs −16.2 → +1.2 (MCTS 100 simulations), −20.9 →
    −1.2 (316), −23.2 → −7.4 (1,000); as trolls +13.5 → +26.7, +3.7 → +21.6, −9.2 →
    +15.0. Every cell improved.
  - Head-to-head (`az_match`, 100 simulations, 20 pairs, summed margin per pair for the
    final network): **against its untrained start +3.9** (95% interval +1.5 to +6.3; 11
    pairs won, 6 drawn, 3 lost; better on both sides by ~4 points); **against step 8
    −5.1** (−8.4 to −1.8; 2 won, 6 drawn, 12 lost; worse on both sides by ~2.5 points).
    So it learned, then its final network fell below step 8's — the learning curve
    (below) shows the fall came only at steps 15-16, after a steady rise to step 14.
    That is one of the triggers noted for the replay buffer (a newer network losing to an
    older one): with 65,536 positions (~240 games, the last ~3 learning steps' worth) the
    network trains only on its latest games. Other possible causes: the dwarfs'
    near-flat policy targets (breadth searches) — since tested: run C, which fixed them,
    did not collapse, and the 2x buffer alone (A2) changed less — and the noise of
    20-pair matches.
  - Because the networks rise and fall, one final checkpoint against another is a noisy
    comparison; so every network also plays a **common anchor**, run A's untrained start.

  **Run B (400 simulations), 2026-09-27 02:30 to 08:30** (`~/thud-runs/stage1_sims400/`):
  6 hours of machine time without a pause, **4 learning steps**, 88,078 positions (a
  quarter of run A's: with the default buffer a learning step needs 21,845 new
  positions). The trolls won all 339 self-play games. The dwarfs' searches, broad only at
  first (62 of 400 simulations on different moves with the untrained network), spread to
  220 of 400 moves (the most visited 3.8%) once the network judged the dwarfs lost — the
  same untried-move effect as at 100. Its evaluation against MCTS (25 games; opponents of
  400 to 4,000 simulations, so not comparable with run A's) was lost heavily as dwarfs.

  **Stage 1 result: at equal machine time, 100 simulations beat 400 — early in training.**
  All at 100 simulations, summed margin per pair for the first network named:

  | Match | Pairs | Margin per pair (95% interval) | Pairs won / drawn / lost |
  |---|---|---|---|
  | A final vs B final | 30 | **+2.9** (+1.5 to +4.2) | 15 / 14 / 1 |
  | A final vs the anchor (A's untrained start) | 20 | +3.9 (+1.5 to +6.3) | 11 / 6 / 3 |
  | B final vs the anchor | 20 | +1.9 (−4.2 to +8.1) | 10 / 3 / 7 |
  | A final vs A step 8 | 20 | −5.1 (−8.4 to −1.8) | 2 / 6 / 12 |

  Limits: the matches searched with 100 simulations, which may suit run A's network (a
  400-simulation match is still to run); run B learned only 4 times against 16; 6 hours
  is early training. In every match the dwarfs lost by 19-32 points whichever network
  played them: both sides improve slowly, and the trolls' side dominates the results.
  Stage 1's gate looked only half met at this point (A's step 8 beats its final); run A's
  learning curve, below, settled it: met (user confirmed).

  **Run A's learning curve (2026-09-27): it learned steadily, then its dwarf play
  collapsed in the last two steps.** Against the anchor (run A's untrained start), 100
  simulations, 20 pairs each, summed margin per pair:

  | Run A network | Against the anchor (95% interval) | Pairs won | As dwarfs | As trolls |
  |---|---|---|---|---|
  | step 8 | +18.9 (+15.8 to +22.1) | 20 of 20 | −4.0 | +22.9 |
  | step 12 | +23.1 (+19.0 to +27.2) | 20 of 20 | −2.3 | +25.4 |
  | step 14 | +26.1 (+23.0 to +29.3) | 20 of 20 | +0.1 | +26.0 |
  | step 16 (final) | +3.9 (+1.5 to +6.3) | 11 of 20 | −24.9 | +28.8 |

  "Final" really is step 16: the trainer saves `checkpoint--1` at every step next to the
  numbered one (`alpha_zero.cc:428-432`), and both were written at 01:37:01. So the rise
  is real and the collapse too — and it is **only the dwarfs' play**; the trolls' kept
  improving. A plausible mechanism, not proven: the trolls win (self-play in steps 15-16:
  260 of 285 games), the value network judges every dwarf position lost, the dwarfs'
  searches spread one visit per move, their moves are chosen by nearly equal one-ply
  values, they play worse, the trolls win more — the loop the user expected to keep the
  dwarfs' policy flat, and the one the value of untried moves (stage 3a) targets.
  Consequences: **stage 1's gate — learning works — is met** for steps 0-14; the
  collapse is the first problem for stage 3a to solve. **100 simulations beat 400 even
  more clearly** than the final-against-final match said: run B's final network scored
  +1.9 against the anchor, run A's from step 8 on +18.9 to +26.1. Run A2 (2x buffer)
  must run past A's step 16 (~5.8 hours, ~351,000 positions) to show whether a larger
  buffer prevents the collapse.

  **Run C (stage 3a: untried moves valued at the siblings' mean minus 0.2 × √(their
  prior mass)), 2026-09-27 20:31 to 2026-09-28 02:32** (`~/thud-runs/stage3a_sims100/`,
  our copy's `az_trainer`, run A's settings otherwise): 6 hours without a pause, 16
  learning steps, 352,553 positions (run A: 350,988). Its dwarf searches visited a median
  19 moves at every step (the most visited 13-32%), run A's 95-99 (1-3%). Policy loss
  4.77 → 4.05 (A: 4.78 → 4.09). Self-play stayed the trolls' (the dwarfs won at most 8
  games a step), without run A's swings. Matches, 100 simulations, 20 pairs, the
  searches using upstream's rule unless noted:

  | Match | Margin per pair (95% interval) | Pairs won / drawn / lost | First network as dwarfs / as trolls |
  |---|---|---|---|
  | C step 8 vs the anchor | +10.4 (+5.2 to +15.6) | 15 / 1 / 4 | +3.1 / +7.3 |
  | C step 12 vs the anchor | +43.5 (+40.0 to +46.9) | 20 / 0 / 0 | +19.9 / +23.6 |
  | C step 14 vs the anchor | +40.1 (+36.9 to +43.3) | 20 / 0 / 0 | +12.5 / +27.6 |
  | C step 16 vs the anchor | +46.4 (+42.8 to +50.0) | 20 / 0 / 0 | +21.6 / +24.8 |
  | C step 16 vs A step 16 (equal positions) | **+20.9** (+17.3 to +24.4) | 20 / 0 / 0 | −3.4 / +24.2 |
  | C step 16 vs A step 14 (A's best) | −1.2 (−6.6 to +4.1) | 6 / 2 / 12 | −2.4 / +1.1 |
  | the same, both searching with the new rule | +3.9 (−0.2 to +7.9) | 14 / 0 / 6 | −12.3 / +16.2 |

  **How sure** (the user asked, 2026-09-28): certain that the rule changes the search as
  intended (19 moves visited against 95-99, at every step); certain that C's final
  network is far stronger than A's final (+20.9, t-interval +17.0 to +24.7); **not shown**
  that C beats A's best checkpoint (−1.2, −6.9 to +4.4; with the new rule in both
  searches +3.9, −0.2 to +7.9 — first reported as +0.1 to +7.6 with the normal factor
  1.96, see below). And none of these intervals covers the variation between training
  runs: one run per rule cannot show that the rule trains better — run A's collapse may
  have been bad luck. The evidence is consistent with the rule and with its mechanism,
  and AlphaZero and KataGo both avoid valuing untried moves as even games, so it stays
  the default. **Run C resumed** (user, 2026-09-28 12:27) for another ~6 hours — does it
  keep improving without collapsing, and does resuming work in our copy (needed before
  growing the network) — as the baseline for stage 3b, then played against A's best in
  100 pairs; replicates of runs A and C (6 hours each) would settle the causal question,
  and are skipped unless it matters.

  **Match intervals until 2026-09-28 used the normal factor 1.96**; with 20 pairs Student's
  t (2.09 for 19 degrees of freedom) is right, about 7% wider. `az_match` uses t since.
  Recomputed: C step 16 against A step 16 +17.0 to +24.7; against the anchor +42.5 to
  +50.3; A step 14 against the anchor +22.8 to +29.5. No conclusion changed except that
  the new-rule match against A's best is no longer significant.
  Against the anchor C scored far more than A's best (+46 against +26) yet only matched it
  head-to-head: the anchor measures how badly a network beats an untrained one, which
  stops ranking strong networks — head-to-head matches are the measure between them.
  One run each: the regression might also have been avoided by chance.

  **Run A2 (2x buffer) resumed, 2026-09-28 04:35 to 07:16** with upstream's trainer from
  its `config.json`: it continued at step 6 from 263,020 positions with the full buffer
  and the latest model loaded — **resuming works** — to step 8 (step 9 was cut off by the
  time limit); no pause. Against the anchor: step 5 (219,092 positions) +9.4 (+5.5 to
  +13.4), step 8 (350,867) +16.6 (+11.5 to +21.7; as dwarfs −13.2, as trolls +29.8). At
  equal positions run A scored +18.9 at 176,000 and +23.1 at 264,000, but +3.9 at 351,000
  (the collapse). So the 2x buffer learned more slowly, and did not collapse by 351,000;
  one run, so weak evidence that the buffer helps stability — the untried-move rule
  changed far more.

- [ ] Cloud GPU (x86 + NVIDIA): OpenSpiel's documented setup, whose default LibTorch
  download is the CUDA build (`global_variables.sh:89`).

**Margins as the value target** (checked 2026-09-25). Thud's returns are the final margin
over 32 (`THUD_RULES.md` §7), which is the objective we want: a match is won on the
combined margin (the match question is in *Deferred decisions*).

- Learning handles it. Every position's value target is player 0's actual return, a real
  number (`alpha_zero.cc:363-369`), learned with squared error (`model.cc:353`) through a
  `tanh` output (`:204`), so the value head learns the **expected margin**. The code
  assumes only two players and zero-sum (`vpnet.cc:120-124`). Only monitoring statistics
  count by sign: wins and draws (`alpha_zero.cc:364`), and "value accuracy" as "predicted
  the winner" (`:378-379`). Evaluation averages are mean margins, so report the share of
  games won and the mean margin, per side.
- The search's `uct_c` scales only the exploration term (`mcts.cc:103-111`), so it absorbs
  the scale of the values; like every game, Thud needs it tuned, margins or not. What it
  cannot absorb is their level: **a move not yet tried counts as 0**, an even game
  (`mcts.cc:108`). An untried move with prior P scores `c * P * sqrt(N)`; a tried one its
  value v plus `c * P * sqrt(N) / (1 + n)`. So, from the formula (a hypothesis until
  measured), with 50 simulations and `uct_c` 2 (`c * sqrt(N)` ≈ 14):
  - The side that is **losing** (v near −0.8) keeps a tried move only while its bonus
    `14 * P / (1 + n)` exceeds about 0.8, and otherwise tries a new one. With a flat prior
    (an untrained network: P ≈ 0.002 over the dwarfs' ~450 moves) that means a new move
    every time, never deeper, and a near-flat policy target. A sharper prior mitigates
    this: a move with P = 0.3 is kept for about 5 visits. So this side mostly suffers
    early in training — when the dwarfs lose everything (smoke test).
  - The side that is **ahead** (v near +0.8) tries a new move only if `14 * P` exceeds v,
    here P above about 0.057: below the root, where no Dirichlet noise reaches, it follows
    its prior. **That is AlphaZero's own design**, and more strongly so: AlphaZero counts
    a move not yet tried as a **loss**. Its pseudocode gives an unvisited node the value 0
    on a 0-to-1 scale (`Node.value`, and `backpropagate` adds `1 - value` for the other
    side); Leela Chess Zero's reading is "AlphaZero just considered unvisited nodes as
    lost", −1 in their terms (lczero.org blog, 2018-12). MuZero's official pseudocode
    likewise gives an untried move the lowest value seen (`ucb_score`, `MinMaxStats`;
    arXiv 1911.08265, ancillary `pseudocode.py`). Exploration beyond the prior comes from
    the root noise, and the policy learns from the root's visit counts. Not a flaw.
  So what is OpenSpiel's own is the losing side's burst of breadth, from counting untried
  moves as even games; with AlphaZero's choice the losing side would also follow its prior
  (at v = −0.8 an untried move needs `14 * P` above 0.2, P above about 0.014). Other
  choices: KataGo uses the parent's value minus `0.2 * sqrt(prior mass of the moves already
  tried)` (arXiv 1902.10565, section 2); Leela Chess Zero offers a fixed value (default −1)
  or the parent's value minus a reduction. **Measured** (below, and in stage 1): the
  dwarfs' searches spread one visit per move once the network judges them lost, at 100
  and at 400 simulations. No flag in upstream changes it. **Done in our copy** (stage
  3a, 2026-09-27): three rules behind `--untried_move_value`, defaulting to the siblings'
  mean minus a reduction (user) — see the roadmap.

**Measured 2026-09-26: at 100 simulations the dwarfs' searches are pure breadth.** From
the replay buffer of the clean 64 x 4 trainer run (100 simulations, 11 small learning
steps; `thud/experiments/az_buffer_stats.cc`, which reads the visit counts every position
stores as its policy target):

| Per search (median) | Dwarfs | Trolls |
|---|---|---|
| Legal moves | 265 | 48 |
| Moves that got any visit | 99 of 100 simulations | 5 |
| The most visited move's share | 1% | 44% |
| Effective number of moves, exp(entropy) | 99 | 3.5 |

Every simulation of a dwarf search goes to a new move — never a second look — so the
dwarfs' policy targets are nearly flat, while the trolls' searches stay on their prior's
favourites. As predicted from the formula: the dwarfs' values are below 0 (their mean
return target was −0.125, losing by 4 points), so every untried move, counted as an even
game, outranks every tried one. A slightly trained network at 100 simulations. **Stage 1
showed it persists** (a watcher recorded these statistics after every learning step):
through all 16 steps of run A the dwarfs' searches visited 95-99 of 100 moves, and in run
B (400 simulations) 62 moves while untrained, 220 once the network judged the dwarfs lost.
The trolls' searches narrowed to 2-3 moves while they won easily and widened when games
were close.

**Settings to determine empirically** (user asked, 2026-09-25). `alpha_zero_torch_example`
defaults in brackets; the ones Thud makes most uncertain first:

1. `--uct_c` (2): the balance between the values and the exploration bonus — every game
   needs it tuned.
2. Root noise `--policy_alpha` (1) and `--policy_epsilon` (0.25, as AlphaZero's
   `root_exploration_fraction`). AlphaZero: "Dirichlet noise Dir(α) was added to the prior
   probabilities in the root node; this was scaled in inverse proportion to the
   approximate number of legal moves in a typical position, to a value of
   α={0.3,0.15,0.03} for chess, shogi and Go respectively" (arXiv 1712.01815,
   *Configuration*; checked 2026-09-25). Scaled the same way from chess (α times the move
   count constant, with chess's typical ~35 legal moves — a common estimate, not checked
   here), Thud's two very different counts — the dwarfs' median 330-470 and the trolls'
   26-39 (Phase 5) — give α about 0.02-0.03 for the dwarfs and 0.3-0.4 for the trolls,
   but the C++
   bot applies one α to whichever side is at the root (`alpha_zero.cc:178-179`,
   `mcts.cc:284-290`). α·n sets roughly how many moves the noise lands on: AlphaZero's ~10
   is a handful; the default α = 1 gives the dwarfs ~400, noise spread evenly like 25%
   uniform. **Chosen: α = 0.1** (user, 2026-09-25), the geometric-mean compromise — α·n
   about 40 for the dwarfs and 3 for the trolls, 4x and 3x off AlphaZero's rule.
   **Follow-up:** once `uct_c` and the simulations are chosen, compare 0.1 with 0.03 and
   0.3 in short runs, read per side; if the mismatch between the sides turns out to
   matter (for example, one side's policy collapses onto a few moves), the clean fix is
   an option to set α = k / (number of legal moves) per position — the paper's rule, for
   any game, and a candidate upstream PR, but an upstream change, so ask first.
3. `--max_simulations` (300): strength against cost; with 450 dwarf moves, few
   simulations cannot cover the moves. AlphaZero used 800 a move; KataGo recorded only
   searches of 600-1,000; "AlphaZero can fail to improve its policy network, if not
   visiting all actions at the root" (Danihelka et al., ICLR 2022, Gumbel AlphaZero).
   **100 for now** (user, 2026-09-27): it beat 400 at equal machine time early in
   training (stage 1); not assumed good enough later — revisit with playout caps (3b).
4. `--temperature_drop` (10 moves sampled before playing the best; AlphaZero's
   pseudocode has `num_sampling_moves = 30`, as mirrored on GitHub by mikolajblaz): Thud
   always starts from the same position, and games here last 100-450 moves, so this and
   the noise are what make games differ.
5. Resignation, `--cutoff_probability` (0.8) and `--cutoff_value` (0.95): **leave it off**
   (`--cutoff_probability=0`; discussed with the user 2026-09-25). It is **on by default**:
   each self-play game draws a uniform number in [0, 1) and may resign if it is below
   `cutoff_probability`, otherwise its threshold is `MaxUtility() + 1` = 2, which no value
   reaches (`alpha_zero.cc:202-203`); so 0 turns it off completely, and every run so far
   passed it. Evaluation games never resign (`:293`). With margins, 0.95 means
   an expected win by more than 30 of 32 points, so it would rarely fire anyway. Resigning
   also fits margins badly: a resigned game records the search's estimate as its result
   (`alpha_zero.cc:155-157`), and cuts off exactly what margins reward, playing a decided
   battle to the best score. Decided battles instead run into the end-of-battle limits
   (200 turns without a capture, 800 in all; `THUD_RULES.md` §6), whose results are real
   margins. The cost is self-play time spent in dead tails; measure the share of such
   turns in self-play (*Deferred decisions*, end-of-battle limits) — if it is large,
   lower the no-capture cap rather than resign. Off by default for us, without an
   upstream change, through our flagfile (below).
6. Network size, `--nn_width` (128) and `--nn_depth` (10): capacity against speed.
   **Chosen to start: 64 x 4** (user, 2026-09-26). 32 x 2 recognised hurls (97.9% in the
   layout control) but its convolutions see only 11 x 11 squares, 64 x 4's 19 x 19;
   AlphaZero used 19 residual blocks of 256 filters, KataGo grew from 6 x 96 to 20 x 256.
   Growing ours later: *Notes for later*, below.
7. Training: `--learning_rate` (1e-4), `--weight_decay` (1e-4), `--train_batch_size`
   (1,024), `--replay_buffer_size` (65,536) and `--replay_buffer_reuse` (3), the ratio of
   training to self-play.

Speed only, not what is learned: `--actors`, `--inference_batch_size`,
`--inference_threads`, `--inference_cache`, `OMP_NUM_THREADS`. Measurement only:
`--eval_levels`, `--evaluation_window` — but mind the cost: level n plays MCTS with
random rollouts and the solver at `max_simulations * 10^(n/2)` simulations
(`alpha_zero.cc:270-286`), so the default 7 levels at 300 simulations reach 300,000
per move, far too slow for Thud; use fewer levels. Not flags: the value of untried moves (above) and
the game's end-of-battle limits (*Deferred decisions*). There is not budget to tune all of
these: set 2 and 4 from the game's numbers (2 is set to 0.1, 5 stays off), then compare
short runs on the most sensitive (1, 3, 6) at equal cost, read per side against a fixed
opponent. So far 3 is 100 and 6 is 64 x 4 to start (stage 1).

**Our defaults live in `thud/experiments/az_thud.flags`**, passed with `--flagfile` (Abseil
flags, no upstream change): `--game=thud`, `--cutoff_probability=0`, `--policy_alpha=0.1`.
The last value of a flag wins, so flags after `--flagfile` override it; each run's
`config.json` records what it used (checked 2026-09-25, including that `#` comment lines
are ignored). `OMP_NUM_THREADS=1` is an environment variable and goes on the command line.
Add each setting here as it is decided.

**Notes for later** (user, 2026-09-26):

- **Grow the network once improvement flattens** — "this will almost certainly be needed"
  (user). KataGo "began with small nets and progressively increased their size,
  concurrently training the next larger size on the same data and switching when its
  average loss caught up to the smaller size" (6 x 96 to 20 x 256; trained on the data,
  not initialised from the smaller network's weights; arXiv 1902.10565, section 2).
  Feasible without changing any upstream code: on resuming, the trainer takes the
  network's shape from the run's `vpnet.pb` and reloads its checkpoint and replay buffer
  (`alpha_zero.cc:321-322`, `:533-536`, `:592`; the buffer's format is the header-only
  `utils/serializable_circular_buffer.h`). So: stop the run; a program of ours on the
  shared-library route trains the larger network on the saved buffer and writes the new
  `vpnet.pb` and checkpoint; resume. Needs one end-to-end test first.
- **Replay buffer: reconsider later** (user, 2026-09-28: leave it for now). Run A2 (2x)
  alone learned more slowly and had not collapsed by 351,000 positions; every position
  is archived from A2 on. The triggers for a retest come in two kinds (user, 2026-09-28:
  some happen in healthy training too, so they need caveats):

  **Event triggers** — a change we make; act when it happens, nothing to misread:
  - **Before training a bigger network from a smaller one's games** (user): the new
    network learns from stored positions, so how many, and from how many network
    versions, matters (the archive keeps them all).
  - **Much faster self-play**, e.g. a GPU: the same number of positions then covers far
    less playing time and fewer network versions; rescale the buffer to keep its span.
  - **Playout caps** (stage 3b): only full-search positions are recorded, so the same
    buffer spans more games.
  - **A change of reuse, learning rate or batch size**: in OpenSpiel these are tied to the
    buffer (a learning step every `buffer / reuse` new positions).

  **Signal triggers** — measurements that also move in healthy training; each counts
  only if the effect is **pronounced and persistent**, measured against a fixed
  reference (the anchor, or a ladder of past checkpoints) with intervals, over several
  learning steps — never one match or one step:
  - **Learning stalls** (user): plateaus come and go in healthy training. Counts if
    several consecutive checkpoints show no gain head-to-head against earlier ones, and
    after ruling out other causes (learning rate, capacity, the search).
  - **A newer network loses to an older one** (user's caveat): happens in healthy
    training — noise, and cycling between strategies, as AlphaGo Zero's gating assumed.
    Counts if the drop is large (the interval well clear of 0) and lasts, several
    checkpoints below an earlier peak; run A's collapse (+26.1 at step 14 against the
    anchor, +3.9 at step 16) was large, though seen over only two steps.
  - **Swinging self-play results**: win shares swing naturally, as one side's
    improvement shifts them until the other adapts. Counts if large swings recur without
    net progress against the fixed reference.
  - **The value loss on fresh games pulling away from the training loss**: some gap is
    normal (the training loss is on seen positions). Counts if the gap grows steadily
    over several steps while the fresh-game loss rises. Needs a program of ours (the
    trainer does not report it).
  - **Game length changing a lot**: lengths fluctuate with style from step to step (run
    A: 143-310 moves). Counts if the change is large (say a third) and lasts, since the
    buffer then holds correspondingly more or fewer games.

- **Replay buffer size: to be increased** (user, 2026-09-27; first kept at the default
  65,536 positions, ~240 games). Run A's step 8 beat its final network, one of the
  triggers below — though regressions are a known part of self-play: AlphaGo Zero let a
  new network generate games only after it won "by a margin of 55%", while AlphaZero
  "simply maintains a single neural network that is updated continually", as OpenSpiel
  does (arXiv 1712.01815). Two jobs to keep apart: the **training window** (what the
  running network learns from; run A2 tests 2x, 131,072 positions at reuse 3 — a learning
  step every ~43 minutes, the learner's CPU share unchanged) and a **data archive** for
  growing the network or training another head later, which must learn from far more
  than the last ~240 games: the trainer rewrites its buffer file each step and replaces
  the whole buffer every `reuse` steps, so copying the file every `reuse` steps keeps
  every position once (~10 KB each, ~15 GB a day at 64 x 4). **Run A2 (2x) result,
  2026-09-28**: it learned more slowly per position (half as many learning steps) and had
  not collapsed by 351,000 positions, where run A did — weak evidence, one run each; the
  untried-move rule changed far more (run C). The notes from when the default was kept:
  OpenSpiel learns once every `buffer / reuse` new positions (`alpha_zero.cc:324`) and
  rewrites the buffer file each time (`:403`, ~10.3 KB a position), so 3 GB (~290,000
  positions) would update the network only every ~70 minutes at 100 simulations, ~4.7
  hours at 400. Revisit it when: the value loss on new games, measured before they enter
  the buffer, pulls away from the training loss (memorising a few hundred games); or a
  newer network loses to an older checkpoint head-to-head (forgetting); or the evaluation
  oscillates. OpenSpiel reports neither of the first two; a program of ours can compute
  them from checkpoints and the saved buffer. Enlarging is a flag, but slows the updates
  unless reuse rises — and resuming with another size fails: `LoadBuffer` refuses a file
  saved with a different maximum size (`utils/serializable_circular_buffer.h`), so a
  program of ours must rewrite the file's recorded size first (easy on the
  shared-library route).

**Changes to the search and trainer** (found 2026-09-26; each needs changed code, so each
goes into our own copy, decided 2026-09-26 — stages 3 and 5 of the roadmap, one at a time):

- **Playout cap randomisation** (KataGo): on a random 25% of turns a full search (e.g.
  400-600), recorded for training; quick searches (e.g. 100) on the rest, not recorded.
  KataGo measured it as 1.37x training efficiency, better than every fixed cap from 100
  to 600 (arXiv 1902.10565, Table 2). Changes the trainer (`alpha_zero.cc`). With
  separate settings per side (user, 2026-09-26), the dwarfs, with ~10x the trolls' moves,
  can get full searches more often, or larger ones; since only full searches become
  training targets, that also gives the dwarfs more targets. Caution: whatever changes
  the sides' playing strength in self-play tilts its results, and so the value targets.
  KataGo also turns root noise off in quick searches: "For fast searches, we also disable
  Dirichlet noise and other explorative settings, maximizing strength" (section 3.1;
  checked 2026-09-28; it says nothing on how moves are chosen after each kind, nor on its
  tree reuse in this context).
  **Proposal for stage 3b** (Claude, 2026-09-28, to decide with the user):
  1. Each move, with probability `p` for the side to move, a full search of `N`
     simulations, recorded for training (policy and value target, as now); otherwise a
     quick search of `n` simulations, not recorded, without root noise. Moves chosen as
     now after either (sampled from the visits for the first 10 moves, then the most
     visited), keeping the openings varied.
  2. Settings per side; start: trolls `p` 0.25, dwarfs `p` 0.5 (more targets for the side
     with ~300 moves), `N` 400 and `n` 100 for both — KataGo's `p` and `n`, `N` below its
     600 for this CPU. About 2.1x run C's simulations per move, so ~half the games an
     hour, and ~38% of positions recorded. A cheaper variant: `N` 300, `n` 50 (~1.4x).
  3. A switch, off by default (every search full and recorded: today's behaviour), like
     the roadmap's rule — unless the user prefers otherwise, as for 3a.
  4. Budget semantics: without tree reuse a search's budget is its new simulations; when
     3c comes, "top up to `N` visits" (KataGo stops "when the tree reaches a cap of N
     nodes") fits the caps and saves compute — decided then.
  5. Learning steps count recorded positions only, so they come less often per game.
  6. Judged against run C at equal machine time (C's first 6 hours: step 16),
     head-to-head in 100 pairs with the new rule for untried moves in both searches, plus
     the search statistics per side.
  7. Tests: the switch off reproduces today's trainer (the tic-tac-toe control, and the
     share of recorded positions = 1); on, the share of full searches per side ≈ `p`,
     only they are recorded, quick ones have no root noise.
- **Tree reuse**: the played move's subtree, with its statistics, becomes the next root
  (AlphaGo Zero did this; OpenSpiel builds a fresh tree each move, `mcts.cc:356`, and
  `RestartAt` does nothing, `mcts.h:173`). The dwarfs would profit through the trolls'
  search (user): the trolls' few moves concentrate visits, and under the chosen one the
  trolls' search has already explored the dwarfs' replies — so one tree shared by both
  sides, not one per side as OpenSpiel's self-play has. The potential can be measured
  with unchanged code first: the share of root visits in the move played, per side.
  Root noise must be re-applied to the reused root. Changes the search and the trainer.
  Expected (user, 2026-09-28): the share grows as the policy sharpens (seen once: run A's
  trolls held 57-77% while winning easily; run C held 13-17% throughout, not yet); with
  per-side budgets (3b) the dwarfs' extra simulations also serve the trolls' next search
  (their subtree holds the dwarfs' exploration of the trolls' replies: at 300 dwarf
  simulations and 14%, ~42 inherited); with equal budgets, the side whose search is more
  focused helps the other. A design choice decides what reuse buys: "N new simulations
  per search" adds visits, "top up to N visits" (AlphaGo Zero's style) saves compute.
  **Measured 2026-09-26** (`thud/experiments/az_reuse.cc`, 64 x 4, fresh and slightly
  trained networks, 100 / 400 / 1,000 simulations): a side's own tree two plies later
  would inherit almost nothing (0.0-0.4% of its visits in every configuration), so only a
  tree shared by both sides is worth building. The program's absolute shares for the
  shared tree (dwarfs 5-29%, trolls 8-14%, falling with more simulations) come from
  positions after up to 200 random moves, and do not match self-play: there the dwarfs'
  searches spread their visits one per move. The trainer's own buffers are the right
  source — after the first 10 moves the played move is the most visited, so its share is
  what the next search, by the other side, would inherit: in the clean 64 x 4 run's buffer
  (100 simulations) the trolls' played move held a median 44% of the visits (what the
  dwarfs would inherit, as the user expected), the dwarfs' 1% (the trolls would inherit
  nothing). Stage 1's watcher records this after every learning step.
- **A convolutional policy head**: 120 move planes (8 directions x 14 distances + 8
  captures) over the board instead of the 450 x 19,800 linear layer — 64 x 4 would drop
  from ~9.2 to ~0.3 million weights; speed gain modest when batched (roughly 15% for
  64 x 4), larger one at a time; the hoped-for gain is faster policy learning, as
  AlphaZero's heads were convolutional. Test first on the layout-check harness (policy
  mass on hurls, 77.9% at step 1,500 with today's head), with an exhaustive test of the
  action-to-plane map. **Compatibility:** a network with this head is a new model type,
  which unmodified OpenSpiel cannot build or load (user asked, 2026-09-26); every other
  planned change leaves the network file as upstream's. If that matters, publish our copy
  with the network, or distil the final network into a standard resnet on our self-play
  data (the machinery of network growing). Two tree levels (move, then capture yes/no)
  would not help: 18,482 actions instead of 19,800, 7% fewer head weights, and an extra
  network call per choice.
- **The value of untried moves**: the parent's value minus a reduction (KataGo, Leela
  Chess Zero) or a loss (AlphaZero) instead of OpenSpiel's 0. The first runs showed the
  dwarfs' searches far too broad (*Margins as the value target*), so it is stage 3a,
  first (user, 2026-09-27).

**If we ever fall back to the Python route** (user, 2026-09-25): first **re-verify the
layout-bug findings below**, then put the fix into **one concise PR, including experiments
that verify its correctness — for example on tic-tac-toe, chess and Thud**. An upstream PR
can carry only the tic-tac-toe and chess experiments, since Thud is not upstream (see the
pre-PR checklist); the Thud evidence would stay in this fork. The uncompiled inference
(above) is a separate flaw and would need its own fix.

**The Python models' layout bug** (found 2026-09-25):

- Both Python models reshape each observation to `game.observation_tensor_shape()` and pass
  it straight to flax's `nn.Conv` (`model_linen.py:209`, `:217`; `model_nnx.py:311`,
  `:322`). OpenSpiel's observations are planes first ("First dimension interpreted as
  selecting from 2D planes", `docs/api_reference/game_observation_tensor_shape.md:28`);
  `nn.Conv` takes the last axis as channels. Nothing transposes on the way
  (`alpha_zero.py:246` → `:455` → `:589` → `:144` → `model_linen.py:209`). For Thud the
  first kernel has 15 input channels — the board's columns — instead of 6, and the 3x3
  windows slide over (plane, row).
- A regression: the TensorFlow model of March 2020 transposed planes-first input
  (`data_format="channels_first"`, `permute_dimensions(torso, (0, 2, 3, 1))`, passed by the
  tic-tac-toe example); the rewrite of 2020-03-23 (`bcdb0b44`, "Switch to an explicit graph
  model") dropped it, and the 2025 flax port kept that. No upstream issue or PR mentions it
  (searched 2026-09-25).
- It hurts learning (`thud/experiments/az_layout_check.py`): OpenSpiel's own model and
  update step, unmodified, resnet 32 x 2, 1,500 steps of 128, 3 seeds per variant, the same
  positions from random games — fed as they are, or transposed to planes last with a
  matching input shape. On held-out positions, mean of the seeds (range):

  | Task | Baseline | As is | Planes last |
  |---|---|---|---|
  | Dwarfs: is a hurl available? (value head) | 66.9% | 79.0% (77.4-81.0) | **98.0%** (97.9-98.2) |
  | Trolls: is a capture available? (value head) | 91.8% | 91.2% | **99.4%** (99.3-99.5) |
  | Dwarfs: policy mass on hurls | 1.2% | 13.7% | **68.1%** |

  The baseline is a constant guess of the more frequent answer, or a uniform policy. As is,
  the model plateaus by step 750; planes last, it reaches 97% on the dwarfs' question by
  step 500. Planes last costs about 1.7 times as long per step and still wins at equal
  time. (The trolls' policy scored about 99% in both: troll capture steps have action IDs
  of their own, so the policy needs no board reading to prefer them.) Limits: a supervised
  task on a small network, not full training; a deeper network might partly compensate.
- Why nobody noticed is a guess: it still learns, and on a 3x3 board one 3x3 window covers
  the whole board anyway; the damage grows with the board, and serious users probably
  train with the C++ version.

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

**3. The PR branch contains only these files.** Nothing from `CLAUDE.md` or `thud/` — in
particular not `thud/experiments/`, whose scripts run hexparrot/thudgame's engine (cloned
outside this repo) to produce the reference numbers our tests quote.

- `open_spiel/games/thud/{thud.h,thud.cc,thud_test.cc}`
- `open_spiel/games/CMakeLists.txt` — game sources plus the test target
- `open_spiel/python/tests/pyspiel_test.py` — the `thud` short name
- `open_spiel/integration_tests/playthroughs/thud.txt` — generated, not hand-written
- `docs/games.md` — one table row: status badge, name, players, deterministic, perfect-info,
  description

The upstreamed files must not point into `thud/` either. Comments in
`open_spiel/games/thud/` cite `THUD_RULES.md` and its section numbers, `thud/PLAN.md`, and
the scripts in `thud/experiments/`. Rewrite them to stand on their own: keep the facts (for
example, that `TestPerft`'s counts come from hexparrot/thudgame at commit 7b171108) and drop
the pointers. This lists them:
`grep -rniE '[^/]thud/|THUD_RULES|section [0-9]' open_spiel/games/thud/`

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
| ~~Classical MCTS + evaluation function vs AlphaZero-style learning~~ | **Decided 2026-09-25 (user): AlphaZero-style learning, with OpenSpiel's C++ AlphaZero** — Phase 6. The Python AlphaZero's input-layout concern (noticed 2026-09-24) is verified and recorded there; any fix belongs in the training code, not in `ObservationTensorShape`, since every upstream board game reports planes first, as we do. |
| Local CPU training vs rented cloud GPU | **Direction 2026-09-25 (user): this CPU until everything works, then a cloud GPU.** When to switch follows from Phase 6's throughput step. For AlphaZero-style training the network, not the game, dominates the cost, and this machine has no CUDA GPU. |
| ~~`OPEN_SPIEL_BUILD_WITH_LIBTORCH` on aarch64 (needed for C++ AlphaZero)~~ | **Works (2026-09-25)**, with LibTorch 2.10 from PyTorch's aarch64 pip wheel and no change to upstream code; the tic-tac-toe control learns — Phase 6. |
| **Match-aware play** (raised 2026-09-25) | The single-battle agent first (**user, 2026-09-25**). A match is two battles with the sides swapped, won on the combined margin (`THUD_RULES.md` §9.5). Our agent maximises its **expected margin** in each battle, which maximises the expected match total, but not quite the chance of **winning the match**: in the second battle the right play depends on the margin carried over (leading by 10, the trolls only need to lose by less than 10, and should play safe). To learn that, the network needs the battle number and the carried margin as input — the battle number alone is not enough — for example as two constant observation planes. A separate network for the second battle would need the carried margin just the same. Two designs: (a) one OpenSpiel game per match, with the returns from the match result — exact, but twice as long, and the first battle learns from a target that includes the second's noise; (b) a single battle with the carried margin as a game parameter, sampled during self-play, as KataGo does with komi (its network input includes "Komi / 15.0 (current player's perspective)", arXiv 1902.10565, appendix A.1) — simpler, but the first battle is not strictly optimised for the match. A pure win/loss target drops any incentive to score once the match is decided, so a mix of the match result and a small score term is likely better (KataGo's search maximises the sum of a win utility and a bounded score utility, `u_score(x) = c_score * (2/π) * arctan((x - x_0) / b)` with `c_score` 0.5; same paper, appendix F). A rules and representation change, so decided with the user. |
| ~~**Our own copy of OpenSpiel's C++ AlphaZero and its MCTS**~~ | **Decided 2026-09-26 (user): yes**, for stage 3 of the roadmap. Needed for any change inside the search or the trainer (*Changes to the search and trainer*, Phase 6), not for growing the network. About 2,400 lines of AlphaZero plus 714 of MCTS, copied into `thud/` under our own namespace (so they cannot clash with the originals inside `libopen_spiel.so`), their Apache headers kept with our change notices, built with `build_az_program.sh`; upstream stays untouched. Cost: we maintain it, and upstream fixes no longer flow in (the C++ AlphaZero is unmaintained upstream anyway). |
| Whether to attempt upstreaming to OpenSpiel | Thud is commercially published; copyright question unresolved |
| **End-of-battle limits** (`THUD_RULES.md` §6: no-capture cap and hard turn cap) — **re-evaluate once our engine plays strongly** | A strong engine of our own. The current defaults rest on proxies only (random play and hexparrot's heuristic AI; see `PROGRESS.md`, session 2), and strong play may stall in ways the proxies never do. Re-measure on our engine's self-play: how games end (rout, no legal move, no-capture cap, hard cap), the longest no-capture stretches, and the "dead tail" after the last capture. Lower the no-capture cap if games regularly sit out the whole cap; raise it if it cuts off real manoeuvring. Both are game parameters, so no code change is needed. |
