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
| 1 — done 2026-09-27 | **Baseline learning with unmodified OpenSpiel**: 64 x 4, 100 against 400 simulations at equal machine time, with a **match program** (head-to-head, pairs of battles with sides swapped); with unchanged code, the **tree-reuse potential** and the dwarfs' **visit spread**. Result: learning works through step 14, then the dwarfs' play collapsed; 100 beat 400; the dwarfs' searches are pure breadth. Run A2 (2x buffer), since done (2026-09-28): slower learning, no collapse | Learning works: later networks beat earlier ones head-to-head, the evaluation per side improves. And which of 100 or 400 is better |
| 2 — done 2026-09-27 | **Our own copy** of the C++ AlphaZero and its MCTS in `thud/az/`, changed in nothing but its namespace (`open_spiel::thud_az`; parameter names identical) — made by `import_from_upstream.py` from upstream commit 540bba6e, built by `thud/az/build.sh`. Passed: (1) **textual identity** — `import_from_upstream.py --check`: all 12 copies are exactly a fresh import (a one-character change is caught), which covers code no test reaches (the batching queue, the trainer's threads, resuming); (2) **`identity_check.cc`** against upstream's code: equal network outputs, equal searches without and with root noise and with random rollouts and the solver (50 of 50 positions each), a checkpoint of ours loads in upstream's code with equal outputs, an equal learning step; every control differs — which covers the layout check; (3) the **tic-tac-toe control** through `az_trainer` (session 5's settings, 19 minutes): losses 1.55 / 0.50 → 1.27 / 0.07 → 0.96 / 0.04 at steps 1, 17, 26 (upstream's copy: 1.56 / 0.47 → 1.27 / 0.07 → 0.95 / 0.05), self-play draws 38% → 86% (upstream 41% → 83%), against MCTS at 40 / 126 / 400 simulations +0.36 / +0.26 / +0.12 at the end (upstream +0.34 / +0.38 / +0.06), draws against stronger MCTS. **Resuming works** in upstream's trainer (run A2) and in our copy (run C resumed 2026-09-28: step 17 continued from 352,553 positions with the full buffer and the latest model reloaded) | Textual identity; identical outputs, searches, training step; upstream loads our checkpoint; the tic-tac-toe control learns as upstream's did |
| 3a — done 2026-09-28 | **Value of untried moves** — urgent (user): the dwarfs' searches spread over nearly every move once the network judges them lost, so their policy targets stay flat, and the better the trolls get, the worse it becomes. **Implemented** in our copy (`thud/az/mcts.{h,cc}`, passed through the trainer as `--untried_move_value` and `--untried_move_reduction`, saved in `config.json`): three rules — `upstream` (0), `sibling_mean_minus_reduction` (the visit-weighted mean of the visited siblings minus 0.2 × √(their prior mass), KataGo's rule with the siblings' mean for the parent's value), `loss` (AlphaZero's) — at every node, **defaulting to sibling_mean_minus_reduction** (user, 2026-09-27; an exception to the upstream-default rule above). Tests: `untried_move_check.cc` (the formula on a hand-built node; on 85 self-play positions of run A's step 14 at 100 simulations the dwarfs' searches visit a median 99 moves with upstream's rule, 31 with the default, 1 with `loss`, which without root noise never leaves its first move; the trolls' 17, 23, 1); `identity_check.cc` still passes with `upstream`, and the default changes 50 of 50 searches. **Run C** (run A's settings with the default rule, 6 hours, 2026-09-27 20:31 to 2026-09-28 02:32): **no collapse, and much stronger dwarf play** — against the anchor step 8 +10.4, 12 +43.5, 14 +40.1, 16 +46.4 (run A: +18.9, +23.1, +26.1, +3.9); at step 16 as dwarfs +21.6 (A: −24.9); the dwarfs' searches visited a median 19 moves at every step (A: 95-99). Head-to-head at equal positions, C step 16 against A step 16: **+20.9** a pair (+17.3 to +24.4, 20 of 20 pairs); against A's best, step 14: −1.2 (−6.9 to +4.4) searching with upstream's rule, +3.9 (−0.2 to +7.9) with the new one — neither significant. Details in *First training runs*. **Continued to step 29** (2026-09-29): even with step 16 head-to-head (+1.1), but its dwarf play much weaker against A step 14 (−20.0 as dwarfs, step 16 −2.4; −11.5 a pair) and the anchor, with upstream's rule in the searches — forgetting or the evaluation's rule? (*First training runs*). **Evaluated as trained** (Step 1, 2026-09-29): no collapse (C's dwarfs never below −0.6 against the anchor), but a drift from step 16 on — step 29 beats step 16 (+4.1), yet its dwarfs lost 10 points against the anchor and it loses to A step 14 (−8.5): specialisation; Step 2 (the buffer) indicated — the quick forgetting check found the dwarfs' move probabilities drifting; **Step 2** (run D, a 4x memory, 2026-09-30): no policy drift, D29 beats C29 (+7.3) and C16 (+8.4), still loses to A14 (−5.4) — forgetting explains part; **run E** (D continued with a 7x memory to step 44, 2026-10-01): E44 beats D29 (+9.7), its dwarfs recover against the anchor (+8.7, D29 +3.2), no forgetting, still loses to A14 (−3.5) | Beats run A head-to-head at equal positions, and the dwarfs' searches narrow |
| 3b — after Step 1 (below) | **Playout cap randomisation**, with settings per side in the code but **equal to start** (proposal in *Changes to the search and trainer*) — unless the **uneven match** (C's final network, step 29, against itself, 400 against 100 simulations, parked 2026-09-29: per-side settings become a setting to tune later) shows that extra simulations gain the dwarfs clearly more than the trolls; then the dwarfs get full searches more often, or budgets scale with the number of legal moves. (User asked about more simulations for the dwarfs, 2026-09-27. The case for them was upstream's rule, which spread the dwarfs' searches over every move; since 3a their searches are about as focused as the trolls' — but over 7-11% of their legal moves, the trolls' over 34-78% (medians in run C's buffers), so whether more simulations help them more is measured, not assumed. Unequal budgets tilt self-play towards the dwarfs, a bias to measure. The trolls' floor of ~100 simulations came from upstream's rule, under which a search visits every move once before any twice; it no longer applies.) | Beats the previous best setting head-to-head at equal time |
| 3c — postponed 2026-09-29 | **Tree reuse, one tree shared by both sides** — **postponed** (user, 2026-09-29; why, how to build and how to test it in *Changes to the search and trainer*, tree reuse: AlphaZero, KataGo and Leela Chess Zero do not reuse in self-play, because reuse weakens root noise; it would add ~15-30% more visits a search — the cache spares the network calls of re-traversing the previous search's positions, not the simulations; the dwarfs' collapse involves exploration) — only if the reusable share stays large (own-tree reuse ~0. Shared-tree share = the most visited move's share, from the buffer statistics: with upstream's rule the dwarfs' 1-3%, the trolls' 57-77% while they won easily, then 17-20% (run A); **with the new rule ~13-14% for the dwarfs and ~13-17% for the trolls at 100 simulations through step 16 (run C), rising to 18-20% for both by step 23 in its continuation, and for the dwarfs to 21-30% at steps 26-29 (the trolls 18-20%) — roughly 15-30% more simulations per search, against playout caps' 1.37x**; re-read as the policy sharpens, since the watcher records it every step). **Priority** (user asked, 2026-09-28): kept after 3b — on 2026-09-28 it bought ~15-20% more simulations a search (on 2026-09-29 the dwarfs' share reached the ~30% mark below at run C's step 29: one reading so far, to discuss with the user), playout caps 1.37x in KataGo's measurement, and it is the larger change (a tree kept across moves and shared by both sides, root noise re-applied, a larger memory limit than the trainer's 10 MB); 3b's budget semantics (new simulations, or topping up to N visits) are chosen with reuse in mind. **Move it up** if the shared-tree share rises above ~30% for either side in the watcher's statistics, or when 3b is done | Same searches with reuse off; beats 3b's setting head-to-head |
| 4 | Settings: `uct_c`, root noise α (0.03 and 0.3 against 0.1), temperature drop | Each change beats the previous setting head-to-head |
| 5 — built 2026-10-02 | **Convolutional policy head** (user, 2026-09-26: later in the roadmap; started 2026-10-01 at the user's request) — first on the layout-check harness, with an exhaustive test of the action-to-plane map. It is the one planned change that makes our networks unloadable by unmodified OpenSpiel. **Built** behind `--nn_model=resnet_conv_policy` (as AlphaZero's and Leela Chess Zero's heads; 420,988 weights at 64 x 4 against 9,244,626), map and model tested (*Changes to the search and trainer*); **the harness passed** (2026-10-02, `az_head_check`, held-out and training positions, both heads, 32 x 2 and 64 x 4, seeds 1-3): the dwarfs' policy mass on hurls 98.8-99.4% on held-out positions against the linear head's 75-78% (which reproduces the layout control), no overfitting (the linear head: 94-96% on its training positions), 3x closer to symmetric; ~15% slower per full batch of 32 or learning step, faster per small batch, and 12% slower over run C'; then **run C'** (`~/thud-runs/stage5_conv.sh`): run C's settings with the new head as the only change, from scratch to step 16, against C as trained at equal steps and against the anchor — its gate (the harness not clearly failing) was met; trained 2026-10-02 06:50-13:38, 12% slower than C, no overfitting, but **its dwarfs much weaker** than C's at equal steps (against the anchor −10.3 at step 16, C +9.5; the trolls equal); **C'16 loses to C16 head-to-head, −8.5** (30 of 40 pairs): stage 5 fails in this run; the cause to analyse first (for the user) | Faster policy learning on the harness, no more overfitting; then no worse in self-play head-to-head |
| 5b — built 2026-10-02 | **Symmetry augmentation** (user, 2026-10-01: right after the head, as its own change) — each sampled position turned or mirrored by a random one of the board's **8** symmetries (not 16: *Changes to the search and trainer*), behind `--symmetry_augmentation`, default off; tested; on the harness with both heads (the 2 x 2, 2026-10-02): **the linear head gains much** (held-out hurl mass 75.0% → 86.8%, its overfitting gone), **the new head little** (98.8% → 99.1%, policy 20% closer to symmetric); no measurable cost. **Self-play: G'16 beats G16 by +10.2** (2026-10-03; its dwarfs better at every step) — **adopted** (the user, 2026-10-03), in every run from now on | Less overfitting or faster learning on the harness; then no worse in self-play head-to-head |
| 5c — decided 2026-10-02, started 2026-10-03 | **Instrumentation** (the user, 2026-10-02): per-side splits of the trainer's losses and value statistics, self-play results as margins and how games ended, search statistics per side (breadth, the prior's entropy, how far the search moves it) — **built and checked 2026-10-03**; and **a ladder of anchors with a rating** for strength (*How we evaluate networks*) — **the rating script built 2026-10-03** (`ladder.py`; anchors A14, E44, D29, C29, C16; promotion rule proposed). The fixed validation set was **shelved** (the user, 2026-10-03): the pilot showed strong searches agreeing on the best move only ~20% of the time, so its policy targets would measure a reference's style; its value-only variant judged of low value | Each statistic checked against an independent computation (done: the mixed numbers, replayed games, `az_buffer_stats`, a hand-built position per ending); the ladder's ratings reproduce the matches we have |
| 6 | Cloud GPU, longer runs — first a weaker, cheaper GPU with many CPU cores, stronger ones as the network grows (the user, 2026-10-03; options in *Cloud GPU options*) | Throughput measured there first, and **every machine-dependent setting reasoned about or benchmarked on each new machine** (buffer, network and batch sizes, threads, cadence; *Cloud GPU options*) |

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
  Every self-play position is archived (~99.4%: the archiver copies a few hundred
  positions late after each third step).
- **Match-aware play** — after a strong single-battle agent (*Deferred decisions*).
- **End-of-battle limits** — re-measure once the engine plays strongly (*Deferred
  decisions*).
- **Gumbel AlphaZero** — only if few-simulation search remains the bottleneck after 3b
  and 3c; a large change.
- **Convolutional policy heads, again** (the user, 2026-10-03) — for both sides, or for
  the trolls only (the hybrid), once our networks are much stronger: run C' compared
  them at step 16 of a 64 x 4 network, where the dwarfs' policy had barely begun to
  learn; with a stronger, larger network, a warm start or a higher learning rate for
  the head, the verdict may differ (*Changes to the search and trainer*, run C').

**What we have tried, and what to keep** (overview, 2026-10-02, at the user's request;
the evidence is in the sections named in the decision log below):

| Feature | What we found | Verdict |
|---|---|---|
| C++ AlphaZero instead of Python | Python's model has a layout bug and is slow | **Keep** C++ |
| Batched inference (32 actors, 2 inference threads) | 2.7-7x faster self-play | **Keep** |
| Our own copy of the AlphaZero code | Proven identical to upstream; needed for every change | **Keep** |
| 100 simulations a move (against 400) | 100 won at equal machine time early in training | **Keep** for now; revisit with playout caps on the GPU |
| Root noise α 0.1 | A compromise between the sides' move counts; never compared | **Keep**; tune on the GPU |
| Untried moves at the siblings' mean − 0.2 (3a) | Stopped run A's collapse; much stronger dwarfs (C16 beat A16 by +20.9) | **Keep** (default) |
| Larger replay buffer, 2x / 4x / 7x | 2x weak evidence; 4x (D) stopped forgetting, beat C29 +7.3; 7x (E) beat D29 +9.7, no forgetting; 7x from step 1 with augmentation (G') loses to C at step 16 but beats C29 by +13.3 and E44 by +8.0 at equal steps — though it stalls the dwarfs' policy | **Keep 7x**; a growing buffer still worth testing for the dwarfs' policy |
| `learner_batches` (fixed training per step) | Lets a larger buffer keep run C's training per step | **Keep** |
| Margins as the value target | The value learns the expected margin, our objective | **Keep** |
| Matches with each network on its own rule, read per side | Showed the dwarfs' drift that mixed results hid | **Keep** |
| Overfitting check in the trainer | Works (control 7.12 against 3.76); none in run C' | **Keep** (a log only) |
| Symmetry augmentation (5b) | Harness: linear head 75% → 87%, overfitting gone; self-play: G'16 beats G16 by +10.2, its dwarfs better at every step | **Keep**, in every run |
| Convolutional policy head (5) | Harness: 99% against 75-78%; self-play: run C' −8.5 against C — its dwarf policy barely learnt, but its trolls are much better (+18.4 against E44's dwarfs) | **Revisit as a troll-only head** (hybrid with the linear head for the dwarfs); **re-evaluate both variants once our networks are much stronger** |
| Playout caps (3b) | Not built; 7x fewer training positions an hour for moderately better targets | **Revisit** on the GPU |
| Tree reuse (3c) | Not built; ~15-30% more simulations a search, weakens root noise | **Revisit** after 3b, or if the reusable share passes 30% |
| Resignation | Fits margins badly | **Drop** |
| A 32 x 2 network | Its convolutions see only 11 x 11 squares | **Drop**; keep 64 x 4, grow when learning flattens |
| `uct_c`, temperature drop | Not tried | **Later**, on the GPU |
| Gumbel AlphaZero | Not tried; a large change | **Later**, only if few-simulation search stays the bottleneck |

**Ideas raised, not yet tried** (2026-10-02, the user): larger filters in the
convolutional head (5x5, 7x7); training the dwarfs more than the trolls (more learning
on dwarf positions); a convolutional head for the trolls only. Discussed in *Changes to
the search and trainer*, run C'.

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
| 2026-09-28 | Playout caps: settings per side in the code, equal to start (`p` 0.25); the dwarfs get more only if the uneven match shows that extra simulations gain them clearly more than the trolls | Claude's recommendation, recorded at the user's request | roadmap, 3b; *Changes to the search and trainer* |
| 2026-09-29 | The 400-simulation match (A final against B final searching at 400) dropped for the time being; the uneven match waits for the user's word | user | *First training runs*; roadmap, 3b |
| 2026-09-29 | Verify the untried-move rule and the dwarf collapse by evaluating every network as trained (Step 1: C on the new rule, A and the anchor on upstream's); Step 2, a larger buffer, only if C's dwarfs decline as trained; the uneven match parked | user | *First training runs* |
| 2026-09-29 | Matches: each network may search with its own rule (`untried_a`/`untried_b`); several matches at once with OMP 1, or 64 threads with batch 32 for a large match (tuning: 1,720-1,923 against ~1,050 simulations/s) | Claude, measured | *First training runs* |
| 2026-09-29 | Tree reuse (3c) postponed, with why, how to build (with Leela Zero's reset and root noise re-applied) and how to test it recorded | user | *Changes to the search and trainer* |
| 2026-09-30 | Step 2: the quick forgetting check first, then option C (a larger buffer with C's cadence and C's training per step) if warranted — warranted by the check; run D branched from C step 15 | user (green light), Claude (judged warranted) | *First training runs* |
| 2026-09-30 | Option C's new setting (`learner_batches`) and the new meaning of the buffer size and reuse to be revisited later | user | *Settings to determine*, 7 |
| 2026-10-02 | Convolutional policy head built as AlphaZero's and Leela Chess Zero's (3x3 convolution, batch norm, ReLU; 3x3 convolution to 120 planes; a fixed map to the 19,800 actions), behind a switch; first on the layout-check harness, with held-out and training positions for overfitting, after the night's runs | user (go-ahead, queue after the runs), Claude (design, from the sources) | *Changes to the search and trainer* |
| 2026-10-02 | Thud's board has 8 symmetries, not a regular octagon's 16 — kept in every change | user asked, checked | *Changes to the search and trainer*, symmetry augmentation |
| 2026-10-02 | Symmetry augmentation as its own change, a switch off by default, compared with and without the new head on the harness | user | *Changes to the search and trainer* |
| 2026-10-02 | Overfitting check in the trainer: before each learning step, the loss on new positions against positions it has trained on, logged | user | *Notes for later*, replay buffer signals |
| 2026-10-02 | A 2,000-simulation reference is stable enough (4,000 gives the same gain from 100 to 400); the recommendation — no run F on this CPU, playout caps on the GPU — is no longer provisional, still for the user to decide | measured | *Changes to the search and trainer*, playout caps |
| 2026-10-02 | The convolutional policy head passes the harness (no overfitting, hurls learnt on held-out positions); run C' goes ahead by its gate | measured | *Changes to the search and trainer* |
| 2026-10-02 | Symmetry augmentation passes the harness: much better for the linear head, slightly for the new one; its self-play test waits for run C' and the user | measured | *Changes to the search and trainer* |
| 2026-10-02 | Run C' (the new head in run C's settings) loses to C at equal steps (−8.5), its dwarfs far weaker: the head is not adopted for now, to be revisited later; analyse the cause before any further run with it | measured, by the roadmap's rule; the user (revisit later) | *Changes to the search and trainer*, run C' |
| 2026-10-02 | Next runs: G — fresh, a 7x memory from step 1, the old (linear) head — and G', the same with symmetry augmentation, compared at equal steps (one change at a time) | user (Claude recommended the control run) | *Changes to the search and trainer*, symmetry augmentation |
| 2026-10-02 | Instrumentation items 1-4 (per-side splits, self-play margins and endings, search statistics, a fixed validation set) built after runs G and G', as stage 5c | user | *Instrumentation* |
| 2026-10-02 | Training the dwarfs more than the trolls: wait for runs G and G' — augmentation may already help the dwarfs | user | *Changes to the search and trainer*, run C' |
| 2026-10-02 | Are the convolutional head's trolls better? C'16 and C16 each against E44 (40 pairs each), queued after runs G and G' | user | *Changes to the search and trainer*, run C' |
| 2026-10-03 | Symmetry augmentation works in self-play (G'16 beats G16 by +10.2): use it in every run from now on | measured; the user (adopted) | *Changes to the search and trainer*, symmetry augmentation |
| 2026-10-03 | A 7x memory from step 1 loses to C's 1x at step 16 (G16 vs C16 −7.9): grow the buffer from a small start (KataGo's formula with c = 65,536 rather than its 250,000) instead of starting large | measured; for the user to confirm | *Changes to the search and trainer*, symmetry augmentation (runs G and G') |
| 2026-10-03 | Continue run G' (augmented, 7x) tonight from step 16: does the long memory pay off later? Compare at equal steps with C29 and D29 (and E44 if it gets that far) | user (proposed), Claude (agreed) | *Changes to the search and trainer*, symmetry augmentation (runs G and G') |
| 2026-10-03 | Stage 5c (instrumentation) before the next new run | user | roadmap, 5c |
| 2026-10-03 | How we evaluate networks: strength by a ladder of fixed anchors with one rating (linear cost), training health by a fixed validation set; the set versioned, with three signals that a version is outdated (no headroom, decoupling from the ratings, the reference beaten at its own budget); its reference chosen by a pilot | user (with Claude's proposal) | *Instrumentation*, *How we evaluate networks* |
| 2026-10-04 | Ladder promotion rule: beat the top anchor in ≥ 40 pairs with the 95% interval of the pair margin above 0 (≈ Leela Zero's 55% of 400 games); G'44 promoted to top anchor | user | *How we evaluate networks* |
| 2026-10-04 | Next change: KataGo's forced playouts and policy target pruning in our copy (behind a switch); first the noise diagnostic on pruned targets (do two noise seeds agree much more?), then a run against G' at equal steps; the hybrid head after it | user (Claude's proposal) | *Changes to the search and trainer*, runs G and G' |
| 2026-10-04 | Run G' (fresh, 7x, augmentation) is the strongest network: beats C29, D29, E44 at equal steps and A14 (+3.2); top of the ladder (+6.9 against E44) | measured | *Changes to the search and trainer*, runs G and G' |
| 2026-10-03 | No fixed validation set (the pilot: strong searches agree on the best move only ~20% of the time; a value-only set of low value): the ladder for strength, the trainer's statistics for training health, as Leela Zero and Leela Chess Zero do | user | *How we evaluate networks* |
| 2026-10-03 | Re-evaluate convolutional policy heads (for both sides, or the trolls only) once our networks are much stronger | user | *Later, when a trigger fires* |
| 2026-10-03 | Cloud GPU: start on a weaker, cheaper one with many CPU cores, move to stronger ones later; on every machine switch, reason about or benchmark the machine-dependent settings first | user | roadmap, 6; *Cloud GPU options* |
| 2026-10-03 | The convolutional head's trolls are much better (against E44's dwarfs +9.8 against C16's −8.6), its dwarfs much worse: a hybrid head (convolutional for the trolls, linear for the dwarfs) is a strong candidate | measured; for the user to decide | *Changes to the search and trainer*, run C' |
| 2026-09-29 | Step back: on this CPU only "does it work" questions, tuning on the GPU (3c and 4 after the switch) | Claude's proposal, **open** | status block |
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
  conclude anything). Since 2026-09-28 `sims_a` and `sims_b` give each network its own
  number of simulations (default `sims`; for the uneven match, stage 3b): with the
  defaults, and with both set to 100, it plays exactly the earlier binary's games (C
  step 16 against A step 14, 2 pairs), and a 400/100 match mirrors exactly when the
  networks swap, its games differing from the 100/100 ones. That which network gets
  which budget is right is read from the code; the uneven match, where the 400 side
  should win, shows it again. Since 2026-09-29 `untried_a` and `untried_b` give each
  network its own rule for untried moves (default `untried`), so that each can search
  with the rule it was trained with, and `progress=SECONDS` reports the simulations per
  second to stderr. Controls: with the defaults, with and without the reports, it plays
  exactly the earlier binary's games; C step 16 on the new rule against A step 14 on
  upstream's mirrors exactly when the networks swap, its games differing from the
  both-upstream ones; the summary records both rules. **Throughput tuning**
  (2026-09-29, `~/thud-runs/tune_az_match.sh`, 6 minutes per setting on Step 1's long
  games, other programs using ~1% of the machine; games are identical whatever the
  threads, batches and OMP settings — checked on the same pairs with 16 threads and OMP 4
  and with 2 threads and OMP 1): one process with 16 threads, batch 16, OMP 4 — the
  setting of every match so far — 1,315 / 1,367 simulations/s at the start / end; 32
  threads 1,619; **64 threads, batch 32, OMP 4: 1,923** (+43%, and still rising with
  threads); the same with 2 inference threads and OMP 2 1,816; 2 processes of 32 threads,
  OMP 2, 1,694; 3 processes of 20 threads 1,521 with OMP 2, **1,720 with OMP 1**. Whole
  matches averaged only ~1,050: threads idle at the end, when the last long pairs run
  alone. So: a match of 64+ pairs on its own with 64 threads, batch 32, OMP 4; smaller
  matches up to 3 at once with OMP 1 and threads = pairs, the next starting as soon as
  one finishes. Since 2026-09-29 (user: measure what the evaluation cache saves)
  `cache=ENTRIES` sets each evaluator's cache (0: none — our copy of the evaluator now
  builds no cache at size 0, where upstream's `LRUCache` keeps at least 4 entries) and
  `share=1` gives a network playing itself one evaluator for both sides, as in
  self-play; the progress reports and the summary count the requests for values and for
  move probabilities and the share the cache answered (our copy's `VPNetEvaluator`
  counts both kinds; the trainer logs them after every learning step). Controls: with the
  defaults it plays exactly the earlier games; with the cache off, the same games and no
  hits; with a shared evaluator, the same games as with two. First counts, C step 29
  against itself with the new rule, one pair: value requests answered by the cache 38.8%
  with two evaluators, **69.4% with one shared** — each side's search re-evaluates the
  positions the other side's just evaluated — and move-probability requests 100% either
  way (the second request of a position the search expands). C step 16 against A step
  14, both on upstream's rule: value requests 0.1%, move-probability requests 100%, but
  only 5,903 of them against 58,904 values: under that rule the searches spread and
  rarely expand a node. **What the cache is worth** (2026-09-29 23:29-23:59,
  `~/thud-runs/tune_cache.sh`: C step 29 against itself with the new rule, 64 threads,
  batch 32, OMP 4, 6 minutes per setting, alone): with one shared evaluator as in
  self-play and today's cache **3,275-3,538 simulations/s, without a cache 1,301 — the
  cache makes the search 2.6-2.7x faster**; two separate evaluators 2,105. Per 100
  simulations ~100 value requests (40-45% answered by the cache: positions the preceding
  searches of both sides evaluated) and ~48 move-probability requests (all answered).
  A 4x cache (1,048,576 entries) seemed to give +42% (4,783 simulations/s), but that was
  the test's own artifact: a network against itself replays each pair's first battle in
  the second, and from ~4 minutes on, when the first short battles ended, the larger
  cache still held them (its value hit rate jumped from ~45% to 69-81%, today's stayed
  at ~40%). Before any replay (90-240 s) the 4x cache ran 3,438 against 3,275-3,538: no
  gain, as expected. (Engine identity: `identity_check.cc` still passes with the
  counters, 12 of 12.) **In real self-play the value hits are fewer**: run D's trainer
  logs, steps 16-25 (2026-09-30), 13-22% of ~2.2-2.9 million value requests per step
  hit the cache (move-probability requests 100%) — root noise at every root, the sampled
  opening moves and 32 concurrent games make consecutive searches overlap less than in
  the noise-free match. These repeats cost no network call but still a simulation each;
  tree reuse would keep their statistics instead (*Changes to the search and trainer*,
  tree reuse, corrected accordingly).
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
  400-simulation match was planned, then dropped for the time being — user, 2026-09-29:
  3b's playout caps revisit the budget); run B learned only 4 times against 16; 6 hours
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
  100 pairs (results below); replicates of runs A and C (6 hours each) would settle the
  causal question, and are skipped unless it matters.

  **Run C continued to step 29** (2026-09-28 12:27-18:28 and 2026-09-29 08:12-10:08, 12
  hours of machine time in all, 660,943 positions; resuming works in our copy): the
  dwarfs' searches narrowed further (a median 11 moves visited, the most visited 0.30 at
  step 29), and the dwarfs did worse in self-play (mean return target −0.13 at step 25,
  −0.40 at step 29). Matches, 100 simulations, upstream's rule in the searches:

  | Match | Pairs | Margin per pair (95% interval) | Pairs won / drawn / lost | C step 29 as dwarfs / as trolls |
  |---|---|---|---|---|
  | C step 29 vs A step 14 | 100 | **−11.5** (−13.1 to −9.9) | 3 / 2 / 95 | −20.0 / +8.5 (step 16: −2.4 / +1.1) |
  | C step 29 vs C step 16 | 40 | +1.1 (−1.9 to +4.0) | 20 / 0 / 20 | +5.5 / −4.5 |
  | C step 29 vs the anchor | 20 | +21.2 (+16.6 to +25.7) | 19 / 1 / 0 | −4.5 / +25.7 (step 16: +21.6 / +24.8) |

  Step 29 is even with step 16 head-to-head and plays the trolls as well or better, but
  its dwarf play fell far against the networks trained with upstream's rule — against A
  step 14's trolls −20.0 (step 16: −2.4), against the untrained anchor's −4.5 (+21.6; the
  games lasted a median 395 moves, step 16's 131) — while against C step 16's trolls its
  dwarfs won by 5.5: not transitive. Two readings (settled below — as trained, a drift; the
  quick check: forgetting in the dwarfs' move probabilities): (a) **forgetting** — its
  dwarfs play well against the trolls they train against and lost what beat other trolls
  (the buffer holds ~3 learning steps; if so, the buffer's "newer network losing"
  trigger — pronounced here, not yet shown persistent); (b) **the evaluation** —
  upstream's rule spreads a side's search when the network judges that side losing.
  **Checked 2026-09-29** (`untried_move_check`, 3 self-play games per network with
  upstream's rule, 100 simulations; `~/thud-runs/stage1_matches/untried_check_C_step*.txt`):
  under upstream's rule the dwarfs' searches spread for **both** — step 16 a median 99
  moves visited (15 dwarf positions; its games were short), step 29 76 (87) — against 15
  and 9 under the new rule; the trolls' 11 for both under upstream's rule (12 and 18 under
  the new one). So step 29 does not lose because
  its searches spread where step 16's did not. But in these matches neither plays the
  dwarfs as trained: with nearly every move visited once, the search picks, among the
  ~100 moves the policy ranks highest, the one whose single evaluation is best
  (`CompareFinal`: visits, then total value) — a test of the value head one move ahead,
  not of the policy with its own search. The drop may lie there (forgetting in the value
  head, or values that no longer separate dwarf moves). The decisive check is the match
  against A step 14 with the new rule in both searches (backlog, user 2026-09-29);
  fairer still, each network searching with its own training rule (`az_match` has one
  `untried` for both). The user's
  hypothesis (2026-09-29) that dwarf play is harder to learn — the trolls start in lines
  of three around the Thudstone and can shove at once, the dwarfs must first line up
  towards a troll; they have 4-9x the moves; plain MCTS dwarfs need deep searches
  (Phase 5) — fits the trolls' faster start and the self-play results, but not a
  decline against fixed opponents.

  **The plan** (user, 2026-09-29: verify the rule and solve the dwarf collapse). Run A's
  collapse is certain — A was evaluated as trained. Run C's decline was seen only in
  matches where its dwarfs did not play as trained. **Step 1 — every network as
  trained** (`~/thud-runs/step1_as_trained.sh`, ~3.5 hours): C's networks search with
  the new rule, A's and the anchor with upstream's (`az_match`'s `untried_a`/`untried_b`),
  the decisive matches first — C29 against C16 (40 pairs), C16 and C29 against the
  anchor (20 each) — then C8, C12, C20, C24 against the anchor and C29 against A14 (60
  pairs). If C's dwarfs hold up, the rule prevented the collapse for about twice as many
  positions as run A lasted (~640,000 against ~330,000): 3a confirmed, and the per-side
  curve against the anchor becomes a health check at the end of every run. **Step 2,
  only if they decline as trained:** the replay buffer as the second cause — our buffer
  holds 65,536 positions, ~200-300 games, where AlphaGo Zero trained on its most recent
  500,000 games; run A2 (2x) had not collapsed where A did (weak); C29's results do not
  line up, a typical sign of a network specialising against its current opponent.
  First, optionally, ~10 minutes: does C29 predict older archived positions worse than
  C16 (the signature of forgetting)? Then a branch from C's last healthy checkpoint with
  a 4x buffer, trained to C29's number of positions and evaluated as in Step 1. Our copy
  must first learn to resume with a different buffer size (upstream's `LoadBuffer`
  refuses one), and the buffer size also sets how often the network learns (every
  `replay_buffer_size / replay_buffer_reuse` new positions) and how much work each
  learning step does (`replay_buffer.Size() / train_batch_size` batches,
  `alpha_zero.cc:425`), so that ratio is chosen deliberately. **Chosen: option C**
  (2026-09-30, user: the quick check first, then option C if warranted): our copy trains
  a fixed 64 batches per learning step (C's amount) drawn from the larger buffer, with
  C's cadence (a step every 21,845 new positions) — so only the memory changes, from ~3
  learning steps of games to ~12, and each position is still trained on ~3 times. (Option
  A, upstream with `replay_buffer_reuse` 3: updates 4x rarer; option B, reuse 12: 4x the
  training work and each position 12 times.) Its new setting, and the new meaning of the
  old ones, are to be revisited later (*Settings to determine*, 7). If the buffer does not fix
  it: the value targets (the dwarfs' are nearly all losses) or the asymmetry hypothesis.

  **Step 1's results** (2026-09-29 19:21-23:25, `step1_as_trained.sh`, each network as
  trained; 95% intervals):

  | Match | Pairs | Per pair | Won / drawn / lost | C as dwarfs / as trolls |
  |---|---|---|---|---|
  | C step 8 vs the anchor | 20 | +19.9 (+14.1 to +25.6) | 20 / 0 / 0 | +4.0 / +15.8 |
  | C step 12 vs the anchor | 20 | +30.9 (+26.9 to +35.0) | 20 / 0 / 0 | +2.4 / +28.6 |
  | C step 16 vs the anchor | 20 | +36.2 (+31.8 to +40.7) | 20 / 0 / 0 | **+9.5** / +26.8 |
  | C step 20 vs the anchor | 20 | +29.9 (+24.7 to +35.0) | 20 / 0 / 0 | +3.2 / +26.6 |
  | C step 24 vs the anchor | 20 | +29.3 (+26.0 to +32.6) | 20 / 0 / 0 | +0.9 / +28.4 |
  | C step 29 vs the anchor | 20 | +25.9 (+22.6 to +29.1) | 20 / 0 / 0 | **−0.6** / +26.4 |
  | C step 29 vs C step 16 | 40 | **+4.1** (+0.9 to +7.3) | 26 / 2 / 12 | +2.4 / +1.7 |
  | C step 29 vs A step 14 | 60 | **−8.5** (−10.9 to −6.2) | 9 / 1 / 50 | −16.8 / +8.3 |

  - **No collapse like run A's**: C's dwarfs never fall below −0.6 against the anchor
    (A step 16's: −24.9).
  - **But C drifted from step 16 on**: it beats step 16 head-to-head on both sides, yet
    its dwarfs lost 10.1 points against the anchor (95% −15.3 to −4.8, p = 0.001; the
    trolls −0.4, p = 0.80), and it loses clearly to A step 14, which step 16 had been
    even with (−1.2 with upstream's rule in both searches, +3.9 with the new one). Better
    against itself, worse against others: the signature of specialising on its own play
    (forgetting). By the rule above, **Step 2 is indicated** (the user decides).
  - The old evaluation exaggerated the decline (with upstream's rule C's dwarfs fell 26
    points against the anchor, as trained 10).
  - **The dwarfs' move probabilities look like their weak point**: C step 16's dwarfs beat
    the anchor by +9.5 as trained but by +21.6 when upstream's rule spread their search
    over ~99 moves and picked by one evaluation each — the value head judges dwarf moves
    better than the policy's focused search does. That speaks for playout caps (3b) and
    the convolutional policy head (5), which both aim at policy learning.
  - Stage 3a in the narrow sense holds: the rule prevents run A's collapse. Training with
    it has not produced a network better than A's best.

  **The quick forgetting check** (2026-09-30, `thud/experiments/az_forgetting.cc`, 3.5
  minutes; `~/thud-runs/stage1_matches/forgetting_2026-09-30.jsonl`): the same 4,096
  positions from each of C's archives (steps N-2..N for N = 3, 6, ..., 30), and for C's
  networks 0, 8, 12, 16, 20, 24, 29, per side, the value error against the game's
  margin, the move-probability loss against the visit counts (the trainer's losses)
  and how often the network's likeliest move is the most visited. Compare within an
  archive: across archives the targets sharpen, so the losses fall anyway. Control: in
  each archive the network trained most recently on it fits it best (C8 on 9, C12 on 12,
  C16 on 15, C20 on 21, C24 on 24, C29 on 30), and the untrained network worst. Values:
  no forgetting — C29 even predicts the old outcomes slightly better than C16-C24. **The
  dwarfs' move probabilities: strong drift, the trolls' hardly** — on archives 18 and 21
  C29's dwarf loss is 6.57 and 5.63 where C16-C24 had 4.46-5.03, on the oldest (3) 8.03,
  worse than the untrained network's near-uniform 5.44, while on the newest (30) it is
  the best of all (2.51); for the trolls C29 is only 0.1-0.2 worse than earlier networks
  on old archives. Most of it between steps 24 and 29 (archive 18: 5.03 → 6.57). C29's
  dwarf policy has narrowed onto a repertoire for its own current games, giving
  near-zero probability to moves earlier searches found worth visiting — consistent with
  its dwarfs losing to unfamiliar trolls in Step 1. Caveat: old targets come from weaker
  searches, so some disagreement is improvement; dwarf-only, this large and this abrupt,
  it reads as forgetting. **Step 2 started** (Claude, on the user's green light).

  **Step 2: run D** (2026-09-30, `~/thud-runs/step2_buffer4x.sh`, `~/thud-runs/
  stage2_buffer4x/`): run C branched at **step 15** — not 16, because C's archives 6, 9,
  12 and 15 hold exactly the positions of steps 4-15, what a 4x buffer holds at step 15 —
  with option C: `replay_buffer_size` 262,144, `replay_buffer_reuse` 12 (a learning step
  every 21,845 new positions, as in C) and `learner_batches` 64 (as in C; new in our
  copy's trainer, 0 = upstream's one pass over the buffer), to `max_steps` 29 (C's number
  of positions, ~636,000). Its buffer starts full: `thud/experiments/az_merge_buffers.cc`
  wrote the four archives, each in the order its positions were added, into one file of
  262,144 that upstream's `LoadBuffer` accepts, and read it back — 262,144 positions, all
  distinct, exactly the archived ones. (The archives miss ~0.6% of positions: the
  archiver copies a few hundred positions after each third step; e.g. 378 between
  archives 6 and 9.) Then, each network as trained: D29 against C29 (40 pairs), the
  anchor (20), A14 (60), C16 (40), D20 and D24 against the anchor (20 each), and the
  forgetting check of D's networks on C's archives. If D's dwarfs keep their repertoire
  and D beats C29 without losing to A14, forgetting was the cause.

  **Run D's results** (2026-09-30: steps 16-29 from 02:57, frozen 07:40-10:48 by a
  battery freeze, stopped at 10:54 with the Claude Code session that had started it,
  resumed detached at 14:43, step 29 at ~16:40; matches until 19:41, each network as
  trained):

  | Match | Pairs | Per pair (95%) | Won / drawn / lost | D as dwarfs / as trolls |
  |---|---|---|---|---|
  | D29 vs C29 | 40 | **+7.3** (+4.1 to +10.5) | 30 / 3 / 7 | +0.8 / +6.5 |
  | D29 vs C16 | 40 | **+8.4** (+5.1 to +11.8) | 32 / 1 / 7 | +7.5 / +0.9 |
  | D29 vs A14 | 60 | −5.4 (−8.1 to −2.7) | 20 / 4 / 36 | −12.4 / +7.0 |
  | D20 vs the anchor | 20 | +37.5 (+33.0 to +42.0) | 20 / 0 / 0 | +11.7 / +25.8 |
  | D24 vs the anchor | 20 | +33.9 (+29.0 to +38.7) | 20 / 0 / 0 | +8.1 / +25.8 |
  | D29 vs the anchor | 20 | +28.1 (+23.9 to +32.3) | 20 / 0 / 0 | +3.2 / +24.9 |

  - **The longer memory stops the dwarfs' policy from forgetting** (the forgetting
    check of D16-D29 on C's archives, same positions): D29's dwarf move-probability loss
    on archives 3, 18 and 21 is 5.74, 4.89 and 4.48, C16's level (5.73, 4.90, 4.58), where
    C29 had 8.03, 6.57 and 5.63; D's dwarf policy is also less peaked (its likeliest move
    is the most visited in 65% of archive 30's dwarf positions, C29's 88%). The trolls'
    and the values' numbers are much as in C.
  - **D is the strongest network of the C/D family**: D29 beats C29 by +7.3 and C16 by
    +8.4 (C29 beat C16 by +4.1); its dwarfs score +7.5 against C16's trolls.
  - **Against fixed outside opponents D is better than C but still declines**: against
    A14 −5.4 (C29: −8.5); its dwarfs against the anchor fall from +11.7 at step 20 to +3.2
    at step 29 (C: +3.2, +0.9, −0.6 at steps 20, 24, 29; D29 minus C29 +3.8, p = 0.07);
    the trolls stay at +25-26 in both runs.
  - Reading: forgetting was a real cause — the longer memory removes the policy's
    narrowing and makes a stronger network — but not the only one: D's dwarfs slip
    against the anchor without their policy narrowing (perhaps a cautious style learned
    against strong trolls, which exploits random ones less and need not cost strength),
    and the gap to A14 remains. By the rule above: D's dwarfs keep their repertoire and D
    beats C29, but D still loses to A14 — forgetting explains part, not all.
  - Still open (Step 1): C's dwarfs played better with a broad search picking by one
    evaluation per move than with their policy's focused one — their move probabilities
    remain the weak point, which playout caps (3b) and the convolutional policy head (5)
    address.

  **Run E** (2026-10-01, `~/thud-runs/stage3b_E.sh`, `~/thud-runs/stage3b_E_7x/`): run D
  continued from step 29 with a **7x memory** — 458,752 positions, `replay_buffer_reuse`
  21 (C's and D's cadence), `learner_batches` 64 — pre-filled with the newest positions of
  D's lineage (C's steps 7-15 and D's 16-29; `az_merge_buffers` now skips duplicates: D's
  archive 27 and final buffer overlap by 218,061), no playout caps, 6 hours of machine
  time from 13:17. It is stage 3b's control arm: run F starts from the same state
  (`replay_buffer_start.data`, D's checkpoint 29) with playout caps for the same machine
  time; 7x is common to both (the user, 2026-10-01: adopt it as the baseline rather than
  test it against 4x). The trainer runs at normal priority, other work at nice 19, and
  a load monitor (`load.log`) records the CPU the rest of the machine takes. The laptop
  slept on battery at ~15:12 (Windows logged nothing until "Wake from sleep detected" at
  16:43) and WSL restarted on waking, ending E after step 32 (checkpoint 32; the buffer
  after step 33's games was saved complete at 15:09) with 6,919 s of its 21,600 s used;
  resumed at 17:54 for the remaining 14,681 s (`stage3b_E_resume.sh`), its step 33
  learning from slightly more new games than usual. Without the buffer watcher from then
  on: it loaded a second 4.4 GB copy after every step, ~12 GB of WSL's 15 with the
  trainer's ~7 — not the cause (a memory kill ends single processes, not WSL), but too
  close; the statistics can be computed afterwards from the saved buffers. **Finished**
  at 21:59 at step 44 (its time budget, no pause since the resume; ~11 learning steps in
  6 hours of machine time, as expected).

  **Run E's results** (2026-10-02 00:45-03:27, `~/thud-runs/stage3b_E_eval.sh`, each
  network as trained; per-side intervals 95%, side differences by Welch's t-test):

  | Match | Pairs | Per pair (95%) | Won / drawn / lost | E as dwarfs / as trolls |
  |---|---|---|---|---|
  | E44 vs D29 | 40 | **+9.7** (+6.6 to +12.9) | 33 / 2 / 5 | +8.1 / +1.7 |
  | E44 vs A14 | 60 | −3.5 (−5.7 to −1.3) | 22 / 6 / 32 | −13.7 / +10.1 |
  | E36 vs the anchor | 20 | +30.6 (+26.6 to +34.5) | 20 / 0 / 0 | +4.6 / +26.0 |
  | E40 vs the anchor | 20 | +31.1 (+27.6 to +34.5) | 20 / 0 / 0 | +3.5 / +27.6 |
  | E44 vs the anchor | 20 | **+37.7** (+34.0 to +41.3) | 20 / 0 / 0 | +8.7 / +29.0 |

  - **E keeps improving**: E44 beats D29 by +9.7 (D29 had beaten C29 by +7.3), the gain
    mostly on the dwarf side (E's dwarfs +8.1 against D's trolls, D's −1.7 against E's).
  - **The dwarfs' slip against the anchor reversed, late**: +8.7 at step 44 against
    D29's +3.2 (p = 0.03), back at D20's +11.7 and C16's +9.5 (neither difference
    significant); E36 and E40 were still at D29's level (+4.6, +3.5), so the recovery
    came in the last steps — with 20 pairs, one reading. The trolls +29.0, the best yet
    (D29 +24.9, p = 0.005).
  - **A14 still wins, by less**: −3.5 (D29 −5.4, C29 −8.5); E44 minus D29 +1.9 (p =
    0.29, not significant) — E's trolls better (+10.1 against D29's +7.0, p = 0.04), its
    dwarfs not (−13.7 against −12.4).
  - **No forgetting** (`az_forgetting`, E33-E44 on C's archives, the same positions as
    before; `forgetting_E_2026-10-02.jsonl`): E44's dwarf move-probability loss on
    archives 18 and 21 is 4.96 and 4.48 (D29 4.89, 4.48; C29 6.57, 5.63); on the oldest
    archives (3-12, positions E's buffer no longer holds) a slow rise of ~0.2 over 11
    steps (archive 3: 5.74 → 5.97; C29 had 8.03); its likeliest dwarf move is the most
    visited in 64% of archive 30's positions (D29 65%, C29 88%). Trolls and values much
    as D29.
  - Reading: the 7x memory continues D's trend without its slip — the strongest network
    so far, still behind A14. What is left of A14's edge is on the dwarf side.

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
  download is the CUDA build (`global_variables.sh:89`). Or ARM + NVIDIA (GH200, below):
  our aarch64 setup unchanged, with PyTorch's CUDA wheel for aarch64 instead of the CPU
  one — `torch-2.10.0+cu128-cp314-cp314-manylinux_2_28_aarch64.whl` exists on
  download.pytorch.org/whl/cu128 (checked 2026-10-03), our exact version and Python.

  **Cloud GPU options** (researched 2026-10-03, the user asked; prices change — recheck
  before renting). What decides it for us: the network is small (64 x 4, ~0.15 GFLOP an
  evaluation), so any current GPU has far more compute than our search can feed; the
  limit will be the **CPU cores** running the tree searches — a GPU wants batches of
  hundreds, i.e. hundreds of games searched in parallel. So cores per GPU matter more
  than the GPU model, until the network grows. Single-GPU instances, on-demand:

  | Provider, instance | GPU | vCPUs | RAM | $/hour | Note |
  |---|---|---|---|---|---|
  | Lambda GH200 | H100-class, 96 GB | 64 (ARM Neoverse V2) | 432 GiB | 2.29 | Same architecture as this laptop |
  | Lambda A10 | A10, 24 GB | 30 | 226 GiB | 1.29 | x86 |
  | Lambda A100 (SXM or PCIe) | A100, 80 GB | 30 | ~220 GiB | 1.99 | x86 |
  | Lambda H100 PCIe | H100, 80 GB | 26 | 225 GiB | 3.29 | x86 |
  | RunPod L40S | L40S, 48 GB | 16 | 94 GB | 0.79 community / 1.09 secure | x86 |
  | RunPod L4 | L4, 24 GB | 12 | 50 GB | 0.44 / 0.49 | x86 |
  | RunPod RTX 4090 | 4090, 24 GB | 6 | 41 GB | 0.34 / 0.74 | Too few cores for us |
  | RunPod RTX 5090 | 5090, 32 GB | 9 | 35 GB | 0.69 / 0.99 | Few cores |
  | Vast.ai (marketplace) RTX 4090 | 4090, 24 GB | varies, filterable | varies | ~0.3-0.6 | Hosts vary in reliability |
  | Hetzner GEX45 (dedicated, monthly) | RTX PRO 4000 Blackwell, 24 GB | i5-13500, 14 cores / 20 threads | 64 GB | €214 a month + €209 setup (~€0.29/h at 24/7) | Germany/Finland |
  | Hetzner GEX131 (dedicated) | RTX PRO 6000 Blackwell, 96 GB | Xeon Gold 5412U, 24 cores | 256 GB | €889-1,197 a month | |

  Sources: lambda.ai/pricing and runpod.io/pricing (read 2026-10-03), Hetzner's GEX45
  and GEX131 announcements and reviews, getdeploying.com and Vast.ai's listings.
  **Recommendation** (Claude, for the user to decide): first measure here, free, how
  much CPU our search costs per simulation without the network (a trivial evaluator):
  that says how many cores a GPU needs. Then a 2-3 hour throughput test (~$10-20) on
  two contrasting machines — **Lambda GH200** (64 ARM cores, our setup unchanged) and a
  cheap x86 machine with many cores (Lambda A10, 30 vCPUs, or a Vast.ai 4090 host with
  ≥ 24 cores) — simulations a second at 64 x 4 and 128 x 6 with 64, 128 and 256 games
  in parallel; then rent by measured throughput per dollar. For months of continuous
  runs a dedicated monthly server (Hetzner) becomes cheaper per hour, if its cores
  suffice.
  **Budget** (the user, 2026-10-03): a first phase of tens of dollars — tune the
  machine-dependent settings and get a feel for how well training works on a GPU —
  then decide how much to spend on pushing further.
  **Start on a weaker, cheaper GPU, move to stronger ones later** (the user,
  2026-10-03): at 64 x 4 even a mid-range GPU outruns our search, so the first machine
  should be cheap with many cores — e.g. Lambda A10 (30 vCPUs, $1.29/h) or a Vast.ai
  4090 host with ≥ 24 cores (~$0.3-0.6/h), both x86 — and the GH200 (or bigger) once
  the network grows. A cheap ARM option exists but does not fit: AWS g5g (Graviton2 +
  NVIDIA T4G; g5g.8xlarge 32 vCPUs, 64 GiB, $1.37/h) needs CUDA for the T4G's compute
  capability 7.5, which PyTorch's official aarch64 CUDA wheels leave out; only AWS's own
  builds (2.7, 2.9.1) have it.

  **On every machine switch, first reason about or benchmark the settings that depend
  on the machine** (the user, 2026-10-03) — this CPU's values (`CLAUDE.md`,
  *Environment*) do not carry over: the replay buffer's size (memory: ~9,700 bytes a
  position, plus the copy written each step, and how long that write takes); the
  network's size (what the machine runs fast enough); the inference batch size and the
  training batch size; the threads — actors (games searched in parallel), inference
  threads, `OMP_NUM_THREADS`, and for matches threads and batch; the learning cadence
  and the batches per step (`replay_buffer_reuse`, `learner_batches`), which set how
  often each position is trained on; simulations per move; the inference cache's
  size; and the time budget of runs. As on this CPU in Phase 6 (the throughput report),
  measured, not assumed — and the stage-1-style comparisons (100 against 400
  simulations, network sizes) may come out differently there.

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
   training to self-play. **Option C of Step 2** (2026-09-30) separates what upstream
   ties together — a learning step every `replay_buffer_size / replay_buffer_reuse` new
   positions, each training `replay_buffer_size / train_batch_size` batches — by fixing
   the batches per learning step, so it adds a setting and changes what the others mean:
   the buffer size becomes only the memory (how many past positions the network keeps
   learning from), the batches per step the amount of training, and together with the
   cadence they set how often each position is trained on. Step 2 picks values only to
   test forgetting (a 4x memory, C's cadence and C's 64 batches); **the values themselves
   are to be revisited later** (user, 2026-09-30) — the memory, the training per step
   and the times each position is trained on — once learning is stable, and again before
   growing the network or moving to the GPU. **The reference for the memory** (user asked
   about an even bigger buffer, 2026-09-30): KataGo samples "uniformly from a growing
   moving window of the most recent data, with window size beginning at 250,000 samples
   and increasing to about 22 million by the end of the main run" — N_window = c(1 +
   β((N_total/c)^α − 1)/α), c = 250,000, α = 0.75, β = 0.4, sublinear in the positions
   generated so far (arXiv 1902.10565, section 3 and appendix C); an ablation "showed
   major overfitting due to lack of data" until its buffer (its "window") was doubled. At our scale it
   gives ~280,000 positions at C's step 15 and ~390,000 at step 29: run D's 262,144 (4x)
   is in line, our former 65,536 a quarter of KataGo's smallest buffer. A larger fixed
   buffer costs no training time with `learner_batches` (64 batches whatever the size;
   saving it each step took ~16 s at 4x), but memory: the trainer used 5.3 GB at 4x, ~8 GB
   at 8x, plus the buffer watcher's copy — WSL's 15 GB allow ~8x at most, 16x needs more of
   the host's 31.6 GB. And a longer memory learns from older, weaker games, which is why
   KataGo grows it sublinearly. So, for long runs (the GPU): a growing buffer in our copy
   — sample only from the most recent N_buffer positions — rather than a bigger fixed
   buffer. KataGo's 22 million came after ~225 million positions (the buffer then 10% of
   all data); by its rule our buffer should be ~4.3x the former 65,536 after 330,000
   positions (85% of all data), ~5.9x after 640,000, ~7.5x after 1 million, ~21x after 5
   million. Sizes need not be powers of two (65,536 is only upstream's default; the
   buffer indexes with `% max_size`): 6x = 393,216 with `replay_buffer_reuse` 18, or 7x =
   458,752 with 21, keeps C's cadence; at ~9,700 bytes a position the trainer needs ~6.5
   GB at 6x and ~7.1 GB at 7x, the buffer watcher's copy ~3.8 or ~4.4 GB more at its peak
   (2026-09-30).

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

**Instrumentation** (the user asked, 2026-10-02: what can we see in our logs, would we
see overfitting, what should we add to judge runs faster?).

*What we have.* The trainer's `learner.jsonl`, per step: throughput (positions and
games a second, inference batch sizes, cache hits); game length (mean, histogram);
wins per side (nearly useless while the trolls win almost every game); the training
losses (policy, value, L2), averaged over both sides; since 2026-10-02 the losses on
new against already trained positions before each step (both sides mixed);
`value_accuracy` (whether the search's value had the winner's sign) and
`value_prediction` (its size, i.e. confidence) at 7 points of each game; the built-in
evaluator against MCTS with random rollouts (a moving average per level, both sides
mixed, a weak opponent). The text logs hold more: each actor logs **every self-play game
with its returns and full move list**, the evaluator every game against MCTS (its side
recoverable from the returns line). Saved on disk: a checkpoint per step and the final
replay buffer — for runs with a 7x buffer through step 16 that is every position of
the run, otherwise only the last ~3 steps (runs C and E had an archive besides). Our
tools work from these after the fact: `az_buffer_stats` (search breadth per side),
`az_forgetting` (a network's fit to older positions), `az_target_quality` (targets
against deeper searches), `az_match` (strength as trained, per side), `az_head_check`.

*Can we see overfitting?* Partly. The new-against-trained check shows a generalisation
gap, but (1) it mixes the sides, whose policy losses differ (~5 against ~3 nats), so
a dwarf-only problem can hide; (2) new positions come from the current network's own
play, so they flatter it (in run C' new positions often had the lower loss); (3) it
compares against self-play targets that change as the network does, so losses are not
comparable across steps or runs. What would show it cleanly is a **fixed validation
set**: a few thousand positions never trained on, from varied games, each with a deep
search's visit counts as the policy target and its game's final margin, evaluated by
every checkpoint, per side. Its losses are comparable across steps and across runs
(C, C', G, G' on the same positions), and they measure what we want — closeness to a
deep search — rather than fit to the run's own targets. It would have shown C''s
dwarf weakness during the run.

*What we wanted in the past, and the signal that would have shown it early:* run A's
dwarf collapse — the dwarfs' search breadth per step (99 moves visited); C29's
narrowing — the dwarf policy's entropy falling and its top move matching the search
more often (88%); C''s weak dwarfs — the dwarfs' mean final margin in self-play per
step, a validation loss per side, an early match against the anchor; throughput and
pauses — already logged (the scripts' clock logs).

*Proposed, in order of value for effort* — **items 1-4 decided** (the user, 2026-10-02:
"I think this will benefit us in the future"), **to build after runs G and G'**, as
roadmap stage 5c; items 1-3 built 2026-10-03, item 4 shelved the same day (the pilot);
items 5-7 not decided:

1. **Per-side splits** of everything now mixed: the training losses, the
   new-against-trained check, `value_accuracy` and `value_prediction`. The learner has
   each position's side to move. Small change in our copy.
2. **Self-play results as margins, per step**: the dwarfs' mean final margin and its
   spread, and how games ended (all of one side captured, no legal move, the 200- or
   800-turn limit). Wins alone say nothing while one side wins everything.
3. **Search statistics per step and side**, from the new positions: moves visited,
   the most visited move's share, effective moves (what `az_buffer_stats` computes
   after the fact); plus, with the root's prior kept in the trajectory, the prior's
   entropy, how far the search moved the policy (KL divergence of the visit counts
   from the prior) and how often the prior's top move is the search's. A search that
   no longer changes the prior teaches the network nothing; one that overrides it
   every time shows a weak policy.
4. **The fixed validation set** above (built once with `az_target_quality`'s
   references; evaluated each step, or afterwards from the checkpoints).
5. **An evaluation watcher** beside the trainer: every few checkpoints a short match
   as trained against the anchor and a strong fixed opponent (E44), per side, at nice
   19 — a strength curve during the run (C' would have shown its dwarfs at −17 at step
   8, 3.5 hours before its training ended). Harmless for equal-steps comparisons.
6. **After-the-fact tools**: a game analyser over the actors' logs (margins, captures,
   hurls and shoves per side, how games ended, how varied the openings are), and a
   per-step slicing of a 7x run's buffer for the statistics above.
7. Lower: gradient and weight norms, batch-norm statistics — only if training turns
   unstable.

Runs G and G' (from 2026-10-02) need none of this to be judged: their final buffers
hold every position, with checkpoints for every step and the actors' game logs, so
items 2, 3 (without the prior: it can be recomputed from the checkpoints), 4 and 6
can be computed afterwards.

**Items 1-3 built and checked** (2026-10-03, our copy's trainer; no switch, they only
log): `learner.jsonl` gains `loss_by_side` (each side's policy and value loss and
positions; the model now also returns each position's losses, detached), the
new-against-trained check's `new_by_side` / `trained_by_side`, `selfplay` (player 0's
return — in Thud the dwarfs' — with a histogram of 21 buckets of 0.1, and the counts of
each `endings` category) and `by_side` (per side: positions, median legal moves, moves
visited, top share, effective moves, the prior's median effective moves, the mean KL
divergence of the visit counts from the prior, how often the prior's top move is the
most visited, and the values' sign accuracy and size at 7 points of the side's own
turns); the learner's text log a line of losses by side, and each actor's game line
its ending. **Checks:** `identity_check` 12 of 12 (one learning step equal to
upstream's: the losses unchanged); in a 3-step smoke run (32 x 2, 20 simulations,
augmentation on) the two sides' losses weighted by their positions reproduce the mixed
ones to ~1e-7 (training losses and both halves of the new-against-trained check); all
32 logged games replayed in `pyspiel` give the logged ending and return, the margin
matching the pieces left; the per-step search statistics match `az_buffer_stats` on the
same 6,736 positions; `instrumentation_check` (new) plays one hand-built position to
each of the five endings (each with a control one move earlier) plus a cut-off game
and another game's end, 12 of 12, and catches a planted error (the turn limit's `>=`
as `>`); `conv_policy_check` and `augmentation_check` still pass. The prior statistics
are plausible on the smoke run's untrained network (its prior near uniform: 269.7
effective moves of 270) but were not recomputed independently. **First real use:** run
G' continued from 2026-10-03 (`~/thud-runs/stage5b_Gaug_continue.sh`).

*Design of items 1-4* (Claude, 2026-10-02; each behind no switch — they only log — but
each with a check that it measures what it claims, as every change here):

- **1. Per-side splits.** The side to move is in every stored position: the
  observation's "trolls to move" plane (`thud::kTrollsToMovePlane`, plane 3 counting
  from 0, one value over the whole plane; `thud.h`), so no format change. The model's loss function gets a variant returning per-position
  policy and value losses (the trainer's means are their averages), and the learner
  averages them per side: the training losses, the new-against-trained check, and
  `value_accuracy` / `value_prediction` (from each state's `current_player`). Check:
  the two sides' averages, weighted by their counts, reproduce today's mixed numbers.
- **2. Self-play results as margins.** Per step: the dwarfs' mean final margin, its
  spread and a histogram (the trajectories' returns), and how each game ended — all
  dwarfs or all trolls captured, no legal move, the 200-turn no-capture limit, the
  800-turn limit — read from the final state in the actor and carried in the
  trajectory. Check: on a few logged games, replayed from their move lists, the same
  margins and endings.
- **3. Search statistics per side.** Per step, from the new positions' targets: moves
  visited, the most visited move's share, effective moves (exp of the entropy), as
  `az_buffer_stats` computes them (check: equal on a saved buffer). And from each
  root, recorded by the actor as three numbers per position: the network's prior's
  entropy (its own prior, before root noise: the evaluator's cached one, no extra
  network call), the KL divergence of the visit counts from it (how far the search
  moved the policy), and whether the prior's top move is the most visited.
- **4. A fixed validation set** — **shelved 2026-10-03** (the user; *How we evaluate
  networks*, the pilot's result; kept here for reference). Built by a new tool (from `az_target_quality`):
  positions from self-play games of several networks (the untrained anchor, A14, C16,
  D29, E44: varied play, both sides), every 7th position, ~4,000 (~2,000 per side:
  top-move agreement to about ±2 points); each searched without root noise by the
  **reference chosen by a pilot** (*How we evaluate networks*, below; the candidates
  A14 at 8,000 simulations with upstream's rule, pure MCTS, E44) — its visit counts the
  policy target, its root value and the game's final margin the value targets. Stored
  **versioned**: the positions in one file, each version's targets in another (the
  replay buffer's format, so our tools read it), with a note of how the targets were
  made. Never in any training buffer (its games are its own). Scored per checkpoint
  and side: policy loss against the reference visits, top-move agreement, distance (as
  `az_target_quality`), value error. Two ways to use it: the trainer loads it
  (`--validation_set=PATH`) and logs the scores before each learning step (~4,000
  inferences, a few seconds), and an offline tool scores any run's saved checkpoints
  — so runs A, C, C', D, E, G and G' can be compared on the same positions
  retroactively. Checks: the untrained network scores worst; the set ranks our
  networks as the matches did (G'16 above G16, C16 above C'16, E44 above D29 above
  C29); a second build with another seed gives the same ranking. The reference
  network itself is excluded from judging (its own score is inflated).

**How we evaluate networks** (decided with the user, 2026-10-03 — the reasoning, to
keep for the rest of the project):

- **Two jobs, two tools.** (1) *Strength*, by matches. Playing every new network
  against every past one costs quadratically; instead a **ladder of fixed anchors**:
  each new network plays the 2-3 anchors nearest its strength (as trained, pairs with
  the sides swapped, read per side), and one rating fitted to all ladder results
  (Elo-like, e.g. Bradley-Terry on the margins) places it — linear cost. **Built
  2026-10-03: `thud/experiments/ladder.py`** — every game's margin fitted by least
  squares, a player having a dwarf strength D and a troll strength T, a battle with x
  as the dwarfs and y as the trolls expected to end at 32 tanh((D_x − T_y) / 32) (the
  margin's ceiling of ±32; fits better than linear: 42 against 40 of 53 matches
  inside their intervals with every match in); a player is a network as it searched
  (another rule is another player); rating R = D + T, the expected margin per pair
  against the anchor, **E44**, with standard errors and a fit check against every
  match. **The untrained network does not belong on the ladder**: with its matches in,
  the misfits were exactly those — A14, which beats our strong networks, beats it by
  only +26.1 where C16 searching broadly beats it by +46.4: beating a near-random
  opponent does not predict strength between strong ones. So its matches are left out
  (they stay per-side diagnostics). **First ratings** (17 matches between trained
  networks, 13 within their intervals; as trained, against E44): A14 +1.7 ± 2.1, E44 0,
  D29 −5.4 ± 2.1, C29 −10.3 ± 2.5, G'16 −10.5 ± 5.2, C16 −13.3 ± 2.2, G16 −21.4 ± 3.9,
  C'16 −21.5 ± 3.3 — the order of every head-to-head we have (G'16 above G16 by 10.9,
  observed 10.2; C16 above C'16); per side, C''s trolls +7.4 against E44's (its dwarfs
  −28.9), A14's +12.4 (its dwarfs −10.6). **Anchors:** G'44 (promoted 2026-10-04), A14,
  E44, D29, C29, C16 (as trained). **Promotion rule** (decided by the user,
  2026-10-04): a network joins the anchors when it beats the top anchor head-to-head
  in at least 40 pairs with the 95% interval of its pair margin above 0 — about 97.5%
  one-sided confidence, as Leela Zero's gate (55% of 400 games, ~2 standard errors
  above 50%); the margin carries more information per game than a win, so fewer
  games suffice. Unlike Leela Zero's gate, ours does not choose which network
  generates the training games — it only adds a reference point — so a wrong
  promotion costs little. **G'44 promoted** (2026-10-04: beat A14 by +3.2, +0.7 to
  +5.8, 60 pairs). Each new network plays the 2-3 anchors nearest its rating, 40
  pairs each. When a network
  clearly beats the top anchor, a frozen copy joins the ladder. Anchors so far: the
  untrained network (A0), C16, D29, E44, A14. Rating against fixed opponents is, as far
  as we know, the usual practice in AlphaZero-style projects (cite a source before
  relying on details). (2) *Training health*, by the **validation set**: minutes, every
  step, per side, comparable across steps and runs — whether a run still learns and
  where (plateaus, overfitting, one side falling behind). It need not measure absolute
  strength, and stays informative while our networks are clearly below its reference.
- **The reference and its bias.** Any target made by a search carries the bias of what
  guides the search; the search corrects the network only partly (the user: it
  "partially eliminates" the network's effect, not wholly) — moves with a low prior get
  few visits, and the value head judges every leaf. Considered:
  - *A network's deep search, e.g. E44's*: favours its own lineage (E44's targets
    favour C, D and E) and inflates its own score.
  - *Pure MCTS* (random rollouts, no network): network-free, but shallow — with ~260
    dwarf moves even 100,000 simulations reach 2-3 plies, below which random play
    decides, while battles last ~300 moves — and blind to hurls (0.2-0.4% of the
    dwarfs' moves, so rollouts almost never play one: it misjudges dwarf threats; Phase
    5). As our networks improve, their agreement with it could fall: a metric that
    stops tracking strength. Cost ~0.3 ms a simulation on one core: 100,000
    simulations ~30 s a position, 4,000 positions ~3.5 hours on 10 cores; a million
    ~35 hours. How many it needs is unknown (likely 100,000 or more for its top move to
    settle) — bias, not budget, is the problem.
  - *A14's deep search* (8,000 simulations): the strongest network by the matches (E44
    loses to it by −3.5), outside every lineage now compared (C, C', D, E, G, G' descend
    from other runs), and under upstream's rule, as A trained, a losing side's search —
    usually the dwarfs' — tries every move before concentrating, so at high budgets its
    root depends much less on its prior; the bias left is mainly its value head. ~5
    hours for 4,000 positions (2.5 at 4,000 simulations, 10 at 16,000).
  - **Chosen by a pilot**: A14 at 2,000 / 4,000 / 8,000 / 16,000 (upstream's rule),
    pure MCTS at 25,000 / 100,000, E44 and G'16 at 4,000 (the default rule) — each one's
    stability (agreement with itself at double the budget), the agreement between them
    (where they agree, the target depends on no single network) and the cost. Expected:
    A14 at ~8,000 as the reference, pure MCTS as a network-free cross-check.
    **Result** (2026-10-03 12:17-13:41, `thud/experiments/az_reference_pilot.cc` and
    `.py`, `~/thud-runs/reference_pilot/pilot.jsonl`; 161 positions from E44's and
    G'16's self-play, 80 dwarf and 81 troll; intervals ±1.96 standard errors):

    | | Dwarfs: same top move | Dwarfs: distance | Trolls: same top move | Trolls: distance |
    |---|---|---|---|---|
    | A14 2,000 vs 4,000 | 69% ±10 | 0.15 | 75% ±9 | 0.12 |
    | A14 4,000 vs 8,000 | 65% ±11 | 0.14 | 77% ±9 | 0.11 |
    | A14 8,000 vs 16,000 | 71% ±10 | 0.15 | 85% ±8 | 0.11 |
    | pure MCTS 25,000 vs 100,000 | 25% ±10 | 0.013 | 17% ±8 | 0.013 |
    | A14 8,000 vs E44 4,000 | 19% ±9 | 0.69 | 21% ±9 | 0.42 |
    | A14 8,000 vs G'16 4,000 | 20% ±9 | 0.47 | 31% ±10 | 0.39 |
    | E44 4,000 vs G'16 4,000 | 23% ±9 | 0.60 | 19% ±9 | 0.39 |
    | A14 16,000 vs pure MCTS 100,000 | 21% ±9 | 0.53 | 15% ±8 | 0.41 |

    - **Pure MCTS is unusable**: at 100,000 simulations its visits are almost uniform
      (0.013 from its own 25,000), its top move a near-random pick among ties — random
      rollouts barely tell moves apart.
    - **A14's deep search is moderately stable** (71% / 85% from 8,000 to 16,000), less
      than E44's (88% / 81% from 2,000 to 4,000, *Is 2,000 a stable reference?*) —
      upstream's rule spreads its visits over every legal move (median 204.5 of 204.5
      for the dwarfs; E44 visits 96).
    - **Strong searches disagree on the best move** — ~20% agreement between any two of
      A14, E44 and G'16 — **but agree on who is winning** (root values correlate
      0.89-0.94). A single reference's top move is mostly its style; part of it is that
      many moves are nearly equal, which "same top move" punishes.
    - Cost: A14 at 8,000 ~165 s a search with 32 at once (~1,550 simulations a second
      in all: 4,000 positions ~5.7 hours); pure MCTS at 100,000 12 s with 6 at once.
    - **Proposed instead** (Claude, 2026-10-03; for the user to decide): values as the
      primary signal — error against the consensus of several deep searches' root
      values (A14, E44, G'16: they agree); for the policy, **regret** instead of
      agreement — how much value the network's preferred move gives up against the
      best move by a deep search's per-move values, so that equally good moves cost
      nothing (A14's search, visiting every move, estimates each one; the pilot tool
      would also record each move's mean value); and before trusting either, calibrate
      on known match results (G'16 > G16, C16 > C'16, E44 > D29 > C29). A set of ~1,000
      positions with all three references costs ~2.5 hours — after run G''s matches.
  - **Decided instead (the user, 2026-10-03): no validation set — the ladder for
    strength, the trainer's statistics (items 1-3) for training health**, as Leela Zero
    (a new network promoted only if it beats the best in matches: 400 games, 55%) and
    Leela Chess Zero (matches, plus a held-out tenth of its own self-play data as a test
    set — no external reference) do. Why: items 1-3 already show training health per
    step and side (they would have shown C''s dwarf policy stuck near uniform during the
    run), and the new-against-trained check, now per side, is in effect Leela Chess
    Zero's held-out test loss; the set's unique promise — comparability across runs, a
    cheap stand-in for strength — is what the pilot undermined (the policy targets are
    a reference's style), and the ladder gives both by measuring strength itself. A
    value-only set (the consensus of several deep searches' root values, cheap: A14's
    values settle by 2,000 simulations) was judged of low value: it cannot see the
    policy, our recurring weak point (C''s value head was decent). The pilot tool and
    these results stay, should a specific need arise.
- **Every reference is outgrown; the first set is version 1.** Signals that a version
  is outdated:
  1. *No headroom*: when building it, record the reference's agreement with itself at
     double the budget (e.g. A14 at 8,000 against 16,000, perhaps ~85-90%) — the
     ceiling. When our best networks' raw policies come near it, the set has nothing
     left to measure.
  2. *Decoupling*: at every promotion to the ladder, check that the set ranks the new
     anchor above the older ones; when its ranking stops following the ratings, it is
     outdated. Free.
  3. *Direct* (rare, ~8 hours on this CPU): when a network clearly tops the ladder,
     play it at the reference's budget against the reference searcher; if ours wins,
     rebuild.
- **Rebuilding is cheap**: re-search the *same positions* with the then-best network at
  a high budget (hours here, minutes on a GPU) as a new version; score the last few
  networks on both versions to chain the scales; refresh the positions themselves now
  and then — future networks play different games (distribution shift).

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
    over several steps while the fresh-game loss rises. **Our trainer reports it since
    2026-10-02** (user asked): before each learning step, the losses on 2,048 of the
    positions new since the last step and on 2,048 it trained on, taken right after that
    step — the same network in the same mode as it plays (`log-learner.txt`, "Before
    learning"; `learner.jsonl`, `loss_before_learning`). Part of a gap is the newer
    network's play, not memorising. Control (2026-10-02): a 256-position buffer trained
    on 150 batches a step shows it plainly — policy loss 7.12 on new positions, 3.76 on
    trained ones — where an ordinary short run showed none (4.53 and 4.66).
  - **Game length changing a lot**: lengths fluctuate with style from step to step (run
    A: 143-310 moves). Counts if the change is large (say a third) and lasts, since the
    buffer then holds correspondingly more or fewer games.

- **Replay buffer size: to be increased** (user, 2026-09-27; first kept at the default
  65,536 positions, ~240 games). Run A's step 8 beat its final network, one of the
  triggers below — though regressions are a known part of self-play: AlphaGo Zero let a
  new network generate games only after it won "by a margin of 55%", while AlphaZero
  "simply maintains a single neural network that is updated continually", as OpenSpiel
  does (arXiv 1712.01815). Two jobs to keep apart: the **replay buffer** (what the
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
  separate settings per side (user, 2026-09-26), the dwarfs, with 4-9x the trolls' legal
  moves in run C's self-play (medians 156-273 against 18-64), can get full searches more
  often, or larger ones; since only full searches become
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
  2. Settings per side in the code (two values instead of one), but **equal to start**:
     `p` 0.25 for both, `N` 400 and `n` 100 — KataGo's `p` and `n`, `N` below its 600
     for this CPU. About 1.75x run C's simulations per move, so ~57% of its games an
     hour, and 25% of positions recorded. A cheaper variant: `N` 300, `n` 50 (~1.1x).
     First proposed with the dwarfs' `p` at 0.5 (~2.1x, ~38% recorded); changed
     2026-09-28, because the reason for it has gone. The dwarfs needed more simulations
     under upstream's rule, which spread their searches over every move; since 3a their
     searches are about as focused as the trolls' (run C after step 22: a median 17 moves
     visited, the most visited holding 0.18; the trolls 14 and 0.19). But focus is not
     search quality: the dwarfs' 17 are ~11% of their 156 legal moves, the trolls' 14 are
     ~64% of their 22. A good dwarf move the network overlooks is rarely found at 100
     simulations, so more simulations may still help the dwarfs more. **The uneven
     match** measures it (`~/thud-runs/night_2026-09-28.sh`; parked 2026-09-29 — per-side
     settings become a setting to tune later): C's final network at 400 simulations against
     itself at 100 (pairs with sides swapped; `az_match`'s `sims_a` and `sims_b`), then
     100 against 100 on the same 100 openings, both with the new rule.
     `thud/experiments/az_sims_gain.py` takes per opening the dwarfs' gain (their
     margin with 400 minus the baseline's) and the trolls', with 95% t intervals on
     each and on their difference; battles in C's new-rule match spread ~5 points, so
     that interval should be about ±2.5 points or narrower. The dwarfs get a larger `p`
     only if their gain is clearly larger.
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
  **The cost at our scale** (2026-10-01, the user asked why it is so large): two factors
  multiply — a move costs 0.25 × 400 + 0.75 × 100 = 175 simulations instead of 100
  (1.75x fewer moves an hour), and only the full-search quarter is recorded (4x fewer
  training positions per move): **7x fewer training positions an hour** than run E
  (1.4 instead of 10 per 1,000 simulations; the cheaper variant 300/50 4.5x fewer, 50%
  full searches of 200 with quick ones of 100 3x fewer). With a learning step every
  21,845 recorded positions, 6 hours give run F ~2 steps against E's ~14, and its buffer
  stays ~90% old positions — too little for a comparison. KataGo could afford it: its
  baseline was the expensive search, its runs days of GPU time; our stage 1 found 100
  simulations better than 400 at equal time early on, when data is scarce. **The cheap
  hint first** (the user chose it): `thud/experiments/az_target_quality.cc` measures how
  much better the targets of a deeper search are, without training — self-play positions
  of D29 as the trainer plays them (root noise, the first 10 moves sampled, every 7th
  position, both sides; the program refuses an even step), each searched with root noise
  at 100, 400 and 1,000 simulations and once without noise at 2,000 as the reference; per
  side, how often each search's most visited move is the reference's, the share of its
  visits on the reference's best move, and the total variation distance of its visit
  distribution from the reference's (½ Σ|p − q|, the share of visits that would have to
  move; 1 minus it is their overlap — bounded, unlike the cross-entropy, which is
  infinite wherever the reference visits a move the target never tried); 400 against 100
  paired per position; control: 1,000 should come closer still. Queued after run E
  (`~/thud-runs/target_quality.sh`, ~30 minutes). If 400's targets are far closer, a
  longer comparison or the GPU decides; if barely, playout caps are not worth 7x less data
  at our scale.
  **Result** (2026-10-01 22:00-22:22, 864 positions from 16 games, 427 dwarf and 437 troll
  positions, medians of 165 and 26 legal moves; `~/thud-runs/stage1_matches/
  target_quality_D29.jsonl`; 95% intervals):

  | | 100 | 400 | 1,000 (control) | 400 minus 100 |
  |---|---|---|---|---|
  | Dwarfs: top move = the reference's | 51.8% | 57.4% | 62.5% | +5.6 points (+1.8 to +9.4) |
  | Dwarfs: distance to the reference | 0.525 | 0.464 | 0.395 | −0.061 (−0.071 to −0.051) |
  | Trolls: top move = the reference's | 24.9% | 27.5% | 35.2% | +2.5 (−2.9 to +8.0) |
  | Trolls: distance to the reference | 0.280 | 0.226 | 0.199 | −0.053 (−0.059 to −0.048) |

  The control holds (1,000 closer than 400 on every measure, both sides); the share of
  visits on the reference's best move stays at 12-13% throughout and says little. So
  400-simulation targets are better, but moderately: the distance shrinks by ~12%
  (dwarfs) and ~19% (trolls), the dwarfs' top move agrees ~5 points more often — not
  worth 7x fewer training positions an hour at our scale, consistent with stage 1 (100
  simulations beat 400 at equal time). **Recommendation (for the user to decide): no run
  F on this CPU; playout caps on the GPU**, where data is not the bottleneck. Also: the
  dwarfs' 100-simulation targets agree with a deep search only about half the time — a
  noisy training signal, consistent with their move probabilities being the weak point.
  **Is 2,000 a stable reference?** (the user asked, 2026-10-02): every simulation starts
  at the root and adds one leaf, evaluated by the network — no random playouts, and the
  priors concentrate the search on a few moves, so 2,000 such simulations go much further
  than 2,000 classical MCTS playouts (AlphaZero trained with 800). But the reference's
  stability was never checked: if 2,000 is still unsettled, every distance has a floor and
  the gain from 100 to 400 is understated. So `target_quality2.sh` (2026-10-02 03:28-04:02,
  after run E's evaluation; the program now takes `refs=2000,4000` and compares each
  deeper reference with the first — control: two identical references agree fully at
  distance 0; `target_quality_D29_refs.jsonl`, 400 dwarf and 409 troll positions) measured
  it. **Result: 2,000 is stable enough, and the conclusion stands.** 2,000 and 4,000
  agree on the top move in 87.7% of dwarf positions (84.5-91.0) and 80.9% of troll ones,
  at a distance of 0.126 and 0.069 — a quarter of 100 simulations' distance to either
  (dwarfs 0.53 and 0.56, trolls 0.27 and 0.31). Against the deeper reference the gain from
  100 to 400 is the same: dwarfs' top move +6.2 points (+2.5 to +10.0; against 2,000 in
  this run +5.8), distance −13% (−12%); trolls' distance −16% (−17%). 1,000 stays closer
  than 400 against both (the control). The recommendation above is no longer
  provisional.
- **Tree reuse**: the played move's subtree, with its statistics, becomes the next root
  (AlphaGo Zero did this; OpenSpiel builds a fresh tree each move, `mcts.cc:356`, and
  `RestartAt` does nothing, `mcts.h:173`). The dwarfs would profit through the trolls'
  search (user): the trolls' few moves concentrate visits, and under the chosen one the
  trolls' search has already explored the dwarfs' replies — so one tree shared by both
  sides, not one per side as OpenSpiel's self-play has. The potential can be measured
  with unchanged code first: the share of root visits in the move played, per side.
  Root noise must be re-applied to the reused root. Changes the search and the trainer.
  Expected (user, 2026-09-28): the share grows as the policy sharpens (seen once: run A's
  trolls held 57-77% while winning easily; run C held 13-17% through step 16, then
  18-20% by step 23 in its continuation); with
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
  **The sources re-read** (user asked, 2026-09-29) — AlphaGo Zero reused the tree, its
  successors do not in self-play, because of root noise:
  - *AlphaGo Zero* (Nature 2017, Methods, *Play*; UCL author manuscript): "The search
    tree is reused at subsequent time-steps: the child node corresponding to the played
    action becomes the new root node; the subtree below this child is retained along with
    all its statistics, while the remainder of the tree is discarded." Root noise: "adding
    Dirichlet noise to the prior probabilities in the root node s0 ... this noise ensures
    that all moves may be tried". It does not say how noise reaches a reused root, whether
    its 1,600 simulations per move are new or a total, nor whether the training target
    counts the inherited visits. (Its 40-block run also used a transposition table.)
  - *AlphaZero*: the official pseudocode builds a fresh root for every move
    (`run_mcts`: `root = Node(0)`, then `add_exploration_noise`, then the simulations) —
    no reuse.
  - *KataGo*: its self-play clears the tree before every search — `play.cpp`, when both
    sides are the same bot: "Also in self-play this makes sure root noise is effective on
    each new search"; the paper enables reuse only for gating matches, with noise off
    (arXiv 1902.10565, appendix on gating).
  - *Leela Zero* (issue #538, open since 2018): with reuse plus 1,600 new playouts, a
    forced sequence (a ladder) left 12,257 inherited visits at the root: "Only the
    additional 1600 playouts will have Dirichlet noise, making it impossible for the noise
    to change the move selected"; proposals: normalise the reused first-level visits
    (their PR 315), or count inherited visits as zero for choosing the move.
  - *Leela Chess Zero* (issue #775, closed 2020 as unfeasible): noise was not applied at
    a root made by reuse (the trap of re-applying it only when a node is expanded; ours
    is `mcts.cc:339`); their self-play trims the tree at the new root instead, and a
    maintainer: the network's evaluation cache can replace reuse in training. Also found:
    a reused terminal child made the new root terminal, so its children got no visits.
  **The evaluation cache** (`vpevaluator.cc:93-121`): the key is a hash of the legal
  moves and the observation — the position as the network sees it, including the side
  to move and both counters (Thud's planes 4 and 5: turns without capture, turns in
  all), so the same board at a later turn is a different entry; the entry is the
  network's whole output, the value and the move probabilities. 262,144 entries
  (`--inference_cache`, default `1 << 18`), least recently used out, shared by all actors,
  cleared after every learning step (`alpha_zero.cc:491`: the network changed). Run C's
  last steps hit it in 52-53% of the requests, but much of that is structural: the
  search asks for a position's value on its first visit (`mcts.cc:439`) and for its move
  probabilities when it expands it on a later visit (`mcts.cc:338`), and the network
  computes both at once, so the second request always hits. About 141 requests per move
  (3.09 million per learning step of 21,845 moves) suggest ~100 first requests (one per
  simulation) and ~41 expansions. **Measured** since 2026-09-29 (the evaluator's request
  counts, *First training runs*, the match program): in run D's self-play (steps 16-25)
  13-22% of the value requests hit — positions an earlier search of either side, or a
  transposition, had already evaluated — and all move-probability requests; in a
  noise-free match of C step 29 against itself with one shared evaluator 40-45%, since
  without root noise consecutive searches overlap more. The cache's 262,144 entries hold
  the positions of dozens of moves of every actor, so the repeats are not lost to
  eviction: the searches simply revisit that little. A larger cache adds nothing: 4x
  gave no gain (an apparent +42% came from replayed battles), and the turn counter in the
  key rules out exact repeats later in a game (it would cost ~0.5 GB per 262,144 entries
  at ~2 KB each, the move probabilities of ~100-150 legal moves).
  **The tree's memory limit**: the trainer caps each search at 10 MB
  (`alpha_zero.cc:182`, upstream's value; evaluation searches 1,000 MB), converted to a
  node count (`mcts.cc:266`: ~131,000 nodes of 80 bytes). When a search exceeds it, it
  prunes the children of rarely visited nodes and continues (`mcts.cc:501-520`) — no
  hard stop. Each expansion allocates all legal moves, so a 100-simulation search holds
  ~15,000 nodes (~1.2 MB), a 400-simulation one ~60,000 (~5 MB); pruning would start
  around 850 simulations. Irrelevant today; to raise (to 50-100 MB, a cap, not an
  allocation) together with bigger full searches (3b) or tree reuse.
  **Postponed** (user, 2026-09-29), because: (1) AlphaZero's pseudocode, KataGo and
  Leela Chess Zero do not reuse the tree in self-play, since reuse weakens root noise,
  and Leela Zero, which followed AlphaGo Zero, has an open issue about it; (2) what it
  would add is modest: the inherited visits — ~15-30% more visits a search at 100
  simulations (the buffer statistics; the dwarfs' most visited move reached 30% at C
  step 29), at almost no extra compute, less with the reset below. The cache spares the
  network calls when a search re-traverses positions the previous one evaluated, but not
  the simulations: in self-play 13-22% of the value requests are such repeats, about the
  inherited share, so each search spends roughly a fifth of its budget rebuilding
  statistics that reuse would keep. (Corrected 2026-09-30: this first said the cache
  makes re-searching free and that reuse adds mainly the visits' information, from a
  noise-free match with 40-45% repeats; the self-play counts show the cache saves the
  network calls only, so reuse's gain is the whole inherited share — still modest.) What
  ~20-30% more simulations a search are worth to learning is not measured yet: the
  parked uneven match (400 against 100 simulations, per side, *Changes to the search and
  trainer*, playout caps) would measure it. (3) The risk weighs most on the dwarfs:
  reuse weakens root noise, the exploration their move probabilities need — the
  forgetting check (2026-09-30) found exactly those narrowing onto a small repertoire —
  and weakening it now would also confound Step 2's test of the buffer. Revisit once the collapse is solved and playout caps are
  in, if the dwarfs' most visited move keeps more than ~30% of the visits (the watcher's
  statistics — the reusable share, which grows as the policy sharpens), or on the GPU if
  the search becomes the bottleneck.
  **How to build it:**
  1. In our copy's search (`mcts.{h,cc}`): keep the tree between searches and continue
     from the played move's child (upstream's `RestartAt` does nothing, `mcts.h:173`);
     free the rest, and keep the node count used for the memory limit right.
  2. In the trainer's self-play (`alpha_zero.cc`, `PlayGame`): one tree for both sides —
     the next search starts from the child of the move just played, whoever played it.
  3. A switch, off by default, with which self-play is exactly today's.
  4. **Root noise:** at the start of every search on a reused root, mix fresh noise into
     the priors of every child, before the first new simulation. The new root was an
     ordinary node, so its children carry the network's plain priors (no double noise).
     A root never expanded gets its noise at expansion, as now. This is Leela Chess Zero's
     bug: noise only at expansion never reaches a reused root.
  5. **The reset** (Leela Zero's proposal, their PR 315, "normalizing the visits/evals of
     a reused subtree's first level children nodes to one"): when a child becomes the
     root, give each of its visited children one visit carrying its mean value, and keep
     everything below unchanged. The noise then steers the new simulations, the training
     target (the root's visit counts) is almost only new visits, and the subtrees keep
     their statistics: only the root-level counts are reset, so most of the inherited
     information stays (every visit below the root's children). How much of reuse's gain
     that keeps is for test 6 below to measure. A few lines on top of steps 1-2; the alternative, counting
     inherited visits as zero only for choosing the move, needs separate counters.
  6. Budget: N new simulations (with the reset, inherited visits at the root are at most
     one per child, so they cannot swamp the noise; without it, they can — Leela Zero's
     ladder, 12,257 against 1,600). Decide together with playout caps.
  7. No terminal or solved flag may pass into a new root (Leela Chess Zero's second bug;
     our self-play runs without the solver, `alpha_zero.cc:183`).
  8. Raise the memory limit (above); log per move the share of the root's visits that
     were inherited.
  9. Matches: no gain (a player's own tree two plies later holds almost nothing), leave
     it off.
  **How to test it** (no training run needed for the search itself):
  1. Switched off: identical searches and games to today's (as `identity_check.cc`).
  2. The handover: after a move, the new root's subtree equals the played child's
     subtree before (counts and values); the node count stays consistent.
  3. Noise: fresh noise on every child of a reused root, mixed into plain priors, before
     the first new simulation; control: without the explicit step the priors stay
     noise-free (catches Leela Chess Zero's bug).
  4. The reset: each visited child of the new root has one visit and its mean value;
     every node below is unchanged.
  5. Reuse logs from self-play games, per side: the inherited share should match the
     buffer statistics (dwarfs ~21-30%, trolls ~18-20% for C step 29).
  6. Search quality offline: on self-play positions of a C network, with root noise,
     searches of 100 new simulations without reuse, with full reuse and with the reset,
     against a 2,000-simulation reference without noise: how often the chosen move
     agrees with the reference's, how close the visit distribution is to the reference's
     (a better training target), and how much exploration is left (visits outside the
     network's top moves). Paired on the same positions, several hundred to a thousand;
     controls: a fresh 130-simulation search must beat a fresh 100 (the test can see a
     gain this size), and reuse should match a fresh search with the same total visits.
  7. Then no worse in the next training run (the per-side health check against the
     anchor), or on the GPU.
- **A convolutional policy head** — **built 2026-10-02** (user's go-ahead, 2026-10-01),
  behind a switch: `--nn_model=resnet_conv_policy` (default `resnet`, upstream's head).
  120 move planes (8 directions x 14 distances + 8 capture steps) over the board instead
  of the 450 x 19,800 linear layer. Built as AlphaZero's chess and shogi heads ("an
  additional rectified, batch-normalized convolutional layer, followed by a final
  convolution of 73 filters", Silver et al., *Science* 2018, supplementary materials,
  *Architecture*) and Leela Chess Zero's (lczero-training
  `tf/tfprocess.py`: a 3x3 convolution with batch norm and ReLU, then a 3x3 convolution
  with a bias to its 80 planes, then a fixed 0/1 map to its 1,858 moves): here a 3x3
  convolution of the torso's width with batch norm and ReLU, a 3x3 convolution with a
  bias to the 120 planes, then each action's logit taken from its own entry
  (`thud::PolicyPlaneIndex`: plane d x 14 + k − 1 for a line move in direction d over k
  squares, 112 + d for a capture step, at the moving piece's square; the 60 cut-off
  corner cells of each plane are never used). The value head is unchanged. **Weights at
  64 x 4: 420,988 against 9,244,626** (measured; the earlier estimate said ~0.3M — the
  first 3x3 convolution adds ~37k). KataGo's global pooling bias in its policy head is
  not used: neither AlphaZero nor Leela Chess Zero has one; a candidate if the harness
  shows the head missing whole-board context. Code: `ResConvPolicyOutputBlock` in
  `thud/az/model.{h,cc}`; the map comes from the game when a network loads
  (`WithPolicyMap`, `thud/az/vpnet.cc`) and is not saved in checkpoints. **Tests:**
  `thud_test.cc` `TestPolicyPlaneIndex` (all 19,800 actions on distinct entries, each on
  its plane at its square by the test's own decoding, exactly the corner cells unused,
  two entries worked out by hand); `thud/az/conv_policy_check.cc` (each action's logit
  is its plane's entry at its square on random inputs and weights, 39,600 of 39,600,
  control with rows and columns swapped 2,160 — only the 9 diagonal squares; policies
  over exactly the legal moves summing to 1; every weight gets a gradient; both heads fit
  a fixed batch, policy loss 4.8 → 0.023 in 150 steps; a checkpoint reloads exactly,
  control a fresh network differs); `identity_check` still passes 12 of 12 for the old
  head; the trainer runs with it end to end. A planted error in the map, the symmetries
  or the observation transform is caught (four mutations). **Test on the layout-check
  harness:** `thud/experiments/az_head_check.cc`, the layout check's task on our copy,
  evaluating held-out test positions and as many training positions (overfitting);
  run 2026-10-02 04:02-05:16 (`~/thud-runs/head_check.sh`, `~/thud-runs/head_check/`):
  timing, then both heads at 32 x 2 and 64 x 4, seeds 1-3, 1,500 steps of 128. **Result:
  the new head is far better on the harness** (means of 3 seeds, min-max in brackets;
  mirror images: each test position against one of its 7 images, mapped back):

  | | linear 32 x 2 | new 32 x 2 | linear 64 x 4 | new 64 x 4 |
  |---|---|---|---|---|
  | Dwarfs' policy mass on hurls, test | 77.9% (75.8-81.3) | **99.4%** (99.2-99.6) | 75.0% (66.0-82.9) | **98.8%** (97.7-99.4) |
  | the same at step 500 | 47.2% | 96.1% | 42.3% | 97.3% |
  | the same on training positions | 96.3% | 99.4% | 94.3% | 98.8% |
  | Policy loss, test / training | 2.565 / 2.306 | 2.199 / 2.230 | 2.562 / 2.306 | 2.202 / 2.233 |
  | Value accuracy, dwarfs / trolls (test) | 97.9% / 99.0% | 98.1% / 99.2% | 98.1% / 98.7% | 97.4% / 97.2% |
  | Policy distance to the mirror image | 0.189 | 0.055 | 0.177 | 0.056 |
  | Value gap to the mirror image | 0.025 | 0.022 | 0.029 | 0.038 |

  - **Control: the linear head at 32 x 2 reproduces the layout control exactly**
    (77.9%, 75.8-81.3; values 97.9% and 99.0%), so our copy and the port are faithful.
  - **The linear head memorises**: 94-96% on positions it trained on, 75-78% on new
    ones, and a policy-loss gap of 0.26. The new head has no gap (its test loss is even
    slightly lower: the test targets are a little sharper) and reaches 96-97% by step
    500. Sharing weights across squares is what a hurl needs: the linear head has to see
    each square's hurls to learn them.
  - **It is 3x closer to symmetric without augmentation** (policy distance 0.055 against
    0.18-0.19); the value gaps are alike for both heads (0.02-0.04).
  - Values: no consistent difference. Both heads' value losses jump between evaluations
    (constant learning rate); the new head's 97.2% for the trolls at 64 x 4 is one seed's
    last evaluation (step 1,250: 99.5%).
  - **Timing** (`timing.jsonl`, alone on the machine, `OMP_NUM_THREADS` 4; 1 thread in
    brackets): one position 1.35 ms against 3.06 (2.69 against 6.72), but **a batch of 32
    18.9 ms against 17.0** (67.5 against 59.8) and a learning step of 128 positions 514
    ms against 448 (1,844 against 1,537), at 64 x 4. The linear head's 9.2M weights
    dominate small batches; in a full one, the new head's two 3x3 convolutions over 225
    squares cost more. Which is faster in the trainer depends on its batches, which
    average well below their limit of 32: in run C''s first step 8.6 positions (run C's:
    11.8), and C' produced 14.0 positions a second against C's 12.1 in its first step —
    but over the whole run it was 12% slower (below). At 32 x 2 the
    two heads cost about the same.
  - By its gate, **run C' went ahead** (hurl mass 98.8% against 75.0%; started 06:50;
    results below).
  With symmetry augmentation (5b, the 2 x 2): below, *Symmetry augmentation*.
  **Self-play test, run C'** (2026-10-02 06:50-13:38, `~/thud-runs/stage5_conv.sh`,
  `~/thud-runs/stage5_conv_C/`): run C's command with `--nn_model=resnet_conv_policy` the
  only change, from scratch to step 16, at nice 19 (C at normal priority), then each
  network as trained against C's (C's matches from Step 1):

  | Match | Pairs | Per pair (95%) | Won / drawn / lost | As dwarfs / as trolls |
  |---|---|---|---|---|
  | **C'16 vs C16** | 40 | **−8.5** (−12.6 to −4.3) | 10 / 0 / 30 | −16.4 / +7.9 |
  | C'8 vs the anchor | 20 | +10.4 (+5.6 to +15.2) | 14 / 5 / 1 | −17.4 / +27.8 |
  | C8 vs the anchor | 20 | +19.9 (+14.1 to +25.6) | 20 / 0 / 0 | +4.1 / +15.8 |
  | C'12 vs the anchor | 20 | +19.0 (+12.8 to +25.1) | 18 / 1 / 1 | −6.9 / +25.8 |
  | C12 vs the anchor | 20 | +31.0 (+26.9 to +35.0) | 20 / 0 / 0 | +2.4 / +28.6 |
  | C'16 vs the anchor | 20 | +17.1 (+12.1 to +22.1) | 19 / 0 / 1 | −10.3 / +27.4 |
  | C16 vs the anchor | 20 | +36.3 (+31.8 to +40.7) | 20 / 0 / 0 | +9.5 / +26.8 |

  - **C' loses to C at equal steps**: −8.5 a pair, 30 of 40 pairs lost; its dwarfs score
    −16.4 against C's trolls, C's dwarfs −7.9 against its trolls. By the roadmap's
    criterion (no worse in self-play head-to-head) **stage 5 fails in this run**.
  - **The new head learnt the trolls faster and the dwarfs much worse**: against the
    anchor its trolls +27.8 at step 8 (C +15.8, p < 0.001), level with C from step 12;
    its dwarfs −17.4, −6.9, −10.3 at steps 8, 12, 16 (C +4.1, +2.4, +9.5; p < 0.001 at
    8 and 16, 0.006 at 12). The trainer's own
    evaluator against MCTS, which does not go through `az_match`, agrees (step 16:
    −0.04 / −0.16 / −0.49 at its three levels, C −0.01 / −0.00 / −0.29). In self-play the
    trolls won every game at every step and the searches' values stayed confident (their
    size, the trainer's `value_prediction`, 0.8-0.97); in C's the dwarfs won 8 games at
    step 12 and the values' size fell to 0.2-0.5 by step 16.
  - **No overfitting** (the trainer's new check): before each learning step the new
    positions' policy loss is at or below that of positions already trained on (step 10:
    4.38 against 4.51; step 16: 4.33 against 4.32), the value losses alike after step 3.
  - **Speed**: step 16 after 6.72 hours against C's 5.98, 12% slower (13-16 positions a
    second from step 3 on, C 16-18; the user asleep until ~11:00, nothing else running);
    only step 1, on the untrained network, was faster (14.0 against 12.1). The head
    check's full-batch timing (~15% slower) was the better guide.
  - Caveats: one run each, and run-to-run variation at this scale is unmeasured; the
    harness tests fitting fixed targets, self-play also what the search does with the
    priors.
  - **Diagnosis from its saved data** (2026-10-02 from 17:20, the user's go-ahead,
    `~/thud-runs/stage5b_G.sh`, part 1). **Search breadth: unchanged** (`az_buffer_stats`
    on its final buffer, steps 14-16, against C's after step 16): the dwarfs' searches
    visit a median 19 moves in both, the most visited holding 14.1% in both, 16.7
    effective moves against 15.9; only the outcome differs, the dwarfs' mean return
    −0.86 against C's −0.47. So the first suspicion — stage 3a's problem, strong trolls
    making the dwarfs' searches spread out over moves judged lost, flattening their
    targets — is ruled out, and so is a sharp wrong prior narrowing them: the searches
    look the same, the dwarfs choose worse moves within them. **Mostly the policy,
    partly the value**: searching broadly (upstream's rule for both sides, every move
    tried once, picked by the values; `Cconv_step16_vs_A_start.jsonl`), C'16's dwarfs
    score +14.6 against the anchor, +24.9 over their −10.3 as trained; C16's rise by
    +12.1 (+21.6 against +9.5). So two thirds of the gap to C16 at the dwarf side (19.8
    as trained, 7.1 broad, p = 0.001) is the policy, one third the value; the trolls
    equal either way. **Why the policy: it barely learnt the dwarfs.** On its own final
    buffer (`az_forgetting`, 4,096 positions; the trainer's logged loss agrees: (5.33 +
    3.36) / 2 = 4.35) C'16's dwarf policy loss is 5.33 against the untrained network's
    5.38, its top move the most visited in 2.9% of dwarf positions (untrained 0.6%) —
    near uniform after 16 steps — while its trolls learnt (4.01 → 3.36, 7.4% → 11.2%).
    Run C learnt its dwarfs late too, but did: C8's dwarf loss on C's archive 9 was 5.36
    (untrained 5.39), C16's on archive 15 5.01, its top move the most visited in 12% of
    them, 59% on the later archives. (On C's archives C''s networks fit the dwarfs like
    an untrained one and the trolls worse than one: each run's trolls learnt their own
    repertoire — C16 fits C''s troll positions worse than uniform too, 4.66 against
    4.06 — and a near-uniform dwarf policy predicts nothing.) **Why the harness
    misled:** its learning rate is 10x the trainer's (1e-3 against 1e-4), and its policy
    target is simple and local (uniform over the capturing moves), where self-play's are
    visit counts over ~263 moves. The validation set (*Instrumentation*, item 4) is the
    better proxy: real deep-search targets. **For revisiting the head:** a higher
    learning rate for it, or a warm start — the head trained first on a finished run's
    buffer (E's 458,752 positions) — would skip the slow start; the slow start may also
    be what the dwarfs' policy needs most generally (C's took ~10 steps to begin).
  - **Ideas raised** (the user, 2026-10-02), each to wait for the diagnosis:
    - *Larger filters (5x5, 7x7) in the head.* The size advantage stays (7x7 in both
      head convolutions: ~0.89M weights against the linear head's 9.2M), the speed
      advantage goes (the head's two convolutions would cost ~130M multiply-adds a
      position against the torso's ~70M, ~5x the 3x3 head's). And it may not reach the
      problem: the torso's nine stacked 3x3 convolutions already see 19 x 19 squares, the
      whole board, from every square; a larger filter adds direct sight of 2-3 squares
      around a piece, while a dwarf moves up to 14. If the policy is to blame for long
      moves, a head that scores each move from the features at both its start and its
      destination square fits better (Leela Chess Zero's later attention policy works
      on start and destination squares; to verify before building on it).
    - *Training the dwarfs more than the trolls* — weighting dwarf positions more in
      the learner's batches; better mixed into every batch than in alternating
      dwarf-only phases, which pull the shared torso and value head back and forth.
      Cheap to build (a sampling weight per side). Each position is now sampled ~3
      times in its life, so doubling the dwarfs' share risks fitting their targets'
      noise; augmentation and the validation set proposed under *Instrumentation*
      would show whether it helps. **Waits for runs G and G'** (the user: augmentation
      may already help the dwarfs).
    - *A convolutional head for the trolls only*, the linear one for the dwarfs (both
      heads, chosen by the side to move). Two cautions: the new head's troll advantage
      was early speed (+27.8 against +15.8 at step 8) — by steps 12-16 C's trolls had
      caught up against the anchor, which is too weak to separate strong trolls; and if
      stronger trolls early hurt the dwarfs' learning in self-play, it would repeat
      that. A cleaner test of the trolls first: C'16 and C16 against the same strong
      dwarfs (each against E44), comparing their troll games. **Result** (2026-10-03
      10:19-11:48, `~/thud-runs/conv_trolls_check.sh`, 40 pairs each, as trained):
      **the new head's trolls are much better** — against E44's dwarfs C'16's trolls
      score +9.8, C16's −8.6 (18.4 better, p < 0.001); against E44's trolls C'16's
      dwarfs score −24.2, C16's −4.5 (19.7 worse, p < 0.001); per pair −14.4 and −13.0,
      the two effects cancelling. At step 16 the new head's trolls beat the best dwarfs
      we have (E44, trained to step 44). The diagnosis above weakens the first caution
      (C''s dwarf searches were as broad as C's; its dwarf policy simply did not
      learn), so **the hybrid — the convolutional head for the trolls, the linear one
      for the dwarfs — is a strong candidate** (for the user to decide; after stage
      5c). Building it: both heads in one network, each position's logits from the
      head of its side to move (`thud::kTrollsToMovePlane`), each head trained only on
      its side's positions; ~9.6M weights (the linear head's 9.2M plus the new head's
      ~0.1M).
  **Compatibility:** a network with this head is a new model type, which unmodified
  OpenSpiel cannot build or load (user asked, 2026-09-26); every other planned change
  leaves the network file as upstream's. If that matters, publish our copy with the
  network, or distil the final network into a standard resnet on our self-play data (the
  machinery of network growing). Two tree levels (move, then capture yes/no) would not
  help: 18,482 actions instead of 19,800, 7% fewer head weights, and an extra network
  call per choice. **Our other programs** (`az_match`, `az_forgetting`,
  `az_target_quality`, ...) compile our copy in: rebuild them before using them on a
  network with this head.
- **Symmetry augmentation** — **built 2026-10-02** (user, 2026-10-01: right after the
  head, as its own change), behind a switch, `--symmetry_augmentation` (default off,
  upstream's behaviour): the learner turns or mirrors each sampled position by a random
  one of the board's symmetries before training on it. AlphaGo Zero did this for Go:
  its "training data was augmented by generating 8 symmetries for each position"; the
  AlphaZero paper (arXiv 1712.01815), quoting that, does not augment for chess and shogi,
  whose rules are not symmetric. One random symmetry per sampled position gives all 8 in
  expectation. **Thud's board has 8 symmetries, not the 16 of
  a regular octagon** (user asked, 2026-10-01; checked): the identity, three quarter
  turns and four mirror images. A regular octagon's dihedral group has order 16, but
  Thud's octagon has edges of alternately 5 and 4 squares — an octagon with alternating
  edge lengths keeps only half the symmetry — and a turn by 45 degrees would not take
  squares to squares (rows are up to 15 squares long, diagonals up to 10). The
  transformation, `SymmetricTrainInputs` (`thud/az/vpnet.{h,cc}`), maps the observation
  (`thud::SymmetricObservation`), the legal moves and the policy target's moves
  (`thud::SymmetricAction`); the value stays. **Tests:** `thud_test.cc`
  `TestSymmetryHelpers` (squares, all 19,800 actions under all 8 symmetries against the
  test's own mirroring, each a permutation, all distinct; observations of mirrored
  random-game positions); `thud/az/augmentation_check.cc` against positions mirrored
  independently as text: observations, legal moves, the policy's moves (each leads to the
  mirror image of its child) and values, 1,608 of 1,608; symmetry 0 changes nothing;
  control: untransformed observations differ in 1,379 of 1,407 (the rest: the fully
  symmetric starting position); a planted error in the policy's mapping is caught. The
  harness's `augment=1` does the same, and it measures how differently a network answers
  a position and its mirror image (value gap, policy distance). **Harness result**
  (2026-10-02 05:49-06:45, `~/thud-runs/head_check_augment.sh`: 64 x 4, both heads, seeds
  1-3, with the head check's unaugmented runs the 2 x 2; means of 3 seeds, min-max in
  brackets):

  | 64 x 4 | linear | linear, augmented | new | new, augmented |
  |---|---|---|---|---|
  | Dwarfs' policy mass on hurls, test | 75.0% (66.0-82.9) | 86.8% (83.7-90.1) | 98.8% (97.7-99.4) | **99.1%** (98.6-99.6) |
  | the same on training positions | 94.3% | 89.8% | 98.8% | 99.1% |
  | Policy loss, test / training | 2.562 / 2.306 | 2.396 / 2.366 | 2.202 / 2.233 | **2.193** / 2.228 |
  | Value loss, test | 0.051 | 0.034 | 0.088 | 0.045 |
  | Value accuracy, dwarfs / trolls (test) | 98.1% / 98.7% | 98.5% / 99.6% | 97.4% / 97.2% | 97.6% / 99.6% |
  | Policy distance to the mirror image | 0.177 | 0.148 | 0.056 | **0.044** |
  | Value gap to the mirror image | 0.029 | 0.013 | 0.038 | 0.025 |
  | Seconds for 1,500 steps (3 runs at once) | 1,514 | 1,462 | 1,708 | 1,674 |

  - **For the linear head augmentation is a strong regulariser**: the policy-loss gap
    between training and held-out positions falls from 0.26 to 0.03, held-out hurl mass
    rises 12 points and was still rising at the end (72%, 78%, 87% at steps 1,000-1,500).
  - **For the new head it adds little**: +0.3 points of hurl mass, a slightly lower loss,
    the policy 20% closer to symmetric — its shared weights already generalise across
    squares. The new head without augmentation is far ahead of the linear head with it.
  - Values: lower losses with augmentation for both heads at the last evaluation, but
    the value numbers jump between evaluations (constant learning rate); no cost in time.
  - **Self-play test: runs G and G'** (the user, 2026-10-02; `~/thud-runs/stage5b_G.sh`,
    `~/thud-runs/stage5b_G_7x/`, `stage5b_Gaug_7x/`): fresh, run C's settings with E's 7x
    memory, to step 16, G' with augmentation the only difference (G 2026-10-02 17:42 to
    00:09, G' to 07:01; mains throughout, no pause). Each position is sampled ~3 times in
    its life there (64 batches of 1,024 a step, ~21 steps in the buffer), against ~10
    passes on the harness. **Results**, each network as trained (C's matches from Step 1):

    | Match | Pairs | Per pair (95%) | Won / drawn / lost | As dwarfs / as trolls |
    |---|---|---|---|---|
    | **G'16 vs G16** | 40 | **+10.2** (+7.0 to +13.4) | 32 / 5 / 3 | −2.0 / +12.3 |
    | **G16 vs C16** | 40 | **−7.9** (−10.8 to −5.0) | 10 / 0 / 30 | −7.9 / +0.0 |
    | G8 / G'8 / C8 vs the anchor | 20 each | +17.8 / +16.2 / +19.9 | | dwarfs −5.2 / +4.0 / +4.1; trolls +23.0 / +12.2 / +15.8 |
    | G12 / G'12 / C12 vs the anchor | 20 each | +25.7 / +36.7 / +31.0 | | dwarfs +0.5 / +13.1 / +2.4; trolls +25.2 / +23.6 / +28.6 |
    | G16 / G'16 / C16 vs the anchor | 20 each | +26.5 / +35.3 / +36.3 | | dwarfs +5.1 / +12.7 / +9.5; trolls +21.4 / +22.6 / +26.8 |

    - **Augmentation helps clearly, through the dwarfs**: G'16 beats G16 by +10.2 (32 of
      40 pairs); against the anchor G''s dwarfs beat G's at every step (+9.2, +12.7,
      +7.7 at steps 8, 12, 16; p = 0.008, 0.001, 0.016); the trolls equal at 12 and 16
      (at step 8 G''s lower, −10.9). In self-play G''s dwarfs won 18 games at step 16 (G
      13, C 0). G' took 6.75 hours to step 16, G 6.32 (G' ran overnight on an idle
      machine, so the ~7% is likely the learner's transformation of 65,536 positions a
      step on one thread).
    - **A 7x memory from step 1 is worse than C's 1x at step 16**: G16 loses to C16 by
      −7.9 (30 of 40 pairs); against the anchor G's trolls trail C's at steps 12 and 16
      (−3.4, −5.4; p = 0.018, 0.001), its dwarfs at step 8 (−9.3, p = 0.001). The only
      difference is the buffer (same learning cadence, 21,845 positions a step, and the
      same 64 batches a step): through step 16 G keeps training on every position since
      step 1, the weak early games included. KataGo's rule, taken literally, would have
      done much the same: its buffer starts at c = 250,000 positions and holds all data
      below that — every position through our step ~11, ~290,000 at step 16 (*Settings
      to determine*, 7) — so its constants do not transfer to our scale, where C's
      65,536 learnt faster early, and E's gain came from a longer memory *later*, in a
      run already trained. So **grow the buffer from a small start**: KataGo's formula
      with c = 65,536 (α 0.75, β 0.4) gives ~2.3x at step 16, ~4.6x after a million
      positions. Needs a growing buffer in our copy (sample only the newest N_buffer
      positions of a larger buffer).
    - G' against C (not played directly): against the anchor about equal in total
      (+35.3 against +36.3), its dwarfs a little better (+3.2, p = 0.29), its trolls
      worse (−4.2, p = 0.008) — augmentation about makes up for the large buffer's
      handicap.
    - The trainer's overfitting check shows nothing here: with the large buffer, new
      positions score *better* than trained ones (G'16: 4.17 against 4.52), because the
      trained sample spans the run's whole history, its old targets flatter — the drift
      *Instrumentation* predicts; the validation set will measure this properly.
    - Caveat as always: one run per setting.
  - **G' continued** (from 2026-10-03 13:41, `~/thud-runs/stage5b_Gaug_continue.sh`, the
    first run on the instrumented trainer; matches after step 44). What the new
    statistics show by step 39 (22:00):
    - **The dwarfs' prior is almost uniform, all along**: its effective moves equal the
      legal moves within a few percent at every step from 17 to 39 (200 of 201 at step
      17, 244 of 261 at 39), its likeliest move is the search's most visited in 1-2.5%
      of dwarf positions; confirmed by another code path, the new positions' dwarf
      policy loss, only 0.05-0.3 below log(legal moves), a uniform prior's loss (step
      39: 5.25 against 5.56). The trolls' prior learns, slowly (policy loss 3.67 →
      3.03; effective 20.5 of 24 moves; 13% top agreement). The dwarfs' play is
      therefore the value head choosing among ~19 nearly arbitrary candidates — which
      fits the broad-search results (C16's dwarfs +12.1 with every move tried, C''s
      +24.9).
    - **Hypothesis, unchecked: root noise sets the dwarfs' targets.** With a prior of
      ~0.004 per move, the noise (ε 0.25 spread by α 0.1 over ~26 of ~250 moves, ~0.01
      each) outweighs the prior several times, so the ~19 moves a dwarf search visits
      are mostly the noise's picks; the target records them, the network cannot predict
      noise, the prior stays flat — a loop. Remedies to examine: **KataGo's forced
      playouts and policy target pruning** (arXiv 1902.10565, section 3.2, read
      2026-10-03 via ar5iv): each root child gets at least n_forced(c) = (k P(c) Σ N)^½
      playouts from the noised prior, and afterwards, from every child but the most
      visited, up to n_forced playouts are subtracted from the policy target as long as
      that child's PUCT value stays below the best's — because "the vast majority of the
      time, noise moves are bad moves, and in AlphaZero since the policy target is the
      playout distribution, we would train the policy to predict these extra bad
      playouts"; it decouples "the policy target in AlphaZero from the dynamics of MCTS
      or the use of explorative noise". Also weaker or per-side noise, and more
      simulations for the dwarfs.
    - **Tested the same evening — confirmed for G'** (`thud/experiments/az_noise_check.cc`,
      2026-10-03 22:05-22:25: the reference pilot's 80 dwarf and 81 troll positions, each
      searched as self-play records it, 100 simulations with root noise under two seeds,
      and once without noise):

      | Dwarfs | Prior below uniform | Two noise seeds' most visited agree | Agrees with the noise-free search | Visits on noise-boosted moves (chance) |
      |---|---|---|---|---|
      | G'39 | **0.04 nats** | **2%** | **4%** | 71% (60%) |
      | C16 | 0.40 | 25% | 34% | 58% (59%) |
      | E44 | 0.94 | 49% | 60% | 41% (48%) |
      | C29 | 2.50 | 88% | 89% | 10% (26%) |

      In G''s dwarf searches the noise decides: another seed almost never gives the same
      most visited move, and the visits crowd onto the noise-boosted moves. E44's and
      C29's priors have learnt and their searches follow them rather than the noise; C
      had begun by step 16. (Trolls: G'39's seeds agree 7%, E44's 19%, C29's 38%, C16's
      30% — their 20-30 moves leave more room to the noise for every network.) **Why G'
      never started** (22:25-22:40, same positions):

      | Dwarfs | Prior below uniform | Seeds agree | With the noise-free search |
      |---|---|---|---|
      | G16 (7x buffer from step 1) | 0.10 nats | 2% | 6% |
      | G'16 (the same with augmentation) | 0.003 | 0% | 3% |
      | G'29 | 0.01 | 0% | 2% |
      | D29 (C's line, 4x from step 15) | 0.61 | 42% | 47% |

      **The 7x buffer from step 1 stalls the dwarfs' policy** (G16 0.10 against C16
      0.40 nats, the buffer the only difference) **and augmentation stalls it further**
      (G'16 0.003) — a near-uniform start plus noise-made targets kept from the very
      first steps, then eight orientations of them. In C → D → E the long memory came
      after C's small buffer had started the dwarfs' policy. Yet G'16 beat G16 by
      +10.2: augmentation's gain came through the value head (G''s value losses are
      lower), the dwarfs playing by value-guided search among noise-chosen moves. So
      the two directions this points to: **policy target pruning** (keep the noise out
      of the targets, so the policy learns from the value-driven part of the visits) and
      **a buffer grown from a small start**. For the user to decide; the CPU is free
      after G''s matches (tomorrow morning).
    - **G''s results** (trained to step 44 by 2026-10-03 23:40, 10 hours on mains, no
      pause; matches until 2026-10-04 03:23; each network as trained):

      | Match | Pairs | Per pair (95%) | Won / drawn / lost | G' as dwarfs / as trolls |
      |---|---|---|---|---|
      | G'29 vs C29 | 40 | **+13.3** (+9.9 to +16.6) | 36 / 1 / 3 | −4.2 / +17.4 |
      | G'29 vs D29 | 40 | **+11.9** (+8.4 to +15.4) | 36 / 0 / 4 | +1.5 / +10.4 |
      | G'44 vs E44 | 40 | **+8.0** (+4.1 to +11.8) | 30 / 0 / 10 | −2.4 / +10.4 |
      | **G'44 vs A14** | 60 | **+3.2** (+0.7 to +5.8) | 36 / 8 / 16 | −3.5 / +6.8 |
      | G'20 / 24 / 29 / 44 vs the anchor | 20 each | +24.3 / +26.3 / +33.8 / +31.5 | | dwarfs +9.8 / +10.8 / +11.0 / +12.1; trolls +14.6 / +15.5 / +22.8 / +19.4 |

      - **G' is the strongest network so far and the first to beat A14** (+3.2, 36 of 60
        pairs won, 16 lost): it beats C29 and D29 at equal steps by +13.3 and +11.9, and
        E44 at equal steps by +8.0. **Ladder** (`ladder.py`, 21 matches, 17 within
        their intervals): G'44 +6.9 ± 2.5, G'29 +6.3 ± 3.3, A14 +2.0, E44 0, D29 −5.1,
        C29 −9.6. By the proposed promotion rule (the top anchor beaten in ≥ 40 pairs,
        interval above 0) G'44 becomes the top anchor — the rule awaits the user.
      - **Its dwarfs are the best we have**: against A14 −3.5, E44's −13.7 (+10.1, p <
        0.001), its trolls weaker (+6.8 against E44's +10.1, p = 0.009); against the
        anchor its dwarfs +12.1, E44's +8.7 (p = 0.14), its trolls +19.4, E44's +29.0 (p
        < 0.001) — while its dwarf prior was near uniform: the value head carries them.
      - **The long memory from step 1 pays off by step 29**: G16 lost to C16 (−7.9), G'29
        beats C29 by +13.3 — the early handicap is overtaken (G without augmentation was
        not continued, so the two changes are not separated after step 16). This weakens
        the case for a growing buffer as far as strength goes; the stalled dwarf policy
        remains its argument.
      - **The dwarfs' prior begins to learn at the end**: its likeliest move is the most
        visited in 2.5%, 3.1%, 3.6%, 4.1%, 5.6%, 7.7% of dwarf positions at steps 39-44
        (effective moves 244 of 261 → 201 of 219); in self-play the dwarfs' mean return
        stays ~−0.1, battles ending with the dwarfs gone ~50-60 and the trolls gone
        ~20-28 times a step. Huge headroom if the dwarfs' policy learns properly — the
        argument for policy target pruning. **Decided next** (the user, 2026-10-04):
        forced playouts and policy target pruning, behind a switch. Note: pruning
        changes each position's policy target, not the number of positions stored;
        with our 100 simulations it removes only ~(k P N)^½ ≈ 1-4 visits from each
        noise-boosted move (KataGo's searches have 600-1,000 visits), so whether it
        cleans our targets enough is measured first — the noise diagnostic on pruned
        targets (two seeds' agreement, G' raw: 2%); if too little, weaker noise for
        the dwarfs is the next lever. Then a run against G' at equal steps; the hybrid
        head decided after it (the user: first see that the dwarfs learn).
        **Built 2026-10-04** (our copy, behind `--policy_target_pruning`, default off):
        `MCTSBot`'s new `forced_playouts_k` (0 off) gives a root child with playouts
        but fewer than (k P N)^½ an infinite selection value; `PrunedRootVisits` (in
        `thud/az/mcts.h`) subtracts up to that many playouts from every child but the
        most visited while its PUCT value, the mean held, stays below the best's, and
        zeroes a child left with one; k = 2 as KataGo. In self-play the actors search
        with forced playouts and record the pruned counts as the policy target; the move
        is still chosen from the unpruned counts, the evaluator's and the matches'
        searches unchanged. **Tests** (`thud/az/pruning_check.cc`, new): five hand-built
        children against values worked out by hand (the most visited kept; one above
        the best's PUCT at once kept; one pruned to a single playout dropped to 0; one
        stopped by the PUCT bound before n_forced; an unvisited one 0), a planted error
        (the single-playout rule removed) caught; in 40 real searches (G'44, 100
        simulations, root noise) visited root children left below n_forced: 6 of 809
        with forced playouts, 62 of 807 without — at 100 simulations n_forced is only
        ~1-4 playouts, so forcing binds for few children; `identity_check` 12 of 12
        with the switch off. **Test plan** (the user asked, 2026-10-04): equal steps,
        not equal time — pruning is free and forced playouts keep 100 simulations a
        move, so a step costs the same; first the noise diagnostic on pruned targets,
        then a run of G''s settings with pruning to step 16 (one night), judged first
        by the new statistics (the dwarfs' prior moving away from uniform by steps 8-16:
        G'16 0.003 nats, C16 0.40) and then by matches against G'16.
        **The diagnostic says it does not work at our scale** (2026-10-04,
        `az_noise_check` extended to pruned targets; the pilot's positions, two noise
        seeds): G'44's dwarf targets carry 65% of their mass on noise-boosted moves raw
        and 65% pruned, 15.2 against 14.7 effective moves, the two seeds' targets 0.758
        and 0.761 apart; E44 alike (41% / 41%). Why: pruning removes only *forced*
        extra playouts, and at 100 simulations there are almost none; the noise acts
        earlier — with a near-flat prior it reshapes the prior PUCT itself uses, so the
        noise-boosted moves are visited legitimately and pruning leaves them. So no
        training run with it. (The diagnostic also shows G'44's dwarf prior 0.076 nats
        below uniform, against 0.039 at step 39: learning slowly at the end of G'.)
        **Next candidate: a policy target from the search's values, not its visits** —
        Gumbel AlphaZero's (Danihelka et al., "Policy improvement by planning with
        Gumbel", ICLR 2022; the paper cited above for "AlphaZero can fail to improve its
        policy network, if not visiting all actions at the root"): π' = softmax(log π +
        σ(completed Q)), the completed Q being each visited move's search value and, for
        unvisited moves, the mixed value (the network's value blended with the
        prior-weighted mean of the visited moves' values), rescaled to [0, 1]; σ(q) = (50
        + the largest visit count) × 0.1 × q (DeepMind's `mctx`,
        `qtransform_completed_by_mix_value`, read 2026-10-04: `maxvisit_init` 50,
        `value_scale` 0.1, `rescale_values` on). The target raises the moves the search
        judged good and lowers those it judged bad by their values, wherever the
        visits went — so noise choosing which moves get evaluated no longer becomes the
        target. Used here with our PUCT search (the paper pairs it with its own Gumbel
        root search — an adaptation to test). The user asked whether training toward
        values is wise (picking the final move by value instead of visits plays
        badly — the "robust child" rule): the target does not choose the move (we still
        play the most visited), it shifts the prior's log-probabilities by a bounded
        σ(Q) rather than taking the maximum, and unvisited moves get an average; but
        with ~5 visits a move our Q values are noisy, so it trades root noise for value
        noise — not settled by the literature for our case. Kept in reserve. If ever
        used with our PUCT search: σ does not discount rarely visited moves (its scale
        follows the *largest* visit count; a move with 1 visit enters with full weight —
        Gumbel's own search gives its candidates equal visits), so add the user's
        threshold (moves under m visits count as unvisited) or shrink each Q toward
        the mixed value by n / (n + n0).
        **More simulations do not fix it either** (2026-10-04, `az_noise_check
        sims=N`, G'44's dwarf searches): two noise seeds' most visited moves agree 6% at
        100 simulations, 11% at 200, 10% at 400 (with the noise-free search 10%, 16%,
        14%); the extra simulations spread the search (effective moves 15, 24, 37)
        instead of settling it — with a flat prior over ~220 moves even 400 is ~2
        visits a move. E44, whose prior learnt, agrees 49% at 100. **So the cure is a
        prior that learns early**: C's small buffer started the dwarfs' prior by step 16
        (0.40 nats; seeds 25%), G's 7x did not (0.10), the buffer the only difference —
        **the growing buffer** (C's 65,536 at first, then KataGo's formula with c =
        65,536) with augmentation is the best-supported candidate, Gumbel's target in
        reserve. Open risk: augmentation slowed the dwarfs' prior (G'16 0.003 against
        G16 0.10), and C with augmentation was never run; the per-step statistics show
        it within a few steps.
        **Built 2026-10-04** (the user's go-ahead; `--replay_buffer_start_size`, default 0
        = every position stored, as upstream): `GrowingBufferSize` and `SampleNewest` in
        `thud/az/alpha_zero.h` — the learner (and the new-against-trained check's
        trained sample) draws only from the newest W(N) positions, W = N up to the
        start, then start (1 + 0.4 ((N / start)^0.75 − 1) / 0.75); distinct samples by
        Floyd's algorithm; `growing_buffer_size` logged per step. **Tests**
        (`thud/az/growing_buffer_check.cc`, new): the formula against values worked out by hand
        (153,375 at 350,000 generated, 300,431 at a million, start 65,536); samples all
        among the newest, distinct, as many as asked, before and after the buffer wraps;
        uniform by a chi-square test (311 and 332 for 299 degrees of freedom, below the
        99.9% point 375; a first min/max bound was poorly calibrated — the extremes of
        300 counts — and replaced); control, upstream's whole-buffer `Sample` draws
        older positions too; a smoke run (start 3,000) logged buffer sizes 2,160, 3,526,
        4,259 — the formula's values. **Run W** (`~/thud-runs/stage5b_W.sh`, queued
        2026-10-04 after the target-quality run): G''s settings with
        `--replay_buffer_start_size=65536`, fresh, to step 16; judged first by the dwarfs'
        prior (G'16 0.003 nats, C16 0.40), then W16 against G'16 and C16, W against the
        anchor, `az_noise_check` on W16.
        **Do more simulations bring the dwarfs' targets closer to a deep search?** (the
        user, 2026-10-04: with a near-flat prior, 200 simulations may approximate the
        truth better than 100 — the question an equal-time comparison does not answer)
        `~/thud-runs/target_quality_Gaug44.sh`: G'44's positions at 100 / 200 / 400 with
        noise against 4,000 and 16,000 without (a million simulations would take ~10
        minutes a position and still be judged by the same value head; 4,000 against
        16,000 shows whether the reference has settled). If 200 is clearly closer, a
        per-side budget (200 for the dwarfs, 100 for the trolls, ~1.5x the compute) is
        the next test. **Result** (2026-10-04 15:52-16:36, 111 dwarf and 113 troll
        positions, `target_quality_Gaug44.jsonl`): **only slightly closer** — the
        dwarfs' top move agrees with the 16,000-simulation search 5.4% at 100, 7.2% at
        200, 9.9% at 400 (neither gain significant), the distance 0.833, 0.756 (−0.078,
        −0.099 to −0.056), 0.670 (−0.164); against 4,000: 9.9%, 14.4%, 15.3%. All far
        from a deep search — D29, whose prior had learnt, agreed ~52% at 100 against its
        2,000 — and the deep reference itself has not settled for the dwarfs (4,000
        against 16,000: 52% agreement, distance 0.215; trolls 68%). So a per-side budget
        buys a little, not a fix; the prior is the problem. **Parked** for the GPU phase,
        where simulations are cheap; run W (the growing buffer) is the experiment that
        matters. A cheap test: dwarf searches of G'39 at 100 simulations with two
      noise seeds and without noise — how much the visited moves and the most visited
      move follow the noise.
    - **A crisis and a recovery, steps 33-37**: the dwarfs' mean return in self-play fell
      from ~−0.2 to −0.89 and recovered to −0.09 by step 39; the trolls' legal moves
      doubled (~30 → 64: new kinds of positions); the value loss on new dwarf positions
      jumped from 0.04 to 0.54 while staying ~0.05 on trained ones — self-play found
      positions the value head did not know, and it adapted within ~5 steps. Exactly
      what the new-against-trained check per side is for. AlphaGo Zero also evaluated positions in a randomly chosen symmetry during
  the search (the same paper); not built — a candidate if augmentation helps.
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
