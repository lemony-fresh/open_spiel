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
  legal and illegal moves, reach tables and the position after a move, plus a test that
  every test diagram is a position the reader accepts. **Re-run it after changing any
  test.** Its first run found two diagrams with 12 trolls, which the reading rules decided
  that day reject.

### Phase 4 — Integration tests and baselines — **done** 2026-09-25

The full OpenSpiel suite passes 285 of 285.

- The playthrough baseline, `open_spiel/integration_tests/playthroughs/thud.txt` (890 KB),
  from `./open_spiel/scripts/generate_new_playthrough.sh thud`: one seeded random game of
  308 turns, which the trolls win by taking the last dwarf (returns -0.75 / +0.75).
  **Replay it after any change to the rules, the encoding, the text formats or the
  observation** (`thud/run_tests.sh playthrough`: its recorded moves replayed, every line
  compared, as upstream's `playthrough_test.py` does); a difference means behaviour
  changed. Generating a new one is not the test: it plays new random moves (corrected
  2026-10-15; CLAUDE.md has the update command for an intended change).
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
| 2 — done 2026-09-27 | **Our own copy** of the C++ AlphaZero and its MCTS in `thud/az/`, changed in nothing but its namespace (`open_spiel::thud_az`; parameter names identical) — made by `import_from_upstream.py` from upstream commit 540bba6e, built by `thud/az/build.sh`. Passed: (1) **textual identity** — `import_from_upstream.py --check`: all 12 copies are exactly a fresh import (a one-character change is caught), which covers code no test reaches (the batching queue, the trainer's threads, resuming); (2) **`identity_test.cc`** against upstream's code: equal network outputs, equal searches without and with root noise and with random rollouts and the solver (50 of 50 positions each), a checkpoint of ours loads in upstream's code with equal outputs, an equal learning step; every control differs — which covers the layout probe; (3) the **tic-tac-toe control** through `az_trainer` (session 5's settings, 19 minutes): losses 1.55 / 0.50 → 1.27 / 0.07 → 0.96 / 0.04 at steps 1, 17, 26 (upstream's copy: 1.56 / 0.47 → 1.27 / 0.07 → 0.95 / 0.05), self-play draws 38% → 86% (upstream 41% → 83%), against MCTS at 40 / 126 / 400 simulations +0.36 / +0.26 / +0.12 at the end (upstream +0.34 / +0.38 / +0.06), draws against stronger MCTS. **Resuming works** in upstream's trainer (run A2) and in our copy (run C resumed 2026-09-28: step 17 continued from 352,553 positions with the full buffer and the latest model reloaded) | Textual identity; identical outputs, searches, training step; upstream loads our checkpoint; the tic-tac-toe control learns as upstream's did |
| 3a — done 2026-09-28 | **Value of untried moves** — urgent (user): the dwarfs' searches spread over nearly every move once the network judges them lost, so their policy targets stay flat, and the better the trolls get, the worse it becomes. **Implemented** in our copy (`thud/az/mcts.{h,cc}`, passed through the trainer as `--untried_move_value` and `--untried_move_reduction`, saved in `config.json`): three rules — `upstream` (0), `sibling_mean_minus_reduction` (the visit-weighted mean of the visited siblings minus 0.2 × √(their prior mass), KataGo's rule with the siblings' mean for the parent's value), `loss` (AlphaZero's) — at every node, **defaulting to sibling_mean_minus_reduction** (user, 2026-09-27; an exception to the upstream-default rule above). Tests: `untried_move_test.cc` (the formula on a hand-built node; on 85 self-play positions of run A's step 14 at 100 simulations the dwarfs' searches visit a median 99 moves with upstream's rule, 31 with the default, 1 with `loss`, which without root noise never leaves its first move; the trolls' 17, 23, 1); `identity_test.cc` still passes with `upstream`, and the default changes 50 of 50 searches. **Run C** (run A's settings with the default rule, 6 hours, 2026-09-27 20:31 to 2026-09-28 02:32): **no collapse, and much stronger dwarf play** — against the anchor step 8 +10.4, 12 +43.5, 14 +40.1, 16 +46.4 (run A: +18.9, +23.1, +26.1, +3.9); at step 16 as dwarfs +21.6 (A: −24.9); the dwarfs' searches visited a median 19 moves at every step (A: 95-99). Head-to-head at equal positions, C step 16 against A step 16: **+20.9** a pair (+17.3 to +24.4, 20 of 20 pairs); against A's best, step 14: −1.2 (−6.9 to +4.4) searching with upstream's rule, +3.9 (−0.2 to +7.9) with the new one — neither significant. Details in *First training runs*. **Continued to step 29** (2026-09-29): even with step 16 head-to-head (+1.1), but its dwarf play much weaker against A step 14 (−20.0 as dwarfs, step 16 −2.4; −11.5 a pair) and the anchor, with upstream's rule in the searches — forgetting or the evaluation's rule? (*First training runs*). **Evaluated as trained** (Step 1, 2026-09-29): no collapse (C's dwarfs never below −0.6 against the anchor), but a drift from step 16 on — step 29 beats step 16 (+4.1), yet its dwarfs lost 10 points against the anchor and it loses to A step 14 (−8.5): specialisation; Step 2 (the buffer) indicated — the quick forgetting measurement found the dwarfs' move probabilities drifting; **Step 2** (run D, a 4x memory, 2026-09-30): no policy drift, D29 beats C29 (+7.3) and C16 (+8.4), still loses to A14 (−5.4) — forgetting explains part; **run E** (D continued with a 7x memory to step 44, 2026-10-01): E44 beats D29 (+9.7), its dwarfs recover against the anchor (+8.7, D29 +3.2), no forgetting, still loses to A14 (−3.5) | Beats run A head-to-head at equal positions, and the dwarfs' searches narrow |
| 3b — after Step 1 (below) | **Playout cap randomisation**, with settings per side in the code but **equal to start** (proposal in *Changes to the search and trainer*) — unless the **uneven match** (C's final network, step 29, against itself, 400 against 100 simulations, parked 2026-09-29: per-side settings become a setting to tune later) shows that extra simulations gain the dwarfs clearly more than the trolls; then the dwarfs get full searches more often, or budgets scale with the number of legal moves. (User asked about more simulations for the dwarfs, 2026-09-27. The case for them was upstream's rule, which spread the dwarfs' searches over every move; since 3a their searches are about as focused as the trolls' — but over 7-11% of their legal moves, the trolls' over 34-78% (medians in run C's buffers), so whether more simulations help them more is measured, not assumed. Unequal budgets tilt self-play towards the dwarfs, a bias to measure. The trolls' floor of ~100 simulations came from upstream's rule, under which a search visits every move once before any twice; it no longer applies.) | Beats the previous best setting head-to-head at equal time |
| 3c — postponed 2026-09-29 | **Tree reuse, one tree shared by both sides** — **postponed** (user, 2026-09-29; why, how to build and how to test it in *Changes to the search and trainer*, tree reuse: AlphaZero, KataGo and Leela Chess Zero do not reuse in self-play, because reuse weakens root noise; it would add ~15-30% more visits a search — the cache spares the network calls of re-traversing the previous search's positions, not the simulations; the dwarfs' collapse involves exploration) — only if the reusable share stays large (own-tree reuse ~0. Shared-tree share = the most visited move's share, from the buffer statistics: with upstream's rule the dwarfs' 1-3%, the trolls' 57-77% while they won easily, then 17-20% (run A); **with the new rule ~13-14% for the dwarfs and ~13-17% for the trolls at 100 simulations through step 16 (run C), rising to 18-20% for both by step 23 in its continuation, and for the dwarfs to 21-30% at steps 26-29 (the trolls 18-20%) — roughly 15-30% more simulations per search, against playout caps' 1.37x**; re-read as the policy sharpens, since the watcher records it every step). **Priority** (user asked, 2026-09-28): kept after 3b — on 2026-09-28 it bought ~15-20% more simulations a search (on 2026-09-29 the dwarfs' share reached the ~30% mark below at run C's step 29: one reading so far, to discuss with the user), playout caps 1.37x in KataGo's measurement, and it is the larger change (a tree kept across moves and shared by both sides, root noise re-applied, a larger memory limit than the trainer's 10 MB); 3b's budget semantics (new simulations, or topping up to N visits) are chosen with reuse in mind. **Move it up** if the shared-tree share rises above ~30% for either side in the watcher's statistics, or when 3b is done | Same searches with reuse off; beats 3b's setting head-to-head |
| 4 | Settings: `uct_c`, root noise α (0.03 and 0.3 against 0.1), temperature drop | Each change beats the previous setting head-to-head |
| 5 — built 2026-10-02 | **Convolutional policy head** (user, 2026-09-26: later in the roadmap; started 2026-10-01 at the user's request) — first on the layout-probe harness, with an exhaustive test of the action-to-plane map. It is the one planned change that makes our networks unloadable by unmodified OpenSpiel. **Built** behind `--nn_model=resnet_conv_policy` (as AlphaZero's and Leela Chess Zero's heads; 420,988 weights at 64 x 4 against 9,244,626), map and model tested (*Changes to the search and trainer*); **the harness passed** (2026-10-02, `az_head_probe`, held-out and training positions, both heads, 32 x 2 and 64 x 4, seeds 1-3): the dwarfs' policy mass on hurls 98.8-99.4% on held-out positions against the linear head's 75-78% (which reproduces the layout control), no overfitting (the linear head: 94-96% on its training positions), 3x closer to symmetric; ~15% slower per full batch of 32 or learning step, faster per small batch, and 12% slower over run C'; then **run C'** (`~/thud-runs/stage5_conv.sh`): run C's settings with the new head as the only change, from scratch to step 16, against C as trained at equal steps and against the anchor — its gate (the harness not clearly failing) was met; trained 2026-10-02 06:50-13:38, 12% slower than C, no overfitting, but **its dwarfs much weaker** than C's at equal steps (against the anchor −10.3 at step 16, C +9.5; the trolls equal); **C'16 loses to C16 head-to-head, −8.5** (30 of 40 pairs): stage 5 fails in this run; the cause to analyse first (for the user) | Faster policy learning on the harness, no more overfitting; then no worse in self-play head-to-head |
| 5b — built 2026-10-02 | **Symmetry augmentation** (user, 2026-10-01: right after the head, as its own change) — each sampled position turned or mirrored by a random one of the board's **8** symmetries (not 16: *Changes to the search and trainer*), behind `--symmetry_augmentation`, default off; tested; on the harness with both heads (the 2 x 2, 2026-10-02): **the linear head gains much** (held-out hurl mass 75.0% → 86.8%, its overfitting gone), **the new head little** (98.8% → 99.1%, policy 20% closer to symmetric); no measurable cost. **Self-play: G'16 beats G16 by +10.2** (2026-10-03; its dwarfs better at every step) — **adopted** (the user, 2026-10-03), in every run from now on | Less overfitting or faster learning on the harness; then no worse in self-play head-to-head |
| 5c — decided 2026-10-02, started 2026-10-03 | **Instrumentation** (the user, 2026-10-02): per-side splits of the trainer's losses and value statistics, self-play results as margins and how games ended, search statistics per side (breadth, the prior's entropy, how far the search moves it) — **built and checked 2026-10-03**; and **a ladder of anchors with a rating** for strength (*How we evaluate networks*) — **the rating script built 2026-10-03** (`ladder.py`; anchors A14, E44, D29, C29, C16, and G'44 at the top since the promotion rule was decided, 2026-10-04). The fixed validation set was **shelved** (the user, 2026-10-03): the pilot showed strong searches agreeing on the best move only ~20% of the time, so its policy targets would measure a reference's style; its value-only variant judged of low value. **The evaluation watcher and the game analyser** (items 5 and 6) **built 2026-10-05**, for the GPU phase's tuning | Each statistic checked against an independent computation (done: the mixed numbers, replayed games, `az_buffer_stats`, a hand-built position per ending); the ladder's ratings reproduce the matches we have |
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
  Every self-play position of runs A2 to D is archived (~99.4%: the archiver copies a
  few hundred positions late after each third step). Since the 7x buffer (run E on) no
  archiver has run (run E's archive folder is empty): a run's final buffer holds its
  last ~21 steps, which for our branch runs of 19 steps is every position they made
  (run K: ~415,000 new positions, the buffer 458,752), and the actors' logs keep every
  self-play game's moves (not its policy targets); long runs will need the archiver
  again (`~/thud-runs/archive_buffer.sh`, a copy every `reuse` steps; corrected
  2026-10-09).
- **Match-aware play** — after a strong single-battle agent (*Deferred decisions*).
- **End-of-battle limits** — re-measure once the engine plays strongly (*Deferred
  decisions*).
- **Gumbel AlphaZero** — only if few-simulation search remains the bottleneck after 3b
  and 3c; a large change.
- **Convolutional policy heads, again** (the user, 2026-10-03) — **done: since runs L and M
  (2026-10-11/12) the convolutional head serves both sides; the hybrid (the trolls only)
  is superseded** (the user, 2026-10-15). The original note: for both sides, or for
  the trolls only (the hybrid), once our networks are much stronger: run C' compared
  them at step 16 of a 64 x 4 network, where the dwarfs' policy had barely begun to
  learn; with a stronger, larger network, a warm start or a higher learning rate for
  the head, the verdict may differ (*Changes to the search and trainer*, run C').

**What we have tried, and what to keep** (overview, 2026-10-02, at the user's request;
rows added through 2026-10-08; the evidence is in the sections named in the decision log
below):

| Feature | What we found | Verdict |
|---|---|---|
| C++ AlphaZero instead of Python | Python's model has a layout bug and is slow | **Keep** C++ |
| Batched inference (32 actors, 2 inference threads) | 2.7-7x faster self-play | **Keep** |
| Our own copy of the AlphaZero code | Proven identical to upstream; needed for every change | **Keep** |
| 100 simulations a move (against 400) | 100 won at equal machine time early in training | **Keep** for now; revisit with playout caps on the GPU |
| Root noise α 0.1 | A compromise between the sides' move counts; never compared | **Keep**; tune on the GPU |
| Untried moves at the siblings' mean − 0.2 (3a) | Stopped run A's collapse; much stronger dwarfs (C16 beat A16 by +20.9) | **Keep** (default) |
| Larger replay buffer, 2x / 4x / 7x | 2x weak evidence; 4x (D) stopped forgetting, beat C29 +7.3; 7x (E) beat D29 +9.7, no forgetting; 7x from step 1 with augmentation (G') loses to C at step 16 but beats C29 by +13.3 and E44 by +8.0 at equal steps — though it stalls the dwarfs' policy | **Keep 7x from step 1** |
| Growing buffer (from 65,536, KataGo's rule; run W) | Started the dwarfs' policy ~15-20 steps earlier than G', but W29 loses to G'29 by −20.2 — its trolls drifted (against the anchor +3.8 at step 29, G' +22.8) | **Drop** (`--replay_buffer_start_size` stays, off) |
| `learner_batches` (fixed training per step) | Lets a larger buffer keep run C's training per step | **Keep** |
| Margins as the value target | The value learns the expected margin, our objective | **Keep** |
| Matches with each network on its own rule, read per side | Showed the dwarfs' drift that mixed results hid | **Keep** |
| Overfitting measure in the trainer | Works (control 7.12 against 3.76); none in run C' | **Keep** (a log only) |
| Symmetry augmentation (5b) | Harness: linear head 75% → 87%, overfitting gone; self-play: G'16 beats G16 by +10.2, its dwarfs better at every step | **Keep**, in every run |
| Convolutional policy head (5) | Harness: 99% against 75-78%; self-play: run C' −8.5 against C — its dwarf policy barely learnt, but its trolls are much better (+18.4 against E44's dwarfs) | ~~Revisit as a troll-only head~~ — **superseded** (2026-10-15): the convolutional head for both sides since runs L and M (L102 beat K102 by +10.72 a pair, its dwarfs included) |
| Convolutional policy head on a trained trunk (2026-10-10) | The long-capture probe: K83's linear head ignores multi-square hurls; the same frozen trunk with a convolutional head trained on run K's buffer (convK83) makes 2-square hurls its top move 81% of the time and shoves visible at every distance; convK83 beat K83 by +16.8 a pair (39 of 40) | **Testing** in training (run L against run K continued) |
| Playout caps (3b) | Not built; 7x fewer training positions an hour for moderately better targets | **Revisit** on the GPU |
| Tree reuse (3c) | Not built; ~15-30% more simulations a search, weakens root noise | **Revisit** after 3b, or if the reusable share passes 30% |
| Resignation | Fits margins badly | **Drop** |
| Policy target pruning with forced playouts (KataGo; 2026-10-04) | Built behind a switch; the diagnostic: at 100 simulations there are almost no forced playouts to prune — G'44's dwarf targets hold 65% of their mass on noise-boosted moves raw and 65% pruned | **Drop** here (`--policy_target_pruning` stays, off); revisit with more simulations |
| A replay buffer that moves positions into their slots and compacts itself after loading (2026-10-05) | Upstream's kept every slot at the largest position it had held and loaded with 45% to spare | **Keep**; the fix sent upstream (PR #1637) |
| The dwarfs' policy temperature 1.5 (run H, 2026-10-07) | No collapse in 19 steps, where G' collapsed within 8 from the same start; H83 beat G'64 by +11.75 (37 of 40 pairs), its dwarfs ~+10 stronger on the ladder; one run each, no other value tried | **Keep** (run H's settings carried into runs I-M; K83, from that line, the top anchor) |
| Players without a network as yardsticks (hexparrot, alpha-beta depths 1-3, MCTS; 2026-10-07) | Every network loses — to hexparrot by 47-57 a pair, to depth 2 by ~50; they exposed the dwarfs' blunders, which self-play hid | **Keep**, for evaluation; to join the ladder and the trainer's evaluator |
| More search against the blunders (G'64 at 800 simulations) | Lost as badly: −53.9 against hexparrot (−55.4 at 100), −49.5 against depth 2 (−49.5) | **Not the fix** |
| Training against alpha-beta, 30% of the positions at depth 2 (run I) | Its trolls learnt to punish (an adjacent dwarf taken 46% of the time in self-play, run H 23.5%), its dwarfs did not (every game against depth 2 ended at −32: no signal); I83 against H83 +2.73 (−0.55 to +6.00), not promoted | **Keep as run K's base**; not enough alone |
| Alpha-beta's depth per side, depth-1 trolls (run J) | No gain for the dwarfs (they still lose every dwarf against depth 1); J83 against I83 −10.20 (−14.48 to −5.92) | **Drop** |
| Kickstarting from alpha-beta's move scores, depth 2 (2026-10-08) | Run K (weight 0.3, temperature 0.25): K83 beat I83 by +30.0 a pair (40 of 40) and H83 by +30.6 (40 of 40); the first win against an outside opponent (alpha-beta depth 1, +8.7); mostly the trolls | **Keep** (K83 the top anchor) |
| Input planes from the rules (20 planes 2026-10-08, replaced by 50 on 2026-10-11) | 48 move planes (direction x distance, at the moving piece) and 2 maps; every test passes (hand-built positions, the move generator, all 8 symmetries, each plane on its own channel, the surgery exact) | **Testing** (run M against run L) |
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
| 2026-09-26 | Convolutional head: later in the roadmap (stage 5), first on the layout-check harness — a Thud game is far easier to get accepted upstream than a new network type, so upstream compatibility matters | user | *Changes to the search and trainer* |
| 2026-09-26 | Our own copy of the C++ AlphaZero and its MCTS, for the changes of stage 3 | user | *Deferred decisions* |
| 2026-09-26 | Untried moves: measure first | user | *Margins as the value target* |
| 2026-09-26 | Replay buffer: default 65,536 positions for now | user | *Notes for later* |
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
| 2026-09-29 | Step back: on this CPU only "does it work" questions, tuning on the GPU (3c and 4 after the switch) | Claude's proposal, **open** | status block |
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
| 2026-10-03 | The convolutional head's trolls are much better (against E44's dwarfs +9.8 against C16's −8.6), its dwarfs much worse: a hybrid head (convolutional for the trolls, linear for the dwarfs) is a strong candidate | measured; for the user to decide | *Changes to the search and trainer*, run C' |
| 2026-10-03 | Cloud GPU: start on a weaker, cheaper one with many CPU cores, move to stronger ones later; on every machine switch, reason about or benchmark the machine-dependent settings first | user | roadmap, 6; *Cloud GPU options* |
| 2026-10-03 | Re-evaluate convolutional policy heads (for both sides, or the trolls only) once our networks are much stronger | user | *Later, when a trigger fires* |
| 2026-10-03 | No fixed validation set (the pilot: strong searches agree on the best move only ~20% of the time; a value-only set of low value): the ladder for strength, the trainer's statistics for training health, as Leela Zero and Leela Chess Zero do | user | *How we evaluate networks* |
| 2026-10-04 | Run G' (fresh, 7x, augmentation) is the strongest network: beats C29, D29, E44 at equal steps and A14 (+3.2); top of the ladder (+6.9 against E44) | measured | *Changes to the search and trainer*, runs G and G' |
| 2026-10-04 | Next change: KataGo's forced playouts and policy target pruning in our copy (behind a switch); first the noise diagnostic on pruned targets (do two noise seeds agree much more?), then a run against G' at equal steps; the hybrid head after it | user (Claude's proposal) | *Changes to the search and trainer*, runs G and G' |
| 2026-10-04 | Ladder promotion rule: beat the top anchor in ≥ 40 pairs with the 95% interval of the pair margin above 0 (≈ Leela Zero's 55% of 400 games); G'44 promoted to top anchor | user | *How we evaluate networks* |
| 2026-10-05 | The growing buffer is not adopted: run W started the dwarfs' policy earlier but W29 loses to G'29 by −20.2 (its trolls drifted); G''s settings (7x from step 1, augmentation) stay the baseline; the dwarfs' policy is not what limits strength at this stage | measured; the user (confirmed) | *Changes to the search and trainer*, run W |
| 2026-10-05 | Tune the algorithm settings (`uct_c` with the untried-move reduction, root noise, the temperature's drop) on the GPU, not on this CPU; continue run G' tonight meanwhile | user | *Cloud GPU options*, How we tune the algorithm settings |
| 2026-10-05 | Instrumentation items 5 and 6 (the evaluation watcher, the game analyser) built now, for the GPU tuning; item 7 only if training turns unstable | user | *Instrumentation* |
| 2026-10-05 | The replay buffer moves each position into its slot and compacts itself after loading (our copy, `thud/az/replay_buffer.h`; upstream's kept every slot at the largest position it held and loaded with 45% spare); run G' stopped after step 54 to resume on it | user | *Instrumentation*, the buffer's memory |
| 2026-10-05 | The batching evaluator's moves reverted: they saved neither compute nor RAM, and similarity to upstream helps | user | *Instrumentation*, the buffer's memory |
| 2026-10-05 | The load test's second half not run: its first half's answer is clear (the evaluation watcher costs ~25-35% of self-play beside the trainer on this machine, mostly efficiency) | user | *Instrumentation*, item 5 |
| 2026-10-05 | Send the replay buffer fix upstream as its own small PR (generic, no Thud content; after the Google CLA, with the GitHub noreply address) — opened 2026-10-06 as [#1637](https://github.com/google-deepmind/open_spiel/pull/1637) | user | *Upstreaming*, a first, smaller PR |
| 2026-10-06 | G'44 stays the top anchor; promotion also needs the ladder rating above the top anchor's, the 95% interval of the difference above 0 (G'73 met the head-to-head test only by exploiting G'44's dwarfs: ladder difference +0.4, −2.0 to +2.8). Part 2 of G''s evaluation (the strength curve, steps 48-72) before deciding how to go on | user | *How we evaluate networks*; runs G and G' |
| 2026-10-07 | G'64 the top anchor (it passes both tests of the tightened promotion rule: +8.60 against G'44 in 40 pairs, ladder difference +5.8, +3.7 to +7.9); G'44 stays an anchor | user | *How we evaluate networks* |
| 2026-10-07 | Outside opponents: hexparrot's AI through a bridge first; OpenSpiel's MCTS with far more than 1,000 simulations and its alpha-beta at a strong depth, all overnight; equal-time comparisons later, by an iteratively deepening alpha-beta | user | *How we evaluate networks* (outside opponents) |
| 2026-10-07 | Training against alpha-beta: 30% of the training positions (not games: those games are short and half recorded) from games of our network against OpenSpiel's alpha-beta at depth 2 (each side half the time), only our network's positions recorded; first test a constant share in run I (G'64 with run H's settings plus these games) against run H's steps 65-83; in long runs tapered by results per side to a ~10% floor, stronger opponents as ours improve; not imitation first | user | *How we evaluate networks* (training against alpha-beta) |
| 2026-10-07 | H83 the top anchor (both tests of the promotion rule: +11.75 against G'64 in 40 pairs, ladder difference +12.2, +8.7 to +15.6); run H the main line proposed; G'64 stays an anchor | user | *How we evaluate networks*; run H |
| 2026-10-08 | Run J: run I with alpha-beta's depth per side (1 as the trolls, so that our dwarfs' games against it carry a signal; 2 as the dwarfs), against run I from the same start | user | *How we evaluate networks* (training against alpha-beta) |
| 2026-10-08 | Kickstarting first, the input features from the rules as their own experiment later: the capture check showed the prior following the targets (G'64 → I83: captures 0.123 → 0.169 of the trolls' prior), so the network can represent captures, and the targets ask for them too weakly (the confirmation, the other half of what the user agreed would decide, runs on the night of 2026-10-08) | measured (the check the user agreed would decide) | *How we evaluate networks*: input features from the rules, kickstarting |
| 2026-10-08 | Kickstarting's teacher at depth 2 for both sides: depth 3 is affordable for the trolls (~20-35% of the trainer's throughput) but what the diagnosis found lies within depth 2 | user | kickstarting |
| 2026-10-08 | Run K: run I plus kickstarting (weight 0.3, temperature 0.25, depth 2), from run I's start, against run I; launched on the user's go | Claude (proposed), user (adopted) | kickstarting |
| 2026-10-08 | The settings built since 2026-10-05 join the list to revisit on a new machine: alpha-beta's CPU cost (group 1), simulations per side and alpha-beta's depths (2), the policy temperature per side (3), the scaffolds' schedules (a new group 5) | user (asked), Claude | *Cloud GPU options* |
| 2026-10-09 | A tool to restructure a network — copy weights (function-preserving) or train from a saved replay buffer — designed now, built when run K's results pick the first change of structure (input features, a convolutional or hybrid head, or none yet) | user (asked), Claude (design) | *How we evaluate networks*, a tool to restructure a network |
| 2026-10-10 | K83 the top anchor (both tests of the promotion rule: +30.57 against H83 in 40 pairs, +27.28 to +33.87; ladder difference +42.9, +37.2 to +48.6); H83 stays an anchor | user | *How we evaluate networks*; kickstarting |
| 2026-10-11 | Run L (convK83, the convolutional head, trained on with run K's recipe) against run K continued, same start and steps; then run M (the input planes) against run L: one change a run | user (Claude's order; the user: clean evaluation) | the long-capture probe; a tool to restructure a network |
| 2026-10-11 | The input planes: 58 at first (56 move planes, direction x distance 1-7 at the moving piece's square, shared by both sides; 2 maps at the affected squares, takeable and danger merged); tests before code; learning traced over many steps; the destination risk re-evaluated after run M | user | *How we evaluate networks*: the input planes, as decided |
| 2026-10-11 | 50 input planes, not 58: the 8 diagonal move planes of 6 and 7 squares left out — a capture of k squares needs 2k squares in a line, the longest diagonals hold 10, so they are zero in every position (checked: the board has room for every kept kind, direction and distance and none of these); a uniform layout brings nothing (a convolution ignores its channels' order) | user | the input planes, as decided |
| 2026-10-12 | Run M (the input planes) ties run L (M102 against L102 +0.65 a pair, −1.63 to +2.93): **the main line is M** — the planes kept in case they help later (longer dwarf lines, long hurls), as the network does use them; promotion test M102 against K83; the standing checks on M102 and L102 | user | the input planes, as decided |
| 2026-10-12 | M102 the top anchor (both tests of the promotion rule: +11.97 against K83 in 40 pairs, +8.97 to +14.98; ladder difference +35.4, +15.8 to +55.1); K83 stays an anchor | user | *How we evaluate networks*; run M |
| 2026-10-13 | The dwarfs' training signal: test runs from M102 — a control and **D1** (a higher kickstarting weight for the dwarfs, 0.7 by check 2's rule, the trolls at 0.3); check 2b first (teacher-guided dwarfs under the trainer's noise: D1 alone, or with per-side noise C3); B1′ or C7 only if D1 leaves the side-to-move gap open. **0.7 is the cap** (the user: "That needs to be enough. If it's not enough then we need to taper … later than anticipated") — at 0.85-1.0 the dwarfs' target would be (nearly) the imitation ruled out on 2026-10-07; check 2 at those mixes is not run | user (Claude recommended, after the checks of the night and the third review) | *the dwarfs' training signal*; `thud/DESIGN_dwarf_signal.md` §8 |
| 2026-10-13 | D1's taper slow and gated: −0.1 at most once per buffer length (~20 steps), only once our dwarfs match the teacher against the same trolls and the regression gates hold ("I'd rather be safe than sorry"); after the third review also only once the search no longer amplifies unsafe dwarf moves (the visits' mass on them no higher than the prior's) | user; the last condition Claude's, from the review | the design, §5 D1 |
| 2026-10-13 | Regression gates added to the promotion rule (test 3): alpha-beta per side, blunders, troll kills, short hurls, the trolls' captures; long hurls monitored; each at 2 standard errors against the top anchor, each with a positive control (the set refined after the design's third review: troll kills added, the trolls measured against alpha-beta 2's dwarfs, standard errors by bootstrap) | user (asked for them and approved the first set; Claude proposed and refined it) | *How we evaluate networks*, promotion rule; the design, §7 |
| 2026-10-14 | The test runs' win rule (the design's §7): the head-to-head's 95% interval above 0 (one-sided at 2.5%, as the promotion rule; 5% considered and declined — a win is rerun anyway, and power is not the problem), the dwarf side against hexparrot and alpha-beta 1 and the trolls against alpha-beta 2's dwarfs not worse than the control's by more than 2 standard errors, not worse than M102 on the dwarf side; settled before the evaluation's results | user | the design, §7 |
| 2026-10-15 | The replication of run N (D1 + C3): a rerun of run N only (run N2, from M102's state, the same config), compared with C121 as run N was — not of the control too: the control proved no weak straw man (C121 beat M102 by +7.20), so one night covers the main risk, a lucky run | user (Claude proposed it instead of the design's rerun of both) | the design, §7 and §10 |
| 2026-10-15 | D1's taper gets a further condition: the value's side-to-move gap closed (check 4) — once the teacher is tapered the dwarfs' policy targets come from the search, which relies on that value | user (Claude proposed) | the design, §5 D1 and §10 |
| 2026-10-15 | Before renting a GPU, every feature it needs built and tested here, every relevant setting exposed (the *GPU readiness checklist* in *Cloud GPU options*; when decided, of its 28 rows 9 done, 2 built but untested, 17 missing — among them Net2Net growth, a learning-rate schedule, a device option for our tools, profiling the batching path) | user | *Cloud GPU options* |
| 2026-10-15 | The hybrid head (convolutional for the trolls, linear for the dwarfs; 2026-10-03) superseded: since runs L and M the convolutional head serves both sides, the dwarfs included (L102 against K102 +10.72; against hexparrot L102 −21.4, K102 −37.1) | user | the head table above; *the long-capture probe* |
| 2026-10-15 | Tests kept complete and run by one command: every feature with a check (the new ones: `grow_test`, `teacher_mix_test`, `trainer_test.sh`, `az_match_test.sh`, the playthrough replay, the sanitized checks), all built without `-DNDEBUG` and run by `thud/run_tests.sh` (quick ~20 s, `--full` ~3 min); git hooks run them — pre-commit the quick ones when a commit changes code, pre-push all when the pushed commits do; the *Tests* list tracks each check, its control, and every gap | user | *Cloud GPU options*, *Tests* |
| 2026-10-15 | After the replication: **a longer D1 + C3 run first** (run N continued from N121, 40 steps to 161, check 4 and the share of self-play games keeping dwarfs every 10 steps), not yet a value test run (B1′, or C7 reversing the rule of 2026-10-07) — the cheapest way to learn whether the value's gap closes as our own dwarfs survive more | user (Claude proposed) | *How we evaluate networks* (the dwarfs' training signal); the design's §10 |
| 2026-10-15 | The GPU checklist's stable features built and checked while run N continues: a learning-rate schedule and a value-loss weight, a watchdog with crash-safe saving, a transfer script, a setup script, the throughput sweep; the batching path profiled (the trainer paused briefly for exclusive measurements, the user's permission) | user | *Cloud GPU options* |
| 2026-10-15 | The legal policy on by default (the priors from the legal moves' logits alone; equal to ~1e-7, every compared search identical); upstream's computation kept behind the switch, for identity_test | user (Claude proposed) | *Cloud GPU options*, the batching path |
| 2026-10-15 | Upstream: the json.h comment fix not opened; the LoadBuffer fix as its own PR after #1637 is merged | user | *Upstreaming*, further candidates |
| 2026-10-15 | One word per concept for what "check" had meant: **tests** (our test programs, files `*_test`, `thud/run_tests.sh`, `build-tests/`), **measurements** (the design's numbered ones: measurement 4 = the value by side to move), **probes** (analysis programs: `az_blunder_probe`, `az_noise_probe`, `az_head_probe`, `az_layout_probe`), **assertions**, **measures** (logged statistics); the session log keeps the old words | user | *Tests*; CLAUDE.md |
| 2026-10-15 | The taper's condition (d): the gap counts as closed when, with the dwarfs to move, a far dwarf is worth at least 80% of its worth with the trolls to move, in two measurements in a row | user (Claude proposed) | the design's §5 |
| 2026-10-15 | The taper's condition (c), stricter: step the dwarfs' teacher weight from w to w′ only if the unsafe-move loop's gain after the step, (1 − w′) × R, is at most 0.8, R (the search's visit share on unsafe dwarf moves over the prior's) at its 95% upper bound; to revisit once in use | user (Claude proposed) | the design's §5 |
| 2026-10-17 | The run programs built by `thud/programs.mk` through `thud/build_programs.sh` (incremental; a second when nothing changed), run first by every queue and by `run_tests.sh --full`; it builds nothing while one of them runs (a stale `az_capture_targets` had aborted night12's probe) | user (Claude proposed) | CLAUDE.md, *Build and test*; *Tests* |
| 2026-10-17 | E1 tested offline first: N161 trained on on run N's last buffer as the trainer learns (its own optimizer, the batch norms in evaluation mode), 640 batches at value-loss weights 1, 4 and 16 and a control at twice the rate, seeds 1 and 2; read by measurement 4 with intervals that resample whole games, paired between networks; the best weight then online against weight 1 (10 steps each); a finer sweep around it only if its neighbours differ beyond the seeds' noise — **stopped the same day** (the user: the batch-norm finding first, kept apart from E1) | user (Claude proposed) | the design's §10, *E1's offline sweep* |
| 2026-10-17 | The batch-norm race first, kept apart from E1 (with proper batch norms the gap may change): fixed in our copy (a lock on the network; each learning step checks its batch norms' counts and stops the program if they are off), no upstream change; N161 converted (its batch norms set to the true statistics of what reaches them, the function unchanged — Net2Net's step — not retrained from scratch: the buffer holds only ~21 steps, and a start other than N161 would confound the comparison); **N′** = the converted N161 trained on 640 batches in training mode, against w1s1 (the same the old way): held-out losses, measurement 4, a match; then 10 self-play steps each, the fix (our trainer, two copies, `explicit_learning`) against the control (`az_trainer_d1`, the race), value-loss weight 1 | user (Claude proposed; the user named N′) | the design's §10; *Upstreaming*, candidate 4 |
| 2026-10-17 | Upstream: the race's fix as a PR with a reproduction on Connect Four, prepared in a separate worktree once tonight's runs are going (a slow PR review says nothing about upstream's upkeep: the user) | user | *Upstreaming*, candidate 4 |
| 2026-10-17 | Upstream: wait for the new maintainers (#1621) a few weeks, then ping them on #1637 and #1646 — around 2026-11-07 if nothing has moved | user | *Upstreaming*, upstream's state |
| 2026-10-18 | The input planes kept: they cost ~0-5% of the speed (search within the rounds' spread, learning +5%) and ~0.1 GB, never hurt (run M tied run L), and may help long hurls and shoves; game-specific inputs speed learning in KataGo (arXiv 1902.10565) | user | *the input planes, as decided* |
| 2026-10-18 | The capacity tests before training on (the user: a network that lost capacity is restarted, cheaply): dead channels, random targets, a reset from the buffer — the reset network beats Np181 by +13.2 (39 of 40 pairs), and fits random targets as fresh networks do where Np181 hardly does | user | `thud/DESIGN_dwarf_signal.md` §10, *The capacity tests* |
| 2026-10-18 | The capacity tests (dead channels, plasticity) monitored with every new best network: `regression_gates.py --dormant --plasticity` reports them, never failing; each network measured gets a row in the capacity record (`DESIGN_dwarf_signal.md` §7) | user | `DESIGN_dwarf_signal.md` §7 |
| 2026-10-19 | WSL's memory stays as it is (no `memory=20GB`); a larger buffer, if any, through compact storage (a position took 9.4 KB: 1,350 observation floats and 24 bytes per legal move — **built 2026-10-19**, `thud/az/position_buffer.h`: R201's buffer 4.47 GB → 0.66 GB, 6.8x; training on it bit for bit the same) and buffer snapshots on disk for resets (one per ~20 steps, merged with `az_merge_buffers`; the first: `~/thud-runs/runR_snap201/`); not all games ever played: old games' targets come from weaker searches — KataGo grows its window sublinearly (250,000 positions at first, ~22 million at the end of its main run; arXiv 1902.10565), ours is a fixed 458,752 | user | *Changes to the search and trainer*, the buffer |
| 2026-10-19 | The no-capture limit 100 in self-play only (`--game=thud(max_turns_without_capture=100)`; every evaluation keeps 200, so results stay comparable): in run R's games 94% ended at the 200 limit, 37% of the positions in that final stall, 3.6% of the captures after more than 100 quiet turns (80% the trolls'). The indicators to revisit it: late captures per battle in evaluation games (our trolls 0.26-0.31, hexparrot's 0.01), the trolls' margins, and in self-play the share of games ending at the limit | user | the design's §10 |
| 2026-10-19 | One change at a time from R201: RL (the limit 100) and RLT (the limit 100, the teacher 0.35 on the dwarfs' positions), 10 steps each, against R's own steps 202-211; a round robin and hexparrot after; R pauses after step 211 until the results | user | status block, `~/thud-runs/limit_teacher_2026-10-19.sh` |
| 2026-10-19 | The window: a cap of 6x (2,752,512 positions, ~4.1 GB compact — today's buffer's memory) with KataGo's rule (`--replay_buffer_start_size=250000`: ~2.3x now, 3x at 5 million positions generated, ~6x at 14 million, ~10 days at our pace), `replay_buffer_reuse` 126 (a step every 21,845 new positions); tested first offline (`~/thud-runs/forward_window_2026-10-20.sh`: R201 trained on today's window and on KataGo's, scored on the next 10 steps' positions) | user | the buffer |
| 2026-10-18 | **The reset network is the main line, run R** (R181: the reset at the run's step 181), continued in self-play from N′'s run's state at step 181, run N's recipe unchanged; started after 02:00 once the laptop has been idle 20 minutes; each segment adds the plasticity test, to see whether R keeps learning | user | status block, `~/thud-runs/r_run_2026-10-19.sh` |

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
  board correctly.** `az_layout_probe.py`'s supervised task, trained with upstream's C++
  model, unmodified (`VPNetModel::Learn`), on exactly the Python runs' data and settings:
  `az_layout_probe.py --export` writes the random games (0-399 for training,
  10,000-10,099 for testing), checksums of the positions taken from them (every 7th turn),
  and each seed's batch order; the new `az_layout_probe.cc` replays the games with Thud's
  C++ code, stops unless its 20,036 + 4,816 positions reproduce the checksums (a
  tampered count and a tampered plane sum were both caught), and trains on the same
  batches in the same order — resnet 32 x 2, learning rate 1e-3, weight decay 1e-4,
  1,500 steps of 128, seeds 1-3. Built as OpenSpiel's `docs/library.md` describes, with no
  CMake change: `build-shared/libopen_spiel.so` plus upstream's `model.cc` and `vpnet.cc`
  compiled alongside (`thud/experiments/build_az_program.sh az_layout_probe`). On the
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
    wall-clock ones do not — a first trainer test lost ~39 and ~9 minutes to sleep, and
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
  gain, as expected. (Engine identity: `identity_test.cc` still passes with the
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
  quick measurement: forgetting in the dwarfs' move probabilities): (a) **forgetting** — its
  dwarfs play well against the trolls they train against and lost what beat other trolls
  (the buffer holds ~3 learning steps; if so, the buffer's "newer network losing"
  trigger — pronounced here, not yet shown persistent); (b) **the evaluation** —
  upstream's rule spreads a side's search when the network judges that side losing.
  **Checked 2026-09-29** (`untried_move_test`, 3 self-play games per network with
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
  head, or values that no longer separate dwarf moves). The decisive measurement is the match
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
  curve against the anchor becomes a health measurement at the end of every run. **Step 2,
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
  (2026-09-30, user: the quick measurement first, then option C if warranted): our copy trains
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

  **The quick forgetting measurement** (2026-09-30, `thud/experiments/az_forgetting.cc`, 3.5
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
  forgetting measurement of D's networks on C's archives. If D's dwarfs keep their repertoire
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
    measurement of D16-D29 on C's archives, same positions): D29's dwarf move-probability loss
    on archives 3, 18 and 21 is 5.74, 4.89 and 4.48, C16's level (5.73, 4.90, 4.58), where
    C29 had 8.03, 6.57 and 5.63; D's dwarf policy is also less peaked (its likeliest move
    is the most visited in 65% of archive 30's dwarf positions, C29's 88%). The trolls'
    and the values' numbers are much as in C.
  - **D is the strongest network of the C/D family**: D29 beats C29 by +7.3 and C16 by
    +8.4 (C29 beat C16 by +4.1); its dwarfs score +7.5 against C16's trolls.
  - **Against fixed opponents (other runs' networks) D is better than C but still declines**: against
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
  **How many CPU cores a GPU needs** — `thud/experiments/az_search_cost.cc` (new,
  2026-10-04): our search's CPU cost per simulation with a stand-in network that
  costs nothing (per side, one core, then threads), and the rate through the
  trainer's batching evaluator with a nearly free network (an MLP of width 8, the
  trainer's cache). Smoke numbers under load (run W running; not results): ~14 µs a
  simulation for dwarf positions, ~6 µs for troll positions, on one core; batched only
  ~5,500 simulations a second with 32 threads — if that holds on an idle machine, the
  batching path, not the search, caps throughput, and a GPU would not help past it.
  **Result** (2026-10-05 13:02-13:20, idle machine, `~/thud-runs/search_cost.jsonl`;
  a first run of 200 positions took 8 seconds, too short, and was redone with 2,000):
  - **The search itself is cheap**: 3.8 µs a simulation for dwarf positions, 2.7 µs
    for troll positions, on one core (~306,000 a second); 1.2 million with 4 threads,
    1.9-2.3 million with 8 — near linear.
  - **Through the trainer's batching evaluator, with a network that costs almost
    nothing (an MLP of width 8), it caps**: with 2 inference threads (the trainer's
    setting) 4,400 / 12,400 / 24,400 / 24,800 / 24,800 simulations a second at 8 / 32 /
    64 / 128 / 256 search threads (batches of 32; of 128 it climbs more slowly to the
    same 25,600); **with 4 inference threads ~38,600**, with 8 no more (36,300); more
    cache shards (8, 32) change nothing (~37,000).
  - **It is not the cores**: at 256 search threads and 4 inference threads the program
    used 29.9 CPU-seconds in 6.9 s of wall time (user 23.7 s, sys 6.2 s) — ~4-5 of the 10
    cores; the kernel time suggests threads waking and sleeping on each other. Each
    simulation costs ~140 µs of CPU through the batching path against 3.3 µs for the
    search: **the evaluator's overhead is ~40x the search**. (`DeviceManager` does not
    lock the model; the cause is somewhere in the evaluator's queue, futures and the
    per-batch tensor work — 19,800 actions a position — to profile before scaling.)
  - **What it means for the GPU**: our trainer self-plays at ~1,700 simulations a
    second here (64 x 4), so a GPU can speed self-play up by up to ~20x before this
    CPU-side cap of ~25,000-40,000 a second binds, whatever the GPU — worthwhile, but
    capped. **Cores per GPU**: ~8-16 suffice for that cap (the cap, not the cores,
    limits); beyond it, more cores do not help — profiling and fixing the batching path
    does (fewer wake-ups per simulation, cheaper output handling). On the first GPU
    machine: measure the same cap there (`az_search_cost`), then the real network.
  `az_search_cost` takes `batched_threads=`, `inference_threads=`, `cache_shards=`.
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

  **The settings to revisit on a new machine, by kind** (agreed with the user,
  2026-10-05):
  1. *Machine settings* — speed only, not what is learnt; re-measured on every machine
     by a short benchmark, no training (`az_search_cost parts=2 net=... batched_threads=...
     batch=... inference_threads=... precision=float32,mixed,bfloat16`, OMP_NUM_THREADS
     varied): **the inference precision** (since 2026-10-19 — on a GPU's tensor cores
     float16 or bfloat16 run several times faster than float32, and the agreement test,
     `az_precision_probe`, and a match against float32 decide whether the lower
     precision plays the same; this CPU: bfloat16 1.56x float32 at each one's best
     setting (4,384 against 2,815 simulations/s), float16 has no fast kernel); **tuned
     apart for each use** (the user, 2026-10-19) — self-play inside the trainer (the
     learner shares the cores, and alpha-beta as opponent and teacher runs on the actors'
     threads: measured by the trainer's own step time, as `arms3` benchmarks two
     settings), matches (`az_match`: threads as pairs, batch, OMP_NUM_THREADS, how many
     matches at once, precision for both networks alike; opponents without a network —
     hexparrot, alpha-beta, MCTS — change the balance; measured by its simulations/s,
     `progress=`, or a match's wall time), and the probes and offline tools (`az_restructure`,
     `az_dwarf_value_probe`, `az_plasticity`: OMP_NUM_THREADS and their batches); actors
     (games searched at once: ~32 here, a GPU wants hundreds to fill its
     batches), inference threads and the inference batch size together — **keep at least
     batch x inference threads actors waiting**, else the threads take turns (2026-10-19:
     32 actors with batches of 32 and 2 threads ran at 1,512 simulations/s, with batches
     of 16 at 2,828), and pick the batch where the kernel is efficient (bfloat16 here
     needs 32 or more); `OMP_NUM_THREADS`;
     the inference cache's size and shards; network copies (devices — several on one
     GPU may get past the batching cap); match threads and batch; memory and disk (a
     7x buffer ~4.4 GB of RAM plus its copy written each step, ~26 s here; checkpoints
     ~110 MB a step at 64 x 4); and the CPU cores beside the GPU, which alpha-beta
     needs — as training opponent and as kickstarting's teacher it runs on the actors'
     CPU threads, ~21 ms a position at depth 2 here, ~4% of a self-play move: where the
     network gets much faster, that share grows, and alpha-beta may set the pace
     (added 2026-10-08).
  2. *Compute-budget settings* — what becomes affordable; they change learning, and
     our CPU answers were shaped by scarce compute: simulations a move (100 won here at
     equal time early in training — try 200-800); playout caps (3b); the network's size
     (grow when learning flattens); batches per learning step and the training batch
     size; the learning rate (tied to the batch size, never tuned); the buffer and the
     cadence (defined per position, so they carry over, but a faster machine fills the
     buffer in fewer hours); simulations per side (`--player_simulations`, built
     2026-10-06, not yet used); alpha-beta's depth as opponent and as teacher
     (`--alphabeta_depth`, `--kickstart_depth`: depth 3 costs ~180x depth 2 for the dwarfs
     (3.8 s a position) and ~23x for the trolls (0.5 s) with OpenSpiel's alpha-beta —
     more cores, or an alpha-beta of our own, make it affordable; added 2026-10-08).
  3. *Algorithm settings* — machine-independent but never tuned; on the GPU with
     several runs per setting: `uct_c` (2), the untried-move reduction (0.2), root noise
     (α 0.1, perhaps per side; ε 0.25), the temperature and its drop (10 moves); the
     policy temperature per side (`--player_policy_temperature` 1.5 for the dwarfs
     since run H, which beat G' continued from the same start — one run each — and
     never compared with other values; added 2026-10-08); augmentation stays on.
  4. *Evaluation settings* — pairs per match (40), simulations in matches (100), the
     ladder's anchors: independent of the machine, only the cost per pair changes.
  5. *Scaffold settings* (added 2026-10-08) — teachers meant to fade, not to stay:
     training against alpha-beta (`--alphabeta_share` 0.3; its taper by results, per
     side, to a ~10% floor, not yet built) and kickstarting (`--kickstart_weight` 0.3,
     `--kickstart_temperature` 0.25; its decay to zero, not yet built). Machine-
     independent; what to tune is less their values than their schedule — how long
     they stay and how fast they fade — judged per side by the ladder and by matches
     against players without a network, and only if they are still in the recipe by
     then (on a faster machine, a run from scratch may need them less, or for fewer
     steps).
  **Order on a new machine**: (1) set up and check correctness — the build, our test
  programs, `identity_test`, a short tic-tac-toe control; (2) benchmark group 1, ~1
  hour — `az_search_cost`, then actors x inference threads x batch with the real
  network; (3) **a baseline repeat**: G''s settings to step ~16, which should learn
  per step as on this CPU — the positive control for the move; (4) then groups 2 and 3,
  one change at a time, 2-3 runs per setting, and group 5's schedules if the scaffolds
  are still used. **Rule**: what is defined per step (the
  cadence, the buffer in positions, equal-steps comparisons) carries over between
  machines; what is defined per hour (time limits, equal-time comparisons) does not.

  **GPU readiness checklist** (the user, 2026-10-15: time on a rented GPU is expensive
  even when used badly, so every feature it needs is to be ready and tested here first,
  and every relevant setting exposed). State: **done** = built and tested (a test
  program, or real runs); **built, untested** = built but never run where it matters;
  **missing** = not built. *Before renting* = to do here first.

  | Feature | Why on the GPU | State | To do |
  |---|---|---|---|
  | *Setup* | | | |
  | A build on the rented machine: LibTorch with CUDA (the pip wheel, as here), `build-torch/`, `build-shared/`, our programs | nothing runs without it | **built, partly tested** (2026-10-15): `thud/setup_machine.sh` — install.sh's packages and dependencies, the venv with torch 2.10.0 (on x86_64 PyPI's CUDA wheel), `~/.bashrc`, `build/` and `build-shared/`, our programs, hexparrot at its commit, every test, and with a GPU a CUDA smoke test; each step skipped when done; `build.sh`, `tests.mk` and `programs.mk` link LibTorch's CUDA libraries when the wheel has them (without, a CUDA device fails: "no CUDA backend"); our programs since 2026-10-17 by `thud/build_programs.sh` (`programs.mk`, incremental). The dry run on a fresh clone here passed every step (2026-10-15, `~/setup-test`); never on x86 or with CUDA | *before renting*: on a cheap x86 box if one is at hand |
  | The tests on the new machine (`thud/run_tests.sh --full`: every test, *Tests* below; the tic-tac-toe control) | correctness before any spending | **done** (here, 2026-10-15: 28 tests, ~4 min, none needs a run folder — the networks they read are made by `make_test_network`) | run them there first (the order above) |
  | The trainer on a GPU (`--devices=cuda:0`; several copies, `cuda:0,cuda:0`) | the network on the GPU | **built, untested on a GPU** (upstream's device handling; no GPU here); the setup script's step 8 runs it there: 2 steps on `cuda:0` and on `cuda:0,cuda:0`, the learner on `cuda:0` | on the machine: step 8 (`--gpu-smoke-only`) |
  | Our tools on a GPU: `az_match devices=LIST` (one copy of each network per entry, e.g. `cuda:0,cuda:0`) | matches at GPU speed | **built, untested on a GPU** (2026-10-15): the default reproduces the old binary exactly (pairs and request counts, one thread); two CPU copies play identical games (since 2026-10-15 a permanent test, `az_match_test.sh` 4, with the control that the summary records the copies); `cuda:0` here reaches Torch and fails as this CPU-only LibTorch must ("no CUDA backend") | on the machine: one short match |
  | — `az_restructure device=` | distillation at GPU speed | **built, untested on a GPU** (2026-10-15): the default writes byte-identical checkpoints (the input-plane surgery on convK83) | on the machine |
  | — `az_grow device=` | growth on the GPU | **built, untested on a GPU** | on the machine |
  | — the probes (`az_dwarf_value_probe`, `az_blunder_probe`, `az_risk_probe`, `az_long_capture_probe`, `az_input_plane_trace`, `az_feature_probe`, …) | — | **missing**: they hard-code `/cpu:0` — cheap enough on the CPU | only if one proves slow |
  | *Throughput* | | | |
  | `az_search_cost`: the search's CPU cost, the batching cap | how many cores a GPU needs | **done** (here: the search ~3.3 µs a simulation; with a tiny network the batching path capped at ~25,000-40,000 simulations/s, ~52,000 with the legal policy) | rerun there |
  | The batching path's overhead | possibly the GPU phase's limit (an extrapolation: below) | **profiled and one cause fixed** (2026-10-15, `perf`, the trainer paused 38 s): with the tiny network of `az_search_cost` (nearly free to evaluate), half the inference threads' time went to the policy over all 19,800 distinct actions — the mask and the softmax — not to queues or locks (< 2%), the searching threads mostly waiting on the 2 inference threads (~25% of the cores busy). **The legal policy** (`VPNetModel::SetLegalPolicy`, on by default): the logits of the legal moves alone, normalised in double — the priors within ~1e-7, the values bit for bit, all 120 compared searches identical (`legal_policy_test`); measured (the trainer paused 62 s): the tiny network 2.4x (Thud) and 1.5x (chess), **our real network on this CPU +1% (M102), chess at 64 x 4 ±0**: with real networks here the network's own forward passes dominate — the bottleneck on this CPU, as before. Whether the bookkeeping limits a GPU is unmeasured | on the machine: the sweep with the legal policy on and off (`legal_policy=`) |
  | A throughput sweep (actors x inference threads x batch with the real network) | the machine settings, group 1 | **built** (2026-10-15): `az_search_cost parts=2 net=RUN:STEP devices=... batched_threads=LIST batch=LIST inference_threads=LIST` — a line per setting, then the fastest; tested here on a test network (the format, the grid) | on the machine |
  | *Training: settings exposed* | | | |
  | The trainer's settings: `uct_c`, the untried-move rule and reduction, simulations (per player), policy temperatures (per player), root noise α and ε (per player), temperature and its drop, the learning rate, weight decay, batch size, batches per step, the buffer (size, reuse, growing start), symmetry augmentation, policy target pruning, alpha-beta's share and depth (per player), kickstarting's weight (per player), depth and temperature, network size and model, actors, inference threads, batch and cache | the tuning groups 1-5 | **done** — each with a test (`untried_move_test`, `conv_policy_test`, `augmentation_test`, `pruning_test`, `growing_buffer_test`, `search_settings_test`, `alphabeta_training_test`, `kickstart_test`), used in runs, and since 2026-10-15 all on at once end to end (`trainer_test.sh`) | — |
  | A learning-rate schedule (warm-up, decay) | long runs, bigger batches | **done** (2026-10-15): `--lr_warmup_steps`, `--lr_warmup_start` (a linear warm-up), `--lr_schedule=STEP:FACTOR,...` (AlphaZero's and KataGo's step-wise factors), on the network's own rate (vpnet.pb), the rate logged per step (`learning_rate` in learner.jsonl); `learning_settings_test` (the factors by hand, the config's round trip and old configs' defaults, the rate set, a step at rate 0 moving no parameter) and `trainer_test.sh` (the rates of 3 steps); found on the way: a checkpoint restores its optimizer's rate (so without a schedule a run trains at its checkpoint's rate), and `az_grow`'s default rate now the source's vpnet.pb's | — |
  | A value-loss weight (E1 of the design) | the value learns small effects poorly | **done** (2026-10-15): `--value_loss_weight` (the losses reported unweighted); `learning_settings_test`: at weight 0 the value head's own parameters exactly unchanged, at 1 they move, weight 2 moves the trunk differently | — |
  | Playout cap randomisation (3b) | recommended for the GPU | **done** (2026-10-15): `--playout_cap_full_share` (each of the network's moves searched in full and recorded with that probability) and `--playout_cap_fast_simulations` (the rest: no root noise, no forced playouts, not recorded; KataGo, arXiv 1902.10565, 3.1); `playout_cap_test` (off: the old games exactly; on at a quarter: 25.4% of 6,150 moves recorded, every one a full search; the fast bots without noise) and `trainer_test.sh` (half in full) | the setting to use: by test runs on the GPU (stage 3b) |
  | Tapers by results (kickstarting per side, the alpha-beta share), gated | long unattended runs | **missing** — by hand, run to run. Its parts (the design's §5, conditions a-d): (a) the noise-free matches and (b) the regression gates are built; (c) measured by `az_unsafe_targets` (built 2026-10-15, `unsafe_targets_test`; N131: R = 0.70, upper end 0.85; a stricter rule decided 2026-10-15: the loop's gain after the step at most 0.8), and (d) is decided (the user, 2026-10-15: closed when a far dwarf with the dwarfs to move is worth at least 80% of its worth with the trolls to move, twice in a row; measurement 4 runs every 10 steps in the longer run) | after (c)'s probe: a supervisor between segments, as `night11` does for measurement 4 (moderate) |
  | The alpha-beta share per side; kickstarting's temperature per side | finer scaffolds | **missing** | if needed (small) |
  | A faster alpha-beta of our own (captures first, a table of positions, iterative deepening) | a depth-3 teacher; alpha-beta then sets the pace on the CPU beside the GPU | **missing** | before deeper teachers |
  | The value-target options B1′, C7, E2 (the design) | if D1 leaves the side-to-move gap open | **missing** — designed only | when needed |
  | Mixed precision | speed on large networks | **missing**; measured 2026-10-19 on this CPU (it has FP16, BF16 and int8 instructions: `fphp asimdhp bf16 i8mm`): LibTorch 2.10 runs a 64-channel 3x3 convolution on a batch of 32 at 7.7 ms in float32, 5.6 ms in bfloat16 (1.4x; relative error 3e-3), 112 ms in float16 (no fast kernel here); a network of our shape (PyTorch, batches of 32, 4 threads): float32 19.5 ms, bfloat16 14.5 ms (1.34x), int8 after static quantization 17.7 ms with qnnpack (1.10x), 115 ms with onednn; **the network is ~90% of a self-play step** (`az_search_cost`: 15,578 simulations/s behind a nearly free network, 1,526 with R201), so bfloat16 inference would make self-play ~1.3x faster here; **built** (2026-10-19: `--inference_precision=float32|mixed|bfloat16`, an inference copy of the network refreshed after every weight change; `precision_test`); the agreement on R201 (`az_precision_probe`, 600 positions of its games, 100-simulation searches without noise): mixed — values within 0.0016 on average (0.0058 at most), the top prior move the same in 98.7%, the searches' chosen move in 94.7% (bfloat16: 0.0023, 0.0073, 97.8%, 93.5%), where it differs a near tie in float32's own search (the visit gap's median 1%, 90th percentile 4%; mixed's value gap 0.006 and 0.027), against 99.5% for float32 searched again with another seed (the control) | **revisit on the GPU** (the user, 2026-10-19: tensor cores give FP16/BF16 their usual large gain there, and int8 via TensorRT); here bfloat16 inference behind a flag, tested by the agreement on buffer positions (values, top moves, the search's chosen move) and a match, after the limit and teacher arms; training stays in float32 |
  | The batching path's profile | the trainer's speed | **done** (2026-10-19, `az_search_cost` on a quiet machine): the network ~90% of self-play; the sweep (32/64/96 actors x batches of 16/32/64 x inference threads x OMP 2 or 4 x precision): today's setting (float32, 32 actors, batches of 32, 2 threads x 4) 1,495 simulations/s; the best float32 2,815 (32 actors, batches of 16, 2 x 4), mixed 3,604 and bfloat16 4,384 (both 64 actors, batches of 16, 4 inference threads x 2) — 2.9x | the arms with the chosen precision and setting, checked in the trainer's own states/s |
  | *Network growth* | | | |
  | Net2Net widening (more filters), function-preserving: `thud/az/az_grow.cc` (2026-10-15) | grow the network without starting over | **done** — every new trunk channel copies a random old one, the same choice in every layer (the input block, every residual block, the policy head's first convolution and batch norm, the value head's hidden layer), the weights reading them divided by the copies; tested on real networks: M102 (convolutional head, input planes) 64 → 96 and 64 → 128, K83 (linear head) 64 → 96, the outputs equal within 7e-7 (values) and 5e-6 (policies) on 200 positions of random games; **the control** (the same growth without the division) differs by 0.7-1.9, as it must; the saved network reloads with identical outputs. **Unit tests** (`grow_test`, 2026-10-15; the growth in `thud/az/net2net.h`): on small networks with random batch norms, the mapping's properties, the function kept (1e-7) for wider, deeper, both, the linear head, no growth; three controls (no division; a batch norm's statistics left uncopied; the new blocks not the identity), every tensor accounted for | — |
  | — symmetry breaking (`noise=`) | copies would stay copies under training | **done** — noise of that many times each tensor's standard deviation on the new channels' incoming weights; at 0.01 the outputs move by ≤ 0.005 (values) and ≤ 0.035 (policies); `grow_test` 5: without noise the copies stay equal through 30 learning steps, with 0.01 they part | the value to use: by the live test |
  | — the value head's hidden layer grown apart from the trunk; other models (`mlp`) | — | **missing** (the hidden layer follows the trunk's width; `az_grow` refuses other models) | if ever needed |
  | Net2Net deepening (extra residual blocks starting as the identity) | the same, in depth | **done** (`az_grow nn_depth=`, 2026-10-15): the new blocks after the old ones, their last batch norm's scale and shift at 0 (Goyal et al. 2017); the outputs equal the source's exactly (0) for M102 4 → 6 blocks, and with widening (128 x 6; K83 96 x 5) as above | — |
  | — a live test: training on from a grown network | that it learns, not just computes | **partly**: in miniature (`trainer_test.sh` 3, 2026-10-15: the trainer takes a grown 12 x 2 network, its buffer and log, and learns a step with finite losses); the real test still **missing** | a trainer segment from a grown M102 (e.g. 96 x 6, `noise=0.01`) in a gap between runs: no loss spike, the losses falling below the source's |
  | — the access it needs: `VPNetModel::NamedTensors()` (every parameter and buffer by name, sharing storage) | — | **done** (our copy, `thud/az/vpnet.h`) | — |
  | Distillation into a new structure from a buffer (`az_restructure nn_width= nn_depth= nn_model= freeze=none`) | a bigger network warm-started from a buffer | **built for a new head only** — tested for a new head (convK83; since 2026-10-15 `trainer_test.sh` 4: a new head on the frozen trunk) and new input planes; **into another width or depth it stops by design** (found 2026-10-15): its copy test requires the copied trunk to give the source's values, and at a new size almost nothing is copied | only if distillation is wanted beside `az_grow`: the test limited to what was copied (small) |
  | Input-plane surgery (`az_restructure input_planes=1 copy=all`) | new inputs | **done** (`input_planes_test`; run M) | — |
  | A fresh optimizer, a new learning rate (`az_fresh_optimizer`) | after surgery or growth | **done** | — |
  | *Evaluation and monitoring* | | | |
  | `az_match` (networks, alpha-beta, MCTS, hexparrot; self-play's conditions; the teacher in a prior) | matches | **done** | the device option (above) |
  | The ladder (`ladder.py`), the watcher (`watch_run.py`), the probes, the evaluation queues (`night9`-style) | evaluation | **done** | — |
  | The regression gates (standard errors, positive controls) | safe promotions and tapers | **built** (2026-10-15): `thud/experiments/regression_gates.py` — the design's seven measures, paired with the anchor on the same openings or positions (alpha-beta per side, blunders with a battle-clustered bootstrap, troll kills, short hurls, the trolls' captures; long hurls monitored); its blunder count equals `az_blunder_probe`'s on 4 networks; on real data N121 passes against M102, the positive controls fail (K83 short hurls, G'64 blunders, G'64 against I83 the trolls' captures); `regression_gates_test.py` on made inputs (itself: all pass at 0; each control fails its gate alone). The measurements it reads are matches and probes run by hand or a queue | a queue that measures a network (the matches and probes) and runs the gates |
  | *Robustness and logistics* | | | |
  | Resuming a run after an interruption | preemption, crashes | **done** (run C; run N after a Windows restart, 2026-10-14; since 2026-10-15 `trainer_test.sh` 2: the step follows on, the logs appended) | — |
  | Restarting a run by itself after a crash or a preemption (a watchdog) | spot instances | **done** (2026-10-15): `thud/az/watchdog.sh` (restarts on a crash or kill, resumes from config.json; gives up after 5 restarts without progress; a stop asked of it passed on as SIGINT), and crash-safe saving in our copy: the buffer and the checkpoints written under a temporary name and renamed; the resume skipping a truncated last record of learner.jsonl and ending the logs' unfinished lines; our `LoadBuffer` refusing a cut-off file (upstream's loads it without an error, half the positions lost: an upstream bug, *Upstreaming*). `watchdog_test.sh`: kills during a buffer write and a checkpoint write, a truncated record — the run finishes with every step once; the control: a half-written buffer refused; the stop; giving up | after a machine restart: start the watchdog again (cron `@reboot`) |
  | Moving runs to and from the machine (buffers ~4.5 GB, checkpoints, logs) | start from our networks, bring results home | **built, tested locally** (2026-10-15): `thud/transfer_run.sh FROM TO [--state-only]` (rsync, resumable; every file's SHA-256 compared; the copy's config.json path set to its folder; a running run refused); `transfer_test.sh` between local folders: the copy resumes into itself, the original untouched, `--state-only`, an interrupted copy completed. Over ssh untested (no second machine) | the first real transfer: a state-only push of a run |
  | A time and cost budget per run (timeouts) | money | **done** (the runners' timeouts) | — |

  **Tests** (the user, 2026-10-15: unit tests for the growth, "pretty complete" tests of
  the other features, commit hooks for them, and "carefully keep track of what's
  implemented and what's tested and what's missing, even if small subfeatures are
  missing"). `thud/run_tests.sh` builds what is out of date (`thud/tests.mk`, into
  `build-tests/`, **without `-DNDEBUG`**) and runs the quick tests (~20 s, four at once)
  or with `--full` all 33 (150 s on an idle machine, 2026-10-17) and then brings the
  programs our runs use up to date (`thud/build_programs.sh`, `thud/programs.mk`, into
  `build-shared/`: a stale `az_capture_targets` had aborted a night's probe; while one of
  them runs it builds nothing and says so); none needs a run folder — the networks the tests read
  are small untrained ones from `thud/az/make_test_network.cc`. **The hooks**:
  `.githooks/pre-commit` runs the quick tests when a commit changes code (the game, `thud/az/`,
  `thud/experiments/`, the tests, the hooks; docs alone go through), `.githooks/pre-push`
  all of them when the pushed commits change code, after its upstream guard (both tested
  2026-10-15 on copies with the tests replaced by a marker: docs-only and deletions skip,
  code runs, a failing test stops the push, the guard still refuses `thud/` upstream).
  Beside a running experiment the tests run one at a time in the idle class. Every
  test, with what it covers (and its control, the evidence that it can fail):

  | Test | Set, time | What it covers | Its control |
  |---|---|---|---|
  | `thud_test` (`open_spiel/games/thud/thud_test.cc`) | quick, 2 s | the rules: every move kind, captures, endings, the encoding, text formats, symmetries; OpenSpiel's generic tests | hand-worked expectations |
  | `crosscheck_tests.py` | quick, 0.2 s | `thud_test`'s expectations recomputed by hexparrot's engine (needs hexparrot set up) | 2 known differences listed apart |
  | `playthrough` (in `run_tests.sh`) | full, 3.5 s | the baseline's recorded moves replayed, every line compared (as upstream's `playthrough_test.py`) | a copy with one board line changed must not replay equal |
  | `features_test` | quick, 3 s | the 50 input planes: hand-built positions, the planes against the move generator | — (exact expectations) |
  | `input_planes_test` | full, 65 s | the planes inside the network: net surgery keeps the outputs, each plane reaches exactly its positions, learning reaches exactly the planes present | judged against the next plane's positions it must fail |
  | `conv_policy_test` | full, ~6 s (247 s → 8 s beside a trainer, 2026-10-15: the learning part stops once the loss halves, in 4-5 steps of the 150 allowed) | the convolutional policy head: each logit at its own plane and square, the masking, learning, checkpoints | rows and columns swapped must fail; a fresh network's outputs must differ from a reloaded one's |
  | `augmentation_test` | quick, 0.2 s | the 8 symmetries of observations, moves and policy targets | positions mirrored independently, as text |
  | `identity_test` | full, 88 s | our AlphaZero copy computes as upstream's, bit for bit: outputs (ours with the legal policy off, `SetLegalPolicy(false)`), a learning step, searches with and without noise, rollouts | another network, seed, step must differ |
  | `legal_policy_test` (new) | full, 44 s | the legal policy, our default, against upstream's computation, for the three models, batches of 1 and 32: the values bit for bit, the priors within 1e-6 (measured ~1e-7) summing to 1; reported: searches with it on and off (all 120 identical) | another network's priors must differ |
  | `batch_norm_test` (new, 2026-10-17) | quick, 3 s | the race's fix (`vpnet.h`): beside a thread running Inference without pause our network learns exactly as alone and each batch norm counts one batch a step; `BatchNormCountsDiffer`; evaluation mode after learning; `ConvertBatchNorms` on stale statistics (outputs unchanged, training mode then like evaluation mode, counts and momenta kept); `LossInTrainingMode` changes nothing; since 2026-10-18 `ConvertBatchNorms` without compensation (the dormant probe's fresh network: evaluation mode then computes training mode's losses, to 1.4e-6) | upstream's `VPNetModel` beside the thread must depart (it does: by 1.01); the stale network's training-mode losses must differ (by 0.19); the fresh network's, before, by at least 0.01 (0.08) |
  | `learning_settings_test` (new) | quick, 0.8 s | the learning-rate schedule's factors, the config's round trip and old configs' defaults, `SetLearningRate`, the rate a checkpoint restores, the value-loss weight, the saves under temporary names; since 2026-10-17 `Gradients` (each loss's own gradient over each head, the L2 term's against its definition to 1e-5, unweighted, no step, an empty set all 0) and `SetBatchNormsInEvaluation` (a step at rate 0 moves no output, its losses equal `Loss()`'s, the scales still learn) | a step at the network's rate must move the parameters; weight 1 must move the value head; a step at rate 0 in training mode must move the outputs; training mode's losses must differ from `Loss()`'s |
  | `watchdog_test.sh` (new) | full, 90 s | the watchdog and crash-safe saving: kills during a buffer write and a checkpoint write, a truncated record — the run finishes, every step once; the stop; giving up | a half-written buffer must be refused |
  | `transfer_test.sh` (new) | full, 24 s | `transfer_run.sh` between local folders: checksums, the copy's path, the copy resumes into itself, the original untouched, `--state-only`, a running run refused, an interrupted copy completed | the original's config.json must still point at the original |
  | `playout_cap_test` (new) | quick, 1.3 s | playout cap randomisation: off plays the old games; on records about the share, every recorded position a full search; the fast bots | off, every move must be recorded |
  | `precision_test` (new, 2026-10-19) | quick, 1.4 s | the inference precision (`VPNetModel::SetInferencePrecision`) on built networks (random weights and batch norms, a peaked prior): float32 set explicitly bit for bit the default; mixed and bfloat16 within 0.01 of float32 (found 0.0012/0.0005 and 0.0017/0.0038), either policy computation alike; the copy follows learning and LoadCheckpoint bit for bit; learning stays float32 (the same losses and weights, tensor by tensor) | neither lower precision bit for bit float32, mixed and bfloat16 apart, another network 0.27 away (the tolerance tells networks apart); the copy moved by learning. A test-side fix: comparing saved checkpoint files failed — LibTorch writes a file's name into it, so files of different names differ; the weights are compared tensor by tensor |
  | `position_buffer_test` (new, 2026-10-19) | quick, 0.4 s | the compact buffer (`position_buffer.h`): the round trip (every field exact, the policy's probabilities as the learner's floats) for 1-300 distinct observation values, -0.0, aligned, subset and empty policies; 2,000 positions of random games exact at 6.8x fewer bytes; the same contents and samples as upstream's `CircularBuffer`; its file, upstream's file and `SaveLegacyBuffer`'s each read back the same; growth (a buffer of 1,000 into 2,500: oldest first, then the last 2,500 added); a file cut short refused; also under the sanitizers | a changed observation entry and a changed index byte must show; a file of other positions loads unequal; a file of 2,500 into 1,000 refused (the file of 1,000 loads); the whole file loads |
  | `regression_gates_test.py` (new) | quick, 0.9 s | the gates' arithmetic on made inputs: a network against itself; the blunder rule by hand; positions that do not match skipped; since 2026-10-18 the capacity tests' two monitored rows (dead channels by trunk and heads, the random targets' fit, beside the anchor's and a fresh network's); since 2026-10-19 the trunk's gradients (the weighted ratio and the cosine from the report before the first batch), the late captures (against hexparrot, after 100+ quiet turns; the count equals one by hand) and the inference precision (az_precision_probe's agreement, the near ties, the control) | each made regression must fail its gate alone; lower long hurls only monitored; a network with every channel dead and no fit reported, never failed; opposed gradients (cosine −1) reported, never failed; battles cut short give fewer late captures, reported, never failed; searches that mostly differ in lower precision, reported, never failed |
  | `dwarf_value_report_test.py` (new, 2026-10-17) | quick, 0.8 s | measurement 4's intervals (`dwarf_value_report.py`): exact values, the 95% intervals' coverage (37 of 40), paired differences, reading the probe's records | resampling positions one by one must narrow the interval when each position is repeated in its game (it does, to 0.51; by games 1.00); a copy of a network must differ by exactly 0 |
  | `ladder_test.py` (new) | quick, 0.7 s | the ladder: known strengths recovered exactly (both links, a chain of matches), the 95% intervals' coverage with noise (77 of 80), differences, player names, loading az_match's files | one player 5 points a battle better must rise by 10 |
  | `position_edits_test` (new) | quick, 1.5 s | the value probe's position edits (`position_edits.h`): the far pieces, a piece removed, a dwarf moved — each against a computation of its own, on 707 positions | removing another piece must give another board |
  | `unsafe_targets_test` (new) | quick, 1 s | the taper's condition (c): the unsafe-move rule (by hand and against the board on 64,633 moves), R, its bootstrap upper end, the design's table of allowed steps | just above each bound must be refused |
  | `untried_move_test` | full, 107 s | the untried-move rules: the formula, and their effect on searches | upstream's rule must spread the dwarfs' searches |
  | `pruning_test` | quick, 13 s | forced playouts and policy target pruning | k 0 must leave 3x more children short |
  | `search_settings_test` | quick, 0.3 s | per-player simulations, policy temperatures, root noise α and ε (InitAZBot) | temperatures {1, 1} must change nothing, {2, 1} must spread a sharp prior; with no list both sides noised |
  | `growing_buffer_test`, `replay_buffer_test`, `az_buffer_truncate self_test` | quick, < 1 s | the growing buffer's size; our replay buffer against upstream's (contents, samples, saved bytes, memory; since 2026-10-15 a file cut short: ours refuses it); truncating a buffer (upstream's format only: files of the compact buffer need not be truncated, they are written under a temporary name); since 2026-10-19 the compact buffer's SampleNewest against upstream's buffer's, before and after it wraps (a different seed: other positions) | upstream's ratchet must show; upstream's must load the cut file without an error (its bug) |
  | `instrumentation_test` | quick, 0.2 s | how a game's end is classified, every ending | a position one move short |
  | `alphabeta_training_test` | quick, 0.2 s | games against alpha-beta: only our network's positions recorded, its moves AlphaBetaPlayer's, per-side depths, the log line, the configuration's round trip | another seed must change a move; self-play must record both sides |
  | `kickstart_test` | full, ~25 s (189 s → 37 s beside a trainer, 2026-10-15: the printed cost timing now behind `cost=1`) | the teacher's scores and policy, the mixed targets, the weight per player | depth 1 must score the position's moves alike; the mixed target must differ from the unmixed |
  | `teacher_mix_test` (new) | quick, 0.3 s | `az_match teacher_mix_a`: the mix per player, exactly; the values unchanged; the cache | the teacher must differ from the prior at both nodes |
  | `grow_test` (new) | quick, 2.4 s | Net2Net (`net2net.h`): the mapping; the function kept for wider, deeper, both, the linear head, no growth; every tensor accounted for; copies stay equal under training without noise, part with it; a grown network reloads | three: no division; a batch norm's statistics left uncopied; new blocks not the identity — each must exceed 10x the tolerance |
  | `trainer_test.sh` (new) | quick, 26 s | the trainer end to end with every feature of ours on (two device copies, batching, augmentation, pruning, 2 learner batches, alpha-beta games, kickstarting and noise per player, a learning-rate warm-up and schedule, a value-loss weight): checkpoints, buffer, logs, config; resuming (the step follows on, the rates the schedule's, logs appended); training on from an `az_grow` network; `az_restructure` (a new head, frozen trunk; since 2026-10-17 two copies with `--explicit_learning` in the first run and one device in the grown one, each step's batch norms counting exactly its batches; `az_batch_norm_convert` (part 6); also training on at value-loss weight 4 with a warm-up and keep=every — the rates, the gradient sizes, the summary — and with `optimizer=source`); `az_fresh_optimizer`; since 2026-10-18 the capacity tools (part 7): `az_dormant_probe`, `az_plasticity`, `az_restructure copy=none` | the tools' own fatal tests (az_grow's control among them; `optimizer=source`'s: a fresh optimizer's step must differ; the dormant probe's killed channel must be found and its values equal Inference's; the plasticity test's wider fresh network must fit better; `copy=none`'s first outputs must differ from the source's) |
  | `az_match_test.sh` (new) | full, 97 s | `az_match`: reproducible; the record (openings, sides, the summary's arithmetic); `noise=`, `sample_plies=` and players without a network (alpha-beta, MCTS, hexparrot) the same whichever thread; `devices=`; `teacher_mix_a=` | each option must change the games; another seed too; the summary must record the devices |
  | `sanitized` (in `run_tests.sh`) | full, 10 s | `replay_buffer_test`, `features_test`, `az_buffer_truncate` under AddressSanitizer and UBSan | the binaries must carry ASan's instrumentation (and ASan was seen catching a heap overflow here) |

  **Faster tests** (the user, 2026-10-15/16): done, no weaker — `kickstart_test` (its cost
  timing behind `cost=1`) and `conv_policy_test` (stops once the loss halves); done,
  barely weaker — `legal_policy_test` reports its search comparison on 20 positions
  (required parts unchanged; 50 s → 18 s), `az_match_test` at 4 simulations (128 s → 55 s).
  **Tried and reverted:** `untried_move_test` with its position-supplying game at 25
  simulations — the positions do matter to it: upstream's rule then visited a median of 87
  dwarf moves, below its 90, and lowering the threshold would weaken it (my claim that it
  would lose nothing was wrong). `identity_test` unchanged (the user).

  **Not covered by any test** (the gaps, to be kept current):

  - *The probes and analysis programs* (`az_blunder_probe`, `az_dwarf_value_probe`,
    `az_risk_probe`, `az_long_capture_probe`, `az_feature_probe`, `az_input_plane_trace`,
    `az_capture_targets`, `az_teacher_stats`, `az_search_cost`, `az_target_quality`,
    `az_head_probe`, `az_noise_probe`, `az_forgetting`, `az_merge_buffers`): measurements
    with their own placebos or controls where they have one, but no unit test of their
    helpers (e.g. the blunder probe's safe-move rule) — the value probe's position edits
    are tested (`position_edits_test`), since the taper's condition (d) acts on that probe.
  - *The Python tools*: `watch_run.py`, `analyse_games.py`, the night queues — tested by
    hand once each, no tests (`ladder.py`, whose ratings decide promotions, has
    `ladder_test.py`).
  - *`az_match` options outside `az_match_test`*: `capture_floor_a`, `share=1`,
    `progress`, `cache`, `inference_threads`, `untried_a`/`untried_b`,
    `temperature_a`/`temperature_b`, `sims_a`/`sims_b`; `SelfPlayChooser`'s sampling only
    end to end. Its default path to `hexparrot_player.py` is found from the binary's folder
    and is right only for `build-shared/` (the test passes it explicitly).
  - *Moving a build folder* breaks its incremental build: each object's dependency file
    names it by its full path, so after the rename `build-checks/` → `build-tests/`
    (2026-10-15) nine objects had no header dependencies left and were never rebuilt —
    harmless until a header changed a class's layout (`VPNetModel`'s new member,
    2026-10-17): `pruning_test` crashed and `trainer_test` aborted, both linked with a stale
    `alpha_zero.o`, and the pre-commit hook stopped the commit. Fixed by a clean rebuild of
    `build-tests/obj/`; after moving a build folder, delete its objects.
  - *The trainer's batch norms* (2026-10-17): until the fix no test pinned the mode its
    learning steps ran in — on one device the inference threads decided it
    (*Upstreaming*, candidate 4). Now `trainer_test` requires each step's batch norms to
    count exactly its batches, with two copies and with one, and `batch_norm_test` has
    upstream's class as the control.
  - *The trainer*: its evaluators (`trainer_test` runs none), `explicit_learning`,
    resignation (`cutoff_probability`, off in our runs), `inference_cache` sizes, the
    learner's statistics' numbers (tested once by hand, 2026-10-03).
  - *Sanitizers on the LibTorch programs*: only the three tests without LibTorch run
    sanitized; ThreadSanitizer was never run on the trainer's or az_match's threads.
  - *CUDA*: nothing here (no GPU); the device code runs with CPU copies only. The setup
    script's smoke test (step 8) covers it on the machine.
  - *The setup script* (`thud/setup_machine.sh`): run on a fresh clone here (2026-10-15,
    every step passed); never on x86, with apt, or with a CUDA wheel. *`transfer_run.sh` over ssh*: untested (no second
    machine; between local folders it is tested).
  - *The warnings*: `-Wno-everything` remains in `build.sh`, `tests.mk` and `programs.mk` (to be narrowed,
    CLAUDE.md); without it the game's header warns once — `ThudState`'s defaulted copy
    assignment is deleted, as in 25 upstream games: harmless, left as upstream writes it.
  - `az_restructure` into another width or depth stops by design (the checklist above).
  - *Stage-1 programs on upstream's AlphaZero* (`az_layout_probe`, `az_throughput`,
    `az_reuse`, `az_buffer_stats`) are not in the suite.

  **Where a test, not the code, had to change** (the user asked that this be noted): no
  test needed a change in the game or in our AlphaZero code. `pruning_test` and
  `untried_move_test` were written for trained networks: on untrained ones their
  controls fail (a near-uniform prior binds no forced playouts; values near 0 spread no
  search), so `make_test_network` gives the networks a trained one's shape
  (`policy_scale=5`, `value_bias=-1`) — the tests themselves unchanged, `untried_move_test`
  run on 1 game instead of 6 (a whole game of 800 plies at 100 simulations each). Fixed in
  the new tests while writing them: `az_match_test` compared the games files byte for
  byte, but battles are written as they finish (sorted now); `trainer_test` tested
  `az_restructure` into 12 x 2 (now a new head, after the finding above); `grow_test`
  left LibTorch unseeded (its controls then varied from 5e-4 to 7e-3 between runs; seeded,
  each control must exceed 10x the tolerance); the sanitized test's control tripped over
  `pipefail`; `trainer_test` required a game against alpha-beta in 2 steps at a share of
  0.3, which fails 4% of the time (0.7^9; now 0.5 over 3 steps, 1 in 2,000). Refactored for the tests, each shown to change nothing: the growth into
  `net2net.h` (byte-identical checkpoints), `TeacherMixEvaluator` into
  `thud/experiments/teacher_mix.h` (identical games and summary). Found on the way: the
  playthrough instruction in CLAUDE.md and Phase 4 (regenerate and diff) could never pass —
  a new playthrough plays new random moves; corrected to upstream's replay. Later the
  same day, with the GPU checklist's features: `learning_settings_test` first compared
  outputs after a step at rate 0, which move through the batch norms' running statistics
  at any rate (now the parameters); `watchdog_test`'s killer was too slow to catch a
  checkpoint being written (now a larger network and a tight poll), and its control's
  test was malformed; `transfer_test`'s stand-in trainer exited at once. **The tests
  found four defects in the code** (none in the game): upstream's `json::FromString`
  stops the program on a line that does not parse, so the first tolerant resume died on
  a truncated record (now only whole lines are parsed); the record after a fragment
  joined it and was lost (now the logs' unfinished lines are ended); upstream's
  `LoadBuffer` loads a cut-off file without an error (ours now refuses it); `az_grow`'s
  default learning rate was the optimizer's, which a schedule scales (now vpnet.pb's).

  **How we tune the algorithm settings** (group 3; decided with the user, 2026-10-05: on
  the GPU, not here): (1) a cheap filter first, search only — a fixed strong network
  (e.g. G''s latest) against itself with different search settings (`uct_c` 1 / 2 / 4,
  the untried-move reduction 0.1 / 0.2 / 0.4, …), 40 pairs each: minutes on a GPU; it
  shows which settings make a trained network *play* better, not which make it *learn*
  better, so it only prunes the candidates; (2) training branches from a common
  checkpoint (as run D branched from C), each candidate ~10 steps, **2-3 runs per
  value** — branching is far cheaper than training from scratch and tests the setting
  where training stands (early-training effects are not covered); (3) evaluated by the
  ladder (each branch against the 2-3 nearest anchors, 40 pairs; adopted only if its
  rating beats the baseline's across the runs) and by the trainer's statistics per side
  (losses, the dwarfs' prior, self-play margins — catching a setting that helps one side
  and stalls the other, or a drift like run W's); (4) one or two settings at a time:
  `uct_c` with the untried-move reduction (both shape exploration), then root noise α
  (perhaps per side), then the temperature's drop. At the GPU's speed a 10-step branch
  takes ~15-30 minutes: 3 values x 3 runs, a few hours per setting. The evaluation
  watcher (*Instrumentation*, item 5) shows early when a branch falls behind, so it can
  be stopped, and the game analyser (item 6) how its play differs.

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
   (2026-09-30, estimated; measured on run G''s trainer, 2026-10-05: ~9.9 GB once its
   inference cache was full, 7.4 GB in its first hour).

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
the run, otherwise only the last ~3 steps (run C had an archive besides; run E's
archive folder is empty). Our
tools work from these after the fact: `az_buffer_stats` (search breadth per side),
`az_forgetting` (a network's fit to older positions), `az_target_quality` (targets
against deeper searches), `az_match` (strength as trained, per side), `az_head_probe`.

*Can we see overfitting?* Partly. The new-against-trained measure shows a generalisation
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
items 5 and 6 built 2026-10-05 (the user: "if that's helpful for our GPU work we should
implement it now" — the tuning's branches are judged by them); item 7 only if training
turns unstable:

1. **Per-side splits** of everything now mixed: the training losses, the
   new-against-trained measure, `value_accuracy` and `value_prediction`. The learner has
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
   as trained against the ladder's anchors nearest its strength, rated on the ladder,
   overall and per side, at a priority that leaves the trainer its CPU — a strength
   curve during the run (C' would have shown its dwarfs at −17 at step 8, 3.5 hours
   before its training ended). Harmless for equal-steps comparisons. (Proposed against
   the anchor and E44 at nice 19; built as described below.)
6. **After-the-fact tools**: a game analyser over the actors' logs (margins, captures,
   hurls and shoves per side, how games ended, how varied the openings are), and a
   per-step slicing of a 7x run's buffer for the statistics above (not built: item 3
   logs them per step since 2026-10-03; only older runs would need it).
7. Lower: gradient and weight norms, batch-norm statistics — only if training turns
   unstable.

Runs G and G' (from 2026-10-02) need none of this to be judged: their final buffers
hold every position, with checkpoints for every step and the actors' game logs, so
items 2, 3 (without the prior: it can be recomputed from the checkpoints), 4 and 6
can be computed afterwards.

**Items 1-3 built and checked** (2026-10-03, our copy's trainer; no switch, they only
log): `learner.jsonl` gains `loss_by_side` (each side's policy and value loss and
positions; the model now also returns each position's losses, detached), the
new-against-trained measure's `new_by_side` / `trained_by_side`, `selfplay` (player 0's
return — in Thud the dwarfs' — with a histogram of 21 buckets of 0.1, and the counts of
each `endings` category) and `by_side` (per side: positions, median legal moves, moves
visited, top share, effective moves, the prior's median effective moves, the mean KL
divergence of the visit counts from the prior, how often the prior's top move is the
most visited, and the values' sign accuracy and size at 7 points of the side's own
turns); the learner's text log a line of losses by side, and each actor's game line
its ending. **Tests:** `identity_test` 12 of 12 (one learning step equal to
upstream's: the losses unchanged); in a 3-step smoke run (32 x 2, 20 simulations,
augmentation on) the two sides' losses weighted by their positions reproduce the mixed
ones to ~1e-7 (training losses and both halves of the new-against-trained measure); all
32 logged games replayed in `pyspiel` give the logged ending and return, the margin
matching the pieces left; the per-step search statistics match `az_buffer_stats` on the
same 6,736 positions; `instrumentation_test` (new) plays one hand-built position to
each of the five endings (each with a control one move earlier) plus a cut-off game
and another game's end, 12 of 12, and catches a planted error (the turn limit's `>=`
as `>`); `conv_policy_test` and `augmentation_test` still pass. The prior statistics
are plausible on the smoke run's untrained network (its prior near uniform: 269.7
effective moves of 270) but were not recomputed independently. **First real use:** run
G' continued from 2026-10-03 (`~/thud-runs/stage5b_Gaug_continue.sh`).

**Items 5 and 6 built and checked** (2026-10-05, Python beside our programs; neither
changes training):

- **5. The evaluation watcher**, `thud/experiments/watch_run.py RUN_DIR`. Every 4 steps
  (`--every`) it plays the run's newest complete checkpoint (the trainer writes the
  network, then the optimizer: complete once `checkpoint-N-optimizer.pt` exists) as
  trained against the 2 ladder anchors nearest its last rating (`--nearest`; the first
  checkpoint against the weakest, or the anchors nearest `--guess`), 20 pairs each
  (`--pairs`), each run's rule for untried moves from its `config.json`. Then it refits
  the ladder (`ladder.py`, whose fit is now a function, `fit()`, its output unchanged)
  and appends the checkpoint's rating with its standard error and its dwarf and troll
  strengths, against E44, to `RUN_DIR/watch.jsonl` and to the progress log. Its matches
  join the ladder (`~/thud-runs/stage1_matches/watch_RUN_stepN_vs_ANCHOR.jsonl`). When
  several checkpoints are due it takes the newest, so it falls behind gracefully; it
  stops once `command.txt` has a "finished" line (or at `--until`), and resumes where
  `watch.jsonl` ends; `--all` (2026-10-06) evaluates every due checkpoint, oldest first,
  for a stopped or finished run after the fact. It must not slow or endanger the run. Its matches run in the
  idle scheduling class (SCHED_IDLE, `chrt --idle 0`, allowed unprivileged), not at
  nice 19, which competes as an equal with our trainers (themselves at nice 19); but
  the idle class is not "leftovers only": it weighs 3 against a nice-19 task's 15 (the
  kernel's `WEIGHT_IDLEPRIO`, `sched_prio_to_weight`), so it still gets ~1/6 of a core
  the two contend for — in a first measurement (two 20-second blocks each way, beside G''s
  trainer) the trainer used ~6.0 cores with two idle-class matches running and ~7.3
  with them paused. Whether that costs the trainer speed is the **load test** below.
  `--nice` runs the matches at nice 19 instead. And only with memory to spare: its
  matches keep a cache of 32,768 evaluations (`az_match`'s default, 262,144, took a
  match to 1.4 GB; a cache changes no game, and in matches it answers only ~5% of the
  value requests), which levels off at ~0.76 GB a match (20 threads); with 1.0 GB per
  match plus 1.5 GB available they run at once (OMP 1, a thread per pair, as
  `CLAUDE.md` advises), else one at a time, else it waits — G''s trainer levels off at
  ~9.9 GB of WSL's 15 (7.4 GB in its first hour, its cache still filling), and the
  out-of-memory killer would pick the largest process. (Both numbers were first
  measured too early, at 0.6 and 7.4 GB: on 2026-10-05 the first test's two matches
  grew to 1.4 GB each, swap reached 2.9 of 4 GB and 0.8 GB stayed available before the
  test was stopped.) It only reports: stopping a run stays a decision. **Test:** the
  first test (W24 against C16 and C29, 6 pairs each; expected near W16/W29's −16) was
  stopped for memory before a pair finished; to redo once the machine is free. Its
  first real use was the load test: G'46 against G'44 (20 pairs, finished) and A14
  (stopped with G' at 19:16).
- **The load test** (the user asked, 2026-10-05: does the evaluation watcher slow the
  trainer, e.g. through shared memory bandwidth?), running the night of 2026-10-05/06
  beside G''s trainer (`~/thud-runs/stage5b_load_test.sh`): the evaluation watcher on
  G' every 2 steps from 46 (so it always has matches to play) is the load, and
  `thud/experiments/load_toggle.py` pauses it (SIGSTOP; a match plays the same games
  however it is timed) in one random block of each pair of 20-minute blocks, until
  07:45, logging per block the trainer's and the load's CPU time.
  `load_test_analysis.py` then compares the trainer's self-play speed: the seconds per
  position of the games played entirely inside a block (an actor's game starts where
  its previous one ended), corrected for overlap with learning phases (a full overlap
  costs ~65%), paired within each pair, with a 95% interval. A per-step comparison
  would not do: G''s throughput varies 12-13% from step to step, and spreading games
  over blocks would smear the contrast; on quiet nights (runs E and G) the
  per-block measure varies 2-5%, on W's night (the user on the laptop) 15-17%.
  **Tests:** on G''s and E's old logs with made-up blocks, a planted 5% slowdown is
  found (+6.0%, +2.6 to +9.4; +7.2%, +3.1 to +11.2) and nothing planted reads as
  nothing (19 of 20 random schedules' intervals contain 0); the toggler's 20-second
  test had the load at 0.0 CPU seconds when paused, and resumed it at the end. With
  the CPU time per block it can tell a lost CPU share (the trainer gets less CPU)
  from lost efficiency (the same CPU, less done: memory bandwidth or caches; the test
  does not tell those two apart). If the watcher does slow the trainer, the likely fix
  is the trainer at nice 10 (weight 110, so the idle class gets under 3%, while the
  user's programs at nice 0, weight 1,024, still come first).
  **First result** (5 pairs, 15:48-19:16, until G' was stopped for the buffer fix):
  with the load, self-play was slower in every pair — 34-45% a position in 4 pairs (the
  games inside a block; the fifth +227% on only 6 games), and 13-51%, mean 25%, counting
  every position spread over its game's duration (a measure that smears the contrast,
  so it can only underestimate) — while the trainer's CPU fell only from 7.4-8.0 to
  6.6-7.0 cores (~10%). **So the evaluation watcher is not harmless beside training on
  this machine: it costs ~1/4-1/3 of self-play, mostly in efficiency, not CPU share** —
  the matches' two networks and 20 threads compete with the trainer for the shared
  caches and memory bandwidth (the trainer gets as much CPU but does less with it). The
  trainer at nice 10 would not fix that (it addresses the CPU share only). The second
  half (on the fixed trainer, from step 56) was not run (the user: leave G' alone — the
  answer is clear, and it would have cost G' ~3-4 steps). Consequences: on this
  machine, evaluate between runs, or with far fewer match threads; on the GPU machine,
  measure again (its trainer's inference runs on the GPU).
- **6. The game analyser**, `thud/experiments/analyse_games.py RUN_DIR [--by 4]
  [--jsonl FILE]`. It replays the actors' logged games in `pyspiel` (~2 s for 700
  games) and reports per group of steps: the dwarfs' margin and how games ended; per
  side the captures — hurls (and long ones, 2+ squares), the trolls' capture steps and
  shoves, dwarfs taken per troll capture; the median turn of each side's first capture;
  and the openings' variety (distinct sequences of the first 2, 4, 8 and 16 moves, as
  a share of the games). A game belongs to step k when it was logged between the
  learner's "Step: k−1" and "Step: k" lines — the same games the trainer's own per-step
  statistics count. Two limits of the logs: upstream logs only the first 20 actors (with
  our 32, ~62% of the games), and each resume starts the actor logs anew (they hold the
  latest segment only: G''s steps 1-44 are gone). **Done at the next trainer build**
  (2026-10-05, the buffer fix below): the actor and evaluator logs now append, as the
  learner's does, so a resumed run keeps its games (G' from its resume at 19:56).
  **Tests:** every replay of W's
  698 logged games reproduces its logged returns, and its ending, computed from the
  replay by the trainer's rules, matches the logged one (698 of 698, 4 of them at the
  no-capture limit); the captures counted move by move add up to the pieces missing;
  the per-step margins match the trainer's `selfplay` statistics within sampling noise
  (W steps 17-29, ~62% of each step's games: step 19 −18.66 against −18.60, step 20
  −15.82 against −15.25 — an assignment off by one step would show there); three
  corrupted logs are caught (returns −0.25 for −0.125, a wrong ending, a dropped move).
  It shows known events: **run A's dwarf collapse** at steps 17-18 (the dwarfs' margin
  −8.4 → −22.9, hurls per game 5.6 → 2.3); **run C's narrowing** (65-68% of 2-move
  openings distinct at steps 26-31, against 97-100% in E, G, C' and W before step 25);
  **C''s dwarfs not learning** (~1 hurl per game through step 16, G 6.1 at steps
  13-16). And one new: **W's trolls give pieces away early** — from step 25 the
  dwarfs' first hurl comes at turn 4 (the median; E 24-30, G 31-32 at steps 13-17), before the
  trolls' first capture (turn 7-9), and only 79-80% of W's 2-move openings are distinct
  — consistent with its trolls' drift on the ladder.
- **The buffer's memory, and copies a move would save** (2026-10-05; the memory log
  beside G' found its trainer growing, 9.75 GB at 15:39 to 10.62 GB at 19:16, 6.1-6.4 GB
  of it in the main heap, where the learner writes the buffer). Two causes in upstream's
  buffer (`open_spiel/utils/circular_buffer.h`, `serializable_circular_buffer.h`), shown
  by `thud/az/replay_buffer_test.cc` on positions shaped like ours: (1) once full,
  `Add` copy-assigns a position into the oldest slot, and a copy-assigned `std::vector`
  keeps its storage when that is large enough, so each slot keeps the largest position
  it ever held — a slot that held a dwarf position (median ~240 legal moves, 24 bytes
  each in the legal moves and the policy target) keeps that size when a troll position
  (~16-22) replaces it: +72% over what the positions need after 3 turnovers; (2)
  `LoadBuffer` (libnop reads each vector element by element) leaves spare capacity in
  every vector: +45% right after a resume — for G''s 7x buffer (4.66 GB on disk) ~6.8
  GB. **Fixed in our copy, no upstream change** (`thud/az/replay_buffer.h`,
  `ReplayBuffer`): `Add` moves the position into its slot (the learner takes its
  reservoir sample's copy first), and `LoadBuffer` replaces every element by a copy of
  itself, which holds exactly its size. **Tests:** the same contents, samples and saved
  file as upstream's buffer, and exactly the bytes the positions need after loading and
  after 3 turnovers (upstream's +45% and +72% are the controls); a smoke run of the
  trainer, fresh to step 2 and resumed to 3 (8 actors, 20 simulations, 32 x 2, a buffer
  of 1,024 overwritten every step). **The audit of our other C++** (the user asked: what
  else would a move improve?): the game code has nothing worth changing; the learner
  copies each sampled batch (per step, negligible) and upstream's trajectory queue
  copies each finished game twice (~2.5 MB, once a game: small); the hot spot is the
  batching evaluator, where a simulation's inputs (~7 KB) go through upstream's
  `ThreadedQueue`, which copies in `Push` and again in `Pop` under its lock, and the
  results are copied into the outputs (`vpnet.cc`) and into the promises (and the
  cache's results: upstream's `LRUCache::Get` returns a copy, two of ~4 KB per hit). We
  passed them by pointer and moved them, checked that, and **reverted it** (the user,
  2026-10-05: they save neither compute nor RAM, and similarity to upstream helps).
  **Tests:** a match plays the
  same games before and after (4 pairs, identical); the CPU study's batching benchmark:
  12,025 / 12,275 → 12,293 / 12,459 simulations a second at 32 threads, 24,287 / 24,472
  → 24,277 / 24,571 at 64 — **no measurable gain**: these copies are not what caps the
  batching path (its ~140 µs a simulation lie elsewhere, most likely the thread
  handoffs), and they hold no memory (freed within microseconds), so they save no RAM.
  Both changes ran under AddressSanitizer and
  UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`; a match and the smoke run,
  no report) — which needed `-DNDEBUG_SANITIZER`: with AddressSanitizer, abseil adds
  generation fields to its hash tables unless that is defined (`raw_hash_set.h`), so
  our sanitized code and the uninstrumented abseil in `libopen_spiel.so` disagreed on
  the layout, giving two false reports (a null load in the cache's map, a new-delete
  size mismatch in the flag registry) until it was set. **Run G' resumed on the fixed
  trainer** from step 54 at 19:56 (stopped at 19:16 right after its checkpoint, the
  user allowed it); its memory: the memory log. Freed spare capacity mostly stays with
  the process as reusable space, so the fix should show as no further growth rather
  than a drop (`malloc_trim` after loading would return it; not built).

*Design of items 1-4* (Claude, 2026-10-02; each behind no switch — they only log — but
each with a test that it measures what it claims, as every change here):

- **1. Per-side splits.** The side to move is in every stored position: the
  observation's "trolls to move" plane (`thud::kTrollsToMovePlane`, plane 3 counting
  from 0, one value over the whole plane; `thud.h`), so no format change. The model's loss function gets a variant returning per-position
  policy and value losses (the trainer's means are their averages), and the learner
  averages them per side: the training losses, the new-against-trained measure, and
  `value_accuracy` / `value_prediction` (from each state's `current_player`). Test:
  the two sides' averages, weighted by their counts, reproduce today's mixed numbers.
- **2. Self-play results as margins.** Per step: the dwarfs' mean final margin, its
  spread and a histogram (the trajectories' returns), and how each game ended — all
  dwarfs or all trolls captured, no legal move, the 200-turn no-capture limit, the
  800-turn limit — read from the final state in the actor and carried in the
  trajectory. Test: on a few logged games, replayed from their move lists, the same
  margins and endings.
- **3. Search statistics per side.** Per step, from the new positions' targets: moves
  visited, the most visited move's share, effective moves (exp of the entropy), as
  `az_buffer_stats` computes them (test: equal on a saved buffer). And from each
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
  retroactively. Tests: the untrained network scores worst; the set ranks our
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
  against the anchor, **E44**, with standard errors and a goodness of fit against every
  match. **The untrained network does not belong on the ladder**: with its matches in,
  the misfits were exactly those — A14, which beats our strong networks, beats it by
  only +26.1 where C16 searching broadly beats it by +46.4: beating a near-random
  opponent does not predict strength between strong ones. So its matches are left out
  (they stay per-side diagnostics). **First ratings** (17 matches between trained
  networks, 13 within their intervals; as trained, against E44): A14 +1.7 ± 2.1, E44 0,
  D29 −5.4 ± 2.1, C29 −10.3 ± 2.5, G'16 −10.5 ± 5.2, C16 −13.3 ± 2.2, G16 −21.4 ± 3.9,
  C'16 −21.5 ± 3.3 — the order of every head-to-head we have (G'16 above G16 by 10.9,
  observed 10.2; C16 above C'16); per side, C''s trolls +7.4 against E44's (its dwarfs
  −28.9), A14's +12.4 (its dwarfs −10.6). **Anchors:** M102 (the top anchor, promoted
  2026-10-12: the convolutional head and the 50 input planes; as trained), K83
  (promoted 2026-10-10; as trained, with its run's dwarf temperature 1.5), H83 (promoted
  2026-10-07 evening; as trained, likewise), G'64 (promoted
  2026-10-07 morning), G'44 (promoted 2026-10-04), A14, E44, D29, C29, C16 (as trained). **Promotion rule** (decided by the user,
  2026-10-04; **tightened 2026-10-06**): a network joins the anchors when (1) it beats
  the top anchor head-to-head in at least 40 pairs with the 95% interval of its pair
  margin above 0 — about 97.5% one-sided confidence, as Leela Zero's gate (55% of 400
  games, ~2 standard errors above 50%); the margin carries more information per game
  than a win, so fewer games suffice — **and (2) its ladder rating is above the top
  anchor's with the 95% interval of the difference above 0** (`ladder.py --compare
  NEW,TOP`, from the fit's covariance, which accounts for the two ratings sharing
  opponents). Test (2) was added after G'73 met (1) by exploiting G'44's dwarfs (+4.7,
  +1.9 to +7.5) while rating no higher (+0.4, −2.0 to +2.8): head-to-head alone can
  reward a network that beats the top anchor in particular (a non-transitive result);
  the ladder weighs every opponent. **(3) The regression gates** (decided by the user
  2026-10-13, refined the same day after the design's third review,
  `thud/DESIGN_dwarf_signal.md` §7; to be set up before the next promotion): tests (1)
  and (2) ask whether a network is better, the gates whether it lost a skill, against
  the top anchor — our dwarfs against alpha-beta depth 1 and our trolls against depth 2's
  dwarfs (40 battles each); blunders against hexparrot (the standard error clustered by
  battle); the trolls our dwarfs kill a battle against hexparrot and alpha-beta depth 1;
  the long-capture probe's built 2-3-square hurls; the trolls' captures (the prior's top
  move a capture where one exists) — each failing if worse than the anchor's by more than
  2 standard errors (analytic or bootstrap; the blunders: ≥ 4 percentage points); long
  hurls (4+ squares) monitored, never a gate. Each gate with a positive control (a
  network known to be worse must fail it: K83 the short-hurl gate, G'64 the blunder
  gate). Together the gates trip on an unchanged network in up to ~15% of evaluations, so
  a failed gate stops the promotion — and a step of D1's taper — until looked into (first
  a rerun of that measure with more battles), not automatically. Unlike Leela Zero's gate, ours does not choose
  which network generates the training games — it only adds a reference point — so a
  wrong promotion costs little. **G'44 promoted** (2026-10-04: beat A14 by +3.2, +0.7
  to +5.8, 60 pairs); **G'73 not promoted** (2026-10-06; the user: keep G'44 for the
  moment); **G'64 promoted** (2026-10-07, the user: beat G'44 by +8.60, +5.87 to +11.33,
  40 pairs; ladder difference +5.8, +3.7 to +7.9); **H83 promoted** (2026-10-07
  evening, the user: beat G'64 by +11.75, +9.0 to +14.5, 37 of 40 pairs; ladder
  difference +12.2, +8.7 to +15.6 — though it lost every pair to hexparrot's AI the same
  evening, as G'64 did: the ladder compares our networks only). Each new network plays the 2-3 anchors nearest its rating, 40 pairs each.
  When a network passes both tests, a frozen copy joins the ladder. Rating against fixed opponents is, as far
  as we know, the usual practice in AlphaZero-style projects (cite a source before
  relying on details). (2) *Training health*, by the **validation set**: minutes, every
  step, per side, comparable across steps and runs — whether a run still learns and
  where (plateaus, overfitting, one side falling behind). It need not measure absolute
  strength, and stays informative while our networks are clearly below its reference.
- **Outside opponents: players without a network** (2026-10-07; the user asked to look
  for other Thud AIs, expecting them to be stronger for now — rightly). No strong Thud
  engine is published. Those with a computer player: hexparrot/thudgame (hand-written
  rules — greedy captures, blocks against troll lines, an influence map; its
  "lookahead" plays its own rules forward, for the dwarfs only), Saucistophe/Thud
  (alpha-beta, depth 3, material plus troll clustering and mobility with a genetic
  algorithm's weights; there a troll step captures exactly one dwarf, not all or
  none), Schkliba/thud-game (alpha-beta on material), dstu/thud (UCT with random
  playouts), THFlowers/Thud-CLI (one random playout per root move); a Code Golf
  contest (AJFaraday/Thud) got no entries; nothing in Ludii or Zillions of Games.
  `az_match` plays three kinds as side B: `b=hexparrot:LOOKAHEAD` (its own AI, through
  `thud/experiments/hexparrot_player.py`; positions cross, not moves, so a move our
  rules do not allow stops the match), `b=alphabeta:DEPTH` (OpenSpiel's
  `AlphaBetaSearch`, in plies, scoring by material as the battle is scored, plus a
  random tie-break below a quarter point), `b=mcts:SIMS` (OpenSpiel's MCTS with random
  rollouts, as the trainer's evaluator plays it); `games=PATH` logs every move. Checked:
  matches between networks play the same games as the previous binary (3 pairs; the
  control, another seed, differs); reruns against hexparrot repeat their games. **First
  pairs (one each, G'64 at 100 simulations): hexparrot −57 and −53, alpha-beta depth 2
  −46, depth 1 −44 (1 ms a move against G'64's 1.8 s), MCTS-300 +20.** G'64's dwarfs
  step next to a troll to threaten a hurl; hexparrot's and alpha-beta's trolls take that
  dwarf every time (27 of 27, 29 of 29) — all 32 dwarfs gone in 70-80 moves. In G''s
  own self-play the trolls take such a dwarf at once in 23% of cases (1,690 in 60
  games), MCTS-300's in 27% (7 of 26): neither self-play nor the evaluator against MCTS
  ever punished it. Its trolls walk into hurling range (hexparrot's dwarfs hurled 7 of
  8). **The ladder rated our networks only against each other and could not see
  this.** Full matches against hexparrot (2026-10-07 evening): G'64 −55.4 a pair (−56.8
  to −54.0, 40 pairs), H83 −51.6 (−53.5 to −49.8, 40), G'44 −56.4 (20), E44 −55.7 (20)
  — none won a pair; as dwarfs −31.4 to −31.6 (nearly every dwarf lost in every
  battle), as trolls −20.2 to −24.8; our networks thought ~2.4 s a move, hexparrot
  ~0.03 s. **The night's answers** (2026-10-07 16:28 to 2026-10-08 02:47, all against
  the same seeded openings; a match killed by mistake replayed its 9 finished pairs
  exactly): (1) **every run has the blind spot** — D29 −55.0, C16 −56.6, and A14 −47.3
  (its dwarfs −31.2 like all; only its trolls better, −16.1), so upstream's rule for
  untried moves is not the cause: it is self-play co-adaptation throughout; (2) **more
  search does not fix it** — G'64 at 800 simulations −53.9 against hexparrot (−55.4 at
  100), −49.5 against depth 2 (−49.5): the network misjudges these positions; (3) the
  alpha-beta rungs, G'64 / H83: depth 1 −28.1 / −25.2 (H83 won 1 of 20 pairs; as trolls
  only −4.5 / −3.4), depth 2 −49.5 / −49.2, depth 3 −56.4 / −55.0 — **depth 3 thinks
  1.3-1.6 s a move, as long as our networks (1.5-1.65 s): at equal time we lose by 55**.
  Deferred: MCTS-10,000/-100,000 and depth 4. The night of 2026-10-07 (`~/thud-runs/outside_2026-10-07.py`, 21 matches,
  seed 1 for all; **reordered at 17:15 with the user** — `night2_2026-10-07.py`: the
  informative ones below up to depth 3, then run I and its evaluation; MCTS-10,000 and
  -100,000 and depth 4 deferred until our networks beat the lower rungs): hexparrot against G'64 and H83 (40 pairs) and G'44, E44, D29, C16,
  A14 (20 — is the weakness common to every run, A14's upstream rule for untried moves
  included?); alpha-beta depths 1 and 2; G'64 at 800 simulations against hexparrot and
  depth 2 (does more search fix it?); depth 3, MCTS-10,000, depth 4, MCTS-100,000
  (measured: depth 3 0.08-0.9 s a move, depth 4 4-12 s; MCTS 10,000 2-4 s and ~12 MB
  of tree, 100,000 35-49 s and up to 1.03 GB). Each side's seconds per move are recorded, for
  **equal-time comparisons** later (the user), with an alpha-beta of our own that
  deepens iteratively under a deadline — OpenSpiel's has none, and a ply costs 10-140x
  more here — checked against `AlphaBetaSearch`'s values at fixed depths. **Proposed,
  to decide with the night's results:** the trainer's evaluator against alpha-beta
  depths 1-2, not only MCTS; hexparrot and alpha-beta as ladder anchors.
- **Training against alpha-beta** (decided by the user, 2026-10-07). The user asked
  whether to use alpha-beta in training while our network is weak — giving up learning
  from rules alone at first — since our dwarfs lack the essentials. The evidence: in
  every pairing we have where one player plays both sides, the dwarfs come out ahead
  (hexparrot's AI 85% of 3,000 games, MCTS-5,000 19 of 20 at +17.7, MCTS-1,000 7 of
  10), while in our networks' self-play the trolls do (G''s dwarfs −1.9 to −9.3, run
  H's −1 to −2); only random play favours the trolls. **Decided:** (1) a share of the
  training positions comes from games of **our network against OpenSpiel's alpha-beta
  at depth 2** (~2 ms a move), alpha-beta playing the trolls in half and the dwarfs in
  the other half — a share of positions, not of games (below); **only
  our network's positions enter the replay buffer**, with its own search's targets and
  the outcome — alpha-beta only punishes, nothing imitates it (imitation first was
  rejected: a copy of a 2-ply material player learns its blind spots; AlphaGo Zero's
  from-scratch run overtook the one started from human games within about a day). The
  rest stays self-play, so the network still finds its own strategy. (2) The first
  test: **a constant 30% of the positions**, as **run I**: G'64 with run H's settings
  plus these games, against run H's own steps 65-83 from the same start — one run
  instead of two (Claude's proposal, 2026-10-07; the user: "sounds all great"), judged
  per side against hexparrot and alpha-beta depths 1-3 and on the ladder;
  if it fixes the blunders, a fresh run with it (do the dwarfs learn early?). (3) In
  long runs, **taper by results, per side, to a floor of ~10%**: once a side clearly
  beats alpha-beta in the trainer's evaluation, its share falls and the opponent gets
  stronger (depth 3, hexparrot); the floor keeps new co-adapted habits punished (as
  AlphaStar's league keeps its exploiters). In reserve, if that is not enough:
  kickstarting (Schmitt et al., 2018) — pulling the policy toward alpha-beta's move
  values (built 2026-10-08 as a mix of the policy targets, which changes neither the
  buffer's format nor the loss; below). The night's results could still change the
  lever: if G'64 at 800 simulations stopped blundering, search depth would be part of
  it; if A14 (upstream's rule for untried moves) blundered alike, it would be self-play
  co-adaptation throughout — both lost as badly (above): co-adaptation. **Built 2026-10-07** (our
  copy, off by default): `--alphabeta_share` and `--alphabeta_depth`; the opponent is
  `thud/az/alphabeta_player.h`, which `az_match`'s `b=alphabeta:N` now shares (its
  games byte-identical to before, network matches identical to the old binary); each
  actor alternates the side alpha-beta plays, and the actor's log line says so
  ("Opponent: alphabeta:2 as trolls;", which `analyse_games.py` reads and leaves out of
  its self-play statistics); the learner adds these games' positions to the buffer but
  reports them apart (`vs_alphabeta` in `learner.jsonl`: our network's return as each
  side, the positions), the self-play statistics staying self-play's. Checked by
  `thud/az/alphabeta_training_test.cc` (all pass at the first run): only the
  network's positions recorded, one per its move, the returns the final position's;
  every alpha-beta move the player's choice with the game's seed (control: another seed
  differs); self-play still records both sides; the log lines; depth 1 takes a free
  capture, depth 2 never steps a lone dwarf next to a troll over 40 seeds (control:
  depth 1 does on 2); the configuration round-trips and older configs load. A 3-step
  trainer run (16 x 1 network, 20 simulations, share 0.3): 29% of 59 games against
  alpha-beta (8 and 9 per side), and per step self-play plus these games add up
  exactly to the learner's games and positions. **A share of games is not a share of
  positions**: alpha-beta routs a weak network fast, and only our half of the moves is
  recorded, so there the games gave 1-10% of the positions; for stronger networks
  (G'64's battles against alpha-beta 66-256 moves) perhaps ~15%. **So the share is one
  of positions** (the user asked to raise it, as it is tapered anyway; Claude: change
  the unit — it is the share of the training data that carries the punishment, and
  of the actors' time): each actor plays alpha-beta whenever the positions its games
  against it recorded fall short of the share of all it recorded. For a network like
  G'64 (~400 positions a self-play game, ~70 against alpha-beta) 30% of the positions
  is ~70% of the games and ~30% of the actors' time; it stays where it is set as the
  network survives longer against alpha-beta, and tapering is lowering it. Checked by a
  second 3-step trainer run (share 0.3 of positions): 0.31, 0.32 and 0.24 per step,
  **0.29 over all** (3,697 of 12,803 positions), 64-74% of the games; each actor near
  the target (17-20 of 24-28 games); the sides 36 and 37; the books balanced at every
  step; `alphabeta_training_test` passes again.
  **Run I** (2026-10-07 19:29 to 2026-10-08 02:08, steps 65-83, mains, no pause;
  `~/thud-runs/runI_Gaug64_T15_AB30`: G'64 with run H's settings plus 30% of the
  positions against alpha-beta depth 2; its control run H's steps 65-83): **its trolls
  learnt to punish, its dwarfs learnt nothing against punishing opponents.** 24-37% of
  each step's positions came from alpha-beta games. In its self-play the trolls take a
  dwarf left next to them 46% of the time (run H 23.5%, G' 24.1%, its last 40 games
  each), its dwarfs offer 15.0 such moves a game (H 17.4, G' 26.4); its dwarfs'
  self-play margin fell from +1.2 to −11..−14 (the trolls punishing). Against players
  without a network, I83 / H83: hexparrot −52.7 / −51.6 (dwarfs −31.6 / −31.4, trolls
  −21.1 / −20.2), alpha-beta depth 1 −20.8 / −25.2 (**trolls +2.8 / −3.4**, dwarfs −23.6
  / −21.9), depth 2 −49.6 / −49.2, depth 3 −52.8 / −55.0 (trolls −21.2 / −23.6). On the
  ladder (58 matches): I68-I83 +24.4 to +28.1 (I80 the highest yet), H83 +22.6 (refitted);
  the gain is the trolls' (I +12.6 to +17.1, H +3.3 to +7.0), the dwarfs equal (~+11).
  I83 against H83 +2.73 a pair (−0.55 to +6.00), 28 of 40 pairs; ladder difference +2.8
  (−0.6 to +6.1): **the promotion rule is not met**; H83 stays the top anchor. **Why
  the dwarfs did not learn: their games against alpha-beta carried no signal.** Our
  dwarfs lost every dwarf in every training game against depth 2 (−32 at every step;
  in the evaluation 19 of 20 games), and the value target is the final result, so every
  dwarf position of those games got the same target and nothing told a blunder from a
  good move; the trolls' games ended between −27 and −7. Against depth 1 our dwarfs end
  between −32 and −7 (its trolls take every dwarf left next to them but walk into hurls
  themselves). **Decided** (the user, 2026-10-08): **run J**, run I with alpha-beta's
  depth per side — 1 when it plays the trolls, 2 when it plays the dwarfs
  (`--alphabeta_depth=2,1`) — against run I from the same start; the trolls' depth to
  rise as our dwarfs improve (the tapering by results). Further levers if that is not
  enough: value targets that see a blunder directly (material a few moves ahead,
  n-step returns), kickstarting. **Built 2026-10-08:** `alphabeta_depth` is a per-player
  list ("2,1"; one value for both; a number in older configs loads as that number),
  `AlphaBetaDepth` in our copy; `alphabeta_training_test` parts 7-8 (all pass); a
  2-step trainer run played 24 games against depth-1 trolls and 23 against depth-2
  dwarfs; a malformed list stops the trainer at its start. Run J's folder
  (`~/thud-runs/runJ_Gaug64_T15_AB30_d21`) is run I's start, its config differing only
  in the depth; `~/thud-runs/runJ_2026-10-08.py` (dry-run with fakes) waits for mains
  power, trains to step 83, then evaluates as run I was, plus J83 against I83.
  **Run J** (2026-10-08 07:19-14:44 on mains, no pause; its evaluation to 19:26, paused
  59 minutes when the laptop lost power and slept, results unaffected): **no gain for
  the dwarfs.** Its dwarfs' training games against depth-1 trolls stayed at −26.3 (step
  65) to −26.5 (75); in self-play they offered 16.7 dwarfs a game (run I 13.4), its
  trolls took 38.8% at once (I 42.1%). Against players without a network J83 / I83:
  hexparrot −52.5 / −52.7 (dwarfs −31.8 / −31.6), depth 1 −18.1 / −20.8 (dwarfs −22.8 /
  −23.6, trolls +4.8 / +2.8), depth 2 −50.4 / −49.6, depth 3 −54.9 / −52.8. **Against
  depth 1 every margin of our dwarfs is a multiple of 4** (−32 to −16): they lose every
  dwarf in every game, and the result varies only with the trolls they killed first —
  so the variation carried a signal about hurling, none about keeping dwarfs alive.
  J83 against I83 **−10.20 a pair (−14.48 to −5.92)**, 7 of 40 pairs, as dwarfs −12.12
  (against I83's punishing trolls), as trolls +1.93; ladder J68 +16.9, J72 +14.1, J76
  +25.9, J80 +20.8, J83 +13.0 (its dwarfs −0.2 ± 4.0, against I83 alone); J83 − I83
  −10.8 (−14.4 to −7.3). Run I stays the better branch. **The dwarfs need a signal
  that registers a dwarf lost soon after the move that lost it**, not 40 moves later:
  short-term value targets (KataGo trains auxiliary heads on discounted future values a
  few turns ahead) or kickstarting their policy from alpha-beta's scores of each move —
  to choose after a diagnosis on the blunder positions (is it the dwarfs' prior, the
  trolls' prior inside the dwarfs' search, or the value of the position after the
  capture?). **The diagnosis** (2026-10-08, `thud/experiments/az_blunder_probe.cc` on
  the games against hexparrot, 150 blunders each of G'64 and I83, searched again as
  played): a blunder is a non-capturing dwarf move next to a troll taken at once —
  **82% of G'64's dwarf moves, 50% of I83's**. (1) The dwarfs' prior favours them: such
  moves hold 39% (G'64) / 50% (I83) of the prior's mass against 18-19% of the legal
  moves, the blunder its top move 47% / 53% of the time. (2) **The trolls' prior inside
  the dwarfs' search hardly sees the capture**: its prior median 0.006 / 0.008 (below
  0.02 in 72% / 75% of cases), visited at all in 22% / 19% of the searches at 100
  simulations (28% / 30% at 800), 4% of that node's visits — so the dwarfs' search
  takes the troll's reply to be harmless: the co-adaptation sits inside the search. At
  800 simulations the search drops the blunder in 56% / 62% of these positions, still
  leaving far too many (G'64 at 800 lost as badly). (3) The value head registers
  material at about a third of its face value for both sides — a dwarf lost −0.023 /
  −0.011 (its material 0.031), a troll removed +0.054 / +0.039 (0.125): compressed,
  not blind to dwarfs. So the cause is the priors — above all the trolls' — not the
  value: kickstarting both sides' policies from alpha-beta's move scores fits,
  short-term value targets less.
- **Input features from the rules** (proposed by the user, 2026-10-08; after the
  capture measurement below, kickstarting first and the features as their own experiment
  later; **superseded on 2026-10-11 by *the input planes, as decided* — 50 planes,
  below**). Extra input planes that state facts the rules decide
  at once — what can be taken next move, what can be hurled or shoved and how far —
  so that the network need not learn to compute them. The user: not much expert
  knowledge, "still fairly obvious game knowledge", the network still learning the
  game from the ground up. **Precedent:** AlphaGo (Silver et al., *Nature* 2016) fed
  rule-derived features (liberties, capture size, ladders); AlphaGo Zero dropped them
  with far more compute; KataGo (Wu, arXiv 1902.10565) put such features back
  (liberties, ladder status, pass-alive areas) and measured that they speed learning
  up considerably. **The planes** (all computed from the board and the side to move):
  - direction-free: (1) dwarfs a troll move could take next turn; (2) trolls a hurl
    could take next turn; (3) **squares where a dwarf would be taken if it stood there**
    (the diagnosed blunder, as a map); (4) squares a hurl could land on;
  - per direction, 8 planes each (the layout of our actions, square × direction ×
    distance): for each dwarf, how far it can be hurled in that direction (k/7, 0 if
    not), and for each troll how far it can be shoved — a piece that can go several
    ways lights several planes; optionally whether that hurl or shove captures.
  **Where, and the buffer's memory:** not in the game's observation (it stays as
  upstream-facing code keeps it, with its playthrough), and not stored in the replay
  buffer: 20+ more planes would make a position ~4x larger, the 7x buffer ~10 GB —
  beyond this machine. Instead the network's input is built inside our copy
  (`thud/az`, the model's input for inference and learning alike) from the stored
  6-plane observation, which holds the whole board. Computed after symmetry
  augmentation, the per-direction planes come out right for the turned board by
  themselves. **Starting from a trained network** (a function-preserving start, the
  principle of Net2Net — Chen, Goodfellow and Shlens, ICLR 2016, arXiv 1511.05641; an
  earlier version of this line said KataGo and Leela do this when adding features,
  which no source checked on 2026-10-09 confirmed): I83's or H83's weights with the new
  input channels' weights set to zero, so that it first plays exactly as before — the
  first mode of *a tool to restructure a network*, below. **Tests:** each
  plane against the move generator on random and tangled positions (a plane says
  "can be taken" iff a legal capture takes it — with a planted error as the control);
  the zero-weight network's outputs identical to the original's; the planes of a
  turned board equal to the turned planes. **Test:** a branch with the features
  against the same branch without, equal steps, judged per side against hexparrot,
  alpha-beta depths 1-3 and on the ladder. **Relation to the diagnosis:** the blunders
  come from the priors, above all the trolls' prior on captures inside the dwarfs'
  search (median 0.006-0.008). Features help there if the network *cannot easily
  represent* "this troll move captures" — the linear policy head must assemble that
  from the trunk for each action, which fits run I's trolls finding captures at their
  own root (46%) while their prior stays tiny; a conv policy head (stage 5's, whose
  trolls learnt faster) points the same way. Kickstarting helps if the network has
  simply not been *told*. One measurement separates the two: in run I's self-play, troll
  positions with a capture available — the search's visits on captures (the policy
  target) against the network's prior on them; a large gap that training does not
  close says representation (features first), little visit mass says the targets
  never asked for it (kickstarting first). **The measurement** (2026-10-08,
  `thud/experiments/az_capture_targets.cc`: 200 troll positions with a capture
  available from run I's last self-play games, searched as self-play does, root noise
  on): G'64's prior gives captures 0.123 of its mass — their uniform share (12.4% of
  the legal moves) — and its search 0.125 (the capture the most visited move in 18%);
  I83's prior 0.169, its search 0.220 (43%). **The prior follows the targets** (run
  I's alpha-beta games raised both), so the network can represent "this troll move
  captures"; **the targets ask for it too weakly**: kickstarting first, the features
  as their own experiment later. **The features are built and checked** (2026-10-08;
  these 20 planes were replaced on 2026-10-11 by the 50 below, and `features_test`
  rewritten for them): `thud/az/thud_features.h` computes the 20 planes from the
  6-plane observation;
  `thud/az/features_test.cc` (all pass at the first run, 5 s): on 2,385 positions
  (979 from run I's self-play) the exact planes match the move generator with no
  difference (7,472 takeable dwarfs, 2,351 takeable trolls), the danger maps disagree
  with a placed piece on 3 of 333,730 squares (dwarfs; the placed dwarf blocking a
  shove's path) and 0 of 222,551 (trolls), the planes of turned boards are the turned
  planes, and both planted errors are caught. Not yet wired into the network.
- **Kickstarting** (Schmitt et al., "Kickstarting deep reinforcement learning", 2018;
  chosen by the capture measurement above; built 2026-10-08). `--kickstart_weight` w,
  `--kickstart_depth`, `--kickstart_temperature` T (points): every recorded position's
  policy target becomes (1 − w) × the search's visits + w × the teacher's policy, the
  softmax over each legal move's material score (`AlphaBetaPlayer::Scores`: the
  material after the move and the opponent's best reply to depth − 1, no tie-break)
  over T points; the move played is still the search's, and w = 0 is upstream's
  target. (Mixing the targets is the same as adding w × the cross-entropy to the
  teacher, kickstarting's distillation term.) Checked by `thud/az/kickstart_test.cc`
  (all pass at the first run): a capture scores one point above every other troll move
  at depth 1; a lone dwarf's three steps next to a troll one point below the rest at
  depth 2, and alike at depth 1 (the control); the teacher's policy; weight 0 records
  exactly the old targets, weight 0.3 plays the same moves and records exactly the
  mix (control: they differ); the settings round-trip. **The teacher's cost:** depth 2
  ~21 ms a position for either side (~4% of a self-play move); **depth 3 3.8 s for a
  dwarf position, 0.5 s for a troll's** — with OpenSpiel's unordered alpha-beta not
  affordable for both sides (the user asked), for the trolls alone at ~20-35% of the
  trainer's throughput (below); an alpha-beta of our own with captures searched first
  and a table of positions would make it so, the same search the equal-time matches
  need. Also built for the confirmation: `MCTSBot::SetPriorFloors` / `ApplyPriorFloor`
  (`search_settings_test` part 5, all pass: hand-worked numbers, floors 0 change no
  search, the trolls' roots get the floored prior, the floor reaches inside dwarf
  searches) and `az_match capture_floor_a=LIST`.
  **Depth 2 for both sides** (the user, 2026-10-08). Depth 3 would be affordable for
  the trolls alone (0.5 s a troll position, half the recorded positions, against ~0.46
  core-seconds a position in run I's trainer, 17-18 positions/s on ~8 cores: ~20-35%
  slower), but what the diagnosis found — the trolls' capture, which depth 1 sees, and
  the dwarfs' safety, which depth 2 sees from their side — lies within depth 2; and odd
  depths have their own blind spot (a capture on the last ply goes unanswered: a troll
  move setting up a capture that a hurl then punishes scores +1, not −3; even depths
  miss the reply to a hurl) — a capture search at the leaves, in an alpha-beta of our
  own, removes both. **What the teacher asks for** (`thud/experiments/az_teacher_stats.cc`,
  depth 2, 200 positions a side from run I's last self-play games, ~1 core-minute; at
  T = 0.25 / 0.5 / 1 / 2): the dwarfs have 255 legal moves on average, 39% of them tied
  for the best score, which get 91 / 73 / 53 / 45% of the teacher's mass; the blunder
  probe's unsafe moves (15% of the legal moves, 11% of them tied for the best) get
  5.9 / 7.8 / 11.3 / 13.2%. The trolls have 32.5, 36% tied for the best (92 / 70 / 54 /
  45%); at the 138 positions with a capture available, captures are 11% of the legal
  moves (74% of them tied for the best, the rest captures that a hurl punishes) and get
  89 / 61 / 39 / 25% — against G'64's prior on them 0.123 and its search's 0.125 (the
  capture measurement). **Decided for run K** (proposed by Claude, adopted by the user,
  2026-10-08): **T = 0.25** — the
  teacher decisive about material (at 0.5 it still puts over a quarter of its mass on
  worse moves); **w = 0.3** — the captures' share of the trolls' targets rises from
  ~0.125 (G'64's search) to ~0.35, enough for a capture to be the troll prior's top
  move, while 70% of each target stays the search's: the teacher is uniform over the
  ~100 dwarf moves tied for the best, so a larger weight flattens the dwarfs' policy
  among quiet moves, where it knows nothing. **Prepared** (2026-10-08 evening): run K's
  folder (`~/thud-runs/runK_Gaug64_T15_AB30_ks`: run I's start, its config differing
  only in the three settings; the step-64 buffer rebuilt when the run starts); the
  trainer rebuilt with kickstarting, each actor logging "Kickstarting: weight W, depth
  D, temperature T" (the previous binary kept as `build-shared/az_trainer_before_kickstart`);
  `~/thud-runs/night3_2026-10-08.py` (dry-run with fakes, a failing and a passing
  measurement): tests of `az_match_next` and a kickstarting smoke run, then the
  capture-floor confirmation, run K and its evaluation as run J's.
  **The capture-floor confirmation** (2026-10-09 11:51-12:27, `~/thud-runs/night3_2026-10-08/`;
  the prediction: if the trolls' prior inside the dwarfs' search causes the blunders,
  they mostly vanish): G'64 with `capture_floor_a=0,0.5` — the trolls' captures get at
  least half the prior at every troll node of its searches — against hexparrot:3 and
  alpha-beta depths 1 and 2, 20 pairs each, the same seeds as its matches of
  2026-10-07, compared pair by pair. **Its trolls transform:** as trolls −24.8 → −4.4
  against hexparrot (+20.5 ± 4.0), −4.6 → **+27.0** against depth 1 (+31.6 ± 6.0),
  −17.8 → −1.4 against depth 2 (+16.4 ± 5.4). **Its dwarfs' results do not move**
  (−31.8 → −31.6, −23.6 → −23.2, −31.8 → −31.8): they still lose nearly every dwarf.
  But they lose them more slowly: counted from the games (a non-capturing dwarf move
  ending next to a troll, that dwarf taken by the very next move; the games with G'64
  as the dwarfs), the blunders fall from 76% to 52% of their moves against hexparrot,
  50% → 25% against depth 1, 76% → 41% against depth 2, the games lasting 70 → 104,
  84 → 139 and 74 → 133 plies — with about as many blunders a game (27, 21 → 18, 28 →
  27), since a game ends when the dwarfs run out. **So the mechanism is half
  confirmed**: the trolls' prior inside the dwarfs' search accounts for about half of
  the blunders a move; the rest come from the dwarfs' own prior (39% of its mass on
  unsafe moves) and from values that register a lost dwarf at a third of its worth.
  And **the trolls' prior on captures is our trolls' main weakness**, more than their
  search. For kickstarting this is the case for both teachers: the trolls' teacher
  raises that prior (in their play and inside the dwarfs' searches), the dwarfs' own
  teacher is needed for the other half. The choice stands (kickstarting first): the
  input features would raise the same prior.
  **Run K** (2026-10-09 12:30-20:02, steps 65-83; its evaluation 20:08-20:55 and,
  paused at the user's request overnight, 10:06-12:45 on 2026-10-10): **kickstarting
  works — K83 against I83 +30.0 a pair (+27.6 to +32.5), all 40 pairs won**, as dwarfs
  +2.6, as trolls +27.4. Against players without a network K83 / I83: hexparrot −36.9
  / −52.7, alpha-beta depth 1 **+8.7** / −20.8 (the first win of any network against an
  outside opponent; +5.9 to +11.4), depth 2 −33.7 / −49.6, mostly the trolls (against
  depth 1 as trolls +25.2 / +2.8, as dwarfs −16.6 / −23.6). Ladder (against E44): K68
  +31.9, K72 +40.8, K76 +57.9, K80 +65.9, **K83 +67.8 ± 9.9** (dwarfs +15.4, trolls
  +52.4), against I83 +24.2 and the top anchor H83 +21.5. **The promotion tests**
  (2026-10-10 16:34-17:27, `~/thud-runs/promoteK_2026-10-10.sh`): **K83 against H83
  +30.57 a pair (+27.28 to +33.87), all 40 pairs won** (as dwarfs +4.6, as trolls
  +26.0); ladder difference **K83 − H83 +42.9 (+37.2 to +48.6)** (refitted: K83 +64.3 ±
  6.3, H83 +21.4 ± 2.7) — both tests passed: **K83 promoted to top anchor** (the user,
  2026-10-10; H83 stays an anchor; `watch_run.py` lists K83 first). Run K averaged ~13%
  slower than run I (15.6 against 17.9 positions/s), its first steps slowest, cause
  unknown (`PROGRESS.md`, 2026-10-09).
- **Do we need the input features? The diagnostics** (the user, 2026-10-09, asked for
  experiments that tell; run on the night of 2026-10-09 by `~/thud-runs/night4_2026-10-09.py`
  into `~/thud-runs/probe_2026-10-09/`). (1) *Does the prior now follow targets that
  ask strongly for captures?* Run K's targets put ~0.3 x the teacher's ~0.9 plus 0.7 x
  the search's visits on captures: `az_capture_targets` with K83, I83 and G'64 on run
  K's troll positions, and `az_teacher_stats` on run K's positions. If K83's prior
  matches its targets, the network can represent captures and the planes would mainly
  save capacity; if it lags far behind after 19 steps of such targets, representation
  (or the 2-number policy head) is the bottleneck. (2) *The blunder diagnosis of K83*
  (`az_blunder_probe` on its games against hexparrot), against G'64's and I83's: the
  dwarfs' prior on unsafe moves, the trolls' prior on the capture inside the dwarfs'
  search. (3) **The feature probe** (`thud/experiments/az_feature_probe.cc`, new):
  linear probes (Alain and Bengio, arXiv 1610.01644; McGrath et al., "Acquisition of
  chess knowledge in AlphaZero", PNAS 2022, who probed AlphaZero's layers against an
  untrained network's) read each feature plane's fact at a square — from the trunk's
  output there (what the network computes) and from the policy head's 2 planes in the
  5 x 5 squares around it (what reaches the linear policy layer nearby; a lower bound,
  as that layer reads every square) — trained on two thirds of the games' positions,
  scored by AUC on the rest, for K83, I83, G'64 and an untrained network of the same
  shape. Our copy's model gained `ModelImpl::Activations` / `VPNetModel::Activations`
  (one pass returning the trunk's output, the policy head's planes and the outputs;
  the probe checks the outputs equal `Inference`'s — exactly, difference 0 — so the
  activations are the ones that make them). Controls: a dwarf on the square (an
  untrained trunk 1.000; K83's 0.959 at 100 optimiser steps, 0.984 at 400 — so 1,000
  steps tonight) and the danger plane with shuffled training labels (5 shuffles, mean
  0.52, range 0.48-0.59: one shuffle alone strays, a random direction in a space where
  the fact is encoded). **A first look** (the program's test, 300 positions only, ±0.05
  for the troll facts' ~30 positives): random features already make several facts
  readable (an untrained trunk: dwarf danger 0.88, troll captures per direction
  0.71-0.92), and K83's trunk seemed to read the troll captures worse than an untrained
  one (0.70-0.86) — **not confirmed** on 4,000 positions (below: equal).
  **Results** (2026-10-10 09:25-10:07, after the laptop slept on battery 02:31-09:24;
  `~/thud-runs/probe_2026-10-09/`):
  - *The trolls' captures — a matter of targets, now largely fixed.* At run K's troll
    positions with a capture (200 of 1,287; captures 10.4% of the legal moves): K83's
    prior puts **0.460** on captures (I83 0.148, G'64 0.091), its search 0.531; the
    prior's top move is a capture in **100%** of them (I83 38%, G'64 22%). Its targets
    asked for ~0.64 (0.7 x the search's ~0.53 plus 0.3 x the teacher's 0.886 —
    `az_teacher_stats` on run K's positions), so the prior follows them, a little
    behind after 19 steps. The network can represent "capture here" well enough.
  - *The dwarfs — half fixed, and the rest looks like the signal.* Against hexparrot
    K83's dwarfs blunder in **36%** of their moves (577 of 1,609; I83 50%, G'64 82%).
    The trolls' prior on the capture inside the dwarfs' search: median **0.069** (I83
    0.008), the capture visited in 89% of the searches (I83 19%). But the dwarfs' own
    prior still puts 0.337 on unsafe moves (I83 0.498, G'64 0.389) against their 18.9%
    of the legal moves, the blunder its top move 51% of the time; and the value head
    still registers a lost dwarf at ~1/3 of its material (v_captured − v_before
    −0.011, its material 0.031; a troll removed +0.034 of 0.125) — plausibly because in
    self-play the dwarfs lose every dwarf anyway (saturated outcomes), so a dwarf lost
    now barely changes the final result the value is trained on.
  - *The probe* (4,000 positions, 1,000 optimiser steps; AUC on held-out games, troll
    facts with 295-536 positives a direction). **Torso** — K83 / I83 / G'64 / untrained:
    dwarf takeable 0.88 / 0.90 / 0.91 / 0.77, dwarf danger 0.93 / 0.93 / 0.93 / 0.86
    (learnt: better than random features); troll takeable 0.93 / 0.94 / 0.91 / 0.93,
    troll danger 0.98 / 0.98 / 0.95 / 0.98, hurls (mean of 8 directions) 0.98 / 0.98 /
    0.99 / 0.98, **troll captures (mean of 8) 0.865 / 0.858 / 0.862 / 0.867** — no
    better than random features, at the troll's square. **Policy planes** (5 x 5
    around): troll captures 0.76 / 0.76 / 0.70 / 0.75, dwarf danger 0.89 / 0.88 / 0.89 /
    0.75 — the 2-number head keeps less than the trunk, as expected. Controls: a dwarf
    on the square 0.99 / 0.99 / 0.99 / 1.00 (torso); the shuffled-label control's means
    0.61 / 0.59 / 0.47 / 0.53 over 5 shuffles, single shuffles ranging 0.12-0.82 — far
    noisier than at 300 positions, since in a trained trunk the danger fact lies along
    a dominant direction that a noise-fitted probe catches by chance; consistent with
    no leak, but too noisy to check a single probe.
  **Reading:** the trunk does not encode "this troll captures in direction d" at the
  troll's square better than random features — yet K83's prior finds a capture as its
  top move every time, so the network encodes capture chances some other way (likely at
  the landing squares, which the linear policy layer can read for any troll move). The
  dwarfs' danger map is learnt (0.93 against 0.86 untrained). **So these diagnostics
  make no case for the input planes now**: the trolls' failure was the targets
  (kickstarting fixed most of it), and the dwarfs' remaining blunders point to their
  training signal — the saturated outcome behind the compressed value, and a teacher
  weight of 0.3 against search targets that still favour unsafe moves — not to a fact
  the network cannot compute. Caveats: a linear probe measures what is linearly
  readable at one square, not what the network could learn; AUC 0.86-0.93 is not
  certainty, and exact planes would make the facts certain (the capacity argument
  stands, unmeasured); only an A/B run decides. (Refined the same day by the
  long-capture probe, next: a real gap, in the policy head more than the inputs.)
  **The long-capture probe** (the user, 2026-10-10: do the networks see a capture at
  the end of a long hurl or shove, across the board, in every direction?
  `thud/experiments/az_long_capture_probe.cc`, `~/thud-runs/probe_2026-10-10/`).
  *Built* positions in minimal pairs: a line of k dwarfs (k trolls) whose k-square hurl
  (shove) is the side to move's one capture, k = 1-7 (7, the longest the board allows),
  8 directions, up to 12 places x 4 random surroundings (3,744 positions; checked by eye
  on printed samples), and its twin with the line one piece short; *real* positions:
  every capture in run K's last 10 self-play games of each logged actor (36,124). The
  move's probability against a blind prior's (1 / legal moves), its rank, and in the
  built pairs how much more the network likes it when it captures; K83, I83, H83, G'64
  and an untrained network (the control: exactly 1.00x, no gain). **Results** (K83):
  real troll steps 4.4x (rank median 3; I83 1.4x, H83 and G'64 below 1); real 1-square
  hurls 42x (rank 3; the others 23-30x) — **but real 2- and 3-square hurls 1.1x and
  0.9x, never the top move (I83, H83, G'64 0.5-1.0x), although kickstarting's teacher
  rates them the best move 99% and 91% of the time** (308 and 122 of them; 4 at 4
  squares); built hurls of 2-7 squares 0.40-0.58x (below blind) in every direction,
  the twins showing no gain beyond the control's, the value moving +0.01-0.02 (a troll
  is 0.125). Shoves: real 2-4 squares 4.0-4.8x (rank 2-3, top 22-37%; I83, H83, G'64
  0.1-1.4x) — kickstarting taught them — though the teacher rates only 60% / 44% / 38%
  of them the best move; built shoves 1.8x at 2 squares, ~1x at 3-7 (the built
  positions, emptier and more evenly spread than real ones, damp what the network does
  in real ones: real shoves of 3-4 4.6x, built ~1x). **Reading:** the networks do not
  act on multi-square hurls at all — in real games, where they are the best move and
  run K's targets asked for them, and built, at every distance and direction — while
  the trunk holds the fact ("a capturing hurl in direction d", AUC ~0.98 at the dwarf's
  square, the feature probe). So the gap lies between the trunk and the move logits:
  the linear policy head, which reads 2 numbers a square, cannot say "this dwarf, this
  direction, this distance" for many dwarfs at once; the trolls, 8 pieces, get their
  shoves through it. This fits the head probe (the linear head learnt the dwarfs'
  hurls to 75-78% of their policy mass, the convolutional head 99%) and AlphaZero's
  chess and shogi heads (a convolutional policy of 73 planes a square for moves by
  direction and distance). **Next, to confirm cheaply** (done the same day: convK83,
  next): K83's trunk with a
  convolutional policy head trained on run K's buffer (*a tool to restructure a
  network*, mode 2), then this probe on it — if it then prefers the multi-square hurls,
  the head is the bottleneck, and the trained head is the start of a run with it; the
  input planes (per direction) could follow, as they suit a per-direction head.
  (Also seen: built 1-square hurls range 0.6x-6.2x by direction, east and west far
  above north and south — unexplained; in real games they are 42x.)
  **The confirmation: the same trunk with a convolutional head** (convK83: K83's trunk
  and value head frozen, a convolutional policy head trained on run K's buffer by
  `az_restructure`; the same probe, 2026-10-10). Against K83: real 1-square hurls
  **89x** (top move 55%; K83 42x, 23%), **real 2-square hurls 53x, the top move 81%
  of the time (K83 1.1x, never)**, real 3-square hurls 2.5x (rank 9; K83 0.9x, rank
  43); built hurls 63x at 1 square, **54x at 2** (top 76%; K83 0.6x), 1.6x at 3, and
  still 0.4-0.6x at 4-7 squares (K83 0.4-0.5x); built shoves **2.4-4.0x at every
  distance 2-7** (top 27-81%; K83 ~1x); real shoves 4.1-5.4x (K83 4.1-4.8x); steps 4.7x
  (4.4x); the hurl gain in the built pairs now positive at 2-3 squares (+0.15; K83
  −0.06). Every direction gains (built hurls of 2-7: 8.3-18.6x by direction, the
  diagonals highest). **Reading:** with the trunk unchanged, the head alone turns
  2-square hurls from ignored into the top move and makes shoves visible at every
  distance — **the linear policy head was the bottleneck** for these. Hurls of 3
  squares improve only a little and of 4-7 not at all: the facts must span 2k squares
  of a 4-block trunk, and run K's games hold almost none to learn from (122 real
  3-square hurls in 200 games, 4 of 4 squares). That is where the input planes (a
  capturing hurl per direction, with its distance, at the dwarf's square — exactly what
  a per-square, per-direction head reads) would help: long-range facts the trunk
  rarely needs and cannot easily compute. **And it plays better**: convK83 against
  K83 **+16.8 a pair (+14.1 to +19.5), 39 of 40 pairs** (2026-10-10 23:24-00:11;
  convK83 as dwarfs −9.6, as trolls +26.4) — with the trunk and value head unchanged.
  **Decided** (the user, 2026-10-11: this order, and a clean evaluation): **run L** —
  convK83 trained on as a whole with run K's recipe, steps 84-102 — against **run K
  continued** from K83 over the same steps (the head's test), both started by
  `~/thud-runs/night5_2026-10-11.py` (dry-run with fakes) from 00:20, then evaluated as
  run K was plus L102 against K102; then **run M**, run L's start plus the input planes
  (net surgery, the tool's mode 1, built meanwhile), against run L (the planes' test).
  Run L's folder (`~/thud-runs/runL_K83_conv`): run K's config with
  `nn_model=resnet_conv_policy` and max_steps 102; vpnet.pb the convolutional structure
  at run K's learning rate 0.0001; convK83's weights with a fresh optimizer
  (`thud/az/az_fresh_optimizer.cc`, new: az_restructure's optimizer had trained at
  0.001, and LibTorch may restore that with the optimizer's state; checked: the saved
  checkpoint gives convK83's outputs exactly on 200 positions, at rate 0.0001); run K's
  learner.jsonl to step 83 and a copy of its final buffer. Run K continued
  (`~/thud-runs/runK_Gaug64_T15_AB30_ks_cont`): run K's checkpoint 83 with its
  optimizer, the same learner.jsonl and buffer, max_steps 102. Both use run K's trainer
  binary. **Results** (run L 2026-10-11 00:20-08:00, run K continued 08:00-16:06, the
  evaluation to 22:29): **the convolutional head wins in training too — L102 against
  K102 +10.72 a pair (+7.81 to +13.64), 34 of 40 pairs won, 2 lost** (L102 as dwarfs
  −19.4, as trolls +30.1). Against players without a network L102 / K102 / K83:
  hexparrot **−21.4** / −37.1 / −36.9 (L102's dwarfs −21.9 against K102's −29.1, its
  trolls +0.6 against −8.0), alpha-beta depth 1 **+19.3** / +12.4 / +8.7, depth 2
  **−20.6** / −28.6 / −33.7 (L102's trolls +10.6), depth 3 **−37.9** / −46.5 / −45.7.
  Ladder (against E44): L88 +89.1, L92 +92.3, L96 +87.6, L100 +95.3; K88 +63.1, K92
  +69.3, K96 +75.5, K100 +64.7 (K83 +59.2 refitted). L102 and K102 have so far played
  only each other, so the ladder rates their difference (+33.5, +13.6 to +53.4) but not
  their level. In training, L's trolls ended at +19.6 against alpha-beta's depth-2
  dwarfs (K continued +1.9 at step 90); both runs' dwarfs still lost every dwarf to
  alpha-beta's trolls (−32 throughout).
- **The input planes, as decided** (the user, 2026-10-11, after the long-capture
  probe; replacing the 20 planes built on 2026-10-08, never wired in): **50 planes**
  (`thud/az/thud_features.h`, whose comments give each plane's meaning in full),
  computed inside our copy's model from the stored 6-plane observation (the buffer
  unchanged; computed after symmetry augmentation, so they turn with the board), for
  both sides at once whoever is to move, all values 0 or positive.
  - **48 move planes** — direction d (8) x distance k — **at the moving piece's
    square**, where the convolutional head scores the move (s, d, k): on a dwarf, 1 if
    its hurl of k squares in direction d captures a troll (at most one k a direction);
    on a troll, the dwarfs its move of k squares takes, over 8 (k = 1 a capturing step,
    2 and more a shove; several k a direction possible). **Distances 1-7 straight, 1-5
    diagonal:** a capture of k squares needs 2k squares in a straight line (k pieces, k
    − 1 empty squares, the target or landing square); the longest straight lines (rows
    and columns 5, 6, 8, 9) hold 15 squares, the longest diagonals 10 (such as (5,0) to
    (14,9) — the cut-off corners shorten every diagonal, the Thudstone cuts the central
    ones), so k ≤ 7 straight and k ≤ 5 diagonal, in every position (other pieces only
    take squares away). **The 8 diagonal planes of 6 and 7 squares, always zero, are
    left out** (the user, 2026-10-11: no benefit in a uniform layout — a convolution
    does not care about its channels' order, and the index needs only an offset per
    direction; AlphaZero's output keeps impossible moves because a move impossible from
    one square is possible from another, ours are impossible from every square).
    **Shared by both sides**: a square holds one piece, and the board planes say which
    (AlphaZero separates its players' pieces because both own the same kinds — Silver
    et al., *Science* 2018: the player's and the opponent's planes, "oriented to the
    perspective of the current player"; in Thud the kind says whose); the network tells
    a hurl from a troll's move by the piece on that square, one channel over. The
    trolls' capturing step has its own action id (the head's 8 extra planes), its input
    still in plane (d, 1).
  - **2 maps, at the affected squares**: "a troll could land next to here" (a step or
    a shove geometrically open — the path empty, the troll line long enough — whether
    or not it captures now): on a dwarf, takeable now; on an empty square, danger for a
    dwarf moving there. "A hurl could land here": on a troll, takeable now; on an empty
    square, danger for a troll moving there. (The 20 planes' separate takeable and
    danger planes were one fact each on different squares — takeable is the map and
    the piece plane — so they are merged; the user.)
  - **Precedent for the count**: AlphaZero's inputs had 17 planes for Go, 119 for chess
    and 362 for shogi; Leela Chess Zero's 112; AlphaGo's policy network 48, among them
    "capture size", how many stones a move at a point would capture, one-hot over 8
    planes — the move-aligned idea — and all 48 predicted expert moves better (57.0%)
    than the raw position and history alone (55.7%; Silver et al., *Nature* 2016).
    Extra planes enlarge only the first layer (64 x 56 x 9 against ~300,000 trunk
    weights), and the planes are almost all zeros.
  - **Tests, before the code** (the user: many): hand-built positions with hand-worked
    expected planes — the longest hurl and shove in each direction, lines one short,
    blocked paths, the Thudstone and the cut-off corners in the way, a troll capturing
    at several distances, several captures at once; every plane against the move
    generator on random and real positions (exact for the move planes; the maps against
    placing a piece); **rotating the board then computing the planes equals computing
    then rotating** (the direction planes permuted), on all 7 non-identity symmetries,
    hand-built positions included, with a wrong direction map as the control; the
    model's input (the 6 planes then the 50, at inference and in training, after the
    augmentation; one input channel weighted, the network moved only where that plane
    is nonzero, a zero plane the control); the surgery (the new channels zero: outputs
    exactly convK83's).
    **Built and passed at the first run** (2026-10-11): `thud/az/features_test.cc`
    (rewritten; 12 hand-built positions — the longest hurl and shove, 7 east, the
    longest diagonal hurl, 5, a line one short, a dwarf and the Thudstone in the path,
    two trolls in a path, a lone dwarf next to a troll, a troll line shoving at three
    distances, two captures at once, the two maps by hand — each plane exact, a wrong
    one failing; 96 positions with one capture of every kind, direction and distance
    the board has room for, from `thud/az/capture_patterns.h`; on 2,360 positions the
    move planes exactly the move generator's 18,658 captures, none longer than allowed;
    the maps exact on all 44,022 pieces, against a placed piece wrong on 3 of 333,841
    empty squares (a dwarf blocking a shove's path) and 0 of 227,089; rotating then
    computing equals computing then rotating on all 7 symmetries, 0 squares differ; the
    board has room for exactly the kept distances and none of the dropped; the planted
    errors caught) and `thud/az/input_planes_test.cc` (new; for the convolutional and
    the linear head: after the surgery the outputs exactly the original's on 2,708
    positions; weight on one plane's channel moves the outputs exactly where that plane
    is nonzero, for all 50 planes, the next plane's positions as the control failing;
    one learning step moves the weights of all 39 planes present in its batch and of
    none of the 11 absent). `VPNetModel` (our copy) builds the input with
    `ModelInput`, widens the first layer in `CopyParametersFrom`, and `Parameter` reads a
    weight by name; `CreateGraphDef` takes extra input planes; `az_restructure` does the
    surgery (`input_planes=1 copy=all max_batches=0`, its test: outputs exactly the
    source's). Run M's start made so (`~/thud-runs/runM_K83_conv_planes`, 56 input
    planes, outputs exactly convK83's on 200 positions).
  - **Learning, over many steps** (the user): one backward pass gives every plane
    present a nonzero gradient; over the offline training and over run M's 19
    checkpoints, the new weights grow and differ across filters and planes, the held-out
    loss is no worse than without the planes, the old weights stay in range, and zeroing
    the planes changes the outputs more and more — a small tool reading the checkpoints,
    no change to the trainer.
  - **The offline measurement** (2026-10-12 01:44-03:25, after the user's pause;
    `~/thud-runs/probe_2026-10-11/`): convK83 trained on as a whole on run K's buffer,
    without the planes (A) and with them (B), learning rate 0.0001. **Both overfit at
    once** — the training loss falls, the held-out loss rises from the first report (A:
    3.653 → 3.665 → 3.681 → 3.683 → 3.704 at 0-1,000 batches; the value loss 0.057 →
    0.087): K83's network already fits these positions, and a fixed buffer cannot teach
    it more, so this offline setting is no test of the planes' worth; self-play (run M)
    is. **Even so the planes help**: B stays ~0.014 below A at every report (3.651 at
    250 batches, its best, against A's 3.665), its value loss lower too (0.061 against
    0.067). **The zero-initialised weights learn and do not collapse** (the user's
    question): their size per plane grows steadily (median 0.065 → 0.177, largest 0.12
    → 0.46 over 1,250 batches), they spread apart across filters (0.002 → 0.008), and
    the network leans on them more and more (zeroing them moves the policy by 0.071 →
    0.092 of its mass, the value by 0.11 → 0.19). The long-capture probe on B (250
    batches): real 2-square hurls 60x (top 87%; convK83 53x, 81%), 3-square 3.0x (2.5x);
    4-7 squares still below blind; shoves lower than convK83's (real 2.8-4.0x against
    4.1-5.4x — the overfitting, likely). **Reading: the long hurls are now a matter of
    examples.** Each direction and distance has its own input plane and its own output
    plane in the head, and a weight learns only from examples: in 200 of run K's
    self-play games there are 122 hurls of 3 squares and 4 of 4. The symmetries share
    examples across directions, not distances. If long hurls come to matter, the remedy
    is examples — built positions with long captures mixed into training, the teacher
    giving their targets — not another input design. **Deferred** (the user,
    2026-10-12: not a major problem yet — stronger dwarfs will form longer lines, so the
    examples should come with strength). The caveat: self-play can also keep them rare
    (dwarfs that do not see long hurls do not set them up, trolls do not learn to avoid
    them — the co-adaptation of the dwarfs' blunders again), leaving a blind spot an
    outside opponent could use. **Monitor:** the hurls by length in each run's
    self-play (the actors' logs) and in the games against hexparrot and alpha-beta,
    above all theirs against us; and **the long-capture probe**
    (`thud/experiments/az_long_capture_probe.cc`: built minimal pairs at every distance
    and direction, and every capture in the run's own last self-play games; ~3 minutes a
    network on 4 threads) on each run's final network and on every network promoted to
    an anchor, with the previous main line beside it (so far G'64, H83, I83, K83,
    convK83 and the offline B — `~/thud-runs/probe_2026-10-10/`, `probe_2026-10-11/`;
    next L102 and M102, once run M is evaluated); it tells whether the prior
    picks up long hurls as they become more common. **M102 and L102** (2026-10-12, on run
    M's last games): real 3-square hurls M102 13.9x (top 50%) against L102 4.8x (14%) —
    14 cases; built 7.4x against 2.4x — the planes help where examples exist (their
    weights for distance 3 grew most); 4-7 still below blind for both; built shoves of
    5-7 squares now 4.6-5.4x (top 77-94%) for both (K83 ~1x); real 2-square hurls
    ~140-150x, the top move 95% of the time. **Revisit if** long hurls (4+
    squares) grow common in self-play while the probe still finds them ignored by the
    prior (below a blind prior's probability, built or real), or show up in our losses
    to outside opponents. The dwarfs' training signal stays the larger problem (they lose every
    dwarf in self-play and to alpha-beta's trolls in training; *the dwarfs' training
    signal*, below).
  - **Run M** (2026-10-12 03:25-12:43, steps 84-102, no pause; its evaluation to
    16:37): **a tie with run L — M102 against L102 +0.65 a pair (−1.63 to +2.93), 14
    won, 9 drawn, 17 lost**; ladder M88 +87.8, M92 +82.0, M96 +98.8, M100 +114.2 ± 74
    (L88-L100 +86 to +94; the ladder saturating at the top: L and M beat its anchors
    almost always); against players without a network M102 / L102 hexparrot −22.1 /
    −21.4, alpha-beta depth 1 +14.8 / +19.3, depth 2 −18.1 / −20.6, depth 3 −29.6 /
    −37.9 — pair by pair on the same seeds M − L +8.3 (+0.3 to +16.2) at depth 3, all
    of it the trolls, −4.6 (−9.3 to +0.2) at depth 1 through the dwarfs, the others
    within noise (one of four comparisons barely clearing zero: suggestive only).
    **Decided** (the user, 2026-10-12): **M is the main line** — the planes kept, as
    they may help later (longer dwarf lines, long hurls) and the network uses them.
    **The weights over run M's 19 checkpoints** (`thud/experiments/az_input_plane_trace.cc`,
    new; the user asked to see the learning over many steps): from zero they grow every step
    (median 0.026 at step 84 → 0.170 at 102, the largest 0.04 → 0.62, still rising), by
    distance as captures occur in the games (distances 1-3 0.28 / 0.34 / 0.50, 4 0.09, 5
    ~0.005, 6-7 exactly 0 — no such capture in run M's games, so no gradient), the maps
    0.13 and 0.13; the filters grow apart (spread 0.0006 → 0.009); and the network leans
    on the planes more and more (zeroing them moves 1.4% → 16.5% of the policy's mass,
    the value by 0.03 → 0.3-0.5). No collapse. As M is no stronger than L, the network
    seems to route facts it already computed through the planes rather than to gain new
    ones.
  - **Re-evaluate the destination risk** (the user, 2026-10-11): a move's risk lies at
    its destination — for a dwarf, which moves like a queen, up to 14 squares from the
    origin where the move is scored (13 already in the opening, (0,5)-(13,5)) — and only
    the danger map holds it, there. **When:** after run M, with the blunder diagnosis
    (`az_blunder_probe`) on M's games against hexparrot, and a probe of risks by distance
    built like the long-capture probe (minimal pairs: a dwarf move of k squares onto a
    danger square or a safe one; does the prior avoid the first at every k?). **Gap if**
    the dwarfs' prior still favours unsafe moves (K83: 0.337 of its mass on 18.9% of the
    legal moves) or blunders stay above ~20% of the dwarfs' moves, and the long moves'
    risks are missed more than the short ones'. **Then:** risk planes at the origin
    ("this move lands in danger", per direction and distance — 8 x 14 for dwarfs), or a
    head scoring each move from both its ends (Leela Chess Zero's attention policy reads
    the origin and the destination squares). **First reading** (2026-10-12, after run
    M; `~/thud-runs/probe_2026-10-12/`): the blunder diagnosis on the games against
    hexparrot — blunders M102 25.4% of the dwarfs' moves (647 of 2,544), L102 25.6%
    (K83 35.9%, I83 50.2%, G'64 82.2%); the dwarfs' prior on unsafe moves 0.256 / 0.254
    against their 18.8% / 17.9% of the legal moves (K83 0.337); the trolls' prior on
    the capture inside the dwarfs' search 0.117 / 0.121 (K83 0.069), the capture
    visited in 95% / 96% of those searches; the value's change when the blunder's dwarf
    is taken −0.002 / −0.007 (its worth 0.031; K83 −0.011), a troll removed +0.045 /
    +0.059 (0.125) — **corrected 2026-10-13** (`thud/DESIGN_dwarf_signal.md` §1.4): each
    network had been measured on its own games; on the same 150 positions every line
    registers a lost dwarf at about −0.01 (I83, K83) or −0.006 (K102, L102, M100), only
    M102 at −0.002 ± 0.003, and L102 and M102 blunder alike — at this size the value
    does not drive the blunders. **Two of the three conditions hold** (blunders above
    20%, the prior still favouring unsafe moves); the third — long moves' risks missed
    more than short ones' — needs the probe of risks by distance. The dwarfs' larger
    problem is their training signal (*the dwarfs' training signal*, below). **The probe
    of risks by distance** (`thud/experiments/az_risk_probe.cc`, new; 600 dwarf positions
    of run M's last self-play games, 181,937 non-capturing dwarf moves): per distance,
    in each position, the prior's mean probability on the "bad" moves of that distance
    over that on the good ones, the geometric mean over positions (below 1: avoided; an
    untrained network gives 1.00 at every distance). Judged first by "a troll can take
    the moved dwarf next move" — too blunt: 56% of the moves, and a dwarf stepping next
    to a troll may be the better move, threatening a hurl worth 4 (M102 then 0.93 at 1
    square and 1.49 at 4) — then by **the teacher (depth 2) scoring the move below the
    best**: M102 0.68 at 1 square, **0.59 at 2-3, 0.58-0.69 at 5-9, 0.73-0.76 at 10-11,
    0.87-0.91 at 12-14** (L102 a little weaker at every distance, 0.64-0.93; K83
    0.85-1.02, and 1.19 at 1 square — it preferred them). **Long moves' risks are missed
    more** (12-14 squares; few positions: 206, 94, 15) — the third condition, weakly
    met. **Distance 4 is an anomaly for every network** (M102 1.00, L102 1.02, K83
    1.42): the "losing" moves of 4 squares get as much as the best, likely a confound —
    the teacher's 2-ply horizon counts as losing a dwarf the trolls can take even when
    a hurl then wins a troll (a bait), and 4-square moves may often be such (unchecked:
    to look at examples; baits proved rare since — after a blunder the dwarfs take a
    troll next move 0.7% of the time, against 3.6% for any dwarf move,
    `thud/DESIGN_dwarf_signal.md` §1.3 — so likely another cause). Reading: the prior
    learnt destination risk well at short and middle distances (M102 better than L102,
    perhaps the danger map), less at the longest — modest, compared with the dwarfs'
    training signal.
- **The dwarfs' training signal** (the user, 2026-10-12, asked for a design comparing
  approaches — upsides, downsides, costs of development, training and running, validation
  — reviewed adversarially by a reviewer agent: **`thud/DESIGN_dwarf_signal.md`**; revised
  2026-10-13 with the results of its measurements). **Findings:** outcome saturation — 98-100% of
  self-play games since run K's step ~77 end with every dwarf gone (M102: 162 of 162) —
  came with training against alpha-beta (run I saturated without kickstarting, 80-95% at
  steps 70-83; run H, without alpha-beta, 41-67%). **Losing every dwarf is not forced by
  the game; it is how our network plays the dwarfs**: alpha-beta at depth 2 playing the
  dwarfs keeps dwarfs alive against M102's trolls in 20 of 20 battles (−13.2) and in all
  236 logged training games against our trolls (−16.2); our dwarfs lose every one
  (self-play −25.2). **The measurements** (the night of 2026-10-13, `~/thud-runs/night7_2026-10-13.py`,
  results in `~/thud-runs/checks_2026-10-13/`; matches only):
  - **Measurement 4, the value by side to move** (`az_dwarf_value_probe`, new): a far dwarf
    removed costs the value clearly with the trolls to move (M102 +0.021, K83 +0.050; a
    dwarf 0.031) but hardly with the dwarfs to move (M102 +0.002 to +0.011); H83 (the
    clean control: run I's steps without its alpha-beta games) and G'64 have no such gap
    — it came with those games, because the only positions from games in which dwarfs
    survive are our trolls' against alpha-beta's dwarfs, all trolls to move.
  - **Measurement 1, the value at the blunders** (`az_blunder_probe`, extended): one ply down
    (trolls to move) the value prefers the teacher's safe move by a third of a dwarf. (Its
    comparison of the search's Q is biased by selection — the positions are moves the
    search chose — and shows nothing; the third review.) In self-play's conditions two
    thirds of the blunders come from the noise.
  - **Measurement 2, the teacher in the dwarfs' search at play time** (`az_match
    teacher_mix_a`, new; no noise): the dwarf side gains +7.6 / +10.6 / +15.9 points a
    battle against L102's trolls at mixes 0.3 / 0.5 / 0.7, +5.5 / +9.8 / +12.8 against
    K83's; 37-40 of 40 battles keep dwarfs; the dwarfs kill as many trolls or more; our
    trolls unchanged within the noise. **But the dwarfs survive by stalling** (38 of 40
    battles at 0.7 end at the no-capture limit, after 435 plies on average).
  - **Measurement 3, self-play's root noise** (`az_match noise=… sample_plies=10`, new; the
    noisy cell reproduces the trainer's self-play, 91 plies and −25.9): the dwarfs' own
    noise costs them 6.3 dwarfs by ply 60 and ~1.8 points a battle (its rule for C3 met),
    but without it outcomes stay saturated (1% of battles keep a dwarf); either side's
    noise breaks the stalls in which dwarfs survive without noise (26% → 0-1%).
  - **Measurement 5, older trolls under noise**: M102's dwarfs keep dwarfs in 9 of 40 battles
    against K83's trolls, margin −9.4 — all nine by killing every troll.
  - **Also found** (the third review, rerun): self-play slides within every run — run
    M's blunders 8.6% → 14.9% → 23.2% of the dwarfs' moves by thirds of the run — so a
    control run will slide too: test runs are compared with M102 as well.
  **Decided** (the user, 2026-10-13; the decision log): test runs from M102, a control
  and **D1** (the dwarfs' kickstarting weight 0.7, the trolls at 0.3); its taper slow
  and gated; **regression gates** in the promotion rule (above). **Built 2026-10-13**:
  `--player_kickstart_weight` (D1; `kickstart_test` extended, all pass) and
  `--player_policy_epsilon` / `--player_policy_alpha` (C3, the root noise per player;
  `search_settings_test` extended, all pass), both off unless set, in
  `build-shared/az_trainer_d1` for both runs; the run folders from M102's state,
  `~/thud-runs/runM_K83_conv_planes_cont` (the control, "M continued") and
  `~/thud-runs/runN_M102_d1` (run N: D1), to step 121; the runner
  `~/thud-runs/night8_2026-10-13.py control|d1`. D1's weight is capped at 0.7 (the user;
  if it is not enough, the teacher stays on longer). The win rule settled (the user,
  2026-10-14).
  **Measurement 2b** (2026-10-13 18:22-20:12; the 0.7 mix in M102's dwarf prior against plain
  M102 in self-play's conditions): under the trainer's noise no battle keeps a dwarf (0
  of 40; the dwarfs lose dwarfs more slowly, −22.0 against −25.9), with the dwarfs' roots
  noise-free 28 of 40 do (no mix: 1 of 80), mostly in stalls of ~620 plies — by its rule,
  **the test run is D1 together with C3** (run N's config: `player_kickstart_weight`
  "0.7", `player_policy_epsilon` "0,0.25"). **The test runs** (the control 2026-10-13
  20:13 to 10-14 05:29; run N 05:30 to 16:55, cut off by a Windows update and resumed;
  the design's §10): **run N wins by the rule** — N121 against C121 +5.50 a pair (+1.91
  to +9.09); the dwarf side against hexparrot −10.2 against −19.6, against alpha-beta 1
  −9.5 against −13.2; the trolls not worse; blunders against hexparrot 9.8% against
  21.6%; N121 against M102 +12.62. The value's side-to-move gap not yet closed (the gain
  is the policy's). **The replication** (run N2, a rerun of run N only — the user —
  2026-10-15 01:56-10:13, evaluated 10:16-12:33; the design's §10): **it meets the rule
  too** — N2121 against C121 +6.10 (+3.09 to +9.11), the per-side measures as N121's
  (hexparrot +8.9 ± 2.4 against the control), blunders 13.1%; N2121 against N121 +0.95
  (−2.04 to +3.94): the gain replicates. **Decided** (the user, 2026-10-15): a longer
  D1 + C3 run first — run N continued from N121, 40 steps (`~/thud-runs/runN_d1_cont`,
  from 14:30), measurement 4 every 10 steps beside the survival share. **Next:** its segments
  and evaluation; the regression gates and the promotion tests of N121 or N2121.
- **A tool to restructure a network** (the user, 2026-10-09: "I feel like we are going
  to need that repeatedly"; designed now, built when run K's results pick the first
  change of structure). **Uses:** the input features (more input channels); the
  convolutional or hybrid policy head (a new head — the linear head reads only 2
  numbers a square before its 450 x 19,800 layer, `thud/az/model.cc:406-411`, so
  per-direction facts reach the move logits poorly: likely why it learnt the dwarfs'
  hurls to 75-78% where the convolutional head reached 99%, and why the per-direction
  feature planes may pay most with a convolutional or hybrid head); a larger network
  (wider or deeper, likely on the GPU). **Precedent:** Leela Zero trained bootstrap
  networks on existing self-play games (a 128 x 10 net on 600,000 games, a 64 x 6 and a
  128 x 10 on 1.5 million; github.com/leela-zero/leela-zero issue #1197); Leela Chess
  Zero's 15 x 192 and 20 x 256 networks were bootstrapped from existing data
  (lczero.org, *Project history*); KataGo trained each larger size on the same data
  (*Notes for later*). **One program, two modes** (sharing the loading and saving):
  1. *Copy weights* — function-preserving (Net2Net, above): build the new network from
     its configuration, copy every tensor whose name and shape still fit; for the input
     layer copy the old channels and set the new channels' weights to zero, so it
     computes exactly the old network's outputs; the zero weights still learn, since
     a weight's update is its input times the error and the new planes hold real data.
     Enough for the input features. A wider trunk: Net2Net's widening (a new filter
     copies an existing one, and both get half the original's outgoing weights, plus a
     little noise so that they can drift apart) — or new filters with random incoming
     and zero outgoing weights; never zero both ways, which leaves a unit no learning
     signal. A deeper one: a new residual block whose last batch norm starts at scale and
     shift zero adds nothing to the block's input, which is already past a ReLU, so the
     network is unchanged. A new head: mode 2. **Test:** outputs identical to the old
     network's on held-out positions; the control: one planted nonzero weight on a new
     channel changes them.
  2. *Train from a replay buffer* — for what cannot be copied: load a saved buffer
     (`replay_buffer.data`), hold out a share of it (in blocks, as positions of one game
     are alike), train the new network with the trainer's loss, learning rate and
     symmetry augmentation (optionally only its new parts first, the copied ones frozen,
     then all), report the held-out loss per side every few hundred batches, stop when
     it stops falling (the overfitting measure's logic). **Tests:** the trained network
     fits held-out positions better than the untrained one (the positive control) and
     better than one trained on shuffled targets (the negative one).
  **Output:** a run folder — `config.json` with the new structure, `vpnet.pb`, the
  checkpoint as the branch's step, the buffer — that the trainer resumes like any
  branch (checked by one learning step). The buffer stores only the 6-plane
  observation and the feature planes are computed from it (`thud/az/thud_features.h`),
  so every saved buffer serves every structure. **Caveats:** mode 2 learns the old
  searches' targets — a warm start at about the old network's level, then self-play
  has to carry it further; and an offline comparison of two structures on the same
  buffer is a filter, not a verdict (the stage-5 head probe favoured the convolutional
  head, run C' then lost to run C). Its time on this CPU is measured when built.
  **Mode 2 built 2026-10-10** (the user: build it, after the long-capture probe):
  `thud/az/az_restructure.cc`, with `VPNetModel::CopyParametersFrom` (by name and
  shape) and `VPNetModel::Freeze` (no gradient; frozen leaf modules kept in evaluation
  mode while learning, so a frozen batch norm neither uses batch statistics nor
  updates its running ones) in our copy. It copies every tensor without "policy" in
  its name (the trunk and the value head, named alike in both output blocks), freezes
  them with freeze=copied, trains the new head on a saved buffer with Thud's 8
  symmetries, every 20th block of 1,024 positions held out, keeps the best by held-out
  policy loss. Tests built in, fatal on failure: the new network's values equal the
  source's before training (the copy) and after it when frozen (the freeze). Tested on
  the 4,096-position smoke buffer (16 x 1 network): 32 tensors copied, 20 parameters
  frozen, both tests exactly 0, the held-out policy loss falling 4.25 → 4.17 in 200
  batches (the source's linear head 4.24) while the held-out value loss stayed 0.7805.
  Mode 1 (net surgery for new input channels) built 2026-10-11 (`input_planes=1
  copy=all`; *the input planes, as decided*). The first use queued
  on 2026-10-10 (`~/thud-runs/convK83_2026-10-10.sh`, after K83's promotion match): a
  convolutional policy head on K83's frozen trunk, trained on run K's final buffer,
  then the long-capture probe on it and on K83. **Done** (2026-10-10 17:27-19:19,
  `~/thud-runs/convK83_2026-10-10/`, outputs in `~/thud-runs/probe_2026-10-10/`): 74
  tensors copied, 44 parameters frozen, both tests exactly 0; best at 5,500 batches
  (~1 s a batch; stopped at 6,500 by patience): held-out policy loss **3.653** (dwarfs
  4.518, trolls 2.973) against K83's linear head 3.665 (4.573, 2.952) — K83's on
  positions it trained on, so the comparison favours it — the value loss 0.0570
  unchanged (frozen). Results under the long-capture probe, below.
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
    run), and the new-against-trained measure, now per side, is in effect Leela Chess
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
  `vpnet.pb` and checkpoint; resume. Needs one end-to-end test first. (Designed
  2026-10-09 as *a tool to restructure a network*, under *How we evaluate networks*.)
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
  forgetting measurement (2026-09-30) found exactly those narrowing onto a small repertoire —
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
  1. Switched off: identical searches and games to today's (as `identity_test.cc`).
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
  7. Then no worse in the next training run (the per-side health measurement against the
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
  two entries worked out by hand); `thud/az/conv_policy_test.cc` (each action's logit
  is its plane's entry at its square on random inputs and weights, 39,600 of 39,600,
  control with rows and columns swapped 2,160 — only the 9 diagonal squares; policies
  over exactly the legal moves summing to 1; every weight gets a gradient; both heads fit
  a fixed batch, policy loss 4.8 → 0.023 in 150 steps; a checkpoint reloads exactly,
  control a fresh network differs); `identity_test` still passes 12 of 12 for the old
  head; the trainer runs with it end to end. A planted error in the map, the symmetries
  or the observation transform is caught (four mutations). **Test on the layout-probe
  harness:** `thud/experiments/az_head_probe.cc`, the layout probe's task on our copy,
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
  - **No overfitting** (the trainer's new measure): before each learning step the new
    positions' policy loss is at or below that of positions already trained on (step 10:
    4.38 against 4.51; step 16: 4.33 against 4.32), the value losses alike after step 3.
  - **Speed**: step 16 after 6.72 hours against C's 5.98, 12% slower (13-16 positions a
    second from step 3 on, C 16-18; the user asleep until ~11:00, nothing else running);
    only step 1, on the untrained network, was faster (14.0 against 12.1). The head
    probe's full-batch timing (~15% slower) was the better guide.
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
  expectation. AlphaGo Zero also evaluated positions in a randomly chosen symmetry during
  the search (the same paper); not built — a candidate if augmentation helps. **Thud's
  board has 8 symmetries, not the 16 of
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
  random-game positions); `thud/az/augmentation_test.cc` against positions mirrored
  independently as text: observations, legal moves, the policy's moves (each leads to the
  mirror image of its child) and values, 1,608 of 1,608; symmetry 0 changes nothing;
  control: untransformed observations differ in 1,379 of 1,407 (the rest: the fully
  symmetric starting position); a planted error in the policy's mapping is caught. The
  harness's `augment=1` does the same, and it measures how differently a network answers
  a position and its mirror image (value gap, policy distance). **Harness result**
  (2026-10-02 05:49-06:45, `~/thud-runs/head_check_augment.sh`: 64 x 4, both heads, seeds
  1-3, with the head probe's unaugmented runs the 2 x 2; means of 3 seeds, min-max in
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
    - The trainer's overfitting measure shows nothing here: with the large buffer, new
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
    - **Tested the same evening — confirmed for G'** (`thud/experiments/az_noise_probe.cc`,
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
        searches unchanged. **Tests** (`thud/az/pruning_test.cc`, new): five hand-built
        children against values worked out by hand (the most visited kept; one above
        the best's PUCT at once kept; one pruned to a single playout dropped to 0; one
        stopped by the PUCT bound before n_forced; an unvisited one 0), a planted error
        (the single-playout rule removed) caught; in 40 real searches (G'44, 100
        simulations, root noise) visited root children left below n_forced: 6 of 809
        with forced playouts, 62 of 807 without — at 100 simulations n_forced is only
        ~1-4 playouts, so forcing binds for few children; `identity_test` 12 of 12
        with the switch off. **Test plan** (the user asked, 2026-10-04): equal steps,
        not equal time — pruning is free and forced playouts keep 100 simulations a
        move, so a step costs the same; first the noise diagnostic on pruned targets,
        then a run of G''s settings with pruning to step 16 (one night), judged first
        by the new statistics (the dwarfs' prior moving away from uniform by steps 8-16:
        G'16 0.003 nats, C16 0.40) and then by matches against G'16.
        **The diagnostic says it does not work at our scale** (2026-10-04,
        `az_noise_probe` extended to pruned targets; the pilot's positions, two noise
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
        **More simulations do not fix it either** (2026-10-04, `az_noise_probe
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
        `thud/az/alpha_zero.h` — the learner (and the new-against-trained measure's
        trained sample) draws only from the newest W(N) positions, W = N up to the
        start, then start (1 + 0.4 ((N / start)^0.75 − 1) / 0.75); distinct samples by
        Floyd's algorithm; `growing_buffer_size` logged per step. **Tests**
        (`thud/az/growing_buffer_test.cc`, new): the formula against values worked out by hand
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
        anchor, `az_noise_probe` on W16. **Result to step 16** (2026-10-04 16:36 to
        2026-10-05 01:16; the buffer 22,036 → 153,778 positions): **the dwarfs' prior did
        not start** — effective moves within 1-2 of the legal moves at every step, its
        top move the most visited in 0.6-1.2% of dwarf positions; on the pilot's positions
        0.010 nats below uniform, two noise seeds agreeing 0% (C16 0.40 and 25%, G16 0.10
        and 2%, G'16 0.003 and 0%). **And W16 is weaker than G'16**: W16 vs G'16 −10.4
        (−14.7 to −6.1, 31 of 40 pairs lost), vs C16 −1.5 (−5.5 to +2.6); against the
        anchor W8 / W12 / W16 −2.3 / +18.4 / +25.6 — its dwarfs as G''s (+13.4 at 12 and
        16, G' +13.1, +12.7), **its trolls far behind** (+0.2, +5.0, +12.2 against G''s
        +12.2, +23.6, +22.6; p < 0.001 at each step). Ladder: W16 −16.1 ± 3.7 against E44
        (G'16 −7.1, C16 −13.1). Reading, with the caveat of one run per setting (early
        training varies between runs, and run-to-run variation is unmeasured): the pattern
        of the dwarfs' prior at step 16 — C 0.40 (small buffer, no augmentation), G 0.10
        (7x, none), W 0.010 (growing, augmentation), G' 0.003 (7x, augmentation) — points
        to **augmentation as the main brake on the dwarfs' early policy**, the buffer's
        size second; the growing buffer did not help it and coincided with slower trolls.
        Run W continues to step 29 (`stage5b_W_continue.sh`), to see whether its prior
        starts later and how W29 compares with G'29.
        **Result to step 29** (resumed 2026-10-05 01:17, step 29 at 06:42, matches until
        09:23; the user working on the laptop overnight — equal steps, unaffected):
        - **The dwarfs' prior started at step ~21**, 15-20 steps before G''s: its top move
          the most visited in 3.0%, 4.6%, 7.0%, 8.7% … 15.4% of dwarf positions at steps
          21-29 (G' at 39-44: 2.5% → 7.7%); on fresh positions its dwarf policy beats a
          uniform prior by 0.72 nats at step 29 (0.23 at 20); on the pilot's positions
          0.165 nats below uniform, two noise seeds agreeing 20% (G'29 0.013 and 0%).
        - **But W29 is far weaker**: against G'29 **−20.2** (−24.5 to −15.9, 35 of 40
          pairs lost), C29 −6.6, D29 −11.7. Against the anchor its dwarfs equal G''s at
          every step (+8.5, +14.6, +10.2 at 20, 24, 29), **its trolls fall away** — +12.2,
          +8.8, −3.2, +3.8 at steps 16, 20, 24, 29 against G''s +22.6, +14.6, +15.5, +22.8
          (p < 0.001 at 24 and 29). In its own self-play the sides became balanced (the
          dwarfs' mean return ~0 from step 22; at step 29 the trolls captured out 51 times,
          the dwarfs 24): its trolls learnt to play its own dwarfs and lost strength
          against everyone else — the drift run C showed for the dwarfs with a small
          buffer, now on the trolls' side. Steps 19-20 a crisis like G''s at 33-37 (the
          trolls' legal moves ~22 → 56; the value loss on new troll positions 0.49 against
          0.055 trained).
        - Ladder: W29 −16.9 ± 3.0, W16 −16.3 against E44; G'29 +5.5, G'44 +6.9.
        - **Reading** (one run per setting): the growing buffer starts the dwarfs' policy
          earlier but cost far more strength than it bought; **G' — a stalled dwarf
          policy, a large buffer, augmentation — is still the strongest**, its dwarfs
          carried by the value head. So the dwarfs' policy is not, at this stage, what
          limits strength (the user had wondered whether we were worrying about it too
          much: the evidence now says so, for the CPU's short runs). **G''s settings stay
          our baseline** (confirmed by the user, 2026-10-05; G' and W differ only in
          `replay_buffer_start_size`, besides their clock-seeded random numbers); the
          growing buffer is not adopted. The dwarfs' policy remains
          worth watching in the long GPU runs (G''s started by itself at ~40).
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
      what the new-against-trained measure per side is for.
    - **G' continued again, steps 45-73** (2026-10-05 14:44 to 2026-10-06 03:07; stopped
      at step 54 for the replay buffer fix and resumed at 19:56; stopped after step 73 at
      the user's request, the power adapter very hot; its evaluation not yet run). **The
      dwarfs' prior took off, then collapsed onto few moves**: its effective moves 92% of
      the legal ones at step 44, 69% at 54, 32% at 62, 5% at 72 (~14 of ~285); its top
      move the search's choice 8% → 41% → 68% → 78%; its gain over uniform on new
      positions 0.32 → 1.12 → 2.18 → 3.40 nats; the search visiting a median 20 → 16 →
      12 → 7 moves and moving the prior ever less (2.34 → 1.90 → 1.46 → 0.79 nats).
      Meanwhile the dwarfs' self-play margin rose from ~−4 to +0.6 (step 60) and fell to
      −9.3 (72); games more than doubled (238 → 540 turns a game, steps 45-48 against
      69-72), more of them at a limit (2 → 7-11 a step), the dwarfs' wipe-outs of the
      trolls 29 a step (56) → 3 (72); long hurls nearly gone (0.12 → 0.04 a game) and the
      2-move openings 97% → 70% distinct (`analyse_games.py`, 1,082 games). The trolls'
      policy sharpened too (0.08 → 0.37 nats over uniform). No value crisis (new and
      trained value losses 0.03-0.04). This looks much like run C's narrowing (its long
      hurls 0.02-0.04, openings 65-68% distinct) — a policy teaching itself to narrow,
      the search no longer correcting it — but G' has the 7x buffer and augmentation, and
      a sharper policy may simply be right: only fixed opponents can tell, per side.
    - **G'73's evaluation (2026-10-06 14:35-16:26, part 1 of
      `~/thud-runs/stage5b_Gaug_eval73.sh`; mains, no pause): from step 44 to 73 the
      dwarfs lost and the trolls gained — no overall progress — but the strength curve
      below shows a peak at step 64 in between.**

      | G'73 as trained against | Pairs | Per pair (95%) | W / D / L | As dwarfs / as trolls | G'44 against the same: per pair; as dwarfs / as trolls |
      |---|---|---|---|---|---|
      | G'44 | 40 | **+4.70** (+1.89 to +7.51) | 27 / 1 / 12 | −3.33 / +8.03 | — |
      | E44 | 40 | +0.12 (−3.71 to +3.96) | 19 / 1 / 20 | −8.60 / +8.72 | +7.97; −2.40 / +10.38 |
      | A14 | 40 | +3.10 (+0.62 to +5.58) | 27 / 0 / 13 | −6.90 / +10.00 | +3.23; −3.52 / +6.75 |
      | the anchor A0 | 20 | +26.55 (+23.97 to +29.13) | 20 / 0 / 0 | +1.55 / +25.00 | +31.50; +12.10 / +19.40 |

      Per side against the same fixed opponents (Welch): **the dwarfs worse every time**
      — against E44 −6.20 (p = 0.006), A14 −3.38 (p = 0.015), A0 −10.55 (p < 0.001); the
      trolls better against A14 +3.25 (p = 0.006) and A0 +5.60 (p < 0.001), even against
      E44 (−1.65, p = 0.17). The noise diagnostic: root noise no longer decides the
      dwarfs' searches — two seeds' most visited agree 78% (G'44 6%, E44 49%), the prior
      1.47 nats below uniform (G'44 0.08, E44 0.94); the trolls' 16% and 0.30. **Ladder:
      G'73 +4.8 ± 2.3** (as dwarfs −2.6, as trolls +7.4) against G'44's +4.4 ± 2.2 (+1.4,
      +3.0) and G'29's +5.1 ± 3.0 — the same within the errors. The fit flags G'73's
      matches as non-transitive: against G'44 observed +4.7, predicted +0.4; against E44
      +0.1, predicted +4.6 — its trolls beat G'44's dwarfs in particular. The promotion
      rule as it stood was met (G'73 beat the top anchor G'44 in 40 pairs, interval
      above 0) while the ladder showed no overall gain and weaker dwarfs: **G'44 stays the
      top anchor, and the rule now also needs a higher ladder rating** (the user,
      2026-10-06; *How we evaluate networks*) — G'73 − G'44 +0.4, −2.0 to +2.8.
      So the dwarfs' policy, once it began to learn (step ~40), narrowed past what helps:
      the search stopped correcting it (KL 2.34 → 0.79 nats), much like run C's drift
      but with the 7x buffer and augmentation.
    - **The strength curve** (part 2, 2026-10-06 16:54-19:38, the evaluation watcher's
      first real use: each checkpoint against G'44 and A14, 20 pairs each; the ladder
      refitted with every match, G'44 now +3.2 ± 2.1):

      | Checkpoint | Rating (95%) | As dwarfs | As trolls | The dwarfs' prior, share of legal |
      |---|---|---|---|---|
      | G'44 | +3.2 ± 2.1 | −0.4 | +3.6 | 92% |
      | G'48 | +5.0 ± 4.0 | −2.6 | +7.6 | 83% |
      | G'56 | +5.6 ± 3.9 | +1.5 | +4.0 | 62% |
      | **G'64** | **+10.4 ± 3.9** | **+5.2** | +5.2 | 20% |
      | G'72 | +6.1 ± 3.8 | +3.5 | +2.6 | 5% |
      | G'73 | +4.4 ± 2.3 | −2.5 | +6.9 | ~5% |

      **G'64 is the peak**: G'64 − G'44 +7.2 (+3.7 to +10.8), G'64 − G'73 +6.0 (+2.0 to
      +10.1); its neighbours borderline (G'64 − G'56 +4.9, −0.0 to +9.8; G'72 − G'64
      −4.4, −9.2 to +0.5). So the dwarfs' narrowing first helped — at step 64 their prior
      covered ~20% of the legal moves and agreed with the search 67% of the time, the
      best dwarfs of the run — and then hurt, by step 72-73 (5%, 78%). G'64 meets the
      tightened rule's ladder test; its head-to-head with G'44 (+9.6) has only 20 pairs
      (the rule needs 40). Caveat: the per-side numbers of the watcher's checkpoints
      rest on two opponents and 20 pairs each — G'72's dwarfs +3.5 against G'73's −2.5
      one step later shows how noisy they are; the overall ratings are firmer.
    - **G'64 confirmed** (2026-10-07 02:28-04:14; 40 pairs each, as trained): against
      G'44 with a new seed **+8.60 (+5.87 to +11.33), 32 / 1 / 7** — as dwarfs +3.25, as
      trolls +5.35; against E44 +2.58 (−0.82 to +5.97), 24 / 2 / 14 — as dwarfs −4.58, as
      trolls +7.15. Against E44, per side (Welch): its dwarfs better than G'73's (+4.02,
      p = 0.025) and not significantly different from G'44's (−2.18, p = 0.31), its
      trolls worse than G'44's (−3.22, p = 0.021). Ladder (refitted, 3,000 games):
      **G'64 +7.4 ± 2.3** (dwarfs +1.5, trolls +5.8), G'72 +4.8, G'73 +3.6, G'44 +1.6;
      **G'64 − G'44 +5.8 (+3.7 to +7.9)**. So **G'64 passes both tests of the tightened
      promotion rule — promoted to top anchor** (the user, 2026-10-07). E44 stays non-transitive with the G'
      family: G'44 beats it by more than the ratings predict (+8.0 against +1.6), G'64
      by less (+2.6 against +7.3).
    - **Run H** (2026-10-07 04:14-11:00, steps 65-83, mains, no pause; G' branched at
      step 64 with the dwarfs' policy softmax temperature 1.5): **the dwarfs' collapse
      did not happen** in 19 steps, where G' collapsed within 8 from the same start —
      the dwarfs' prior stays at 13-34% of the legal moves (mostly 15-25%; G' 18% → 5%
      by step 73), its top move the search's choice 39-57% (G' 71% → 78%), the search
      moving it 1.46-1.86 nats (G' 1.18 → 0.79), their searches over 10.4-12.1 effective
      moves (G' 5.7 → 4.2), their self-play margin −3.2 to +1.5 (G' −1.9 → −9.3), the
      trolls wiped out in 15-31 games a step (G' 14 → 3-5). Self-play ~12% slower (~17.5
      positions a second against ~20; wider searches find fewer positions in the
      cache, likely). Its checkpoints are evaluated as trained: `az_match` takes
      `temperature_a` / `temperature_b` (built 2026-10-07; test: `1,1` plays the same
      games as none, `1.5,1` different ones — the control), the evaluation watcher reads
      each run's temperature from its `config.json`, and the ladder counts a network
      searching with another temperature than its run's as another player ("H72
      [temperature none]"). **Result** (`~/thud-runs/eval_runH.sh`, 2026-10-07
      12:14-16:28: the watcher on H68-H80 against G'64 and G'44, 20 pairs each; H83
      against G'64, 40 pairs): **run H is stronger at every step, and it is the dwarfs.**
      Ladder (refitted, 49 matches; against E44): H68 +14.9 ± 3.8, H72 +15.9, H76 +12.4,
      H80 +18.7 ± 3.9, H83 +18.7 ± 4.1 — where G' from the same start fell (G'64 +6.5 ±
      2.1, G'72 +5.0, G'73 +3.7). As dwarfs H68-H83 +9.2 to +14.2 (H83 +14.2 ± 2.9)
      against G'64's +3.3, G'72's +3.2, G'73's −2.6; as trolls +3.2 to +6.9, G''s +1.7 to
      +6.3 — no clear difference. H76's dip is within the noise (± 3.8). **H83 against G'64: +11.75
      a pair (+9.0 to +14.5), 37 of 40 pairs**, as dwarfs +10.55, as trolls +1.2; ladder
      difference **H83 − G'64 +12.2 (+8.7 to +15.6)**, H80 − G'64 +12.1 (+8.6 to +15.7).
      **H83 meets both tests of the promotion rule — promoted by the user** (2026-10-07
      evening; its rating rests on G'64 alone, H80's on G'64 and G'44). Proposed: **run H the main line**
      (no collapse, its dwarfs ~+10 stronger on our ladder) — with the caveat of the same
      day's outside opponents (*How we evaluate networks*): the ladder measures our
      networks against each other, and how H83 fares against players without a network
      is being measured the same night (`~/thud-runs/outside_2026-10-07.py`, hexparrot
      first).
    - **Training the dwarfs better** (discussed with the user, 2026-10-06; sources: Leela
      Chess Zero's `--policy-softmax-temp`, default 1.36, "higher values make priors
      closer to each other, widening the search"; Danihelka et al., ICLR 2022; MiniZero,
      arXiv 2310.11305: with 200 simulations plain AlphaZero matched or beat the Gumbel
      variants on 9x9 Go and Othello, Gumbel wins with few, and "progressive
      simulation" — more simulations as training goes on — helped): the collapse fits
      too few simulations for a confident prior. Order agreed: (1) per-side search
      settings in our copy — **built 2026-10-06**: `--player_policy_temperature`
      (Leela's policy softmax temperature per player, at every node of that player,
      before root noise; `MCTSBot::SetPolicyTemperatures`) and `--player_simulations`
      (self-play simulations per player), both recorded in `config.json`, off by
      default; checked by `thud/az/search_settings_test.cc` (temperature 1 changes no
      visit; 2 flattens the dwarfs' root priors and not the trolls'; the trainer's
      searches get their settings; old configs load). (2) The search-only filter on
      G'64's positions (100 / 200 / 400 / 800 against 4,000 and 16,000, with the binary
      of G'44's filter, so the two compare directly) — how much more search changes G'64's
      own targets; not the temperature (its reference would share the prior's bias).
      Stopped on 2026-10-07 at 91 positions: with 32 threads and the 16,000 reference
      it held 8.0 GB and, beside the confirmation matches, filled RAM and swap; rerun
      alone with 16 threads (2026-10-07 11:30-12:13, 143 dwarf and 144 troll positions,
      `target_quality_Gaug64.jsonl`). **Result: more search barely moves the dwarfs'
      choice; it helps the trolls more.** Against 16,000 simulations the dwarfs' most
      visited move agrees 28.0% at 100, 28.7% at 200, 30.1% at 400, 32.2% at 800 (over
      100: +0.7, +2.1, +4.2 points, none significant — 800's interval −0.9 to +9.3); the
      distance falls 0.641, 0.616, 0.577, 0.527 (every step significant, 800 −0.114,
      −0.136 to −0.092; no flattening by 800); the mass on the reference's best falls
      0.174 → 0.146 (more simulations spread the visits of a sharp prior the reference
      mostly shares). The trolls: 6.9%, 14.6%, 15.3%, 19.4% (+7.6, +8.3, +12.5 points,
      200 and up significant), distance 0.413 → 0.324. Against G'44's filter: the
      dwarfs' agreement at 100 rose from 5.4% to 28.0% — the sharper prior, not the
      search, made that difference — while what 4x more search adds stays small (G'44
      +4.5, G'64 +2.1 points at 400). The reference has not settled: 4,000 against
      16,000 agree 48% for the dwarfs (G'44 52%), 62% for the trolls (68%); against
      4,000, 800 gains the dwarfs +9.8 points (+3.2 to +16.3), but that reference shares
      more of 100's errors. A limit made concrete the same day: the reference is the
      same network searching longer and shares its blind spots — against players
      without a network G'64 loses material it never sees (*How we evaluate networks*,
      outside opponents) — so the filter measures closeness to the network's own deep
      search, not to good play.
      (3) **Run H**: G' branched at step 64 with the dwarfs' temperature 1.5, its
      control G''s own steps 65-73; the buffer as it was at step 64 rebuilt from step
      73's by `thud/experiments/az_buffer_truncate.cc` (259,963 of 458,752 positions
      kept, the 198,789 added after step 64 dropped; its control: a loaded and re-saved
      buffer byte-identical to the trainer's file). (4) If the filter shows more
      simulations clearly help: a branch with 400 dwarf simulations — **not shown**
      (2026-10-07: 400 and 800 bring the dwarfs' choice no significantly closer to a
      deep search; the trolls' gain more), so no such branch for now. On the GPU:
      playout caps with deep dwarf searches, raised over training; tree reuse once
      searches are deep; Gumbel AlphaZero only if the dwarfs' budget has to stay
      small.
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
- It hurts learning (`thud/experiments/az_layout_probe.py`): OpenSpiel's own model and
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

### A first, smaller PR: the replay buffer's memory (agreed with the user, 2026-10-05)

Generic, no Thud content, so none of the copyright steps above apply: upstream's
`CircularBuffer::Add` copies into full slots, which keep their largest size, and
`SerializableCircularBuffer::LoadBuffer` leaves spare capacity (*Instrumentation*, the
buffer's memory). Upstream's guide calls bug fixes "straightforward to be included" and asks
for an issue first only for large or design changes, so a PR directly. None of the three
files has changed upstream since our copy (checked against upstream `master`, 2026-08-31).

1. **The Google CLA**, signed by the user for the GitHub account `lemony-fresh`
   (<https://cla.developers.google.com/>), covering the commit address.
2. **Commits only as `lemony-fresh <6112333+lemony-fresh@users.noreply.github.com>`**, the
   GitHub noreply address (the personal one is never to appear in a public commit: the fork's
   32 commits were rewritten to the noreply address on 2026-10-05).
3. **A branch off upstream `master`**, not `thud` — **prepared 2026-10-06** (branch
   `replay-buffer-memory`, a git worktree in `~/open_spiel-buffer-memory`, not committed;
   the draft description, for the maintainers, in `~/open_spiel-buffer-memory_PR.md`):
   `open_spiel/utils/circular_buffer.h` — `Add(T value)`, moved into its slot (the
   pass-by-value idiom for a stored argument: every caller is fixed without a change, an
   lvalue copied into the parameter as before, so upstream's `alpha_zero.cc` needs none);
   `serializable_circular_buffer.h` — `LoadBuffer` replaces each element by a copy of
   itself (libnop reads vectors of non-integral elements by `push_back`, deliberately
   without `reserve()`, `nop/base/vector.h`); a test in each of `circular_buffer_test.cc`
   (a slot replaced by a small vector keeps less than the large one's capacity) and
   `serializable_circular_buffer_test.cc` (a loaded vector of 1,025 floats keeps less than
   1.5 times its size) — no exact-capacity assertions (implementation-dependent). **Checked
   so far** (compiled directly against `libopen_spiel.so`): both test programs pass with
   the change, and **fail at exactly the new tests without it** (the controls), and run
   cleanly under AddressSanitizer and UndefinedBehaviorSanitizer.
4. **Tests — done 2026-10-06** (the worktree with its own CMake builds, upstream's default
   "Testing" one and a Release one with LibTorch, 6 jobs): `circular_buffer_test` and
   `serializable_circular_buffer_test` pass through `ctest`; so do `torch_integration_test`,
   `torch_model_test`, `torch_vpnet_test` and `dqn_torch_test` (DQN, the buffer's third
   user upstream); and upstream's `alpha_zero_torch_example`
   trains tic-tac-toe 2 steps (its buffer of 512 full at step 2) and resumes from its
   `config.json` to step 3 (loading the buffer, then overwriting full slots), exit 0 both.
5. **The user reviews** the diff and the description (our measurements, in our own words);
   push and PR only on the user's go-ahead. **Done 2026-10-06: PR
   [#1637](https://github.com/google-deepmind/open_spiel/pull/1637)** (commit `0c73c409`,
   only the noreply address; no co-author line, which Google's CLA check would also
   check). The CLA check (`cla/google`) passed at once: the user's CLA, signed with their
   Google account and the GitHub username, covers commits with the noreply address. Next:
   upstream's CI (`check-changes`) and the maintainers' review; merged in batches, at
   least every two weeks.

**Upstream's state** (checked 2026-10-17, the user asked): merges come in batches — 05-12,
05-22, 07-13 (a large one), 07-16/17, 08-11/12, 08-25, 08-31 — with releases 2.0, 2.0.1 and
2.0.2 in July and August; none since 08-31. On 2026-09-18 Marc Lanctot, OpenSpiel's lead,
announced he had left Google DeepMind and lost admin rights, handing maintenance to John
Schultz (@jhtschultz) and Zun Li (@rezunli96) (issue #1621; accepted 09-21). Since then no
merge, and on 10-01 Lanctot wrote that he "can't import PRs anymore" (PRs are imported
into Google's repository and synced out, hence the batches). Open PRs are piling up (51:
17 opened in September, 11 in October so far). So: maintained, in a handover — our #1637
and #1646 wait for the new maintainers' first batch. **Decided (the user):** wait a few
weeks, then ping them — around 2026-11-07 if nothing has moved.

### Further upstream candidates (2026-10-15)

Found while building the GPU checklist's features; each checked in upstream's own code
(`upstream/master`, 48401890).

1. **`json::FromString`'s comment** (`open_spiel/utils/json.h:161`) says it returns
   `nullopt` on a parse failure; `ParseError` in `json.cc` deliberately stops the program
   instead (its own comment says so). A PR fixing the comment was prepared (branch
   `json-fromstring-doc`, worktree `~/open_spiel-json-doc`, one commit with the noreply
   identity, the description in `~/open_spiel-json-doc_PR.md`); **the user decided not to
   open it** (2026-10-15). Nothing pushed.
2. **`SerializableCircularBuffer::LoadBuffer` ignores its reads' status**
   (`open_spiel/utils/serializable_circular_buffer.h:55-64`): a file cut short — which
   upstream's AlphaZero leaves when killed while it overwrites the buffer in place
   (`alpha_zero.cc:403`; a second Ctrl-C exits at once, `alpha_zero_torch_example.cc:90`)
   — loads without an error, only the positions before the cut, its count of positions
   added still the file's, which breaks the slots' order (`replay_buffer_test`, part 5,
   on upstream's class). A clear bug; no issue or PR upstream. **Decided (the user):** a
   PR of its own after #1637 is merged (#1637 changes the same function), with the
   read validation of our `ReplayBuffer::LoadBuffer` and a test.
3. **The legal policy** (our `VPNetModel::SetLegalPolicy`): upstream's AlphaZero masks and
   normalises the policy over every distinct action (`model.cc:210-211, 259-260, 335`,
   `vpnet.cc:161, 199-214`) — a general design whose cost grows with the action count
   (Thud 19,800, chess 4,674, Go 362, Hex 121). An optimisation, not a fix; measured here
   only with networks cheap to evaluate (chess, a tiny network: +49%), none with a real
   one on this CPU (chess 64 x 4: ±0). Revisit after GPU measurements.
4. **The learner's batch norms on a shared device** (found 2026-10-17, the design's §10,
   *E1's offline sweep*): `VPNetModel::Learn` sets the network to training mode
   (`vpnet.cc:270`) and `Inference` to evaluation mode (`:191`), on one shared module;
   `DeviceManager`'s mutex guards only its count of requests (`device_manager.h:71, 97`),
   and with one device `explicit_learning` must be off (`alpha_zero.cc:579`), so the
   inference threads serving the actors use the learner's network while it learns, and
   set it to evaluation mode during its steps. Reproduced on our copy (the same code):
   N161's training-mode losses on batches of its buffer, value 0.12-0.14, fall to
   0.023-0.027 with one inference thread beside the steps — the level our trainer
   reported throughout; so the batch norms normalise mostly by their running statistics
   while learning, and those are rarely updated. The batch norms' own counters
   (`num_batches_tracked`, one per batch in training mode) agree: in run N's step 161 the
   input batch norm counted 7 of the 64 learning batches, the deeper ones 11-51, rising
   with depth — an inference call under way when the learner switches to training
   apparently finishes its deeper layers in training mode, normalised by its few
   positions' statistics (run H's step 65 alike: 16-53). A real bug (the step's mode is not the one
   `Learn` sets; the parameters are also written while inference reads them). **Fixed in
   our copy** (2026-10-17: `thud/az/vpnet.h` — a reader-writer lock, the network kept in
   evaluation mode outside learning, each learning step's counts checked;
   `batch_norm_test` has upstream's class as its control). **Decided (the user):** a PR
   with a reproduction on Connect Four, prepared in a separate worktree once the runs are
   going; reviewed by the user before anything is pushed. **Prepared 2026-10-17**: branch
   `vpnet-batch-norm-race` (worktree `~/open_spiel-bn-race`, commit `cb154b7b`, the noreply
   identity, not pushed; the description `~/open_spiel-bn-race_PR.md`): a reader-writer
   lock in `vpnet.h`/`vpnet.cc` (Inference shared, Learn and LoadCheckpoint exclusive, the
   model in evaluation mode outside Learn), and `TestLearnUnaffectedByConcurrentInference`
   in `vpnet_test.cc` (Connect Four: two copies learn the same batches, one beside a thread
   calling Inference; equal losses required) — failing on unchanged upstream at the first
   step in 3 runs of 3, passing with the fix in 3 of 3; `torch_model_test`,
   `torch_integration_test` and the tic-tac-toe example pass. The comments shortened at
   the user's review (commit `cb154b7b`). **Opened 2026-10-17 as
   [#1646](https://github.com/google-deepmind/open_spiel/pull/1646)** (the user's
   go-ahead; only the noreply identity, no co-author line): the CLA check passed;
   upstream's CI (`check-changes`) and the maintainers' review next.

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
