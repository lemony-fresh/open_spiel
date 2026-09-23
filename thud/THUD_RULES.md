> # DO NOT UPSTREAM THIS FILE
>
> **THIS FILE QUOTES THE OFFICIAL THUD RULES VERBATIM AND MUST NEVER BE INCLUDED IN A PULL
> REQUEST TO `google-deepmind/open_spiel`, OR IN ANY OTHER PUBLIC DISTRIBUTION.**
>
> Thud is commercially published by Trevor Truran / The Cunning Artificer. Game *mechanics*
> are not copyrightable, but the *expression* of them is, and this file reproduces that
> expression directly.
>
> Anything upstream-facing — header comments, a `docs/games.md` entry — must be written from
> scratch in our own words, never copied from here. See the pre-PR checklist in
> `thud/PLAN.md`.

# Thud — rules specification

The exact ruleset we implement. Where the published rules are ambiguous, the ambiguity is
recorded and a decision made explicitly. **Implement this file, not recollection of Thud.**

Sources: the official rules as reproduced on BoardGameGeek and in the Tabletop Simulator
workshop edition, cross-checked against `wiki.lspace.org/Thud` and `spaxegames.wordpress.com/thud`.
The official PDF at `tesera.ru/images/items/1543265/THUD_RULES.pdf` returns HTTP 403; the
lspace wiki's prose account is loose and wrong in at least one place (it implies trolls hurl).

---

## 1. Board

15x15 grid, rows and columns indexed `0..14`, with a 15-square triangle removed from each
corner, giving an octagon of **165 squares**.

Playable columns per row:

| Row | Cols | Row | Cols |
|---|---|---|---|
| 0 | 5–9 | 8 | 0–14 |
| 1 | 4–10 | 9 | 0–14 |
| 2 | 3–11 | 10 | 1–13 |
| 3 | 2–12 | 11 | 2–12 |
| 4 | 1–13 | 12 | 3–11 |
| 5 | 0–14 | 13 | 4–10 |
| 6 | 0–14 | 14 | 5–9 |
| 7 | 0–14 | | |

`5+7+9+11+13 + 15x5 + 13+11+9+7+5 = 165`.

The **Thudstone** occupies the centre square `(7,7)` for the whole game. It is not a piece,
belongs to neither player, and may never be moved onto or through.

## 2. Setup

**Trolls (8)** — the eight squares orthogonally and diagonally adjacent to the Thudstone:
`(6,6) (6,7) (6,8) (7,6) (7,8) (8,6) (8,7) (8,8)`.

**Dwarfs (32)** — every square on the octagon's perimeter *except* the four in the same row
or column as the Thudstone. The perimeter is 36 squares:

- straight edges: row 0 cols 5–9; row 14 cols 5–9; col 0 rows 5–9; col 14 rows 5–9 (20)
- diagonal edges: `(1,4)(2,3)(3,2)(4,1)`, `(1,10)(2,11)(3,12)(4,13)`, `(10,1)(11,2)(12,3)(13,4)`, `(10,13)(11,12)(12,11)(13,10)` (16)

Excluded: `(0,7) (7,0) (7,14) (14,7)`. Hence `36 - 4 = 32`.

**Dwarfs move first.**

## 3. Directions

Eight directions throughout: the four orthogonals and the four diagonals. "Line" always
means a maximal run of same-type pieces on consecutive squares along one direction.

## 4. Dwarf turn — exactly one of

### 4a. Move

Choose a dwarf and a direction; move it `1..k` squares. Every square traversed **and** the
destination must be empty — not a piece, not the Thudstone. A dwarf never captures by moving.

### 4b. Hurl

Let a line of `N` dwarfs run in direction `d`. The **front** dwarf (the one furthest along
`d`) is hurled along `d` onto a square occupied by a troll, provided:

- every square strictly between the front dwarf and that troll is empty, and
- the number of such intervening squares is **< N**.

Equivalently, the hurled dwarf travels `1..N` squares. The dwarf leaves its origin, lands on
the troll's square, and that troll is captured. A hurl captures **exactly one** troll and may
only land on a troll — never on an empty square.

## 5. Troll turn — exactly one of

### 5a. Move

Choose a troll and a direction; move it **exactly one** square onto an empty square (not the
Thudstone). Having moved, it **may** capture any subset of the dwarfs on the eight squares
adjacent to its destination — including none. Capturing is not compulsory.

### 5b. Shove

Let a line of `N` trolls run in direction `d`. The **endmost** troll (furthest along `d`) is
shoved `1..N` squares along `d`, provided every square traversed and the destination is
empty. Having landed, it captures any subset of the dwarfs adjacent to its destination.

**A shove is legal only if it captures at least one dwarf.** A shove that would capture
nothing is not a legal move. Confirmed by implementation: `dstu/thud` returns `None` when the
capture count is zero (`if i == 0 { None }`), and `hexparrot/thudgame` returns no moves
(`if not capturable: return []`).

## 6. End of the battle

Officially: "when both players agree that no more captures can be made by continuing to play,
or when one player has no more valid moves to make."

## 7. Scoring

Surviving dwarfs score **1 point each** for the dwarf player; surviving trolls score
**4 points each** for the troll player. The result is the difference.

Maxima are `32` and `8 x 4 = 32`, so the game is naturally balanced. A full match is **two
battles with the sides swapped**; the higher two-battle total wins.

---

## 8. Ambiguities and our decisions

These are the points where the published rules do not determine behaviour. Each needs a
decision before the action encoding is designed; **items 1 and 4 must be settled first**
because they change the action space.

Each item records how three existing implementations resolved it, which settles two of the
four outright:

| Implementation | Language | Notes |
|---|---|---|
| [`dstu/thud`](https://github.com/dstu/thud) | Rust | Has MCTS; most carefully structured |
| [`THFlowers/Thud-CLI`](https://github.com/THFlowers/Thud-CLI) | Java | Has MCTS |
| [`hexparrot/thudgame`](https://github.com/hexparrot/thudgame) | Python | Has AI engine and self-play |

**Contested rules should be OpenSpiel game parameters**, declared in
`parameter_specification` rather than hard-coded. That turns items 1–3 from arguments into
experiments, and costs almost nothing to build in up front.

### 1. Do trolls choose which dwarfs to capture? — affects action space

The rules say a troll "may" capture "any (all)" adjacent dwarfs and that "capturing is not
compulsory". Read literally, the troll player chooses an arbitrary subset of up to 8 adjacent
dwarfs: 256 possibilities per landing square.

Declining a capture is genuinely strategic in Thud (a dwarf left alive can shield the troll,
and captures expose the capturing troll), so "capture all" is a real rule change, not a
simplification of a dead option.

**Options:**
- **(a) Faithful, as a second decision node.** Apply the move, then have the same player
  choose captures — legal actions become "capture the dwarf at square X" plus "done". Reuses
  the square encoding, keeps the action space small, and is exactly correct. Cost: roughly
  doubles tree depth on capturing moves.
- **(b) Capture all, mandatory.** Simplest and smallest tree; a documented deviation from the
  rules.

**Implementations are genuinely split, so this one stays a decision:**
- `THFlowers/Thud-CLI` — **chosen subset**, and implemented exactly as option (a): after a
  move or shove it sets `turn.setRemoveTurn(true)`, and `removePlay()` then takes the
  player's explicitly listed positions via a separate `R` command.
- `hexparrot/thudgame` — **capture all**: `capturable = self.tokens_adjacent(dest, 'dwarf')`
  is captured wholesale.
- `dstu/thud` — **unclear from a first reading.** `Action::Shove` carries a capture count and
  a `[Coordinate; 7]` list, but generation appears to emit one action per `(start, end)` with
  the captured set already computed, which reads more like capture-all than a choice. Needs a
  closer look before being counted as evidence either way.

**Option (a) is idiomatic OpenSpiel, not a workaround.** `open_spiel/games/amazons/` is an
in-tree precedent: Amazons has a three-phase turn (move a piece, then shoot an arrow) built
as `enum MoveState { amazon_select, destination_select, shot_select }`, where
`CurrentPlayer()` simply returns `current_player_` and `DoApplyAction` only flips it in the
final phase:

```cpp
case shot_select: {
  board_[shoot_] = CellState::kBlock;
  current_player_ = 1 - current_player_;
  state_ = amazon_select;
}
```

`LegalActions()` switches on `state_` to return the right actions per sub-phase. Nothing in
OpenSpiel requires players to alternate — a multi-phase turn is a supported pattern with a
working reference implementation to copy.

**Settled: (a) — a separate dwarf-removal phase.** Faithful to the rules, idiomatic in
OpenSpiel, and `THFlowers` shows it works in practice. Only tree depth suffers, not the
action space. Revisit only if Phase 5 shows depth is the bottleneck.

Concretely: after a troll move or shove resolves, if any dwarfs are adjacent to the
destination, the state enters a removal phase with the **same player** still to act. Legal
actions are "remove the dwarf at square X" for each adjacent dwarf, plus "done". After a
*move* the "done" action is available immediately (capturing is not compulsory); after a
*shove* at least one dwarf must be removed before "done" becomes legal, since a shove is only
legal if it captures.

### 2. Can a single dwarf hurl itself? (`N = 1`)

"Anywhere there is a line of adjacent dwarfs" suggests `N >= 2`. But with `N = 1` the
condition "intervening squares < 1" permits a lone dwarf to capture an **adjacent** troll.
Implementations differ.

**Implementations lean toward requiring `N >= 2`, 2 to 1:**
- `dstu/thud` — **requires `N >= 2`**. Hurl generation walks a ray forwards while walking the
  reverse ray backwards, and bails immediately with `if !self[previous].is_dwarf() { return
  None }` — so the square *behind* the dwarf must hold a dwarf, and a lone dwarf fails at the
  first step.
- `hexparrot/thudgame` — **requires `N >= 2`**.
- `THFlowers/Thud-CLI` — **allows `N = 1`**, and deliberately: `distanceAttackCheck()` throws
  `"Shove must be at least 2 trolls"` only when `numInLine == 1 && turn == TROLL`. The
  troll-only guard shows the asymmetry was intentional, not an oversight.

**Settled: require `N >= 2`.** The decisive argument is textual. The official rules phrase the
two moves *identically* — "anywhere there is a straight (orthogonal or diagonal) line of
adjacent **trolls/dwarfs** on the board, they may **shove/hurl**". Identical phrasing must be
read identically, and the `N >= 2` requirement for shove is unanimous across implementations
(item 3). So hurl requires `N >= 2` too, and `THFlowers` is simply wrong here.

Supporting: allowing `N = 1` would let any dwarf take any adjacent troll, gutting the game's
central asymmetry — trolls capture by contact, dwarfs must strike at range — and making rule
4a's "a dwarf never captures by moving" nearly vacuous.

### 3. Can a single troll shove? (`N = 1`)

With `N = 1` a shove is a one-square move that captures — i.e. rule 5a. Treating it as a
shove would duplicate 5a and make troll captures effectively mandatory.

**Settled: require `N >= 2`.** All three implementations agree, one of them with an explicit
error message to that effect (`"Shove must be at least 2 trolls"`), and `dstu/thud` enforces
it structurally by requiring the square behind the shoved troll to hold a troll.

### 4. Termination — the official condition is not implementable

"Both players agree" cannot be evaluated, and dwarfs can shuffle indefinitely. Note that in a
margin-scored game the player who is ahead wants to stop and the player behind does not, so
mutual agreement is not a mechanism that survives contact with self-play.

**The two implementations take opposite approaches, and one of them is a warning:**
- `hexparrot/thudgame` — a **no-progress cap plus a stalemate check**:
  `DEFAULT_MAX_PLIES = 400` produces a `'cutoff'` result, and `has_legal_move(...)` returning
  false ends the game as a no-move stalemate. Tellingly, their cutoff is annotated
  *"self-play only"* — i.e. added for exactly our use case.
- `dstu/thud` — models the **agreement** faithfully: *"A Thud game traditionally ends when
  both players agree that it should end. This is implemented as a proposal/counter-proposal
  process"*, with a `Decision` enum of `Accept`/`Decline`. Faithful, but it adds actions to
  the space and inherits the problem above: under self-play the trailing player simply never
  accepts, so games would run to whatever cap exists anyway.

**Implement `hexparrot`'s approach.** The battle ends when **either**

- the current player has no legal move (this *is* an official condition), **or**
- `K` consecutive plies have passed with no capture — start with `K = 50` and tune, **or**
- a hard `MaxGameLength` cap is reached (required by OpenSpiel regardless). `hexparrot`'s
  400 plies is a reasonable starting anchor.

Terminal utility is the score difference, normalised to `[-1, 1]` by dividing by 32. This
gives a richer learning signal than win/loss, and 32 is the exact maximum margin.

### 5. Whether a hurl may use a sub-line

If five dwarfs are in a line, may the player treat them as a line of three to hurl a shorter
distance? Since `N` sets only an upper bound on distance (`1..N`), a shorter throw is already
legal from the maximal line, so **use the maximal line** and the question is moot. Noted only
so it is not rediscovered.

### 6. Match structure

We model **one battle** as one OpenSpiel game, not the two-battle match. Score-difference
utility already captures the margin that the match structure aggregates, and side-swapping is
an evaluation-harness concern, not a game-rules one.
