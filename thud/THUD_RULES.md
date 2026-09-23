> # DO NOT UPSTREAM THIS FILE
>
> **THIS FILE QUOTES THE OFFICIAL THUD RULES VERBATIM AND MUST NEVER BE INCLUDED IN A PULL
> REQUEST TO `google-deepmind/open_spiel`, OR IN ANY OTHER PUBLIC DISTRIBUTION.**
>
> Thud is commercially published by Trevor Truran / The Cunning Artificer. Game *mechanics*
> are not copyrightable, but the *expression* of them is, and this file quotes that
> expression. Anything upstream-facing — header comments, a `docs/games.md` entry — must be
> written from scratch in our own words. See the pre-PR checklist in `thud/PLAN.md`.

# Thud — rules specification

The exact ruleset and action encoding we implement. **Implement this file, not recollection
of Thud.** Sections 1–8 are the specification; section 9 explains the non-obvious choices.

---

## 1. Board

A 15×15 grid of squares `(row, col)`, both `0..14`, row 0 at the top ("north"). A triangle
of 15 squares is cut from each corner, leaving an octagon of **165 squares**:

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

The **Thudstone** stands on `(7,7)` for the whole game. It belongs to neither player, never
moves, and no piece may land on it or pass over it.

**Adjacent** means one of the up to 8 neighbouring squares, orthogonal or diagonal. Every
move runs in a straight line along one of the 8 directions.

## 2. Setup

**Trolls (8):** the 8 squares adjacent to the Thudstone —
`(6,6) (6,7) (6,8) (7,6) (7,8) (8,6) (8,7) (8,8)`.

**Dwarfs (32):** all 36 perimeter squares except the 4 in line with the Thudstone,
`(0,7) (7,0) (7,14) (14,7)`:

- straight edges: row 0 cols 5–9; row 14 cols 5–9; col 0 rows 5–9; col 14 rows 5–9 (20)
- diagonal edges: `(1,4) (2,3) (3,2) (4,1)`, `(1,10) (2,11) (3,12) (4,13)`,
  `(10,1) (11,2) (12,3) (13,4)`, `(10,13) (11,12) (12,11) (13,10)` (16)

## 3. Turns

Player 0 commands the dwarfs, player 1 the trolls. **Dwarfs move first**, then the players
alternate. A turn is exactly one of the moves in sections 4–5; there is no passing. Pieces
capture only as part of their own move.

## 4. Dwarf moves

### 4a. Move

- **Allowed:** one dwarf moves 1 or more squares in one direction; every square it passes
  over and its landing square are empty.
- **Not allowed:** passing over or landing on any piece or the Thudstone. A move never
  captures.

### 4b. Hurl

A dwarf on square `s` is hurled in direction `d`. `N` is the number of dwarfs in the
unbroken line that starts at `s` and runs backwards (direction `−d`), counting the hurled
dwarf itself.

- **Allowed:** the dwarf travels `k` squares, `1 ≤ k ≤ N`; the `k − 1` squares it passes
  over are empty, and its landing square holds a troll. That troll is captured and the dwarf
  takes its square. **`N = 1` is allowed:** a lone dwarf may hurl onto an adjacent troll
  (§9.2).
- **Not allowed:** `k > N`; passing over any piece or the Thudstone; landing on an empty
  square, a dwarf or the Thudstone. A hurl captures exactly one troll.

## 5. Troll moves

### 5a. Step

- **Allowed:** one troll moves exactly 1 square to an empty square. If at least one dwarf
  is adjacent to the landing square, the troll player chooses: **capture all** of those
  dwarfs, or **capture none** (§9.1).
- **Not allowed:** moving more than 1 square (except by shove); landing on any piece or the
  Thudstone; capturing only some of the adjacent dwarfs; capturing without moving.

### 5b. Shove

A troll on square `s` is shoved in direction `d`. `N` is the number of trolls in the
unbroken line that starts at `s` and runs backwards (direction `−d`), counting the shoved
troll itself. The other trolls in the line stay where they are.

- **Allowed:** the troll travels `k` squares, `2 ≤ k ≤ N`; every square it passes over and
  its landing square are empty; and **at least one dwarf is adjacent to the landing
  square**. **All** dwarfs adjacent to the landing square are captured.
- **Not allowed:** `k = 1` (that is a step, §9.3); `k > N`, so a lone troll never shoves;
  passing over or landing on any piece or the Thudstone; a shove that captures nothing;
  declining the captures.

## 6. End of the battle

The battle ends as soon as one of these holds:

1. The player to move has no legal move. This includes a side with no pieces left.
2. `max_turns_without_capture` consecutive turns (default **200**) have passed without a
   capture by either side.
3. `max_turns` turns (default **800**) have been played.

Both limits are game parameters. The official ending, by agreement, cannot be implemented
(§9.4). The defaults are to be re-evaluated once our engine plays strongly (`PLAN.md`,
*Deferred decisions*).

## 7. Scoring

The dwarf player scores 1 point per surviving dwarf, the troll player 4 points per surviving
troll. The margin `m = dwarfs − 4 × trolls` lies in `[−32, 32]`. Returns are `m / 32` for
player 0 (dwarfs) and `−m / 32` for player 1 (trolls). One OpenSpiel game is one battle
(§9.5).

## 8. Action encoding

Every turn is exactly one action, so turns and actions coincide and `MaxGameLength` equals
`max_turns`.

- **Square `s`**, 0–164: the 165 squares in row-major order — row 0 cols 5–9 are 0–4, row 1
  cols 4–10 are 5–11, and so on. The Thudstone has an index but never moves.
- **Direction `d`**, 0–7: N `(−1,0)`, NE `(−1,+1)`, E `(0,+1)`, SE `(+1,+1)`, S `(+1,0)`,
  SW `(+1,−1)`, W `(0,−1)`, NW `(−1,−1)`.
- **Distance `k`**, 1–14.

| Actions | Formula | Meaning |
|---|---|---|
| 0 – 18,479 | `(s × 8 + d) × 14 + (k − 1)` | the piece on `s` goes `k` squares in direction `d`: **dwarf move**, **dwarf hurl**, **troll step without capturing** (`k = 1`) or **troll shove** (`k ≥ 2`) |
| 18,480 – 19,799 | `18,480 + s × 8 + d` | **troll step capturing all**: the troll on `s` steps 1 square in direction `d` and captures all adjacent dwarfs |

`NumDistinctActions` is **19,800**. Actions whose path leaves the board are never legal. The
kind of move follows from the mover, the action range and the landing square:

| Mover | Action | Move |
|---|---|---|
| dwarf | first range, landing square empty | move (4a) |
| dwarf | first range, landing square holds a troll | hurl (4b) |
| troll | first range, `k = 1` | step, captures none (5a) |
| troll | second range | step, captures all (5a); legal only if a dwarf is adjacent to the landing square |
| troll | first range, `k ≥ 2` | shove, captures all (5b); legal only if a dwarf is adjacent to the landing square — a shove never captures nothing |

This is the from-square × direction × distance layout of OpenSpiel's `chess`.

---

## 9. Why the rules read this way

**Sources, most authoritative first:**

- **The official rules** from the official Thud website, "Rules for Classic Thud and Koom
  Valley Thud" (© 2001/2005 Terry Pratchett, Trevor Truran and Bernard Pearson), archived:
  page 1 `https://web.archive.org/web/20071113030431/http://shop.thudgame.com/rules`,
  page 2 `https://web.archive.org/web/20071103213932/http://www.thudgame.com/rules2`.
- The printed rulebook's wording, as reproduced on BoardGameGeek and in the Tabletop
  Simulator workshop edition. Its PDF (`tesera.ru/images/items/1543265/THUD_RULES.pdf`)
  returns HTTP 403 to us. `wiki.lspace.org/Thud` is unreliable (it implies trolls hurl).
- Three implementations, read in full on 2026-09-22: `dstu/thud` (Rust, MCTS),
  `THFlowers/Thud-CLI` (Java, MCTS) and `hexparrot/thudgame` (Python, heuristic AI). Code
  citations are in `PROGRESS.md`, session 2.

### 9.1 Troll captures: all or none, declining allowed

The official text says only *"A troll captures one or more dwarfs by moving to a square next
to it (them)"* and *"Capturing is not compulsory."* It says nothing about choosing among
adjacent dwarfs; only the printed wording "any (all) … may be captured" hints at a subset.
All three implementations' AIs use all or none, so we do too — it keeps one action per turn.

Declining stays legal because the rules say so explicitly, although it is almost always a
blunder: the declined dwarf is adjacent to the troll and can hurl it on the next turn (§9.2).
dstu's and THFlowers' MCTS keep the option as well. A shove must capture, so it captures all.

### 9.2 A lone dwarf may hurl

Official: *"(Note: 1 lone dwarf can form a line of 1 by moving to a square adjacent to a
troll then hurling himself and capture a troll in this way. He can't move and capture on
the same move, though, but must wait for his next move assuming the troll hasn't captured
him by then.)"* All three implementations agree.

The distance rule is official too: *"4 dwarfs can attack a troll if there are 0, 1, 2, or 3
empty squares between the front dwarf and the troll"*, i.e. `1 ≤ k ≤ N`. Because `N` only
bounds the distance, counting the full line never loses a legal hurl.

### 9.3 Shoves travel 2 to N squares

A one-square shove and its captures produce exactly the position of a one-square step that
captures all, which 5a already allows. Leaving it out loses no position and keeps every
action unambiguous. THFlowers' MCTS generates shoves the same way. The official description
— *"the front troll is shoved in the back by the rest in the line"* — implies at least two
trolls anyway.

### 9.4 Termination

Officially a battle ends when the players agree that no more captures can be made, or when
a player cannot move. Agreement cannot be evaluated, and in self-play the trailing player
never agrees. We follow `hexparrot/thudgame`: a no-progress cap (its 400-ply cutoff is
annotated "self-play only") plus the official no-legal-move ending. `dstu/thud`'s
propose/accept actions are faithful but stall the same way.

The defaults 200/800 were chosen by measurement (2026-09-22; details in `PROGRESS.md`,
session 2). In 3,000 games of hexparrot's AI they never ended a game: the longest lasted 152
turns and its longest stretch without a capture was 41 turns. In 1,000 random games — the
regime of early training — they cut 1.5% of games short, against 16–57% for the shorter
pairs tested, while playing only 3–16% more turns than those. Removing the limits altogether would add only 0.3%
more turns, because random games end on their own.

### 9.5 One battle per game

Officially a game is two battles with sides swapped, won on the combined margin. We model a
single battle: its margin-based return already carries what the match adds up, and swapping
sides is a job for the evaluation harness, not the rules.
