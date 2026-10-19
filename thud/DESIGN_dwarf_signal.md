# The dwarfs' training signal — a design

Final, 2026-10-13: revised with the results of its measurements and after three adversarial
reviews (each review's findings were checked at the source before they were taken over —
*Review log* at the end); the test runs' results and the replication added 2026-10-14 and
2026-10-15 (§10). Written for the user to decide the next training runs. Context:
`PLAN.md` (*How we evaluate networks*: the blunder diagnosis, kickstarting, training
against alpha-beta, the input planes, the destination risk; runs G' to M) and
`PROGRESS.md`.

## Glossary

- **Approaches** (§5) carry a group letter and a number: **A** value targets with the same
  fixed point (A1-A3), **B** value targets that value dwarfs directly (B1, B1′, B2, B3),
  **C** data from play in which dwarfs survive (C1-C7), **D** the policy and the search
  (D1-D3), **E** how well the value learns small effects (E1, E2). The ones that matter
  most: **D1** a higher kickstarting weight for the dwarfs; **C7** alpha-beta's own dwarf
  positions as value-only data; **C3** per-side root noise; **C1** games against an older
  network's trolls; **B1** material ahead, and **B1′** the teacher's score, in the value
  target; **E1** a larger value-loss weight.
- **Measurements** (§4) are taken without training, before choosing training runs: **measurement 1**
  the search's own comparison at the blunders; **measurement 2** the teacher in the dwarfs'
  search; **measurement 3** self-play's root noise; **measurement 4** the value by side to move;
  **measurement 5** older trolls in self-play's conditions; **measurement 2b** (to run before any
  training night) the teacher in the dwarfs' search under the trainer's noise; **measurements
  6-7** (later, only if needed) what a lost dwarf costs in playouts, handicap starts. (In the conversation of
  2026-10-13 measurements 1-4 were called Q1, Q2, Q3 and Q5, measurement 5 P5.)
- **The value** is the network's estimate of the battle's result, from the dwarfs' side
  (−1 to 1). **The prior** is its probability for each move. **The teacher** is
  kickstarting's: the softmax of OpenSpiel's alpha-beta scores at depth 2 (material after
  the opponent's best reply) over 0.25 points. **Q** of a move is the search's mean value
  over the simulations through it; **visits** how many there were.
- **Outcome saturation**: every dwarf lost in nearly every game, so the dwarfs-left term
  of the result is always 0.
- **A test run** is one training run in a comparison, against its **control run**: the
  same start (a network and its buffer), the same number of steps, one change.
  **Starting from M102** means continuing training from run M's step-102 network and
  buffer.

## Summary

- **The problem.** Since run K's step ~77, 98-100% of self-play games end with every dwarf
  gone (M102: 162 of 162); it began with training against alpha-beta (run I, without
  kickstarting, saturated too; run H, without alpha-beta, did not), and within each run
  since, the dwarfs' blunders in self-play keep rising (run M: 9% → 15% → 23% of their
  moves by thirds of the run). **Losing every dwarf is not forced by the game; it is how
  our network plays the dwarfs**: a plain 2-ply alpha-beta search (the rules and material,
  nothing else) playing the dwarfs against the same trolls, M102's, keeps dwarfs alive in
  every battle (20 of 20) and loses by 13.2 points a battle, where our network's dwarfs
  lose every dwarf and by 25.2 (a 1-ply search loses every dwarf too: the second ply is
  what keeps them alive).
- **The mechanism, measured (measurement 4).** The value registers a far dwarf when the
  **trolls** are to move (M102 +0.021, K83 +0.050; a dwarf is 0.031) but hardly when the
  **dwarfs** are to move (M102 +0.002 to +0.011). H83 and G'64, trained without the games
  against alpha-beta, have no such gap: it came with those games, because the only
  positions of games in which dwarfs survive — our trolls' against alpha-beta's dwarfs —
  are all trolls-to-move positions. At a blunder the value one ply down (trolls to move)
  does prefer the teacher's safe move, by a third of a dwarf (measurement 1).
- **The strongest lever, measured at play time (measurement 2).** Mixing the teacher into M102's
  dwarf prior, in matches without noise: the dwarf side gains **+7.6 / +10.6 / +15.9
  points a battle at mixes 0.3 / 0.5 / 0.7** against L102's trolls (+5.5 / +9.8 / +12.8
  against K83's), 37-40 of 40 battles keep dwarfs, the dwarfs kill as many trolls or more,
  and our trolls are unchanged within the noise. **But the dwarfs survive by stalling**
  (38 of 40 battles at mix 0.7 end at the no-capture limit), and the trainer's root noise
  breaks stalls (measurement 3: either side's noise alone takes the unmixed network's surviving
  battles from 26% to 0-1%). **Measurement 2b**: under the trainer's noise the teacher-guided
  dwarfs keep no dwarf (0 of 40; they lose dwarfs more slowly and score +3.9); with the
  dwarfs' roots noise-free (C3) they keep dwarfs in **28 of 40** battles (no mix: 1 of 80).
- **The other measurements.** The dwarfs' own root noise (measurement 3) costs them 6 dwarfs by ply 60
  and ~1.8 points a battle (its pre-set rule for C3 is met), but without it outcomes stay
  saturated (1% of battles keep a dwarf). Against K83's older trolls (measurement 5) M102's
  dwarfs keep dwarfs in 9 of 40 battles under noise — all nine by killing every troll.
- **The test runs (§10).** From M102, 19 steps each: the control run (run M's recipe) and
  **run N, D1 + C3** (the dwarfs' teacher weight 0.7, the trolls' 0.3; no root noise at
  the dwarfs' roots), as measurement 2b's rule chose. **Run N wins by §7's rule** — N121 against
  the control's C121 +5.50 a pair (+1.91 to +9.09); its dwarfs against hexparrot +9.3 ±
  2.2 better, blunders 9.8% against 21.6%; its trolls not worse — and **the replication,
  run N2** (a rerun of run N only, the user), meets it too: +6.10 (+3.09 to +9.11) against
  C121, +0.95 (−2.04 to +3.94) against N121. The control did not slide (C121 beat M102 by
  +7.20). The gain is the policy's: the value's side-to-move gap has not closed in 19
  steps. **Decided** (the user, 2026-10-15): a longer D1 + C3 run first — run N continued
  from N121 for 40 steps, measurement 4 every 10 (§10).
- **Decided by the user (2026-10-13):** test runs from M102 with D1; its slow, gated taper
  (§5); the regression gates, also in the promotion rule (§7; refined since by review 3).
  D1's weight capped at 0.7 (above it the dwarfs' policy target nears pure imitation of
  alpha-beta, ruled out on 2026-10-07; if 0.7 is not enough, the teacher stays on longer).
  The win rule of §7 (2026-10-14). **For the user to decide** later, only if needed:
  whether C7 may reverse the rule of 2026-10-07 (only our network's positions recorded).

## 1. The problem, as measured

A battle's return is (dwarfs left − 4 × trolls left) / 32 for the dwarfs (`THUD_RULES.md`
§7): a dwarf is worth 1/32 = 0.031, a troll 0.125, of a value range [−1, 1].

### 1.1 Outcome saturation

How each step's self-play games ended (the trainer's `endings`, `learner.jsonl`; runs G',
K and M are one line of descent):

| Steps | Run | Every dwarf gone | Every troll gone | Mean length |
|---|---|---|---|---|
| 40-64 | G' (self-play only) | 27-75% | 20-61% | 230-500 plies |
| 65-69 | K (alpha-beta games from step 65, kickstarting) | 49-84% | 10-36% | 300-470 |
| 70-76 | K | 92-99% | 0-8% | 160-260 |
| 77-102 | K, then M | **98-100%** (M102: 162 of 162) | 0-2% | 95-135 |

- **The natural control**, from the same start (G'64), steps 70-83: run H (the dwarfs'
  policy temperature 1.5, self-play only) 41-67% of games with every dwarf gone, the
  dwarfs' mean return at step 83 −0.02; run I (run H plus 30% of positions from games
  against alpha-beta, no kickstarting) 80-95%, −0.43; run K (run I plus kickstarting) −0.73.
  The saturation came with training against alpha-beta — its trolls learnt to punish (an
  adjacent dwarf taken 46% of the time, run H 23.5%, `PLAN.md`) while its dwarfs did not
  improve; run K went further (one run each, so not necessarily kickstarting's doing).
- M102's dwarfs: self-play margin −25.2 (~1.7 trolls killed a game, the last dwarf lost
  after ~95 plies). Against alpha-beta's depth-2 trolls in training (`vs_alphabeta`, ~20
  games a step): −32 in every game from step 65, but for one game each at steps 88, 93
  and 96. Against hexparrot's trolls (40 battles, replayed): no dwarf left in 39, one in
  the last (at the no-capture limit); K83 40 of 40.

### 1.2 Survival is possible against our trolls

| Dwarfs | Trolls | Battles with dwarfs left | Dwarfs left | Dwarfs' mean margin |
|---|---|---|---|---|
| alpha-beta depth 1 (matches) | M102 | 0 of 20 | 0 | −30.0 |
| alpha-beta depth 2 (matches) | M102 | **20 of 20** | 5-20 | −13.2 |
| alpha-beta depth 3 (matches) | M102 | 20 of 20 | 2-23 | −1.8 |
| alpha-beta depth 2 (run M's training games in the actor logs: 236 of the run's 366; 20 of the 32 actors log) | run M's network, with root noise | **236 of 236** | 4.4 on average | −16.2 (189 at the no-capture limit, 46 at the turn limit) |
| hexparrot (lookahead 3) | M102 | 30 of 40 | 2-25 | −1.3 |
| M102 (matches) | K83 | ≥15 of 40 (margins not a multiple of 4) | | −18.2 |
| M102 (matches) | L102 | ≥13 of 40 (7 of them certainly a single dwarf) | | −25.5 |
| M102 (self-play) | M102 | 0 of 162 | 0 | −25.2 |

A full-width 2-ply material search plays the dwarfs about 10-12 points a battle better
than our network does against our trolls; a 1-ply search loses every dwarf — the
knowledge that keeps dwarfs alive is exactly the second ply, the trolls' reply. (Matches: no root noise, the most visited move;
`az_match`.)

### 1.3 How the dwarfs lose

In run M's self-play (324 games of actors 0-3, replayed):

- 32 dwarfs lost a game, in captures of 1.12 dwarfs on average (2+ dwarfs in 10.4% of
  the captures); 11% of the dwarfs lost to shoves.
- **Blunders** — a non-capturing dwarf move to a square next to a troll, the dwarf taken
  at once — account for 10.2 dwarfs a game, **32%** of the losses (16% of the dwarfs'
  moves after ply 10; against hexparrot 25% of their moves, 51% of their losses). The rest
  are trolls reaching dwarfs that did not just move.
- **Baits are rare**: after a blunder the dwarfs take a troll with their very next move
  0.7% of the time, against 3.6% for any dwarf move (against hexparrot 0.6%, against
  3.4%).
- **Self-play is sliding, not standing still**: by thirds of run M (the same games) the
  blunder rate is 8.6% → 14.9% → 23.2% of the dwarfs' moves (the 16% above is the run's
  average; M102's own rate in measurement 3's emulation is 28%), the dwarfs lost that way 5.8 →
  8.1 → 10.5 a game. By the reviewer's count (actors 0-1) the dwarfs' unsafe moves rose
  from 11.8% to 25.1% of their moves — above their ~21% share of the legal moves — and the
  trolls' punishing them at once from 85% to 97%; run K slid likewise (blunders 4% → 9% →
  12.5%), K continued and L too (to ~20%). Against hexparrot, without noise, M102 still
  blunders less than K83 (25% against 36%).

### 1.4 The value's sensitivity at the blunders

v_captured − v_before (the value after the blunder and its capture, minus before the
blunder) on **fixed** positions — M102's and K102's 150 blunder positions against
hexparrot — for each network (`az_blunder_probe`, values only; the control reproduces
M102's published values exactly):

| Network | On M102's positions (± SE) | On K102's positions | Divided by 1 − v² (pre-tanh) |
|---|---|---|---|
| G'64 | −0.016 ± 0.005 | −0.011 | −0.021 / −0.014 |
| I83 | −0.009 ± 0.002 | −0.012 | −0.025 / −0.031 |
| K83 | −0.012 ± 0.002 | −0.010 | −0.026 / −0.020 |
| K102 (K continued) | −0.005 ± 0.002 | −0.007 | −0.015 / −0.021 |
| L102 | −0.007 ± 0.002 | −0.007 | −0.021 / −0.019 |
| M84 / M92 / M100 | −0.010 / −0.008 / −0.005 | −0.008 / −0.007 / −0.005 | |
| M102 | **−0.002 ± 0.003** | +0.001 ± 0.002 | −0.003 / +0.007 |

- The fall from G'64 to K83 in `PLAN.md`'s first readings (−0.023 → −0.011) was mostly
  two artefacts: each network measured on its own games, and tanh's compression (G'64's
  v_before is about −0.2, the others' about −0.75).
- All continuations of K83 halve alike (K102, L102 to about −0.006); only run M's last
  checkpoints fall further (paired, L102 − M102: −0.005 ± 0.002 and −0.008 ± 0.002 on the
  two position sets) — specific to run M, its cause unknown.
- **L102 and M102 blunder at the same rate** (25.6% / 25.4%) and tie head-to-head while
  their sensitivity differs 3-4x.
- Both positions of this proxy are dwarf-to-move positions — exactly where measurement 4 found
  the value weakest; the measurements by side to move and the search's own comparison
  are measurements 4 and 1.

## 2. Diagnosis

**What holds the dwarfs back is how our network plays them, not the game.** Against our
trolls a plain 2-ply material search playing the dwarfs keeps dwarfs alive; our network's
dwarfs do not — so better dwarf play exists, and our network has not found it. Under our
dwarfs' play the outcome hardly depends on any single dwarf (every one is lost anyway), so
self-play teaches the value little about dwarfs and the search's visits (70% of the
policy target) learn little. **And it is getting worse, not standing still** (§1.3): in
every run since training against alpha-beta began, the dwarfs choose unsafe moves more
and more often in self-play. A possible reason, an inference to measure: if the prior
fits its target, M102's mass on unsafe moves, 0.256 = 0.7 × (the visits' mass) + 0.3 ×
(the teacher's, ~0.06), puts the visits' at ~0.34 — the search amplifies the prior's
unsafe mass by ~1.3, the visits feed back at 0.7, a loop gain of ~0.93 at the teacher
weight 0.3 (and ~0.4 at 0.7). (Exact policy iteration cannot stall below the optimum —
the policy improvement theorem — but with approximate values, noisy targets and 100
simulations the search here does not improve on the prior's unsafe moves.)

**Measured since (measurement 4): the value's blindness is by side to move, and it comes from
the data.** The only games in the buffer in which dwarfs survive are those against
alpha-beta's dwarfs, and from them only our trolls' positions are recorded (~26% of all
positions) — trolls-to-move positions. There the value learnt a dwarf's worth; at
dwarf-to-move positions, recorded only from games in which every dwarf is lost, it did
not. H83 and G'64, trained without those games, show no gap. For the dwarfs' search this
means: a dwarf move's child (trolls to move) is valued fairly — at the blunders it
prefers the teacher's safe move by a third of a dwarf (measurement 1) — but every evaluation
after a troll reply is a dwarf-to-move position, where a lost dwarf barely counts. (That
the search therefore prefers blunders the deeper it looks is plausible but not shown:
measurement 1's comparison of Q is biased by how its positions were selected.)

Why our dwarfs' search does not do what a 2-ply search does: it is not full-width. With
~285 legal moves and ~40 troll replies each, 100 simulations cannot see each move's
replies; the prior must carry that knowledge, and it learns it from the visits (70%) and
from the teacher — exactly alpha-beta at depth 2 — at only 30%. Measurement 2 shows how much
the teacher's knowledge in the prior is worth.

Contributing factors:

1. **Self-play's root noise** (Dirichlet α 0.1, ε 0.25, at every root of both sides) and
   10 plies (5 a side) sampled at temperature 1: the dwarfs' own noise costs them ~6
   dwarfs by ply 60 and ~1.8 points a battle, and either side's noise breaks the stalls in
   which dwarfs survive without it (measurement 3).
2. **The dwarfs' games against alpha-beta's trolls** give every one of those dwarf
   positions the target −1 (10.9% of the dwarfs' recorded positions in run M).
3. **The search's resolution.** A lost dwarf (0.031) competes with the untried-move
   reduction (0.2 × √ of the visited siblings' prior mass, i.e. several dwarfs), with the
   value's dispersion between neighbouring positions (v_captured − v_before: standard
   deviation 0.034) and with a winner's curse over ~46 effectively visited moves. PUCT's
   exploration term (Q + 2 · P · √N / (1 + n), `mcts.cc`) is about 0.07 for the dwarfs'
   near-uniform prior at 100 simulations: their search is driven by Q, not the prior
   (KL(visits ‖ prior) 2.09 nats; the prior's top move the most visited in only 13% of
   their searches).
4. **The value learns small effects poorly**: dwarf positions sit at v ≈ −0.75, where tanh
   compresses differences by 1 − v² ≈ 0.4; the value loss (0.035) is small beside the
   policy loss (3.86) and the L2 term (0.048).

### What cannot fix it, and what can

Value targets whose fixed point is the expected outcome under the current play — the
result z; z mixed with the search's root value q; KataGo's short-term value targets; n-step
or TD(λ) targets; potential-based shaping with the material as potential (Ng, Harada and
Russell, ICML 1999: policy-invariant by design) — **are unlikely to help**: they give the
value nothing new about dwarfs. (Our data are off-policy — root noise, temperature,
alpha-beta games — so q-targets do differ from z-targets where those matter; measurement 3 found
the dwarfs' noise worth ~1.8 points a battle.) Reweighting the final score (a dwarf left
worth more) does nothing while no dwarf is left.

What can: **(1) better dwarf play generating the data** (D1, with C3 if the noise
blocks it); **(2) dwarf-to-move data from play in which dwarfs survive** (C7, C1, C2);
**(3) a target that values dwarfs directly** — a bias, tapered (B1, B1′); and helping the
value learn small effects (E1, E2).

## 3. Constraints

- No rule changes; the objective stays the margin; evaluation at the real rules.
- Learning from the rules: the material-scoring alpha-beta is accepted as teacher
  (kickstarting) and opponent. **The user's rule of 2026-10-07: only our network's
  positions are recorded — alpha-beta only punishes, nothing imitates it** (`PLAN.md`,
  training against alpha-beta). C7 would reverse it (for value targets only) and needs
  the user's decision; outside engines' moves as targets are excluded; an outside
  engine's games as value data would be a further step (C6).
- A clean evaluation: one change against a control run from the same network, at equal
  steps (as runs L and M against K continued and L).
- **Compute**: a 19-step run takes ~9.3 hours here (run M: 03:25-12:43, 28.7 minutes a
  step, ~21,900 positions a step); one training run a night. **Evaluation** of 2-3
  networks takes about a day of machine time per round, between runs, not beside them
  (`CLAUDE.md`). Matches side by side do not add up: three matches gave ~1,500
  simulations/s together, six ~1,400-1,700 (the measurements, 2026-10-13).
- **Targets are fixed when a position enters the buffer**, which holds ~21 steps (458,752
  positions, ~21,850 a step; each position sampled ~3 times during its stay: 64 batches of
  1,024 a step). Any change to targets reaches the buffer gradually: over a 19-step run,
  roughly 90% of the way.
- **Memory**: a second network in the trainer (C1) costs a cache and inference threads;
  WSL's swap is nearly full (3.9 of 4 GB, RAM free) — a `wsl --shutdown` before the next
  long run also brings in the 8 GB swap set on 2026-10-07.
- **No seed**: the trainer seeds its actors from the clock (`alpha_zero.cc`), so a
  replicate is a rerun, and run-to-run variance is unmeasured.

## 4. The measurements and their results (2026-10-13)

Run in the night of 2026-10-13 (`~/thud-runs/night7_2026-10-13.py`, results in
`~/thud-runs/checks_2026-10-13/`; paused 9.3 hours by a standby when the charger came
loose, results unaffected). The new options and tools were checked first: `az_match` with
them off reproduces the old binary's matches exactly (pairs and request counts), its
noisy battles are the same on 1 and 3 threads, and the extended `az_blunder_probe` prints
the old fields unchanged.

### Measurement 1 — the search's own comparison at the blunders

`az_blunder_probe` (extended): at each blunder position, the blunder against the best safe
move — the non-capturing move other than the blunder that the teacher scores best, the
network's prior breaking ties; both root children's Q and visits in the search as played
(100 simulations, no noise); each child's own value (trolls to move); each line's value
after the trolls' reply (the capture after the blunder; after the safe move the reply
taking the most material, the prior breaking ties). M102; 150 blunders each.

| | Against hexparrot | In self-play's conditions (measurement 3's noisy cell) |
|---|---|---|
| Blunders, share of the dwarfs' moves | 25% | 28% (489 of 1,750) |
| The noise-free search plays the blunder again | 100% | 35% |
| Teacher: the safe move better by (points, mean) | 0.76 | 0.85 |
| Visits: blunder / safe move | 29.4 / 6.5 | 12.4 / 9.2 |
| Q: safe − blunder, where both were visited | −0.013 ± 0.003 (114 cases, the blunder chosen in all; safe preferred in 29) | where the search chose the blunder (47): −0.011 ± 0.002; where it did not (32): −0.002 ± 0.007 |
| Child's value: safe − blunder (trolls to move) | **+0.011 ± 0.002** | **+0.010 ± 0.002** |
| After the trolls' reply: safe − blunder (dwarfs to move) | −0.004 ± 0.003 | −0.010 ± 0.003 |
| The teacher sees no better non-capturing move | 24% of the blunders | 17% |

**Reading:** one ply down the value prefers the safe move by about a third of a dwarf —
the solid result here, since the way the positions were picked works against it. The Q
row and the row after the reply are **biased by selection** (review 3): the positions are
moves the search chose, and a chosen move's Q beating one alternative among ~46 visited
moves is close to automatic (in the hexparrot set the search chose the blunder in every
case; in the self-play set, where it did not choose it, Q shows no preference). So measurement 1
does not show that the search prefers blunders the deeper it looks; measurement 4 carries the
claim about dwarf-to-move positions. To measure the search itself: unselected dwarf
positions, the teacher's unsafe against its safe moves, each given a fixed number of
simulations (not run). In self-play's conditions two thirds of the blunders come from the
noise and the sampled plies (the noise-free search plays only 35% of them again).

### Measurement 2 — the teacher in the dwarfs' search, without training

`az_match teacher_mix_a=w,0`: M102's prior at every dwarf node of its searches (as a
network trained toward the teacher would have it) is (1 − w) × its own + w × the teacher's;
40 pairs against L102 at the promotion matches' settings (seed 3, the same openings pair
by pair as the match of 2026-10-12 without the mix; no noise).

| Mix w | Dwarf side | Dwarf side vs w = 0 (paired) | Battles keeping dwarfs | Dwarfs left | Trolls killed | Troll side vs w = 0 | Pair margin vs L102 |
|---|---|---|---|---|---|---|---|
| 0 (2026-10-12; no games file: from the margins) | −25.5 | | ≥13 of 40, 7 of them certainly one | – | ≈1.4 | | +0.65 (14 won, 17 lost) |
| 0.3 | −17.9 | **+7.6 ± 1.4** | 37 of 40 | 8.4 | 1.43 | −0.2 ± 0.9 | +8.0 (32 won) |
| 0.5 | −14.9 | **+10.6 ± 1.7** | 40 of 40 | 11.6 | 1.40 | +0.8 ± 0.9 | +12.0 (34 won) |
| 0.7 | −9.6 | **+15.9 ± 1.4** | 40 of 40 | 14.6 | 1.98 | −0.6 ± 0.8 | +16.0 (40 won) |

Against K83's trolls (the replication; mix 0: the match of 2026-10-12, −18.2):

| Mix w | Dwarf side | Dwarf side vs w = 0 (paired) | Battles keeping dwarfs | Dwarfs left | Trolls killed | Troll side vs w = 0 | Pair margin vs K83 |
|---|---|---|---|---|---|---|---|
| 0 (2026-10-12; from the margins) | −18.2 | | ≥15 of 40 | – | ≈2.5 | | +12.0 (38 won) |
| 0.3 | −12.6 | **+5.5 ± 1.5** | 38 of 40 | 9.9 | 2.38 | −1.2 ± 0.7 | +16.3 (38 won) |
| 0.5 | −8.3 | **+9.8 ± 1.4** | 40 of 40 | 12.7 | 2.75 | −1.0 ± 0.8 | +20.8 (40 won) |
| 0.7 | −5.4 | **+12.8 ± 1.9** | 40 of 40 | 13.9 | 3.20 | −0.2 ± 0.7 | +24.6 (39 won) |

**Reading:** the teacher's knowledge in the dwarfs' prior keeps them alive, more with every
step of the dose, against both opponents alike, and at 0.7 the dwarfs even kill more
trolls (the feared loss of hurl set-ups did not show); our trolls play the same within
the noise (their battles are other games — the mix also sits at the dwarf nodes inside
M102's troll searches — and against K83 at 0.3 they are −1.2 ± 0.7). **The rule set in
advance ("the best mix that does not cost troll kills") chooses 0.7**, the largest tried
— the effect had not levelled off. (The pairing by opening buys little: the dwarf
margins of a pair correlate at 0.04-0.34.)

**How the dwarfs survive: by stalling** (review 3; the games replayed):

| Games (M102 as the dwarfs, no noise) | Dwarfs lost by ply 60 / 100 / 200 | Battles ending at the no-capture limit | Mean length |
|---|---|---|---|
| Mix 0.7 against L102 | 13.5 / 14.7 / 15.8 | 38 of 40 | 435 plies |
| Mix 0.3 against L102 | 16.0 / 18.0 / 20.2 | 34 of 40 | 531 |
| No mix, against itself (measurement 3's noise-free cell) | 19.4 / 23.5 / 27.6 | 12 of 40 | 303 |

The teacher does not prevent the early losses (13.5 dwarfs by ply 60 even at 0.7); it stops
the bleeding afterwards, and the battle freezes until the no-capture limit. **And these
matches had no root noise**, while measurement 3 shows that either side's noise alone breaks
the unmixed network's stalls (battles keeping dwarfs 26% → 0-1%). So what D1 needs —
teacher-guided dwarfs keeping dwarfs *in self-play*, to give the value dwarf-to-move data
— measurement 2 does not show. Other caveats: at play time the teacher is a real 2-ply search at
every dwarf node; trained, the network only approximates it, and the buffer delays it
(§3).

### Measurement 2b — the teacher in the dwarfs' search under the trainer's noise (to run first)

`az_match` combines the options: M102 with `teacher_mix_a=0.7,0` against plain M102 in
self-play's conditions (`sample_plies=10 opening=0`), once with `noise=1,1` (the
trainer's) and once with `noise=0,1` (the dwarfs' roots noise-free: C3's condition); 40
pairs each, ~1-1.5 hours together. Measured on the battles in which the mixed network
plays the dwarfs: battles keeping dwarfs, dwarfs lost by ply 60, length, how they end —
against measurement 3's cells without the mix (0% and 1% keeping dwarfs; 23.6 and 18.0 dwarfs
lost by ply 60). **Decides, fixed in advance:**

- dwarfs kept in a clear share of battles (≥ 20%) under both sides' noise → **D1 alone**;
- only with the dwarfs' noise off → **D1 together with C3** (per-side noise);
- in neither → D1 does not desaturate self-play *at the start*. Training may still get
  there — the network learns the teacher's prior and the visits improve on it, a sharper
  prior is displaced less by the noise, the trolls and the value adapt — and the D1 run
  shows whether it does (the share of self-play games keeping dwarfs, measurement 4's gap, by
  step). If both stay flat through the run, the value gets its own help next (**B1′**,
  or **C7** with the user's word, or C3).

**Result** (2026-10-13 18:22-20:12, `~/thud-runs/checks_2026-10-13/Q2b_*`, `analyse_q2b.py`;
the battles in which the mixed network plays the dwarfs):

| Noise at the roots of | Battles keeping dwarfs | Dwarfs lost by ply 60 | Length | Dwarfs' margin | Trolls killed | Endings |
|---|---|---|---|---|---|---|
| both (the trainer's), mix 0.7 | **0 of 40** | 17.1 ± 0.7 | 129 plies | −22.0 ± 1.1 | 2.50 | every dwarf gone, 40 |
| both, no mix (measurement 3) | 0 of 80 | 23.6 ± 0.4 | 91 | −25.9 ± 0.5 | 1.51 | every dwarf gone, 80 |
| the trolls only, mix 0.7 | **28 of 40 (70%)** | 13.4 ± 0.4 | 621 | −17.0 ± 1.2 | 3.10 | no-capture limit 23, every dwarf gone 12, other 5 |
| the trolls only, no mix (measurement 3) | 1 of 80 | 18.0 ± 0.3 | 167 | −24.2 ± 0.6 | 1.95 | every dwarf gone 79 |

(The mixed network as the trolls, with the mix at the dwarf nodes inside its troll
searches, noise both: −25.3, every dwarf gone in 40 — as without the mix.)

**Reading:** under the trainer's noise the teacher makes the dwarfs lose their dwarfs more
slowly and kill a troll more a battle (+3.9 points), but every dwarf is still lost; with
the dwarfs' roots noise-free, 70% of the battles keep dwarfs (2.6 on average), mostly in
stalls. **The rule set in advance chooses D1 together with C3** (only with the dwarfs'
noise off). Caveats: two changes in one run (D1 alone as a further night if they must be
told apart); the long, stalled games below.

If stalls do survive in training, expect games of 400-600 plies — ~40-50 games a step
instead of ~160, about half the positions from capture-free stalls: to be watched (the
share of positions after a battle's last capture).

### Measurement 3 — self-play's root noise

M102 against itself in self-play's conditions (`az_match noise=… sample_plies=10
opening=0`: the trainer's Dirichlet noise per side, its 10 plies sampled from the visit
counts), 80 independent battles a cell:

| Noise at the roots of | Dwarfs lost by ply 60 | 16th dwarf lost at ply | Trolls killed | Battles keeping dwarfs | Mean length | Dwarfs' margin |
|---|---|---|---|---|---|---|
| neither | 19.2 ± 0.3 | 50 ± 5 | 1.25 | 26% | 296 plies | −25.9 |
| both (the trainer's self-play) | 23.6 ± 0.4 | 38 ± 1 | 1.51 | 0% | 91 | −25.9 |
| the dwarfs only | 26.1 ± 0.3 | 35 ± 1 | 1.02 | 0% | 82 | −27.9 |
| the trolls only | 18.0 ± 0.3 | 47 ± 2 | 1.95 | 1% | 167 | −24.2 |

The "both" cell reproduces the trainer's self-play (91 plies and −25.9 against run M's 95
and −25.2): the emulation is faithful. 2 × 2 effects on dwarfs lost by ply 60: the dwarfs'
noise +6.3 ± 0.3, the trolls' −1.9 ± 0.3, interaction −1.4 ± 0.7.

**Reading:** **the rule set in advance is met** (the dwarf-noise effect, 6.3, is at least
half the both-minus-neither difference, 4.4, its interval above 0), so C3 qualifies as a
run of its own. What C3 would change is the dwarfs' noise alone — the "trolls only" row
against "both": the dwarfs lose 18.0 instead of 23.6 dwarfs by ply 60, kill 1.95 instead
of 1.51 trolls, the battles last 167 instead of 91 plies, and the margin is −24.2 instead
of −25.9 (+1.8 ± 0.75). **But outcomes stay saturated without the dwarfs' noise** (1% of
battles keep a dwarf, and none with the dwarfs' noise alone): C3 by itself would not give
the value games in which dwarfs survive. It may be a precondition for D1, though — measurement
2b decides.

### Measurement 4 — the value by side to move

`az_dwarf_value_probe` (new): in positions from run M's actor logs (~750 a kind), one dwarf
at least 4 king's moves from every troll removed, the value's drop (a dwarf's material:
0.031); as positive control a far troll removed (0.125); as placebo the same dwarf moved
one square (no material changed: ~0 expected). Six networks.

| Network | Trolls to move: games against alpha-beta's dwarfs / self-play | Dwarfs to move: games against alpha-beta's dwarfs / self-play |
|---|---|---|
| G'64 (before alpha-beta games) | +0.009 / +0.017 | +0.009 / +0.018 |
| H83 (G'64 plus run I's 19 steps and settings, but no alpha-beta games; ~190 positions a kind) | +0.013 / +0.014 | +0.014 / +0.015 |
| I83 | +0.042 / +0.059 | +0.017 / +0.027 |
| K83 | +0.050 / +0.055 | +0.016 / +0.022 |
| K102 | +0.040 / +0.037 | +0.007 / +0.015 |
| L102 | +0.027 / +0.028 | +0.009 / +0.017 |
| M102 | +0.021 / +0.021 | **+0.002** / +0.011 |

(Standard errors 0.0004-0.0017; the troll control +0.05 to +0.12 everywhere; the placebo
−0.007 to +0.007 — but +0.003 to +0.007 at self-play's dwarf-to-move positions for every
network since G'64, so the last column slightly overstates a dwarf's worth there: M102's
+0.011 is ~+0.004 beyond its placebo.)

**Reading:** with the trolls to move every network trained with the games against
alpha-beta registers a far dwarf clearly (I83 to K102 above its material of 0.031, L102
and M102 at 0.021-0.028; all at or above it once divided by 1 − v²); with the dwarfs to
move far less, least of all in the positions of those games — which the buffer never
holds (alpha-beta's dwarf positions are not recorded). **H83 is the clean control** — the
same 19 steps and settings as I83 but for the alpha-beta games — and has no gap; neither
has G'64. The gap is not the probe's doing: the two kinds are alternate plies of the same
games, 94% of the sampled positions have a far dwarf, and it survives the division by
1 − v² (M102: 0.031 against 0.005). **The rule set in advance ("registered at
trolls-to-move but not at dwarf-to-move positions → C7") is met.** Both sides weaken over
training (K83 → M102). The value's *level* splits by side too: in the positions of the
games against alpha-beta's dwarfs M102's value is −0.54 with the trolls to move and −0.75
with the dwarfs to move, on consecutive plies — so a search's Q there depends on the
parity of its leaves' depth; to be watched in the D1 run, whose self-play may come to
look like those games.

### Measurement 5 — older trolls in self-play's conditions

M102 against K83 with the trainer's noise on both sides and its sampled plies, 40 pairs:
M102's dwarfs keep dwarfs in **9 of 40** battles (1.9 left on average), margin **−9.4**
(against −25.9 against its own trolls in measurement 3's "both" cell; −18.2 against K83 without
noise); the pair margin +19.7 (+15.8 to +23.6). **All nine are battles in which the
dwarfs killed every troll** (5.2 trolls killed a battle on average); in the other 31
every dwarf is lost. Older trolls desaturate the outcomes, but by collapsing, not by our
dwarfs learning to stay alive — the "exploiting weak trolls" risk of C1, in the data.

## 5. The approaches

**Dev** = development and tests (estimates at our pace so far); **train** = extra compute
in training; **run** = extra compute in play. Shares are given in recorded positions.

### Group A — value targets with the same fixed point

**A1. z mixed with the search's root value: z′ = (1 − a)·z + a·q.** Oracle's AlphaZero
series (*Lessons from AlphaZero, part 4*) found the average better than either; Leela
Chess Zero has trained against such a mix.
- Up: lower variance; `root_value` is already in every recorded position.
- Down: q is built from the same value; its errors reinforce. **`root_value` is from the
  side to move's view** (the trainer's cutoff code sets `returns[player] = root_value`),
  the target is the dwarfs' — negate at troll positions, with a test.
- Dev ~2 hours; train 0; run 0.

**A2. KataGo's short-term value heads** (exponential averages of future search values,
horizons ~6, 16 and 50 turns; since g170, `docs/KataGoMethods.md`: "slightly faster"
training, "slightly better" value loss). Same values averaged; new heads, losses, a
buffer field. Dev ~1 day; train small; run 0.

**A3. n-step / TD(λ); potential-based shaping.** Dev ~2 hours; nothing new for the value.

Group A: not now.

### Group B — value targets that value dwarfs directly (a bias, tapered)

**B1. Material ahead.** z′_t = (1 − b)·z + b·m_t, m_t the material ((dwarfs − 4 × trolls)
/ 32) averaged over the following plies with weights (1 − λ)λ^j (the final result once
the game has ended).
- Up: the game's own scoring; the learner has every trajectory, and each recorded
  position's material is in its observation.
- Down:
  - **The buffer delays it** (§3): b falling from 0.5 to 0 over 13 steps puts at most a
    mean b of ~0.17 into the buffer, highest just when the taper "ends". Fix: store m_t
    per position and apply b when sampling (a buffer field; old buffers must still load),
    or rewrite the inherited buffer's targets, or start from a fresh buffer.
  - **The signal is diluted by the saturation it targets.** m_t is measured under our
    dwarfs' play too: the safe move's dwarf is usually lost a few plies later anyway (our
    dwarfs lose ~0.68 dwarfs per two plies in self-play), so the target difference is about
    b · 0.031 · (λ − λ^(1+d)) for a delay of d plies — ~0.25 · b · 0.031 at d = 6, λ = 0.95.
    B1 mostly rewards delay.
  - It raises the dwarf values' level (toward −0.5), which alone raises their sensitivity
    through tanh — a confound for the mechanism's measurement.
  - Materialism while b > 0; two hyperparameters and a schedule; both sides share the
    head (consistent: a troll is 4); games against alpha-beta record every other ply, so
    the horizon is counted in plies.
- Dev ~3 hours at insertion (with the delay), ~1 day at sampling. Train 0; run 0.

**B1′. The teacher's score in the value target.** z′ = (1 − b)·z + b·s, s the teacher's
depth-2 material score of the position (its best move's score, /32). Stockfish's NNUE
training mixes the game result and the search's evaluation the same way (`lambda`,
nnue-pytorch).
- Up: immediate and full-size at both sides to move (measurement 4's gap would be addressed
  directly); the scores are already computed at every recorded position for kickstarting
  — no extra compute.
- Down: myopic (2 plies) by construction; the same buffer delay as B1; a bias. The score
  is from the side to move's view — negate at troll positions, as for A1.
- Dev as B1.

**B2. A material head in the search's utility** (KataGo's search maximises win/loss
utility plus a scaled score utility, Wu 2019): a second head learns m_t; leaf utility =
value + c × material; c tapered, 0 in evaluation if wanted. Up: the main value unbiased; c
tunable without retraining. Down: a head, a loss, a buffer field, the search's utility;
visits biased by c. Dev ~1½ days; train small; run small.

**B3. Material or survival as auxiliary heads only** (like KataGo's ownership head). The
main value's fixed point is unchanged; the trunk already reads takeable dwarfs and danger
(AUC 0.88-0.93). Dev ~1 day. Not now.

### Group C — data from play in which dwarfs survive

**C1. Trolls from an older network** (a league in miniature; OpenAI Five played 20% of its
games against past versions of itself, OpenAI 2018; AlphaStar a league). A share of the
dwarfs' positions from games against an older network's trolls (K83's), only our dwarfs'
positions recorded, as for alpha-beta.
- Up: real outcomes, no bias in targets; dwarf-to-move positions from games whose result
  varies — **measurement 5**: against K83's trolls with the trainer's noise M102's dwarfs keep
  dwarfs in 9 of 40 battles, margin −9.4.
- Down: the value learns a mix of opponents it cannot tell apart; **exploiting weak trolls
  may not transfer — and measurement 5 shows just that**: all nine survivals are battles in
  which every troll was killed, none a battle in which dwarfs stayed alive beside living
  trolls; a second shared evaluator with its own cache (~0.6 GB at the default)
  and inference threads, and the trainer's batching already caps throughput, so these
  games cost more than twice per recorded position; memory (§3).
- Dev ~1 day; train >2× for those games.

**C2. Handicap starts** (KataGo plays a share of games with handicap stones and randomises
komi, Wu 2019): a share of self-play from 7- or 6-troll starts. A variant: **start
positions from the dwarfs' positions of our own games against alpha-beta** (Go-Exploit,
Trudeau and Bowling, AAMAS 2023: self-play from an archive of the agent's own states of
interest, for a value that generalises better). Starts from hexparrot's games would bring
in an outside engine's influence (as C6).
- Up: no bias; the rules only; both sides recorded; no extra compute.
- Down: not the real opening; transfer to 8-troll games; the trolls' statistics mix games.
- **Measurement 7 first**, at the trainer's settings with an 8-troll baseline. Dev ~½ day
  (`ThudGame::NewInitialState(const std::string&)` exists).

**C3. Per-side self-play noise**: ε per side, and α per side by AlphaZero's rule (α ≈ 10 /
legal moves: ~0.035 for the dwarfs' ~285, ~0.2 for the trolls; Silver et al. 2018 used 0.3
/ 0.15 / 0.03 for chess, shogi, Go). **Measurement 3: its rule is met** — without their own
noise the dwarfs lose 18.0 instead of 23.6 dwarfs by ply 60 and score −24.2 instead of
−25.9 (+1.8 ± 0.75) — **but outcomes stay saturated** (1% of battles keep a dwarf): alone
it gives the value no games in which dwarfs survive. Up: removes an artefact, no bias;
per-side settings exist for simulations and temperatures; would remove much of
self-play's blunders (measurement 1: the noise-free search repeats only 35% of them); **may be
what lets D1's dwarfs survive in self-play (measurement 2b)**. Down: less exploration for the
dwarfs. **Built 2026-10-13**: `--player_policy_epsilon` and `--player_policy_alpha` in our
trainer ("0,0.25": no root noise in the dwarfs' self-play searches; empty: as before);
`search_settings_test` extended (with "0,0.25" the dwarfs' search has exactly the
un-noised prior at its root, the trolls' a noised one, the evaluation search none; the
control: with no list both are noised) — all pass; in `az_trainer_d1`.

**C4. No dwarf games against alpha-beta's trolls** (keep alpha-beta as the dwarfs): those
games end at −32 and give 10.9% of the dwarfs' positions the target −1. Down: the dwarfs'
policy loses the teacher's signal in hard games; run J's depth-1 trolls did not help. Dev
~1 hour.

**C5. Weaker current trolls** (fewer troll simulations, `--player_simulations`; a smaller
trolls' teacher weight; fewer games against alpha-beta's dwarfs, whose games made the
trolls strong in run I). Down: weakens the trolls on purpose; their prior picks a capture
as its top move whenever there is one (K83: 100%). Dev 0-1 hour. Not now.

**C6. An outside engine's dwarfs as value data** (hexparrot's dwarfs keep dwarfs in 30 of
40 battles against M102). Down: an outside engine in training, its licence to check, a
Python process per actor. Superseded by C7. Not recommended.

**C7. Alpha-beta's own dwarf positions as value-only data.** In the games against
alpha-beta's dwarfs that the trainer already plays (366 in run M), alpha-beta's positions
are discarded. Record them with their outcome as the value target and no policy target.
- **Reverses the user's rule of 2026-10-07** (§3) — for the value only, nothing imitates
  alpha-beta's moves; the user decides.
- Up: no extra games; the rules only; exactly the data measurement 4 found missing —
  dwarf-to-move positions of games in which dwarfs survive (the trolls-to-move side of
  those games is already recorded, ~26% of the positions, and there the value is right).
- Down: off-policy — the value learns outcomes of alpha-beta's dwarf play, mixed with our
  own and indistinguishable from it; those games end mostly at the no-capture limit
  (stalls), a style our dwarfs may not reach.
- **Mechanics** (the trainer's code, checked): an empty policy target gives a zero row of
  policy loss (`vpnet.cc` fills the targets from the stored pairs; `model.cc`'s
  cross-entropy over masked, finite logits; augmentation maps an empty policy to an empty
  one; the buffer copies the vectors) — but recorded naively, ~5,800 such positions a
  step would:
  - dilute the policy gradient by about a quarter (`torch::mean` over the batch);
  - displace ~26% of each step's own positions (the learner fills a step by counting
    positions, `num_states < learn_rate`), shortening the buffer's memory of our own play
    from ~21 to ~16 steps;
  - change the alpha-beta share (the actor counts `trajectory.states` towards it: the
    30% would fill with fewer games, for both sides);
  - corrupt the statistics (per-side losses, `vs_alphabeta`, the overfitting measure's
    samples, `Trajectory::State::value`).
  So: a separate stream outside the share's and the step's accounting; the policy loss
  averaged over the rows that have a target; C7's positions capped at ~10% of a batch;
  statistics reported without them.
- Dev ~1 day. Train: buffer space and learner batches only (or ~1.2 GB more buffer to keep
  21 steps of our own data — not with swap as it is); run 0.

### Group D — the policy and the search

**D1. A higher kickstarting weight for the dwarfs** (per side: the trolls stay at 0.3).
The teacher shapes the policy that **generates the data**, so a larger teacher share for
the dwarfs moves self-play toward the play that keeps dwarfs alive (§1.2) — a change of
distribution, which should then give the value the dwarf-to-move data measurement 4 found
missing. (This revised, on 2026-10-12, the earlier assessment "treats the symptom".)
- **Measurement 2**: the teacher in M102's dwarf prior at play time gains the dwarf side +7.6 /
  +10.6 / +15.9 points a battle at 0.3 / 0.5 / 0.7 against L102's trolls, all battles
  keeping dwarfs at 0.5 and 0.7, troll kills not lower, our trolls unchanged within the
  noise — in matches without root noise, the dwarfs surviving by stalls.
- Up: the most direct lever on our network's dwarf play, measured at play time; the
  teacher already runs at every recorded position (no extra compute); at 0.7 the
  target's mass on unsafe moves falls from ~0.26 to ~0.14, below their ~0.19-0.21 share of
  the legal moves, and the loop that amplifies unsafe moves (§2) weakens (its gain ~0.93
  at 0.3, ~0.4 at 0.7, if the inference holds).
- Down:
  - **Its chain is open at the second step**: better dwarf play → self-play keeps dwarfs →
    dwarf-to-move data for the value. Whether teacher-guided dwarfs keep dwarfs under the
    trainer's root noise is not known — measurement 3: either side's noise breaks the stalls
    in which the unmixed network's dwarfs survive. **Measurement 2b decides before the run**:
    D1 alone, D1 with C3, or D1 for the policy only (the value then needing C7 or B1′).
  - **The network must learn it**: at play time measurement 2 ran a real 2-ply search at every
    dwarf node; trained, the prior only approximates the teacher (the dwarfs' policy loss
    is 4.92 now, against ln 285 = 5.65 for a uniform prior).
  - **It builds up slowly**: its targets are fixed at insertion (§3) — at 0.7, the
    buffer's mean dwarf weight rises from 0.30 to ~0.66 by step 121. The run tests D1
    mostly, not fully (run K's kickstarting worked through the same delay).
  - **Defence, not attack**: measurement 2's gain is mostly dwarfs kept (14.6 left at 0.7), the
    trolls killed rising only from ~1.4 to ~2.0 — the 2-ply teacher teaches safety;
    attacking has to come from the search and the value. A 2-ply teacher cannot see a
    quiet move that sets up a hurl; the troll-kills gate (§7) watches for that loss.
  - **Long, stalled games** if it works (measurement 2: 435 plies at 0.7): fewer games a step,
    many capture-free positions (measurement 2b).
  - Dependence on the teacher until it is tapered; the prior (now near uniform, ~205
    effective moves of 285) concentrates onto the teacher's best moves, many tied at
    depth 2, among which the search chooses.
- **Weight: 0.7, capped** (the user, 2026-10-13: "let's keep the 0.7 cap. That needs to be
  enough. If it's not enough then we need to taper … later than anticipated"). Measurement 2's
  rule (the best mix that does not cost troll kills) picks 0.7, the largest tried; the
  effect had not levelled off, and 0.85 or 1.0 might do better still at play time. But as
  a training weight, 1.0 makes the dwarfs' policy target pure 2-ply alpha-beta, and 0.85
  nearly — the imitation the user ruled out on 2026-10-07 ("not imitation first",
  `PLAN.md`, training against alpha-beta) — and its cost (what the visits know beyond 2
  plies) shows only in training, not in measurement 2. At 0.7 the visits keep 30% of the
  target. If 0.7 proves too little, the teacher stays on longer, not stronger. (Measurement 2 at
  0.85 and 1.0 is therefore not run; `~/thud-runs/check2_more_2026-10-13.sh` would do it.)
- **Taper, slow and gated** (the user, 2026-10-13: "I'd rather be safe than sorry";
  approved): not within the first 19 steps; then −0.1 at a time, at most once per buffer
  length (~20 steps, the time a change takes to fill the buffer), and only when (a) our
  dwarfs play the dwarfs at least as well as the teacher does against the same trolls,
  both measured without noise (matches: our network's dwarfs against alpha-beta 2's
  dwarfs, each against our trolls; in training the teacher's −16 is noise-free while ours
  is under root noise, so that comparison alone might never open), (b) the regression
  gates of §7 hold at that step, and (c) **the search no longer amplifies unsafe moves**:
  the visits' mass on unsafe dwarf moves is no higher than the prior's (from the recorded
  targets; §2's loop — at a lower weight the loop's gain rises again; **decided stricter**
  (the user, 2026-10-15, to revisit once in use): the loop's gain after the step, (1 − w′) × R with
  R the visits' unsafe mass over the prior's at its 95% upper bound, at most 0.8 — R ≤ 2.0
  for 0.7 → 0.6, ≤ 1.33 for 0.5 → 0.4, ≤ 1.0 for 0.3 → 0.2, ≤ 0.8 without a teacher — so
  the rule tightens as the teacher fades; R needs a probe not yet built), and (d) **the
  value's side-to-move gap has closed** (measurement 4 on the current network: a far dwarf
  counts about as much with the dwarfs to move as with the trolls to move — the user,
  2026-10-15; **closed** (the user, 2026-10-15): with the dwarfs to move a far dwarf worth
  at least 80% of its worth with the trolls to move, in two measurements in a row, every
  10 steps; while the teacher carries the dwarfs' policy the gap costs little, but
  once it is tapered the policy targets come from the search, which relies on that
  value, and an open gap would bring the trap back). 0.7 → 0.3 (the trolls' weight) thus
  takes ≥ 80 steps (~4-5 nights here); later both sides together
  toward a floor of ~0.1, as planned for the alpha-beta share. Never raised again
  automatically. (Kickstarting's paper adapted the weight by population-based training,
  the teacher's influence falling as training went on, Schmitt et al. 2018; we cannot
  afford a population, hence results and gates.)
- **Built 2026-10-13**: `--player_kickstart_weight` in our trainer (`thud/az/`; "0.7": the
  dwarfs' positions 0.7, the trolls' the single `kickstart_weight`; empty: as before);
  `kickstart_test` extended (the same moves played; every dwarf position's target 0.3 ×
  the visits + 0.7 × the teacher's; every troll position's unchanged; an empty list
  changes nothing; the configuration round-trips) — all pass; the trainer built as
  `build-shared/az_trainer_d1`, for both the control run (empty list) and the D1 run; a
  4-minute dry start from a copy of M102's state resumed and logged "Kickstarting: weight
  0.3 (per player 0.7, 0.3)". The run folders are prepared from M102's state (its
  network, optimizer, buffer and learner log; to step 121):
  `~/thud-runs/runM_K83_conv_planes_cont` (the control) and `~/thud-runs/runN_M102_d1`
  (`player_kickstart_weight` "0.7"); the runner `~/thud-runs/night8_2026-10-13.py
  control|d1`. Train 0; run 0.

**D2. The teacher in the search's prior at play time** — measurement 2's mechanism (`az_match
teacher_mix_a`). Not a training change; costs ~1.5× the time of a search. An option for
play only, and a measuring tool.

**D3. A moves-left head** (Leela Chess Zero since v0.26: remaining game length, used in
search to prefer shorter wins and longer losses). With every dwarf lost anyway it only
delays the loss. Dev ~1½ days. Not now.

### Group E — how well the value learns small effects

**E1. A larger value-loss weight** (the value loss 0.035 against the policy loss 3.86 and
the L2 term 0.048). Up: one flag; the trainer's overfitting measure shows no value
overfitting now (step 102: 0.021 on trained positions, 0.015 on new ones), and each
position is sampled only ~3 times. Down: measurement 4 says the data, not the learning, are the
limit — the value learnt a dwarf's worth wherever its data showed it; the trunk's
capacity shifts toward the value; may start to overfit (watch the same measure). Dev ~1
hour. **Measured 2026-10-17** (§10, *E1's offline sweep*): the loss sizes mislead — on the
trunk the value's gradient is already 1.5-6.6 times the policy's at weight 1 (N161, as the
trainer learns), the two at a cosine of about −0.5; under Adam, which normalises each
parameter's step, a larger weight mostly turns the trunk further toward the value, at the
policy's expense, so little is expected; the sweep tests it.

**E2. A categorical margin head**: 65 bins over the margins −32..32, the value its
expectation (MuZero's categorical support, KataGo's score distribution, Leela Chess Zero's
win/draw/loss head). Removes tanh's compression (factor 4 in §2); the targets are the
integer margins, z × 32 — no buffer change (mixing in B1 or B1′ later needs MuZero's
two-hot encoding). Does not change the fixed point. Dev ~1-2 days (model, loss, inference,
tests); train small; run small.

## 6. Comparison

| | Lever | Evidence (measurements) | Dev | Train | Run | Main risk |
|---|---|---|---|---|---|---|
| A1 z + q | variance | the dwarfs' noise is worth ~1.8 points (3) | 2 h | – | – | no effect |
| A2 short-term heads | variance | – | 1 day | small | – | no effect; buffer change |
| A3 n-step / shaping | variance | – | 2 h | – | – | no effect |
| B1 material ahead | bias, tapered | – | 3 h / 1 day | – | – | delay by the buffer; diluted signal |
| B1′ teacher score in the value | bias, tapered | the gap is at dwarf-to-move positions (4) | 3 h / 1 day | – | – | myopic; delay |
| B2 material head in the search | bias in the search | – | 1½ days | small | small | complexity |
| B3 auxiliary heads | representation | – | 1 day | small | – | no effect |
| C1 older trolls | data | 9 of 40 under noise, all by wiping out the trolls (5) | 1 day | >2× for those games | – | weak-troll exploits; memory |
| C2 handicap / archive starts | data | – | ½ day | – | – | transfer |
| C3 per-side noise | data; perhaps D1's precondition | rule met; outcomes stay saturated (3); measurement 2b | 1-2 h | – | – | alone, no surviving dwarfs |
| C4 no games against alpha-beta's trolls | data | – | 1 h | saves | – | small |
| C5 weaker trolls | data | – | 0-1 h | lower | – | weaker trolls |
| C6 outside engine | data | – | 1 day | Python | – | not ours |
| C7 alpha-beta's dwarf positions, value only | data | exactly the missing data (4) | 1 day | buffer share | – | off-policy value; the user's rule; mechanics |
| **D1 dwarfs' teacher weight** | **the play generating the data** | **+7.6 to +15.9 at play time, by stalls, without noise (2); measurement 2b** | **built** | – | – | survival under noise; learning it; defence only |
| D2 teacher at play time | search | (2) | done | – | 1.5× | play only |
| D3 moves-left head | search | – | 1½ days | – | – | delay |
| E1 value-loss weight | learning | the data are the limit (4) | 1 h | – | – | overfitting |
| E2 categorical margin head | learning | – | 1-2 days | small | small | complexity |

## 7. Validation

**The control run**: M102 continued with run M's recipe to the same step (the same
binary, `az_trainer_d1`, with no per-player weight); one control serves every test run
that starts from M102 with the same number of steps. **Both final networks are also
compared with M102**: self-play has been sliding within every run (§1.3), so the control
may get worse, and a test run that merely slides less would "win" against it.

**The rule, settled by the user on 2026-10-14** — before any of the evaluation's results
had come in (the precisions are review 3's estimates from M102 and L102 on the same
openings):

| Measure | Battles or pairs | 95% half-width | Use |
|---|---|---|---|
| Head-to-head, the test run's last network against the control run's | 40 pairs | ~2.3 a pair | primary |
| The dwarf side against hexparrot, against the control's and M102's on the same openings | 40 battles | ~2.9 | primary |
| The dwarf side against alpha-beta depth 1 (M102's dwarfs −15.2: room both ways) | 40 battles | ~3.3 (4.7 at 20) | primary |
| The trolls against alpha-beta depth 2's dwarfs (M102's trolls +13.2: room both ways) | 40 battles | ~3.3 (4.6 at 20) | non-inferiority |
| The trolls against alpha-beta depth 3's dwarfs; against hexparrot | 20; 40 battles | 7.9; 6.5 | for the record: too imprecise to decide |
| The dwarf side against alpha-beta depth 2 | | | not used: at the floor (−31.2) |

- **A test run wins if** the head-to-head's 95% interval lies above 0 — a one-sided test
  at 2.5%, as the promotion rule's (the user kept it so: power is not the problem, and a
  win is rerun anyway) — **and** on each
  dwarf-side measure it is not worse than the control's by more than 2 standard errors,
  **and** on the trolls' measure it is not worse than the control's by more than 2
  standard errors; and it is not worse than M102 on the dwarf-side measures.
- **Replication**: before adopting a winner, rerun it (no seed: a rerun), and the control
  too unless the control proved strong. (For run N the user chose a rerun of run N only,
  2026-10-15: C121 beat M102 by +7.20, so the main risk is a lucky run N.)
- **Secondary, for the record**: the ladder; alpha-beta depth 2 as trolls; the
  head-to-head per side.

**Regression gates** (the user asked for them and approved them, 2026-10-13; for the
promotion rule too, `PLAN.md`; refined after review 3): the promotion rule asks "is the
new network better?" (head-to-head and ladder); the gates ask "did it lose a skill?" — a
network must pass both. **The anchor is the top anchor, M102.**

| Gate | Measure | Fails if |
|---|---|---|
| Alpha-beta, per side | our dwarfs against depth 1; our trolls against depth 2's dwarfs; 40 battles each on the anchor's openings | worse than the anchor's by more than 2 standard errors of the paired difference (~3.3 points) |
| Blunders | the blunder rate against hexparrot (`az_blunder_probe`); the standard error clustered by battle (~2.0 percentage points for the difference of two networks) | ≥ 4 points higher than the anchor's |
| Troll kills | trolls our dwarfs kill a battle, against hexparrot and alpha-beta depth 1 — what a 2-ply teacher cannot teach (set-ups) | lower than the anchor's by more than 2 standard errors |
| Short hurls (2-3 squares) | `az_long_capture_probe`'s built positions: the hurl's top-move share and its ratio to a blind prior | lower than the anchor's by more than 2 standard errors (bootstrap over positions) |
| The trolls' captures | at troll positions with a capture, the prior's top move a capture (K83: 100%) | lower by more than 2 standard errors |
| Long hurls (4+ squares) | the same probe | never fails: monitored now, a target for later |
| Dead channels (2026-10-18) | `az_dormant_probe` on 4,096 buffer positions: channels never above 0, in the trunk and in the heads, and dormant ones (τ 0.1), beside the anchor's and a fresh network's | never fails: monitored, recorded below |
| Plasticity (2026-10-18) | `az_plasticity`: random targets on 8,192 positions, 1,000 batches at 0.001, the policy + value loss at the end, beside a fresh network's of the same width in the same run, and the weights' norm | never fails: monitored, recorded below |
| Late captures (2026-10-19) | captures our network makes against hexparrot (the limit 200) after more than 100 turns without one, a battle, as trolls and as dwarfs, beside the anchor's — the indicator for self-play's no-capture limit of 100 (R191: trolls 0.12, dwarfs 0.03; N161: 0.25, 0.00) | never fails: monitored |
| Trunk gradients (2026-10-19) | `az_restructure`'s report before its first batch, on 1,024 positions of run N's last buffer: the weighted value loss's gradient on the shared layers against the policy loss's (the ratio of their sizes) and the cosine between the two (+1: both losses ask the same change, 0: unrelated, −1: opposed — the multi-task conflict) | never fails: monitored, recorded below |

- **Standard errors analytic or by bootstrap** over battles or positions — not from one
  pair of checkpoints (M100 against M102 differ for real: the blunders were rising).
- **False alarms, stated**: each gate at 2 standard errors, one-sided, trips ~2.5% of the
  time on an unchanged network; the five gates (seven measures) together in up to ~15% of
  evaluations. So a failed gate stops a promotion or a taper step **until looked into**
  (a rerun of that measure with more battles first), not automatically.
- **Each gate with a positive control**: a network known to be worse must fail it (K83
  the short-hurl gate against M102 — it ignored 2-square hurls; G'64 the blunder gate).
- The short-hurl gate guards a skill the convolutional head brought; it does not watch
  D1's own risk (an available hurl is a one-ply capture, which the teacher rewards) —
  that is the troll-kills gate.
- **The capacity tests, monitored** (the user, 2026-10-18: not to block a release, but to
  watch how they develop): every evaluation of a candidate runs `az_dormant_probe` and
  `az_plasticity` beside the matches, `regression_gates.py --dormant --plasticity` reports
  them, and each network measured gets a row in the record below. What they caught: Np181,
  a strong network, fitted random targets to 5.0 where fresh networks reach 0.13-0.16 —
  and a fresh network trained on its buffer beat it by +13.2 (§10, *The capacity tests*).
  Dormancy at τ 0.1 has a floor of ~8 (a fresh network's value head); a fresh network's
  plasticity varies 0.13-0.16 between runs (its initialisation), so compare a network with
  the fresh one of its own run.

**The capacity record** (dead channels in the trunk and the heads, dormant at τ 0.1, of
705 rectifier channels at 64 x 4; plasticity: random targets' policy + value loss after
1,000 batches, lower is better; the weights' norm; the trunk's gradients: the weighted
value's pull against the policy's, and their cosine — in training mode, "eval." where
measured in evaluation mode, as those networks learnt):

| Measured | Network | Trunk dead | Heads dead | Dormant | Plasticity | Norm | Value/policy | Cosine |
|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | N161 | 0 | 8 | 11 | — | — | 2.60 | −0.52 (eval.) |
| 2026-10-18 | N161c (N161 converted) | — | — | — | 5.015 | 48.1 | — | — |
| 2026-10-18 | N′ | 0 | 8 | 12 | — | — | 3.67 | +0.06 |
| 2026-10-18 | Np171 | 0 | 8 | 13 | — | — | 8.18 | −0.04 |
| 2026-10-18 | Np181 | 0 | 8 | 14 | 5.012 | 44.8 | 7.79 | −0.20 |
| 2026-10-18 | No171 | 0 | 8 | 11 | — | — | 10.24 | −0.51 (eval.) |
| 2026-10-18 | the reset after 250 batches | — | — | — | 0.325 | 29.8 | — | — |
| 2026-10-18 | the reset after 1,000 batches | 0 | 2 | 24 | 0.419 | 27.3 | — | — |
| 2026-10-18 | the reset after 3,000 batches | 0 | 3 | 23 | 0.218 | 22.0 | — | — |
| 2026-10-18 | R181 (the reset network) | 0 | 3 | 17 | 0.346 | 23.4 | — | — |
| 2026-10-18 | a fresh network (B's run) | 0 | 0 | 8 | 0.127 | 30.3 | — | — |
| 2026-10-18 | a fresh network (the control's run) | 0 | 0 | 8 | 0.157 | 30.3 | — | — |
| 2026-10-19 | R191 (run R, segment 1) | 0 | 3 | 18 | 0.459 (fresh 0.188) | 23.1 | 2.54 | +0.19 |

**Mechanism** (explains, does not decide) — each with the tools of §4, on the test run's,
the control run's and M102's networks:

- the self-play endings, lengths and margins in the trainer's log, and the share of
  positions after a battle's last capture (stalls); measurement 3's "both" cell on the new
  network (dwarfs lost by ply 60, battles keeping dwarfs);
- **measurement 4** (the value by side to move: does the dwarf-to-move gap close? does the
  value's level still split by side?);
- blunders in self-play by thirds of the run (the slide) and against hexparrot; the
  dwarfs' prior and visits on unsafe moves (the amplification of §2); the risk probe by
  distance;
- for D1: the dwarfs' agreement with the teacher's top moves over the run, and trolls
  killed.

**Controls in code**: no per-player weight (D1), no second stream (C7), b = 0 (B1) each
reproduce the old targets exactly (D1: `kickstart_test`).

## 8. Recommendation

1. **Measurement 2b** (done, 2026-10-13): under the trainer's noise teacher-guided dwarfs keep
   no dwarf; with the dwarfs' roots noise-free 70% of the battles keep dwarfs — **the
   test run is D1 together with C3**.
2. **Then two nights from M102, 19 steps each**: the control run and the D1 run at a
   dwarfs' teacher weight of 0.7 (the trolls at 0.3; 0.7 is the cap, §5) and no root
   noise at the dwarfs' roots (C3: `player_policy_epsilon` "0,0.25"; two changes in one
   run, D1 alone as a third night if they must be told apart). Both from the same binary; a `wsl --shutdown` was done
   (the 8 GB swap is in). Meanwhile the regression gates of §7 set up (their standard
   errors, their positive controls; ~½ day). Then about a day of evaluation by §7 —
   against each other and against M102 — the gates included, with the mechanism read off.
3. **Then, by the mechanism:**
   - D1 wins and its self-play keeps dwarfs, measurement 4's gap closing → rerun D1 and the
     control (replication), adopt; taper the dwarfs' weight slowly and gated (§5).
   - D1 wins but the dwarf-to-move gap stays → a further test run for the value: **B1′**
     (the teacher's score in the value target, applied at sampling) or **C7** if the user
     agrees to reverse the rule of 2026-10-07 (~1 day of development each).
   - D1 does not win → C7 or B1′ next; C1 after them (measurement 5: its survivals are troll
     wipe-outs).
4. **Later**: B1 only with b applied at sampling and sized to a measured signal (measurement 6);
   C2 (after measurement 7); E2 on the GPU. **Not now**: A1-A3, B2, B3, C4-C6, D3, E1 (E1
   since tested after all, the longer run's gap having reopened: §10, *E1's offline
   sweep*, 2026-10-17).

## 9. Open questions

- Whether the search amplifies unsafe moves as §2 infers (the visits' against the
  prior's mass on unsafe dwarf moves, by step) — to measure on run M's records and in the
  runs.
- The gates' standard errors and positive controls, once measured (§7).
- Whether run M's late value drop (§1.4) matters for runs that start from M102 — the
  control will show whether it persists.

## 10. Results of the test runs (2026-10-14)

Both from M102's state, steps 103-121, `az_trainer_d1`: the control run (run M's recipe;
done 2026-10-14 05:29) and **run N, D1 + C3** (the dwarfs' teacher weight 0.7, their
roots without noise; 05:30-16:55, cut off at ~07:35 by a Windows update's restart and
resumed from step 107 at 11:00). In training, run N's self-play changed steadily: games
150 → 246-270 plies, the dwarfs' margin −24.4 → −22.9 (the control's −25.3 to −25.8),
3-5% of games ending with dwarfs left (0-1% at the start), the dwarfs' prior
concentrating (~200 → ~110 effective moves); our trolls against alpha-beta's dwarfs
unchanged. **The evaluation** (`~/thud-runs/night9_2026-10-14.py`, results in
`~/thud-runs/eval_2026-10-14/`; 40 pairs each, as trained, no noise):

| Measure | N121 | C121 (control) | M102 | N121 − C121 (paired) | The rule (§7) |
|---|---|---|---|---|---|
| Head-to-head N121 against C121 | | | | **+5.50 a pair (+1.91 to +9.09)**, 30 won, 1 drawn, 9 lost | interval above 0 ✓ |
| The dwarf side against hexparrot | −10.2 | −19.6 | −23.4 | **+9.3 ± 2.2** | not worse ✓ |
| The dwarf side against alpha-beta 1 | −9.5 | −13.2 | −14.7 | +3.7 ± 1.9 | not worse ✓ |
| The trolls against alpha-beta 2's dwarfs | +15.9 | +14.8 | +13.0 | +1.1 ± 1.2 | not worse ✓ |
| Against M102, the dwarf side (hexparrot / alpha-beta 1) | | | | N121 − M102: +13.1 ± 2.0 / +5.2 ± 2.4 | not worse ✓ |

**By the rule, run N wins.** Head-to-head against M102: N121 +12.62 a pair (+9.83 to
+15.42, 36 of 40), C121 +7.20 (+3.95 to +10.45) — the control did not slide this time.
For the record: the trolls against hexparrot +6.5 / +5.2 / +1.3, against alpha-beta 1
+30.3 / +30.7 / +30.2; the dwarfs against alpha-beta 2 at the floor (−30.6 / −30.8 /
−30.9). **Blunders against hexparrot** (`az_blunder_probe`): N121 **9.8%** of its dwarf
moves (429 of 4,385 — longer battles), C121 21.6% (598 of 2,772), M102 25.4%.

**The mechanism:** the gain is the policy's, not yet the value's. Measurement 4 on N121 (run N's
positions): a far dwarf still counts with the trolls to move (+0.019) but hardly with the
dwarfs to move (+0.003 in the games against alpha-beta's dwarfs, +0.010 in self-play) —
as C121 and M102 on the same positions; the gap has not closed in 19 steps, with only
3-5% of run N's self-play keeping dwarfs. In self-play's conditions with noise at *both*
roots (measurement 3's "both" cell, not run N's own C3 setting) N121 still keeps no dwarf (0 of
80; 22.1 dwarfs lost by ply 60 against M102's 23.6).

**The replication** (2026-10-15; the user: rerun run N only): **run N2**, run N's recipe
again from M102 (steps 103-121, 01:56-10:13), evaluated as run N was (10:16-12:33)
(`~/thud-runs/night10_2026-10-15.py`, results in `~/thud-runs/eval_2026-10-15/`; 40
pairs, the same seeds; differences paired by battle, as in the table above):

| Measure | N2121 | N121 | N2121 − C121 (paired) | N2121 − N121 (paired) | The rule (§7) |
|---|---|---|---|---|---|
| Head-to-head against C121 | **+6.10 a pair (+3.09 to +9.11)**, 30 won, 10 lost | +5.50 (+1.91 to +9.09) | | | interval above 0 ✓ |
| The dwarf side against hexparrot | −10.7 | −10.2 | **+8.9 ± 2.4** | −0.4 ± 2.8 | not worse ✓ |
| The dwarf side against alpha-beta 1 | −10.5 | −9.5 | +2.6 ± 2.0 | −1.1 ± 1.8 | not worse ✓ |
| The trolls against alpha-beta 2's dwarfs | +14.2 | +15.9 | −0.6 ± 1.3 | −1.7 ± 1.7 | not worse ✓ (−0.6 > −2.6) |
| Against M102, the dwarf side (hexparrot / alpha-beta 1) | | | N2121 − M102: +12.7 ± 2.1 / +4.2 ± 2.3 | | not worse ✓ |

**Run N2 meets the rule too: the recipe's gain replicates.** The two runs of one recipe
against each other: N2121 against N121 **+0.95 a pair (−2.04 to +3.94)**, 20 won, 2
drawn, 18 lost, and every per-side difference within 1 standard error — run-to-run
variation at 19 steps is small against the recipe's effect (+5.5 and +6.1 against the
control). Blunders against hexparrot: N2121 13.1% of its dwarf moves (462 of 3,532),
N121 9.8%, C121 21.6%, M102 25.4%. Measurement 4 has not been run on N2121.

**Next, by §8:** D1 + C3 is replicated; adopting it for the main line is the user's
decision. It falls under step 3's second branch — it wins, but the dwarf-to-move gap stays
after 19 steps — whose next step is a test run for the value (B1′, or C7 with the user's
word), unless a longer D1 + C3 run comes first (below). Meanwhile the regression gates set
up and the promotion tests (the ladder; the gates) for N121 or N2121; whether to continue D1
+ C3 for longer, so that self-play's surviving games grow and the value's gap can close, is
the next question. If it continues: measurement 4 every ~10 steps beside the share of self-play
games keeping dwarfs — both rising: patience; survival rising but the gap not: the value is
not learning what it sees (E1, E2); survival staying below ~5%: the data never come (B1′, or
C7 with the user's word; C4, dropping the games that give the dwarfs' positions −1, as a
cheap help either way). The gap's closing is a condition of the taper (§5).

**Decided** (the user, 2026-10-15): the longer D1 + C3 run first — run N continued from
N121, steps 122-161 (`~/thud-runs/runN_d1_cont`, from 14:30), in four segments of 10 steps,
after each measurement 4 on the segment's own games (the newest network and N121 on the same
positions) and the share of self-play games keeping dwarfs (run N's last 10 steps: 2.2%
of 693 games, the dwarfs' margin −23.4), read by the rule above.

**The first segment** (steps 122-131, 2026-10-15 14:30 to ~19:00, slowed by work beside
it): **12.7% of 472 self-play games kept dwarfs** (2.2% before), the dwarfs' margin −20.4.
Measurement 4 on the segment's own positions, N131 against N121 on the same positions — a
far dwarf's worth with the dwarfs to move: self-play positions +0.0163 against +0.0121,
65% of its worth with the trolls to move (N121 46%); the games against alpha-beta's
trolls +0.0031 against +0.0019, 18% (N121 10%). Both rising, both below the taper's 80%.
The placebo (the dwarf moved one square) is ~+0.007 at self-play positions, not ~0 (at the
games against alpha-beta ~0): part of the measured worth there is a sensitivity to the
position as such — net of it the self-play ratio is 51% (N121 29%). By the rule above:
survival and the gap both rising — patience.

**Condition (c) measured** (2026-10-15, `az_unsafe_targets`, 200 dwarf positions of the
segment's self-play, 100 simulations): **N131: R = 0.70** (upper end 0.85; prior 0.081,
visits 0.057) — at C3's settings (no noise at the dwarfs' roots) the search reduces unsafe
moves; a step 0.7 → 0.6 would be allowed (gain 0.34). M102 on the same positions: R = 0.87
(1.02) without noise, **1.05 (1.13) with its own noise (ε 0.25)** — the noise is what
made the search amplify unsafe moves (§2 had inferred ~1.3 from M102's own training,
without measuring it); and the teacher halved the prior's unsafe mass (0.16 → 0.08).

**The whole run** (steps 122-161, done 2026-10-16 09:29; measurement 4 on each segment's own
positions, N121 alongside on the same ones; the ratio is a far dwarf's worth with the
dwarfs to move over its worth with the trolls to move):

| Segment | Self-play games keeping dwarfs | The dwarfs' margin | Ratio, self-play positions (N121) | Ratio, games against alpha-beta (N121) |
|---|---|---|---|---|
| run N's last 10 steps | 2.2% | −23.4 | — | — |
| to 131 | 12.7% | −20.4 | 65% (46%) | 18% (10%) |
| to 141 | 23.0% | −20.7 | 73% (40%) | 44% (13%) |
| to 151 | 27.0% | −21.8 | 50% (37%) | 15% (14%) |
| to 161 | 22.7% | −23.4 | 47% (28%) | 3% (14%) |

Survival rose fast to step 151, then fell back a little; the gap narrowed to step 141, then
reopened — below the taper's 80% throughout. The placebo at self-play positions
(~+0.008-0.010) is as large as the dwarf-to-move effect at steps 151-161, so the ratios
there are uncertain. By the rule above: **survival rising but the gap not closing — the
value does not learn what it sees** (E1, the value-loss weight, built 2026-10-15; E2).
The margin's return to −23.4 at higher survival wants a look (the trolls improving too?):
the evaluation of N161 against N121 and the players without a network comes first.

**N161's evaluation** (2026-10-17, `~/thud-runs/night12_2026-10-16.py`,
`~/thud-runs/eval_2026-10-17/`; 40 pairs, as N121's): **N161 beats N121 by +4.12 a pair
(+1.26 to +6.99)**, 30 won, 1 drawn, 9 lost; M102 by +11.22. Against the players without a
network as N121: alpha-beta 1 +17.4 (N121 +20.8; its dwarfs −12.7 against −9.5), alpha-beta
2 −13.8 (−14.6; its trolls +16.2 against +15.9), hexparrot −4.6 (−3.8). **The regression
gates against M102 all pass**, the blunders down to 5.9% of the dwarfs' moves (N121 9.8%,
M102 25.4%). Condition (c): R = 0.65 (upper 0.79) — the search no longer amplifies unsafe
moves, even without a teacher; the taper waits on condition (d), the value's gap.

**Measurement 4 with intervals** (2026-10-17; `az_dwarf_value_probe each=` and
`thud/experiments/dwarf_value_report.py`: 95% intervals from a bootstrap that resamples
whole games — a game's positions are not independent — and differences between networks
paired on the same positions). The table above, self-play positions: N131 65% (56-75),
N141 73% (65-82), N151 50% (39-62), N161 47% (31-65). Each segment's positions differ, and
N121 measured on each falls too (46, 43, 38, 28%), so the table alone cannot separate the
networks from their positions. On the same positions it can: on steps 132-141's, **N151 −
N141 = −24 points (−33 to −14)**, N161 − N151 +1 (−11 to +17); on steps 152-161's, −20
(−29 to −10) and −1 (−11 to +10). So the gap did reopen between steps 141 and 151, and stayed
reopened — the reading above stands. Net of the placebo the self-play ratios are
uncertain at the end (N161 −11%, −54 to +26).

**E1's offline sweep** (2026-10-17, `~/thud-runs/e1_sweep_2026-10-17.sh`, results in
`~/thud-runs/e1_sweep_2026-10-17/`; the user: build it and run it, larger weights too, with
intervals). N161 trained on on run N's last buffer (458,752 positions, steps ~141-161;
every 20th block of 1,024 held out) by `az_restructure`, 640 batches of 1,024 (10 trainer
steps, ~35 minutes an arm), then measurement 4 at 320 and 640 batches on steps 142-161's
games, each arm paired with weight 1 at the same seed. Three findings while building it:

- **Our trainer learns with its batch norms in evaluation mode, in effect.** In true
  training mode N161's losses on batches of its own buffer are value 0.13, policy 3.96
  (evaluation mode: 0.029, 3.84) — yet the trainer reported 0.027 and 3.85 as it trained.
  The cause: on one device the inference threads serving the actors share the learner's
  network, and each `Inference` sets it to evaluation mode while `Learn`'s step runs
  (upstream's design; reproduced: one inference thread beside the steps brings their
  losses from 0.12-0.14 down to 0.023-0.027; PLAN.md, *Upstreaming*, candidate 4). So
  offline training that carries over must keep the batch norms in evaluation mode
  (`batch_norms=evaluation`). The batch norms' own counters agree: in step 161 the input
  batch norm ran in training mode for 7 of the 64 learning batches, the deeper ones 11-51
  (rising with depth: apparently inference calls under way when the learner switches to
  training, finishing their deeper layers in training mode). With the batch norms kept in
  evaluation mode the control at weight 1 holds N161's held-out losses
  (value 0.0311 → 0.0296 → 0.0313, policy 3.797 → 3.794 over 40 batches); in training
  mode they rose to 0.056 and 3.85.
- **The value already leads the trunk** (E1 above): at weight 1 its gradient there is
  1.5-6.6 times the policy's, cosine −0.4 to −0.5. A larger weight therefore changes the
  direction of the trunk's steps more than their size; the expectation is a small effect.
- **The optimizer matters**: a fresh one damages N161 (held-out value loss 0.031 → 0.16 in
  40 batches), so every arm continues N161's own (`optimizer=source`, tested against the
  source's own step). But Adam's second moments are N161's at weight 1: a new weight w
  first raises the effective rate of the parameters the value leads — by about 1.4-2.6x
  over these 640 batches at w = 4 to 16, estimated from Adam's update rule (not measured). Hence a control arm: weight 1 at twice the rate — if a
  weight's effect is the rate's, the two agree.

**Stopped the same day** after arm w1s1 (the user: the batch-norm finding first, kept apart
from E1 — with proper batch norms the gap may change). As planned: weights 1, 4, 16 and
the rate control (the user would have preferred 1, 16, 64), each with seeds 1 and 2 (the training's own
noise), a warm-up of 64 batches for all. **Reading** (set before the results): a weight
helps if its paired difference from weight 1 in the dwarfs' worth with the dwarfs to move
(and the ratio) excludes 0 at both seeds, exceeds the difference between weight 1's two
seeds, and is not matched by the rate control; its cost is the held-out policy loss. Only
then the best weight goes online against weight 1, and a finer sweep (its neighbours at
a factor of 2) only if the neighbours differ beyond the seeds' noise.

**The race fixed, and the two arms** (2026-10-17; the user: the batch-norm finding first,
apart from E1 — with proper batch norms the gap may change). Our copy now locks the
network against concurrent use and checks every learning step's batch-norm counts
(`thud/az/vpnet.h`; `batch_norm_test`, upstream's class as the control). N161 cannot just
go on learning in training mode: its stored statistics are far from those of what reaches
its batch norms (the conversion found channel means up to 1.8 deviations off and true
deviations up to 1.9x the stored ones), so training mode computes another function
(N161's value loss 0.13 against 0.03). **N161c**: its batch norms converted to the true
statistics, scale and shift adjusted (`az_batch_norm_convert`, 32 batches of its buffer):
evaluation-mode outputs within 7.7e-7 of N161's, and training mode's losses now within
0.0002 of evaluation mode's (N161: 0.2048). **N′**: N161c trained on 640 batches of the
buffer in training mode, w1s1's recipe exactly (seed, rate, warm-up), so that N′ and w1s1
differ in the batch norms' mode alone; compared by held-out losses, measurement 4 and a
match (`~/thud-runs/nprime_2026-10-17.sh`). Then **two arms of 10 self-play steps** (162-171,
run N's recipe, value-loss weight 1; `~/thud-runs/bn_arms_2026-10-17.sh`): B, N′ with the
fixed trainer (two copies, `explicit_learning`); A, w1s1 with `az_trainer_d1` (the race
as before); read by measurement 4 on fixed positions (paired), survival, the losses, the
batch norms' counts, self-play speed, and a match of the two step-171 networks.
Arm B's self-play kept dwarfs in 65.8% of its games (steps 162-171), arm A's in ~29%
(162-168), from the first step on — while N′ and w1s1 drew their match. Two causes are
open: the networks (N′ learnt in training mode) or the race in arm A's self-play (its
inference shares the learner's network: modes switched mid-call, weights written
mid-call). **Self-play without training** tells them apart: each of N′, w1s1, Np171 and
No171 against itself in run N's self-play conditions (`az_match share=1 noise=0,1
sample_plies=10 opening=0`), no learner beside it — w1s1 keeping dwarfs like N′ points to
the race, like arm A to the networks. Survival alone cannot say whether the dwarfs got
stronger or the trolls weaker: each side against a fixed player can — hexparrot:3 (it
saturates neither side), pair by pair against N161's match on the same openings, with the
gates' dwarf blunders. Then N′'s run goes on to step 201 in segments of 10, read as §10's
segments are (survival and the gap), with matches against N161 and hexparrot:3 after each
(`~/thud-runs/np_cont_2026-10-18.sh`).

**The arms' results** (2026-10-18; one seed each; `~/thud-runs/bn_fix_2026-10-17/`, the
queue's in `cont/`). Strength: **Np171 beats No171 by +7.6 a pair** (+4.3 to +10.9, 30 of
40) after 10 steps from equal starts (N′ against w1s1 −1.45). Each side against
hexparrot:3, against N161 on the same 40 openings (95% intervals), with the gates' dwarf
blunders (the share of dwarf moves whose dwarf the reply takes):

| Network | Dwarfs | Trolls | Blunders | Against hexparrot:3 |
|---|---|---|---|---|
| N161 | — | — | 5.9% | −4.6 a pair |
| w1s1 (640 batches, evaluation mode) | +8.5 (+4.4 to +12.5) | −0.4 (−4.7 to +3.9) | 2.6% | +3.5 |
| N′ (640 batches, training mode) | +10.6 (+6.2 to +14.9) | −1.0 (−6.3 to +4.3) | 2.3% | +5.0 |
| No171 (arm A, the race) | +1.9 (−1.7 to +5.5) | +2.0 (−3.0 to +6.9) | 4.4% | −0.8 |
| Np171 (arm B, fixed) | **+16.5** (+12.4 to +20.6) | +1.1 (−3.9 to +6.0) | 2.0% | **+13.0** |

So the dwarfs got stronger and the trolls no weaker. Two surprises: 640 offline batches on
run N's buffer improved the dwarfs in either mode — the old trainer's learning was neither
mode but a mix that changed from batch to batch and layer to layer, which clean training
in either mode avoids (an inference, untested) — and arm A's 10 steps lost most of w1s1's
gain while arm B's added to N′'s. Survival, the four networks against themselves in run
N's self-play conditions without a learner (80 battles each): N′ 61%, w1s1 54% (the starts
alike), Np171 61%, No171 34% — the networks diverged; and arm A's own first two steps kept
dwarfs in 34% of their games where w1s1 alone keeps 54% (arm B 61%, N′ alone 61%): the race
also spoilt arm A's self-play directly. The value (measurement 4, steps 142-161's
positions, paired): a far dwarf's worth with the dwarfs to move, in the games against
alpha-beta — where N161 gave it none (−0.0001) — Np171 0.0070, No171 0.0024 (+0.0046,
+0.0037 to +0.0055); its ratio to the trolls-to-move worth 0.27 against 0.10; in self-play
positions +0.0028 (+0.0016 to +0.0041). Still far below material (1/32 = 0.031): the gap
closes, faster with the fix. The trunk's gradients (1,024 positions of run N's buffer):
the weighted value's pull against the policy's grew in both arms (N′ 3.7x → Np171 8.2x,
training mode, cosine ~0; N161 2.6x → No171 10.2x, evaluation mode, cosine −0.5) — the
value already leads the trunk, so a larger value weight (E1) is not what these results
ask for. Self-play speed: 10 steps in 4h09m (B, two copies and `explicit_learning`)
against 4h34m (A).

**The capacity tests** (2026-10-18; the user: if the race cost capacity, restart cheaply;
`~/thud-runs/capacity_2026-10-18/`). A, dead and dormant channels (Sokar et al., ICML
2023): the trunk's 576 channels none in any network of the lineage; the value head's hidden
layer 8 of 64 dead in each, the same since N161. B, random targets (after Lyle, Rowland and
Dabney, ICLR 2022): fresh networks of widths 64, 48 and 32 fit them to 0.12, 0.24 and 0.51
(policy + value) in 1,000 batches; Np181 and N161c only to 5.0 — far below even width 32;
whether that is lost capacity or features specialised to Thud (the user's objection) the
reset network's checkpoints tell. C, the reset (Nikishin et al., ICML 2022: a fresh network
on the replay buffer): 3,500 offline batches on N′'s run's buffer, ~4 hours, give held-out
losses below Np181's on Np181's own data (policy 3.811 against 3.900, value 0.022 against
0.047), and **the reset network beats Np181 by +13.2 a pair (+11.4 to +15.0; 39 of 40
pairs won, none lost)** and hexparrot:3 by +28.7 (Np181 +17.1): against hexparrot its
dwarfs +11.7 better than Np181's on the same openings, its trolls equal. So the lineage
(runs K, M, N, all with the race) held back what its own data could teach: the network,
not the data, was the limit.
The control for whether knowing Thud is what stops a network fitting random targets (the
user's objection): the reset network's checkpoints after 250, 1,000 and 3,000 batches, and
the reset network itself — as good on the buffer as Np181 or better — fit them to
0.17-0.35 (policy), fresh 64 to 0.12: knowing the game does not stop it; Np181's rigidity
is its own, the race's or its far longer training's. **Run R** (the user, 2026-10-18): the
reset network continued in self-play, a plasticity test after each segment of 10 steps.

## Review log

**Review 1** (an adversarial reviewer agent, 2026-10-12, on draft 2; findings checked at
the source before being taken over):

- The value's "steady fall" was measured on different positions and through tanh: on fixed
  positions only run M's end falls (§1.4) — **taken over**, the diagnosis rewritten around
  the dwarfs' policy.
- The strongest evidence was missing: alpha-beta's dwarfs survive against our trolls
  (replayed: 20 of 20; 236 of 236 logged training games) — **taken over** (§1.2); D1
  revised; C7 added; C6 superseded.
- Run H against run I as a control for the onset — **taken over** (§1.1).
- B1 delayed by the buffer and diluted by the saturation; B1′ (Stockfish NNUE's λ) —
  **taken over**; B1 moved down.
- Group A overstated (off-policy data; policy improvement) — **taken over**, softened.
- Scale argument: the exploration term ~0.07, not ~1 — **taken over** (factor 3).
- Blunders 32% of losses; baits rare — **taken over**; draft 2's factor "baits" dropped.
- Measurement 3's design; measurements 5 and 7 at trainer settings; measurement 6's power; a value-only
  retraining measurement's missing control (dropped); A1's sign; C1's costs; a primary endpoint,
  a trolls' margin, replication, the evaluation budget; E1 and E2; Go-Exploit — **taken
  over**.
- Start test runs from L102 too — **not now** (M is the main line).

**Review 2** (the same reviewer, 2026-10-13, on draft 3; checked at the source):

- C7's mechanics: zero policy loss per row is right, but naive recording dilutes the policy
  gradient, displaces a quarter of each step's own data, changes the alpha-beta share and
  the statistics (`alpha_zero.cc`: the actor's and the learner's counting, checked) —
  **taken over**; C7's design and cost (1 day) corrected.
- The buffer already holds our trolls' positions from games where dwarfs survive (~26%) —
  **taken over**: §2 corrected; measurement 4 added; C7 made conditional on it.
- C7 reverses the user's rule of 2026-10-07 (`PLAN.md`, checked) — **taken over**: for the
  user to decide.
- The endpoint's power: the dwarfs against alpha-beta 2 at the floor (−31.2, checked); the
  trolls against hexparrot too imprecise for 3 points — **taken over**: alpha-beta 1 for
  the dwarfs, alpha-beta 3's dwarfs for the trolls, exact rules (the precisions are the
  reviewer's estimates, not rechecked).
- D1: the flattening is rather a concentration; the risk is lost hurl set-ups; the buffer
  delays it (0.30 → ~0.66); 0.7 unsupported; measurement 2 no bound — **taken over**: the weight
  from measurement 2, else 0.5.
- Measurement 3: the same opening in all cells, independent battles, dwarfs lost by ply 60, a
  fixed decision rule, C3 as a run of its own — **taken over**.
- Numbers and wording (H 41-67%, I 80-95%; 98-100%; 10.4%; 7 certain; the baits
  comparison; the dispersion 0.034; measurement 1's children's values; the plane ablation dropped
  as not diagnostic; E1's sampling ~3 times and no value overfitting; E2's two-hot; B1′'s
  sign; the hexparrot variant of C2 flagged; "rerun", no seed; one taper trigger) —
  **taken over**.

**The measurements** (2026-10-13; §4): measurement 1 and measurement 4 located the value's blindness at
dwarf-to-move positions and traced it to the buffer's data; measurement 2 measured D1's lever
(0.7 chosen by its rule); measurement 3 met its rule for C3, though outcomes stay saturated
without the dwarfs' noise too;
measurement 5 looked like support for C1 (review 3: its survivals are troll wipe-outs). The
recommendation changed accordingly: two nights (control and D1) instead of three, a third
test run chosen by D1's mechanism.

**Review 3** (the same reviewer, 2026-10-13, on draft 5 with the measurements' results; checked
at the source — the games replayed, H83 probed, the reviewer's trend script rerun):

- Measurement 2's survival is survival by stall, in matches without noise, and noise breaks
  stalls: D1's chain is untested where it must work — **taken over**: the stall analysis
  (§4), **measurement 2b before any night**, D1 with C3 if the dwarfs' noise blocks survival.
- Measurement 1's "the search prefers the blunder" conditions on the outcome (the search chose
  those moves; where it did not, no preference; "25 of 137" was 25 of 79) — **taken
  over**: the claim dropped, the child's-value result kept, measurement 4 carries the diagnosis.
- Self-play slides within every run (run M's blunders 8.6% → 23.2% by thirds), so the
  control slides — **taken over**: §1.3, §2 (with the reviewer's loop-gain inference,
  marked as such), both final networks compared with M102, an amplification condition in
  the taper.
- "The weight by results" could select 0.85-1.0, the imitation ruled out on 2026-10-07 —
  **taken over**: 0.7 unless the user reopens that decision (flagged for them).
- C3's low rank rested on the wrong contrast (both against neither); the right one
  (the dwarfs' noise alone) is worth +1.8 points; the honest reason — outcomes stay
  saturated — kept; C3 perhaps D1's precondition — **taken over**.
- §7: the trolls against alpha-beta 3's dwarfs too noisy (the reviewer's own round-2
  suggestion) → depth 2's dwarfs, 40 battles; the gates' rule simplified, standard errors
  clustered or bootstrapped rather than from M100 against M102, a troll-kills gate added
  (the short-hurl gate does not watch D1's risk), the false-alarm rate stated, the anchor
  named — **taken over**.
- Measurement 5: all nine survivals are battles with every troll killed — **taken over**; C1
  ranked after C7 and B1′.
- §1.2: the alpha-beta depth-1 row was M102's dwarfs against depth 1's trolls; depth 1's
  dwarfs lose every dwarf (−30.0) — **corrected**.
- Measurement 4: H83 as the clean control (probed: no gap); the gap not a probe artefact; the
  value's level also splits by side — **taken over**. Wording ("at or above its material"
  only for I83-K102; the placebo's range; "mostly a single dwarf"; the pairing; the
  trolls "unchanged within the noise"; the taper's noisy comparison) — **taken over**.

## References

- Silver et al., "A general reinforcement learning algorithm that masters chess, shogi,
  and Go through self-play", *Science* 2018 (Dirichlet α scaled to the legal moves).
- Schmitt et al., "Kickstarting deep reinforcement learning", 2018.
- Ng, Harada, Russell, "Policy invariance under reward transformations", ICML 1999.
- D. J. Wu, "Accelerating Self-Play Learning in Go", arXiv 1902.10565 (handicap games, komi
  randomisation, score utility); KataGo, `docs/KataGoMethods.md` (short-term value
  targets, since g170).
- "Lessons From AlphaZero (part 4): Improving the Training Target", Oracle developers, 2018.
- Stockfish, nnue-pytorch: `lambda` interpolating the evaluation and the game result.
- Trudeau, Bowling, "Targeted Search Control in AlphaZero for Effective Policy
  Improvement" (Go-Exploit), AAMAS 2023, arXiv 2302.12359.
- OpenAI, "OpenAI Five", 2018 (20% of games against past selves).
- Leela Chess Zero, project history (the moves-left head, v0.26).
- Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned
  model" (MuZero), *Nature* 2020 (categorical value support).
