#!/usr/bin/env python3
# Copyright 2026 The Thud-on-OpenSpiel authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checks thud/experiments/regression_gates.py on inputs made here (no run folder): battles
of random play in az_match's games format (40 pairs, 4 opening plies), a probe file and
az_capture_targets files.

  1. A network against itself: every gate passes, every difference exactly 0.
  2. Controls, each must fail its gate and only its gate: the dwarfs' margins against
     alpha-beta 1 lowered by 5 points; the anchor's dwarfs given three more captures a
     battle against alpha-beta 1 (random play hardly captures: the candidate kills fewer
     trolls; that gate reads the moves' notation alone); the probe's short hurls never the top move; the trolls' captures never
     the prior's top move. And the long hurls never the top move only monitored, never a
     failure.
  3. Positions that do not match (az_capture_targets files of other positions): the gate
     skipped, not compared.
  4. The blunder rule by hand: a battle whose one dwarf move ends next to a troll that
     takes it counts 1 blunder of the dwarf moves after the opening; the same battle with
     that capture replaced by another move counts 0.

(The blunder count also equals az_blunder_check's on real games — M102, N121, C121 and
N2121 against hexparrot, checked 2026-10-15 — which needs run folders, so not here.)

  python3 thud/experiments/regression_gates_check.py
"""

import json
import os
import random
import subprocess
import sys
import tempfile

import pyspiel

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = os.path.join(HERE, "regression_gates.py")
failures = 0


def check(ok, what):
  global failures
  print(("PASS " if ok else "FAIL ") + what)
  failures += not ok


def random_battles(game, seed, pairs=40):
  rng = random.Random(seed)
  out = []
  for pair in range(pairs):
    opening_rng = random.Random(1000 + pair)
    for a_player in (0, 1):
      state = game.new_initial_state()
      actions = []
      for ply in range(rng.randrange(60, 160)):
        if state.is_terminal():
          break
        legal = state.legal_actions()
        r = opening_rng if ply < 4 else rng
        a = legal[r.randrange(len(legal))]
        actions.append(state.action_to_string(a))
        state.apply_action(a)
      out.append({"pair": pair, "a_player": a_player, "a_margin": rng.randrange(-30, 31),
                  "actions": actions})
  return out


def write(path, battles):
  with open(path, "w") as f:
    for b in battles:
      f.write(json.dumps(b) + "\n")


def gates(tmp, *args):
  r = subprocess.run([sys.executable, GATES, "--json", *args], capture_output=True, text=True)
  lines = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
  return {l["gate"]: l for l in lines if "gate" in l}, r.returncode


def main():
  game = pyspiel.load_game("thud")
  tmp = tempfile.mkdtemp(prefix="regression_gates_check.")
  base = {k: random_battles(game, s) for k, s in (("ab1", 1), ("ab2", 2), ("hexparrot", 3))}
  for k, battles in base.items():
    write(f"{tmp}/{k}.jsonl", battles)
  probe = []
  for net in ("A", "B"):
    for k in range(1, 6):
      for i in range(30):
        probe.append({"net": net, "source": "built", "kind": "hurl", "k": k, "direction": "N",
                      "move": "x", "p": 0.1, "rank": 1 if i % 3 else 2, "legal": 100})
  write(f"{tmp}/probe.jsonl", probe)
  caps = [{"captures": 2, "legal": 30 + i % 7, "prior_top_captures": i % 2} for i in range(60)]
  write(f"{tmp}/caps.jsonl", caps)
  files = ",".join(f"{k}={tmp}/{k}.jsonl" for k in base)

  # 1. A network against itself.
  results, status = gates(tmp, "--candidate", "A", "--anchor", "B", "--games", f"A:{files}",
                          "--games", f"B:{files}", "--probe", f"{tmp}/probe.jsonl",
                          "--captures", f"A={tmp}/caps.jsonl", "--captures", f"B={tmp}/caps.jsonl")
  compared = [r for r in results.values() if "skipped" not in r]
  check(status == 0 and len(compared) == 8 and all(not r["fails"] and r["difference"] == 0
                                                     for r in compared),
        f"1. against itself: all {len(compared)} measures pass, every difference 0")

  # 2. Controls.
  def variant(key, change):
    battles = [dict(b) for b in base[key]]
    for b in battles:
      change(b)
    path = f"{tmp}/{key}_{change.__name__}.jsonl"
    write(path, battles)
    return path

  def lower(b):
    b["a_margin"] -= 5

  def more_captures(b):  # Three dwarf moves after the opening written as captures.
    b["actions"] = [m + "x" if i in (4, 6, 8) and not m.endswith("x") else m
                    for i, m in enumerate(b["actions"])]

  ab1_lower = variant("ab1", lower)
  ab1_more = variant("ab1", more_captures)
  probe_short = [dict(d, rank=2) if d["net"] == "A" and d["k"] in (2, 3) else d for d in probe]
  write(f"{tmp}/probe_short.jsonl", probe_short)
  probe_long = [dict(d, rank=2) if d["net"] == "A" and d["k"] >= 4 else d for d in probe]
  write(f"{tmp}/probe_long.jsonl", probe_long)
  write(f"{tmp}/caps_none.jsonl", [dict(c, prior_top_captures=0) for c in caps])
  cases = [
      ("the dwarfs' margins against alpha-beta 1 lowered by 5", "alpha-beta, dwarfs (depth 1)",
       ["--games", f"A:ab1={ab1_lower}", "--games", f"B:ab1={tmp}/ab1.jsonl"]),
      ("the anchor's dwarfs capturing more against alpha-beta 1",
       "troll kills (against alpha-beta depth 1)",
       ["--games", f"A:ab1={tmp}/ab1.jsonl", "--games", f"B:ab1={ab1_more}"]),
      ("the short hurls never the top move", "short hurls (2-3 squares)",
       ["--probe", f"{tmp}/probe_short.jsonl"]),
      ("the trolls' captures never the prior's top move",
       "trolls' captures (the prior's top move a capture)",
       ["--captures", f"A={tmp}/caps_none.jsonl", "--captures", f"B={tmp}/caps.jsonl"]),
  ]
  for what, gate, args in cases:
    results, status = gates(tmp, "--candidate", "A", "--anchor", "B", *args)
    failed = [g for g, r in results.items() if r.get("fails")]
    check(status == 1 and failed == [gate], f"2. control, {what}: fails {failed}")
  results, status = gates(tmp, "--candidate", "A", "--anchor", "B", "--probe",
                          f"{tmp}/probe_long.jsonl")
  long = results["long hurls (4+ squares)"]
  check(status == 0 and long["monitored"] and long["difference"] < 0 and not long["fails"],
        f"2. the long hurls lower ({long['difference']:+.2f}): monitored, not a failure")

  # 3. Positions that do not match.
  write(f"{tmp}/caps_other.jsonl", [dict(c, legal=c["legal"] + 1) for c in caps])
  results, _ = gates(tmp, "--candidate", "A", "--anchor", "B", "--captures",
                     f"A={tmp}/caps_other.jsonl", "--captures", f"B={tmp}/caps.jsonl")
  check("skipped" in results["trolls' captures (the prior's top move a capture)"],
        "3. az_capture_targets files of other positions: the gate skipped")

  # 4. The blunder rule by hand: find a dwarf move that steps next to a troll which can
  # take it at once, in a random game after the opening.
  import importlib.util
  spec = importlib.util.spec_from_file_location("gates", GATES)
  g = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(g)
  found = None
  rng = random.Random(7)
  for attempt in range(200):
    state = game.new_initial_state()
    moves = []
    for ply in range(120):
      if state.is_terminal():
        break
      if ply >= 4 and state.current_player() == 0:
        for a in state.legal_actions():
          m = state.action_to_string(a)
          if m.endswith("x"):
            continue
          r, c = map(int, g.DESTINATION.search(m).groups())
          after = state.child(a)
          if not g.next_to_troll(g.board_cells(after), r, c):
            continue
          for reply in after.legal_actions():
            if g.board_cells(after.child(reply))[r][c] != "d":
              other = next((x for x in after.legal_actions()
                            if g.board_cells(after.child(x))[r][c] == "d"), None)
              if other is not None:
                found = (moves + [m, after.action_to_string(reply)],
                         moves + [m, after.action_to_string(other)])
              break
          if found:
            break
        if found:
          break
      legal = state.legal_actions()
      # Random play, avoiding captures of dwarfs so that earlier moves are not blunders.
      a = legal[rng.randrange(len(legal))]
      moves.append(state.action_to_string(a))
      state.apply_action(a)
    if found:
      break
  if found is None:
    check(False, "4. a blunder position found in random games")
  else:
    taken, spared = found
    # The rule looks at a move only if a reply follows it: `taken` ends with the reply.
    ours_t, n_t = g.blunders({"actions": taken}, game)
    ours_s, n_s = g.blunders({"actions": spared}, game)
    before_t, before_n = g.blunders({"actions": taken[:-1]}, game)
    check(n_t - before_n == 1 and n_s - before_n == 0 and ours_t == ours_s,
          f"4. the blunder rule by hand: the dwarf taken counts 1 more blunder ({n_t} against "
          f"{before_n} before the move), the dwarf spared none ({n_s}); dwarf moves {ours_t}")
  print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
