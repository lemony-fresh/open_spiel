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

"""Tests thud/experiments/regression_gates.py on inputs made here (no run folder): battles
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

  5. The capacity tests' rows (2026-10-18), monitored: from an az_dormant_probe file and
     an az_plasticity file made here, the dead channels split by trunk and heads and the
     dormant ones at the largest τ, and the random targets' fit, each beside the anchor's
     and a fresh network's, as written; the control: a candidate with every channel dead
     and no fit at all is reported, never failed (exit 0); without the files both rows
     are skipped.
  6. The trunk's gradients (2026-10-19), monitored: from az_restructure reports made here,
     the weighted ratio and the cosine of the report before the first batch (not a later
     one), the candidate's and the anchor's; the control: opposed gradients (cosine -1)
     are reported, never failed; without a report the row is skipped.
  7. Late captures (2026-10-19), monitored: late_captures counts what an independent
     count by hand counts (our network's captures after more than 2 quiet turns, in the
     random battles); the control: the same battles cut to their first 30 plies give
     fewer, reported, never failed; without hexparrot games the row is skipped.

(The blunder count also equals az_blunder_probe's on real games — M102, N121, C121 and
N2121 against hexparrot, checked 2026-10-15 — which needs run folders, so not here.)

  python3 thud/experiments/regression_gates_test.py
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
  tmp = tempfile.mkdtemp(prefix="regression_gates_test.")
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
  compared = [r for r in results.values() if "skipped" not in r and "text" not in r]
  late = results["late captures (after 100+ quiet turns, against hexparrot, a battle)"]
  check(status == 0 and len(compared) == 8 and all(not r["fails"] and r["difference"] == 0
                                                     for r in compared)
        and late["candidate"] == late["anchor"],
        f"1. against itself: all {len(compared)} measures pass, every difference 0 (the "
        f"late captures equal)")

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
  # 5. The capacity tests' rows.
  layers = [("input", 64), ("torso_1_a", 64), ("torso_1_b", 64), ("value_linear", 64)]
  def dormant_lines(net, dead):
    return [{"net": net, "layer": l, "channels": c, "dead": dead.get(l, 0),
             "dormant": {"0.025": dead.get(l, 0), "0.1": dead.get(l, 0) + 1},
             "median_active_share": 0.5} for l, c in layers]
  write(f"{tmp}/dormant.jsonl", dormant_lines("C", {"torso_1_a": 2, "value_linear": 3}) +
        dormant_lines("A", {"value_linear": 1}) + dormant_lines("fresh", {}) +
        dormant_lines("dead", {l: c for l, c in layers}) +
        [{"summary": True, "net": "C", "channels": 256, "dead": 5}])
  def plastic(net, fresh, policy, value):
    return [{"net": net, "batches": 100, "policy": policy + 1, "value": value + 1},
            {"net": net, "batches": 200, "policy": policy, "value": value},
            {"summary": True, "net": net, "fresh": fresh, "width": 64, "last_policy": policy,
             "last_value": value, "mean_policy": policy, "mean_value": value,
             "weight_norm": 40.0}]
  write(f"{tmp}/plastic.jsonl", plastic("C", False, 0.3, 0.05) + plastic("A", False, 4.0, 1.0)
        + plastic("f64", True, 0.1, 0.02) + plastic("dead", False, 4.3, 1.0))
  results, status = gates(tmp, "--candidate", "C", "--anchor", "A", "--dormant",
                          f"{tmp}/dormant.jsonl", "--plasticity", f"{tmp}/plastic.jsonl")
  dead = results["dead channels (trunk, heads)"]
  fit = results["plasticity (random targets: policy + value loss at the end)"]
  check(status == 0 and dead["monitored"] and not dead["fails"] and
        (dead["candidate"]["trunk_dead"], dead["candidate"]["head_dead"],
         dead["candidate"]["dormant"], dead["anchor"]["head_dead"], dead["fresh"]["dead"]) ==
        (2, 3, 9, 1, 0) and abs(fit["candidate"] - 0.35) < 1e-9 and
        abs(fit["anchor"] - 5.0) < 1e-9 and abs(fit["fresh"] - 0.12) < 1e-9 and
        fit["batches"] == 200 and not fit["fails"],
        f"5. the capacity rows, monitored: {dead['text']} | {fit['text']}")
  results, status = gates(tmp, "--candidate", "dead", "--anchor", "A", "--dormant",
                          f"{tmp}/dormant.jsonl", "--plasticity", f"{tmp}/plastic.jsonl")
  check(status == 0 and results["dead channels (trunk, heads)"]["candidate"]["dead"] == 256 and
        not any(r.get("fails") for r in results.values()),
        "5. control: every channel dead and no fit — reported, never failed (exit 0)")
  results, status = gates(tmp, "--candidate", "C", "--anchor", "A")
  check(status == 0 and "skipped" in results["dead channels (trunk, heads)"] and
        "skipped" in results["plasticity (random targets: policy + value loss at the end)"],
        "5. without the files both rows are skipped")
  # 6. The trunk's gradients.
  def report(path, ratio, cosine):
    write(path, [{"source": True, "policy": 3.7},
                 {"batches": 0, "gradients": {"policy": 0.5, "value": 1.0, "l2": 0.01,
                                              "cosine": cosine, "weighted_value_to_policy": ratio}},
                 {"batches": 1, "gradients": {"policy": 0.4, "value": 0.9, "l2": 0.01,
                                              "cosine": 0.9, "weighted_value_to_policy": 9.0}}])
  report(f"{tmp}/grad_C.jsonl", 2.5, 0.2)
  report(f"{tmp}/grad_A.jsonl", 7.8, -1.0)
  results, status = gates(tmp, "--candidate", "C", "--anchor", "A", "--gradients",
                          f"C={tmp}/grad_C.jsonl", "--gradients", f"A={tmp}/grad_A.jsonl")
  row = results["trunk gradients (the value's pull against the policy's, their cosine)"]
  check(status == 0 and row["monitored"] and not row["fails"] and
        row["candidate"]["weighted_value_to_policy"] == 2.5 and row["candidate"]["cosine"] == 0.2
        and row["anchor"]["cosine"] == -1.0, f"6. the trunk's gradients, monitored: {row['text']}")
  results, status = gates(tmp, "--candidate", "A", "--anchor", "C", "--gradients",
                          f"A={tmp}/grad_A.jsonl")
  row = results["trunk gradients (the value's pull against the policy's, their cosine)"]
  check(status == 0 and not row["fails"] and row["candidate"]["cosine"] == -1.0,
        "6. control: opposed gradients reported, never failed (exit 0)")
  results, status = gates(tmp, "--candidate", "C", "--anchor", "A")
  check("skipped" in results["trunk gradients (the value's pull against the policy's, their cosine)"],
        "6. without a report the row is skipped")
  # 7. Late captures.
  spec7 = importlib.util.spec_from_file_location("gates7", GATES)
  g7 = importlib.util.module_from_spec(spec7)
  spec7.loader.exec_module(g7)
  def by_hand(battle, after):
    state = game.new_initial_state()
    quiet = count = 0
    for action in battle["actions"]:
      if state.is_terminal():
        break
      ours = state.current_player() == battle["a_player"]
      before = sum(str(state).split("to_move")[0].count(p) for p in "dT")
      state.apply_action(state.string_to_action(action))
      captured = sum(str(state).split("to_move")[0].count(p) for p in "dT") < before
      if captured and quiet > after and ours:
        count += 1
      quiet = 0 if captured else quiet + 1
    return count
  hex_battles = base["hexparrot"]
  ours7 = sum(g7.late_captures(b, game, 2) for b in hex_battles)
  hand7 = sum(by_hand(b, 2) for b in hex_battles)
  cut = [dict(b, actions=b["actions"][:30]) for b in hex_battles]
  write(f"{tmp}/hex_cut.jsonl", cut)
  cut7 = sum(g7.late_captures(b, game, 2) for b in cut)
  results, status = gates(tmp, "--candidate", "C", "--anchor", "A", "--late_after", "2",
                          "--games", f"C:hexparrot={tmp}/hex_cut.jsonl",
                          "--games", f"A:hexparrot={tmp}/hexparrot.jsonl")
  row = results["late captures (after 2+ quiet turns, against hexparrot, a battle)"]
  # (The cut battles fail other gates, the blunders' and the kills': the exit status says
  # nothing about this row.)
  check(ours7 == hand7 and ours7 > 0 and cut7 < ours7 and row["monitored"]
        and not row["fails"] and
        row["candidate"]["trolls"] + row["candidate"]["dwarfs"] <
        row["anchor"]["trolls"] + row["anchor"]["dwarfs"],
        f"7. late captures: {ours7} counted, {hand7} by hand; the control cut to 30 plies "
        f"{cut7}, reported, never failed: {row['text']}")
  results, status = gates(tmp, "--candidate", "C", "--anchor", "A")
  check("skipped" in results["late captures (after 100+ quiet turns, against hexparrot, a battle)"],
        "7. without hexparrot games the row is skipped")
  print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
