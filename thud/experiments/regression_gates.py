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

"""The regression gates (thud/DESIGN_dwarf_signal.md, section 7; thud/PLAN.md, the GPU
readiness checklist): did a candidate network lose a skill against the anchor (M102)?
Each gate compares the two on the same openings or positions, paired, one-sided at 2
standard errors (the blunders at 4 percentage points):

  alpha-beta, dwarfs   our dwarfs' margin against alpha-beta depth 1     fails if lower by > 2 SE
  alpha-beta, trolls   our trolls' margin against depth 2's dwarfs       fails if lower by > 2 SE
  blunders             the share of our dwarf moves against hexparrot   fails if >= 4 points higher
                       that blunder a dwarf (az_blunder_check's rule: a
                       move capturing nothing, ending next to a troll,
                       whose dwarf the reply takes; after the 4 opening
                       plies), the SE clustered by battle (a bootstrap
                       over the pairs)
  troll kills          trolls our dwarfs kill a battle, against          fails if lower by > 2 SE
                       hexparrot and against alpha-beta depth 1
  short hurls          the long-capture probe's built hurls of 2-3       fails if lower by > 2 SE
                       squares: the capture the prior's top move
  trolls' captures     troll positions of self-play with a capture       fails if lower by > 2 SE
                       (az_capture_targets on the same logs): the
                       prior's top move a capture
  long hurls           built hurls of 4+ squares                         never fails: monitored

Inputs: az_match games files (games=PATH) of both networks against alpha-beta 1, alpha-beta
2 and hexparrot on the same openings (the same seed); az_long_capture_probe output holding
both networks (net = their names); and az_capture_targets output of each on the same logs
(the same positions in the same order, checked by their legal moves and captures). A gate whose files are not given is skipped,
and said so. A failed gate means "look into it" (a rerun with more battles first), not an
automatic verdict: seven measures at 2 standard errors trip ~15% of the time on an
unchanged network (section 7).

  python3 thud/experiments/regression_gates.py --candidate N121 --anchor M102 \\
    --games N121:ab1=FILE,ab2=FILE,hexparrot=FILE --games M102:ab1=FILE,ab2=FILE,hexparrot=FILE \\
    --probe FILE [--probe FILE ...] --captures N121=FILE --captures M102=FILE [--json]
"""

import argparse
import json
import math
import random
import re
import sys

OPENING = 4  # az_match's default opening plies, which az_blunder_check skips.
DESTINATION = re.compile(r"\(\d+,\d+\)-\((\d+),(\d+)\)")


def battles(path):
  return [json.loads(line) for line in open(path) if line.strip()]


def by_pair(path, player):
  """{pair: battle} for the battles in which our network played `player`."""
  return {b["pair"]: b for b in battles(path) if b["a_player"] == player}


def paired(a, b, f):
  """Mean and standard error of f(a) - f(b) over the pairs both have."""
  d = [f(a[p]) - f(b[p]) for p in a if p in b]
  n = len(d)
  m = sum(d) / n
  se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("inf")
  return m, se, n


def board_cells(state):
  return str(state).split("\n")[:15]


def next_to_troll(rows, r, c):
  for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
      if (dr or dc) and 0 <= r + dr < 15 and 0 <= c + dc < 15 and rows[r + dr][c + dc] == "T":
        return True
  return False


def blunders(battle, game):
  """(our dwarf moves, blunders) in one battle our network played as the dwarfs."""
  state = game.new_initial_state()
  moves = battle["actions"]
  ours = count = 0
  for i in range(len(moves) - 1):
    action = state.string_to_action(moves[i])
    if i >= OPENING and state.current_player() == 0:
      ours += 1
      if not moves[i].endswith("x"):
        r, c = map(int, DESTINATION.search(moves[i]).groups())
        after = state.child(action)
        if next_to_troll(board_cells(after), r, c):
          replied = after.child(after.string_to_action(moves[i + 1]))
          if board_cells(replied)[r][c] != "d":
            count += 1
    state.apply_action(action)
  return ours, count


def kills(battle):
  """Trolls our dwarfs killed: their captures (each a hurl taking one troll)."""
  state_player = 0  # The dwarfs move first; plies alternate.
  n = 0
  for i, move in enumerate(battle["actions"]):
    if i % 2 == state_player and i >= OPENING and move.endswith("x"):
      n += 1
  return n


def probe_cases(paths, net, keep):
  """{(position key): top} for the probe's per-case lines of `net` that `keep` accepts.

  The probe writes each network's cases in the same order, so the case's index among
  the lines of its (source, kind, k) is its key."""
  out, seen = {}, {}
  for path in paths:
    for line in open(path):
      d = json.loads(line)
      if d.get("summary") or d.get("net") != net or not keep(d):
        continue
      group = (d["source"], d["kind"], d["k"], d["direction"])
      i = seen.get(group, 0)
      seen[group] = i + 1
      out[group + (i,)] = 1.0 if d["rank"] == 1 else 0.0
  return out


def proportion_gate(name, cand, anchor, threshold_se=2.0, monitored=False):
  keys = [k for k in cand if k in anchor]
  if not keys:
    return {"gate": name, "skipped": "no positions of both networks"}
  d = [cand[k] - anchor[k] for k in keys]
  n = len(d)
  m = sum(d) / n
  se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("inf")
  return {"gate": name, "candidate": sum(cand[k] for k in keys) / n,
          "anchor": sum(anchor[k] for k in keys) / n, "difference": m, "se": se,
          "positions": n, "fails": (not monitored) and m < -threshold_se * se,
          "monitored": monitored}


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--candidate", required=True)
  parser.add_argument("--anchor", required=True)
  parser.add_argument("--games", action="append", default=[],
                      help="NAME:ab1=FILE,ab2=FILE,hexparrot=FILE (any of them)")
  parser.add_argument("--probe", action="append", default=[])
  parser.add_argument("--captures", action="append", default=[],
                      help="NAME=FILE: az_capture_targets output, one per network")
  parser.add_argument("--bootstrap", type=int, default=2000)
  parser.add_argument("--json", action="store_true", help="one JSON line per gate")
  args = parser.parse_args()

  files = {}
  for spec in args.games:
    name, rest = spec.split(":", 1)
    files[name] = dict(kv.split("=", 1) for kv in rest.split(","))
  cand, anch = files.get(args.candidate, {}), files.get(args.anchor, {})
  results = []

  def margin_gate(name, key, player):
    if key not in cand or key not in anch:
      return {"gate": name, "skipped": f"no {key} games of both networks"}
    m, se, n = paired(by_pair(cand[key], player), by_pair(anch[key], player),
                      lambda b: b["a_margin"])
    return {"gate": name, "difference": m, "se": se, "pairs": n, "fails": m < -2 * se}

  results.append(margin_gate("alpha-beta, dwarfs (depth 1)", "ab1", 0))
  results.append(margin_gate("alpha-beta, trolls (depth 2's dwarfs)", "ab2", 1))

  if "hexparrot" in cand and "hexparrot" in anch:
    import pyspiel  # Only the blunder gate needs the game.
    game = pyspiel.load_game("thud")
    a = {p: blunders(b, game) for p, b in by_pair(cand["hexparrot"], 0).items()}
    b = {p: blunders(x, game) for p, x in by_pair(anch["hexparrot"], 0).items()}
    pairs = [p for p in a if p in b]
    rate = lambda d, ps: sum(d[p][1] for p in ps) / sum(d[p][0] for p in ps)
    diff = rate(a, pairs) - rate(b, pairs)
    rng = random.Random(1)
    boots = []
    for _ in range(args.bootstrap):
      ps = [rng.choice(pairs) for _ in pairs]
      boots.append(rate(a, ps) - rate(b, ps))
    mb = sum(boots) / len(boots)
    se = math.sqrt(sum((x - mb) ** 2 for x in boots) / (len(boots) - 1))
    results.append({"gate": "blunders (against hexparrot)", "candidate": rate(a, pairs),
                    "anchor": rate(b, pairs), "difference": diff, "se": se,
                    "candidate_counts": [sum(a[p][1] for p in pairs), sum(a[p][0] for p in pairs)],
                    "anchor_counts": [sum(b[p][1] for p in pairs), sum(b[p][0] for p in pairs)],
                    "battles": len(pairs), "fails": diff >= 0.04})
  else:
    results.append({"gate": "blunders (against hexparrot)", "skipped": "no hexparrot games of both"})

  for key, opponent in (("hexparrot", "hexparrot"), ("ab1", "alpha-beta depth 1")):
    if key in cand and key in anch:
      m, se, n = paired(by_pair(cand[key], 0), by_pair(anch[key], 0), kills)
      results.append({"gate": f"troll kills (against {opponent})", "difference": m, "se": se,
                      "pairs": n, "fails": m < -2 * se})
    else:
      results.append({"gate": f"troll kills (against {opponent})", "skipped": f"no {key} games"})

  if args.probe:
    short = lambda d: d["source"] == "built" and d["kind"] == "hurl" and d["k"] in (2, 3)
    longer = lambda d: d["source"] == "built" and d["kind"] == "hurl" and d["k"] >= 4
    for name, keep, monitored in (("short hurls (2-3 squares)", short, False),
                                  ("long hurls (4+ squares)", longer, True)):
      results.append(proportion_gate(name, probe_cases(args.probe, args.candidate, keep),
                                     probe_cases(args.probe, args.anchor, keep),
                                     monitored=monitored))
  else:
    for name in ("short hurls (2-3 squares)", "long hurls (4+ squares)"):
      results.append({"gate": name, "skipped": "no probe file"})

  captures = dict(spec.split("=", 1) for spec in args.captures)
  gate = "trolls' captures (the prior's top move a capture)"
  if args.candidate in captures and args.anchor in captures:
    lines = {n: [json.loads(l) for l in open(captures[n]) if l.strip() and "summary" not in l]
             for n in (args.candidate, args.anchor)}
    a, b = lines[args.candidate], lines[args.anchor]
    if len(a) != len(b) or any((x["legal"], x["captures"]) != (y["legal"], y["captures"])
                               for x, y in zip(a, b)):
      results.append({"gate": gate, "skipped": "the two files are not the same positions"})
    else:
      results.append(proportion_gate(
          gate, {i: float(x["prior_top_captures"]) for i, x in enumerate(a)},
          {i: float(y["prior_top_captures"]) for i, y in enumerate(b)}))
  else:
    results.append({"gate": gate, "skipped": "no az_capture_targets files of both"})

  failed = [r["gate"] for r in results if r.get("fails")]
  if args.json:
    for r in results:
      print(json.dumps(r))
    print(json.dumps({"summary": True, "candidate": args.candidate, "anchor": args.anchor,
                      "failed": failed}))
  else:
    print(f"Regression gates: {args.candidate} against the anchor {args.anchor}")
    for r in results:
      if "skipped" in r:
        print(f"  SKIP  {r['gate']}: {r['skipped']}")
        continue
      verdict = "MONITOR" if r.get("monitored") else ("FAIL" if r["fails"] else "pass")
      extra = ""
      if "candidate" in r:
        extra = f" ({r['candidate']:.3f} against {r['anchor']:.3f})"
      print(f"  {verdict:7} {r['gate']}: {r['difference']:+.3f} ± {r['se']:.3f}{extra}")
    print("  " + ("failed: " + ", ".join(failed) + " — look into it (a rerun with more battles "
                  "first)" if failed else "all gates pass"))
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
