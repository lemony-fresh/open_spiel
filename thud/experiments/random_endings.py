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

"""How games of uniformly random moves end, played with our own Thud implementation.

Phase 5 (thud/PLAN.md): checks the proxy behind the end-of-battle limits (THUD_RULES.md
section 6). Session 2 chose 200/800 from 1,000 random games played with hexparrot's engine
(thud/PROGRESS.md, session 2: median 334 turns, mean 343, max 794; 1.5% of games cut short
by 200/800). This plays the same kind of games through pyspiel with our implementation and
reports how they end: a rout (the side to move has no pieces), stuck (pieces, but no legal
move), or one of the two limits; their lengths; the longest stretches without a capture;
and the results.

  python3 random_endings.py --games 1000      # about a minute on one process

Seeds are fixed (0..N-1). Needs pyspiel (the build's python/ directory on PYTHONPATH, as
CLAUDE.md sets up); speed does not matter here, so any build type will do.
"""

import argparse
import collections
import random
import re
import statistics as st

import pyspiel


def status(state):
  """The side to move, turns, turns without a capture, dwarfs and trolls of a state."""
  text = str(state)
  fields = dict(re.findall(r"(\w+)=(\w+)", text.split("\n")[-1]))
  board = "".join(text.split("\n")[:15])
  return (fields["to_move"], int(fields["turns"]), int(fields["turns_without_capture"]),
          board.count("d"), board.count("T"))


def play(game, seed):
  """One random game: how it ended, its length, longest no-capture stretch, margin."""
  rng, state = random.Random(seed), game.new_initial_state()
  longest = 0
  while not state.is_terminal():
    state.apply_action(rng.choice(state.legal_actions()))
    longest = max(longest, status(state)[2])
  to_move, turns, without_capture, dwarfs, trolls = status(state)
  params = game.get_parameters()
  if turns >= params["max_turns"]:
    ending = "turn limit"
  elif without_capture >= params["max_turns_without_capture"]:
    ending = "no-capture limit"
  elif (dwarfs if to_move == "dwarfs" else trolls) == 0:
    ending = "rout"
  else:
    ending = "stuck"
  return ending, turns, longest, dwarfs - 4 * trolls, state.returns()


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--games", type=int, default=1000)
  parser.add_argument("--game", default="thud")
  args = parser.parse_args()
  game = pyspiel.load_game(args.game)
  games = [play(game, seed) for seed in range(args.games)]

  n = len(games)
  endings = collections.Counter(g[0] for g in games)
  lengths = [g[1] for g in games]
  longest = [g[2] for g in games]
  margins = [g[3] for g in games]
  print(f"{n} random games of {args.game}")
  print("endings: " + ", ".join(f"{k} {v} ({v / n:.1%})"
                                for k, v in endings.most_common()))
  print(f"length: median {st.median(lengths):.0f}, mean {st.mean(lengths):.0f}, "
        f"max {max(lengths)}")
  print(f"longest stretch without a capture: median {st.median(longest):.0f}, "
        f"max {max(longest)}")
  print(f"dwarf margin (dwarfs - 4 x trolls): mean {st.mean(margins):.1f}; "
        f"dwarfs win {sum(m > 0 for m in margins) / n:.1%}, "
        f"trolls win {sum(m < 0 for m in margins) / n:.1%}, "
        f"draws {sum(m == 0 for m in margins) / n:.1%}")
  bad = [g for g in games if abs(g[4][0] - g[3] / 32) > 1e-9]
  print(f"returns equal margin / 32 in every game: {not bad}")


if __name__ == "__main__":
  main()
