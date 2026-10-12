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

"""hexparrot/thudgame's AI as a player for az_match (b=hexparrot:LOOKAHEAD).

az_match starts one of these per game thread and talks to it over stdin and stdout, one
line each way:

  seed N            -> ok          reseeds hexparrot's move-selection RNG (each battle)
  move SIDE ROWS    -> ROWS | none SIDE is dwarf or troll, ROWS the board as the 15 rows
                                   of one of our diagrams joined by '/' (thud.h's text
                                   format without its status line); the reply is the
                                   board after hexparrot's move, or none if its AI gives
                                   up (NoMoveException)

Positions, not moves, cross the boundary: the board is rebuilt from the diagram each turn
(perft_reference.board_from), and az_match finds the one legal action of ours that leads
to the board returned — so a move our rules do not allow stops the match instead of being
mapped to something else. hexparrot's AI reads nothing of the history but whose turn it
is, which its board takes from the parity of its ply list.

  python3 thud/experiments/hexparrot_player.py --lookahead 3

Setup as in perft_reference.py (hexparrot cloned outside this repo, commit 7b171108).
"""

import argparse
import importlib
import sys
import types

import perft_reference as ref


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--lookahead", type=int, default=3,
                      help="hexparrot AI lookahead (its GUI/console default is 3)")
  parser.add_argument("--hexparrot", default=ref.DEFAULT_HEXPARROT)
  args = parser.parse_args()
  ref.load_hexparrot(args.hexparrot)
  ai_engine = importlib.import_module("thud.ai_engine")
  no_move = importlib.import_module("thud.ply").NoMoveException

  for line in sys.stdin:
    words = line.split()
    if not words:
      continue
    if words[0] == "seed" and len(words) == 2:
      ai_engine.seed(int(words[1]))
      print("ok", flush=True)
    elif words[0] == "move" and len(words) == 3 and words[1] in ("dwarf", "troll"):
      side = words[1]
      board = ref.board_from("\n".join(words[2].split("/")))
      # turn_to_act() is the parity of the ply list. The stand-in for the dwarfs' last
      # move has the fields validate_move reads of a previous ply (a capture by a troll
      # matters only in Koom Valley Thud, not in classic).
      if side == "troll":
        board.ply_list = [types.SimpleNamespace(token="dwarf", captured=[])]
      assert board.turn_to_act() == side
      try:
        ply = ai_engine.AIEngine.calculate_best_move(board, side, args.lookahead)
      except no_move:
        print("none", flush=True)
        continue
      board.apply_ply(ply)
      print(ref.diagram_of(board).replace("\n", "/"), flush=True)
    else:
      sys.exit(f"hexparrot_player: cannot read {line!r}")


if __name__ == "__main__":
  main()
