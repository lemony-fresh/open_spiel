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

"""Where the time goes in OpenSpiel's Python AlphaZero search, on Thud.

Times, for resnets of three sizes: OpenSpiel's Python MCTS with PUCT (as its AlphaZero
uses it) with a do-nothing evaluator, which isolates the search; the same search with
AlphaZero's network evaluator; and the network alone, evaluated three ways: as upstream's
Model.inference does it (one position at a time, never compiled: its @jax.jit is commented
out, model_linen.py:451), compiled once and one position at a time, and compiled in batches
of 64 (which may use several cores).

  python3 az_speed.py      # about a minute

Needs pyspiel and OpenSpiel's JAX dependencies.
"""

import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyspiel

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model_linen

SIZES = ((32, 2), (64, 4), (128, 6))  # resnet (width, depth)


class UniformEvaluator(mcts.Evaluator):
  """Costs nothing: value 0 and a uniform prior. Isolates the search itself."""

  def evaluate(self, state):
    return np.zeros(2)

  def prior(self, state):
    legal = state.legal_actions()
    return [(a, 1 / len(legal)) for a in legal]


def turn_120(game):
  rng, state = random.Random(3), game.new_initial_state()
  for _ in range(120):
    state.apply_action(rng.choice(state.legal_actions()))
  return state


def sims_per_second(game, state, evaluator, sims):
  bot = mcts.MCTSBot(game, 2.0, sims, evaluator, solve=False,
                     child_selection_fn=mcts.SearchNode.puct_value,
                     random_state=np.random.RandomState(0))
  start = time.time()
  bot.mcts_search(state)
  return sims / (time.time() - start)


def seconds_per_call(fn, repeats):
  jax.block_until_ready(fn())  # Compiles, if it compiles at all.
  start = time.time()
  for _ in range(repeats):
    jax.block_until_ready(fn())
  return (time.time() - start) / repeats


def main():
  game = pyspiel.load_game("thud")
  positions = [game.new_initial_state(), turn_120(game)]
  for state in positions:
    print(f"position with {len(state.legal_actions())} dwarf moves: search alone "
          f"{sims_per_second(game, state, UniformEvaluator(), 400):,.0f} sims/s")
  opening = positions[0]
  obs = np.asarray(opening.observation_tensor(), np.float32)
  mask = np.asarray(opening.legal_actions_mask())
  for width, depth in SIZES:
    model = model_linen.Model.build_model(
        "resnet", game.observation_tensor_shape(), game.num_distinct_actions(), width,
        depth, weight_decay=1e-4, learning_rate=1e-3, path="/tmp")
    evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    print(f"resnet {width} x {depth} ({model.num_trainable_variables:,} variables):")
    for state in positions:
      print(f"  search with the network, {len(state.legal_actions())} moves: "
            f"{sims_per_second(game, state, evaluator, 100):.1f} sims/s")

    params, stats = model._state.params, model._state.batch_stats  # pylint: disable=protected-access
    apply_fn = model._state.apply_fn  # pylint: disable=protected-access

    def forward(p, s, o):
      return apply_fn({"params": p, "batch_stats": s}, o, training=False, mutable=False)

    single = jax.jit(forward)
    batched = jax.jit(jax.vmap(forward, in_axes=(None, None, 0), axis_name="batch"))
    one, many = jnp.asarray(obs), jnp.asarray(np.tile(obs, (64, 1)))
    as_upstream = seconds_per_call(lambda: model.inference(obs, mask), 20)
    compiled = seconds_per_call(lambda: single(params, stats, one), 200)
    in_batches = seconds_per_call(lambda: batched(params, stats, many), 20) / 64
    print(f"  network per position: as upstream {as_upstream * 1e3:.2f} ms, compiled once "
          f"{compiled * 1e3:.3f} ms, compiled in batches of 64 {in_batches * 1e3:.3f} ms")


if __name__ == "__main__":
  main()
