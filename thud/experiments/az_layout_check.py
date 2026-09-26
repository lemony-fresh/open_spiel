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

"""Does OpenSpiel's Python AlphaZero model learn Thud better with the planes as channels?

OpenSpiel's flax model reshapes each observation to game.observation_tensor_shape(), planes
first, and hands it to flax's nn.Conv, which takes the LAST axis as channels
(model_linen.py:209/217). For Thud's (6, 15, 15) the convolution then slides over (plane,
row), with the 15 board columns as channels. This trains OpenSpiel's own model and update
step, unmodified, in two variants on the same data:

  as_is  the observation as AlphaZero feeds it today, input shape (6, 15, 15);
  fixed  the same observation transposed to (15, 15, 6), input shape (15, 15, 6), so the
         unchanged reshape yields planes-last and the convolution runs over the board.

The task is purely about the board, and one AlphaZero needs: the value head learns whether
the side to move has a capture (+1) or not (-1); the policy target is uniform over the
capturing moves if there are any, else over all legal moves. Positions come from random
games, train and test from different games.

  python3 az_layout_check.py --variant as_is --seed 1
  python3 az_layout_check.py --variant fixed --seed 1

Prints one JSON line per evaluation. Needs pyspiel and OpenSpiel's JAX dependencies.
"""

import argparse
import json
import os
import pickle
import random
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyspiel

from open_spiel.python.algorithms.alpha_zero import model_linen
from open_spiel.python.algorithms.alpha_zero import utils

SHAPE = (6, 15, 15)  # ThudGame::ObservationTensorShape()
TROLLS_TO_MOVE_PLANE = 3


def positions(game, seeds, every):
  """Every `every`-th position of random games: (observation, legal, capturing)."""
  data = []
  for seed in seeds:
    rng, state, turn = random.Random(seed), game.new_initial_state(), 0
    while not state.is_terminal():
      legal = state.legal_actions()
      if turn % every == 0:
        captures = [a for a in legal if state.action_to_string(a).endswith("x")]
        data.append((np.asarray(state.observation_tensor(), np.float32),
                     np.asarray(legal), np.asarray(captures, dtype=np.int64)))
      state.apply_action(rng.choice(legal))
      turn += 1
  return data


def load_data(game, cache):
  if os.path.exists(cache):
    with open(cache, "rb") as f:
      return pickle.load(f)
  # Every 7th turn: an odd step, so both sides to move are sampled.
  data = positions(game, range(400), 7), positions(game, range(10_000, 10_100), 7)
  with open(cache, "wb") as f:
    pickle.dump(data, f)
  return data


def model_input(observation, variant):
  if variant == "fixed":
    return observation.reshape(SHAPE).transpose(1, 2, 0).ravel()
  return observation


def batch(data, indices, variant, num_actions):
  n = len(indices)
  mask = np.zeros((n, num_actions), bool)
  policy = np.zeros((n, num_actions), np.float32)
  value = np.empty(n, np.float32)
  for row, i in enumerate(indices):
    _, legal, captures = data[i]
    target = captures if len(captures) else legal
    mask[row, legal] = True
    policy[row, target] = 1 / len(target)
    value[row] = 1.0 if len(captures) else -1.0
  obs = np.stack([model_input(data[i][0], variant) for i in indices])
  return utils.TrainInput(observation=jnp.asarray(obs), legals_mask=jnp.asarray(mask),
                          policy=jnp.asarray(policy), value=jnp.asarray(value))


def evaluate(model, predict, test, variant, num_actions):
  """Value sign accuracy, and policy mass on capturing moves, per side to move."""
  stats = {}
  for start in range(0, len(test), 256):
    idx = list(range(start, min(start + 256, len(test))))
    b = batch(test, idx, variant, num_actions)
    logits, value = predict(model._state.params, model._state.batch_stats,  # pylint: disable=protected-access
                            b.observation)
    logits = jnp.where(b.legals_mask, logits, jnp.finfo(jnp.float32).min)
    probs = np.asarray(jax.nn.softmax(logits, axis=-1))
    for row, i in enumerate(idx):
      obs, _, captures = test[i]
      side = "trolls" if obs.reshape(SHAPE)[TROLLS_TO_MOVE_PLANE].any() else "dwarfs"
      s = stats.setdefault(side, {"n": 0, "right": 0, "with_capture": 0, "mass": 0.0})
      s["n"] += 1
      s["right"] += (float(value[row]) > 0) == bool(len(captures))
      if len(captures):
        s["with_capture"] += 1
        s["mass"] += float(probs[row, captures].sum())
  return {side: {"value_accuracy": round(s["right"] / s["n"], 4),
                 "capture_mass": round(s["mass"] / max(1, s["with_capture"]), 4)}
          for side, s in sorted(stats.items())}


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--variant", choices=["as_is", "fixed"], required=True)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--steps", type=int, default=1500)
  parser.add_argument("--eval_every", type=int, default=250)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--nn_width", type=int, default=32)
  parser.add_argument("--nn_depth", type=int, default=2)
  parser.add_argument("--cache", default="/tmp/az_layout_check_positions.pkl")
  args = parser.parse_args()

  game = pyspiel.load_game("thud")
  train, test = load_data(game, args.cache)
  num_actions = game.num_distinct_actions()
  shape = SHAPE if args.variant == "as_is" else (15, 15, 6)
  model = model_linen.Model.build_model(
      "resnet", shape, num_actions, args.nn_width, args.nn_depth, weight_decay=1e-4,
      learning_rate=1e-3, path="/tmp", seed=args.seed)
  kernel = model._state.params["ConvBlock_0"]["Conv_0"]["kernel"].shape  # pylint: disable=protected-access
  apply_fn = model._state.apply_fn  # pylint: disable=protected-access

  @jax.jit
  def predict(params, batch_stats, observations):
    one = lambda o: apply_fn({"params": params, "batch_stats": batch_stats}, o,
                             training=False, mutable=False)
    return jax.vmap(one, axis_name="batch")(observations)

  baselines = {}  # A constant value guess, and a uniform policy, per side to move.
  for side, trolls in (("dwarfs", False), ("trolls", True)):
    mine = [(l, c) for o, l, c in test
            if o.reshape(SHAPE)[TROLLS_TO_MOVE_PLANE].any() == trolls]
    open_ = [len(c) / len(l) for l, c in mine if len(c)]
    share = len(open_) / len(mine)
    baselines[side] = {"positions": len(mine), "value_accuracy": round(max(share, 1 - share), 4),
                       "capture_mass": round(float(np.mean(open_)), 4)}
  head = {"variant": args.variant, "seed": args.seed, "first_kernel": kernel,
          "train": len(train), "test": len(test), "baselines": baselines}
  print(json.dumps(head), flush=True)
  rng, start = np.random.default_rng(args.seed), time.time()
  for step in range(1, args.steps + 1):
    losses = model.update(batch(train, rng.integers(len(train), size=args.batch_size),
                                args.variant, num_actions))
    if step % args.eval_every == 0:
      print(json.dumps({"variant": args.variant, "seed": args.seed, "step": step,
                        "seconds": round(time.time() - start),
                        "loss_policy": round(float(losses.policy), 4),
                        "loss_value": round(float(losses.value), 4),
                        **evaluate(model, predict, test, args.variant, num_actions)}),
            flush=True)


if __name__ == "__main__":
  main()
