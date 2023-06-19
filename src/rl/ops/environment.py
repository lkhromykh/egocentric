from typing import Callable
import collections

import dm_env.specs
import numpy as np
from jax.tree_util import tree_map
from rltools.dmc_wrappers.base import Wrapper

from src.rl import types_ as types

ActionLogProbFn = Callable[
    [types.Observation],
    tuple[types.Action, types.Array]
]


class FromOneHot(Wrapper):
    """Convert one-hot action to its value + cache action space."""

    def __init__(self, env: dm_env.Environment) -> None:
        super().__init__(env)
        act_spec = env.action_spec()
        self._is_discrete = isinstance(act_spec, dm_env.specs.DiscreteArray)
        dtype = np.int32 if self._is_discrete else np.float32
        self._act_spec = act_spec.replace(dtype=dtype)

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def step(self, action: types.Action) -> dm_env.TimeStep:
        if self._is_discrete:
            action = action.argmax(-1)
        return self.env.step(action)

    def action_spec(self) -> dm_env.specs.Array:
        return self._act_spec


def train_loop(env: dm_env.Environment,
               policy: ActionLogProbFn,
               num_steps: int,
               prev_timestep: dm_env.TimeStep | None = None,
               ) -> tuple[types.Trajectory, dm_env.TimeStep]:
    trajectory = collections.defaultdict(list)
    ts = prev_timestep or env.reset()
    for _ in range(num_steps):
        obs = ts.observation
        action, log_prob = policy(obs)
        ts = env.step(action)
        reward = _from_none(ts.reward)
        discount = _from_none(ts.discount)
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['discounts'].append(discount)
        trajectory['log_probs'].append(log_prob)
    trajectory['observations'].append(ts.observation)

    def stack(xs): return tree_map(lambda *x: np.stack(x), *xs)
    trajectory = tree_map(
        stack, trajectory, is_leaf=lambda x: isinstance(x, list))
    return dict(trajectory), ts


def eval_loop(env: dm_env.Environment, policy: ActionLogProbFn) -> np.floating:
    ts = env.reset()
    shape = np.asanyarray(ts.step_type).shape
    cont = np.ones(shape, dtype=bool)
    reward = np.zeros(shape, dtype=env.reward_spec().dtype)
    while cont.any():
        action, _ = policy(ts.observation)
        ts = env.step(action)
        reward += cont * _from_none(ts.reward)
        cont *= np.logical_not(ts.last())
    return reward


def _from_none(x, fill_value=0., dtype=float):
    """First dm_env.TimeStep may contain None."""
    return np.where(x == None, fill_value, x).astype(dtype)
