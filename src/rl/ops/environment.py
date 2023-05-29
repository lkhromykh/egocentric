from typing import Callable
import collections

import dm_env.specs
import numpy as np
from jax.tree_util import tree_map
from rltools.dmc_wrappers.base import Wrapper
from rltools.dmc_wrappers import AutoReset

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
        if not self._is_discrete:
            act_spec = act_spec.replace(dtype=np.float32)
        self._act_spec = act_spec

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def step(self, action: types.Action) -> dm_env.TimeStep:
        if self._is_discrete:
            action = action.argmax(-1)
        return super().step(action)

    def action_spec(self) -> dm_env.specs.Array:
        return self._act_spec


def train_loop(env: AutoReset,  # continue interacting after termination
               policy: ActionLogProbFn,
               num_steps: int,
               prev_timestep: dm_env.TimeStep | None = None,
               ) -> tuple[types.Trajectory, dm_env.TimeStep]:
    trajectory = collections.defaultdict(list)
    ts = env.reset()
    for _ in range(num_steps):
        assert not np.any(ts.last())
        obs = ts.observation
        action, log_prob = policy(obs)
        ts = env.step(action)
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(ts.reward)
        trajectory['discounts'].append(ts.discount)
        trajectory['log_probs'].append(log_prob)
    trajectory['observations'].append(ts.observation)

    def stack(xs): return tree_map(lambda *x: np.stack(x), *xs)
    trajectory = {
        k: tree_map(stack, v, is_leaf=lambda x: isinstance(x, list))
        for k, v in trajectory.items()
    }
    return trajectory, ts


def eval_loop(env: dm_env.Environment, policy: ActionLogProbFn) -> np.float:
    ts = env.reset()
    shape = np.asanyarray(ts.step_type).shape
    cont = np.ones(shape, dtype=bool)
    reward = np.zeros(shape, dtype=env.reward_spec().dtype)
    while cont.any():
        action, _ = policy(ts.observation)
        ts = env.step(action)
        r = ts.reward
        reward += cont * np.where(r == None, 0., r).astype(float)
        cont *= np.logical_not(ts.last())
    return reward
