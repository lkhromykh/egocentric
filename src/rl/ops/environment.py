from typing import Callable
import collections

import dm_env
import numpy as np
from jax.tree_util import tree_map
from rltools.dmc_wrappers import AutoReset

from src.rl import types_ as types

ActionLogProbFn = Callable[[types.Observation], tuple[types.Action, types.Array]]


def train_loop(env: AutoReset,  # continue interacting after termination
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
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(ts.reward)
        trajectory['discounts'].append(ts.discount * (not ts.last()))
        trajectory['log_probs'].append(log_prob)
    trajectory['observations'].append(ts.observation)

    def stack(xs): return tree_map(lambda *x: np.stack(x), *xs)
    trajectory = {
        k: tree_map(stack, v, is_leaf=lambda x: isinstance(x, list))
        for k, v in trajectory.items()
    }
    return trajectory, ts


def eval_loop(env: dm_env.Environment, policy: ActionLogProbFn) -> float:
    ts = env.reset()
    reward = 0
    while not ts.last():
        action, _ = policy(ts.observation)
        ts = env.step(action)
        reward += ts.reward
    return reward
