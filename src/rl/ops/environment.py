from typing import Callable
import collections

import dm_env.specs
import numpy as np
from jax.tree_util import tree_map
from rltools.dmc_wrappers import AutoReset

from src.rl import types_ as types

ActionLogProbFn = Callable[
    [types.Observation],
    tuple[types.Action, types.Array]
]


def from_one_hot(env, action):
    if isinstance(env.action_spec(), dm_env.specs.DiscreteArray):
        action = action.argmax(-1)
    return action


def train_loop(env: AutoReset,  # continue interacting after termination
               policy: ActionLogProbFn,
               num_steps: int,
               prev_timestep: dm_env.TimeStep | None = None,
               ) -> tuple[types.Trajectory, dm_env.TimeStep]:
    trajectory = collections.defaultdict(list)
    def act_transform(act): return from_one_hot(env, act)
    ts = prev_timestep or env.reset()
    for _ in range(num_steps):
        obs = ts.observation
        action, log_prob = policy(obs)
        ts = env.step(act_transform(action))
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(ts.reward)
        trajectory['discounts'].append(ts.discount * np.logical_not(ts.last()))
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
    def act_transform(act): return from_one_hot(env, act)
    while cont.any():
        action, _ = policy(ts.observation)
        ts = env.step(act_transform(action))
        reward += cont * ts.reward
        cont *= np.logical_not(ts.last())
    return reward
