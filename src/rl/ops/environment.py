from typing import Callable
import collections

import dm_env
import numpy as np
from jax.tree_util import tree_map

from src.rl import types_ as types

ActionLogProbFn = Callable[[types.Observation], tuple[types.Action, types.Array]]


def environment_loop(env: dm_env.Environment,
                     policy: ActionLogProbFn
                     ) -> types.Trajectory:
    trajectory = collections.defaultdict(list)
    ts = env.reset()
    while not ts.last():
        obs = ts.observation
        action, log_prob = policy(obs)
        ts = env.step(action)
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(ts.reward)
        trajectory['discounts'].append(ts.discount)
        trajectory['log_probs'].append(log_prob)
    trajectory['observations'].append(ts.observation)

    def stack(*xs): return tree_map(lambda *x: np.stack(x), *xs)
    return {k: tree_map(stack, v) for k, v in trajectory.items()}
