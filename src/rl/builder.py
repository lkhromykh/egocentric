import os

import dm_env.specs
import numpy as np
from dm_control import suite
import jax
import optax
import haiku as hk

from src.rl.replay_buffer import ReplayBuffer
from src.rl.networks import Networks
from src.rl.config import Config
from src.rl.training_state import TrainingState
from src.rl.alg import vpi, StepFn
from src.rl.ops import environment_loop
from src.rl import types_ as types


class Builder:

    PARAMS = 'params.pkl'
    CONFIG = 'config.yaml'

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.exists(path := self.exp_path()):
            os.makedirs(path)
        if not os.path.exists(path := self.exp_path(Builder.CONFIG)):
            cfg.save(path)

    def make_env(self, rng: int) -> dm_env.Environment:
        return suite.load('cartpole', 'balance', task_kwargs={'random': rng})

    def make_networks(self, env: dm_env.Environment) -> Networks:
        return Networks.make_networks(
            self.cfg,
            env.observation_spec(),
            env.action_spec()
        )

    def make_training_state(self,
                            rng: types.RNG,
                            params: hk.Params,
                            ) -> TrainingState:
        c = self.cfg
        optim = optax.adam(c.learning_rate)
        return TrainingState.init(rng, params, optim, c.polyak_tau)

    def make_replay_buffer(self,
                           rng: int | np.random.Generator,
                           env: dm_env.Environment
                           ) -> ReplayBuffer:
        # Replace this mess. Ensure length consistency.
        act_spec = env.action_spec()
        obs_spec = env.observation_spec()
        test_traj = environment_loop(
            env,
            lambda _: (act_spec.generate_value(), 0)
        )
        traj_len = len(test_traj['actions'])
        if isinstance(act_spec, dm_env.specs.DiscreteArray):
            # Replace by one-hot signature.
            act_spec = np.zeros(act_spec.num_values, act_spec.dtype)

        def time_major(x, times=traj_len):
            return np.zeros((times,) + x.shape, dtype=x.dtype)
        signature = {
            'actions': act_spec,
            'rewards': env.reward_spec(),
            'discounts': env.discount_spec(),
            'log_probs': np.zeros((), act_spec.dtype),
        }
        tm = jax.tree_util.tree_map
        signature = tm(time_major, signature)
        signature['observations'] = tm(
            lambda x: time_major(x, traj_len + 1), obs_spec)
        return ReplayBuffer(rng, self.cfg.buffer_capacity, signature)

    def make_step_fn(self, networks: Networks) -> StepFn:
        fn = vpi(self.cfg, networks)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn

    def exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        path = os.path.join(logdir, path)
        return os.path.abspath(path)
