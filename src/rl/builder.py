import os

import dm_env.specs
import numpy as np
import jax
import optax
import haiku as hk
from rltools.dmc_wrappers import AutoReset, AsyncEnv, SequentialEnv

from src.rl.replay_buffer import ReplayBuffer
from src.rl.networks import Networks
from src.rl.config import Config
from src.rl.training_state import TrainingState
from src.rl.alg import vpi, StepFn
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
        c = self.cfg
        rng = np.random.default_rng(rng)
        match c.task.split('_'):
            case 'dmc', domain, task:
                from dm_control import suite

                def env_fn(seed):
                    return lambda: AutoReset(suite.load(
                        domain, task,
                        task_kwargs={'random': seed},
                        environment_kwargs={'flat_observation': True}
                    ))
            case ['src']:
                from src.suite import load

                def env_fn(seed):
                    return lambda: AutoReset(load(
                        seed,
                        action_mode=c.action_space,
                        img_size=(84, 84),
                        time_limit=5,
                    ))
            case _:
                raise ValueError(self.cfg.task)
        seeds = rng.integers(0, np.iinfo(np.int32).max, c.num_envs)
        return SequentialEnv([env_fn(seed) for seed in seeds])

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
        optim = optax.adamw(c.learning_rate, weight_decay=c.weight_decay)
        optim = optax.chain(optax.clip_by_global_norm(c.max_grad), optim)
        return TrainingState.init(rng, params, optim, c.polyak_tau)

    def make_replay_buffer(self,
                           rng: int | np.random.Generator,
                           env: dm_env.Environment
                           ) -> ReplayBuffer:
        seq_len = self.cfg.sequence_len
        act_spec = env.action_spec()
        if isinstance(act_spec, dm_env.specs.DiscreteArray):
            # Replace by one-hot signature.
            act_spec = np.zeros(act_spec.num_values, act_spec.dtype)
        signature = {
            'actions': act_spec,
            'rewards': env.reward_spec(),
            'discounts': env.discount_spec(),
            'log_probs': np.array(0., act_spec.dtype),
        }

        def time_major(x, times=seq_len):
            return np.zeros((times,) + x.shape, dtype=x.dtype)
        tmap = jax.tree_util.tree_map
        signature = tmap(time_major, signature)
        signature['observations'] = tmap(
            lambda x: time_major(x, seq_len + 1),
            env.observation_spec()
        )
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
