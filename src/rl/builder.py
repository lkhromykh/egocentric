import os
from typing import Callable

import cloudpickle
import dm_env.specs
import numpy as np
import jax
import optax
import haiku as hk
from rltools.dmc_wrappers import AsyncEnv

from src.rl.replay_buffer import ReplayBuffer
from src.rl.networks import Networks
from src.rl.config import Config
from src.rl.training_state import TrainingState
from src.rl.alg import vpi
from src.rl.ops.environment import FromOneHot
from src.rl import types_ as types


class Builder:
    PARAMS = 'params.pkl'
    STATE = 'state.cpkl'
    CONFIG = 'config.yaml'
    REPLAY = 'replay.npz'

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.exists(path := self.exp_path()):
            os.makedirs(path)
        if not os.path.exists(path := self.exp_path(Builder.CONFIG)):
            cfg.save(path)

    def make_env_fn(self, seed: int) -> Callable[[], dm_env.Environment]:
        c = self.cfg
        def wrap(env): return FromOneHot(env)
        match c.task.split('_'):
            case 'dmc', domain, task:
                from dm_control import suite

                def env_fn():
                    return wrap(suite.load(
                        domain, task,
                        task_kwargs={'random': seed},
                        environment_kwargs={'flat_observation': True}
                    ))
            case ['src']:
                from src.suite import load

                def env_fn():
                    return wrap(load(
                        seed,
                        action_mode=c.action_space,
                        img_size=(64, 64),
                        control_timestep=.05,
                        time_limit=3.,
                    ))
            case 'ur', _:
                from ur_env.remote import RemoteEnvClient
                address = None
                def env_fn(): return wrap(RemoteEnvClient(address))
            case _:
                raise ValueError(self.cfg.task)

        return env_fn

    def make_envs(self, rng: int) -> dm_env.Environment:
        rng = np.random.default_rng(rng)
        seeds = rng.integers(0, np.iinfo(np.int32).max, self.cfg.num_envs)
        return AsyncEnv(
            [self.make_env_fn(seed) for seed in seeds],
            context='spawn'
        )

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
        if os.path.exists(path := self.exp_path(Builder.STATE)):
            print('Loading existing state.')
            with open(path, 'rb') as f:
                state = cloudpickle.load(f)
            return jax.device_put(state)
        c = self.cfg
        optim = optax.adamw(c.learning_rate, weight_decay=c.weight_decay)
        optim = optax.chain(optax.clip_by_global_norm(c.max_grad), optim)

        def label_fn(key):
            match key[0]:
                case 'a': return 'actor'
                case 'c': return 'critic'
                case '~': return 'dual'
                case _: raise ValueError(key)

        labels = type(params)({
            k: jax.tree_util.tree_map(lambda t: label_fn(k), v)
            for k, v in params.items()
        })
        optim = optax.multi_transform(
            {'actor': optim,
             'critic': optim,
             'dual': optax.adam(c.temp_learning_rate)},
            labels
        )
        return TrainingState.init(rng, params, optim, c.polyak_tau)

    def make_replay_buffer(self,
                           rng: int | np.random.Generator,
                           env: dm_env.Environment
                           ) -> ReplayBuffer:
        if os.path.exists(path := self.exp_path(Builder.REPLAY)):
            return ReplayBuffer.load(path)
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

    def make_step_fn(self, networks: Networks) -> types.StepFn:
        fn = vpi(self.cfg, networks)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn

    def exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        path = os.path.join(logdir, path)
        return os.path.abspath(path)
