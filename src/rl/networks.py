from collections.abc import Callable
from typing import NamedTuple
import re

import chex
from dm_env import specs
import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp

from src.rl import types_ as types
from src.rl.config import Config

Array = types.Array
tfd = tfp.distributions
act = jax.nn.elu


def layer_norm(x):
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


class TransformedDistribution(tfd.TransformedDistribution):

    def log_prob(self, event):
        threshold = .99999
        event = jnp.clip(event, -threshold, threshold)
        return super().log_prob(event)

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class MLP(hk.Module):

    def __init__(self,
                 layers: types.Layers,
                 activation: Callable = act,
                 name: str | None = None,
                 ) -> None:
        super().__init__(name)
        self.layers = layers
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = hk.Linear(layer, with_bias=False)(x)
            x = layer_norm(x)
            x = self.activation(x)
        return x


class CNN(hk.Module):

    def __init__(self,
                 filters: types.Layers,
                 kernels: types.Layers,
                 strides: types.Layers,
                 name: str | None = None
                 ) -> None:
        super().__init__(name=name)
        self.filters = filters
        self.kernels = kernels
        self.strides = strides

    def __call__(self, x: Array) -> Array:
        chex.assert_type(x, int)
        prefix = x.shape[:-3]
        x = jnp.reshape(x / 255. - 0.5, (-1,) + x.shape[-3:])
        for f, k, s in zip(self.filters, self.kernels, self.strides):
            x = hk.Conv2D(f, (k, k), s, padding='VALID', with_bias=False)(x)
            x = layer_norm(x)
            x = act(x)
        return jnp.reshape(x, prefix + (-1,))


class Encoder(hk.Module):

    def __init__(self,
                 obs_keys: str,
                 mlp_layers: types.Layers,
                 cnn_filters: types.Layers,
                 cnn_kernels: types.Layers,
                 cnn_strides: types.Layers,
                 name: str | None = None
                 ) -> None:
        super().__init__(name=name)
        self.obs_keys = obs_keys
        self.mlp_layers = mlp_layers
        self.cnn_filters = cnn_filters
        self.cnn_kernels = cnn_kernels
        self.cnn_strides = cnn_strides

    def __call__(self, obs: types.Observation) -> Array:
        cnn_feat, mlp_feat, emb = [], [], []
        def concat(x): return jnp.concatenate(x, -1)
        selected_keys = []
        for key, feat in sorted(obs.items()):
            if re.search(self.obs_keys, key):
                selected_keys.append(key)
                match feat.dtype:
                    case jnp.uint8: cnn_feat.append(feat)
                    case _: mlp_feat.append(jnp.atleast_1d(feat))
        if mlp_feat:
            mlp_feat = concat(mlp_feat)
            mlp = MLP(self.mlp_layers, jax.nn.tanh)
            emb.append(mlp(mlp_feat))
        if cnn_feat:
            cnn_feat = concat(cnn_feat)
            cnn = CNN(filters=self.cnn_filters,
                      kernels=self.cnn_kernels,
                      strides=self.cnn_strides)
            emb.append(cnn(cnn_feat))
        return concat(emb)


class Actor(hk.Module):

    def __init__(self,
                 action_spec: types.ActionSpecs,
                 layers: types.Layers,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.action_spec = action_spec
        self.layers = layers

    def __call__(self, state: Array) -> tfd.Distribution:
        state = MLP(self.layers)(state)
        w_init = hk.initializers.VarianceScaling(1e-3)
        match sp := self.action_spec:
            case specs.DiscreteArray():
                logits = hk.Linear(sp.num_values, w_init=w_init)(state)
                dist = tfd.OneHotCategorical(logits=logits, dtype=jnp.int32)
            case specs.BoundedArray():
                fc = hk.Linear(2 * sp.shape[0], w_init=w_init)
                mean, std = jnp.split(fc(state), 2, -1)
                std = jax.nn.sigmoid(std) + 1e-3
                dist = tfd.Normal(mean, std)
                dist = TransformedDistribution(dist, tfp.bijectors.Tanh())
                dist = tfd.Independent(dist, 1)
            case _:
                raise ValueError(sp)
        return dist


class Critic(hk.Module):

    def __init__(self,
                 layers: types.Layers,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.layers = layers

    def __call__(self,
                 state: Array,
                 action: Array,
                 ) -> Array:
        x = jnp.concatenate([state, action.astype(state.dtype)], -1)
        x = MLP(self.layers)(x)
        return hk.Linear(1)(x)


class CriticsEnsemble(hk.Module):

    def __init__(self,
                 ensemble_size: int,
                 *args,
                 name: str | None = None,
                 **kwargs
                 ) -> None:
        super().__init__(name)
        self.ensemble_size = ensemble_size
        self._factory = lambda n: Critic(*args, name=n, **kwargs)

    def __call__(self, *args, **kwargs):
        values = []
        for i in range(self.ensemble_size):
            critic = self._factory('critic_%d' % i)
            values.append(critic(*args, **kwargs))
        return jnp.concatenate(values, -1)


class Networks(NamedTuple):
    init: Callable
    actor: Callable
    critic: Callable
    act: Callable

    @classmethod
    def make_networks(
            cls,
            cfg: Config,
            observation_spec: types.ObservationSpecs,
            action_spec: types.ActionSpecs
    ) -> 'Networks':
        dummy_obs = jax.tree_map(
            lambda sp: sp.generate_value(),
            observation_spec
        )

        @hk.without_apply_rng
        @hk.multi_transform
        def model():
            def encoder(keys, name=None):
                return Encoder(
                    keys,
                    cfg.mlp_layers,
                    cfg.cnn_filters,
                    cfg.cnn_kernels,
                    cfg.cnn_strides,
                    name=name
                )

            def actor(obs):
                state = encoder(cfg.actor_keys, 'actor_encoder')(obs)
                actor_ = Actor(
                    action_spec,
                    cfg.actor_layers,
                    name='actor'
                )
                return actor_(state)

            def critic(obs, action):
                state = encoder(cfg.critic_keys, 'critic_encoder')(obs)
                critic_ = CriticsEnsemble(
                    cfg.ensemble_size,
                    cfg.critic_layers,
                    name='critic'
                )
                return critic_(state, action)

            def act(seed: types.RNG,
                    obs: types.Observation,
                    training: bool
                    ) -> tuple[Array, Array]:
                dist = actor(obs)
                action, log_prob = jax.lax.cond(
                    training,
                    lambda: dist.experimental_sample_and_log_prob(seed=seed),
                    lambda: (dist.mode(), jnp.zeros(dist.batch_shape))
                )
                return action, log_prob

            def init():
                dist = actor(dummy_obs)
                critic(dummy_obs, dist.sample(seed=jax.random.PRNGKey(0)))

            return init, (actor, critic, act)

        init, apply = model
        return cls(
            init=init,
            actor=apply[0],
            critic=apply[1],
            act=apply[2]
        )
