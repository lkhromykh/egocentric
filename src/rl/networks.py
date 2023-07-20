from collections.abc import Callable
from typing import NamedTuple
import re

from dm_env import specs
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp

from src.rl import types_ as types
from src.rl.config import Config

Array = types.Array
tfd = tfp.distributions
_out_w_init = hk.initializers.TruncatedNormal(stddev=1e-3)


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
                 act: str,
                 norm: str,
                 activate_final: bool = True,
                 name: str | None = None,
                 ) -> None:
        super().__init__(name)
        self.layers = layers
        self.act = act
        self.norm = norm
        self.activate_final = activate_final

    def __call__(self, x: Array) -> Array:
        for idx, layer in enumerate(self.layers):
            x = hk.Linear(layer)(x)
            if idx != len(self.layers) - 1 or self.activate_final:
                x = _get_norm(self.norm)(x)
                x = _get_act(self.act)(x)
        return x


class Encoder(hk.Module):

    def __init__(self,
                 obs_keys: str,
                 mlp_layers: types.Layers,
                 cnn_kernels: types.Layers,
                 cnn_depths: types.Layers,
                 cnn_strides: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.obs_keys = obs_keys
        self.mlp_layers = mlp_layers
        self.cnn_kernels = cnn_kernels
        self.cnn_depths = cnn_depths
        self.cnn_strides = cnn_strides
        self.act = act
        self.norm = norm
        
    def __call__(self, obs: types.Observation) -> Array:
        mlp_feat, cnn_feat, emb = [], [], []
        def concat(x): return jnp.concatenate(x, -1)
        selected_keys = []
        for key, feat in sorted(obs.items()):
            if re.search(self.obs_keys, key):
                selected_keys.append(key)
                match feat.dtype:
                    case jnp.uint8: cnn_feat.append(feat)
                    case _: mlp_feat.append(jnp.atleast_1d(feat))
        print(f'{self.name} selected keys: {selected_keys}')
        if mlp_feat:
            mlp_feat = concat(mlp_feat)
            emb.append(self._mlp(mlp_feat))
        if cnn_feat:
            cnn_feat = concat(cnn_feat)
            emb.append(self._cnn(cnn_feat))

        emb = concat(emb)
        emb = _get_norm('layer')(emb)
        return jax.lax.tanh(emb)

    def _mlp(self, x):
        mlp = MLP(self.mlp_layers, self.act, self.norm, activate_final=False)
        return mlp(x)

    def _cnn(self, x):
        x /= 255.
        prefix = x.shape[:-3]
        x = jnp.reshape(x, (np.prod(prefix, dtype=int),) + x.shape[-3:])
        cnn = tuple(zip(self.cnn_depths, self.cnn_kernels, self.cnn_strides))
        for i, (depth, kernel, stride) in enumerate(cnn):
            x = hk.Conv2D(depth, kernel, stride, padding='VALID')(x)
            if i != len(cnn) - 1:
                x = _get_norm(self.norm)(x)
                x = _get_act(self.act)(x)
        return jnp.reshape(x, prefix + (-1,))


class Actor(hk.Module):

    def __init__(self,
                 action_spec: types.ActionSpecs,
                 layers: types.Layers,
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.action_spec = action_spec
        self.layers = layers
        self.act = act
        self.norm = norm

    def __call__(self, state: Array) -> tfd.Distribution:
        state = MLP(self.layers, self.act, self.norm)(state)
        match sp := self.action_spec:
            case specs.DiscreteArray():
                logits = hk.Linear(sp.num_values, w_init=_out_w_init)(state)
                dist = tfd.OneHotCategorical(logits=logits, dtype=jnp.int32)
            case specs.BoundedArray():
                fc = hk.Linear(2 * sp.shape[0], w_init=_out_w_init)
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
                 act: str,
                 norm: str,
                 name: str | None = None
                 ) -> None:
        super().__init__(name)
        self.layers = layers
        self.act = act
        self.norm = norm

    def __call__(self,
                 state: Array,
                 action: Array,
                 ) -> Array:
        x = jnp.concatenate([state, action.astype(state.dtype)], -1)
        x = MLP(self.layers, self.act, self.norm)(x)
        fc = hk.Linear(1, w_init=_out_w_init)
        return fc(x)


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
                    cfg.cnn_kernels,
                    cfg.cnn_depths,
                    cfg.cnn_strides,
                    cfg.activation,
                    cfg.normalization,
                    name=name
                )

            def actor(obs):
                if cfg.asymmetric:
                    keys = cfg.actor_keys
                    name = 'actor_encoder'
                    sg = lambda x: x
                else:
                    keys = cfg.critic_keys
                    name = 'critic_encoder'
                    sg = jax.lax.stop_gradient
                state = encoder(keys, name)(obs)
                state = sg(state)
                actor_ = Actor(
                    action_spec,
                    cfg.actor_layers,
                    cfg.activation,
                    cfg.normalization,
                    name='actor'
                )
                return actor_(state)

            def critic(obs, action):
                state = encoder(cfg.critic_keys, 'critic_encoder')(obs)
                critic_ = CriticsEnsemble(
                    cfg.ensemble_size,
                    cfg.critic_layers,
                    cfg.activation,
                    cfg.normalization,
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
                init_temp = jnp.log(jnp.exp(cfg.init_temperature) - 1)
                hk.get_parameter('temperature',
                                 (),
                                 init=hk.initializers.Constant(init_temp))

            return init, (actor, critic, act)

        init, apply = model
        return cls(
            init=init,
            actor=apply[0],
            critic=apply[1],
            act=apply[2]
        )


def _get_act(act: str) -> Callable[[Array], Array]:
    if act == 'none':
        return lambda x: x
    if hasattr(jax.nn, act):
        return getattr(jax.nn, act)
    raise ValueError(act)


def _get_norm(norm: str) -> Callable[[Array], Array]:
    match norm:
        case 'none':
            return lambda x: x
        case 'layer':
            return hk.LayerNorm(axis=-1,
                                create_scale=True,
                                create_offset=True)
        case 'rms':
            return hk.RMSNorm(axis=-1,
                              create_scale=True)
        case _:
            raise ValueError(norm)
