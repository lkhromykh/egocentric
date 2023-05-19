from typing import Callable

import jax
import jax.numpy as jnp
import haiku as hk
import chex
import optax

from src.rl.config import Config
from src.rl.networks import Networks
from src.rl.training_state import TrainingState
from src.rl.ops import retrace
from src.rl import types_ as types

StepFn = Callable[
    [TrainingState, types.Trajectory],
    tuple[TrainingState, types.Metrics]
]


def vpi(cfg: Config, nets: Networks) -> StepFn:
    sg = jax.lax.stop_gradient
    def tree_slice(t, sl): return jax.tree_util.tree_map(lambda x: x[sl], t)

    def cross_entropy(q_values, log_probs):
        tempered_q_values = sg(q_values) / cfg.entropy_coef
        normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)
        return -jnp.sum(normalized_weights * log_probs, axis=0)

    def loss_fn(
            params: hk.Params,
            target_params: hk.Params,
            rng: types.RNG,
            obs_t: types.Observation,
            a_t: types.Action,
            r_t: types.Array,
            disc_t: types.Array,
            log_mu_t: types.Array,
    ) -> tuple[chex.Numeric, types.Metrics]:
        # time major arrays are expected [TIME, BATCH, ...].
        chex.assert_tree_shape_prefix(obs_t, (cfg.sequence_len + 1, cfg.batch_size))
        chex.assert_tree_shape_prefix(a_t, (cfg.sequence_len, cfg.batch_size))

        policy_t = nets.actor(params, obs_t)
        log_pi_t = policy_t[:-1].log_prob(a_t)
        pi_a_dash_t, log_pi_dash_t = policy_t.experimental_sample_and_log_prob(
            seed=rng, sample_shape=(cfg.num_actions,))
        obs_tTm1 = tree_slice(obs_t, jnp.s_[:-1])
        q_t = nets.critic(params, obs_tTm1, a_t)
        target_q_t = nets.critic(target_params, obs_tTm1, a_t)
        target_q_dash_t = jax.vmap(nets.critic, in_axes=(None, None, 0)
                                   )(target_params, obs_t, pi_a_dash_t)
        v_tp1 = target_q_dash_t.mean(0)[1:]

        # targets = r_t + cfg.gamma * disc_t * v_tp1
        # critic_loss = jnp.square(q_t - sg(targets)).mean()

        in_axes = 5 * (1,) + (None,)
        adv_fn = jax.vmap(retrace, in_axes, out_axes=1)
        log_rho_t = log_pi_t - log_mu_t
        disc_t *= cfg.gamma
        adv_t = adv_fn(target_q_t, v_tp1, r_t, disc_t, log_rho_t, cfg.lambda_)
        critic_loss = jnp.square(q_t - sg(target_q_t + adv_t)).mean()

        # Avoid maximizing terminal states.
        entropy = policy_t[:-1].entropy()
        target_q_dash_t, log_pi_dash_t = map(
            lambda t: t[:, :-1],
            (target_q_dash_t, log_pi_dash_t)
        )
        if cfg.action_space == 'continuous':
            actor_loss = -target_q_dash_t.mean(0)
            actor_loss -= cfg.entropy_coef * entropy
        else:
            actor_loss = cross_entropy(target_q_dash_t, log_pi_dash_t)
        actor_loss = actor_loss.mean()

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'advantage': adv_t.mean(),
            'entropy': entropy.mean(),
            'log_importance': log_rho_t.mean(),
            'reward': r_t.mean(),
            'value': target_q_t.mean()
        }
        return actor_loss + critic_loss, metrics

    @chex.assert_max_traces(1)
    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        params = state.params
        target_params = state.target_params
        rng, subkey = jax.random.split(state.rng)

        args = map(
            batch.get,
            ('observations', 'actions', 'rewards', 'discounts', 'log_probs')
        )
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, metrics = grad_fn(params, target_params, subkey, *args)
        metrics.update(grad_norm=optax.global_norm(grads))
        state = state.update(grads)
        return state.replace(rng=rng), metrics

    return step
