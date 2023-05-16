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

    def shift(x):
        x = sg(x)
        return jnp.concatenate([x[1:], jnp.zeros_like(x[-1:])])

    def cross_entropy_loss(policy, q_values, actions):
        chex.assert_rank([q_values, actions], [1, 2])
        q_values, actions = jax.lax.stop_gradient((q_values, actions))
        normalized_weights = jax.nn.softmax(q_values / cfg.entropy_coef)
        return -jnp.sum(normalized_weights * policy.log_prob(actions))

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
        chex.assert_rank([a_t, r_t, disc_t, log_mu_t], [3, 2, 2, 2])
        actor_fn = hk.BatchApply(jax.vmap(nets.actor, in_axes=(None, 0)))
        critic_fn = hk.BatchApply(jax.vmap(nets.critic, in_axes=(None, 0, 0)))
        import pdb; pdb.set_trace()

        policy_t = actor_fn(params, obs_t)
        log_pi_t = policy_t.log_prob(a_t)
        pi_a_t = policy_t.sample(seed=rng, size=(cfg.num_actions,))

        q_t = critic_fn(params, obs_t, a_t)
        target_q_t = jax.vmap(critic_fn, in_axes=(None, None, 0)
                              )(target_params, obs_t, pi_a_t)
        v_tp1 = shift(target_q_t.mean(0))

        log_rho_t = log_pi_t - log_mu_t
        adv_t = retrace(q_t, v_tp1, r_t, disc_t, log_rho_t, cfg.lambda_)
        critic_loss = jnp.square(q_t - target_q_t - adv_t).mean()
        actor_loss = cross_entropy_loss(policy_t, target_q_t, pi_a_t).mean()
        # consider straight through for continuous action spaces

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'advantage': adv_t.mean(),
            'entropy': policy_t.entropy().mean(),
            'log_importance_factor': log_rho_t.mean(),
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
        grad_fn = jax.grad(loss_fn)
        grads, metrics = loss_fn(params, target_params, subkey, *args)
        metrics.update(grad_norm=optax.global_norm(grads))
        state = state.update(grads)
        return state.replace(rng=rng), metrics

    return step
