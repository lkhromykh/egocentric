import jax
import jax.numpy as jnp
import haiku as hk
import chex
import optax

from src.rl.config import Config
from src.rl.networks import Networks
from src.rl.training_state import TrainingState
from src.rl import ops
from src.rl import types_ as types


def vpi(cfg: Config, nets: Networks) -> types.StepFn:
    sg = jax.lax.stop_gradient
    def tree_slice(t, sl): return jax.tree_util.tree_map(lambda x: x[sl], t)

    def cross_entropy(q_values, log_probs, temperature):
        tempered_q_values = q_values / temperature
        normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)
        normalized_weights = sg(normalized_weights)
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

        tc_aug = jax.vmap(ops.augmentation_fn, in_axes=(None, 0, None))
        target_obs_t = obs_t.copy()
        for key, val in obs_t.items():
            if val.dtype == jnp.uint8:
                rng, k1, k2 = jax.random.split(rng, 3)
                target_obs_t[key] = tc_aug(k1, val, 4)
                obs_t[key] = tc_aug(k2, val, 4)

        policy_t = nets.actor(params, obs_t)
        log_pi_t = policy_t[:-1].log_prob(a_t)
        tau = jnp.maximum(params['~']['temperature'], -18.)
        tau = jax.nn.softplus(tau) + 1e-8
        act_dim = a_t.shape[-1]
        match cfg.action_space:
            # Avoid MC sampling for discrete action spaces.
            case 'discrete':
                pi_a_dash_t = jnp.eye(act_dim)  # as one_hot
                pi_a_dash_t = jnp.expand_dims(pi_a_dash_t, (1, 2))
                shape = (1, *policy_t.batch_shape, 1)
                pi_a_dash_t = jnp.tile(pi_a_dash_t, shape)
                log_pi_dash_t = policy_t.log_prob(pi_a_dash_t)
            case 'continuous':
                pi_a_dash_t, log_pi_dash_t = \
                    policy_t.experimental_sample_and_log_prob(
                        seed=rng, sample_shape=(cfg.num_actions,)
                    )
            case _: raise ValueError(cfg.action_space)
        obs_tTm1, target_obs_tTm1 = tree_slice(
            (obs_t, target_obs_t), jnp.s_[:-1])
        q_t = nets.critic(params, obs_tTm1, a_t)
        target_q_t = nets.critic(target_params, target_obs_tTm1, a_t)
        target_q_dash_t = jax.vmap(
            nets.critic, in_axes=(None, None, 0))(
            target_params, target_obs_t, pi_a_dash_t)
        match cfg.action_space:
            case 'discrete':
                prob_t = jnp.exp(log_pi_dash_t)
                v_t = jnp.expand_dims(prob_t, -1) * target_q_dash_t
                v_t = jnp.sum(v_t, 0)
            case 'continuous':
                v_t = target_q_dash_t.mean(0)

        in_axes = 5 * (1,) + (None,)
        target_fn = jax.vmap(ops.retrace, in_axes,
                             out_axes=1, axis_name='batch')
        in_axes = 2 * (-1,) + 4 * (None,)
        target_fn = jax.vmap(target_fn, in_axes,
                             out_axes=-1, axis_name='q_ensemble')

        log_rho_t = log_pi_t - log_mu_t
        disc_t *= cfg.gamma
        target_q_t = target_fn(target_q_t, v_t[1:], r_t, disc_t,
                               log_rho_t, cfg.lambda_)
        target_q_t = target_q_t.min(-1, keepdims=True)
        critic_loss = jnp.square(q_t - sg(target_q_t))
        not_reset = disc_t > 0  # mask cross episode bootstrap estimation
        critic_loss = jnp.mean(jnp.expand_dims(not_reset, -1) * critic_loss)

        target_q_dash_t = target_q_dash_t.mean(-1)
        match cfg.action_space:
            case 'discrete':
                entropy_t = policy_t.entropy()
                target_entropy = cfg.entropy_per_dim * jnp.log(act_dim)
                actor_loss = cross_entropy(target_q_dash_t, log_pi_dash_t, tau)
            case 'continuous':
                entropy_t = -log_pi_dash_t.mean(0)
                target_entropy = (cfg.entropy_per_dim - 1.) * act_dim
                actor_loss = -target_q_dash_t.mean(0) - sg(tau) * entropy_t
        actor_loss = jnp.mean(actor_loss)
        temp_loss = tau * sg(entropy_t.mean() - target_entropy)

        adv_gap = target_q_dash_t.max(0) - target_q_dash_t.min(0)
        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'temperature_loss': temp_loss,
            'temperature': tau,
            'entropy': entropy_t.mean(),
            'log_importance': log_rho_t.mean(),
            'reward': r_t.mean(),
            'value': target_q_t.mean(),
            'adv_gap': adv_gap.mean(),
        }
        return actor_loss + critic_loss + temp_loss, metrics

    def step(state: TrainingState,
             batch: types.Trajectory
             ) -> tuple[TrainingState, types.Metrics]:
        params = state.params
        target_params = state.target_params
        rng, subkey = jax.random.split(state.rng)

        batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
        args = map(
            batch.get,
            ('observations', 'actions', 'rewards', 'discounts', 'log_probs')
        )
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, target_params, subkey, *args)
        metrics.update(grad_norm=optax.global_norm(grad))
        state = state.update(grad)
        return state.replace(rng=rng), metrics

    @chex.assert_max_traces(1)
    def fuse(state: TrainingState,
             batch: types.Trajectory,
             ) -> tuple[TrainingState, types.Metrics]:
        # Not passing num steps for compliance with the StepFn signature.
        num_steps = len(batch['rewards']) // cfg.batch_size
        for i in range(num_steps):
            b = cfg.batch_size * i
            e = b + cfg.batch_size
            subbatch = tree_slice(batch, jnp.s_[b:e])
            state, metrics = step(state, subbatch)
        print('Fusing %d gradient steps' % num_steps)
        return state, metrics

    return fuse
