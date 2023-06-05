import jax
import jax.numpy as jnp
import chex


def retrace(q_t: chex.Array,
            v_tp1: chex.Array,
            r_t: chex.Array,
            disc_t: chex.Array,
            log_rho_t: chex.Array,
            lambda_: float
            ) -> chex.Array:
    """Original estimation from the 1606.02647"""
    rho_t = lambda_ * jnp.minimum(1., jnp.exp(log_rho_t))
    xs = (q_t, v_tp1, r_t, disc_t, rho_t)
    chex.assert_rank(xs, 1)
    chex.assert_equal_shape(xs)
    chex.assert_scalar_non_negative(lambda_)

    def fn(acc, x):
        q, next_v, r, disc, c = x
        resid = r + disc * next_v - q
        acc = resid + disc * c * acc
        return acc, acc
    _, resid_t = jax.lax.scan(fn, 0., xs, reverse=True)
    return q_t + resid_t


def retrace2(q_t: chex.Array,
             v_tp1: chex.Array,
             r_t: chex.Array,
             disc_t: chex.Array,
             log_rho_t: chex.Array,
             lambda_: float
             ) -> chex.Array:
    """Equivalent estimation from the 1611.01224."""
    rho_t = lambda_ * jnp.minimum(1, jnp.exp(log_rho_t))
    q_tp1 = jnp.concatenate([q_t[1:], jnp.zeros_like(q_t[:1])])
    xs = (q_tp1, v_tp1, r_t, disc_t, rho_t)
    chex.assert_rank(xs, 1)
    chex.assert_equal_shape(xs)
    chex.assert_scalar_non_negative(lambda_)

    def fn(acc, x):
        next_q, next_v, r, disc, c = x
        acc = r + disc * next_v + disc * c * (acc - next_q)
        return acc, acc
    _, target_q_t = jax.lax.scan(fn, q_tp1[-1], xs, reverse=True)
    return target_q_t


def pql(q_t: chex.Array,
        v_tp1: chex.Array,
        r_t: chex.Array,
        disc_t: chex.Array,
        log_rho_t: chex.Array,
        lambda_: chex.Numeric
        ) -> chex.Array:
    """Peng's Q(λ)."""
    log_rho_t = jnp.zeros_like(log_rho_t)  # unsafe but not conservative
    return retrace(q_t, v_tp1, r_t, disc_t, log_rho_t, lambda_)
