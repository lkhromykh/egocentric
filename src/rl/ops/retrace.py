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
    rho_t = lambda_ * jnp.minimum(1., jnp.exp(log_rho_t))
    xs = (q_t, v_tp1, r_t, disc_t, rho_t)
    chex.assert_rank(xs, 1)
    chex.assert_equal_shape(xs)
    chex.assert_scalar_non_negative(lambda_)

    def fn(acc, x):  # 1606.02647
        q, next_v, r, disc, c = x
        resid = r + disc * next_v - q
        acc = resid + disc * c * acc
        return acc, acc
    _, resids = jax.lax.scan(fn, 0., xs, reverse=True)
    return resids
