from typing import NamedTuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax


@jax.tree_util.register_pytree_node_class
class TrainingState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    step: jnp.ndarray
    tx: optax.TransformUpdateFn
    target_update_var: float

    def update(self, grads: hk.Params) -> 'TrainingState':
        params = self.params
        target_params = self.target_params
        opt_state = self.opt_state
        step = self.step

        updates, opt_state = self.tx(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_params = optax.incremental_update(
            params, target_params, self.target_update_var)

        return self._replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            step=step + 1
        )

    @classmethod
    def init(cls,
             rng: jax.random.PRNGKey,
             params: hk.Params,
             optim: optax.GradientTransformation,
             target_update_var: float
             ) -> 'TrainingState':
        return cls(
            params=params,
            target_params=params,
            opt_state=optim.init(params),
            rng=rng,
            step=jnp.int32(0),
            target_update_var=target_update_var,
            tx=optim.update,
        )

    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def tree_flatten(self):
        children = (self.params,
                    self.target_params,
                    self.opt_state,
                    self.rng,
                    self.step
                    )
        aux = (self.tx, self.target_update_var)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)
