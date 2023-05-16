import collections.abc
from typing import TypedDict, Any

import jax
import jax.numpy as jnp

from dm_env import specs

RNG = jax.random.PRNGKey
Array = jnp.ndarray

Action = Array
ActionSpecs = specs.DiscreteArray | specs.BoundedArray
Observation = collections.abc.MutableMapping[str, Array]
ObservationSpecs = collections.abc.MutableMapping[str, specs.Array]

Policy = collections.abc.Callable[[Observation], Action]
Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, Any]


class Trajectory(TypedDict, total=False):
    observations: list[Observation]
    actions: list[Action]
    rewards: list[float]
    discounts: list[float]
    next_observations: list[Observation]
