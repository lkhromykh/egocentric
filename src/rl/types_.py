import collections.abc
from typing import Any, Callable, TypedDict

import jax
from dm_env import specs

from src.rl.training_state import TrainingState

RNG = Array = jax.Array

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
    log_probs: list[float]


StepFn = Callable[[TrainingState, Trajectory], tuple[TrainingState, Metrics]]
