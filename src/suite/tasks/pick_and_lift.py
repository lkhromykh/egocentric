import numpy as np
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions

from src.suite import entities
from src.suite import common
from src.suite.tasks import base


class Box(entities.BoxWithVertexSites):

    def _build_observables(self):
        return entities.StaticPrimitiveObservables(self)


class PickAndLift(base.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prop = Box(
            half_lengths=common.BOX_SIZE,
            mass=common.BOX_MASS
        )
        self._arena.add_free_entity(self._prop)

        def distance(physics):
            pos, _ = self._get_mocap(physics)
            obj_pos, _ = self._prop.get_pose(physics)
            return obj_pos - pos

        self._task_observables['distance'] = observable.Generic(distance)
        for obs in self._task_observables.values():
            obs.enabled = True
        self._gripper.observables.enable_all()
        self._prop.observables.enable_all()

        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*self.workspace.prop_box),
            ignore_collisions=False,
            settle_physics=True
        )

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._physics_variation.apply_variations(physics, random_state)

    def get_reward(self, physics):
        return 1.

