from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.composer.environment import EpisodeInitializationError
from dm_control.utils import rewards

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
            mass=common.BOX_MASS,
        )
        self._prop.geom.rgba = (1., 0, 0, 1.)
        self._arena.add_free_entity(self._prop)

        def distance(physics):
            tcp_pos = physics.bind(self._gripper.tool_center_point).xpos
            obj_pos, _ = self._prop.get_pose(physics)
            return obj_pos - tcp_pos

        self._task_observables['box/distance'] = observable.Generic(distance)
        # self._task_observables['box/distance'].enabled = True
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
        try:
            super().initialize_episode(physics, random_state)
            self._prop_placer(physics, random_state)
        except Exception as exp:
            raise EpisodeInitializationError(exp) from exp

    def get_reward(self, physics):
        lowest = physics.bind(self._prop.vertices).xpos[:, 2].min()
        return rewards.tolerance(
            lowest,
            bounds=(.1, .3),
            margin=.1,
            value_at_margin=0.,
            sigmoid='linear'
        )


