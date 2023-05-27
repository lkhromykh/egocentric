from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.environment import EpisodeInitializationError
from dm_control.utils import rewards
from dm_control.composer.variation import distributions
from dm_control.composer.variation.colors import RgbVariation
from dm_control.manipulation.shared import workspaces

from src.suite import entities
from src.suite import common
from src.suite.tasks import base


class Box(entities.BoxWithVertexSites):

    def _build_observables(self):
        return entities.StaticPrimitiveObservables(self)


class PickAndLift(base.Task):

    MARGIN: float = .15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prop = Box(
            half_lengths=common.BOX_SIZE,
            mass=common.BOX_MASS,
        )
        self._prop.geom.rgba = (1., 0, 0, 1.)
        self._arena.add_free_entity(self._prop)
        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(*self.workspace.prop_box),
            ignore_collisions=False,
            settle_physics=True
        )
        lower = self.workspace.tcp_box.lower.copy()
        upper = self.workspace.tcp_box.upper.copy()
        lower[2] = self.MARGIN
        upper[2] = self.MARGIN + 1e-3
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=lower,
            upper=upper,
            rgba=common.BLUE,
            name='target_height'
        )
        self._build_variations()
        self._build_observables()

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
            bounds=(self.MARGIN, float('inf')),
            margin=self.MARGIN,
            value_at_margin=0.,
            sigmoid='linear'
        )

    # def should_terminate_episode(self, physics):
    #     reward = self.get_reward(physics)
    #     return reward == 1.

    def _build_variations(self):
        super()._build_variations()

        def axis_var(dist, idx):
            def noise_fn(init, cur, rng):
                noise = dist(init, cur, rng)
                return noise[idx]
            return noise_fn
        uni = distributions.Uniform
        rgba = [uni(.5, 1.), uni(0, .5), uni(0, .5)]
        rgba = [axis_var(dist, i) for i, dist in enumerate(rgba)]
        rgba = RgbVariation(*rgba)
        self._mjcf_variation.bind_attributes(
            self._prop.geom,
            rgba=rgba
        )

    def _build_observables(self):
        super()._build_observables()

        def distance(physics):
            tcp_pos = physics.bind(self._gripper.tool_center_point).xpos
            obj_pos, _ = self._prop.get_pose(physics)
            return obj_pos - tcp_pos
        self._task_observables['box/distance'] = observable.Generic(distance)
        for obs in self._task_observables.values():
            obs.enabled = True
        self._gripper.observables.enable_all()
        self._prop.observables.enable_all()

