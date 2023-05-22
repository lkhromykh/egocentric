import abc
from typing import NamedTuple, Literal
from enum import IntEnum

import numpy as np
from dm_env import specs

from dm_control.composer.task import Task as _Task
from dm_control.manipulation.shared import workspaces
from dm_control.composer import variation

from src.suite import common
from src.suite import entities

ActionMode = Literal['discrete', 'continuous']


class DiscreteActions(IntEnum):

    FORWARD = 0
    BACKWARD = 1
    RIGHT = 2
    LEFT = 3
    UP = 4
    DOWN = 5
    OPEN = 6
    CLOSE = 7

    @staticmethod
    def as_array(action: int) -> common.Array:
        idx, val = np.divmod(action, 2)
        ar = np.zeros((4,), dtype=np.float32)
        ar[idx] = -1 if val else 1
        return ar


class BoundingBox(NamedTuple):

    lower: common.Array
    upper: common.Array

    def sample(self, rng: common.RNG) -> common.Array:
        return rng.uniform(low=self.lower, high=self.upper)

    @property
    def center(self):
        return (self.upper + self.lower) / 2


class WorkSpace(NamedTuple):

    prop_box: BoundingBox
    tcp_box: BoundingBox

    @classmethod
    def from_halfsizes(cls,
                       half_sizes: tuple[float, float] = (.1, .1),
                       tcp_height: tuple[float, float] = (.1, .3)
                       ) -> 'WorkSpace':
        x, y = half_sizes
        low, high = tcp_height

        def box_fn(l, h):
            return BoundingBox(
                lower=np.float32([-x, -y, l]),
                upper=np.float32([x, y, h])
            )
        return cls(box_fn(0.05, 0.05), box_fn(low, high))


class Task(abc.ABC, _Task):

    def __init__(self,
                 control_timestep: float = common.CONTROL_TIMESTEP,
                 action_mode: ActionMode = 'discrete',
                 workspace: WorkSpace = WorkSpace.from_halfsizes(),
                 img_size: tuple[int, int] = (84, 84)
                 ) -> None:
        self._control_timestep = control_timestep
        self.action_mod = action_mode.lower()
        self.workspace = workspace
        self.img_size = h, w = img_size

        self._arena = entities.Arena()
        self._gripper = entities.Robotiq2f85()
        self._camera, self._task_observables = entities.add_camera_observables(
            self._gripper.base_mount,
            entities.EGOCENTRIC_REALSENSE,
            height=h, width=w
        )
        offset = self.workspace.tcp_box.center
        self._arena.mocap.pos = offset
        self._arena.add_free_entity(self._gripper, offset)
        self._weld = self._arena.attach_to_mocap(self._gripper.base_mount)

        # self._physics_variation = variation.PhysicsVariator() task randomization
        # self._mjcf_variation = variation.MJCFVariator()  domain randomization
        rb = self.root_entity.mjcf_model.worldbody
        workspaces.add_bbox_site(
            body=rb,
            lower=workspace.prop_box.lower,
            upper=workspace.prop_box.upper,
            rgba=common.BLUE,
            name='prop_spawn'
        )
        workspaces.add_bbox_site(
            body=rb,
            lower=workspace.tcp_box.lower,
            upper=workspace.tcp_box.upper,
            rgba=common.GREEN,
            name='mocap_bbox'
        )

    def initialize_episode(self, physics, random_state):
        pos = self.workspace.tcp_box.center
        quat = common.DOWN_QUATERNION
        self._set_mocap(physics, pos, quat)

    def before_step(self, physics, action, random_state):
        if self.action_mod == 'discrete':
            action = DiscreteActions.as_array(action)
        else:
            action = np.clip(action, -1, 1)
        pos, grip = np.split(action, [3])
        mocap_pos, _ = self._get_mocap(physics)
        self._set_mocap(physics, mocap_pos + common.CTRL_LIMIT * pos)
        self._gripper.set_grasp(physics, float(grip > 0.))

    def action_spec(self, physics):
        num_values = len(DiscreteActions)
        match self.action_mod:
            case 'discrete':
                return specs.DiscreteArray(num_values)
            case 'continuous':
                lim = np.full((num_values // 2,), 1, dtype=np.float32)
                return specs.BoundedArray(
                    shape=lim.shape,
                    dtype=lim.dtype,
                    minimum=-lim,
                    maximum=lim
                )
            case _:
                raise RuntimeError(self.action_mod)

    def _get_mocap(self, physics):
        mocap = physics.bind(self._arena.mocap)
        return mocap.mocap_pos, mocap.mocap_quat

    def _set_mocap(self, physics, pos, quat=None):
        mocap = physics.bind(self._arena.mocap)
        bbox = self.workspace.tcp_box
        pos = np.clip(pos, bbox.lower, bbox.upper)
        mocap.mocap_pos = pos
        if quat is not None:
            mocap.mocap_quat = quat

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables
