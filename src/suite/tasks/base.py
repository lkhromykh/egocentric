import abc
from enum import IntEnum
from typing import NamedTuple, Literal

import numpy as np
from dm_env import specs

from dm_control.composer import variation
from dm_control.composer.task import Task as _Task
from dm_control.manipulation.shared import workspaces
from dm_control.composer.variation import noises, distributions
from dm_control.utils import transformations

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
    CLOSE = 6
    OPEN = 7
    # ROLL_CW = 8
    # ROLL_CCW = 9

    @staticmethod
    def as_array(action: int, dtype=np.float32) -> common.Array:
        idx, val = np.divmod(action, 2)
        ar = np.zeros((4,), dtype=dtype)
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
                       tcp_height: tuple[float, float] = (.17, .4)
                       ) -> 'WorkSpace':
        x, y = half_sizes
        low, high = tcp_height

        def box_fn(l, h):
            return BoundingBox(
                lower=np.float32([-x, -y, l]),
                upper=np.float32([x, y, h])
            )
        return cls(box_fn(0.1, 0.1), box_fn(low, high))


_DEFAULT_WORKSPACE = WorkSpace.from_halfsizes()


class Task(abc.ABC, _Task):

    def __init__(self,
                 control_timestep: float = common.CONTROL_TIMESTEP,
                 action_mode: ActionMode = 'discrete',
                 workspace: WorkSpace = _DEFAULT_WORKSPACE,
                 img_size: tuple[int, int] = (128, 128)
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
        self._arena.add_free_entity(self._gripper, offset)
        relpose = np.array([0, 0, 0, 1., 0, 0, 0])
        self._weld = self._arena.attach_to_mocap(
            self._gripper.base_mount, relpose)

        self._physics_variation = variation.PhysicsVariator()
        self._mjcf_variation = variation.MJCFVariator()
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.tcp_box.lower,
            upper=workspace.tcp_box.upper,
            rgba=common.GREEN,
            name='mocap_bbox'
        )

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variation.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variation.apply_variations(physics, random_state)
        pos = self.workspace.tcp_box.sample(random_state)
        self._set_mocap(physics, pos, common.DOWN_QUATERNION)
        self._gripper.set_pose(physics, pos, common.DOWN_QUATERNION)

    def before_step(self, physics, action, random_state):
        if self.action_mod == 'discrete':
            action = DiscreteActions.as_array(action)
        else:
            action = np.clip(action, -1, 1)
        pos, grip, rot = map(np.squeeze, np.split(action, [3, 4]))
        mocap_pos, mocap_quat = self._get_mocap(physics)
        # rot = common.ROT_LIMIT * np.array([0, 0, rot])
        # rot = transformations.euler_to_quat(rot)
        self._set_mocap(physics,
                        pos=mocap_pos + common.CTRL_LIMIT * pos,
                        # quat=transformations.quat_mul(mocap_quat, rot)
                        )
        if grip:
            self._gripper.set_grasp(physics, float(grip > 0.))

    def action_spec(self, physics):
        num_values = len(DiscreteActions)
        match self.action_mod:
            case 'discrete':
                return specs.DiscreteArray(num_values, dtype=np.int32)
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

    def _build_variations(self):
        """Domain randomization goes here."""
        self._mjcf_variation.clear()
        uni = distributions.Uniform

        def eq_noise(low, high):
            def noise_fn(init, cur, rng):
                noise = rng.uniform(low, high)
                return np.full_like(init, noise)
            return noise_fn

        for light in self.root_entity.mjcf_model.worldbody.find_all('light'):
            self._mjcf_variation.bind_attributes(
                light,
                pos=noises.Additive(uni(-.4, .4)),
                diffuse=eq_noise(.1, .7),
                specular=eq_noise(.05, .3),
                ambient=eq_noise(.05, .4)
            )

        def rgb(init, cur, random_state):
            noise = random_state.uniform(0., 1., len(init) - 1)
            return np.concatenate([noise, [1.]])
        self._mjcf_variation.bind_attributes(
            self._arena.groundplane_material,
            rgba=rgb
        )

        def axis_var(dist, idx):
            def noise_fn(init, cur, rng):
                noise = np.zeros_like(init)
                noise[idx] = dist(init[idx], cur[idx], rng)
                return init + noise
            return noise_fn
        self._mjcf_variation.bind_attributes(
            self._camera,
            pos=axis_var(uni(-.01, .01), 1),
            quat=axis_var(uni(-0.04, 0.04), 3),
            fovy=noises.Additive(uni(-8, 8))
        )

    def _build_observables(self):
        """Enable required observables."""
        def noisy_cam(img, random_state):
            noise = random_state.randint(-20, 20, img.shape)
            img = np.clip(img + noise, 0, 255).astype(img.dtype)
            return img
        self._task_observables['realsense/image'].corruptor = noisy_cam
