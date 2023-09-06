from enum import IntEnum
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from dm_env import specs
import matplotlib.pyplot as plt
from PIL import Image

from ur_env.environment import Task
from ur_env.scene import Scene, nodes
from ur_env.remote import RemoteEnvServer
from ur_env.environment import Environment


class DiscreteActions(IntEnum):

    FORWARD = 0
    BACKWARD = 1
    RIGHT = 2
    LEFT = 3
    UP = 4
    DOWN = 5
    CLOSE = 6
    OPEN = 7
    ROLL_CW = 8
    ROLL_CCW = 9

    @staticmethod
    def as_array(action: int, dtype=np.float32):
        idx, val = np.divmod(action, 2)
        ar = np.zeros(len(DiscreteActions) // 2, dtype=dtype)
        ar[idx] = -1 if val else 1
        return ar


class PickAndLift(Task):
    """Grasp an object and lift it."""

    CTRL_LIMIT = .03
    ROT_LIMIT = np.pi / 6
    IMG_SHAPE = (96, 96)
    IMG_KEY = "realsense/image"
    OBJ_KEY = "robotiq_2f85/object_detected"
    POSE_KEY = "tcp_pose"
    GRIPPER_POS = "robotiq_2f85/length"

    def __init__(self,
                 threshold: float,
                 init_q: list[float],
                 ):
        """
        Args:
            threshold: reward will not increase after this height.
            init_q: initial joints position.
        """
        self._init_q = list(init_q)
        self._threshold = threshold
        # Initialize on reset.
        self._grasped: float = None
        self._init_pos: float = None
        self._num_rot: int = None

    def initialize_episode(self, scene, random_state):
        super().initialize_episode(scene, random_state)
        scene.arm.rtde_control.moveJ(self._init_q)
        scene.gripper.move(scene.gripper.min_position)
        self._grasped = -1.
        self._init_pos = np.asarray(scene.arm.rtde_receive.getActualTCPPose())[:3]
        self._num_rot = 0

    def get_observation(self, scene):
        obs = scene.get_observation()
        img = self._img_fn(obs['realsense/image'])
        pose = self._pose_fn(obs['arm/ActualTCPPose'])
        return {
            self.IMG_KEY: img,
            self.POSE_KEY: pose,
            self.OBJ_KEY: obs['gripper/object_detected'],
            self.GRIPPER_POS: obs['gripper/pos'],
        }

    def observation_spec(self, scene):
        spec = scene.observation_spec()
        img_spec = spec['realsense/image']
        img_spec = img_spec.replace(shape=self.IMG_SHAPE + (3,))
        pos_spec = spec['arm/ActualTCPPose'].replace(shape=(6,))
        return {
            self.IMG_KEY: img_spec,
            self.POSE_KEY: pos_spec,
            self.OBJ_KEY: spec['gripper/object_detected'],
            self.GRIPPER_POS: spec['gripper/pos'],
        }

    def get_reward(self, scene) -> float:
        is_picked = scene.gripper.object_detected
        pose = scene.arm.rtde_receive.getActualTCPPose()
        height = pose[2]
        success = is_picked * (height > self._threshold)
        return float(success)

    def _img_fn(self, img: np.ndarray) -> np.ndarray:
        img = img[:, 144:-224]
        h, w, c = img.shape
        assert (h == w) * (c == 3)
        img = Image.fromarray(img)
        img = img.resize(self.IMG_SHAPE, resample=Image.Resampling.LANCZOS)
        return np.asarray(img)

    def _pose_fn(self, pose):
        # TODO: make sure that pose correspond or transform via rtde
        xyz, axang = np.split(pose, 2)
        xyz -= np.array([-0.4922, 0.0381, -0.01]) # achieve 0.03 at the lowest point
        scale = 1 - 2 * np.pi / np.linalg.norm(axang)
        axang *= scale
        return np.concatenate([xyz, axang])

    def action_spec(self, scene):
        return specs.DiscreteArray(len(DiscreteActions), dtype=np.int32)

    def before_step(self, scene, action, random_state):
        pos = np.asarray(scene.arm.rtde_receive.getActualTCPPose())
        action = DiscreteActions.as_array(action)
        arm, grasp, rot = np.split(action, [3, 4])
        arm *= self.CTRL_LIMIT
        if np.linalg.norm(pos[:3] + arm - self._init_pos) > .2:
            arm = np.zeros_like(arm)
        if grasp:
            self._grasped = grasp
        if rot.size:
            rot = 0 if np.abs(rot + self._num_rot) > 8 else rot
            self._num_rot += rot
            rot = rot * np.array([0, 0, self.ROT_LIMIT])
        else:
            rot = np.zeros((3,))
        scene.step({
            'arm': np.concatenate([arm, rot]),
            'gripper': self._grasped
        })

HOST = "10.201.2.179"
arm = nodes.TCPPose(
        host=HOST,
        port=50002,
        frequency=300,
        speed=.2,
        absolute_mode=False
    )
gripper = nodes.DiscreteGripper(HOST, force=10)
realsense = nodes.RealSense()
scene_ = Scene(arm=arm, gripper=gripper, realsense=realsense)

INIT_Q = [-0.350, -1.452, 2.046, -2.167, 4.712, 0.03]
task = PickAndLift(
    threshold=.1,
    init_q=INIT_Q
)
env = Environment(
    random_state=0,
    scene=scene_,
    task=task,
    time_limit=30,#20
    max_violations_num=2
)
breakpoint()
address = ('', 5555)
env = RemoteEnvServer(env, address)
env.run()
#TODO: apply shift, reorintate axes, compare RPY, maybe use Kinect, correct TCP offset
# First [0.02224612, 0.06145405, 0.20606055, 2.22144147, 2.22144147, 0.]
# Step forward [5.07222968e-02, 6.14540523e-02, 2.12635864e-01,
#               2.22140963e+00, 2.22140972e+00, 3.53468484e-05]
# Clockwise rotation (8): [ 5.87790820e-02,  6.69913161e-02,  2.57667821e-01,
#                           2.69739921e+00, 1.61047878e+00, -7.54914283e-09])