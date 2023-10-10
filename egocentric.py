from enum import IntEnum
import random
from itertools import product
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from scipy.spatial.transform import Rotation
from dm_env import specs
import matplotlib.pyplot as plt
from PIL import Image

from ur_env.environment import Task
from ur_env.scene import Scene, nodes
from ur_env.remote import RemoteEnvServer
from ur_env.environment import Environment


def task_sampler():
    orientations = (
        'N', 'NE', 'E', 'SE',
        'S', 'SW', 'W', 'NW'
    )
    toys = (
        'red santa',
        'grey walrus',
        'orange sponge',
        'black brick',
        'green ball',
        'blue ball',
    )
    variations = list(product(orientations, toys))
    while True:
        yield random.choice(variations)


class DiscreteActions(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    DOWN = 4
    UP = 5
    CLOSE = 6
    OPEN = 7
    ROLL_CW = 8
    ROLL_CCW = 9

    @staticmethod
    def as_array(action: int, dtype=np.float32) -> np.ndarray:
        idx, val = np.divmod(action, 2)
        ar = np.zeros(len(DiscreteActions) // 2, dtype=dtype)
        ar[idx] = -1 if val else 1
        return ar

    @staticmethod
    def sim2real_spoofing(action: int):
        """Sim2Real defects. Identity mapping in case of exact """
        mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
        }
        action = mapping.get(action)
        return DiscreteActions.as_array(action)


class PickAndLift(Task):
    """Grasp an object and lift it."""

    CTRL_LIMIT = .03
    ROT_LIMIT = np.pi / 6
    IMG_SHAPE = (64, 64)
    IMG_KEY = "realsense/image"
    HEIGHT_KEY = "tcp_height"
    OBJ_KEY = "robotiq_2f85/object_detected"
    GRIPPER_POS = "robotiq_2f85/length"

    def __init__(self,
                 threshold: float,
                 init_q
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
        self._task_gen = task_sampler()

    def initialize_episode(self, scene, random_state):
        scene.arm.rtde_control.moveJ(self._init_q)
        scene.gripper.move(scene.gripper.min_position)
        print('Manually setup scene:', next(self._task_gen),
              '\nPress any key to continue.')
        input()
        super().initialize_episode(scene, random_state)
        self._grasped = -1.
        self._init_pos = np.asarray(scene.arm.rtde_receive.getActualTCPPose())[:3]
        self._num_rot = 0

    def get_observation(self, scene):
        obs = scene.get_observation()
        img = self._img_fn(obs['realsense/image'])
        height = obs['arm/ActualTCPPose'][2:]
        return {
            self.IMG_KEY: img,
            self.HEIGHT_KEY: height,
            self.OBJ_KEY: obs['gripper/object_detected'],
            self.GRIPPER_POS: obs['gripper/pos'],
        }

    def observation_spec(self, scene):
        spec = scene.observation_spec()
        img_spec = spec['realsense/image']
        img_spec = img_spec.replace(shape=self.IMG_SHAPE + (3,))
        height_spec = spec['arm/ActualTCPPose']
        height_spec = height_spec.replace(shape=(1,))
        return {
            self.IMG_KEY: img_spec,
            self.HEIGHT_KEY: height_spec,
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
        assert h == w and c == 3
        img = Image.fromarray(img)
        img = img.resize(self.IMG_SHAPE, resample=Image.Resampling.LANCZOS)
        return np.asarray(img)

    def action_spec(self, scene):
        return specs.DiscreteArray(len(DiscreteActions), dtype=np.int32)

    def before_step(self, scene, action, random_state):
        pos = scene.arm.rtde_receive.getActualTCPPose()
        import pdb; pdb.set_trace()
        xyz, rotvec = np.split(pos, 2)
        rmat = Rotation.from_rotvec(rotvec).as_matrix()
        action = DiscreteActions.sim2real_spoofing(action)
        arm, grasp, rot = np.split(action, [3, 4])
        arm = self.CTRL_LIMIT * rmat @ arm
        if np.linalg.norm(xyz + arm - self._init_pos) > .2:
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

HOST = None
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
    time_limit=16,
    max_violations_num=2
)
#breakpoint()
address = ('', 5555)
env = RemoteEnvServer(env, address)
env.run()
#TODO: apply shift, reorintate axes, compare RPY, maybe use Kinect, correct TCP offset
# First [0.02224612, 0.06145405, 0.20606055, 2.22144147, 2.22144147, 0.]
# Step forward [5.07222968e-02, 6.14540523e-02, 2.12635864e-01,
#               2.22140963e+00, 2.22140972e+00, 3.53468484e-05]
# Clockwise rotation (8): [ 5.87790820e-02,  6.69913161e-02,  2.57667821e-01,
#                           2.69739921e+00, 1.61047878e+00, -7.54914283e-09])
