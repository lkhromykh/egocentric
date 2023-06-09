import numpy as np
import numpy.typing as npt

from dm_control.manipulation.shared.constants import (
    RED,
    GREEN,
    BLUE,
    TASK_SITE_GROUP
)
from dm_control.composer.constants import SENSOR_SITES_GROUP

Array = npt.NDArray[np.float32]
RNG = np.random.Generator

DOWN_QUATERNION = (0., 0.70710678118, 0.70710678118, 0.)
CTRL_LIMIT = .03
ROT_LIMIT = np.pi / 6
CONTROL_TIMESTEP = .1
BOX_MASS = .1
BOX_SIZE = (.04, .03, .015)
MOCAP_SITE_GROUP = 5
