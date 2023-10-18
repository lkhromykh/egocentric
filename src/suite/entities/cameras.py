from typing import TypedDict
import collections

from dm_control import mjcf
from dm_control.composer.observation import observable


class CameraSpec(TypedDict, total=False):
    name: str
    pos: tuple[float, float, float]
    fovy: int
    xyaxes: tuple[float, ...]
    quat: tuple[float, ...]


KINECT = CameraSpec(
    name='kinect',
    pos=(-1.1, .3, .6),
    xyaxes=(-.2, -.9, 0, .3, 0, .8),
    fovy=70
)

# https://www.intelrealsense.com/depth-camera-d455/
EGOCENTRIC_REALSENSE = CameraSpec(
    name='realsense',
    pos=(-0.0115, .091, .025),
    quat=(0., 0., 0.99876, .049814),
    fovy=58
)


def add_camera_observables(body: mjcf.Element,
                           camera_specs: CameraSpec,
                           **kwargs
                           ):
    obs_dict = collections.OrderedDict()
    n = camera_specs['name']
    camera = body.add('camera', **camera_specs)
    obs_dict[f'{n}/image'] = observable.MJCFCamera(camera, **kwargs)
    # obs_dict[f'{n}/depth'] = observable.MJCFCamera(
    #     camera, depth=True, **kwargs)
    # obs_dict[f'{n}/segmentation'] = observable.MJCFCamera(
    #     camera, segmentation=True, **kwargs)
    return camera, obs_dict
