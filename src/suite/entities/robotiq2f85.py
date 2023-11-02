import os

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable

from src.suite import common


_ROBOTIQ2F85_XML_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../third_party/mujoco_menagerie/robotiq_2f85/2f85.xml'
)


class Robotiq2f85(composer.Entity):
    """Robotiq 2F85 gripper."""

    def _build(self):
        self._mjcf_model = mjcf.from_path(_ROBOTIQ2F85_XML_PATH)
        self._actuators = self.mjcf_model.find_all('actuator')
        self._base = self.mjcf_model.find('body', 'base_mount')
        self._tcp_site = self._base.add(
            'site',
            name='tcp_center_point',
            pos=[0, 0, .1269],
            group=common.MOCAP_SITE_GROUP
        )
        black = self._mjcf_model.find('material', 'black')
        for geom in self._mjcf_model.find_all('geom'):
            geom.material = black

    def _build_observables(self):
        return RobotiqObservables(self)

    def set_grasp(self, physics, close_factor):
        """[0., 1.] -> uint8"""
        ctrl = int(255 * close_factor)
        physics.set_control(ctrl)
        acc = physics.bind(self._actuators)
        physics.step()
        while abs(acc.velocity[0]) > 5e-3:
            physics.step()

    @property
    def tool_center_point(self):
        return self._tcp_site

    @property
    def base_mount(self):
        return self._base

    @property
    def actuators(self):
        return self._actuators

    @property
    def mjcf_model(self):
        return self._mjcf_model


class RobotiqObservables(composer.Observables):

    @composer.observable
    def tcp_pos(self):
        return observable.MJCFFeature('xpos', self._entity.tool_center_point)

    @composer.observable
    def tcp_quat(self):
        return observable.MJCFFeature('xquat', self._entity.base_mount)

    @composer.observable
    def length(self):
        def normed(len_, random_state): return len_ / 0.7980633
        return observable.MJCFFeature('length', self._entity.actuators,
                                      corruptor=normed)

    @composer.observable
    def force(self):
        return observable.MJCFFeature('force', self._entity.actuators)

    @composer.observable
    def object_detected(self):
        _, max_force = self._entity.actuators[0].forcerange
        def detector(v, random_state): return (v > .8 * max_force).astype(float)
        return observable.MJCFFeature('force', self._entity.actuators,
                                      corruptor=detector)
