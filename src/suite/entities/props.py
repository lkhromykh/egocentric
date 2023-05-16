"""Slightly modified dm_control manipulation props."""
# github.com/deepmind/dm_control/blob/main/dm_control/entities/props/primitive.py
import itertools

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable
import numpy as np

from src.suite import common

_DEFAULT_HALF_LENGTHS = [0.05, 0.1, 0.15]
_MAX_TOUCH = 200.


class Primitive(composer.Entity):
    """A primitive MuJoCo geom prop."""

    def _build(self, geom_type, size, mass=None, name=None, **kwargs):
        """Initializes this prop.
        Args:
          geom_type: a string, one of the types supported by MuJoCo.
          size: a list or numpy array of up to 3 numbers, depending on the type.
          mass: The mass for the primitive geom.
          name: (optional) A string, the name of this prop.
        """
        size = np.reshape(np.asarray(size), -1)
        self._mjcf_root = mjcf.element.RootElement(model=name)

        self._geom = self._mjcf_root.worldbody.add(
            'geom', name='body_geom', type=geom_type,
            size=size, mass=mass, **kwargs)

        # this differs from the source.
        self._touch_site = self._mjcf_root.worldbody.add(
            'site', type=geom_type, name='touch_sensor',
            size=self._geom.size * 1.05,
            rgba=[1, 1, 1, 0.1],  # touch sensor site is almost transparent
            group=common.SENSOR_SITES_GROUP)

        self._touch = self._mjcf_root.sensor.add(
            'touch', site=self._touch_site, cutoff=_MAX_TOUCH)

        self._position = self._mjcf_root.sensor.add(
            'framepos', name='position', objtype='geom', objname=self.geom)

        self._orientation = self._mjcf_root.sensor.add(
            'framequat', name='orientation', objtype='geom',
            objname=self.geom)

        self._linear_velocity = self._mjcf_root.sensor.add(
            'framelinvel', name='linear_velocity', objtype='geom',
            objname=self.geom)

        self._angular_velocity = self._mjcf_root.sensor.add(
            'frameangvel', name='angular_velocity', objtype='geom',
            objname=self.geom)

        self._name = name

    def _build_observables(self):
        return PrimitiveObservables(self)

    @property
    def geom(self):
        """Returns the primitive's geom, e.g., to change color or friction."""
        return self._geom

    @property
    def touch_site(self):
        return self._touch_site

    @property
    def touch(self):
        """Exposing the touch sensor for observations and reward."""
        return self._touch

    @property
    def position(self):
        """Ground truth pos sensor."""
        return self._position

    @property
    def orientation(self):
        """Ground truth angular position sensor."""
        return self._orientation

    @property
    def linear_velocity(self):
        """Ground truth velocity sensor."""
        return self._linear_velocity

    @property
    def angular_velocity(self):
        """Ground truth angular velocity sensor."""
        return self._angular_velocity

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def name(self):
        return self._name


class PrimitiveObservables(composer.Observables,
                           composer.FreePropObservableMixin):
    """Primitive entity's observables."""

    @define.observable
    def position(self):
        return observable.MJCFFeature('sensordata', self._entity.position)

    @define.observable
    def orientation(self):
        return observable.MJCFFeature('sensordata', self._entity.orientation)

    @define.observable
    def linear_velocity(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.linear_velocity)

    @define.observable
    def angular_velocity(self):
        return observable.MJCFFeature('sensordata',
                                      self._entity.angular_velocity)

    @define.observable
    def touch(self):
        obs = observable.MJCFFeature('sensordata', self._entity.touch)
        obs.corruptor = lambda touch, random_state: touch / _MAX_TOUCH
        return obs


# Usage of observation.enable is preferred over duplicating Observables.
class StaticPrimitiveObservables(composer.Observables):
    """Primitive w/o velocity sensors output."""

    @define.observable
    def position(self):
        return observable.MJCFFeature('sensordata', self._entity.position)

    @define.observable
    def orientation(self):
        return observable.MJCFFeature('sensordata', self._entity.orientation)

    @define.observable
    def touch(self):
        obs = observable.MJCFFeature('sensordata', self._entity.touch)
        obs.corruptor = lambda touch, random_state: touch / _MAX_TOUCH
        return obs


class Sphere(Primitive):
    """A class representing a sphere prop."""

    def _build(self, radius=0.05, mass=None, name='sphere'):
        super(Sphere, self)._build(
            geom_type='sphere', size=radius, mass=mass, name=name)


class Box(Primitive):
    """A class representing a box prop."""

    def _build(self, half_lengths=None, mass=None, name='box'):
        half_lengths = half_lengths or _DEFAULT_HALF_LENGTHS
        super(Box, self)._build(geom_type='box',
                                size=half_lengths,
                                mass=mass,
                                name=name)


class BoxWithSites(Box):
    """A class representing a box prop with sites on the corners."""

    def _build(self, half_lengths=None, mass=None, name='box'):
        half_lengths = half_lengths or _DEFAULT_HALF_LENGTHS
        super(BoxWithSites, self)._build(half_lengths=half_lengths, mass=mass,
                                         name=name)

        corner_positions = itertools.product(
            [half_lengths[0], -half_lengths[0]],
            [half_lengths[1], -half_lengths[1]],
            [half_lengths[2], -half_lengths[2]])
        corner_sites = []
        for i, corner_pos in enumerate(corner_positions):
            corner_sites.append(
                self._mjcf_root.worldbody.add(
                    'site',
                    type='sphere',
                    name='corner_{}'.format(i),
                    size=[0.1],
                    pos=corner_pos,
                    rgba=[1, 0, 0, 1.0],
                    group=composer.SENSOR_SITES_GROUP))
        self._corner_sites = tuple(corner_sites)

    @property
    def corner_sites(self):
        return self._corner_sites


class Ellipsoid(Primitive):
    """A class representing an ellipsoid prop."""

    def _build(self, radii=None, mass=None, name='ellipsoid'):
        radii = radii or _DEFAULT_HALF_LENGTHS
        super(Ellipsoid, self)._build(geom_type='ellipsoid',
                                      size=radii,
                                      mass=mass,
                                      name=name)


class Cylinder(Primitive):
    """A class representing a cylinder prop."""

    def _build(self, radius=0.05, half_length=0.15, mass=None, name='cylinder'):
        super(Cylinder, self)._build(geom_type='cylinder',
                                     size=[radius, half_length],
                                     mass=mass,
                                     name=name)


class Capsule(Primitive):
    """A class representing a capsule prop."""

    def _build(self, radius=0.05, half_length=0.15, mass=None, name='capsule'):
        super(Capsule, self)._build(geom_type='capsule',
                                    size=[radius, half_length],
                                    mass=mass,
                                    name=name)


class _VertexSitesMixin:
    """It differs from dm_control version in sites treatment:
    existing sites will alternate instead of creating new every time."""

    def add_vertex_sites(self, box_geom_or_site):
        """Add sites corresponding to the vertices of a box geom or site."""
        offsets = (
            (-half_length, half_length)
            for half_length in box_geom_or_site.size
        )
        site_positions = np.vstack(tuple(itertools.product(*offsets)))
        if box_geom_or_site.pos is not None:
            site_positions += box_geom_or_site.pos
        self._vertices = []
        for i, pos in enumerate(site_positions):
            name = 'vertex_' + str(i)
            site = box_geom_or_site.parent.find('site', name)
            if site is None:
                site = box_geom_or_site.parent.add(
                    'site', name=name,
                    pos=pos, type='sphere', size=[0.002],
                    rgba=common.RED, group=common.TASK_SITE_GROUP)
            else:
                site.pos = pos
            self._vertices.append(site)

    @property
    def vertices(self):
        return self._vertices


class BoxWithVertexSites(Box, _VertexSitesMixin):
    """Subclass of `Box` with sites marking the vertices of the box geom."""

    def initialize_episode_mjcf(self, random_state):
        self.touch_site.size = 1.05 * self.geom.size
        self.add_vertex_sites(self.geom)
