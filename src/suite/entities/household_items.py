import os

import numpy as np
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable

_BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../third_party/mujoco_scanned_objects/models'
))


class HouseholdItem(composer.Entity):
    DATA_DIR = _BASE_PATH

    def _build(self, name, scale='.5 .5 .5'):
        name, _ = os.path.splitext(name)
        self._item_name = name
        path = os.path.join(self.DATA_DIR, name, 'model.xml')
        self._mjcf_model = mjcf.from_path(path)
        for m in self._mjcf_model.find_all('mesh'):
            m.scale = scale
        self.body = self._mjcf_model.find('body', 'model')
        self.geoms = self._mjcf_model.find_all('geom')

    def _build_observables(self):
        return ItemObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    @property
    def item_name(self):
        return self._item_name


class ItemObservables(composer.Observables):

    @composer.observable
    def pos(self):
        return observable.MJCFFeature('xipos', self._entity.body)

    @composer.observable
    def rmat(self):
        return observable.MJCFFeature('ximat', self._entity.body)

    @composer.observable
    def aabb(self):
        def aabb(physics):
            coords = physics.bind(self._entity.geoms).aabb
            center, half_size = np.split(coords, 2, -1)
            high = np.max(center + half_size, 0)
            low = np.min(center - half_size, 0)
            return np.concatenate([low, high])

        return observable.Generic(aabb)
