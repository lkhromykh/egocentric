import os

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer.environment import EpisodeInitializationError

_BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../third_party/mujoco_scanned_objects/models'
))

_VOLERROR = (
    'Seagate_1TB_Backup_Plus_portable_drive_for_Mac',
)


class HouseholdItem(composer.Entity):
    DATA_DIR = _BASE_PATH

    def _build(self, name, scale='.5 .5 .5'):
        if name in _VOLERROR:
            # TODO: fix mesh volume error on scale < 1..
            raise EpisodeInitializationError(name + ' mesh volume error.')
        name, _ = os.path.splitext(name)
        self._item_name = name
        path = os.path.join(self.DATA_DIR, name, 'model.xml')
        self._mjcf_model = mjcf.from_path(path)
        for m in self._mjcf_model.find_all('mesh'):
            m.scale = scale
        geom = filter(
            lambda g: g.conaffinity == 0,
            self._mjcf_model.find_all('geom')
        )
        self.geom = next(geom)

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
        return observable.MJCFFeature('xpos', self._entity.geom)
