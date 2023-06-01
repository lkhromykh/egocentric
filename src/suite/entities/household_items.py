import os
import zipfile
from string import Template

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable


_template = Template("""
<mujoco model="item">
    <asset>
        <mesh name="mesh" file="$obj" scale="$scale"/>
        <texture name="texture" file="$texture" type="2d"/>
        <material name="mat" texture="texture"/>
    </asset>

    <worldbody>
        <body name="item">
            <geom name="item" mesh="mesh" type="mesh" material="mat"
                  friction="3 0.1 0.002"
            />
        </body>
    </worldbody>
</mujoco>
""")

_BASE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../../third_party/scanned_objects_google/'
))


class HouseholdItem(composer.Entity):
    DATA_DIR = _BASE_PATH
    CACHE_DIR = os.path.join(_BASE_PATH, 'cache/')
    MODEL = 'meshes/model.obj'
    TEXTURE = 'materials/textures/texture.png'

    def _build(self, name, scale='.5 .5 .5'):
        name, _ = os.path.splitext(name)
        self._item_name = name
        self._extract(name)
        model_dir = os.path.join(_BASE_PATH, HouseholdItem.CACHE_DIR, name)
        model = _template.substitute(
            obj=HouseholdItem.MODEL,
            texture=HouseholdItem.TEXTURE,
            scale=scale
        )
        self._mjcf_model = mjcf.from_xml_string(model, model_dir=model_dir)
        self.geom = self._mjcf_model.find('geom', 'item')

    def _build_observables(self):
        return ItemObservables(self)

    @staticmethod
    def _extract(name):
        path = os.path.join(HouseholdItem.CACHE_DIR, name)
        if not os.path.exists(path):
            os.makedirs(path)
            zipfile_path = os.path.join(HouseholdItem.DATA_DIR, name)
            with open(zipfile_path + '.zip', 'rb') as file:
                item = zipfile.ZipFile(file)
                item.extract(HouseholdItem.MODEL, path=path)
                item.extract(HouseholdItem.TEXTURE, path=path)

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
