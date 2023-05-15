import os

from dm_control import mjcf
from dm_control.composer import Entity

_BASE_XML_PATH = os.path.join(os.path.dirname(__file__), 'base.xml')


class Arena(Entity):

    def _build(self):
        self._mjcf_root = mjcf.from_path(_BASE_XML_PATH)

        self._mocap = self._mjcf_root.find('body', 'mocap')

        self._skybox = self._mjcf_root.asset.add(
            'texture',
            name='skybox',
            type='skybox',
            builtin='gradient',
            rgb1=(.4, .6, .8),
            rgb2=(0, 0, 0),
            width=800,
            height=800,
            mark='random',
            markrgb='1 1 1',
            random=.01
        )

        self._groundplane_material = self._mjcf_root.asset.add(
            'material',
            name='groundplane',
            specular=.7,
            shininess=.1,
            reflectance=.01,
            rgba=(.4, .4, .4, 1.)
        )
        self._groundplane = self._mjcf_root.worldbody.add(
            'geom',
            name='ground',
            type='plane',
            material=self._groundplane_material,
            size=(4, 1.5, 0.01),
            friction=(.4,),
            solimp=(.95, .99, .001),
            solref=(.002, 1.)
        )
        self._room_light = self._mjcf_root.worldbody.add(
            'light',
            name='room',
            pos=(0, 0, 2.),
            dir=(0, 0, -1),
            diffuse=(.6, .6, .6),
            specular=(.3, .3, .3),
            ambient=(.4, .4, .4),
            directional='true',
            castshadow='false'
        )
        self._hall_light = self._mjcf_root.worldbody.add(
            'light',
            name='hall',
            pos=(0, -2., 1.),
            dir=(0, 1, 0),
            diffuse=(.2, .2, .2),
            specular=(.2, .2, .2),
            ambient=(.1, .1, .1),
            directional='true',
            castshadow='false'
        )

    def add_free_entity(self, entity):
        frame = self.attach(entity)
        frame.add('freejoint')
        return frame

    def attach_to_mocap(self, body):
        return self._mjcf_root.equality.add(
            'weld',
            name='mocap_weld',
            body1=self._mocap,
            body2=body,
            solref='0.02 1'
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def mocap(self):
        return self._mocap

    @property
    def skybox(self):
        return self._skybox

    @property
    def groundplane_material(self):
        return self._groundplane_material

    @property
    def groundplane(self):
        return self._groundplane

    @property
    def room_light(self):
        return self._room_light

    @property
    def hall_light(self):
        return self._hall_light
