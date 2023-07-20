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
        self._groundplane_texture = self._mjcf_root.asset.add(
            'texture',
            name='groundplane',
            type='2d',
            builtin='gradient',
            rgb1=(.4, .4, .4),
            rgb2=(.5, .5, .5),
            mark='random',
            markrgb='1 1 1',
            random=0.,
            width=300,
            height=300
        )

        self._groundplane_material = self._mjcf_root.asset.add(
            'material',
            name='groundplane',
            texture=self._groundplane_texture,
            texrepeat='1 1',
            specular=.7,
            shininess=.1,
            reflectance=0,
        )
        self._groundplane = self._mjcf_root.worldbody.add(
            'geom',
            name='groundplane',
            type='plane',
            material=self._groundplane_material,
            size=(4, 1.5, 0.01),
            friction=(.4,),
            solimp=(.95, .99, .001),
            solref=(.005, 1.)
        )
        self._room_light = self._mjcf_root.worldbody.add(
            'light',
            name='room',
            pos=(0, 0, 2.),
            dir=(0, 0, -1),
            diffuse=(.6, .6, .6),
            specular=(.3, .3, .3),
            ambient=(.4, .4, .4),
            directional=True,
            castshadow=False
        )
        self._hall_light = self._mjcf_root.worldbody.add(
            'light',
            name='hall',
            pos=(0, -2., 1.),
            dir=(0, 1, 0),
            diffuse=(.2, .2, .2),
            specular=(.2, .2, .2),
            ambient=(.1, .1, .1),
            directional=True,
            castshadow=False
        )

    def add_free_entity(self, entity, offset=None):
        frame = self.attach(entity)
        frame.add('freejoint')
        if offset is not None:
            frame.pos = offset
        return frame

    def attach_to_mocap(self, body, relpose=None):
        return self._mjcf_root.equality.add(
            'weld',
            name='mocap_weld',
            body1=self._mocap,
            body2=body,
            relpose=relpose,
            solref='0.01 1'
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
    def groundplane_texture(self):
        return self._groundplane_texture

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
