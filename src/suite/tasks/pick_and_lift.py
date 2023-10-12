import os

from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.environment import EpisodeInitializationError
from dm_control.utils import rewards
from dm_control.composer.variation import distributions
from dm_control.manipulation.shared import workspaces

from src.suite import entities
from src.suite import common
from src.suite.tasks import base

with open(os.path.join(os.path.dirname(__file__), 'banlist')) as f:
    _banlist = f.read().splitlines()
_ITEMS = os.listdir(entities.HouseholdItem.DATA_DIR)
_ITEMS = list(set(_ITEMS) - set(_banlist))
BOXES = [
    'Nestle_Skinny_Cow_Dreamy_Clusters_Candy_Dark_Chocolate_6_pack_1_oz_pouches',
    'Office_Depot_Canon_PG21XL_Remanufactured_Ink_Cartridge_Black',
    'U_By_Kotex_Sleek_Regular_Unscented_Tampons_36_Ct_Box',
    'Epson_Ink_Cartridge_126_Yellow',
    'Office_Depot_HP_61Tricolor_Ink_Cartridge',
    'Sonicare_2_Series_Toothbrush_Plaque_Control',
    'Tune_Belt_Sport_Armband_For_Samsung_Galaxy_S3',
    'Marc_Anthony_True_Professional_Oil_of_Morocco_Argan_Oil_Treatment',
    'Hasbro_Life_Board_Game',
    'Winning_Moves_1180_Aggravation_Board_Game',
    'Crayola_Washable_Sidewalk_Chalk_16_pack',
    'Dell_Ink_Cartridge_Yellow_31',
    'Perricone_MD_Chia_Serum',
    'Asus_Z97IPLUS_Motherboard_Mini_ITX_LGA1150_Socket'
]
CYLINDERS = [
    'Quercetin_500',
    'Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion',
    'Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister',
    '5_HTP',
    'Big_Dot_Aqua_Pencil_Case',
    'Weston_No_33_Signature_Sausage_Tonic_12_fl_oz',
    'Theanine',
    'Mastic_Gum',
    'Whey_Protein_Vanilla',
    'Cole_Hardware_Antislip_Surfacing_Material_White',
    'Big_Dot_Pink_Pencil_Case',
    'Lactoferrin',
    'Room_Essentials_Mug_White_Yellow',
    'QAbsorb_CoQ10',
    'JarroDophilusFOS_Value_Size',
    'Aroma_Stainless_Steel_Milk_Frother_2_Cup',
    'Threshold_Ramekin_White_Porcelain',
    'Prostate_Optimizer',
    'Ecoforms_Plant_Container_GP16AMOCHA'
]


class PickAndLift(base.Task):

    MARGIN: float = .15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prop = entities.HouseholdItem('Ultra_JarroDophilus')
        self._prop_height = None
        self._arena.add_free_entity(self._prop)
        lower = self.workspace.tcp_box.lower.copy()
        upper = self.workspace.tcp_box.upper.copy()
        lower[2] = self.MARGIN
        upper[2] = self.MARGIN + 1e-3
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=lower,
            upper=upper,
            rgba=common.BLUE,
            name='target_height'
        )
        self._build_variations()
        self._build_observables()

    def initialize_episode_mjcf(self, random_state):
        try:
            super().initialize_episode_mjcf(random_state)
            item = random_state.choice(_ITEMS)
            self._prop.detach()
            self._prop = entities.HouseholdItem(item,
                                                scale='.5 .5 .5',
                                                rgb=None)
            self._prop.observables.enable_all()
            self._arena.add_free_entity(self._prop)
        except Exception as exc:
            raise EpisodeInitializationError from exc

    def initialize_episode(self, physics, random_state):
        try:
            self._gripper.set_pose(physics, self.workspace.tcp_box.upper)
            prop_placer = initializers.PropPlacer(
                props=[self._prop],
                position=distributions.Uniform(*self.workspace.prop_box),
                quaternion=workspaces.uniform_z_rotation,
                ignore_collisions=False,
                settle_physics=True,
            )
            prop_placer(physics, random_state)
            super().initialize_episode(physics, random_state)
            physics.forward()
            self._prop_height = self._prop_com_height(physics)
        except Exception as exc:
            raise EpisodeInitializationError from exc

    def get_reward(self, physics):
        diff = self._prop_com_height(physics) - self._prop_height
        return rewards.tolerance(
            diff,
            bounds=(self.MARGIN, float('inf')),
            margin=self.MARGIN,
            value_at_margin=0.,
            sigmoid='linear'
        )

    def _prop_com_height(self, physics):
        prop = physics.bind(self._prop.body)
        return prop.xipos[2]

    def _build_observables(self):
        super()._build_observables()

        def distance(physics):
            tcp_pos = physics.bind(self._gripper.tool_center_point).xpos
            obj_pos = physics.bind(self._prop.body).xipos
            return obj_pos - tcp_pos
        self._task_observables[f'{self._prop.mjcf_model.model}/distance'] =\
            observable.Generic(distance)

        def height(physics):
            tcp_pos = physics.bind(self._gripper.tool_center_point).xpos
            return tcp_pos[2:]
        self._task_observables['tcp_height'] = observable.Generic(height)

        for obs in self._task_observables.values():
            obs.enabled = True
        self._task_observables['realsense/rgbd'].enabled = False
        self._gripper.observables.enable_all()
        self._prop.observables.enable_all()

    # def _build_variations(self):
    #     pass
