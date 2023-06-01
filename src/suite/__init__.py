from dm_control.composer import Environment

from src.suite.tasks import base
from src.suite.tasks.pick_and_lift import PickAndLift
from src.suite import common


def load(random_state: int,
         *,
         action_mode: base.ActionMode = 'discrete',
         control_timestep: float = common.CONTROL_TIMESTEP,
         img_size: tuple[int, int] = (100, 100),
         time_limit: float = float('inf'),
         ) -> Environment:
    task = PickAndLift(
        action_mode=action_mode,
        control_timestep=control_timestep,
        img_size=img_size
    )
    return Environment(
        task,
        random_state=random_state,
        time_limit=time_limit,
        raise_exception_on_physics_error=True,
        max_reset_attempts=5,
        strip_singleton_obs_buffer_dim=True,
    )

