from collections.abc import Sequence

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from guide_frame import guide_forward_speed_range_to_base
from smooth_mdp import effective_schedule_step
from v5_3_schedule import semantic_speed_range_v5_3


class velocity_command_curriculum_v5_3(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = env.command_manager.get_term("base_velocity")

    def __call__(self, env, env_ids: Sequence[int]) -> float:
        del env_ids
        semantic_range = semantic_speed_range_v5_3(effective_schedule_step(env))
        self._command.cfg.ranges.lin_vel_x = guide_forward_speed_range_to_base(
            semantic_range
        )
        return semantic_range[1]
