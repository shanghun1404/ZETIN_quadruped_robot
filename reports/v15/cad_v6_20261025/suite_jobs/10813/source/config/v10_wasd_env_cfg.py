"""Flat blind V10 environment with exact categorical WASD commands."""

from __future__ import annotations

from isaaclab.utils import configclass

import v13_mdp
from v10_3_flat_env_cfg import GuideV2FlatSpawnFixedEnvCfg
from v10_wasd_commands import WASDVelocityCommandCfg


@configclass
class GuideV2V10WASDEnvCfg(GuideV2FlatSpawnFixedEnvCfg):
    """V10.3 flat physics plus the V10.4-compatible WASD command contract."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = WASDVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=WASDVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.3, 0.3),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.4, 0.4),
            ),
        )
        self.rewards.gait.func = v13_mdp.CommandTrackingGaitReward
        self.rewards.gait.params["standing_std"] = 0.05
        self.rewards.base_linear_velocity.func = (
            v13_mdp.command_aware_base_linear_velocity_reward
        )
        self.rewards.base_linear_velocity.params["standing_std"] = 0.08
