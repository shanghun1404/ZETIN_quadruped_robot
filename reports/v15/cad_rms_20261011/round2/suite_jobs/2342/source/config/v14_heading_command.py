"""Keep the episode's initial walking line through an observed yaw command."""

import copy

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from v14_heading_math import lane_yaw_rate, update_lane_reference


class V14LaneVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reference_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.reference_yaw = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        update_lane_reference(
            self.reference_xy, self.reference_yaw, self.robot.data.root_pos_w[:, :2],
            self.robot.data.heading_w, self.command_counter, ids,
        )
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()
        right = torch.stack((torch.cos(self.reference_yaw), torch.sin(self.reference_yaw)), dim=-1)
        cross_track = ((self.robot.data.root_pos_w[:, :2]-self.reference_xy)*right).sum(-1)
        self.vel_command_b[:, 2] = lane_yaw_rate(
            self.robot.data.heading_w, self.reference_yaw, cross_track, standing=self.is_standing_env
        )


@configclass
class V14LaneVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = V14LaneVelocityCommand


def make_lane_command_cfg(original):
    return V14LaneVelocityCommandCfg(
        asset_name=original.asset_name, resampling_time_range=original.resampling_time_range,
        rel_standing_envs=original.rel_standing_envs, rel_heading_envs=0.0,
        heading_command=False, debug_vis=False, ranges=copy.deepcopy(original.ranges),
    )
