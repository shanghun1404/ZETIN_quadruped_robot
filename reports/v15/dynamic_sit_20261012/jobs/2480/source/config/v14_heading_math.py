"""Tensor-only direction feedback for the robot's +y forward convention."""

import torch


def lane_yaw_rate(yaw, target_yaw, cross_track, *, standing=None):
    error = torch.atan2(torch.sin(target_yaw-yaw), torch.cos(target_yaw-yaw))
    rate = torch.clamp(1.5*error + 0.6*cross_track, -0.4, 0.4)
    return torch.where(standing, 0.0, rate) if standing is not None else rate


def update_lane_reference(reference_xy, reference_yaw, root_xy, root_yaw,
                          command_counter, env_ids):
    fresh = env_ids[command_counter[env_ids] == 0]
    reference_xy[fresh] = root_xy[fresh]
    reference_yaw[fresh] = root_yaw[fresh]
