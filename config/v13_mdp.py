"""V13 reward terms that make gait credit conditional on command tracking."""

from __future__ import annotations

import math

import torch

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp
except ModuleNotFoundError:  # Keep the pure gate testable without IsaacLab.
    class _GaitReward:
        pass
else:
    _GaitReward = spot_mdp.GaitReward


def command_alignment_gate(
    target_velocity: torch.Tensor,
    actual_velocity: torch.Tensor,
    *,
    standing_std: float,
) -> torch.Tensor:
    """Keep full gait credit for aligned motion and suppress wrong directions."""

    if not isinstance(standing_std, (int, float)) or not math.isfinite(
        float(standing_std)
    ) or standing_std <= 0.0:
        raise ValueError("standing_std must be a finite positive number")
    if (
        target_velocity.shape != actual_velocity.shape
        or target_velocity.ndim != 2
        or target_velocity.shape[1] != 3
    ):
        raise ValueError("target and actual velocities must have equal shape (N, 3)")
    if not bool(torch.isfinite(target_velocity).all()) or not bool(
        torch.isfinite(actual_velocity).all()
    ):
        raise ValueError("target and actual velocities must be finite")
    target_norm = torch.linalg.vector_norm(target_velocity, dim=1)
    actual_norm = torch.linalg.vector_norm(actual_velocity, dim=1)
    denominator = torch.clamp(
        target_norm * actual_norm,
        min=torch.finfo(target_velocity.dtype).eps,
    )
    alignment = torch.clamp(
        torch.sum(target_velocity * actual_velocity, dim=1) / denominator,
        min=0.0,
        max=1.0,
    )
    standing = torch.exp(-actual_norm / float(standing_std))
    return torch.where(target_norm > 0.0, alignment, standing)


def command_aware_linear_std(
    target_xy: torch.Tensor,
    *,
    moving_std: float,
    standing_std: float,
) -> torch.Tensor:
    """Use a tight planar kernel only when planar motion is not commanded."""

    for name, value in (("moving_std", moving_std), ("standing_std", standing_std)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("target_xy must have shape (N, 2)")
    if not bool(torch.isfinite(target_xy).all()):
        raise ValueError("target_xy must be finite")
    moving = torch.linalg.vector_norm(target_xy, dim=1) > 0.0
    return torch.where(
        moving,
        torch.full_like(moving, float(moving_std), dtype=target_xy.dtype),
        torch.full_like(moving, float(standing_std), dtype=target_xy.dtype),
    )


def command_aware_base_linear_velocity_reward(
    env,
    asset_cfg,
    std: float,
    standing_std: float,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Preserve the stock moving reward and sharpen only zero-planar commands."""

    target_xy = env.command_manager.get_command("base_velocity")[:, :2]
    moving_reward = spot_mdp.base_linear_velocity_reward(
        env,
        asset_cfg=asset_cfg,
        std=std,
        ramp_at_vel=ramp_at_vel,
        ramp_rate=ramp_rate,
    )
    standing_reward = spot_mdp.base_linear_velocity_reward(
        env,
        asset_cfg=asset_cfg,
        std=standing_std,
        ramp_at_vel=ramp_at_vel,
        ramp_rate=ramp_rate,
    )
    use_moving_reward = torch.linalg.vector_norm(target_xy, dim=1) > 0.0
    return torch.where(use_moving_reward, moving_reward, standing_reward)


class CommandTrackingGaitReward(_GaitReward):
    """Stock Spot gait reward weighted by commanded motion alignment."""

    def __call__(
        self,
        env,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg,
        sensor_cfg,
        standing_std: float,
    ) -> torch.Tensor:
        gait = super().__call__(
            env,
            std=std,
            max_err=max_err,
            velocity_threshold=velocity_threshold,
            synced_feet_pair_names=synced_feet_pair_names,
            asset_cfg=asset_cfg,
            sensor_cfg=sensor_cfg,
        )
        target_velocity = env.command_manager.get_command("base_velocity")[:, :3]
        actual_velocity = torch.cat(
            (
                self.asset.data.root_lin_vel_b[:, :2],
                self.asset.data.root_ang_vel_b[:, 2:3],
            ),
            dim=1,
        )
        return gait * command_alignment_gate(
            target_velocity,
            actual_velocity,
            standing_std=standing_std,
        )
