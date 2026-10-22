"""V13 actuator regularization for the V12 depth-policy lineage."""

from __future__ import annotations

import math

import torch


def low_pass_alpha(*, cutoff_hz: float, control_dt_s: float) -> float:
    """Return the backward-Euler coefficient of a first-order command LPF."""
    if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be finite and positive")
    if cutoff_hz >= 0.5 / control_dt_s:
        raise ValueError("cutoff_hz must be below the control-rate Nyquist frequency")
    return control_dt_s / (control_dt_s + 1.0 / (2.0 * math.pi * cutoff_hz))


def low_pass_step(raw: torch.Tensor, previous: torch.Tensor, alpha: float) -> torch.Tensor:
    if raw.shape != previous.shape:
        raise ValueError("raw and previous action targets must have the same shape")
    return previous + alpha * (raw - previous)


def clip_position_target_to_motor_effort(
    target: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    effort_limits: torch.Tensor,
    *,
    stiffness: float,
    damping: float,
    velocity_limit: float,
    fraction: float,
) -> torch.Tensor:
    """Limit PD position targets to a fraction of the DCMotor torque-speed envelope."""
    if target.ndim != 2 or position.shape != target.shape or velocity.shape != target.shape:
        raise ValueError("target, position and velocity must share [env, joint] shape")
    if effort_limits.shape != (target.shape[1],) or bool((effort_limits <= 0).any()):
        raise ValueError("effort limits must be positive and match joint count")
    if not all(math.isfinite(value) for value in (stiffness, damping, velocity_limit, fraction)):
        raise ValueError("motor parameters must be finite")
    if stiffness <= 0.0 or damping < 0.0 or velocity_limit <= 0.0 or not 0.0 < fraction <= 1.0:
        raise ValueError("motor gains, velocity limit and fraction must be valid")

    # V13 uses equal continuous and stall effort, as configured for robot_v2.
    clipped_velocity = velocity.clamp(-2.0 * velocity_limit, 2.0 * velocity_limit)
    maximum = torch.minimum(effort_limits * (1.0 - clipped_velocity / velocity_limit), effort_limits)
    minimum = torch.maximum(effort_limits * (-1.0 - clipped_velocity / velocity_limit), -effort_limits)
    midpoint = 0.5 * (minimum + maximum)
    half_width = 0.5 * fraction * (maximum - minimum)
    target_min = position + (midpoint - half_width + damping * velocity) / stiffness
    target_max = position + (midpoint + half_width + damping * velocity) / stiffness
    return torch.minimum(torch.maximum(target, target_min), target_max)


def action_rate_squared(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Per-environment sum of squared changes in raw policy actions."""
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError("actions must have equal two-dimensional shapes")
    return torch.square(current - previous).sum(dim=1)


def torque_limit_penalty(
    requested_torque_nm: torch.Tensor,
    limits_nm: torch.Tensor,
    *,
    onset_fraction: float = 0.8,
) -> torch.Tensor:
    """Exponential cost near joint limits, with a finite linear over-limit tail."""
    if requested_torque_nm.ndim != 2 or limits_nm.ndim != 1:
        raise ValueError("torque must be [env, joint] and limits must be [joint]")
    if requested_torque_nm.shape[1] != limits_nm.shape[0]:
        raise ValueError("torque and limit joint counts must match")
    if not 0.0 < onset_fraction < 1.0 or bool((limits_nm <= 0).any()):
        raise ValueError("limits must be positive and onset_fraction within (0, 1)")
    ratio = requested_torque_nm.abs() / limits_nm
    excess = ((ratio - onset_fraction) / (1.0 - onset_fraction)).clamp_min(0.0)
    near = torch.expm1(2.0 * excess.clamp_max(1.0))
    tail = 2.0 * math.exp(2.0) * (excess - 1.0).clamp_min(0.0)
    return (near + tail).mean(dim=1)


def power_penalty(
    torque_nm: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    *,
    reference_w: float = 20.0,
) -> torch.Tensor:
    """Mean absolute mechanical power per joint, normalized to reference_w."""
    if torque_nm.shape != velocity_rad_s.shape or torque_nm.ndim != 2:
        raise ValueError("torque and velocity must have equal [env, joint] shapes")
    if not math.isfinite(reference_w) or reference_w <= 0.0:
        raise ValueError("reference_w must be finite and positive")
    return (torque_nm * velocity_rad_s).abs().mean(dim=1) / reference_w


def policy_action_rate_squared(env) -> torch.Tensor:
    return action_rate_squared(env.action_manager.action, env.action_manager.prev_action)


def requested_torque_pressure(env, asset_cfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    names = [asset.joint_names[index] for index in ids]
    limits = torch.tensor(
        [24.8 if name.endswith("_HAA") else 48.0 for name in names],
        device=asset.data.computed_torque.device,
        dtype=asset.data.computed_torque.dtype,
    )
    return torque_limit_penalty(asset.data.computed_torque[:, ids], limits)


def applied_joint_power(env, asset_cfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return power_penalty(asset.data.applied_torque[:, ids], asset.data.joint_vel[:, ids])
