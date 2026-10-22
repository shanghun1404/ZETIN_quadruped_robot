"""Comparable physics-step actuator metrics for V12 and V13 rollouts."""

from __future__ import annotations

import math

import torch


def summarize_actuation(
    applied_nm: torch.Tensor,
    requested_nm: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    alive: torch.Tensor,
    limits_nm: torch.Tensor,
    *,
    physics_dt_s: float,
) -> dict[str, object]:
    if applied_nm.ndim != 3 or requested_nm.shape != applied_nm.shape or velocity_rad_s.shape != applied_nm.shape:
        raise ValueError("actuation arrays must share [time, env, joint] shape")
    if alive.shape != applied_nm.shape[:2] or alive.dtype != torch.bool:
        raise ValueError("alive mask must be boolean [time, env]")
    if limits_nm.shape != (applied_nm.shape[2],) or bool((limits_nm <= 0).any()):
        raise ValueError("positive joint limits must match the joint dimension")
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    if not bool(alive.any()):
        raise ValueError("rollout contains no live actuation samples")

    applied = applied_nm[alive].abs()
    requested = requested_nm[alive].abs()
    velocity = velocity_rad_s[alive].abs()
    limits = limits_nm.to(applied.device)
    result: dict[str, object] = {
        "peak_applied_nm": applied.amax(dim=0).tolist(),
        "peak_requested_nm": requested.amax(dim=0).tolist(),
        "p99_applied_nm": float(torch.quantile(applied.reshape(-1), 0.99)),
        "p99_requested_nm": float(torch.quantile(requested.reshape(-1), 0.99)),
        "applied_saturation_fraction": float((applied >= 0.99 * limits).float().mean()),
        "over_limit_request_fraction": float((requested > limits).float().mean()),
        "mean_joint_power_w": float((applied * velocity).mean()),
        "sampled_physics_steps": int(alive.sum()),
    }
    consecutive = alive[1:] & alive[:-1]
    if bool(consecutive.any()):
        slew = (applied_nm[1:] - applied_nm[:-1]).abs()[consecutive] / physics_dt_s
        result["p99_applied_torque_slew_nm_s"] = float(torch.quantile(slew.reshape(-1), 0.99))
    else:
        result["p99_applied_torque_slew_nm_s"] = 0.0
    return result
