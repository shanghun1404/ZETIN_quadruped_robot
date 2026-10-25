from __future__ import annotations

import math

import torch


def ema_step(raw: torch.Tensor, previous: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return alpha * raw + (1.0 - alpha) * previous


def target_rate_penalty(current: torch.Tensor, previous: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    return torch.clamp(((current - previous) / scale).square(), 0.0, 1.0).mean(dim=1)


def target_jerk_penalty(
    current: torch.Tensor,
    previous: torch.Tensor,
    older: torch.Tensor,
    scale: float = 0.025,
) -> torch.Tensor:
    second_difference = current - 2.0 * previous + older
    return torch.clamp((second_difference / scale).square(), 0.0, 1.0).mean(dim=1)


def normalize_foot_forces(forces: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    masses = masses.to(device=forces.device, dtype=forces.dtype)
    body_weight = masses.sum(dim=1, keepdim=True) * 9.81
    nominal_force = body_weight / max(forces.shape[1], 1)
    return torch.clamp(forces / nominal_force, 0.0, 4.0) / 4.0


def signed_progress_ratio(
    semantic_velocity: torch.Tensor,
    semantic_command: torch.Tensor,
    command_floor: float = 0.05,
) -> torch.Tensor:
    if command_floor <= 0.0:
        raise ValueError("command_floor must be positive")
    denominator = torch.clamp(torch.abs(semantic_command), min=command_floor)
    return torch.clamp(semantic_velocity / denominator, -1.0, 1.0)


def hybrid_signed_progress_ratio(
    forward_velocity: torch.Tensor,
    forward_command: torch.Tensor,
    lateral_velocity: torch.Tensor,
    roll_pitch_rate: torch.Tensor,
    heading_error: torch.Tensor,
    command_floor: float = 0.05,
    stability_floor: float = 0.35,
) -> torch.Tensor:
    signed = signed_progress_ratio(forward_velocity, forward_command, command_floor)
    stability = torch.exp(
        -(lateral_velocity / 0.12).square()
        - (roll_pitch_rate / 0.60).square()
        - (heading_error / math.radians(10.0)).square()
    )
    stable_forward = signed * (stability_floor + (1.0 - stability_floor) * stability)
    return torch.where(signed < 0.0, signed, stable_forward)


def wrong_direction_ratio(
    forward_velocity: torch.Tensor,
    scale: float = 0.10,
) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return torch.clamp(torch.relu(-forward_velocity) / scale, 0.0, 1.0).square()


def flight_indicator(contact_count: torch.Tensor) -> torch.Tensor:
    return (contact_count == 0.0).to(dtype=contact_count.dtype)


def support_deficit_ratio(contact_count: torch.Tensor, minimum_contacts: int = 2) -> torch.Tensor:
    deficit = torch.clamp(minimum_contacts - contact_count, 0.0, float(minimum_contacts))
    return deficit / minimum_contacts


def bounded_square_error(error: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return torch.clamp((error / scale).square(), 0.0, 1.0)


def wrapped_angle_error(angle: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = angle - target
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def heading_error_features(yaw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = wrapped_angle_error(target, yaw)
    return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


def quaternion_yaw_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion width must be 4 in wxyz order")
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def vertical_acceleration_step(
    current: torch.Tensor,
    previous: torch.Tensor,
    initialized: torch.Tensor,
    step_dt: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    acceleration = (current - previous) / step_dt
    penalty = bounded_square_error(acceleration, scale)
    penalty = torch.where(initialized, penalty, torch.zeros_like(penalty))
    return penalty, current.clone(), torch.ones_like(initialized)


def swing_clearance_penalty(
    clearance: torch.Tensor,
    actual_contacts: torch.Tensor,
    low: float,
    high: float,
    margin: float,
) -> torch.Tensor:
    if not 0.0 <= low < high or margin <= 0.0:
        raise ValueError("clearance band and margin are invalid")
    below = torch.clamp(low - clearance, min=0.0) / margin
    above = torch.clamp(clearance - high, min=0.0) / margin
    per_foot = torch.clamp(below.square() + above.square(), 0.0, 1.0)
    swing_weight = 1.0 - actual_contacts
    denominator = torch.clamp(swing_weight.sum(dim=-1), min=1.0)
    return (per_foot * swing_weight).sum(dim=-1) / denominator


def smoothstep(step: int, start: int, end: int) -> float:
    if end <= start:
        raise ValueError("end must be greater than start")
    x = min(max((step - start) / (end - start), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)
