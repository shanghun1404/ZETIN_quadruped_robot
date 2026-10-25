"""Simulator-independent closed-loop metrics for the V12 camera policy."""

from __future__ import annotations

import torch


def summarize_rollout(
    valid: torch.Tensor,
    alive_at_end: torch.Tensor,
    forward_speed_m_s: torch.Tensor,
    scan_mae: torch.Tensor,
    action_mae: torch.Tensor,
) -> dict[str, float | int]:
    """Average time-series metrics only while each environment is alive."""

    expected_shape = tuple(valid.shape)
    for name, tensor in (
        ("forward_speed_m_s", forward_speed_m_s),
        ("scan_mae", scan_mae),
        ("action_mae", action_mae),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if valid.ndim != 2 or alive_at_end.shape != (valid.shape[1],):
        raise ValueError("valid must be [time, env] and alive_at_end must be [env]")
    valid_count = int(valid.sum())
    if valid_count <= 0:
        raise ValueError("rollout has no valid samples")
    mask = valid.to(dtype=forward_speed_m_s.dtype)
    return {
        "survival": float(alive_at_end.float().mean()),
        "mean_forward_speed_m_s": float((forward_speed_m_s * mask).sum() / valid_count),
        "scan_mae": float((scan_mae * mask).sum() / valid_count),
        "action_mae": float((action_mae * mask).sum() / valid_count),
        "valid_samples": valid_count,
    }


def rough_gate(survival: float, mean_forward_speed_m_s: float) -> bool:
    """Apply the V12 operational rough-terrain boundary."""

    return survival >= 0.80 and mean_forward_speed_m_s >= 0.25


def stair_traversal_mask(
    scenario: str,
    alive_at_end: torch.Tensor,
    max_progress_m: torch.Tensor,
    vertical_travel_after_approach_m: torch.Tensor,
    *,
    approach_distance_m: float,
    step_height_m: float,
) -> torch.Tensor:
    """Apply the same survival/progress/two-step contract in either direction."""

    if scenario not in ("stairs_up", "stairs_down"):
        raise ValueError(f"unsupported stair scenario: {scenario}")
    if not (
        alive_at_end.shape
        == max_progress_m.shape
        == vertical_travel_after_approach_m.shape
    ):
        raise ValueError("stair metric tensors must have identical shapes")
    return (
        alive_at_end.bool()
        & (max_progress_m >= approach_distance_m)
        & (vertical_travel_after_approach_m >= 2.0 * step_height_m)
    )
