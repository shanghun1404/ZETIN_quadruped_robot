"""Simulator-free acceptance gates for V11 rough and stair evaluation."""

from __future__ import annotations

import torch


MIN_REALISTIC_STEP_HEIGHT_M = 0.15


def stair_level_assignments(num_envs: int, levels: int) -> list[int]:
    """Assign stair rows cyclically so every configured difficulty is measured."""
    if levels < 1:
        raise ValueError("levels must be positive")
    if num_envs < levels:
        raise ValueError(f"stairs evaluation needs at least {levels} environments")
    return [environment % levels for environment in range(num_envs)]


def peak_height_after_approach(
    height_gain_m: torch.Tensor,
    horizontal_progress_m: torch.Tensor,
    valid_samples: torch.Tensor,
    approach_distance_m: float,
) -> torch.Tensor:
    """Keep peak climb height after reaching the stairs, even after descent."""
    if not (
        height_gain_m.shape
        == horizontal_progress_m.shape
        == valid_samples.shape
    ):
        raise ValueError("height, progress and validity tensors must have equal shape")
    eligible = valid_samples.bool() & (
        horizontal_progress_m >= approach_distance_m
    )
    candidates = torch.where(
        eligible,
        height_gain_m,
        torch.full_like(height_gain_m, -torch.inf),
    )
    peak = candidates.max(dim=0).values
    fallback = torch.where(
        valid_samples.bool(),
        height_gain_m,
        torch.full_like(height_gain_m, -torch.inf),
    ).max(dim=0).values
    return torch.where(torch.isfinite(peak), peak, fallback)


def minimum_level_for_step_height(
    height_range: tuple[float, float], levels: int, minimum_m: float
) -> int:
    """Return the first discrete terrain level meeting a physical step height."""
    low, high = height_range
    if levels < 2:
        raise ValueError("levels must be at least two")
    for level in range(levels):
        height = low + (high - low) * level / (levels - 1)
        if height >= minimum_m:
            return level
    raise ValueError(f"no terrain level reaches {minimum_m} m")


def rough_gates(metrics: dict[str, float]) -> dict[str, bool]:
    """Grade travel and data quality on the generated rough-terrain mix."""
    return {
        "rough_survival": metrics["survival_fraction"] >= 0.90,
        "rough_speed": metrics["mean_forward_speed_m_s"] >= 0.30,
        "direct_phase_valid": metrics["direct_phase_valid_fraction"] >= 0.90,
    }


def stairs_gates(
    metrics: dict[str, float],
    stairs: dict[str, float | int],
    minimum_step_height_m: float = MIN_REALISTIC_STEP_HEIGHT_M,
) -> dict[str, bool]:
    """Grade survival, total climbing and a directly measured physical height."""
    return {
        "stairs_survival": metrics["survival_fraction"] >= 0.80,
        "stairs_climb_success": stairs["climb_success_fraction"] >= 0.50,
        "stairs_clears_real_step": (
            stairs["hardest_step_height_cleared_m"] >= minimum_step_height_m
        ),
    }
