"""Simulator-free helpers for V9 evaluation metrics."""

from __future__ import annotations


def pearson_correlation_or_none(
    dot_product: float,
    first_norm: float,
    second_norm: float,
    *,
    min_norm: float = 1.0e-8,
) -> float | None:
    """Return a Pearson correlation, or ``None`` when either signal is constant."""
    if min_norm <= 0.0:
        raise ValueError("min_norm must be positive")
    if first_norm <= min_norm or second_norm <= min_norm:
        return None
    return dot_product / (first_norm * second_norm)


def climbed_stairs(
    height_gain_m,
    horizontal_progress_m,
    step_height_m,
    approach_distance_m: float,
):
    """Require both two steps of height and enough travel to reach those steps."""
    return (height_gain_m >= 2.0 * step_height_m) & (
        horizontal_progress_m >= approach_distance_m
    )
