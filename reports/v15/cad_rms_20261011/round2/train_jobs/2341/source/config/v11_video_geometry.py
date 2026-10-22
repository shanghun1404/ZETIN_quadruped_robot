"""Pure geometry helpers for recording V11 stair rollouts."""

from __future__ import annotations


def stair_step_height_m(
    height_range_m: tuple[float, float],
    levels: int,
    level: int,
) -> float:
    """Return the physical stair height for one discrete terrain row."""
    if levels < 2:
        raise ValueError("levels must be at least two")
    if not 0 <= level < levels:
        raise ValueError(f"level must be between 0 and {levels - 1}")
    low, high = height_range_m
    return low + (high - low) * level / (levels - 1)


def tracking_camera_pose(
    root_xyz: tuple[float, float, float],
    view: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Place a tracking camera around a robot moving along base +y."""
    x, y, z = root_xyz
    if view == "side":
        eye = (x - 2.6, y, z + 1.0)
        target = (x, y + 0.2, z + 0.05)
    elif view == "rear_quarter":
        eye = (x - 1.8, y - 2.2, z + 1.4)
        target = (x, y + 0.4, z + 0.05)
    else:
        raise ValueError(f"unsupported view: {view}")
    return eye, target


def stair_rollout_passed(
    *,
    alive: bool,
    final_progress_m: float,
    peak_height_after_approach_m: float,
    approach_distance_m: float,
    step_height_m: float,
) -> bool:
    """Apply the evaluator's survival, approach and two-step climb contract."""
    return (
        alive
        and final_progress_m >= approach_distance_m
        and peak_height_after_approach_m >= 2.0 * step_height_m
    )


def stair_descent_passed(
    *,
    alive: bool,
    final_progress_m: float,
    max_descent_after_approach_m: float,
    approach_distance_m: float,
    step_height_m: float,
) -> bool:
    """Require survival, stair approach and at least two steps of descent."""
    return (
        alive
        and final_progress_m >= approach_distance_m
        and max_descent_after_approach_m >= 2.0 * step_height_m
    )
