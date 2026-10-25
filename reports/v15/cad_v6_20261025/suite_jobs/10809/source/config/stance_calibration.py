from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


TARGET_HEIGHT_M = 0.35
HEIGHT_BAND_M = (0.335, 0.365)
THIGH_LENGTH_M = 0.25
CALF_LENGTH_M = 0.26
FOOT_RADIUS_M = 0.020
SPAWN_CLEARANCE_M = 0.005
STANCE_HALF_ANGLES_RAD = (0.82, 0.85, 0.87, 0.90)


@dataclass(frozen=True)
class StanceCandidate:
    half_angle_rad: float
    hfe_rad: float
    kfe_rad: float
    spawn_height_m: float


def _candidate(half_angle_rad: float) -> StanceCandidate:
    vertical_leg_length = (THIGH_LENGTH_M + CALF_LENGTH_M) * math.cos(half_angle_rad)
    return StanceCandidate(
        half_angle_rad=half_angle_rad,
        hfe_rad=-half_angle_rad,
        kfe_rad=2.0 * half_angle_rad,
        spawn_height_m=vertical_leg_length + FOOT_RADIUS_M + SPAWN_CLEARANCE_M,
    )


STANCE_CANDIDATES = tuple(_candidate(angle) for angle in STANCE_HALF_ANGLES_RAD)


@dataclass(frozen=True)
class StanceMetrics:
    candidate: StanceCandidate
    mean_height_m: float
    support_fraction: float
    rms_vertical_velocity_m_s: float
    p95_height_range_m: float
    body_collision: bool
    low_height: bool


def passes_stance(metrics: StanceMetrics) -> bool:
    return (
        HEIGHT_BAND_M[0] <= metrics.mean_height_m <= HEIGHT_BAND_M[1]
        and metrics.support_fraction >= 0.99
        and metrics.rms_vertical_velocity_m_s <= 0.08
        and metrics.p95_height_range_m <= 0.03
        and not metrics.body_collision
        and not metrics.low_height
    )


def select_stance_candidate(values: Iterable[StanceMetrics]) -> StanceMetrics:
    passing = [metrics for metrics in values if passes_stance(metrics)]
    if not passing:
        raise ValueError("no stance candidate passed calibration")
    return min(
        passing,
        key=lambda metrics: (
            abs(metrics.mean_height_m - TARGET_HEIGHT_M),
            metrics.candidate.spawn_height_m,
        ),
    )
