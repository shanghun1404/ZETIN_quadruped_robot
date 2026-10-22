from __future__ import annotations

import math
from collections.abc import Mapping


TARGET_HEIGHT_M = 0.35


def base_x_command(semantic_speed: float) -> float:
    return -float(semantic_speed)


def _height_passes(metrics: Mapping[str, object]) -> bool:
    error = abs(float(metrics["mean_base_height_m"]) - TARGET_HEIGHT_M)
    return error < 0.015 or math.isclose(error, 0.015, rel_tol=0.0, abs_tol=1.0e-12)


def _pilot_gate(metrics: Mapping[str, object]) -> dict[str, bool]:
    return {
        "forward_speed": float(metrics["mean_forward_speed_m_s"]) >= 0.10,
        "wrong_direction": float(metrics["max_wrong_direction_s"]) <= 0.15,
        "lateral_displacement": float(metrics["p95_lateral_displacement_m"]) <= 0.10,
        "heading_drift": float(metrics["p95_heading_drift_deg"]) <= 5.0,
        "yaw_rate": float(metrics["rms_yaw_rate_rad_s"]) <= 0.10,
        "continuous_support": float(metrics["support_ge2_fraction"]) >= 0.85,
        "under_support_streak": float(metrics["max_under_support_s"]) <= 0.12,
        "flight_fraction": float(metrics["flight_fraction"]) <= 0.005,
        "flight_streak": float(metrics["max_flight_s"]) <= 0.04,
        "mean_height": _height_passes(metrics),
        "height_range": float(metrics["p95_height_ptp_m"]) <= 0.05,
        "vertical_velocity": float(metrics["rms_vertical_velocity_m_s"]) <= 0.12,
        "roll_pitch_rate": float(metrics["rms_roll_pitch_rate_rad_s"]) <= 0.35,
        "survival": float(metrics["survival_fraction"]) >= 0.95,
        "finite_terms": bool(metrics["all_terms_finite"]),
    }


def _release_gate(metrics: Mapping[str, object]) -> dict[str, bool]:
    return {
        "forward_speed": float(metrics["mean_forward_speed_m_s"]) >= 0.32,
        "wrong_direction": float(metrics["max_wrong_direction_s"]) <= 0.25,
        "lateral_displacement": float(metrics["p95_lateral_displacement_m"]) <= 0.10,
        "heading_drift": float(metrics["p95_heading_drift_deg"]) <= 5.0,
        "yaw_rate": float(metrics["rms_yaw_rate_rad_s"]) <= 0.05,
        "continuous_support": float(metrics["support_ge2_fraction"]) >= 0.95,
        "under_support_streak": float(metrics["max_under_support_s"]) <= 0.08,
        "flight_fraction": float(metrics["flight_fraction"]) <= 0.001,
        "flight_streak": float(metrics["max_flight_s"]) <= 0.02,
        "mean_height": _height_passes(metrics),
        "height_range": float(metrics["p95_height_ptp_m"]) <= 0.05,
        "vertical_velocity": float(metrics["rms_vertical_velocity_m_s"]) <= 0.08,
        "roll_pitch_rate": float(metrics["rms_roll_pitch_rate_rad_s"]) <= 0.25,
        "foot_impact": float(metrics["mean_normalized_foot_impact"]) <= 0.15,
        "survival": float(metrics["survival_fraction"]) >= 0.95,
        "finite_terms": bool(metrics["all_terms_finite"]),
    }


def evaluate_comparable_gate(
    metrics: Mapping[str, object], protocol: str
) -> dict[str, bool]:
    if protocol == "pilot":
        return _pilot_gate(metrics)
    if protocol == "release":
        return _release_gate(metrics)
    raise ValueError(f"unsupported protocol: {protocol}")
