import torch

import grounded_gate
from grounded_gate import evaluate_gate, evaluate_pilot_gate


def passing_metrics():
    return {
        "mean_forward_speed_m_s": 0.32,
        "max_wrong_direction_s": 0.25,
        "p95_lateral_displacement_m": 0.10,
        "p95_heading_drift_deg": 5.0,
        "rms_yaw_rate_rad_s": 0.05,
        "support_ge2_fraction": 0.95,
        "max_under_support_s": 0.08,
        "flight_fraction": 0.001,
        "max_flight_s": 0.02,
        "mean_base_height_m": 0.35,
        "p95_height_ptp_m": 0.05,
        "rms_vertical_velocity_m_s": 0.08,
        "rms_roll_pitch_rate_rad_s": 0.25,
        "mean_normalized_foot_impact": 0.15,
        "survival_fraction": 0.95,
        "all_terms_finite": True,
        "all_terms_active": True,
    }


def pilot_passing_metrics():
    return {
        "mean_forward_speed_m_s": 0.10,
        "max_wrong_direction_s": 0.15,
        "p95_lateral_displacement_m": 0.10,
        "p95_heading_drift_deg": 5.0,
        "rms_yaw_rate_rad_s": 0.10,
        "support_ge2_fraction": 0.85,
        "max_under_support_s": 0.12,
        "flight_fraction": 0.005,
        "max_flight_s": 0.04,
        "mean_base_height_m": 0.35,
        "p95_height_ptp_m": 0.05,
        "rms_vertical_velocity_m_s": 0.12,
        "rms_roll_pitch_rate_rad_s": 0.35,
        "survival_fraction": 0.95,
        "all_terms_finite": True,
        "all_terms_active": True,
    }


def test_pilot_gate_accepts_every_boundary():
    for height in (0.335, 0.365):
        metrics = pilot_passing_metrics()
        metrics["mean_base_height_m"] = height
        result = evaluate_pilot_gate(metrics)
        assert all(result.values()), (height, result)


def test_pilot_gate_rejects_each_individual_regression():
    cases = {
        "forward_speed": ("mean_forward_speed_m_s", 0.099),
        "wrong_direction": ("max_wrong_direction_s", 0.151),
        "lateral_displacement": ("p95_lateral_displacement_m", 0.101),
        "heading_drift": ("p95_heading_drift_deg", 5.01),
        "yaw_rate": ("rms_yaw_rate_rad_s", 0.101),
        "continuous_support": ("support_ge2_fraction", 0.849),
        "under_support_streak": ("max_under_support_s", 0.121),
        "flight_fraction": ("flight_fraction", 0.0051),
        "flight_streak": ("max_flight_s", 0.041),
        "mean_height": ("mean_base_height_m", 0.366),
        "height_range": ("p95_height_ptp_m", 0.051),
        "vertical_velocity": ("rms_vertical_velocity_m_s", 0.121),
        "roll_pitch_rate": ("rms_roll_pitch_rate_rad_s", 0.351),
        "survival": ("survival_fraction", 0.949),
        "finite_terms": ("all_terms_finite", False),
        "active_terms": ("all_terms_active", False),
    }
    for gate, (key, value) in cases.items():
        metrics = pilot_passing_metrics()
        metrics[key] = value
        assert evaluate_pilot_gate(metrics)[gate] is False, gate


def test_release_gate_accepts_every_boundary():
    result = evaluate_gate(passing_metrics())
    assert all(result.values()), result


def test_release_gate_rejects_each_individual_regression():
    cases = {
        "forward_speed": ("mean_forward_speed_m_s", 0.319),
        "wrong_direction": ("max_wrong_direction_s", 0.251),
        "lateral_displacement": ("p95_lateral_displacement_m", 0.101),
        "heading_drift": ("p95_heading_drift_deg", 5.01),
        "yaw_rate": ("rms_yaw_rate_rad_s", 0.051),
        "continuous_support": ("support_ge2_fraction", 0.949),
        "under_support_streak": ("max_under_support_s", 0.081),
        "flight_fraction": ("flight_fraction", 0.0011),
        "flight_streak": ("max_flight_s", 0.021),
        "mean_height": ("mean_base_height_m", 0.366),
        "height_range": ("p95_height_ptp_m", 0.051),
        "vertical_velocity": ("rms_vertical_velocity_m_s", 0.081),
        "roll_pitch_rate": ("rms_roll_pitch_rate_rad_s", 0.251),
        "foot_impact": ("mean_normalized_foot_impact", 0.151),
        "survival": ("survival_fraction", 0.949),
        "finite_terms": ("all_terms_finite", False),
        "active_terms": ("all_terms_active", False),
    }
    for gate, (key, value) in cases.items():
        metrics = passing_metrics()
        metrics[key] = value
        assert evaluate_gate(metrics)[gate] is False, gate


def test_contact_streak_counts_only_alive_samples_and_resets_on_recovery():
    streak = torch.zeros(3, dtype=torch.long)
    maximum = torch.zeros_like(streak)
    alive = torch.tensor([True, True, False])
    streak, maximum = grounded_gate.record_contact_streak(
        streak, maximum, torch.tensor([True, False, True]), alive
    )
    torch.testing.assert_close(streak, torch.tensor([1, 0, 0]))
    streak, maximum = grounded_gate.record_contact_streak(
        streak, maximum, torch.tensor([True, True, True]), alive
    )
    torch.testing.assert_close(streak, torch.tensor([2, 1, 0]))
    streak, maximum = grounded_gate.record_contact_streak(
        streak, maximum, torch.tensor([False, True, True]), alive
    )
    torch.testing.assert_close(streak, torch.tensor([0, 2, 0]))
    torch.testing.assert_close(maximum, torch.tensor([2, 2, 0]))


def test_active_mask_excludes_post_done_samples():
    alive = torch.tensor([True, True])
    alive = grounded_gate.advance_alive_mask(alive, torch.tensor([True, False]))
    assert alive.tolist() == [False, True]
