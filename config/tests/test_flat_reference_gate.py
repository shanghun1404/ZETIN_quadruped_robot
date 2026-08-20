from __future__ import annotations

from pathlib import Path

import pytest

from flat_reference_gate import base_x_command, evaluate_comparable_gate
from grounded_gate import evaluate_gate, evaluate_pilot_gate


def passing_metrics() -> dict[str, object]:
    return {
        "mean_forward_speed_m_s": 0.40,
        "max_wrong_direction_s": 0.0,
        "p95_lateral_displacement_m": 0.0,
        "p95_heading_drift_deg": 0.0,
        "rms_yaw_rate_rad_s": 0.0,
        "support_ge2_fraction": 1.0,
        "max_under_support_s": 0.0,
        "flight_fraction": 0.0,
        "max_flight_s": 0.0,
        "mean_base_height_m": 0.35,
        "p95_height_ptp_m": 0.0,
        "rms_vertical_velocity_m_s": 0.0,
        "rms_roll_pitch_rate_rad_s": 0.0,
        "mean_normalized_foot_impact": 0.0,
        "survival_fraction": 1.0,
        "all_terms_finite": True,
        "all_terms_active": False,
    }


def test_semantic_forward_maps_to_negative_base_x():
    assert base_x_command(0.15) == -0.15
    assert base_x_command(-0.15) == 0.15


@pytest.mark.parametrize(
    ("protocol", "current_gate"),
    (("pilot", evaluate_pilot_gate), ("release", evaluate_gate)),
)
def test_comparable_gate_matches_current_numeric_contract_without_reward_inventory(
    protocol, current_gate
):
    metrics = passing_metrics()
    expected = current_gate(metrics)
    expected.pop("active_terms")

    gates = evaluate_comparable_gate(metrics, protocol)

    assert "active_terms" not in gates
    assert gates == expected
    assert all(gates.values())


def test_comparable_gate_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="unsupported protocol"):
        evaluate_comparable_gate(passing_metrics(), "unknown")


def test_evaluator_uses_historical_flat_contract():
    source = Path("evaluate_flat_reference.py").read_text()

    assert "ForFlatEnvCfg" in source
    assert "ForFlatPPORunnerCfg" in source
    assert 'base_x_command(protocol["speed"])' in source
    assert "env_cfg.actions.joint_pos.scale = 0.05" in source
    assert "load_optimizer=False" in source
    assert '"actor_observation_dim": 48' in source
    assert "WARMUP_DURATION_S = 2.0" in source
    assert '"z": (0.4, 0.4)' in source
    assert '"warmup_duration_s": WARMUP_DURATION_S' in source
