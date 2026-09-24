"""Behavioral checks for V13 action filtering and actuator penalties."""

import math

import pytest
import torch

import v13_torque_mdp as mdp
from v13_torque_mdp import action_rate_squared, low_pass_alpha, low_pass_step, power_penalty, torque_limit_penalty


def test_25_hz_filter_smooths_each_200_hz_physics_step() -> None:
    alpha = low_pass_alpha(cutoff_hz=25.0, control_dt_s=0.005)
    assert alpha == pytest.approx(0.439900, abs=1e-6)
    raw = torch.tensor([[1.0, -1.0]])
    previous = torch.zeros_like(raw)
    assert torch.allclose(low_pass_step(raw, previous, alpha), raw * alpha)
    after_four_steps = previous
    for _ in range(4):
        after_four_steps = low_pass_step(raw, after_four_steps, alpha)
    assert torch.allclose(after_four_steps, raw * 0.901584, atol=1e-5)


def test_filter_rejects_cutoff_at_or_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        low_pass_alpha(cutoff_hz=100.0, control_dt_s=0.005)


def test_action_rate_uses_squared_policy_output_difference() -> None:
    current = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
    previous = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    assert action_rate_squared(current, previous).tolist() == [4.0, 1.0]


def test_torque_pressure_uses_each_joint_limit_and_rises_near_saturation() -> None:
    limits = torch.tensor([24.8, 48.0])
    torques = torch.tensor([[19.84, 38.4], [24.8, 0.0], [0.0, 48.0]])
    values = torque_limit_penalty(torques, limits)
    assert values[0] == pytest.approx(0.0, abs=1e-5)
    assert values[1] == pytest.approx((math.exp(2.0) - 1.0) / 2.0)
    assert values[2] == pytest.approx(values[1])


def test_torque_pressure_still_rises_for_over_limit_motor_request() -> None:
    limits = torch.tensor([48.0])
    values = torque_limit_penalty(torch.tensor([[48.0], [100.0]]), limits)
    assert torch.isfinite(values).all()
    assert values[1] > values[0] * 5.0


def test_power_penalizes_braking_and_is_zero_at_rest() -> None:
    torques = torch.tensor([[10.0, -10.0], [10.0, 10.0]])
    velocities = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
    assert power_penalty(torques, velocities, reference_w=20.0).tolist() == [1.0, 0.0]


def test_effort_guard_caps_requests_at_speed_dependent_motor_bounds() -> None:
    target = torch.tensor([[1.0, -1.0, 0.02]])
    position = torch.zeros_like(target)
    velocity = torch.tensor([[10.0, -10.0, 0.0]])
    bounded = mdp.clip_position_target_to_motor_effort(
        target,
        position,
        velocity,
        torch.tensor([48.0, 48.0, 24.8]),
        stiffness=300.0,
        damping=8.0,
        velocity_limit=50.0,
        fraction=0.95,
    )
    requested = 300.0 * (bounded - position) - 8.0 * velocity
    assert requested[0, 0] == pytest.approx(36.24, abs=1e-4)
    assert requested[0, 1] == pytest.approx(-36.24, abs=1e-4)
    assert bounded[0, 2] == pytest.approx(0.02)
