"""V13 LPF + torque guard stage and its lockstep ordering; CPU tensors, no ROS."""

import math

import numpy as np
import pytest
import torch

from v14_ros2_control import contract
from v14_ros2_control.joint_command import JointCommandFilter, LockstepJointCommander


LIMITS = np.array([contract.joint_effort_limit(n) for n in contract.JOINT_NAMES])


def reference_step(raw, filtered, position, velocity, fraction):
    """Independent float64 statement of the V13 action: LPF, then DCMotor-envelope target clip."""
    alpha = 0.005 / (0.005 + 1.0 / (2.0 * math.pi * 25.0))
    filtered = filtered + alpha * (raw - filtered)
    v = np.clip(velocity, -100.0, 100.0)
    upper = np.minimum(LIMITS * (1.0 - v / 50.0), LIMITS)
    lower = np.maximum(LIMITS * (-1.0 - v / 50.0), -LIMITS)
    middle, half = 0.5 * (upper + lower), 0.5 * fraction * (upper - lower)
    low = position + (middle - half + 8.0 * velocity) / 300.0
    high = position + (middle + half + 8.0 * velocity) / 300.0
    return filtered, np.minimum(np.maximum(filtered, low), high)


def test_filter_matches_independent_v13_statement_over_a_rollout():
    rng = np.random.default_rng(14)
    command_filter = JointCommandFilter()
    position = rng.normal(size=12) * 0.3
    command_filter.reset(torch.tensor(position, dtype=torch.float32))
    filtered = position.copy()
    for step in range(200):
        if step % 4 == 0:
            raw = position + rng.normal(size=12) * 0.4
            command_filter.set_target(torch.tensor(raw, dtype=torch.float32))
        position = position + rng.normal(size=12) * 0.01
        velocity = rng.normal(size=12) * 8.0
        actual = command_filter.step(torch.tensor(position, dtype=torch.float32),
                                     torch.tensor(velocity, dtype=torch.float32))
        filtered, expected = reference_step(raw, filtered, position, velocity, 0.99)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-5)
    # The filter state is the unguarded LPF output, as in V13LowPassJointPositionAction.
    np.testing.assert_allclose(command_filter.filtered.numpy(), filtered, atol=2e-5)


def test_guard_keeps_requested_pd_torque_inside_the_envelope():
    command_filter = JointCommandFilter()
    position = torch.zeros(12)
    command_filter.reset(position)
    command_filter.set_target(torch.full((12,), 3.0))
    for _ in range(40):
        velocity = torch.full((12,), 12.0)
        command = command_filter.step(position, velocity)
        torque = 300.0 * (command - position) - 8.0 * velocity
        upper = torch.tensor(LIMITS, dtype=torch.float32) * (1.0 - 12.0 / 50.0)
        assert bool((torque <= 0.99 * upper + 1e-3).all())


def test_unguarded_filter_is_a_plain_first_order_lpf():
    command_filter = JointCommandFilter(guard_fraction=None)
    command_filter.reset(torch.zeros(12))
    command_filter.set_target(torch.ones(12))
    outputs = [float(command_filter.step(torch.zeros(12), torch.zeros(12))[0]) for _ in range(400)]
    assert outputs[0] == pytest.approx(command_filter.alpha, rel=1e-6)
    assert outputs[-1] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        command_filter.set_target(torch.full((12,), float("nan")))


def test_policy_ticks_wait_for_their_own_target():
    commander = LockstepJointCommander(JointCommandFilter())
    zeros = [0.0] * 12
    for step in range(6):
        commander.add_state(step * contract.PHYSICS_DT_NS, zeros, zeros)
    assert commander.drain() == []  # step 0 is a policy tick
    commander.add_target(0, [0.1] * 12)
    ready = commander.drain()
    assert [stamp for stamp, _ in ready] == [0, 5_000_000, 10_000_000, 15_000_000]
    commander.add_target(4 * contract.PHYSICS_DT_NS, [0.2] * 12)
    assert [stamp for stamp, _ in commander.drain()] == [20_000_000, 25_000_000]
    with pytest.raises(ValueError):
        commander.add_target(contract.PHYSICS_DT_NS, zeros)
