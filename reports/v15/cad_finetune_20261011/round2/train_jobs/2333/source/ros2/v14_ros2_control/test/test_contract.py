"""Lockstep timing and joint-order contract; pure Python."""

from types import SimpleNamespace

import pytest

from v14_ros2_control import contract


def test_stamps_are_exact_on_the_200hz_grid():
    for step in (0, 1, 3, 4, 199, 200, 12_345):
        sec, nanosec = contract.step_stamp(step)
        stamp = contract.stamp_ns(SimpleNamespace(sec=sec, nanosec=nanosec))
        assert stamp == step * 5_000_000
        assert contract.is_policy_tick(stamp) == (step % 4 == 0)
    assert contract.step_stamp(200) == (1, 0)
    with pytest.raises(ValueError):
        contract.is_policy_tick(5_000_001)
    with pytest.raises(ValueError):
        contract.step_stamp(-1)


def test_joint_fields_are_reordered_by_name():
    names = list(reversed(contract.JOINT_NAMES))
    values = list(range(12))
    ordered = contract.ordered(names, values)
    assert ordered == [float(11 - i) for i in range(12)]
    with pytest.raises(ValueError):
        contract.ordered(names[:-1], values[:-1])
    with pytest.raises(ValueError):
        contract.ordered(names[:-1] + ["FR_HAA"], values)  # duplicate, FL_HAA missing


def test_robot_v2_defaults_and_limits():
    assert [contract.joint_default(n) for n in ("FR_HAA", "RL_HFE", "RR_KFE")] == [0.0, -0.55, 1.10]
    assert [contract.joint_effort_limit(n) for n in ("FL_HAA", "FL_HFE", "FL_KFE")] == [24.8, 48.0, 48.0]
    assert contract.PHYSICS_DT_S * contract.DECIMATION == pytest.approx(0.02)
