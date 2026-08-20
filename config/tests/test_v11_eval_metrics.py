import pytest
import torch

from v11_eval_metrics import (
    minimum_level_for_step_height,
    peak_height_after_approach,
    rough_gates,
    stair_level_assignments,
    stairs_gates,
)


def test_realistic_stair_level_is_first_level_at_or_above_15_cm():
    assert minimum_level_for_step_height((0.05, 0.20), levels=8, minimum_m=0.15) == 5


def test_rough_gate_requires_survival_speed_and_valid_direct_phase():
    passing = {
        "survival_fraction": 0.90,
        "mean_forward_speed_m_s": 0.30,
        "direct_phase_valid_fraction": 0.90,
    }

    assert all(rough_gates(passing).values())
    assert rough_gates({**passing, "survival_fraction": 0.89})["rough_survival"] is False
    assert rough_gates({**passing, "mean_forward_speed_m_s": 0.29})["rough_speed"] is False
    assert rough_gates({**passing, "direct_phase_valid_fraction": 0.89})[
        "direct_phase_valid"
    ] is False


def test_stairs_gate_requires_real_step_not_only_easy_levels():
    passing_metrics = {"survival_fraction": 0.80}
    passing_stairs = {
        "climb_success_fraction": 0.50,
        "hardest_level_cleared": 5,
        "hardest_step_height_cleared_m": 0.15,
    }

    assert all(
        stairs_gates(
            passing_metrics,
            passing_stairs,
            minimum_step_height_m=0.15,
        ).values()
    )
    failed = stairs_gates(
        passing_metrics,
        {**passing_stairs, "hardest_step_height_cleared_m": 0.149},
        minimum_step_height_m=0.15,
    )
    assert failed["stairs_clears_real_step"] is False


def test_stair_assignment_covers_each_difficulty_equally():
    assert stair_level_assignments(num_envs=16, levels=8) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]


def test_stair_assignment_rejects_too_few_environments():
    with pytest.raises(ValueError, match="at least 8"):
        stair_level_assignments(num_envs=7, levels=8)


def test_peak_height_keeps_a_climb_when_fast_robot_has_already_descended():
    height_gain = torch.tensor(
        [
            [0.00, 0.00],
            [0.10, 0.05],
            [0.24, 0.08],
            [-0.05, 0.02],
        ]
    )
    progress = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.5, 2.5],
            [7.0, 7.0],
        ]
    )
    valid = torch.ones_like(height_gain, dtype=torch.bool)

    peak = peak_height_after_approach(
        height_gain,
        progress,
        valid,
        approach_distance_m=2.1,
    )

    assert torch.equal(peak, torch.tensor([0.24, 0.08]))
