import pytest
import torch

from v12_eval_metrics import (
    rough_gate,
    stair_traversal_mask,
    summarize_rollout,
)


def test_rollout_summary_masks_samples_after_each_environment_dies():
    valid = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, True],
        ]
    )
    alive_at_end = torch.tensor([False, True])
    speed = torch.tensor([[1.0, 3.0], [2.0, 4.0], [100.0, 5.0]])
    scan_mae = torch.tensor([[0.1, 0.3], [0.2, 0.4], [9.0, 0.5]])
    action_mae = torch.tensor([[0.2, 0.4], [0.3, 0.5], [8.0, 0.6]])

    summary = summarize_rollout(
        valid,
        alive_at_end,
        speed,
        scan_mae,
        action_mae,
    )

    assert summary["survival"] == pytest.approx(0.5)
    assert summary["mean_forward_speed_m_s"] == pytest.approx(3.0)
    assert summary["scan_mae"] == pytest.approx(0.3)
    assert summary["action_mae"] == pytest.approx(0.4)
    assert summary["valid_samples"] == 5


@pytest.mark.parametrize(
    ("survival", "speed", "expected"),
    [
        (0.80, 0.25, True),
        (0.79, 0.25, False),
        (0.80, 0.24, False),
    ],
)
def test_rough_gate_uses_the_literal_operational_boundaries(survival, speed, expected):
    assert rough_gate(survival, speed) is expected


def test_stair_traversal_uses_survival_progress_and_two_physical_steps():
    alive = torch.tensor([True, True, False, True])
    progress = torch.tensor([2.1, 2.09, 3.0, 3.0])
    vertical = torch.tensor([0.314, 1.0, 1.0, 0.313])

    ascent = stair_traversal_mask(
        "stairs_up",
        alive,
        progress,
        vertical,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    descent = stair_traversal_mask(
        "stairs_down",
        alive,
        progress,
        vertical,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )

    assert torch.equal(ascent, torch.tensor([True, False, False, False]))
    assert torch.equal(descent, ascent)


def test_stair_traversal_rejects_an_unknown_scenario():
    with pytest.raises(ValueError, match="unsupported stair scenario"):
        stair_traversal_mask(
            "flat",
            torch.ones(1, dtype=torch.bool),
            torch.ones(1),
            torch.ones(1),
            approach_distance_m=0.1,
            step_height_m=0.1,
        )
