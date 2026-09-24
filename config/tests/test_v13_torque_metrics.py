import pytest
import torch

from v13_torque_metrics import summarize_actuation


def test_summary_separates_applied_limit_from_over_limit_request() -> None:
    applied = torch.tensor([[[0.0, 0.0]], [[24.8, 48.0]]])
    requested = torch.tensor([[[0.0, 0.0]], [[100.0, 48.0]]])
    velocity = torch.tensor([[[0.0, 0.0]], [[2.0, 0.0]]])
    alive = torch.ones((2, 1), dtype=torch.bool)
    summary = summarize_actuation(
        applied, requested, velocity, alive, torch.tensor([24.8, 48.0]), physics_dt_s=0.005
    )
    assert summary["peak_applied_nm"] == pytest.approx([24.8, 48.0])
    assert summary["peak_requested_nm"] == pytest.approx([100.0, 48.0])
    assert summary["applied_saturation_fraction"] == pytest.approx(0.5)
    assert summary["over_limit_request_fraction"] == pytest.approx(0.25)
    assert summary["mean_joint_power_w"] == pytest.approx(12.4)
