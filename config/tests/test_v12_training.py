import math

import pytest
import torch
from torch import nn

from v12_training import (
    BestMetricTracker,
    EarlyStopper,
    checkpoint_is_finite,
    compute_distillation_batch,
    distillation_metrics,
)


def test_distillation_metrics_use_literal_elementwise_errors():
    predicted_scan = torch.tensor([[0.0, 2.0]])
    target_scan = torch.tensor([[1.0, 0.0]])
    student_action = torch.tensor([[1.0, 4.0]])
    teacher_action = torch.tensor([[0.0, 0.0]])

    metrics = distillation_metrics(
        predicted_scan,
        target_scan,
        student_action,
        teacher_action,
    )

    assert metrics["scan_mae"] == pytest.approx(1.5)
    assert metrics["scan_rmse"] == pytest.approx(math.sqrt(2.5))
    assert metrics["action_mae"] == pytest.approx(2.5)
    assert metrics["action_rmse"] == pytest.approx(math.sqrt(8.5))


def test_distillation_metrics_are_zero_for_exact_targets():
    scan = torch.randn((3, 187))
    action = torch.randn((3, 12))

    assert distillation_metrics(scan, scan, action, action) == {
        "scan_mae": 0.0,
        "scan_rmse": 0.0,
        "action_mae": 0.0,
        "action_rmse": 0.0,
    }


def test_checkpoint_finite_scan_checks_every_model_tensor():
    assert checkpoint_is_finite(
        {"model_state_dict": {"a": torch.ones(2), "b": torch.zeros(3)}}
    )
    assert not checkpoint_is_finite(
        {"model_state_dict": {"a": torch.tensor([float("nan")])}}
    )
    assert not checkpoint_is_finite({"wrong_key": {"a": torch.ones(1)}})


def test_early_stopper_waits_for_five_consecutive_non_improvements():
    stopper = EarlyStopper(patience=5)

    assert not stopper.update(1.0)
    assert not stopper.update(0.9)
    for value in (0.91, 0.92, 0.93, 0.94):
        assert not stopper.update(value)
    assert stopper.update(0.95)
    assert stopper.best == pytest.approx(0.9)
    assert stopper.bad_epochs == 5


def test_compute_distillation_batch_supports_real_gradient_optimization():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 187), nn.Tanh())
    actor = nn.Sequential(nn.Linear(235, 12))
    actor.requires_grad_(False)
    batch = {
        "depth": torch.full((8, 1, 2, 2), 0.25),
        "proprioception": torch.zeros((8, 48)),
        "teacher_scan": torch.zeros((8, 187)),
        "teacher_action": torch.zeros((8, 12)),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    initial = float(compute_distillation_batch(model, actor, batch)["total"])
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        result = compute_distillation_batch(model, actor, batch)
        result["total"].backward()
        optimizer.step()
    final = float(compute_distillation_batch(model, actor, batch)["total"])

    assert final < initial * 0.2


def test_best_metric_tracker_preserves_the_resume_baseline_until_loss_improves():
    baseline = {"total_loss": 0.2, "scan_mae": 0.1}
    tracker = BestMetricTracker(baseline, epoch=0)

    assert not tracker.consider({"total_loss": 0.3, "scan_mae": 0.05}, epoch=1)
    assert tracker.best_epoch == 0
    assert tracker.best_metrics is baseline
    improved = {"total_loss": 0.19, "scan_mae": 0.09}
    assert tracker.consider(improved, epoch=2)
    assert tracker.best_epoch == 2
    assert tracker.best_metrics is improved
