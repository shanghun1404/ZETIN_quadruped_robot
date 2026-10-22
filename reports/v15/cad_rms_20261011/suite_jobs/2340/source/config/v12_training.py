"""Pure loss, metric, and stopping helpers for V12 distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from v12_vision_student import distillation_loss, reconstruct_teacher_observation


def distillation_metrics(
    predicted_scan: torch.Tensor,
    target_scan: torch.Tensor,
    student_action: torch.Tensor,
    teacher_action: torch.Tensor,
) -> dict[str, float]:
    """Return literal scan/action MAE and RMSE values."""

    if predicted_scan.shape != target_scan.shape:
        raise ValueError("predicted and target scan shapes must match")
    if student_action.shape != teacher_action.shape:
        raise ValueError("student and teacher action shapes must match")
    scan_error = predicted_scan.detach().float() - target_scan.detach().float()
    action_error = student_action.detach().float() - teacher_action.detach().float()
    return {
        "scan_mae": float(scan_error.abs().mean()),
        "scan_rmse": math.sqrt(float(scan_error.square().mean())),
        "action_mae": float(action_error.abs().mean()),
        "action_rmse": math.sqrt(float(action_error.square().mean())),
    }


def compute_distillation_batch(
    model,
    frozen_actor,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run one differentiable pseudo-scan and frozen-action distillation batch."""

    predicted_scan = model(batch["depth"])
    student_observation = reconstruct_teacher_observation(
        batch["proprioception"],
        predicted_scan,
    )
    student_action = frozen_actor(student_observation)
    losses = distillation_loss(
        predicted_scan,
        batch["teacher_scan"],
        student_action,
        batch["teacher_action"],
    )
    return {
        **losses,
        "predicted_scan": predicted_scan,
        "student_action": student_action,
    }


def checkpoint_is_finite(checkpoint: dict[str, object]) -> bool:
    """Return true only for a non-empty, entirely finite model state dictionary."""

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        return False
    return all(
        isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all().item()
        for tensor in state.values()
    )


@dataclass
class BestMetricTracker:
    """Retain a resume baseline until a strictly lower total loss appears."""

    best_metrics: dict[str, float]
    epoch: int = 0

    def __post_init__(self) -> None:
        if "total_loss" not in self.best_metrics:
            raise ValueError("best metrics must include total_loss")
        if not math.isfinite(float(self.best_metrics["total_loss"])):
            raise ValueError("best total_loss must be finite")
        self.best_epoch = self.epoch

    def consider(self, metrics: dict[str, float], *, epoch: int) -> bool:
        value = float(metrics.get("total_loss", float("nan")))
        if not math.isfinite(value):
            raise ValueError("candidate total_loss must be finite")
        if value >= float(self.best_metrics["total_loss"]):
            return False
        self.best_metrics = metrics
        self.best_epoch = epoch
        return True


@dataclass
class EarlyStopper:
    """Stop after a fixed number of consecutive non-improving validation epochs."""

    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")

    def update(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError("early-stopping value must be finite")
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
