from __future__ import annotations

import torch

from v5_6_math import evaluate_phase_gate


EVAL_CONTACT_FLOOR = 0.30
EVAL_SWING_FLOOR = 0.07


def per_foot_usage_floor_penalty(
    duty: torch.Tensor,
    moving: torch.Tensor,
    contact_floor: float = 0.35,
    swing_floor: float = 0.10,
) -> torch.Tensor:
    if duty.ndim < 2 or duty.shape[-1] != 4:
        raise ValueError("duty must end with four ordered feet")
    if moving.shape != duty.shape[:-1]:
        raise ValueError("moving mask must match the leading duty dimensions")
    if contact_floor <= 0.0 or swing_floor <= 0.0 or contact_floor + swing_floor > 1.0:
        raise ValueError("usage floors must be positive and non-overlapping")

    duty_float = duty.float()
    contact_deficit = torch.relu((contact_floor - duty_float) / contact_floor)
    swing_deficit = torch.relu((swing_floor - (1.0 - duty_float)) / swing_floor)
    penalty = (contact_deficit.square() + swing_deficit.square()).amax(dim=-1)
    return torch.clamp(penalty, 0.0, 1.0) * moving.float()


def summarize_usage_floor_metrics(
    metrics: dict[str, object],
    contact_floor: float = EVAL_CONTACT_FLOOR,
    swing_floor: float = EVAL_SWING_FLOOR,
) -> dict[str, object]:
    contact = metrics["foot_contact_fraction"]
    swing = metrics["foot_swing_fraction"]
    if not isinstance(contact, dict) or not contact:
        raise ValueError("foot_contact_fraction must be a non-empty mapping")
    if not isinstance(swing, dict) or set(swing) != set(contact):
        raise ValueError("foot swing and contact mappings must name the same feet")

    least_contact_foot = min(contact, key=contact.get)
    least_swing_foot = min(swing, key=swing.get)
    min_contact = float(contact[least_contact_foot])
    min_swing = float(swing[least_swing_foot])
    return {
        "min_foot_contact_fraction": min_contact,
        "min_foot_swing_fraction": min_swing,
        "contact_floor_margin": min_contact - contact_floor,
        "swing_floor_margin": min_swing - swing_floor,
        "least_contact_foot": least_contact_foot,
        "least_swing_foot": least_swing_foot,
    }


def evaluate_v5_7_phase_gate(metrics: dict[str, object]) -> dict[str, bool]:
    gates = evaluate_phase_gate(metrics)
    gates["phase_min_contact_floor"] = (
        float(metrics["min_foot_contact_fraction"]) >= EVAL_CONTACT_FLOOR
    )
    gates["phase_min_swing_floor"] = (
        float(metrics["min_foot_swing_fraction"]) >= EVAL_SWING_FLOOR
    )
    return gates
