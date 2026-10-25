from __future__ import annotations

import torch

from v5_6_math import evaluate_phase_gate


# Evaluation floors identical to V5.7/V5.8 so the V5.9 pilot gate stays directly
# comparable to the usage-floor and usage-gate baselines.
EVAL_CONTACT_FLOOR = 0.30
EVAL_SWING_FLOOR = 0.07


def usage_gate_factor(
    duty: torch.Tensor,
    moving: torch.Tensor,
    contact_floor: float = 0.35,
    swing_floor: float = 0.10,
    gain: float = 1.5,
    min_gate: float = 0.0,
) -> torch.Tensor:
    """Symmetric multiplicative gate in ``[min_gate, 1]`` for the forward reward.

    V5.8 gated the forward-velocity reward only on the *contact* side, so an
    abandoned leg (duty too low) shrank the reward. That broke the three-leg
    cheat (rear-left contact rose 0.21 -> 0.43) but left the front feet planted
    ~93% of the time, so the minimum swing floor and duty balance still failed.

    V5.9 adds the *swing* side. The gate now closes when the worst foot is either
    used too little (contact below ``contact_floor``) **or** planted too much
    (swing below ``swing_floor``). A foot must therefore spend a real fraction of
    its time both on and off the ground, so the forward reward requires every leg
    to actually walk. The two floors do not overlap, so at most one deficit is
    active per foot; the healthy duty band is ``[contact_floor, 1 - swing_floor]``.

    When every foot lies in the healthy band the gate is exactly ``1.0`` and the
    reward reduces to the stock ``track_lin_vel_xy_exp`` value. Standing
    environments (``moving`` False) keep the gate fully open.

    The deficit is linear so an over-planted foot is cut hard: at ``swing_floor``
    ``0.10``, ``gain`` ``1.5`` a foot planted 93% of the time (swing 0.07) yields
    a gate of about ``0.50``, removing half the forward reward until it lifts.
    """
    if duty.ndim < 2 or duty.shape[-1] != 4:
        raise ValueError("duty must end with four ordered feet")
    if moving.shape != duty.shape[:-1]:
        raise ValueError("moving mask must match the leading duty dimensions")
    if not 0.0 < contact_floor <= 1.0:
        raise ValueError("contact_floor must lie in (0, 1]")
    if not 0.0 < swing_floor <= 1.0:
        raise ValueError("swing_floor must lie in (0, 1]")
    if contact_floor + swing_floor > 1.0:
        raise ValueError("contact and swing floors must not overlap")
    if gain < 0.0:
        raise ValueError("gain must be non-negative")
    if not 0.0 <= min_gate <= 1.0:
        raise ValueError("min_gate must lie in [0, 1]")

    duty_float = duty.float()
    contact_deficit = torch.relu((contact_floor - duty_float) / contact_floor)
    swing_deficit = torch.relu((swing_floor - (1.0 - duty_float)) / swing_floor)
    worst_deficit = (contact_deficit + swing_deficit).amax(dim=-1)
    gate = torch.clamp(1.0 - gain * worst_deficit, min_gate, 1.0)
    return torch.where(moving, gate, torch.ones_like(gate))


def summarize_usage_gate_metrics(
    metrics: dict[str, object],
    contact_floor: float = EVAL_CONTACT_FLOOR,
    swing_floor: float = EVAL_SWING_FLOOR,
) -> dict[str, object]:
    """Report the least-used feet, mirroring the V5.7/V5.8 usage diagnostics."""
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


def evaluate_v5_9_phase_gate(metrics: dict[str, object]) -> dict[str, bool]:
    """V5.9 shares the V5.7/V5.8 phase-gate floors: min contact 0.30, min swing 0.07."""
    gates = evaluate_phase_gate(metrics)
    gates["phase_min_contact_floor"] = (
        float(metrics["min_foot_contact_fraction"]) >= EVAL_CONTACT_FLOOR
    )
    gates["phase_min_swing_floor"] = (
        float(metrics["min_foot_swing_fraction"]) >= EVAL_SWING_FLOOR
    )
    return gates
