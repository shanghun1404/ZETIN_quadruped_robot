from __future__ import annotations

import torch

from v5_6_math import evaluate_phase_gate


# Evaluation floors are identical to V5.7 so the V5.8 pilot gate stays directly
# comparable to the usage-floor baseline.
EVAL_CONTACT_FLOOR = 0.30
EVAL_SWING_FLOOR = 0.07


def usage_gate_factor(
    duty: torch.Tensor,
    moving: torch.Tensor,
    contact_floor: float = 0.35,
    gain: float = 1.5,
    min_gate: float = 0.0,
) -> torch.Tensor:
    """Multiplicative gate in ``[min_gate, 1]`` for the forward-velocity reward.

    The gate closes as the *worst-used* foot's contact duty falls below
    ``contact_floor``. Applied to the forward linear-velocity tracking reward it
    removes the payoff of the three-leg cheat: a policy that keeps one leg in the
    air can no longer collect forward-tracking reward while it drifts forward.

    When every foot clears the floor the gate is exactly ``1.0`` and the reward
    reduces to the stock ``track_lin_vel_xy_exp`` value, so the only behavioural
    change versus V5.6/V5.7 is confined to abandoned-leg gaits. Standing
    environments (``moving`` False) keep the gate fully open so a planted stance
    under a zero command is never penalized.

    The gate uses a linear (not squared) worst-foot deficit so that a leg used
    only ~20% of the time is cut hard rather than diluted: at ``contact_floor``
    ``0.35``, ``gain`` ``1.5`` a duty of ``0.21`` yields a gate of ``0.40``,
    removing 60% of the forward reward for that step.
    """
    if duty.ndim < 2 or duty.shape[-1] != 4:
        raise ValueError("duty must end with four ordered feet")
    if moving.shape != duty.shape[:-1]:
        raise ValueError("moving mask must match the leading duty dimensions")
    if not 0.0 < contact_floor <= 1.0:
        raise ValueError("contact_floor must lie in (0, 1]")
    if gain < 0.0:
        raise ValueError("gain must be non-negative")
    if not 0.0 <= min_gate <= 1.0:
        raise ValueError("min_gate must lie in [0, 1]")

    duty_float = duty.float()
    contact_deficit = torch.relu((contact_floor - duty_float) / contact_floor)
    worst_deficit = contact_deficit.amax(dim=-1)
    gate = torch.clamp(1.0 - gain * worst_deficit, min_gate, 1.0)
    return torch.where(moving, gate, torch.ones_like(gate))


def summarize_usage_gate_metrics(
    metrics: dict[str, object],
    contact_floor: float = EVAL_CONTACT_FLOOR,
    swing_floor: float = EVAL_SWING_FLOOR,
) -> dict[str, object]:
    """Report the least-used feet, mirroring the V5.7 usage-floor diagnostics."""
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


def evaluate_v5_8_phase_gate(metrics: dict[str, object]) -> dict[str, bool]:
    """V5.8 shares the V5.7 phase-gate floors: min contact 0.30, min swing 0.07."""
    gates = evaluate_phase_gate(metrics)
    gates["phase_min_contact_floor"] = (
        float(metrics["min_foot_contact_fraction"]) >= EVAL_CONTACT_FLOOR
    )
    gates["phase_min_swing_floor"] = (
        float(metrics["min_foot_swing_fraction"]) >= EVAL_SWING_FLOOR
    )
    return gates
