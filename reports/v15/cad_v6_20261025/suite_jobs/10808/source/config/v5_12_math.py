"""Pure tensor helpers for the V5.12 anti-cheat gait penalties.

The V5.11 pilot walked at the commanded 0.50 m/s but the measured gait was a
two-cheat exploit rather than a natural walk:

1. **Front-pair bounding.** The two front hips moved *in phase*
   (HFE FL-FR correlation ``+0.70``), i.e. the robot hopped both front legs
   together instead of stepping them alternately. Every natural quadruped walk
   and trot moves the left/right leg of a pair in *opposite* phase; only bound
   and pace gaits synchronize a pair, and both read as the stuttering
   "tuk-tuk" motion the operator rejected.
2. **Rear-left abandonment.** ``RL`` was in contact only ``36%`` of the time
   while ``FL``/``FR``/``RR`` were at ``77-82%`` -- the same abandoned-leg cheat
   that has survived every version since V5.6.

These helpers score exactly those two behaviours as bounded ``[0, 1]``
penalties, so a weighted penalty can never exceed the forward reward and make
standing still (or dying) profitable -- the failure mode that killed the
``action_rate_l2 -0.05`` experiment and the V5.9 fresh symmetric gate.
"""

from __future__ import annotations

import torch


# Ordered foot indices used by every V5 contact tensor: FL, FR, RL, RR.
FL, FR, RL, RR = 0, 1, 2, 3


def pair_synchrony_penalty(
    first_contact: torch.Tensor,
    first_air: torch.Tensor,
    moving: torch.Tensor,
) -> torch.Tensor:
    """Penalize left/right feet of a pair leaving or landing on the same step.

    ``first_contact`` and ``first_air`` are ``[N, 4]`` boolean-ish tensors that
    are true on the step a foot touches down / lifts off. A bound or pace gait
    lands the two front feet (or the two rear feet) on the *same* step, so the
    penalty fires; a walk or trot staggers them, so it stays at zero.

    Only same-pair events are scored -- diagonal timing is left completely free,
    so this does NOT prescribe trot over walk. It only forbids the specific
    front-pair (and rear-pair) synchronization measured in the V5.11 rollout.

    Returns a ``[N]`` tensor in ``[0, 1]``, zeroed wherever ``moving`` is false
    so a standing robot is never penalized for keeping its feet planted.
    """
    if first_contact.shape[-1] != 4 or first_air.shape[-1] != 4:
        raise ValueError("first_contact and first_air must end with four feet")

    contact = first_contact.to(dtype=torch.bool)
    air = first_air.to(dtype=torch.bool)

    events = torch.stack(
        (
            contact[..., FL] & contact[..., FR],
            air[..., FL] & air[..., FR],
            contact[..., RL] & contact[..., RR],
            air[..., RL] & air[..., RR],
        ),
        dim=-1,
    )
    penalty = events.to(dtype=torch.float32).sum(dim=-1) / 4.0
    return penalty * moving.to(dtype=penalty.dtype)


def worst_foot_contact_deficit(
    duty: torch.Tensor,
    moving: torch.Tensor,
    contact_floor: float = 0.35,
) -> torch.Tensor:
    """Bounded penalty for the least-used foot falling below ``contact_floor``.

    ``duty`` is the ``[N, 4]`` per-foot exponential moving average of the binary
    contact flag. The deficit of the *worst* foot is used (not the mean) so the
    penalty targets the one abandoned leg directly instead of being diluted by
    three healthy legs -- the dilution that made the additive V5.7 floor too
    weak to matter.

    Returns a ``[N]`` tensor in ``[0, 1]``: ``0`` when every foot clears the
    floor, ``1`` when a foot never touches the ground at all.
    """
    if duty.shape[-1] != 4:
        raise ValueError("duty must end with four feet")
    if not 0.0 < contact_floor <= 1.0:
        raise ValueError("contact_floor must lie in (0, 1]")

    deficit = torch.clamp((contact_floor - duty) / contact_floor, min=0.0, max=1.0)
    worst = deficit.amax(dim=-1)
    return worst * moving.to(dtype=worst.dtype)


# ---------------------------------------------------------------------------
# Evaluation-side gate
# ---------------------------------------------------------------------------

# A natural walk/trot must keep the left/right legs of a pair out of phase and
# must not abandon a leg. These thresholds are what the V5.11 rollout failed.
EVAL_MAX_PAIR_SYNC = 0.05
EVAL_MIN_CONTACT = 0.30
EVAL_MAX_PAIR_CORRELATION = 0.30


def evaluate_v5_12_phase_gate(phase_metrics: dict) -> dict:
    """Gate the two cheats V5.12 targets, on top of the shared phase floors."""
    correlations = phase_metrics.get("same_joint_pair_correlation", {})
    hfe = correlations.get("HFE", {}) if isinstance(correlations, dict) else {}
    front_pair = hfe.get("FL-FR")
    rear_pair = hfe.get("RL-RR")

    gates = {
        "min_foot_contact": phase_metrics.get("min_foot_contact_fraction", 0.0)
        >= EVAL_MIN_CONTACT,
    }
    if front_pair is not None:
        gates["front_pair_out_of_phase"] = front_pair <= EVAL_MAX_PAIR_CORRELATION
    if rear_pair is not None:
        gates["rear_pair_out_of_phase"] = rear_pair <= EVAL_MAX_PAIR_CORRELATION
    return gates
