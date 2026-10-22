"""V5.14 swing-floor (over-planted foot) penalty.

Why V5.14 exists
----------------
V5.13's dense in-phase penalty flipped the rear pair out of phase
(RL-RR ``+0.349 -> -0.037``) but left the front pair untouched
(``+0.677 -> +0.685``) after 4000 iterations. The same formula and weight fixed
one pair and not the other, which means the front-pair synchrony is not a freely
chosen cheat -- it is forced by posture:

=====  ==============  =============
foot   contact (V5.13)  swing (V5.13)
=====  ==============  =============
FL     0.843            0.157
FR     0.830            0.170
RL     0.367            0.633
RR     0.792            0.208
=====  ==============  =============

Both forefeet stay planted about 84% of the time. Two feet simultaneously in
stance both retract to push the body forward, so their hip velocities share a
sign and the correlation is positive by construction. The robot is not bounding;
it is leaning on its forelimbs while crouched 10 cm below the 0.45 m target.

So the lever is not more phase shaping -- it is making the feet leave the ground
and making the posture penalty actually apply. See ``v5_14_env_cfg`` for the
second half of that fix.
"""

from __future__ import annotations

import torch


def worst_foot_swing_deficit(
    duty: torch.Tensor,
    moving: torch.Tensor,
    swing_floor: float = 0.25,
) -> torch.Tensor:
    """Bounded penalty for the most over-planted foot.

    ``duty`` is the ``[N, 4]`` per-foot contact-duty exponential moving average,
    so ``1 - duty`` is the swing duty. The *worst* (least airborne) foot is used,
    mirroring :func:`v5_12_math.worst_foot_contact_deficit`, which is the one
    V5.12 penalty that demonstrably moved its metric.

    Returns ``[N]`` in ``[0, 1]``: ``0`` when every foot lifts for at least
    ``swing_floor`` of the time, ``1`` when a foot never leaves the ground.
    """
    if duty.shape[-1] != 4:
        raise ValueError("duty must end with four feet")
    if not 0.0 < swing_floor <= 1.0:
        raise ValueError("swing_floor must lie in (0, 1]")

    swing = 1.0 - duty
    deficit = torch.clamp((swing_floor - swing) / swing_floor, min=0.0, max=1.0)
    worst = deficit.amax(dim=-1)
    return worst * moving.to(dtype=worst.dtype)


# Must sit above the measured V5.13 forefoot swing (0.157) or the gate would
# pass the very defect it exists to catch, while staying below the 0.25 training
# floor so evaluation is not stricter than the objective.
EVAL_MIN_SWING = 0.20


def evaluate_v5_14_phase_gate(phase_metrics: dict) -> dict:
    """V5.13's gates plus an explicit minimum-swing floor."""
    from v5_13_math import evaluate_v5_13_phase_gate

    gates = evaluate_v5_13_phase_gate(phase_metrics)
    minimum_swing = phase_metrics.get("min_foot_swing_fraction")
    if minimum_swing is None:
        per_foot = phase_metrics.get("foot_swing_fraction", {})
        minimum_swing = min(per_foot.values()) if per_foot else 0.0
    gates["min_foot_swing"] = minimum_swing >= EVAL_MIN_SWING
    return gates
