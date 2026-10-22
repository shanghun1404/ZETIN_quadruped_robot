"""V5.13 continuous in-phase penalty for the front/rear leg pairs.

Why V5.12 failed and this replaces it
-------------------------------------
V5.12 penalized the *discrete event* "both feet of a pair touch down on the same
control step". After 2000 iterations at weight -2.0 the front-pair HFE
correlation was unchanged (+0.698 -> +0.677) and the episodic penalty was
negligible (-0.0011 -> -0.0404): the policy simply offset its two front
footfalls by one or two control steps while continuing to swing both front legs
as a single unit. The cheat is a *continuous phase relationship*, so an
event-based penalty has a trivial loophole and almost no density.

Measured signature of the cheat (v5.12 ``model_1999`` rollout, 300 steps x 64
envs, HFE joint velocities):

===========================  =========  ========  =======
quantity                     front      rear      diagonal
===========================  =========  ========  =======
same-direction fraction      0.749      0.765     0.638
mean ``relu(v_a * v_b)``     7.21       1.65      2.04
===========================  =========  ========  =======

The front pair's in-phase coupling is over four times the rear and diagonal
values -- it is the outlier, exactly the bounding motion seen on video.

The penalty
-----------
``relu(v_a * v_b)`` is large only when the two legs of a pair swing hard *in the
same direction at the same instant*. It is zero whenever they move oppositely
(a walk or trot) and small during gentle shared stance, so it targets the
violent in-phase bounce rather than banning normal stance overlap. It is
evaluated every step, so there is no timing loophole.

Normalization uses ``vel_scale = 4.0 rad/s``, chosen from the measured HFE speed
distribution (mean 1.82, p90 5.48, p95 6.74 rad/s): the current cheat scores
about 0.45 and only the top few percent of in-phase bursts saturate at 1.0.
"""

from __future__ import annotations

import torch


# Ordered HFE joints: FL, FR, RL, RR.
FL, FR, RL, RR = 0, 1, 2, 3

# Chosen from the measured HFE velocity distribution -- see the module docstring.
DEFAULT_VEL_SCALE = 4.0


def pair_in_phase_penalty(
    hfe_velocity: torch.Tensor,
    moving: torch.Tensor,
    vel_scale: float = DEFAULT_VEL_SCALE,
) -> torch.Tensor:
    """Penalize the left/right legs of a pair swinging in the same direction.

    ``hfe_velocity`` is ``[N, 4]`` ordered FL, FR, RL, RR. Returns ``[N]`` in
    ``[0, 1]``: the mean over the front pair and the rear pair of the clamped,
    normalized positive velocity product. Diagonal timing is untouched, so a
    trot and a walk are both free; only same-pair synchrony is priced.

    Zeroed wherever ``moving`` is false so a standing robot is never penalized.
    """
    if hfe_velocity.shape[-1] != 4:
        raise ValueError("hfe_velocity must end with four HFE joints")
    if vel_scale <= 0.0:
        raise ValueError("vel_scale must be positive")

    front = hfe_velocity[..., FL] * hfe_velocity[..., FR]
    rear = hfe_velocity[..., RL] * hfe_velocity[..., RR]
    denominator = vel_scale * vel_scale
    front_penalty = torch.clamp(torch.relu(front) / denominator, max=1.0)
    rear_penalty = torch.clamp(torch.relu(rear) / denominator, max=1.0)
    penalty = 0.5 * (front_penalty + rear_penalty)
    return penalty * moving.to(dtype=penalty.dtype)


# ---------------------------------------------------------------------------
# Evaluation-side gate
# ---------------------------------------------------------------------------

EVAL_MAX_PAIR_CORRELATION = 0.30
EVAL_MIN_CONTACT = 0.30


def evaluate_v5_13_phase_gate(phase_metrics: dict) -> dict:
    """Gate the two cheats V5.13 targets.

    ``min_foot_contact_fraction`` is not emitted by every evaluator variant, so
    it is derived from the per-foot table when absent. V5.12's gate defaulted a
    missing key to 0.0 and reported a false failure; this does not.
    """
    correlations = phase_metrics.get("same_joint_pair_correlation", {})
    hfe = correlations.get("HFE", {}) if isinstance(correlations, dict) else {}

    minimum_contact = phase_metrics.get("min_foot_contact_fraction")
    if minimum_contact is None:
        per_foot = phase_metrics.get("foot_contact_fraction", {})
        minimum_contact = min(per_foot.values()) if per_foot else 0.0

    gates = {"min_foot_contact": minimum_contact >= EVAL_MIN_CONTACT}
    for label, key in (("front_pair_out_of_phase", "FL-FR"), ("rear_pair_out_of_phase", "RL-RR")):
        value = hfe.get(key)
        if value is not None:
            gates[label] = value <= EVAL_MAX_PAIR_CORRELATION
    return gates
