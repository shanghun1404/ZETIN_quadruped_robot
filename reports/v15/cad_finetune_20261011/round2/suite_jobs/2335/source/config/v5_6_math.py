from __future__ import annotations

from itertools import combinations

import torch


PHASE_GATE_THRESHOLDS = {
    "max_all_feet_stance_fraction": 0.80,
    "min_pair_contact_disagreement_fraction": 0.10,
    "min_foot_swing_fraction": 0.05,
    "max_contact_duty_spread": 0.20,
    "max_liftoff_burst_gt2_fraction": 0.02,
    "max_touchdown_burst_gt2_fraction": 0.02,
}


def _moving_float(moving: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if moving.shape != reference.shape[:-1]:
        raise ValueError("moving mask must match the leading contact dimensions")
    return moving.to(dtype=torch.float32)


def all_feet_stance_penalty(
    contacts: torch.Tensor,
    moving: torch.Tensor,
) -> torch.Tensor:
    if contacts.ndim < 2 or contacts.shape[-1] != 4:
        raise ValueError("contacts must end with four ordered feet")
    all_stance = contacts.bool().all(dim=-1).to(dtype=torch.float32)
    return all_stance * _moving_float(moving, contacts)


def contact_transition_burst_penalty(
    first_contact: torch.Tensor,
    first_air: torch.Tensor,
    moving: torch.Tensor,
    max_per_direction: int = 2,
) -> torch.Tensor:
    if first_contact.shape != first_air.shape or first_contact.shape[-1] != 4:
        raise ValueError("transition tensors must have identical four-foot shapes")
    if not 0 <= max_per_direction < first_contact.shape[-1]:
        raise ValueError("max_per_direction must be between zero and three")
    denominator = float(first_contact.shape[-1] - max_per_direction)
    touchdown = torch.relu(first_contact.float().sum(dim=-1) - max_per_direction) / denominator
    liftoff = torch.relu(first_air.float().sum(dim=-1) - max_per_direction) / denominator
    penalty = torch.maximum(touchdown.square(), liftoff.square())
    return torch.clamp(penalty, 0.0, 1.0) * _moving_float(moving, first_contact)


def foot_usage_imbalance_penalty(
    duty: torch.Tensor,
    moving: torch.Tensor,
) -> torch.Tensor:
    if duty.ndim < 2 or duty.shape[-1] != 4:
        raise ValueError("duty must end with four ordered feet")
    variance = duty.float().var(dim=-1, unbiased=False)
    penalty = torch.clamp(variance / 0.25, 0.0, 1.0)
    return penalty * _moving_float(moving, duty)


def summarize_phase_traces(
    contacts: torch.Tensor,
    first_contact: torch.Tensor,
    first_air: torch.Tensor,
    foot_names: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR"),
) -> dict[str, object]:
    if contacts.ndim != 3 or contacts.shape[-1] != 4:
        raise ValueError("contacts must have shape [steps, environments, four feet]")
    if first_contact.shape != contacts.shape or first_air.shape != contacts.shape:
        raise ValueError("transition traces must match contact traces")
    if len(foot_names) != 4:
        raise ValueError("exactly four foot names are required")

    flat_contacts = contacts.bool().reshape(-1, 4)
    flat_first_contact = first_contact.bool().reshape(-1, 4)
    flat_first_air = first_air.bool().reshape(-1, 4)
    duty = flat_contacts.float().mean(dim=0)
    swing = 1.0 - duty
    contact_count = flat_contacts.sum(dim=-1)
    pair_disagreement = torch.stack(
        [
            (flat_contacts[:, left] != flat_contacts[:, right]).float().mean()
            for left, right in combinations(range(4), 2)
        ]
    ).mean()

    return {
        "foot_contact_fraction": {
            name: float(duty[index].item()) for index, name in enumerate(foot_names)
        },
        "foot_swing_fraction": {
            name: float(swing[index].item()) for index, name in enumerate(foot_names)
        },
        "contact_count_fraction": {
            str(count): float((contact_count == count).float().mean().item())
            for count in range(5)
        },
        "all_feet_stance_fraction": float((contact_count == 4).float().mean().item()),
        "mean_pair_contact_disagreement_fraction": float(pair_disagreement.item()),
        "contact_duty_spread": float((duty.max() - duty.min()).item()),
        "liftoff_burst_gt2_fraction": float(
            (flat_first_air.sum(dim=-1) > 2).float().mean().item()
        ),
        "touchdown_burst_gt2_fraction": float(
            (flat_first_contact.sum(dim=-1) > 2).float().mean().item()
        ),
    }


def summarize_action_correlations(
    actions: torch.Tensor,
    joint_names: tuple[str, ...] | list[str],
) -> dict[str, object]:
    if actions.ndim < 2 or actions.shape[-1] != len(joint_names):
        raise ValueError("action width must match joint_names")
    flat = actions.float().reshape(-1, actions.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=0)
    grouped: dict[str, list[tuple[str, int]]] = {}
    for index, name in enumerate(joint_names):
        suffix = name.rsplit("_", 1)[-1]
        if suffix not in {"HAA", "HFE", "KFE"}:
            continue
        grouped.setdefault(suffix, []).append((name.split("_", 1)[0], index))
    if set(grouped) != {"HAA", "HFE", "KFE"} or any(
        len(entries) != 4 for entries in grouped.values()
    ):
        raise ValueError("joint names must contain four HAA, HFE, and KFE channels")

    result: dict[str, dict[str, float]] = {}
    correlations: list[float] = []
    for suffix in ("HAA", "HFE", "KFE"):
        entries = sorted(grouped[suffix])
        pairs: dict[str, float] = {}
        for left, right in combinations(entries, 2):
            left_leg, left_index = left
            right_leg, right_index = right
            denominator = norms[left_index] * norms[right_index]
            if denominator <= 1.0e-8:
                correlation = 0.0
            else:
                value = torch.dot(centered[:, left_index], centered[:, right_index])
                correlation = float(torch.clamp(value / denominator, -1.0, 1.0).item())
            pairs[f"{left_leg}-{right_leg}"] = correlation
            correlations.append(correlation)
        result[suffix] = pairs
    return {
        "same_joint_pair_correlation": result,
        "mean_same_joint_pair_correlation": float(sum(correlations) / len(correlations)),
        "max_same_joint_pair_correlation": float(max(correlations)),
    }


def evaluate_phase_gate(metrics: dict[str, object]) -> dict[str, bool]:
    swing = metrics["foot_swing_fraction"]
    if not isinstance(swing, dict) or not swing:
        raise ValueError("foot_swing_fraction must be a non-empty mapping")
    return {
        "phase_all_stance": float(metrics["all_feet_stance_fraction"])
        <= PHASE_GATE_THRESHOLDS["max_all_feet_stance_fraction"],
        "phase_contact_disagreement": float(
            metrics["mean_pair_contact_disagreement_fraction"]
        )
        >= PHASE_GATE_THRESHOLDS["min_pair_contact_disagreement_fraction"],
        "phase_min_swing": min(float(value) for value in swing.values())
        >= PHASE_GATE_THRESHOLDS["min_foot_swing_fraction"],
        "phase_duty_balance": float(metrics["contact_duty_spread"])
        <= PHASE_GATE_THRESHOLDS["max_contact_duty_spread"],
        "phase_liftoff_burst": float(metrics["liftoff_burst_gt2_fraction"])
        <= PHASE_GATE_THRESHOLDS["max_liftoff_burst_gt2_fraction"],
        "phase_touchdown_burst": float(metrics["touchdown_burst_gt2_fraction"])
        <= PHASE_GATE_THRESHOLDS["max_touchdown_burst_gt2_fraction"],
    }
