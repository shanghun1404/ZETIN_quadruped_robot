"""Direct foot-contact phase metrics shared by locomotion evaluators."""

from __future__ import annotations

import torch


FEET = ("FL", "FR", "RL", "RR")
PAIRS = (
    ("FL", "FR"),
    ("RL", "RR"),
    ("FL", "RR"),
    ("FR", "RL"),
    ("FL", "RL"),
    ("FR", "RR"),
)


def _finite_summary(values: torch.Tensor) -> dict[str, float | None]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            "median": None,
            "p10": None,
            "p90": None,
            "valid_fraction": 0.0,
        }
    return {
        "median": float(torch.quantile(finite, 0.50)),
        "p10": float(torch.quantile(finite, 0.10)),
        "p90": float(torch.quantile(finite, 0.90)),
        "valid_fraction": float(finite.numel() / values.numel()),
    }


def contact_phase_metrics(contacts: torch.Tensor) -> dict[str, object]:
    """Summarize phase from binary foot contacts shaped ``[time, env, 4]``."""
    if contacts.ndim != 3 or contacts.shape[-1] != 4:
        raise ValueError("contacts must have shape [time, environments, 4]")

    values = contacts.to(dtype=torch.float32)
    centered = values - values.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(centered, dim=0)
    index = {name: i for i, name in enumerate(FEET)}
    pair_metrics = {}
    for left, right in PAIRS:
        left_index, right_index = index[left], index[right]
        denominator = norms[:, left_index] * norms[:, right_index]
        correlation = (
            centered[:, :, left_index] * centered[:, :, right_index]
        ).sum(dim=0) / denominator.clamp(min=1e-8)
        correlation = torch.where(
            denominator > 1e-8,
            correlation,
            torch.full_like(correlation, torch.nan),
        )
        pair_metrics[f"{left}-{right}"] = _finite_summary(correlation)

    before = values[:-1].bool()
    after = values[1:].bool()
    landing = ~before & after
    lifting = before & ~after

    def coincidence(left_index: int, right_index: int) -> float:
        same_event = (
            landing[:, :, left_index] & landing[:, :, right_index]
        ) | (lifting[:, :, left_index] & lifting[:, :, right_index])
        any_event = (
            landing[:, :, left_index]
            | landing[:, :, right_index]
            | lifting[:, :, left_index]
            | lifting[:, :, right_index]
        )
        return float(same_event.sum() / any_event.sum().clamp(min=1))

    return {
        "pairs": pair_metrics,
        "transition_coincidence": {
            "front": coincidence(0, 1),
            "rear": coincidence(2, 3),
        },
    }
