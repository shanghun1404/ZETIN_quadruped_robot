"""Strict aggregation for repeated locomotion checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


REAR_CORRELATION_WEAK_ABOVE = -0.20
REAR_TRANSITION_WEAK_ABOVE = 0.05
MIN_VALID_FRACTION = 0.90


def aggregate_gate_payloads(
    payloads: list[dict], expected_seeds: tuple[int, ...]
) -> dict[str, object]:
    """Validate and combine one checkpoint's evaluations across exact seeds."""
    rows_by_seed: dict[int, dict[str, object]] = {}
    checkpoints: set[str] = set()
    for payload in payloads:
        metadata = payload.get("metadata", {})
        seed = metadata.get("seed")
        checkpoint = metadata.get("checkpoint")
        if not isinstance(seed, int):
            raise ValueError("each payload must contain an integer metadata.seed")
        if seed in rows_by_seed:
            raise ValueError(f"duplicate seed: {seed}")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("each payload must contain metadata.checkpoint")
        checkpoints.add(checkpoint)

        contact_phase = payload.get("phase_metrics", {}).get("contact_phase", {})
        pairs = contact_phase.get("pairs", {})
        for pair_name, pair in pairs.items():
            valid_fraction = pair.get("valid_fraction")
            if not isinstance(valid_fraction, (int, float)) or valid_fraction < MIN_VALID_FRACTION:
                raise ValueError(
                    f"seed {seed} pair {pair_name} valid_fraction "
                    f"{valid_fraction!r} is below {MIN_VALID_FRACTION}"
                )
        rear = pairs.get("RL-RR", {})
        rear_median = rear.get("median")
        rear_coincidence = contact_phase.get("transition_coincidence", {}).get("rear")
        if not isinstance(rear_median, (int, float)):
            raise ValueError(f"seed {seed} has no finite RL-RR median")
        if not isinstance(rear_coincidence, (int, float)):
            raise ValueError(f"seed {seed} has no rear transition coincidence")

        weak = (
            rear_median > REAR_CORRELATION_WEAK_ABOVE
            or rear_coincidence > REAR_TRANSITION_WEAK_ABOVE
        )
        rows_by_seed[seed] = {
            "seed": seed,
            "rear_contact_correlation_median": float(rear_median),
            "rear_transition_coincidence": float(rear_coincidence),
            "rear_weak": weak,
            "survival": payload.get("metrics", {}).get("survival_fraction"),
            "speed": payload.get("metrics", {}).get("mean_forward_speed_m_s"),
        }

    expected = set(expected_seeds)
    actual = set(rows_by_seed)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError("missing seeds: " + ", ".join(str(seed) for seed in missing))
    extra = sorted(actual - expected)
    if extra:
        raise ValueError("unexpected seeds: " + ", ".join(str(seed) for seed in extra))
    if len(checkpoints) != 1:
        raise ValueError("multiple checkpoints in one aggregate")

    rows = [rows_by_seed[seed] for seed in expected_seeds]
    weak_count = sum(bool(row["rear_weak"]) for row in rows)
    return {
        "checkpoint": next(iter(checkpoints)),
        "seeds": list(expected_seeds),
        "thresholds": {
            "rear_contact_correlation_weak_above": REAR_CORRELATION_WEAK_ABOVE,
            "rear_transition_coincidence_weak_above": REAR_TRANSITION_WEAK_ABOVE,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "confirmation_seed_count": 4,
        },
        "per_seed": rows,
        "weak_seed_count": weak_count,
        "rear_weakness_confirmed": weak_count >= 4,
        "median_across_seeds": {
            "rear_contact_correlation": statistics.median(
                float(row["rear_contact_correlation_median"]) for row in rows
            ),
            "rear_transition_coincidence": statistics.median(
                float(row["rear_transition_coincidence"]) for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-seeds", default="42,43,44,45,46")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.expected_seeds.split(","))
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    result = aggregate_gate_payloads(payloads, seeds)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
