"""Strict all-seed aggregation for V11 rough and stairs evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v11_eval_metrics import (
    MIN_REALISTIC_STEP_HEIGHT_M,
    rough_gates,
    stairs_gates,
)


REQUIRED_SEEDS = (42, 43, 44, 45, 46)


def aggregate_v11_payloads(
    payloads: list[dict],
    expected_seeds: tuple[int, ...],
    scenario: str,
) -> dict[str, object]:
    """Require one matching checkpoint result for every expected seed."""
    if expected_seeds != REQUIRED_SEEDS:
        raise ValueError("expected seeds must be exactly 42, 43, 44, 45, 46")
    if scenario not in {"rough", "stairs"}:
        raise ValueError(f"unsupported scenario: {scenario}")

    by_seed: dict[int, dict] = {}
    checkpoints: set[str] = set()
    for payload in payloads:
        metadata = payload.get("metadata", {})
        seed = metadata.get("seed")
        checkpoint = metadata.get("checkpoint")
        if not isinstance(seed, int):
            raise ValueError("each payload must contain an integer metadata.seed")
        if seed in by_seed:
            raise ValueError(f"duplicate seed: {seed}")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("each payload must contain metadata.checkpoint")
        checkpoints.add(checkpoint)
        by_seed[seed] = payload

    expected = set(expected_seeds)
    actual = set(by_seed)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError("missing seeds: " + ", ".join(map(str, missing)))
    extra = sorted(actual - expected)
    if extra:
        raise ValueError("unexpected seeds: " + ", ".join(map(str, extra)))
    if len(checkpoints) != 1:
        raise ValueError("multiple checkpoints in one aggregate")

    rows = []
    for seed in expected_seeds:
        payload = by_seed[seed]
        metrics = dict(payload.get("metrics", {}))
        if scenario == "rough":
            pairs = (
                payload.get("phase_metrics", {})
                .get("contact_phase", {})
                .get("pairs", {})
            )
            if not pairs:
                raise ValueError(f"seed {seed} has no direct contact phase pairs")
            metrics["direct_phase_valid_fraction"] = min(
                float(pair["valid_fraction"]) for pair in pairs.values()
            )
            gates = rough_gates(metrics)
            row = {
                "seed": seed,
                "survival_fraction": float(metrics["survival_fraction"]),
                "mean_forward_speed_m_s": float(
                    metrics["mean_forward_speed_m_s"]
                ),
                "direct_phase_valid_fraction": float(
                    metrics["direct_phase_valid_fraction"]
                ),
                "gates": gates,
                "passed": all(gates.values()),
            }
        else:
            stairs = payload.get("stairs", {})
            realistic_step = payload.get("realistic_step", {})
            realistic_level = realistic_step.get("first_level")
            if not isinstance(realistic_level, int):
                raise ValueError(f"seed {seed} has no realistic stair level")
            minimum_height = realistic_step.get("minimum_height_m")
            if minimum_height != MIN_REALISTIC_STEP_HEIGHT_M:
                raise ValueError(
                    f"seed {seed} realistic minimum must be "
                    f"{MIN_REALISTIC_STEP_HEIGHT_M} m"
                )
            gates = stairs_gates(
                metrics,
                stairs,
                minimum_step_height_m=MIN_REALISTIC_STEP_HEIGHT_M,
            )
            row = {
                "seed": seed,
                "survival_fraction": float(metrics["survival_fraction"]),
                "mean_forward_speed_m_s": float(
                    metrics["mean_forward_speed_m_s"]
                ),
                "climb_success_fraction": float(
                    stairs["climb_success_fraction"]
                ),
                "hardest_level_cleared": int(stairs["hardest_level_cleared"]),
                "hardest_step_height_cleared_m": float(
                    stairs["hardest_step_height_cleared_m"]
                ),
                "realistic_level": realistic_level,
                "gates": gates,
                "passed": all(gates.values()),
            }
        rows.append(row)

    worst_case = {
        "survival_fraction": min(row["survival_fraction"] for row in rows),
        "mean_forward_speed_m_s": min(
            row["mean_forward_speed_m_s"] for row in rows
        ),
    }
    if scenario == "rough":
        worst_case["direct_phase_valid_fraction"] = min(
            row["direct_phase_valid_fraction"] for row in rows
        )
    else:
        worst_case["climb_success_fraction"] = min(
            row["climb_success_fraction"] for row in rows
        )
        worst_case["hardest_level_cleared"] = min(
            row["hardest_level_cleared"] for row in rows
        )
        worst_case["hardest_step_height_cleared_m"] = min(
            row["hardest_step_height_cleared_m"] for row in rows
        )

    passing_seed_count = sum(bool(row["passed"]) for row in rows)
    return {
        "scenario": scenario,
        "checkpoint": next(iter(checkpoints)),
        "seeds": list(expected_seeds),
        "per_seed": rows,
        "passing_seed_count": passing_seed_count,
        "required_seed_count": len(expected_seeds),
        "worst_case": worst_case,
        "passed": passing_seed_count == len(expected_seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--scenario", choices=("rough", "stairs"), required=True)
    parser.add_argument("--expected-seeds", default="42,43,44,45,46")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected_seeds = tuple(int(value) for value in args.expected_seeds.split(","))
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    result = aggregate_v11_payloads(payloads, expected_seeds, args.scenario)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
