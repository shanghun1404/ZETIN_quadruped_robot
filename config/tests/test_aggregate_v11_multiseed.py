import pytest

from aggregate_v11_multiseed import aggregate_v11_payloads


EXPECTED_SEEDS = (42, 43, 44, 45, 46)


def _rough_payload(seed: int, survival: float = 0.95, speed: float = 0.4) -> dict:
    pair = {"median": -0.8, "p10": -0.9, "p90": -0.7, "valid_fraction": 1.0}
    return {
        "metadata": {"checkpoint": "/runs/model_3000.pt", "seed": seed},
        "metrics": {
            "survival_fraction": survival,
            "mean_forward_speed_m_s": speed,
        },
        "phase_metrics": {
            "contact_phase": {
                "pairs": {name: dict(pair) for name in ("FL-FR", "RL-RR")}
            }
        },
    }


def _stairs_payload(
    seed: int,
    hardest: int = 5,
    hardest_height_m: float = 0.157,
) -> dict:
    return {
        "metadata": {"checkpoint": "/runs/model_3000.pt", "seed": seed},
        "metrics": {
            "survival_fraction": 0.85,
            "mean_forward_speed_m_s": 0.35,
        },
        "stairs": {
            "climb_success_fraction": 0.6,
            "hardest_level_cleared": hardest,
            "hardest_step_height_cleared_m": hardest_height_m,
        },
        "realistic_step": {"minimum_height_m": 0.15, "first_level": 5},
    }


def test_rough_aggregate_is_strict_across_all_expected_seeds():
    payloads = [_rough_payload(seed) for seed in EXPECTED_SEEDS]

    result = aggregate_v11_payloads(payloads, EXPECTED_SEEDS, scenario="rough")

    assert result["passed"] is True
    assert result["passing_seed_count"] == 5
    assert result["worst_case"]["survival_fraction"] == 0.95
    assert result["worst_case"]["mean_forward_speed_m_s"] == 0.4


def test_one_failed_rough_seed_fails_the_aggregate():
    payloads = [_rough_payload(seed) for seed in EXPECTED_SEEDS]
    payloads[-1]["metrics"]["survival_fraction"] = 0.89

    result = aggregate_v11_payloads(payloads, EXPECTED_SEEDS, scenario="rough")

    assert result["passed"] is False
    assert result["passing_seed_count"] == 4


def test_stairs_aggregate_requires_realistic_level_in_every_seed():
    payloads = [_stairs_payload(seed) for seed in EXPECTED_SEEDS]
    payloads[-1]["stairs"]["hardest_level_cleared"] = 4
    payloads[-1]["stairs"]["hardest_step_height_cleared_m"] = 0.136

    result = aggregate_v11_payloads(payloads, EXPECTED_SEEDS, scenario="stairs")

    assert result["passed"] is False
    assert result["worst_case"]["hardest_level_cleared"] == 4


def test_stairs_aggregate_uses_physical_height_not_claimed_level():
    payloads = [_stairs_payload(seed) for seed in EXPECTED_SEEDS]
    payloads[-1]["stairs"]["hardest_step_height_cleared_m"] = 0.149
    payloads[-1]["realistic_step"]["first_level"] = 0

    result = aggregate_v11_payloads(payloads, EXPECTED_SEEDS, scenario="stairs")

    assert result["passed"] is False
    assert result["per_seed"][-1]["gates"]["stairs_clears_real_step"] is False


def test_missing_seed_is_rejected():
    with pytest.raises(ValueError, match="missing seeds: 46"):
        aggregate_v11_payloads(
            [_rough_payload(seed) for seed in EXPECTED_SEEDS[:-1]],
            EXPECTED_SEEDS,
            scenario="rough",
        )


def test_caller_cannot_weaken_the_required_seed_set():
    with pytest.raises(ValueError, match="exactly 42, 43, 44, 45, 46"):
        aggregate_v11_payloads(
            [_rough_payload(42)],
            (42,),
            scenario="rough",
        )
