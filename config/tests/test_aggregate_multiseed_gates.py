import pytest

from aggregate_multiseed_gates import aggregate_gate_payloads


EXPECTED_SEEDS = (42, 43, 44, 45, 46)


def _payload(seed: int, rear_median: float, rear_coincidence: float = 0.01) -> dict:
    pair = {
        "median": rear_median,
        "p10": rear_median - 0.1,
        "p90": rear_median + 0.1,
        "valid_fraction": 1.0,
    }
    pairs = {
        name: dict(pair)
        for name in ("FL-FR", "RL-RR", "FL-RR", "FR-RL", "FL-RL", "FR-RR")
    }
    return {
        "metadata": {"checkpoint": "/runs/model_2000.pt", "seed": seed},
        "metrics": {"survival_fraction": 1.0, "mean_forward_speed_m_s": 0.5},
        "phase_metrics": {
            "contact_phase": {
                "pairs": pairs,
                "transition_coincidence": {"front": 0.01, "rear": rear_coincidence},
            }
        },
    }


def test_four_weak_seeds_confirm_the_rear_phase_weakness():
    payloads = [
        _payload(42, -0.10),
        _payload(43, -0.15),
        _payload(44, -0.05),
        _payload(45, -0.19),
        _payload(46, -0.70),
    ]

    result = aggregate_gate_payloads(payloads, EXPECTED_SEEDS)

    assert result["rear_weakness_confirmed"] is True
    assert result["weak_seed_count"] == 4
    assert result["checkpoint"] == "/runs/model_2000.pt"
    assert result["seeds"] == [42, 43, 44, 45, 46]


def test_three_weak_seeds_do_not_confirm_the_rear_phase_weakness():
    payloads = [
        _payload(42, -0.10),
        _payload(43, -0.15),
        _payload(44, -0.05),
        _payload(45, -0.50),
        _payload(46, -0.70),
    ]

    result = aggregate_gate_payloads(payloads, EXPECTED_SEEDS)

    assert result["rear_weakness_confirmed"] is False
    assert result["weak_seed_count"] == 3


def test_transition_coincidence_can_confirm_a_weak_seed():
    payloads = [
        _payload(seed, -0.50, rear_coincidence=0.06 if seed != 46 else 0.01)
        for seed in EXPECTED_SEEDS
    ]

    result = aggregate_gate_payloads(payloads, EXPECTED_SEEDS)

    assert result["rear_weakness_confirmed"] is True
    assert result["weak_seed_count"] == 4


def test_missing_seed_is_a_failure_not_a_partial_summary():
    payloads = [_payload(seed, -0.5) for seed in EXPECTED_SEEDS[:-1]]

    with pytest.raises(ValueError, match="missing seeds: 46"):
        aggregate_gate_payloads(payloads, EXPECTED_SEEDS)


def test_low_valid_fraction_is_rejected():
    payloads = [_payload(seed, -0.5) for seed in EXPECTED_SEEDS]
    payloads[0]["phase_metrics"]["contact_phase"]["pairs"]["RL-RR"][
        "valid_fraction"
    ] = 0.89

    with pytest.raises(ValueError, match="valid_fraction"):
        aggregate_gate_payloads(payloads, EXPECTED_SEEDS)


def test_mixed_checkpoints_are_rejected():
    payloads = [_payload(seed, -0.5) for seed in EXPECTED_SEEDS]
    payloads[-1]["metadata"]["checkpoint"] = "/runs/model_2999.pt"

    with pytest.raises(ValueError, match="multiple checkpoints"):
        aggregate_gate_payloads(payloads, EXPECTED_SEEDS)
