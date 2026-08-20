import pytest
import torch

from gait_phase_metrics import contact_phase_metrics


def test_perfect_trot_has_opposed_same_end_pairs_and_matched_diagonals():
    fl = torch.tensor([0, 0, 1, 1] * 20, dtype=torch.float32)
    fr = 1.0 - fl
    contacts = torch.stack((fl, fr, fr, fl), dim=-1)[:, None, :]

    result = contact_phase_metrics(contacts)

    assert result["pairs"]["FL-FR"]["median"] == pytest.approx(-1.0)
    assert result["pairs"]["RL-RR"]["median"] == pytest.approx(-1.0)
    assert result["pairs"]["FL-RR"]["median"] == pytest.approx(1.0)
    assert result["pairs"]["FR-RL"]["median"] == pytest.approx(1.0)
    assert result["transition_coincidence"]["front"] == pytest.approx(0.0)
    assert result["transition_coincidence"]["rear"] == pytest.approx(0.0)


def test_constant_feet_are_invalid_instead_of_fake_zero_correlations():
    contacts = torch.zeros((40, 3, 4), dtype=torch.float32)

    result = contact_phase_metrics(contacts)

    pair = result["pairs"]["RL-RR"]
    assert pair["valid_fraction"] == 0.0
    assert pair["median"] is None
    assert pair["p10"] is None
    assert pair["p90"] is None


def test_same_direction_pair_transitions_are_counted_as_coincident():
    contacts = torch.tensor(
        [
            [[0, 0, 1, 1]],
            [[1, 1, 0, 0]],
            [[0, 0, 1, 1]],
        ],
        dtype=torch.float32,
    )

    result = contact_phase_metrics(contacts)

    assert result["transition_coincidence"]["front"] == pytest.approx(1.0)
    assert result["transition_coincidence"]["rear"] == pytest.approx(1.0)


def test_quantiles_are_across_environments_not_time_samples():
    alternating = torch.tensor([0, 1] * 20, dtype=torch.float32)
    matched = alternating.clone()
    opposed = 1.0 - alternating
    first_env = torch.stack((alternating, opposed, opposed, alternating), dim=-1)
    second_env = torch.stack((alternating, matched, matched, alternating), dim=-1)
    contacts = torch.stack((first_env, second_env), dim=1)

    result = contact_phase_metrics(contacts)

    front = result["pairs"]["FL-FR"]
    assert front["valid_fraction"] == 1.0
    assert front["median"] == pytest.approx(0.0)
    assert front["p10"] == pytest.approx(-0.8)
    assert front["p90"] == pytest.approx(0.8)


def test_contact_phase_metrics_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\[time, environments, 4\]"):
        contact_phase_metrics(torch.zeros((10, 4)))
