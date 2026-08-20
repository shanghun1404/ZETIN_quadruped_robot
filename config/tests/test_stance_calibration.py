import math
from dataclasses import FrozenInstanceError, replace

import pytest

from stance_calibration import (
    HEIGHT_BAND_M,
    STANCE_CANDIDATES,
    TARGET_HEIGHT_M,
    StanceCandidate,
    StanceMetrics,
    passes_stance,
    select_stance_candidate,
)


def passing_metrics(candidate: StanceCandidate | None = None, **overrides):
    values = {
        "candidate": candidate or STANCE_CANDIDATES[2],
        "mean_height_m": TARGET_HEIGHT_M,
        "support_fraction": 0.99,
        "rms_vertical_velocity_m_s": 0.08,
        "p95_height_range_m": 0.03,
        "body_collision": False,
        "low_height": False,
    }
    values.update(overrides)
    return StanceMetrics(**values)


def test_declares_exact_35cm_target_and_symmetric_candidate_geometry():
    assert TARGET_HEIGHT_M == 0.35
    assert HEIGHT_BAND_M == (0.335, 0.365)
    assert tuple(candidate.half_angle_rad for candidate in STANCE_CANDIDATES) == (
        0.82,
        0.85,
        0.87,
        0.90,
    )

    for candidate in STANCE_CANDIDATES:
        assert candidate.hfe_rad == pytest.approx(-candidate.half_angle_rad)
        assert candidate.kfe_rad == pytest.approx(2.0 * candidate.half_angle_rad)
        expected_spawn = 0.51 * math.cos(candidate.half_angle_rad) + 0.020 + 0.005
        assert candidate.spawn_height_m == pytest.approx(expected_spawn)


def test_selector_rejects_nonpassing_values_and_chooses_closest_height():
    values = [
        passing_metrics(STANCE_CANDIDATES[0], mean_height_m=0.366),
        passing_metrics(STANCE_CANDIDATES[1], mean_height_m=0.359),
        passing_metrics(STANCE_CANDIDATES[2], mean_height_m=0.349),
    ]
    assert select_stance_candidate(values).candidate is STANCE_CANDIDATES[2]


def test_passes_stance_accepts_all_boundaries():
    assert passes_stance(passing_metrics(mean_height_m=0.335))
    assert passes_stance(passing_metrics(mean_height_m=0.365))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean_height_m", 0.3349),
        ("mean_height_m", 0.3651),
        ("support_fraction", 0.989),
        ("rms_vertical_velocity_m_s", 0.081),
        ("p95_height_range_m", 0.031),
        ("body_collision", True),
        ("low_height", True),
    ],
)
def test_passes_stance_rejects_each_violation(field, value):
    assert not passes_stance(replace(passing_metrics(), **{field: value}))


def test_selector_breaks_equal_height_error_ties_by_lower_spawn_height():
    higher_spawn = StanceCandidate(0.82, -0.82, 1.64, 0.40)
    lower_spawn = StanceCandidate(0.90, -0.90, 1.80, 0.36)
    higher = passing_metrics(higher_spawn, mean_height_m=0.355)
    lower = passing_metrics(lower_spawn, mean_height_m=0.345)

    assert select_stance_candidate([higher, lower]) is lower


def test_selector_raises_when_no_candidate_passes():
    with pytest.raises(ValueError):
        select_stance_candidate([passing_metrics(low_height=True)])


def test_stance_contract_types_are_immutable():
    candidate = STANCE_CANDIDATES[0]
    metrics = passing_metrics(candidate)

    with pytest.raises(FrozenInstanceError):
        candidate.hfe_rad = -1.0
    with pytest.raises(FrozenInstanceError):
        metrics.mean_height_m = 0.34
