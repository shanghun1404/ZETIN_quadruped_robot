"""Exact capsule clearance for inter-leg collision evaluation."""

import numpy as np
import pytest

from v14_clearance import (
    capsule_gap_m,
    interleg_clearances_m,
    segment_distance_m,
    summarize_interleg_clearance,
)


def test_crossing_segments_have_zero_centerline_distance() -> None:
    distance = segment_distance_m(
        np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
    )
    assert distance == pytest.approx(0.0)
    assert capsule_gap_m(
        np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        0.02, 0.02,
    ) == pytest.approx(-0.04)


def test_parallel_and_skew_segments_use_boundary_or_interior_minimum() -> None:
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.0, 0.0])
    assert segment_distance_m(
        start, end, np.array([0.5, 0.1, 0.0]), np.array([1.5, 0.1, 0.0]),
    ) == pytest.approx(0.1)
    assert segment_distance_m(
        start, end, np.array([1.2, -1.0, 0.0]), np.array([1.2, 1.0, 0.0]),
    ) == pytest.approx(0.2)
    assert segment_distance_m(
        start, end, np.array([0.5, -1.0, 0.05]), np.array([0.5, 1.0, 0.05]),
    ) == pytest.approx(0.05)


def test_interleg_clearance_uses_link_local_capsules() -> None:
    names = tuple(
        f"{leg}_{part}_profile"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf")
    )
    positions = np.zeros((1, 8, 3))
    positions[0, 2:4, 1] = 0.1
    positions[0, 4:6, 1] = 1.0
    positions[0, 6:8, 1] = -1.0
    quats = np.zeros((1, 8, 4))
    quats[..., 0] = 1.0
    gaps = interleg_clearances_m(positions, quats, names)
    assert gaps.shape == (1, 24)
    assert gaps.min() == pytest.approx(0.06)
    assert interleg_clearances_m(positions, quats, names, radius_m=0.015).min() == pytest.approx(0.07)


def test_clearance_summary_ignores_finished_environments() -> None:
    gaps = np.full((2, 2, 24), 0.1)
    gaps[0, 0, 0] = -0.003
    gaps[1, 0, 0] = -0.02
    gaps[1, 1, 0] = 0.01
    alive = np.array([[True, True], [False, True]])
    summary = summarize_interleg_clearance(gaps, alive)
    assert summary["measured_env_frames"] == 3
    assert summary["minimum_gap_m"] == pytest.approx(-0.003)
    assert summary["negative_gap_frame_fraction"] == pytest.approx(1 / 3)
    assert summary["gap_below_minus_2mm_frame_fraction"] == pytest.approx(1 / 3)
