"""The V14 USD overlay may filter only contacts within one leg."""

from pathlib import Path

import pytest

from v14_usd_plan import (
    SOURCE_SHA256,
    collision_capsules,
    same_leg_filtered_pairs,
    verify_source_sha256,
)


def test_capsules_cover_both_profiles_on_each_leg() -> None:
    capsules = collision_capsules()
    assert len(capsules) == 8
    assert {capsule.link_name for capsule in capsules} == {
        f"{leg}_{part}_profile"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("thigh", "calf")
    }
    assert all(capsule.radius_m == 0.02 for capsule in capsules)
    assert all(capsule.height_m == 0.23 for capsule in capsules)
    assert all(capsule.center_link_m == (0.125, 0.09, 0.24) for capsule in capsules)


def test_capsule_radius_can_be_tuned_within_profile_bounds() -> None:
    assert {capsule.radius_m for capsule in collision_capsules(radius_m=0.015)} == {0.015}
    with pytest.raises(ValueError, match="radius"):
        collision_capsules(radius_m=0.009)


def test_filtered_pairs_are_only_within_a_leg() -> None:
    pairs = same_leg_filtered_pairs()
    assert len(pairs) == 12
    assert len(set(pairs)) == 12
    for first, second in pairs:
        assert first[:2] == second[:2]
        assert first != second
    assert ("FL_thigh_profile", "FL_calf_profile") in pairs
    assert ("FL_thigh_profile", "FR_thigh_profile") not in pairs


def test_source_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "wrong.usd"
    source.write_bytes(b"wrong robot")
    assert len(SOURCE_SHA256) == 64
    with pytest.raises(ValueError, match="SHA-256"):
        verify_source_sha256(source)
