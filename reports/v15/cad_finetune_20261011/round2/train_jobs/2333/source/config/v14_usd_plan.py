"""Source-locked collision geometry for V14 inter-leg contacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


SOURCE_SHA256 = "3df1d83c036f4679ab183440a3de9520537f037a461f7e1e0a10e43ec0df6e18"
LEGS = ("FL", "FR", "RL", "RR")
PROFILE_PARTS = ("thigh", "calf")
CAPSULE_RADIUS_M = 0.02
CAPSULE_HEIGHT_M = 0.23
CAPSULE_CENTER_LINK_M = (0.125, 0.09, 0.24)


@dataclass(frozen=True)
class CapsuleSpec:
    link_name: str
    center_link_m: tuple[float, float, float]
    radius_m: float
    height_m: float


def collision_capsules(radius_m: float = CAPSULE_RADIUS_M) -> tuple[CapsuleSpec, ...]:
    """Return one capsule for each measured thigh/calf visual profile."""
    if not 0.01 <= radius_m <= 0.025:
        raise ValueError("capsule radius must be between visual radius 0.01 m and 0.025 m")
    return tuple(
        CapsuleSpec(
            link_name=f"{leg}_{part}_profile",
            center_link_m=CAPSULE_CENTER_LINK_M,
            radius_m=radius_m,
            height_m=CAPSULE_HEIGHT_M,
        )
        for leg in LEGS
        for part in PROFILE_PARTS
    )


def same_leg_filtered_pairs() -> tuple[tuple[str, str], ...]:
    """Exclude the three mechanically adjacent link pairs in each leg."""
    return tuple(
        (first, second)
        for leg in LEGS
        for first, second in (
            (f"{leg}_thigh_profile", f"{leg}_calf_profile"),
            (f"{leg}_thigh_profile", f"{leg}_foot"),
            (f"{leg}_calf_profile", f"{leg}_foot"),
        )
    )


def verify_source_sha256(path: str | Path) -> str:
    """Refuse any USD other than the reviewed robot_v2 source asset."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"robot_v2 USD SHA-256 mismatch: {actual}")
    return actual
