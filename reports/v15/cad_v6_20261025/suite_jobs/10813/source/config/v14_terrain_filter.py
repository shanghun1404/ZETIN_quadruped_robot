"""Author V14 capsule-to-terrain filters on the composed IsaacLab scene."""

from __future__ import annotations

from collections.abc import Sequence

from v14_usd_plan import LEGS, PROFILE_PARTS


GROUND_PATH = "/World/ground"


def author_v14_ground_filter(stage: object, env_prim_paths: Sequence[str]) -> int:
    """Keep the original feet on terrain while excluding V14 leg capsules."""
    from pxr import UsdPhysics

    ground = stage.GetPrimAtPath(GROUND_PATH)
    if not ground.IsValid():
        raise ValueError(f"ground prim is missing: {GROUND_PATH}")
    if not env_prim_paths:
        raise ValueError("at least one environment prim path is required")

    capsule_paths = []
    for env_path in env_prim_paths:
        for leg in LEGS:
            for part in PROFILE_PARTS:
                path = (
                    f"{env_path}/Robot/base/{leg}_{part}_profile/v14_leg_collision"
                )
                if not stage.GetPrimAtPath(path).IsValid():
                    raise ValueError(f"V14 capsule prim is missing: {path}")
                capsule_paths.append(path)

    relation = UsdPhysics.FilteredPairsAPI.Apply(ground).CreateFilteredPairsRel()
    existing = {str(path) for path in relation.GetTargets()}
    added = 0
    for path in capsule_paths:
        if path not in existing:
            relation.AddTarget(path)
            added += 1
    return added


def apply_v14_ground_filter(env: object, env_ids: object) -> None:
    """IsaacLab prestartup event; called after scene creation and before sim.reset()."""
    if env_ids is not None:
        raise ValueError("V14 ground filter must run for the entire stage")
    count = author_v14_ground_filter(
        env.sim.get_initial_stage(), env.scene.env_prim_paths
    )
    expected = len(env.scene.env_prim_paths) * len(LEGS) * len(PROFILE_PARTS)
    if count != expected:
        raise RuntimeError(f"V14 ground filters added {count}, expected {expected}")
