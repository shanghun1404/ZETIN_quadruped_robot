"""Scene-level ground filtering preserves the USD overlay's leg contacts."""

import os

import pytest


def test_ground_filters_only_v14_capsules() -> None:
    overlay = os.environ.get("V14_OVERLAY_USD")
    if not overlay:
        pytest.skip("set V14_OVERLAY_USD to run the Isaac Sim USD contract")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from pxr import Usd, UsdGeom, UsdPhysics

    from v14_terrain_filter import author_v14_ground_filter

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/ground")
    env_paths = [f"/World/envs/env_{index}" for index in range(2)]
    for env_path in env_paths:
        robot = UsdGeom.Xform.Define(stage, f"{env_path}/Robot").GetPrim()
        robot.GetReferences().AddReference(overlay, "/quadruped")

    assert author_v14_ground_filter(stage, env_paths) == 16
    ground = stage.GetPrimAtPath("/World/ground")
    targets = {
        str(path)
        for path in UsdPhysics.FilteredPairsAPI(ground).GetFilteredPairsRel().GetTargets()
    }
    assert len(targets) == 16
    assert "/World/envs/env_0/Robot/base/FL_thigh_profile/v14_leg_collision" in targets
    assert "/World/envs/env_1/Robot/base/RR_calf_profile/v14_leg_collision" in targets
    assert not any("_foot" in path for path in targets)
