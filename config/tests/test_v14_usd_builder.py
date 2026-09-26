"""Integration contract for the composed V14 robot asset (run in Isaac Sim)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_derived_usd_preserves_feet_and_filters_only_same_leg(tmp_path: Path) -> None:
    source_text = os.environ.get("V14_SOURCE_USD")
    if not source_text:
        pytest.skip("set V14_SOURCE_USD to run the Isaac Sim USD contract")

    from v14_usd_builder import build_v14_overlay
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        from v14_usd_plan import collision_capsules, same_leg_filtered_pairs

        source = Path(source_text)
        output = tmp_path / "v14_robot.usda"
        build_v14_overlay(source, output)
        source_stage = Usd.Stage.Open(str(source))
        stage = Usd.Stage.Open(str(output))
        assert stage is not None
        for capsule_spec in collision_capsules():
            path = f"/quadruped/base/{capsule_spec.link_name}/v14_leg_collision"
            prim = stage.GetPrimAtPath(path)
            assert prim.IsValid()
            capsule = UsdGeom.Capsule(prim)
            assert capsule.GetRadiusAttr().Get() == pytest.approx(0.02)
            assert capsule.GetHeightAttr().Get() == pytest.approx(0.23)
            assert capsule.GetAxisAttr().Get() == "X"
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()

        for leg in ("FL", "FR", "RL", "RR"):
            foot_path = f"/quadruped/base/{leg}_foot/collisions/foot/foot"
            original = source_stage.GetPrimAtPath(foot_path)
            derived = stage.GetPrimAtPath(foot_path)
            assert original.IsValid() and derived.IsValid()
            assert original.GetTypeName() == derived.GetTypeName()
            assert UsdPhysics.CollisionAPI(derived).GetCollisionEnabledAttr().Get()

        for first, second in same_leg_filtered_pairs():
            first_prim = stage.GetPrimAtPath(f"/quadruped/base/{first}")
            targets = UsdPhysics.FilteredPairsAPI(first_prim).GetFilteredPairsRel().GetTargets()
            assert f"/quadruped/base/{second}" in {str(path) for path in targets}

        root = stage.GetPrimAtPath("/quadruped/base/base")
        assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True

        tuned_output = tmp_path / "v14_robot_radius_015.usda"
        tuned_report = build_v14_overlay(source, tuned_output, radius_m=0.015)
        tuned_stage = Usd.Stage.Open(str(tuned_output))
        assert tuned_report["capsule_radius_m"] == pytest.approx(0.015)
        assert UsdGeom.Capsule(
            tuned_stage.GetPrimAtPath("/quadruped/base/FL_thigh_profile/v14_leg_collision")
        ).GetRadiusAttr().Get() == pytest.approx(0.015)

    finally:
        # Kit closes with the process. app.close() exits before pytest reports failures.
        pass
