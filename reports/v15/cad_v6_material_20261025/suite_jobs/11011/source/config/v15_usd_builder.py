"""Author the V15 recovery USD: the V14 USD plus a trunk collider.

The source asset collides with terrain only through its feet (V14 capsules are
terrain-filtered by the V14 locomotion env, not by the USD), so a fallen trunk
sinks through the ground. V15-R adds one box on the trunk body, sized from the
measured trunk panel geometry in the trunk frame (probe_v15_trunk_geometry.py,
job 280): x +/-0.20 m, y +/-0.26 m, z 0-0.15 m. The box is filtered against the
leg capsules and feet so hip/thigh proximity cannot create internal contacts.
The V14 USD and the original asset are referenced, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

TRUNK_PATH = "/quadruped/base/base"
TRUNK_BOX_CENTER_M = (0.0, 0.0, 0.075)
TRUNK_BOX_SIZE_M = (0.40, 0.52, 0.15)
LEG_COLLIDERS = tuple(
    path
    for leg in ("FL", "FR", "RL", "RR")
    for path in (f"/quadruped/base/{leg}_thigh_profile/v14_leg_collision",
                 f"/quadruped/base/{leg}_calf_profile/v14_leg_collision",
                 f"/quadruped/base/{leg}_foot/collisions/foot/foot")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_v15_recovery_overlay(v14_usd: str | Path, output_usd: str | Path) -> dict[str, object]:
    source = Path(v14_usd).resolve()
    output = Path(output_usd).resolve()
    if output == source or output.exists():
        raise FileExistsError(f"V15 output already exists or aliases source: {output}")

    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output))
    try:
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/quadruped")
        root.GetPrim().GetReferences().AddReference(str(source), "/quadruped")
        stage.SetDefaultPrim(root.GetPrim())

        trunk = stage.GetPrimAtPath(TRUNK_PATH)
        mass = UsdPhysics.MassAPI(trunk).GetMassAttr().Get() if trunk.HasAPI(UsdPhysics.MassAPI) else None
        if not trunk.HasAPI(UsdPhysics.RigidBodyAPI) or not mass or mass <= 0.0:
            raise ValueError("trunk must be a rigid body with authored mass so the collider adds no mass")
        box = UsdGeom.Cube.Define(stage, f"{TRUNK_PATH}/v15_trunk_collision")
        box.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(box.GetPrim())
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*TRUNK_BOX_CENTER_M))
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*TRUNK_BOX_SIZE_M))
        UsdGeom.Imageable(box.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        pairs = UsdPhysics.FilteredPairsAPI.Apply(box.GetPrim()).CreateFilteredPairsRel()
        for path in LEG_COLLIDERS:
            if not stage.GetPrimAtPath(path).IsValid():
                raise ValueError(f"missing V14 leg collider: {path}")
            pairs.AddTarget(path)
        if not stage.GetRootLayer().Save():
            raise RuntimeError("failed to save V15 recovery USD")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "source_usd": str(source),
        "source_sha256": _sha256(source),
        "output_usd": str(output),
        "output_sha256": _sha256(output),
        "trunk_box_center_m": TRUNK_BOX_CENTER_M,
        "trunk_box_size_m": TRUNK_BOX_SIZE_M,
        "trunk_authored_mass_kg": mass,
        "filtered_leg_colliders": list(LEG_COLLIDERS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14-usd", required=True, type=Path)
    parser.add_argument("--output-usd", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()
    if args.report_json.exists():
        parser.error(f"report already exists: {args.report_json}")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        report = build_v15_recovery_overlay(args.v14_usd, args.output_usd)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_uncancellable_quit(1)
        app.close()
        raise SystemExit(1)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
