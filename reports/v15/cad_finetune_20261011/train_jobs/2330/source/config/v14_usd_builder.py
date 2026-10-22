"""Author a derived robot_v2 USD with selective inter-leg collisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from v14_usd_plan import CAPSULE_RADIUS_M, collision_capsules, same_leg_filtered_pairs, verify_source_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_v14_overlay(
    source_usd: str | Path,
    output_usd: str | Path,
    *,
    radius_m: float = CAPSULE_RADIUS_M,
) -> dict[str, object]:
    """Reference the reviewed source and add only V14 collision opinions."""
    source = Path(source_usd).resolve()
    output = Path(output_usd).resolve()
    source_digest = verify_source_sha256(source)
    if output == source or output.exists():
        raise FileExistsError(f"V14 output already exists or aliases source: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")

    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(output))
    if stage is None:
        raise RuntimeError(f"could not create V14 USD: {output}")
    try:
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/quadruped")
        root.GetPrim().GetReferences().AddReference(str(source), "/quadruped")
        stage.SetDefaultPrim(root.GetPrim())

        for spec in collision_capsules(radius_m=radius_m):
            path = f"/quadruped/base/{spec.link_name}/v14_leg_collision"
            capsule = UsdGeom.Capsule.Define(stage, path)
            capsule.CreateAxisAttr("X")
            capsule.CreateHeightAttr(spec.height_m)
            capsule.CreateRadiusAttr(spec.radius_m)
            UsdGeom.Xformable(capsule.GetPrim()).AddTranslateOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(*spec.center_link_m))
            UsdGeom.Imageable(capsule.GetPrim()).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(capsule.GetPrim())

        for first, second in same_leg_filtered_pairs():
            first_prim = stage.GetPrimAtPath(f"/quadruped/base/{first}")
            second_prim = stage.GetPrimAtPath(f"/quadruped/base/{second}")
            if not first_prim.IsValid() or not second_prim.IsValid():
                raise ValueError(f"missing robot_v2 leg link: {first}, {second}")
            UsdPhysics.FilteredPairsAPI.Apply(first_prim).CreateFilteredPairsRel().AddTarget(
                second_prim.GetPath()
            )

        articulation = stage.GetPrimAtPath("/quadruped/base/base")
        if not articulation.IsValid():
            raise ValueError("robot_v2 articulation root is missing")
        PhysxSchema.PhysxArticulationAPI.Apply(
            articulation
        ).CreateEnabledSelfCollisionsAttr(True)
        if not stage.GetRootLayer().Save():
            raise RuntimeError("failed to save V14 USD")
    except BaseException:
        output.unlink(missing_ok=True)
        raise

    return {
        "source_usd": str(source),
        "source_sha256": source_digest,
        "output_usd": str(output),
        "output_sha256": _sha256(output),
        "capsule_count": len(collision_capsules(radius_m=radius_m)),
        "same_leg_filtered_pair_count": len(same_leg_filtered_pairs()),
        "capsule_radius_m": radius_m,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", required=True, type=Path)
    parser.add_argument("--output-usd", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--radius-m", default=CAPSULE_RADIUS_M, type=float)
    args = parser.parse_args()
    if args.report_json.exists():
        parser.error(f"report already exists: {args.report_json}")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        report = build_v14_overlay(
            args.source_usd,
            args.output_usd,
            radius_m=args.radius_m,
        )
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
