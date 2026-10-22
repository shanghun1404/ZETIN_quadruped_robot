"""Measure robot_v2 body/joint/collider layout in the trunk frame for the V15 recovery USD.

Only feet collide with terrain in the source asset, so a fallen trunk sinks through
the ground. This probe records, relative to the trunk body, every rigid body's
geometry bounds (instance proxies included), the joint tree and existing
colliders, so a trunk collider can be placed from measurements, not frame guesses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

TRUNK = "/quadruped/base/base"


def box(bound) -> dict[str, list[float]]:
    aligned = bound.ComputeAlignedRange()
    return {"min": list(aligned.GetMin()), "max": list(aligned.GetMax())}


def main() -> None:
    stage = Usd.Stage.Open(args_cli.usd_path)
    trunk = stage.GetPrimAtPath(TRUNK)
    if not trunk.IsValid():
        raise RuntimeError(f"missing trunk prim {TRUNK}")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    bodies, joints, colliders = {}, [], []
    trunk_to_world = UsdGeom.Xformable(trunk).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            origin = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            bodies[path] = {
                "bound_in_trunk": box(cache.ComputeRelativeBound(prim, trunk)),
                "origin_in_trunk": list((origin * trunk_to_world.GetInverse()).ExtractTranslation()),
                "active": prim.IsActive(),
            }
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append({
                "path": path, "type": prim.GetTypeName(),
                "body0": [str(t) for t in joint.GetBody0Rel().GetTargets()],
                "body1": [str(t) for t in joint.GetBody1Rel().GetTargets()],
            })
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            colliders.append({"path": path, "type": prim.GetTypeName(), "enabled": enabled,
                              "bound_in_trunk": box(cache.ComputeRelativeBound(prim, trunk))})
    report = {
        "usd_path": args_cli.usd_path,
        "trunk": TRUNK,
        "trunk_world_translation": list(trunk_to_world.ExtractTranslation()),
        "bodies": bodies,
        "joints": joints,
        "colliders": colliders,
    }
    Path(args_cli.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"[INFO] bodies={len(bodies)} joints={len(joints)} colliders={len(colliders)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
