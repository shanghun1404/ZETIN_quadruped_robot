"""[HISTORICAL -- written before the V10.3 fix. The constants it argues about
have since been corrected; it now imports the fixed config, so its framing
describes the defect as it stood, not the current state.]

Read the foot-to-trunk offset straight out of the USD. No simulator, no GPU.

``guide_robot_v2_cfg.py`` asserts ``FEET_ABOVE_ROOT_FRAME_M = 0.42`` -- that the
feet sit above the root frame -- and derives every spawn height from it as
``clearance - 0.42``. Three measurements disagree with the sign:

    MJCF ``stand`` keyframe, base qpos z   +0.4640
    V10.1 probe, settled root z           +0.4357
    V10.2 probe, settled root z           +0.4455

This script adds a fourth that touches neither the simulator nor the training
config: it opens the USD and compares the world-space bounds of the trunk
against the world-space bounds of the four foot colliders, in the asset's own
rest pose. Only the sign matters here -- the rest pose is not the stand pose, so
the magnitude will differ from the numbers above.

Bounds, not frames, deliberately. The conversion leaves body *frames* displaced
from their geometry on this asset, which is the defect that has already broken
four separate mechanisms. A bounding box is where the collider actually is.

Note ``Usd.TraverseInstanceProxies``: a plain ``Stage.Traverse()`` does not
descend into instanced prims, and reading this asset without it previously
produced a false "no collision geometry" conclusion.

``pxr`` is only importable once the kit extensions are loaded, so this launches
the app headless and imports afterwards -- the same order every other asset
script in this directory uses. It builds no scene and steps no physics.
"""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure foot-vs-trunk z in the USD.")
parser.add_argument("--usd", default="/home/kang/robot_v2/usd/v2_plain.usd")
parser.add_argument("--report", default="/tmp/usd_foot_offset_check.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(args.usd)
if stage is None:
    raise SystemExit(f"could not open {args.usd}")

cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy],
    useExtentsHint=False,
)

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
report: dict = {"usd": args.usd}
bounds: dict[str, dict] = {}

for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    name = prim.GetName()
    if name != "base" and name not in FOOT_NAMES:
        continue
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        continue
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        continue
    lo, hi = box.GetMin(), box.GetMax()
    entry = {
        "path": str(prim.GetPath()),
        "z_min": round(float(lo[2]), 4),
        "z_max": round(float(hi[2]), 4),
        "z_centre": round(float((lo[2] + hi[2]) / 2.0), 4),
    }
    # Keep the first (outermost) hit per name; nested prims repeat the name.
    bounds.setdefault(name, entry)

report["bounds"] = bounds
missing = [n for n in ("base",) + FOOT_NAMES if n not in bounds]
report["missing"] = missing

if not missing:
    trunk = bounds["base"]["z_centre"]
    feet = [bounds[n]["z_centre"] for n in FOOT_NAMES]
    feet_mean = sum(feet) / len(feet)
    report["trunk_z_centre"] = round(trunk, 4)
    report["feet_z_centre_mean"] = round(feet_mean, 4)
    # Positive means the feet sit BELOW the trunk, i.e. a normal quadruped and a
    # root frame that is above the ground when standing.
    report["trunk_minus_feet_m"] = round(trunk - feet_mean, 4)
    report["feet_are_below_trunk"] = bool(trunk - feet_mean > 0.0)
    report["config_claims_feet_above_root"] = True
    report["verdict"] = (
        "feet below trunk -- FEET_ABOVE_ROOT_FRAME_M has the wrong sign"
        if report["feet_are_below_trunk"]
        else "feet above trunk -- the config constant is consistent"
    )

with open(args.report, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
