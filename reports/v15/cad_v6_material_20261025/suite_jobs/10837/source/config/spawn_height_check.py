"""[HISTORICAL -- written before the V10.3 fix. The constants it argues about
have since been corrected; it now imports the fixed config, so its framing
describes the defect as it stood, not the current state.]

Measure where the root frame sits when the feet are on the ground.

``guide_robot_v2_cfg.py`` records ``SETTLED_ROOT_FRAME_Z_M = -0.4192`` and
``FEET_ABOVE_ROOT_FRAME_M = 0.42``, both annotated as measured. Today's probes
measure the settled root z at +0.4455 (V10.2) and +0.4357 (V10.1) on the same
asset on flat ground, with 0.95 of the robot's weight carried by the feet. The
sign is inverted: the feet are roughly 0.44 m *below* the root frame, not above.

The consequence is the spawn arithmetic. ``SPAWN_ROOT_FRAME_Z_M`` is written as
``clearance - FEET_ABOVE_ROOT_FRAME_M``, which puts the feet a full
``2 x 0.44`` below where it intends:

    flat     0.05 - 0.42 = -0.37   feet land 0.81 m under the floor
    terrain  0.32 - 0.42 = -0.10   feet land 0.54 m under the tile origin

On a flat GroundPlane the solver ejects the robot and it settles correctly, so
this never showed up in a gate. On generated terrain it is exactly the two
observed failures: falling through the mesh from underneath at a low spawn, and
a 165-166 degree flip from any spawn high enough to make contact.

This script does not read those constants. It measures three things directly:

1. where the root frame settles when the robot is dropped and left alone;
2. how far the robot travels vertically between spawn and settle, which is the
   size of the interpenetration recovery every episode currently begins with;
3. whether spawning at ``settled + clearance`` starts the robot already standing
   -- support near 1.0 within a few steps and no large vertical excursion.

Run it with the GPU free; it builds a small scene twice.
"""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Flat-Guide-RobotV2-SpawnCheck"
parser = argparse.ArgumentParser(description="Measure the settled root frame height.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--report", default="/tmp/spawn_height_check.json")
parser.add_argument("--clearance", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from v10_2_flat_env_cfg import GuideV2FlatDutyLiveEnvCfg
from v10_2_flat_ppo_cfg import V10_2_FLAT_PPO_CFG

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": GuideV2FlatDutyLiveEnvCfg,
        "rsl_rl_cfg_entry_point": V10_2_FLAT_PPO_CFG,
    },
)

report = {"ok": False, "clearance_requested": args_cli.clearance}


def _measure(spawn_z, settle_steps=400):
    """Drop the robot from ``spawn_z`` and report how it ends up."""
    cfg = GuideV2FlatDutyLiveEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 7
    cfg.scene.robot.init_state.pos = (0.0, 0.0, float(spawn_z))
    # Randomised reset pose would add noise to a geometry measurement.
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }

    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped
    robot = base.scene["robot"]
    sensor = base.scene.sensors["contact_forces"]
    weight = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
    action = torch.zeros((args_cli.num_envs, 12), device=base.device)

    trace = []
    for step in range(settle_steps):
        env.step(action)
        if step in (0, 4, 9, 24, 49, 99, 199, 399):
            force = sensor.data.net_forces_w.norm(dim=-1).sum(dim=1)
            trace.append({
                "step": step + 1,
                "root_z": round(float(robot.data.root_pos_w[:, 2].mean()), 4),
                "support": round(float(force.mean()) / weight, 3),
            })

    force = sensor.data.net_forces_w.norm(dim=-1).sum(dim=1)
    gravity = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
    out = {
        "spawn_root_z": round(float(spawn_z), 4),
        "settled_root_z": round(float(robot.data.root_pos_w[:, 2].mean()), 4),
        "support_fraction": round(float(force.mean()) / weight, 3),
        "tilt_deg_mean": round(float(tilt.mean()), 3),
        "tilt_deg_max": round(float(tilt.max()), 3),
        "trace": trace,
    }
    out["vertical_excursion_m"] = round(out["settled_root_z"] - out["spawn_root_z"], 4)
    env.close()
    return out


try:
    from v10_flat_env_cfg import SPAWN_ROOT_FRAME_Z_M as CURRENT_SPAWN_Z

    # 1. What the configuration does today.
    report["as_configured"] = _measure(CURRENT_SPAWN_Z)
    settled = report["as_configured"]["settled_root_z"]

    # 2. The distance from the root frame down to the ground, which is the
    #    quantity init_state.pos.z actually controls. Measured, not assumed.
    report["root_above_ground_m"] = settled
    corrected = settled + args_cli.clearance
    report["corrected_spawn_root_z"] = round(corrected, 4)

    # 3. Does the corrected spawn start the robot standing?
    report["as_corrected"] = _measure(corrected)

    failures = []
    a, b = report["as_configured"], report["as_corrected"]
    if abs(a["vertical_excursion_m"]) < 0.10:
        failures.append(
            "the configured spawn shows no large vertical excursion; the premise "
            "of this check does not hold and the constants may be right"
        )
    if abs(b["vertical_excursion_m"]) > 0.10:
        failures.append(
            f"the corrected spawn still moves {b['vertical_excursion_m']} m vertically"
        )
    # Standing from the first steps is the whole point: no ejection transient.
    early = b["trace"][1]["support"] if len(b["trace"]) > 1 else 0.0
    report["corrected_support_at_step_5"] = early
    if not (0.5 < early < 1.6):
        failures.append(f"corrected spawn is not on its feet by step 5 (support {early})")
    if not (0.80 < b["support_fraction"] < 1.25):
        failures.append(f"corrected spawn settles at support {b['support_fraction']}")
    if b["tilt_deg_max"] > 20.0:
        failures.append(f"corrected spawn tilts to {b['tilt_deg_max']} deg")

    report["failures"] = failures
    report["ok"] = not failures
except Exception:
    report["error"] = traceback.format_exc()[-1500:]

with open(args_cli.report, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
