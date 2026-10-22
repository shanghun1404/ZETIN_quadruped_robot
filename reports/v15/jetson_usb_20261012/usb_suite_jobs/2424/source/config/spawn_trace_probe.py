"""[HISTORICAL -- written before the V10.3 fix. The constants it argues about
have since been corrected; it now imports the fixed config, so its framing
describes the defect as it stood, not the current state.]

Trace the root frame from spawn to settle. A minimal edit of the V10.2 probe.

Question: does the robot spawn deep inside the floor and get ejected?

``guide_robot_v2_cfg.py`` says the feet sit 0.42 m *above* the root frame, so the flat
config spawns the root frame at ``0.05 - 0.42 = -0.37``. Both probes then measure the
settled root frame at **+0.44** with the ground carrying 94-95% of the robot's weight.
If the spawn really is at -0.37 and the settled state really is +0.44, the robot rises
0.81 m with no actuation, which only happens if it started underground.

The alternative is that ``init_state.pos.z`` is not the quantity ``root_pos_w[:, 2]``
reports -- in which case the robot never was at -0.37 and there is no interpenetration.
A trace distinguishes these two immediately: read ``root_pos_w[:, 2]`` on the first
step and see whether it is near -0.37 or near +0.44.

``spawn_height_check.py`` was written to do this and hung -- 4 h 19 m at 105% CPU and
0% GPU, twice, before any output. Rather than debug it while holding the GPU, this
copies the structure of ``probe_v10_2_flat.py``, which is known to run in about a
minute, and changes only what is needed. Prints are flushed so a hang is visible.
"""

import argparse
import json
import traceback

from isaaclab.app import AppLauncher

TASK_ID = "Isaac-Velocity-Flat-Guide-RobotV2-SpawnTrace"
parser = argparse.ArgumentParser(description="Trace root z from spawn to settle.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--report", default="/tmp/spawn_trace_probe.json")
parser.add_argument("--spawn-z", type=float, default=None,
                    help="override init_state.pos.z; default keeps the config value")
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

report = {"ok": False}
try:
    cfg = GuideV2FlatDutyLiveEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 42
    if args_cli.spawn_z is not None:
        pos = cfg.scene.robot.init_state.pos
        cfg.scene.robot.init_state.pos = (pos[0], pos[1], float(args_cli.spawn_z))
    report["configured_spawn_z"] = float(cfg.scene.robot.init_state.pos[2])
    print(f"[trace] spawn z = {report['configured_spawn_z']}", flush=True)

    env = gym.make(TASK_ID, cfg=cfg)
    env.reset()
    base = env.unwrapped
    robot = base.scene["robot"]
    sensor = base.scene.sensors["contact_forces"]
    weight = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
    print(f"[trace] env built, weight {weight:.1f} N", flush=True)

    # Read before stepping: this is the spawn state itself.
    report["root_z_before_any_step"] = round(float(robot.data.root_pos_w[:, 2].mean()), 4)
    print(f"[trace] root z before stepping = {report['root_z_before_any_step']}", flush=True)

    action = torch.zeros((args_cli.num_envs, 12), device=base.device)
    trace = []
    for step in range(400):
        env.step(action)
        if step in (0, 1, 2, 4, 9, 19, 49, 99, 199, 399):
            force = sensor.data.net_forces_w.norm(dim=-1).sum(dim=1)
            row = {
                "step": step + 1,
                "root_z": round(float(robot.data.root_pos_w[:, 2].mean()), 4),
                "support": round(float(force.mean()) / weight, 3),
            }
            trace.append(row)
            print(f"[trace] {row}", flush=True)
    report["trace"] = trace

    force = sensor.data.net_forces_w.norm(dim=-1).sum(dim=1)
    gravity = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
    report["settled_root_z"] = round(float(robot.data.root_pos_w[:, 2].mean()), 4)
    report["support_fraction"] = round(float(force.mean()) / weight, 3)
    report["tilt_deg_mean"] = round(float(tilt.mean()), 3)
    report["rise_from_spawn_m"] = round(
        report["settled_root_z"] - report["root_z_before_any_step"], 4
    )
    # The verdict this script exists to deliver.
    report["spawned_underground"] = bool(report["root_z_before_any_step"] < 0.0)
    report["ok"] = True
    env.close()
except Exception:
    report["error"] = traceback.format_exc()[-1500:]

with open(args_cli.report, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2), flush=True)
simulation_app.close()
raise SystemExit(0 if report.get("ok") else 1)
