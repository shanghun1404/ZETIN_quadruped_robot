"""Is the body frame's 'upright' the same as the world's?

bad_orientation terminates on the angle between the body-frame projected gravity
and straight down. If the MJCF->USD conversion baked a ROTATION into the root
prim (it demonstrably baked a translation), the standing robot would read as
permanently tilted and the term would fire on the first step -- which is exactly
what the run shows: 100% base_tilt terminations, mean episode length 8.3/1000.
"""
import argparse, json, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--report", default="/tmp/tilt_check.json")
p.add_argument("--spawn_z", type=float, default=None)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
import gymnasium as gym, torch
from v10_1_env_cfg import GuideV2BaselineEnvCfg
from v10_1_ppo_cfg import V10_1_PPO_CFG
TASK = "Isaac-Tilt-Check-v10-1"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": GuideV2BaselineEnvCfg, "rsl_rl_cfg_entry_point": V10_1_PPO_CFG})
out = {}
try:
    cfg = GuideV2BaselineEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = 3
    cfg.observations.policy.enable_corruption = False
    # remove reset randomisation so we see the asset's own orientation
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x","y","z","roll","pitch","yaw")}
    cfg.events.push_robot = None
    cfg.terminations.base_tilt = None          # observe rather than terminate
    if a.spawn_z is not None:
        cfg.scene.robot.init_state.pos = (0.0, 0.0, a.spawn_z)
    env = gym.make(TASK, cfg=cfg); env.reset()
    base = env.unwrapped; robot = base.scene["robot"]
    action = torch.zeros((a.num_envs, 12), device=base.device)
    trace = []
    for step in range(400):
        env.step(action)
        if step in (0, 5, 50, 150, 399):
            g = robot.data.projected_gravity_b          # unit gravity in body frame
            # bad_orientation uses acos(-g_z) style tilt from vertical
            tilt = torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))
            q = robot.data.root_quat_w
            trace.append({
                "step": step,
                "proj_gravity_mean": [round(float(g[:, i].mean()), 4) for i in range(3)],
                "tilt_deg_mean": round(float(torch.rad2deg(tilt).mean()), 3),
                "tilt_deg_max": round(float(torch.rad2deg(tilt).max()), 3),
                "quat_w_mean": round(float(q[:, 0].mean()), 4),
                "root_z": round(float(robot.data.root_pos_w[:, 2].mean()), 4),
            })
    # do the feet actually touch anything?
    cs = base.scene.sensors["contact_forces"]
    f = cs.data.net_forces_w
    out["contact_bodies"] = list(cs.body_names)
    out["total_contact_force_N"] = round(float(f.norm(dim=-1).sum(dim=1).mean()), 2)
    out["expected_weight_N"] = round(float(robot.root_physx_view.get_masses()[0].sum()) * 9.81, 1)
    out["trace"] = trace
    out["tilt_limit_deg"] = 60.0
    out["fires_at_rest"] = trace[-1]["tilt_deg_mean"] > 60.0
    out["ok"] = True
    env.close()
except Exception:
    import traceback; out["error"] = traceback.format_exc()[-1200:]
with open(a.report, "w") as h: json.dump(out, h, indent=2)
app.close()
