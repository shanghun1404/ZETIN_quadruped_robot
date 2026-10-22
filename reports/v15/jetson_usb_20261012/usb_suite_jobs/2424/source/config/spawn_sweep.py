"""Find the spawn frame z that lands the robot upright on generated terrain."""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--report", default="/tmp/spawn_sweep.json")
p.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
import gymnasium as gym, torch
from v10_1_env_cfg import GuideV2BaselineEnvCfg
from v10_1_ppo_cfg import V10_1_PPO_CFG
TASK = "Isaac-Spawn-Sweep-v10-1"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": GuideV2BaselineEnvCfg, "rsl_rl_cfg_entry_point": V10_1_PPO_CFG})
rows = []
try:
    for z in (0.0, 0.2, 0.4, 0.6, 1.0):
        cfg = GuideV2BaselineEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = 3
        cfg.observations.policy.enable_corruption = False
        cfg.events.reset_base.params["pose_range"] = {"x": (0.0,0.0), "y": (0.0,0.0), "yaw": (0.0,0.0)}
        cfg.events.reset_base.params["velocity_range"] = {k:(0.0,0.0) for k in ("x","y","z","roll","pitch","yaw")}
        cfg.events.push_robot = None
        cfg.terminations.base_tilt = None
        cfg.scene.robot.init_state.pos = (0.0, 0.0, z)
        env = gym.make(TASK, cfg=cfg); env.reset()
        base = env.unwrapped; robot = base.scene["robot"]; cs = base.scene.sensors["contact_forces"]
        action = torch.zeros((a.num_envs, 12), device=base.device)
        for _ in range(400):
            env.step(action)
        g = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:,2], -1.0, 1.0)))
        force = float(cs.data.net_forces_w.norm(dim=-1).sum(dim=1).mean())
        weight = float(robot.root_physx_view.get_masses()[0].sum()) * 9.81
        rows.append({
            "spawn_frame_z": z,
            "settled_root_z": round(float(robot.data.root_pos_w[:,2].mean()), 4),
            "origin_z": round(float(base.scene.env_origins[:,2].mean()), 4),
            "tilt_deg_mean": round(float(tilt.mean()), 2),
            "tilt_deg_max": round(float(tilt.max()), 2),
            "support_fraction": round(force / weight, 3),
            "upright_and_supported": bool(tilt.mean() < 20.0 and 0.8 < force/weight < 1.25),
        })
        env.close()
except Exception:
    import traceback
    rows.append({"error": traceback.format_exc()[-800:]})
with open(a.report, "w") as h: json.dump({"rows": rows}, h, indent=2)
app.close()
