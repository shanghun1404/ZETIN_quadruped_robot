"""Decompose the height scan: what is pos_w, what is hit_z, and when does it settle?"""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--report", default="/tmp/scan_parts.json")
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
import gymnasium as gym, torch
from v10_1_env_cfg import GuideV2BaselineEnvCfg
from v10_1_ppo_cfg import V10_1_PPO_CFG
TASK = "Isaac-Scan-Parts-v10-1"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": GuideV2BaselineEnvCfg, "rsl_rl_cfg_entry_point": V10_1_PPO_CFG})
out = {}
try:
    cfg = GuideV2BaselineEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = 7
    cfg.observations.policy.enable_corruption = False
    env = gym.make(TASK, cfg=cfg); env.reset()
    base = env.unwrapped
    robot = base.scene["robot"]; scanner = base.scene.sensors["height_scanner"]
    action = torch.zeros((a.num_envs, 12), device=base.device)
    trace = []
    for step in range(1000):
        env.step(action)
        if step in (0, 100, 300, 600, 999):
            hits = scanner.data.ray_hits_w[..., 2]
            finite = torch.isfinite(hits)
            trace.append({
                "step": step,
                "sensor_pos_w_z": round(float(scanner.data.pos_w[:, 2].mean()), 4),
                "robot_root_z":   round(float(robot.data.root_pos_w[:, 2].mean()), 4),
                "env_origin_z":   round(float(base.scene.env_origins[:, 2].mean()), 4),
                "hit_z_mean":     round(float(hits[finite].mean()), 4),
                "pos_minus_hit":  round(float((scanner.data.pos_w[:, 2].unsqueeze(1) - hits)[finite].mean()), 4),
                "min_foot_z":     round(float(robot.data.body_pos_w[:, [robot.body_names.index(f"{s}_foot") for s in ("FL","FR","RL","RR")], 2].min()), 4),
            })
    out["trace"] = trace
    out["scanner_offset_cfg"] = list(base.cfg.scene.height_scanner.offset.pos)
    out["obs_offset_param"] = float(base.cfg.observations.policy.height_scan.params["offset"])
    # what offset would centre the scan at the settled state?
    last = trace[-1]
    out["suggested_obs_offset"] = round(last["pos_minus_hit"], 4)
    out["ok"] = True
    env.close()
except Exception:
    import traceback; out["error"] = traceback.format_exc()[-1200:]
with open(a.report, "w") as h: json.dump(out, h, indent=2)
app.close()
