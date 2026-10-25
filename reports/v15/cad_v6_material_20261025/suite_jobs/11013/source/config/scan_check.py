"""Is the height scan actually informative, or clipped flat?

Codex flagged that mdp.height_scan returns `sensor.data.pos_w.z - hit_z - offset`
where pos_w is the PARENT BODY position. This asset's root frame sits at
z = -0.4192 while the robot stands on the ground, so the stock 0.5 offset would
put flat ground at -0.919 and the [-1, 1] observation clip would saturate after
~8 cm of rise. The fix sets offset = -0.4192. This measures whether it worked.
"""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--report", default="/tmp/scan_check.json")
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym, torch
from v10_1_env_cfg import GuideV2BaselineEnvCfg
from v10_1_ppo_cfg import V10_1_PPO_CFG

TASK = "Isaac-Scan-Check-v10-1"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": GuideV2BaselineEnvCfg, "rsl_rl_cfg_entry_point": V10_1_PPO_CFG})
out = {"ok": False}
try:
    cfg = GuideV2BaselineEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.seed = 7
    # measure the raw signal, without the observation noise
    cfg.observations.policy.enable_corruption = False
    env = gym.make(TASK, cfg=cfg); env.reset()
    base = env.unwrapped
    term = base.cfg.observations.policy.height_scan
    out["scan_func"] = getattr(term.func, "__name__", str(term.func))
    out["scan_params"] = sorted(term.params)
    out["clip"] = list(base.cfg.observations.policy.height_scan.clip)
    action = torch.zeros((a.num_envs, 12), device=base.device)
    for _ in range(60):
        env.step(action)
    obs = base.observation_manager.compute()["policy"]
    scan = obs[:, 48:]                      # 187 scan points follow the 48 proprio
    out["scan_width"] = int(scan.shape[1])
    lo, hi = base.cfg.observations.policy.height_scan.clip
    out["mean"] = round(float(scan.mean()), 4)
    out["std"] = round(float(scan.std()), 4)
    out["min"] = round(float(scan.min()), 4)
    out["max"] = round(float(scan.max()), 4)
    out["frac_at_low_clip"] = round(float((scan <= lo + 1e-4).float().mean()), 4)
    out["frac_at_high_clip"] = round(float((scan >= hi - 1e-4).float().mean()), 4)
    out["saturated_fraction"] = round(out["frac_at_low_clip"] + out["frac_at_high_clip"], 4)
    # A useful scan varies across the grid and is mostly inside the clip.
    out["ok"] = out["saturated_fraction"] < 0.25 and out["std"] > 0.01
    env.close()
except Exception:
    import traceback; out["error"] = traceback.format_exc()[-1200:]
with open(a.report, "w") as h: json.dump(out, h, indent=2)
app.close()
