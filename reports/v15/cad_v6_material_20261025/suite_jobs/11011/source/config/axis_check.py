"""Which base-frame axis is the robot's fore-aft axis?

Per-body frames on this asset are individually scrambled (a foot reported z
+0.19, above the trunk), so single-body positions are unreliable. Two independent
signals are used instead:
  1. the foot GROUPING -- which axis separates the front pair from the rear pair;
  2. the whole-robot bounding box in the base frame -- a quadruped is longer
     fore-aft than it is wide.
"""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--report", default="/tmp/axis.json")
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
import gymnasium as gym, torch
from isaaclab.utils.math import quat_apply_inverse
from v10_flat_env_cfg import GuideV2FlatEnvCfg
from v10_flat_ppo_cfg import V10_FLAT_PPO_CFG
TASK = "Isaac-Axis-Check-v10-flat"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": GuideV2FlatEnvCfg, "rsl_rl_cfg_entry_point": V10_FLAT_PPO_CFG})
out = {}
try:
    cfg = GuideV2FlatEnvCfg(); cfg.scene.num_envs = 4; cfg.seed = 1
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0,0.0), "y": (0.0,0.0), "yaw": (0.0,0.0)}
    env = gym.make(TASK, cfg=cfg); env.reset()
    base = env.unwrapped; robot = base.scene["robot"]
    for _ in range(200):
        env.step(torch.zeros((4, 12), device=base.device))
    names = list(robot.body_names)
    rp, rq = robot.data.root_pos_w[0], robot.data.root_quat_w[0]
    local = quat_apply_inverse(rq.unsqueeze(0).expand(len(names), -1),
                              robot.data.body_pos_w[0] - rp)
    def g(n): return local[names.index(n)]
    front = (g("FL_foot") + g("FR_foot")) / 2
    rear  = (g("RL_foot") + g("RR_foot")) / 2
    left  = (g("FL_foot") + g("RL_foot")) / 2
    right = (g("FR_foot") + g("RR_foot")) / 2
    out["front_minus_rear"] = [round(float(v), 4) for v in (front - rear)]
    out["left_minus_right"] = [round(float(v), 4) for v in (left - right)]
    fr = (front - rear).abs()
    out["fore_aft_axis"] = ["x", "y", "z"][int(fr[:2].argmax())]
    out["head_sign"] = "+" if float((front - rear)[int(fr[:2].argmax())]) > 0 else "-"
    # whole-robot extent in the base frame
    ext = local.max(dim=0).values - local.min(dim=0).values
    out["body_extent_xyz"] = [round(float(v), 4) for v in ext]
    out["longest_horizontal_axis"] = ["x", "y"][int(ext[:2].argmax())]
    out["ok"] = True
    env.close()
except Exception:
    import traceback; out["error"] = traceback.format_exc()[-1000:]
with open(a.report, "w") as h: json.dump(out, h, indent=2)
app.close()
