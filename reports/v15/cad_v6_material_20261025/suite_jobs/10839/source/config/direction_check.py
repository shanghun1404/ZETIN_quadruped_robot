"""Which way does the new robot's head point, in its own base frame?

The v1 robot's head pointed along base -x, which is why the whole V8+ line
converts guide-forward speeds to negative base lin_vel_x. That was established by
measuring foot positions, not assumed, and it matters: a sign error trains the
robot to walk backwards while every metric still looks healthy.

The MJCF cannot answer this -- its body transforms are all (0,0,0) with the
geometry baked into the meshes -- so measure it on the loaded articulation.
"""
import argparse, json
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--report", default="/tmp/direction.json")
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
import gymnasium as gym, torch
from v10_flat_env_cfg import GuideV2FlatEnvCfg
from v10_flat_ppo_cfg import V10_FLAT_PPO_CFG
TASK = "Isaac-Direction-Check-v10-flat"
gym.register(id=TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": GuideV2FlatEnvCfg, "rsl_rl_cfg_entry_point": V10_FLAT_PPO_CFG})
out = {}
try:
    cfg = GuideV2FlatEnvCfg(); cfg.scene.num_envs = 8; cfg.seed = 1
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0,0.0), "y": (0.0,0.0), "yaw": (0.0,0.0)}
    env = gym.make(TASK, cfg=cfg); env.reset()
    base = env.unwrapped; robot = base.scene["robot"]
    action = torch.zeros((8, 12), device=base.device)
    for _ in range(200):
        env.step(action)
    names = list(robot.body_names)
    root_p = robot.data.root_pos_w[0]
    root_q = robot.data.root_quat_w[0]
    from isaaclab.utils.math import quat_apply_inverse
    def body_in_base(name):
        w = robot.data.body_pos_w[0, names.index(name)] - root_p
        return quat_apply_inverse(root_q.unsqueeze(0), w.unsqueeze(0))[0]
    feet = {}
    for f in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        v = body_in_base(f)
        feet[f] = [round(float(v[i]), 4) for i in range(3)]
    out["feet_in_base_frame"] = feet
    front_x = (feet["FL_foot"][0] + feet["FR_foot"][0]) / 2
    rear_x  = (feet["RL_foot"][0] + feet["RR_foot"][0]) / 2
    out["front_pair_mean_x"] = round(front_x, 4)
    out["rear_pair_mean_x"] = round(rear_x, 4)
    out["head_points_along"] = "+x" if front_x > rear_x else "-x"
    out["v1_convention_head_along"] = "-x"
    out["same_as_v1"] = out["head_points_along"] == "-x"
    for b in ("body_head", "body_front", "body_back"):
        if b in names:
            v = body_in_base(b)
            out[f"{b}_in_base"] = [round(float(v[i]), 4) for i in range(3)]
    out["ok"] = True
    env.close()
except Exception:
    import traceback; out["error"] = traceback.format_exc()[-1200:]
with open(a.report, "w") as h: json.dump(out, h, indent=2)
app.close()
