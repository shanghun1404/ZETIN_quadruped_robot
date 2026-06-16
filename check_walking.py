import argparse
import os
import gymnasium as gym
from isaaclab.app import AppLauncher

# 1. 앱 런처 설정 (렌더링 관련 기능을 완전히 끔)
parser = argparse.ArgumentParser(description="Verify walking via coordinates.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 필수 라이브러리 임포트
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import importlib.metadata as metadata

from for_env_cfg import ForFlatEnvCfg
from for_ppo_cfg import ForFlatPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-For-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForFlatEnvCfg,
        "rsl_rl_cfg_entry_point": ForFlatPPORunnerCfg,
    },
)

def main():
    # 환경 설정
    env_cfg = ForFlatEnvCfg()
    # evaluation시 랜덤 커맨드 대신 앞으로 무조건 직진하도록 설정 고정
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, -1.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(env_cfg.commands.base_velocity.ranges, "heading"):
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.scene.num_envs = 1  
    agent_cfg = ForFlatPPORunnerCfg()
    
    # 설정 변환
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 환경 생성 (렌더링 모드 없음)
    env = gym.make("Isaac-Velocity-Flat-For-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 최신 모델 찾기
    model_path = "/home/scienceowl/quadruped_training/logs/rsl_rl/for_flat/2026-06-03_07-08-40/model_7999.pt"

    print(f"[INFO] Loading model: {model_path}")

    # 모델 로드
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="temp", device=agent_cfg.device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    
    obs_results = env.get_observations()
    if isinstance(obs_results, dict):
        obs = obs_results["policy"]
    elif isinstance(obs_results, tuple):
        obs = obs_results[0]
    else:
        obs = obs_results

    start_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2].clone()
    
    print("\n" + "="*50)
    print("VERIFICATION RUN: CHECKING IF THE ROBOT ACTUALLY WALKS")
    print("="*50)
    
    rnn_states = None
    for i in range(3001):
        if rnn_states is None:
            # First step without rnn state or if it resets we might need to reset state. But OnPolicyRunner might handle it.
            # RSL_RL's get_inference_policy returns a function. We can just use the underlying module.
            if hasattr(runner.alg, "actor_critic") and hasattr(runner.alg.actor_critic, "act_inference"):
                # Use the module directly
                pass
                
        # Actually just try calling policy(obs) but we might need to reset it. Wait, RSL-RL get_inference_policy maintains state internally!
        actions = policy(obs)
        step_results = env.step(actions)
        obs_res = step_results[0]
        if isinstance(obs_res, dict):
            obs = obs_res.get("policy", obs_res)
        else:
            obs = obs_res
        
        if i % 100 == 0:
            current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :2]
            dist = torch.norm(current_pos - start_pos)
            print(f"Step {i:4d}: Pos [x={current_pos[0]:.2f}, y={current_pos[1]:.2f}], Moved: {dist.item():.2f} meters")
        
        reset_buf = env.unwrapped.reset_buf[0]
        if reset_buf:
            print(f"  -> ENV RESET at step {i}!")
            # reset rnn internal state if policy supports it
            if hasattr(policy, "reset"):
                policy.reset()

    print("="*50)
    print("Verification finished! If 'Moved' distance increased, your robot is walking perfectly.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
