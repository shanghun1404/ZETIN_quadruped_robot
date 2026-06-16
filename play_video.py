import argparse
import os
import gymnasium as gym
from isaaclab.app import AppLauncher

# 1. 앱 런처 설정
import sys

parser = argparse.ArgumentParser(description="Play and record video of the trained agent.")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()

# 노트북 환경을 위해 설정을 극단적으로 낮춤
args_cli.enable_cameras = True 
args_cli.headless = True
args_cli.width = 160 # 더 작게
args_cli.height = 120 # 더 작게

# 메모리 예산을 강제로 줄이는 설정을 kit_args에 추가
kit_settings = [
    "--/rtx/sceneDb/maxMemoryBudget=512",
    "--/rtx/sceneDb/maxTlasSize=256",
    "--/rtx/sceneDb/maxBlasSize=256",
    "--/renderer/multiGpu/enabled=false",
    "--/app/renderer/resolution/width=160",
    "--/app/renderer/resolution/height=120",
    "--/rtx/raytracing/enabled=false", # 아예 끄기 시도
]
args_cli.kit_args = " ".join(kit_settings)

# 렌더링 설정을 직접 주입하여 메모리 절약
import omni
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 시뮬레이션 앱 실행 후 설정을 더 낮춤
import omni.kit.commands
omni.kit.commands.execute("SetSetting", setting="/rtx/realtime/enabled", value=False) # 리얼타임 끔
omni.kit.commands.execute("SetSetting", setting="/rtx/hydra/enabled", value=True)

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
    env_cfg.scene.num_envs = 1  
    # evaluation시 랜덤 커맨드 대신 앞으로 무조건 직진하도록 설정 고정
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, -1.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) # 제자리 직진을 위해 0으로 변경
    if hasattr(env_cfg.commands.base_velocity.ranges, "heading"):
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    agent_cfg = ForFlatPPORunnerCfg()
    
    # 가상환경 버전 확인 및 설정 변환
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # 환경 생성 및 비디오 레코더 래핑
    video_dir = os.path.abspath("output_video")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    env = gym.make("Isaac-Velocity-Flat-For-v0", cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=video_dir, 
        step_trigger=lambda step: step == 0, 
        video_length=500 
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 확실하게 성공한 최신 모델 가중치로 강제 지정 (다른 컴퓨터 이동 시 에러 방지)
    model_path = os.path.abspath("logs/rsl_rl/for_flat/2026-06-03_07-08-40/model_7999.pt")

    print(f"[INFO] Loading model from: {model_path}")

    # 러너 생성 및 모델 로드
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=video_dir, device=agent_cfg.device)
    runner.load(model_path)
    
    # 테스트 실행 (Policy 추론)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    obs_results = env.get_observations()
    if isinstance(obs_results, dict):
        obs = obs_results["policy"]
    elif isinstance(obs_results, tuple):
        obs = obs_results[0]
    else:
        obs = obs_results
    
    print("[INFO] Recording started...")
    for _ in range(500):
        actions = policy(obs)
        step_results = env.step(actions)
        obs_res = step_results[0]
        if isinstance(obs_res, dict):
            obs = obs_res.get("policy", obs_res)
        else:
            obs = obs_res
    
    print(f"[INFO] Recording finished! Video saved in: {video_dir}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
