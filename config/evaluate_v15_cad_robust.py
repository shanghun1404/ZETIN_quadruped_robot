"""Compare V15 CAD actors under pinned push, motor-delay and sensor-noise settings.

Each actor runs every scenario on the CAD training environment (rms flat/rough
columns, the same command distribution) from the same seed. Falls are
``base_tilt``, ``low_height`` or ``physics_diverged`` terminations; ``tile_exit``
and time-outs are counted separately and are not falls.
"""
import argparse
import copy
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--actor',action='append',required=True,help='label=checkpoint; repeat per actor')
p.add_argument('--student',required=True);p.add_argument('--usd',required=True);p.add_argument('--output',required=True)
p.add_argument('--num_envs',type=int,default=128);p.add_argument('--steps',type=int,default=1500)
p.add_argument('--seed',type=int,default=2026);p.add_argument('--scenarios',default='')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
faulthandler.dump_traceback_later(90.,repeat=True)
app=AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from v14_leg_ppo_cfg import V14_LEG_PPO_CFG
from v13_depth_policy import make_v13_depth_wrapper
from v15_math import make_finite_rollout
from v15_cad_training_env import GuideV15CadEnvCfg,enable_stair_curriculum,enable_rms_finetune
from v15_cad_robust import RobustSettings,ROBUST_END,MOTOR_END,PUSH_INTERVAL_S,enable_robust_finetune,make_noisy_proprioception,settings_record

FALLS=('base_tilt','low_height','physics_diverged')
SCENARIOS={
    'nominal':RobustSettings(),
    'push_0.5':RobustSettings(push_speed_mps=(.5,.5)),
    'push_1.0':RobustSettings(push_speed_mps=(1.,1.)),
    'push_1.5':RobustSettings(push_speed_mps=(1.5,1.5)),
    'delay_30ms':RobustSettings(delay_steps=(6,6)),
    'delay_40ms':RobustSettings(delay_steps=(8,8)),
    'missed_10pct':RobustSettings(hold_probability=.10),
    'noise_1x':RobustSettings(noise_scale=1.),
    'noise_2x':RobustSettings(noise_scale=2.),
    'combined_train_end':ROBUST_END,
    'combined_hard':RobustSettings(push_speed_mps=(1.,1.),delay_steps=(6,6),hold_probability=.10,noise_scale=1.5),
    # Motor side (motor profile): per-step command jitter, old feedback, lagged motor torque.
    'delay_jitter':RobustSettings(delay_steps=(2,2),delay_jitter_steps=2),
    'feedback_10ms':RobustSettings(feedback_delay_steps=(2,2)),
    'motor_lag_10ms':RobustSettings(motor_lag_s=(.01,.01)),
    'motor_lag_20ms':RobustSettings(motor_lag_s=(.02,.02)),
    'combined_motor_end':MOTOR_END,
    'combined_motor_hard':RobustSettings(push_speed_mps=(1.,1.),delay_steps=(4,4),delay_jitter_steps=2,hold_probability=.10,
        noise_scale=1.5,feedback_delay_steps=(2,2),motor_lag_s=(.01,.01)),
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(env,raw,policy,settings,seed):
    unwrapped=raw.unwrapped
    unwrapped.v15_robust_settings=settings
    torch.manual_seed(seed);np.random.seed(seed)
    terminations=unwrapped.termination_manager
    counts={name:0 for name in terminations.active_terms}
    count=env.num_envs
    error_xy=torch.zeros((),device=env.device);error_yaw=torch.zeros((),device=env.device)
    # Steps create inference-mode state buffers, so later resets must run in the same mode.
    with torch.inference_mode():
        observations,_=env.reset()
        for _ in range(args.steps):
            command=unwrapped.command_manager.get_command('base_velocity')
            data=unwrapped.scene['robot'].data
            error_xy+=torch.linalg.vector_norm(command[:,:2]-data.root_lin_vel_b[:,:2],dim=-1).sum()
            error_yaw+=(command[:,2]-data.root_ang_vel_b[:,2]).abs().sum()
            observations,_,dones,_=env.step(policy(observations))
            for name in counts:counts[name]+=int(terminations.get_term(name).sum())
    falls=sum(counts[name] for name in FALLS if name in counts)
    ended=sum(counts.values())
    robot_seconds=count*args.steps*unwrapped.step_dt
    return dict(settings=settings_record(settings),terminations=counts,falls=falls,
        falls_per_robot_minute=falls/(robot_seconds/60.),fall_fraction_of_ended_episodes=falls/ended if ended else 0.,
        mean_planar_velocity_error_mps=float(error_xy)/(count*args.steps),
        mean_yaw_rate_error_radps=float(error_yaw)/(count*args.steps),
        nonfinite_steps=env.nonfinite_steps,robot_seconds=robot_seconds)


def main():
    actors=[]
    for item in args.actor:
        label,_,path=item.partition('=')
        if not label or not Path(path).is_file():raise ValueError(f'bad actor {item}')
        actors.append((label,path))
    names=[name for name in args.scenarios.split(',') if name] or list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:raise ValueError(f'unknown scenario {name}')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cfg=GuideV15CadEnvCfg()
    cfg.scene.robot.spawn.usd_path=args.usd
    cfg.scene.num_envs=args.num_envs
    cfg.seed=args.seed
    # Same order as train_v15_cad.py --terrain mixed --profile rms_retain.
    enable_stair_curriculum(cfg)
    cfg.scene.terrain.terrain_generator.num_cols=args.num_envs
    enable_rms_finetune(cfg)
    enable_robust_finetune(cfg,ramp_steps=0)
    # Pinned settings: no ramp term, so the scenario values stay fixed.
    cfg.curriculum=None
    cfg.actions.joint_pos.max_delay_steps=max(settings.delay_steps[1]+settings.delay_jitter_steps for settings in SCENARIOS.values())
    cfg.actions.joint_pos.max_feedback_delay_steps=max(settings.feedback_delay_steps[1] for settings in SCENARIOS.values())
    agent=copy.deepcopy(V14_LEG_PPO_CFG)
    gym.register(id='Isaac-V15-CAD-Robust-Eval',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True)
    raw=gym.make('Isaac-V15-CAD-Robust-Eval',cfg=cfg)
    raw.unwrapped.v15_robust_settings=RobustSettings()
    env=make_finite_rollout(make_noisy_proprioception(make_v13_depth_wrapper()))(
        raw,student_checkpoint=args.student,clip_actions=agent.clip_actions)
    faulthandler.cancel_dump_traceback_later()
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=cfg.sim.device)
    record=dict(schema=1,seed=args.seed,num_envs=args.num_envs,steps=args.steps,step_dt=raw.unwrapped.step_dt,
        push_interval_s=list(PUSH_INTERVAL_S),falls=list(FALLS),student_sha256=digest(args.student),usd_sha256=digest(args.usd),
        actors={label:dict(path=str(Path(path).resolve()),sha256=digest(path)) for label,path in actors},results={})
    for label,path in actors:
        runner.load(path,load_optimizer=False,map_location=cfg.sim.device)
        policy=runner.get_inference_policy(device=cfg.sim.device)
        record['results'][label]={}
        for name in names:
            result=run(env,raw,policy,SCENARIOS[name],args.seed)
            record['results'][label][name]=result
            print('ROBUST EVAL '+json.dumps(dict(actor=label,scenario=name,falls=result['falls'],
                falls_per_robot_minute=result['falls_per_robot_minute'],
                planar_error=result['mean_planar_velocity_error_mps'])),flush=True)
            (output/'robust_eval.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    record['completed']=True
    (output/'robust_eval.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('ROBUST EVAL FINISHED',flush=True)
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
