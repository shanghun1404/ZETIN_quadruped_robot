"""Resume adopted V15 PPO on unchanged CAD and a ROS2-equivalent action path."""
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
p.add_argument('--actor',required=True);p.add_argument('--student',required=True)
p.add_argument('--usd',required=True);p.add_argument('--output',required=True)
p.add_argument('--num_envs',type=int,default=128);p.add_argument('--iterations',type=int,default=2000)
p.add_argument('--terrain',choices=['flat','mixed'],default='mixed')
p.add_argument('--profile',choices=['balanced','targeted','rms','rms_retain'],default='balanced')
p.add_argument('--retention-teacher',default=None)
p.add_argument('--retention-weight',type=float,default=5.)
p.add_argument('--seed',type=int,default=73);p.add_argument('--learning-rate',type=float,default=5e-5)
p.add_argument('--minimum-policy-std',type=float,default=.18)
p.add_argument('--max-delay-steps',type=int,default=3)
p.add_argument('--preflight-steps',type=int,default=100)
p.add_argument('--contract-only',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
faulthandler.dump_traceback_later(90.,repeat=True)
print('Starting CAD fine-tuning application',flush=True)
app=AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from pxr import UsdPhysics
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner
from v14_leg_ppo_cfg import V14_LEG_PPO_CFG
from v13_depth_policy import make_v13_depth_wrapper
from v15_math import make_finite_rollout
from v15_cad_training_env import GuideV15CadEnvCfg,enable_stair_curriculum,enable_targeted_finetune,enable_rms_finetune,ground_at,MODEL_METADATA
from v15_cad_training_mdp import cad_self_contact_filters
from v15_cad_training_preflight import validate_preflight_state
from v14_ros2_control import contract


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,record):Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def main():
    if args.num_envs<1 or args.iterations<1 or args.preflight_steps<1 or not 0<args.learning_rate<.001:
        raise ValueError('Invalid training limits')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    cfg=GuideV15CadEnvCfg()
    cfg.scene.robot.spawn.usd_path=args.usd
    cfg.scene.num_envs=args.num_envs
    cfg.seed=args.seed
    cfg.actions.joint_pos.max_delay_steps=args.max_delay_steps
    if args.terrain=='mixed':
        enable_stair_curriculum(cfg)
        # One robot per column prevents overlapping peers in the depth images.
        cfg.scene.terrain.terrain_generator.num_cols=args.num_envs
    if args.profile=='targeted':enable_targeted_finetune(cfg)
    if args.profile in ('rms','rms_retain'):
        if args.terrain!='mixed':raise ValueError('RMS profile requires its flat/rough generated terrain')
        enable_rms_finetune(cfg)
    agent=copy.deepcopy(V14_LEG_PPO_CFG)
    agent.experiment_name='v15_cad_finetune'
    agent.run_name=args.terrain+'_'+args.profile+'_commands_rated_load'
    agent.seed=args.seed
    agent.algorithm.learning_rate=args.learning_rate;agent.algorithm.schedule='fixed'
    if args.profile=='rms_retain':
        if not args.retention_teacher or not np.isfinite(args.retention_weight) or args.retention_weight<=0.:
            raise ValueError('Retention requires its passing teacher and positive weight')
        from v15_cad_policy_retention import RetentionPPO
        import rsl_rl.runners.on_policy_runner as runner_module
        runner_module.RetentionPPO=RetentionPPO
        agent.algorithm.class_name='RetentionPPO'
    agent.save_interval=100
    gym.register(id='Isaac-V15-CAD-Finetune',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True)
    raw=gym.make('Isaac-V15-CAD-Finetune',cfg=cfg)
    env=make_finite_rollout(make_v13_depth_wrapper())(raw,student_checkpoint=args.student,clip_actions=agent.clip_actions)
    faulthandler.cancel_dump_traceback_later()
    robot=raw.unwrapped.scene['robot'];action=raw.unwrapped.action_manager.get_term('joint_pos')
    metadata=json.loads(MODEL_METADATA.read_text())
    masses=robot.root_physx_view.get_masses().sum(-1)
    assert len(robot.body_names)==13 and torch.allclose(masses,torch.full_like(masses,metadata['source_total_mass_kg']),atol=5e-5,rtol=0.)
    assert digest(args.usd)==metadata['source_usd_root_sha256']
    root_prim=raw.unwrapped.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
    from pxr import Usd
    colliders=sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(root_prim,Usd.TraverseInstanceProxies()))
    assert colliders==67,colliders
    observations=env.get_observations()
    assert observations['policy'].shape==(args.num_envs,235)
    validate_preflight_state(env.nonfinite_steps,robot.data,observations)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(output),device=cfg.sim.device)
    runner.load(args.actor,load_optimizer=True,map_location=cfg.sim.device)
    if args.profile=='rms_retain':
        teacher=copy.deepcopy(runner.alg.policy)
        teacher.load_state_dict(torch.load(args.retention_teacher,map_location=cfg.sim.device,weights_only=False)['model_state_dict'])
        teacher.requires_grad_(False);teacher.eval()
        runner.alg.retention_teacher=teacher
        runner.alg.retention_weight=args.retention_weight
    start=runner.current_learning_iteration+1;runner.current_learning_iteration=start
    before_std=runner.alg.policy.std.detach().cpu().tolist()
    with torch.no_grad():runner.alg.policy.std.clamp_(min=args.minimum_policy_std)
    runner.alg.learning_rate=args.learning_rate;runner.alg.schedule='fixed'
    for group in runner.alg.optimizer.param_groups:group['lr']=args.learning_rate
    inherited_config=raw.unwrapped.reward_manager
    command_cfg=cfg.commands.base_velocity
    terrain_columns={}
    generator=cfg.scene.terrain.terrain_generator
    if generator is not None:
        from v15_cad_training_env import _terrain_kind_mask
        terrain_columns={row.kind:int(_terrain_kind_mask(raw.unwrapped,(row.kind,)).sum())
                         for row in generator.sub_terrains.values()}
    if args.profile in ('rms','rms_retain'):
        assert set(terrain_columns)=={'flat','rough'}
        assert float(raw.unwrapped.scene.terrain.terrain_levels.max())==0.
        # Check the analytical ground lookup against the authored box height.
        from v15_cad_training_terrain import rough_tile_specs
        tile=rough_tile_specs()[0]
        point=raw.unwrapped.scene.env_origins.clone()
        point[:,0]+=float(tile[0]);point[:,1]+=float(tile[1])
        height=ground_at(raw.unwrapped,point)-raw.unwrapped.scene.env_origins[:,2]
        rough_mask=_terrain_kind_mask(raw.unwrapped,('rough',))
        assert bool(torch.allclose(height[rough_mask],torch.full_like(height[rough_mask],float(tile[4])),atol=1e-6,rtol=0.))
        assert bool((height[~rough_mask]==0.).all())
    manifest=dict(schema=1,training_seed=args.seed,terrain=args.terrain,profile=args.profile,num_envs=args.num_envs,
        command_probabilities=list(command_cfg.command_probabilities),
        command_hold_seconds=list(command_cfg.resampling_time_range),
        terrain_forward_fraction_start=command_cfg.force_forward_terrain_fraction_start,
        rough_training_seed=73 if args.profile in ('targeted','rms','rms_retain') and args.terrain=='mixed' else None,
        retention_teacher_sha256=digest(args.retention_teacher) if args.profile=='rms_retain' else None,
        retention_weight=args.retention_weight if args.profile=='rms_retain' else None,
        retention_scope='Forward/zero commands only; frozen actor mean targets' if args.profile=='rms_retain' else None,
        terrain_columns=terrain_columns,
        iterations_requested=args.iterations,starting_iteration=start,actor=str(Path(args.actor).resolve()),
        actor_sha256=digest(args.actor),student_sha256=digest(args.student),usd_sha256=digest(args.usd),
        mass_min_max_kg=[float(masses.min()),float(masses.max())],body_count=len(robot.body_names),collider_count=colliders,
        action_joint_order=list(contract.JOINT_NAMES),physical_joint_order=robot.joint_names,
        policy_std_before=before_std,policy_std_after=runner.alg.policy.std.detach().cpu().tolist(),
        exploration_std_semantics='Initial boost only; std remains trainable during PPO',
        restored_optimizer=True,optimizer_groups_lr=[group['lr'] for group in runner.alg.optimizer.param_groups],
        learning_rate=runner.alg.learning_rate,schedule=runner.alg.schedule,
        physics_dt=.005,policy_dt=.02,cutoff_hz=action.cfg.cutoff_hz,
        two_ros2_guards=[action.cfg.effort_guard_fraction,action.cfg.walk_guard_fraction],
        max_delay_physics_steps=action.cfg.max_delay_steps,
        training_rms_time_constant_s=action.cfg.rms_time_constant_s,
        stiffness_min_max=[float(robot.data.joint_stiffness.min()),float(robot.data.joint_stiffness.max())],
        damping_min_max=[float(robot.data.joint_damping.min()),float(robot.data.joint_damping.max())],
        self_collision_filters=cad_self_contact_filters(),
        reward_weights={name:inherited_config.get_term_cfg(name).weight for name in inherited_config.active_terms},
        current_semantics='Pre-step bounded PD motor phase estimate; contact reactions are separate',
        adoption='Candidate only; original339 and unchanged physical acceptance retained')
    write(output/'training_contract.json',manifest)
    dump_yaml(str(output/'params/env.yaml'),cfg);dump_yaml(str(output/'params/agent.yaml'),agent)
    print('CAD TRAINING CONTRACT '+json.dumps(manifest),flush=True)
    checks=[]
    with torch.inference_mode():
        for step in range(args.preflight_steps):
            actions=runner.alg.policy.act_inference(observations)
            observations,rewards,dones,extras=env.step(actions)
            validate_preflight_state(env.nonfinite_steps,robot.data,observations,rewards,
                raw.unwrapped.termination_manager.get_term('physics_diverged'))
            limits=torch.tensor([contract.joint_effort_limit(name) for name in contract.JOINT_NAMES],device=env.device)
            assert bool((action.estimated_torque_policy.abs()<=limits+1e-4).all())
            assert bool(torch.isfinite(action.estimated_torque_rms_policy).all())
            assert bool(((action.estimated_torque_rms_policy>=0.)&(action.estimated_torque_rms_policy<=limits+1e-4)).all())
            if step%25==0:
                checks.append(dict(step=step,finite=True,episode_resets=int(dones.sum()),
                    reward_mean=float(rewards.mean()),estimated_torque_abs_max_nm=float(action.estimated_torque_policy.abs().max()),
                    contact_force_count=int((raw.unwrapped.reward_manager.get_term_cfg('self_contact_count').func(raw.unwrapped)>0).sum())))
    write(output/'preflight.json',dict(passed=True,steps=args.preflight_steps,
        nonfinite_steps=env.nonfinite_steps,raw_state_checked=True,checks=checks))
    print('CAD PREFLIGHT PASS '+json.dumps(checks),flush=True)
    if args.contract_only:
        runner.save(str(output/'contract_actor.pt'))
        write(output/'completed.json',dict(contract_only=True,training_completed=False))
    else:
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        checkpoint=output/f'model_{runner.current_learning_iteration}.pt'
        assert checkpoint.is_file()
        assert all(bool(torch.isfinite(value).all()) for value in runner.alg.policy.state_dict().values())
        assert bool((runner.alg.policy.std>0).all())
        write(output/'completed.json',dict(training_completed=True,updates=args.iterations,
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),last_iteration=runner.current_learning_iteration))
        print('CAD TRAINING FINISHED '+str(checkpoint),flush=True)
    writer=getattr(runner,'writer',None)
    if writer is not None:writer.flush();writer.close()
    torch.cuda.synchronize()
    sys.stdout.flush();sys.stderr.flush()
    # Previous Kit teardown held theGPUafter checkpoints; Slurm handles process cleanup.
    os._exit(0)


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
