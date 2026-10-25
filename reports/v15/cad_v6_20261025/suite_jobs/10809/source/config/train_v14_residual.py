"""Create a zero residual or train a bounded residual with immutable base/student."""

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--actor-checkpoint', required=True)
parser.add_argument('--student-checkpoint', required=True)
parser.add_argument('--usd-path', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--seed', type=int, default=46)
parser.add_argument('--iterations', type=int, default=300)
parser.add_argument('--save-interval', type=int, default=50)
parser.add_argument('--resume-residual', help='Validated trained residual checkpoint to continue')
parser.add_argument('--zero-only', action='store_true')
parser.add_argument('--zero-proof', help='Comparison JSON required before training')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1 or args.save_interval < 1:
    parser.error('positive environment and iteration counts required')
if args.zero_only and args.resume_residual:
    parser.error('--zero-only cannot resume a trained residual')
if not args.zero_only and not args.zero_proof:
    parser.error('--zero-proof is required before training')
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

from v13_depth_policy import make_v13_depth_wrapper
from v14_leg_ppo_cfg import V14_LEG_PPO_CFG
from v14_residual_env_cfg import GuideV14ResidualEnvCfg
from v14_residual_policy import (ResidualPolicyWrapper, ResidualOnPolicyRunner, file_digest, initialize_residual_actor,
                                 load_residual_manifest, module_digest, residual_agent_cfg, resume_residual,
                                 save_initial_residual, seal_residual_checkpoints)


def main():
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    hashes = dict(base_sha256=file_digest(args.actor_checkpoint), student_sha256=file_digest(args.student_checkpoint),
                  usd_sha256=file_digest(args.usd_path))
    if not args.zero_only:
        proof = json.loads(Path(args.zero_proof).read_text())
        if proof.get('passed') is not True or any(proof.get(k) != v for k,v in hashes.items()):
            raise ValueError('missing/mismatched zero-residual simulation proof')
    if args.resume_residual:
        load_residual_manifest(args.resume_residual, base_checkpoint=args.actor_checkpoint,
                               student_checkpoint=args.student_checkpoint, usd_path=args.usd_path,
                               heading_control=True)
    cfg = GuideV14ResidualEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.robot.spawn.usd_path = args.usd_path
    cfg.observations.policy.enable_corruption = False
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator.num_cols = args.num_envs
    cfg.seed = args.seed
    if cfg.rewards.base_angular_velocity.params['std'] != 2.0 or cfg.rewards.base_angular_velocity.weight != 5.:
        raise ValueError('residual pilot must preserve original yaw reward')
    agent_cfg = residual_agent_cfg(V14_LEG_PPO_CFG)
    agent_cfg.seed = args.seed
    agent_cfg.device = cfg.sim.device
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = args.save_interval
    manifest = dict(schema=1, **hashes, base_checkpoint=args.actor_checkpoint,
                    student_checkpoint=args.student_checkpoint, usd_path=args.usd_path,
                    max_rad=.05, action_scale=.2, observation_dim=307, actor_hidden_dims=[128,128],
                    heading_control=True, num_envs=args.num_envs, seed=args.seed, iterations=args.iterations,
                    save_interval=args.save_interval, rollout_steps_per_env=agent_cfg.num_steps_per_env,
                    additional_transitions=0 if args.zero_only else args.num_envs*agent_cfg.num_steps_per_env*args.iterations,
                    zero_only=args.zero_only, zero_proof=args.zero_proof, learning_rate=1e-4,
                    initial_noise_std=.05, noise_std_type='log', entropy_coef=0., initial_terrain_level=0,
                    yaw_reward_std=2., yaw_reward_weight=5., effort_guard_fraction=.99, action_lpf_hz=25.,
                    reward_weights={'clearance':-2., 'correction':-.25, 'correction_change':-.05})
    (root/'v14_residual_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'[INFO] Residual output: {root}', flush=True)
    gym.register(id='V14-Residual-Train', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True)
    raw = gym.make('V14-Residual-Train', cfg=cfg)
    depth = make_v13_depth_wrapper()(raw, student_checkpoint=args.student_checkpoint, clip_actions=None)
    env = None
    try:
        base_cfg = V14_LEG_PPO_CFG.to_dict()
        base_runner = OnPolicyRunner(depth, base_cfg, log_dir=None, device=cfg.sim.device)
        base_runner.load(args.actor_checkpoint, load_optimizer=False, map_location=cfg.sim.device)
        env = ResidualPolicyWrapper(depth, base_runner.alg.policy)
        base_hash = module_digest(env.frozen_actor)
        student_hash = module_digest(depth._student)
        origins = raw.unwrapped.scene.terrain.env_origins
        if len(torch.unique(origins,dim=0)) != args.num_envs:
            raise RuntimeError('residual training requires unique terrain origins')
        runner = ResidualOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(root), device=cfg.sim.device)
        obs = env.get_observations()
        assert obs['policy'].shape == (args.num_envs,307)
        if args.resume_residual:
            resume = resume_residual(runner,args.resume_residual,root/'model_resume.pt')
            manifest['resume'] = resume
            print(f'[INFO] Residual resumed: {resume}', flush=True)
        else:
            initialize_residual_actor(runner.alg.policy)
            with torch.no_grad():
                if torch.count_nonzero(runner.alg.policy.act_inference(obs)):
                    raise RuntimeError('initial residual mean is not zero')
            save_initial_residual(runner, root/'model_zero.pt')
            print('[INFO] Zero residual saved; frozen base/student verified', flush=True)
        start_iteration = runner.current_learning_iteration
        manifest['first_iteration'] = start_iteration
        manifest['last_iteration'] = start_iteration+args.iterations-1 if not args.zero_only else None
        (root/'v14_residual_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        seal_residual_checkpoints(root)
        dump_yaml(str(root/'env.yaml'), cfg)
        dump_yaml(str(root/'agent.yaml'), agent_cfg)
        if not args.zero_only:
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
        verification = dict(base_unchanged=module_digest(env.frozen_actor)==base_hash,
                            student_unchanged=module_digest(depth._student)==student_hash,
                            base_tensor_sha256=base_hash, student_tensor_sha256=student_hash,
                            first_iteration=start_iteration, last_iteration=runner.current_learning_iteration,
                            updates_completed=0 if args.zero_only else runner.current_learning_iteration-start_iteration+1,
                            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            max_observed_delta_rad=env.max_observed_delta_rad,
                            residual_parameters_finite=all(bool(torch.isfinite(p).all()) for p in runner.alg.policy.parameters()))
        (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        if not (verification['base_unchanged'] and verification['student_unchanged']
                and verification['residual_parameters_finite'] and env.max_observed_delta_rad <= .050001):
            raise RuntimeError(f'residual verification failed: {verification}')
        seal_residual_checkpoints(root)
        print(f'[INFO] Residual verified: {verification}', flush=True)
    finally:
        (env if env is not None else depth).close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        import omni.kit.app
        omni.kit.app.get_app().post_uncancellable_quit(1)
        app.close()
        raise SystemExit(1)
    app.close()
