"""Bounded V15.1 fixed-command flat/rough torque pilot, starting from339."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import math
import os
from pathlib import Path

import train_v151_locomotion as base

BASE_OVERLAY_SHA256 = 'd7e5c58debf0e4df03b9f90bd61c3b6a0d8b48d99227b5968e44bfb6252b9581'
RATED_WEIGHT = -.1


def checkpoint_chunks():
    previous=0
    for total in (25,50,100):
        yield total,total-previous
        previous=total


def validate_pilot_args(args):
    if (not args.resume or args.num_envs!=512 or args.max_iterations!=100
            or args.max_delay_steps!=3 or args.seed!=47):
        raise ValueError('Targeted pilot requires resume339,512env,100updates,delay0-3,seed47')


def restrict_terrain(generator):
    if generator.num_rows<5 or 'random_rough' not in generator.sub_terrains:
        raise ValueError('Expected inherited rough terrain with evaluation level4')
    difficulty=4/(generator.num_rows-1)
    rough=copy.deepcopy(generator.sub_terrains['random_rough'])
    flat=copy.deepcopy(rough)
    flat.noise_range=(0.,0.)
    flat.proportion=rough.proportion=.5
    generator.sub_terrains={'flat':flat,'random_rough':rough}
    generator.num_rows=1
    generator.curriculum=False
    generator.difficulty_range=(difficulty,difficulty)
    return dict(terrain_proportions={'flat':.5,'random_rough':.5},difficulty=difficulty,
                rough_noise_range=list(rough.noise_range),
                flat_geometry='zero-height copy of rough heightfield',stairs_validated=False)


def validate_actual_command(class_name,values):
    if class_name!='UniformVelocityCommand' or not values:
        raise ValueError('Runtime command must be nonempty UniformVelocityCommand, without lane yaw feedback')
    if any(len(row)!=3 or any(not math.isclose(float(v),expected,abs_tol=1e-6)
                             for v,expected in zip(row,(0.,.4,0.))) for row in values):
        raise ValueError('Runtime command drifted from fixed body[0,.4,0]')
    return dict(command_class=class_name,num_environments=len(values),
                command_min=[min(row[i] for row in values) for i in range(3)],
                command_max=[max(row[i] for row in values) for i in range(3)])


def actual_scope_contract(env,scope):
    term=env.command_manager.get_term('base_velocity')
    actual=validate_actual_command(type(term).__name__,term.command.tolist())
    ranges=term.cfg.ranges
    assert tuple(ranges.lin_vel_x)==(0.,0.) and tuple(ranges.lin_vel_y)==(.4,.4)
    assert tuple(ranges.ang_vel_z)==(0.,0.) and not term.cfg.heading_command
    assert term.cfg.rel_standing_envs==0.
    terrain=env.scene.terrain
    generator=terrain.cfg.terrain_generator
    assert set(generator.sub_terrains)=={'flat','random_rough'}
    assert generator.num_rows==1 and not generator.curriculum and env.cfg.curriculum is None
    assert tuple(generator.difficulty_range)==(scope['difficulty'],scope['difficulty'])
    assert tuple(generator.sub_terrains['flat'].noise_range)==(0.,0.)
    assert list(generator.sub_terrains['random_rough'].noise_range)==scope['rough_noise_range']
    assert all(t.proportion==.5 for t in generator.sub_terrains.values())
    assert tuple(terrain.terrain_origins.shape)==(1,generator.num_cols,3)
    assert bool((terrain.terrain_levels==0).all())
    types=terrain.terrain_types.tolist()
    actual.update(command_ranges=dict(x=list(ranges.lin_vel_x),y=list(ranges.lin_vel_y),yaw=list(ranges.ang_vel_z)),
        command_module=type(term).__module__,terrain_generator_class=type(generator).__name__,
        terrain_classes={name:type(cfg).__name__ for name,cfg in generator.sub_terrains.items()},
        terrain_proportions={name:float(cfg.proportion) for name,cfg in generator.sub_terrains.items()},
        terrain_noise_ranges={name:list(cfg.noise_range) for name,cfg in generator.sub_terrains.items()},
        difficulty_range=list(generator.difficulty_range),curriculum_enabled=False,
        terrain_origins_shape=list(terrain.terrain_origins.shape),
        environment_terrain_column_counts={str(i):types.count(i) for i in sorted(set(types))},
        terrain_levels_min_max=[int(terrain.terrain_levels.min()),int(terrain.terrain_levels.max())])
    return actual


def configure_scope(env_cfg):
    from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
    scope=restrict_terrain(env_cfg.scene.terrain.terrain_generator)
    old=env_cfg.commands.base_velocity
    ranges=copy.deepcopy(old.ranges)
    ranges.lin_vel_x=(0.,0.);ranges.lin_vel_y=(.4,.4);ranges.ang_vel_z=(0.,0.)
    env_cfg.commands.base_velocity=UniformVelocityCommandCfg(asset_name=old.asset_name,
        resampling_time_range=(1e6,1e6),rel_standing_envs=0.,rel_heading_envs=0.,
        heading_command=False,debug_vis=False,ranges=ranges)
    env_cfg.curriculum=None
    scope.update(command_body_xyz=[0.,.4,0.],lane_yaw_feedback=False,
                 actuator_randomization='inherited V15 startup gains0.8-1.2',
                 other_events='inherited V15',teacher_loss_added=False,
                 checkpoint_updates=[total for total,_ in checkpoint_chunks()])
    return scope


def configure_reward(rewards):
    contract=base.configure_torque_reward(rewards)
    rewards.rated_motor_load.weight=RATED_WEIGHT
    contract['weights_after']['rated_motor_load']=RATED_WEIGHT
    contract['rated_motor_load_weight']=RATED_WEIGHT
    return contract


def build_source(original):
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()!=BASE_OVERLAY_SHA256:
        raise RuntimeError('Unreviewed V15.1 base training overlay')
    source=base.build_source(original)
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1: raise RuntimeError(f'Expected one targeted anchor: {old!r}')
        source=source.replace(old,new,1)
    replace('default=500)','default=100)')
    replace('    env_cfg.seed = args_cli.seed\n','    env_cfg.seed = args_cli.seed\n    targeted_scope = targeted_configure_scope(env_cfg)\n')
    replace('"guide_v151_rated_load"','"guide_v151_targeted_load"')
    replace('"339_rated_squared_w0p02_lr1e-5"','"339_fixed04_flatrough_w0p1_lr1e-5"')
    replace('        "prepared_motor_adaptation": True,','        "prepared_motor_adaptation": True,\n        "targeted_scope": targeted_scope,\n        "base_overlay_sha256": TARGETED_BASE_SHA256,')
    replace('assert reward_manager.get_term_cfg("rated_motor_load").weight == -0.02',
            'assert reward_manager.get_term_cfg("rated_motor_load").weight == -0.1')
    replace('        action = raw_env.unwrapped.action_manager.get_term("joint_pos")',
            '        actual_targeted_scope = targeted_actual_scope(raw_env.unwrapped, targeted_scope)\n        action = raw_env.unwrapped.action_manager.get_term("joint_pos")')
    replace('            **lr_contract, "resume_optimizer": True,',
            '            **lr_contract, "resume_optimizer": True,\n            "targeted_scope": targeted_scope,\n            "actual_targeted_scope": actual_targeted_scope,')
    replace('        for updates in (250, 500):\n            runner.learn(num_learning_iterations=250, init_at_random_ep_len=(updates == 250))',
            '        for updates, chunk_size in targeted_checkpoint_chunks():\n            runner.learn(num_learning_iterations=chunk_size, init_at_random_ep_len=(updates == 25))')
    replace('            if updates == 250:\n','            if updates < 100:\n')
    ast.parse(source)
    return source


def main():
    parser=argparse.ArgumentParser(add_help=False)
    for name in base.INPUTS: parser.add_argument('--'+name.replace('_','-'),required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--num_envs',type=int,default=512)
    parser.add_argument('--max_iterations',type=int,default=100)
    parser.add_argument('--max-delay-steps',type=int,default=3)
    parser.add_argument('--seed',type=int,default=47)
    args,_=parser.parse_known_args()
    validate_pilot_args(args)
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU training requires Slurm allocation')
    for name,(expected_path,expected_hash) in base.INPUTS.items():
        path=Path(getattr(args,name))
        if path.resolve()!=Path(expected_path).resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
            raise RuntimeError(f'Unreviewed frozen input {name}')
    original=Path(os.environ.get('V15_TRAINER','/home/shanghun1404/v15_large_scale/config/train_v15_locomotion.py'))
    source=build_source(original)
    namespace=dict(__name__='__main__',__file__=str(Path(__file__).resolve()),
        FINETUNE_FIXED_LR=base.FIXED_LR,FINETUNE_TRAINER_SHA256=base.TRAINER_SHA256,
        FINETUNE_SOURCE_HASHES=base.SOURCE_HASHES,finetune_sync_lr=base.synchronize_fixed_lr,
        finetune_configure_torque_reward=configure_reward,targeted_configure_scope=configure_scope,
        targeted_checkpoint_chunks=checkpoint_chunks,targeted_actual_scope=actual_scope_contract,
        TARGETED_BASE_SHA256=BASE_OVERLAY_SHA256)
    exec(compile(source,str(original),'exec'),namespace)


if __name__=='__main__': main()
