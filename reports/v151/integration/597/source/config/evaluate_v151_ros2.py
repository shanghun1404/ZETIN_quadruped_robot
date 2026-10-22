"""Single robot V15.1 ROS2/Isaac Sim integration and collision evaluation."""
import argparse
import json
import hashlib
import os
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--usd-path',required=True);p.add_argument('--output',required=True)
p.add_argument('--initial-pose');p.add_argument('--seconds',type=float,default=28.)
p.add_argument('--seed',type=int,default=42);p.add_argument('--scenario',choices=['flat','rough','stairs_up'],default='flat')
p.add_argument('--video-dir');p.add_argument('--namespace',default='/v151')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
from v14_leg_env_cfg import GuideV14LegEnvCfg
from v14_ros2_bridge import implicit_joint_drives
from v151_ros2_bridge import V151Ros2ActionCfg
from v14_clearance import interleg_clearances_m
from v151_ros2_control.motion import CANONICAL
from v151_collision import capsule_trunk_gaps_m

def main():
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    cfg=GuideV14LegEnvCfg();cfg.scene.num_envs=1;cfg.seed=args.seed
    cfg.scene.robot.spawn.usd_path=args.usd_path;cfg.scene.env_spacing=12.
    cfg.actions.joint_pos=V151Ros2ActionCfg(asset_name='robot',namespace=args.namespace)
    implicit_joint_drives(cfg.scene.robot)
    cfg.curriculum=None;cfg.events.push_robot=None;cfg.observations.policy.enable_corruption=False
    cfg.episode_length_s=args.seconds+10.
    cfg.events.reset_base.params['pose_range']={'x':(0.,0.),'y':(0.,0.),'yaw':(0.,0.)}
    cfg.events.reset_base.params['velocity_range']={k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    # Initialize exactly once. No subsequent teleport/reset is permitted by the bridge.
    if args.initial_pose:
        initial=json.loads(Path(args.initial_pose).read_text())
        cfg.scene.robot.init_state.pos=(0.,0.,initial['height'])
        cfg.scene.robot.init_state.joint_pos=dict(zip(CANONICAL,initial['joint_position']))
        # Avoid random reset offsets; preserve fixed initial pose.
    cfg.events.reset_robot_joints.params['position_range']=(1.,1.)
    cfg.events.reset_robot_joints.params['velocity_range']=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_y=(.4,.4)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(1e6,1e6)
    cfg.commands.base_velocity.debug_vis=False
    geometry={}
    if args.scenario=='flat':
        cfg.scene.terrain.terrain_type='plane';cfg.scene.terrain.terrain_generator=None
    else:
        from v14_eval_protocol import fixed_eval_generator
        if args.scenario=='stairs_up':
            import copy
            from v9_env_cfg import STAIRS_UP_EVAL_TERRAIN_CFG
            cfg.scene.terrain.terrain_generator=copy.deepcopy(STAIRS_UP_EVAL_TERRAIN_CFG)
        if args.scenario=='rough':
            rough=cfg.scene.terrain.terrain_generator.sub_terrains['random_rough']
            rough.proportion=1.
            cfg.scene.terrain.terrain_generator.sub_terrains={'random_rough':rough}
        gen,geometry=fixed_eval_generator(cfg.scene.terrain.terrain_generator,args.scenario,5 if args.scenario=='stairs_up' else 4,1)
        cfg.scene.terrain.terrain_generator=gen;cfg.scene.terrain.max_init_terrain_level=0
    names=[f'{leg}_{part}_profile' for leg in ('FL','FR','RL','RR') for part in ('thigh','calf')]
    contact_names=names+[f'{leg}_foot' for leg in ('FL','FR','RL','RR')]
    for i,name in enumerate(contact_names):
        others=[f'{{ENV_REGEX_NS}}/Robot/base/{n}' for n in contact_names if n[:2]!=name[:2]]
        setattr(cfg.scene,f'leg_contact_{i}',ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/base/{name}',
            update_period=0.,history_length=1,filter_prim_paths_expr=others))
    env=ManagerBasedRLEnv(cfg=cfg)
    robot=env.scene['robot'];ids,_=robot.find_joints(list(CANONICAL),preserve_order=True)
    bridge=env.action_manager.get_term('joint_pos')
    records={k:[] for k in ('position','quaternion','velocity','world_velocity','angular_velocity','drive_kp','drive_kd','drive_limit','feedforward','position_target','foot_force','foot_com_position','joint_position','joint_velocity','estimated_torque','isaac_estimated_torque','gaps','trunk_gaps','contact_pair_force')}
    failure=None;video=None
    if args.video_dir:
        from v14_video import RolloutVideos
        video=RolloutVideos(env.sim,robot,args.video_dir,[0],view='side',fps=25.)
    env.reset()
    def forbid_reset(env_ids):
        terms=[name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])]
        raise RuntimeError(f'Robot terminated before reset; terms={terms}')
    env._reset_idx=forbid_reset
    foot_ids,_=robot.find_bodies(['FL_foot','FR_foot','RL_foot','RR_foot'],preserve_order=True)
    support=env.scene['contact_forces'];support_ids,_=support.find_bodies(['FL_foot','FR_foot','RL_foot','RR_foot'],preserve_order=True)
    original=env.scene.write_data_to_sim
    def record():
        original()
        data=robot.data
        values=dict(position=data.root_link_pos_w[0],quaternion=data.root_link_quat_w[0],velocity=data.root_lin_vel_b[0],
            joint_position=data.joint_pos[0,ids],joint_velocity=data.joint_vel[0,ids],isaac_estimated_torque=data.applied_torque[0,ids],
            world_velocity=data.root_link_lin_vel_w[0],angular_velocity=data.root_ang_vel_b[0],
            drive_kp=data.joint_stiffness[0,ids],drive_kd=data.joint_damping[0,ids],drive_limit=data.joint_effort_limits[0,ids],
            feedforward=data.joint_effort_target[0,ids],position_target=data.joint_pos_target[0,ids],
            foot_force=support.data.net_forces_w[0,support_ids],foot_com_position=data.body_com_pos_w[0,foot_ids])
        pd=values['drive_kp']*(values['position_target']-values['joint_position'])-values['drive_kd']*values['joint_velocity']
        values['estimated_torque']=torch.clamp(pd,-values['drive_limit'],values['drive_limit'])+values['feedforward']
        frame={k:v.detach().cpu().numpy().copy() for k,v in values.items()}
        frame['gaps']=interleg_clearances_m(data.body_pos_w.cpu().numpy(),data.body_quat_w.cpu().numpy(),robot.body_names)[0]
        frame['trunk_gaps']=capsule_trunk_gaps_m(data.body_pos_w.cpu().numpy(),data.body_quat_w.cpu().numpy(),robot.body_names)[0]
        pair=[]
        for i in range(12):
            forces=env.scene[f'leg_contact_{i}'].data.force_matrix_w
            pair.append(float(torch.linalg.vector_norm(forces,dim=-1).max()))
        frame['contact_pair_force']=pair
        for k,v in frame.items(): records[k].append(v)
    env.scene.write_data_to_sim=record
    try:
        for step in range(round(args.seconds/env.step_dt)):
            if video is not None and step%2==0: video.capture(torch.ones(1,dtype=torch.bool,device=env.device))
            env.step(torch.zeros(1,12,device=env.device))
    except Exception as error:
        failure=repr(error);traceback.print_exc()
    finally:
        if video is not None: video.close()
    arrays={k:np.stack(v) for k,v in records.items() if v}
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    output.with_suffix('.status.json').write_text(json.dumps(bridge.history,indent=2))
    if not arrays:
        result=dict(schema=1,completed=False,task_success=False,failure=failure or 'no physics samples',seconds_recorded=0)
        output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return 1
    tau=arrays.get('estimated_torque',np.zeros((0,12)))
    gaps=arrays.get('gaps',np.zeros((0,24)))
    rated=np.tile([8.3,18.,18.],4)
    result=dict(schema=1,scenario=args.scenario,seed=args.seed,failure=failure,completed=failure is None,
        seconds_recorded=len(tau)*.005,joint_order=CANONICAL,geometry=geometry,
        command_schema=2,drive_gain_checks=bridge.drive_gain_checks,drive_limit_checks=bridge.drive_limit_checks,reset_after_initialization=False,terminal_position=robot.data.root_link_pos_w[0].tolist(),
        terminal_quaternion=robot.data.root_link_quat_w[0].tolist(),
        torque_semantics='hybrid: feedforward command at zero gains; drive-cap-aware instantaneous implicit PD estimate when gains nonzero, not measured PhysX drive force',
        torque_rms=np.sqrt(np.mean(tau*tau,axis=0)).tolist(),torque_peak=np.max(np.abs(tau),axis=0).tolist(),
        rated_exceed_fraction=np.mean(np.abs(tau)>rated,axis=0).tolist(),
        mean_absolute_power=float(np.mean(np.sum(np.abs(tau*arrays['joint_velocity']),axis=1))),
        min_interleg_gap_m=float(gaps.min()),negative_gap_fraction=float(np.mean(np.min(gaps,axis=1)<0)),
        contact_pair_force_max_N=float(arrays['contact_pair_force'].max()),
        min_leg_trunk_proxy_gap_m=float(arrays['trunk_gaps'].min()),
        trunk_proxy_overlap_fraction=float(np.mean(np.min(arrays['trunk_gaps'],axis=1)<0)),
        collision_scope='24 cross-leg thigh/calf capsule gaps plus 12 thigh/calf/foot filtered contact channels; adjacent same-leg and leg-ground excluded; no trunk collider in V14 asset',
        final_position=arrays['position'][-1].tolist(),modes=sorted(set(s['mode'] for s in bridge.history)),
        compute_ms_p50_p99=np.quantile([s['compute_ms'] for s in bridge.history],[.5,.99]).tolist())
    result['demo_sequence_required']=bool(args.initial_pose)
    result['task_success']=result['completed']
    if args.initial_pose:
        required={'SIT','RISE','STABILIZE','STAND','BLEND_WALK','WALK','BRAKE','BLEND_STAND','LOWER','SETTLE_SIT'}
        modes=[s['mode'] for s in bridge.history]
        result['task_success']=bool(result['completed'] and required.issubset(set(modes)) and modes[-1]=='SIT'
            and arrays['position'][-1,1]-arrays['position'][0,1]>.5
            and np.linalg.norm(arrays['velocity'][-1])<.15
            and abs(arrays['position'][-1,2]-initial['height'])<.04)
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if failure is None else 1
try: code=main()
except Exception:
    traceback.print_exc();code=1
import sys
sys.stdout.flush();sys.stderr.flush();os._exit(code)
