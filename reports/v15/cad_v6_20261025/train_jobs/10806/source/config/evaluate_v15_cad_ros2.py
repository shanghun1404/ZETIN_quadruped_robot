"""V15 depth policy on the unchanged CAD USD in ROS2 / Isaac Sim."""
import argparse
import faulthandler
import json
import os
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--usd-path',required=True);p.add_argument('--output',required=True)
p.add_argument('--seconds',type=float,default=14.);p.add_argument('--seed',type=int,default=42)
p.add_argument('--spawn-height',type=float,default=.47);p.add_argument('--video-dir')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
faulthandler.dump_traceback_later(90.,repeat=True)
print('Starting Isaac Sim application',flush=True)
app=AppLauncher(args).app
print('Isaac Sim application started',flush=True)

import numpy as np
import torch
from pxr import UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg,AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from v14_ros2_control import contract
from v12_camera import make_v12_depth_camera_cfg
from v15_cad_contract import joint_mapping
from v15_cad_ros2_bridge import V15CadBridge
from v151_ros2_control.motion import CANONICAL

physical_names,sign_np=joint_mapping(contract.JOINT_NAMES)
bodies=('base',)+tuple(f'{leg}_{part}' for leg in ('FL','FR','BL','BR') for part in ('hip','thigh','shank'))
adjacent={frozenset(('base',f'{leg}_hip')) for leg in ('FL','FR','BL','BR')}
adjacent.update(frozenset((f'{leg}_{a}',f'{leg}_{b}')) for leg in ('FL','FR','BL','BR') for a,b in [('hip','thigh'),('thigh','shank')])

@configclass
class CadSceneCfg(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7,restitution=0.)))
    light=AssetBaseCfg(prim_path='/World/light',spawn=sim_utils.DomeLightCfg(intensity=2500.))
    robot=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',spawn=sim_utils.UsdFileCfg(
        usd_path=args.usd_path,activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,args.spawn_height),
            joint_pos={n:float(s*contract.joint_default(q)) for n,s,q in zip(physical_names,sign_np,contract.JOINT_NAMES)},
            joint_vel={'.*':0.}),soft_joint_pos_limit_factor=1.,
        actuators={
            'haa':ImplicitActuatorCfg(joint_names_expr=['.*_hip_roll'],stiffness=300.,damping=8.,
                effort_limit_sim=24.8,armature=.02,friction=.3),
            'pitch_knee':ImplicitActuatorCfg(joint_names_expr=['.*_hip_pitch','.*_knee'],stiffness=300.,damping=8.,
                effort_limit_sim=48.,armature=.02,friction=.3)})
    depth_camera=make_v12_depth_camera_cfg(.02)
    depth_camera.prim_path='{ENV_REGEX_NS}/Robot/base/front_depth'
    contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*_shank',update_period=0.,history_length=1)

def main():
    print('Creating CAD physics scene',flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,render_interval=4,device=args.device))
    cfg=CadSceneCfg(num_envs=1,env_spacing=12.,replicate_physics=False)
    pair_order=[]
    for i,body in enumerate(bodies):
        others=[n for n in bodies if n!=body and frozenset((body,n)) not in adjacent]
        pair_order.append(dict(body=body,targets=others))
        setattr(cfg,f'pair_{i}',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+body,
            update_period=0.,history_length=1,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+n for n in others]))
    scene=InteractiveScene(cfg)
    print('CAD scene created',flush=True)
    faulthandler.cancel_dump_traceback_later()
    root='/World/envs/env_0/Robot'
    mass_api=UsdPhysics.MassAPI(sim.stage.GetPrimAtPath(root+'/base'))
    offset=list(mass_api.GetCenterOfMassAttr().Get())
    metadata=dict(base_com_offset=offset,physical_joint_names=list(physical_names),sign=sign_np.tolist(),
                  expected_policy_joint_order=list(contract.JOINT_NAMES),physics_dt_s=.005,
                  camera_prim_path=root+'/base/front_depth',pair_order=pair_order)
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    output.with_suffix('.controller_start').write_text('ready\n')
    sim.set_camera_view(eye=(2.,-2.,1.5),target=(0.,0.,.3))
    sim.reset()
    robot=scene['robot']
    # InteractiveScene.reset clears buffers; it does not set the PhysX pose.
    root_state=robot.data.default_root_state.clone()
    root_state[:,:3]+=scene.env_origins
    robot.write_root_pose_to_sim(root_state[:,:7])
    robot.write_root_velocity_to_sim(root_state[:,7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.reset();scene.write_data_to_sim();scene.update(.005)
    if not torch.allclose(robot.data.joint_pos,robot.data.default_joint_pos,atol=1e-6):
        raise RuntimeError('PhysX initial joint state differs from CAD policy stand pose')
    if not torch.allclose(robot.data.root_link_pos_w,root_state[:,:3],atol=1e-5):
        raise RuntimeError('PhysX initial root state differs from requested spawn pose')
    metadata['initial_joint_position_physical_rad']=robot.data.joint_pos[0].tolist()
    metadata['articulation_joint_order']=robot.joint_names
    metadata['initial_root_position_m']=robot.data.root_link_pos_w[0].tolist()
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2))
    for _ in range(3):sim.render()
    bridge=V15CadBridge(scene,sim.device)
    ids=bridge._joint_ids;sign=bridge.sign;canonical=[contract.JOINT_NAMES.index(n) for n in CANONICAL]
    actual_mass=float(robot.root_physx_view.get_masses().sum())
    video=None;failure=None;records=[];depth_samples=[]
    if args.video_dir:
        from v14_video import RolloutVideos
        video=RolloutVideos(sim,robot,args.video_dir,[0],view='side',fps=25.)
    try:
        for step in range(round(args.seconds/.005)):
            bridge.apply_actions()
            data=robot.data
            q=data.joint_pos[0,ids];dq=data.joint_vel[0,ids]
            target=data.joint_pos_target[0,ids];limits=data.joint_effort_limits[0,ids]
            estimated=torch.clamp(300*(target-q)-8*dq,-limits,limits)*sign
            # Depth is the image consumed for this command at the policy stamp.
            if step%4==0:
                depth=scene['depth_camera'].data.output['distance_to_image_plane'][0,:,:,0]
                valid=depth[torch.isfinite(depth)]
                depth_samples.append(dict(stamp_ns=step*5_000_000,finite_fraction=float(torch.isfinite(depth).float().mean()),
                    min_m=float(valid.min()) if valid.numel() else None,max_m=float(valid.max()) if valid.numel() else None))
            if video is not None and step%8==0:video.capture(torch.ones(1,dtype=torch.bool,device=sim.device))
            scene.write_data_to_sim()
            physics_index=sim.current_time_step_index
            sim.step(render=False)
            if (step+1)%4==0:sim.render()
            if sim.current_time_step_index!=physics_index+1:
                raise RuntimeError('Expected exactly one PhysX step per ROS2 command')
            scene.update(.005)
            data=robot.data
            q_after=data.joint_pos[0,ids];dq_after=data.joint_vel[0,ids]
            gravity=quat_apply_inverse(data.root_link_quat_w,torch.tensor([[0.,0.,-1.]],device=sim.device))[0]
            pairs=[];net=[]
            for i in range(len(bodies)):
                sensor=scene[f'pair_{i}'];forces=sensor.data.force_matrix_w[0,0]
                pairs.append(torch.linalg.vector_norm(forces,dim=-1).cpu().numpy().copy())
                net.append(sensor.data.net_forces_w[0,0].cpu().numpy().copy())
            records.append(dict(command_stamp_ns=step*5_000_000,state_stamp_ns=(step+1)*5_000_000,
                position=data.root_link_pos_w[0].cpu().numpy().copy(),
                quaternion=data.root_link_quat_w[0].cpu().numpy().copy(),velocity=data.root_lin_vel_b[0].cpu().numpy().copy(),
                gravity=gravity.cpu().numpy().copy(),joint_position=(q_after*sign)[canonical].cpu().numpy().copy(),
                joint_velocity=(dq_after*sign)[canonical].cpu().numpy().copy(),estimated_torque=estimated[canonical].cpu().numpy().copy(),
                position_target=(target*sign)[canonical].cpu().numpy().copy(),
                nonadjacent_pair_force_n=np.concatenate(pairs),body_net_force=np.stack(net)))
            if not torch.isfinite(data.root_state_w).all() or not torch.isfinite(q_after).all():raise RuntimeError('Nonfinite physics state')
            if float(gravity[2])>-.5:raise RuntimeError('Trunk tilt exceeded60 degrees')
            if float(data.root_link_pos_w[0,2])<.2:raise RuntimeError('Trunk fell below0.20m')
    except Exception as error:
        failure=repr(error);traceback.print_exc()
    finally:
        if video is not None:video.close()
    arrays={k:np.stack([f[k] for f in records]) for k in records[0]} if records else {}
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    output.with_suffix('.status.json').write_text(json.dumps(bridge.history,indent=2))
    output.with_suffix('.depth.json').write_text(json.dumps(depth_samples,indent=2))
    result=dict(schema=1,scenario='flat',seed=args.seed,seconds_requested=args.seconds,seconds_recorded=len(records)*.005,
        failure=failure,completed=failure is None and len(records)==round(args.seconds/.005),reset_after_initialization=False,
        actual_mass_kg=actual_mass,body_names=robot.body_names,joint_names=robot.joint_names,joint_order=CANONICAL,
        joint_mapping=metadata,command=[0.,.4,0.],torque_semantics='instantaneous drive-cap-aware implicit PD estimate, not measured PhysX drive force',
        collision_scope='All nonadjacent pairs among13 real CAD rigid bodies; original67 convexHull colliders, self-collisions enabled. Adjacent12 joint-connected pairs excluded by PhysX.',
        collider_source_unchanged=True,drive_gain_checks=bridge.drive_gain_checks,drive_limit_checks=bridge.drive_limit_checks,
        transport_retry_count=bridge.retry_count,video=video.metadata() if video is not None else [])
    if records:
        tau=arrays['estimated_torque'];rated=np.tile([8.3,18.,18.],4)
        result.update(progress_y_m=float(arrays['position'][-1,1]-arrays['position'][0,1]),
            command_mae_m_s=float(np.mean(abs(arrays['velocity'][min(400,len(records)-1):,1]-.4))),
            torso_min_height_m=float(arrays['position'][:,2].min()),
            max_nonadjacent_contact_force_n=float(arrays['nonadjacent_pair_force_n'].max()),
            self_contact_frames=int((arrays['nonadjacent_pair_force_n'].max(axis=1)>1.).sum()),
            torque_rms_nm=np.sqrt(np.mean(tau*tau,axis=0)).tolist(),torque_peak_nm=np.max(abs(tau),axis=0).tolist(),
            ideal_current_rms_a=(np.sqrt(np.mean(tau*tau,axis=0))/np.tile([1.23,1.44,1.44],4)).tolist(),
            rated_exceed_fraction=np.mean(abs(tau)>rated,axis=0).tolist(),
            compute_ms_p50_p99=np.quantile([s['compute_ms'] for s in bridge.history],[.5,.99]).tolist())
        result['motion_success']=bool(result['completed'] and result['progress_y_m']>=.3*max(0.,args.seconds-2.)
            and result['command_mae_m_s']<=.15)
        result['collision_checks_pass']=result['max_nonadjacent_contact_force_n']<=1.
    else:result.update(motion_success=False,collision_checks_pass=False)
    result['task_success']=bool(result['motion_success'] and result['collision_checks_pass'])
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if result['task_success'] else 1

code=1
try:code=main()
except Exception as error:
    traceback.print_exc()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists():output.write_text(json.dumps(dict(schema=1,completed=False,task_success=False,
        failure=repr(error),seconds_recorded=0),indent=2))
import sys
sys.stdout.flush();sys.stderr.flush();os._exit(code)
