"""Independent CAD trials, one Isaac application, external ROS2 controllers.

Terrain lanes are created before PhysX starts. Resets occur only between
independent trials. Every recorded physics step has a same-stamp ROS2 reply.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--usd-path',required=True)
parser.add_argument('--output-dir',required=True)
parser.add_argument('--actor',required=True)
parser.add_argument('--student',required=True)
parser.add_argument('--model-path')
parser.add_argument('--cases',required=True)
parser.add_argument('--seed',type=int,default=42)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
faulthandler.dump_traceback_later(90.,repeat=True)
print('Starting Isaac Sim suite application',flush=True)
app=AppLauncher(args).app

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
from v15_cad_suite_protocol import case_config,get_case,ground_height,terrain_cuboids,evaluate_case
from v15_cad_suite_metrics import estimated_torque,torque_summary,standing_verdict
from v151_ros2_control.motion import CANONICAL

physical_names,sign_np=joint_mapping(contract.JOINT_NAMES)
canonical=[contract.JOINT_NAMES.index(n) for n in CANONICAL]
to_policy=[CANONICAL.index(n) for n in contract.JOINT_NAMES]
bodies=('base',)+tuple(f'{leg}_{part}' for leg in ('FL','FR','BL','BR') for part in ('hip','thigh','shank'))
adjacent={frozenset(('base',f'{leg}_hip')) for leg in ('FL','FR','BL','BR')}
adjacent.update(frozenset((f'{leg}_{a}',f'{leg}_{b}')) for leg in ('FL','FR','BL','BR') for a,b in [('hip','thigh'),('thigh','shank')])
stand_names={'standing_hold','sit_rise_walk_stop_sit','deep_crouch_rise_walk_stop_sit'}


@configclass
class CadSceneCfg(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg(size=(1000.,1000.),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7,restitution=0.)))
    light=AssetBaseCfg(prim_path='/World/light',spawn=sim_utils.DomeLightCfg(intensity=2500.))
    robot=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',spawn=sim_utils.UsdFileCfg(
        usd_path=args.usd_path,activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.47),
            joint_pos={n:float(s*contract.joint_default(q)) for n,s,q in zip(physical_names,sign_np,contract.JOINT_NAMES)},
            joint_vel={'.*':0.}),soft_joint_pos_limit_factor=1.,
        actuators={
            'haa':ImplicitActuatorCfg(joint_names_expr=['.*_hip_roll'],stiffness=300.,damping=8.,effort_limit_sim=24.8,armature=.02,friction=.3),
            'pitch_knee':ImplicitActuatorCfg(joint_names_expr=['.*_hip_pitch','.*_knee'],stiffness=300.,damping=8.,effort_limit_sim=48.,armature=.02,friction=.3)})
    depth_camera=make_v12_depth_camera_cfg(.02)
    depth_camera.prim_path='{ENV_REGEX_NS}/Robot/base/front_depth'
    contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*_shank',update_period=0.,history_length=1)


def strict(value):
    if isinstance(value,dict):return {k:strict(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [strict(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(strict(value),indent=2,allow_nan=False))


def start_controller(name,namespace,offset,folder,drop):
    command=[sys.executable,'-m','v15_cad_suite_controller','--ros-args','-r','__ns:='+namespace,
        '-p','actor_checkpoint:='+args.actor,'-p','student_checkpoint:='+args.student]
    if name in stand_names:
        command+=['-p','model_path:='+args.model_path,'-p','initial_mode:='+('STAND' if name=='standing_hold' else 'SIT'),
            '-p','demo:='+('false' if name=='standing_hold' else 'true'),
            '-p','height_drop:='+str(drop),'-p','stand_forward_shift_m:=0.0','-p','crouch_forward_shift_m:=0.0']
    else:
        command+=['-p','base_com_offset:='+json.dumps(offset)]
    env=dict(os.environ,V15_CAD_CASE=name)
    log=open(folder/'controller.log','w')
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    write_json(folder/'controller_command.json',command)
    return process,log


def trial(sim,scene,name,lane,offset,pair_order):
    folder=Path(args.output_dir)/name
    folder.mkdir(parents=True,exist_ok=False)
    standing=name in stand_names
    spec=None if standing else get_case(name)
    duration=12. if name=='standing_hold' else 30. if standing else spec.duration_s
    config=dict(name=name,duration_s=duration,lane_origin_world_m=[lane,0,0],seed=args.seed,
        reset_policy='Explicit initialization once; no resets during this trial.') if standing else case_config(spec)
    config['lane_origin_world_m']=[lane,0,0]
    write_json(folder/'case_config.json',config)
    drop=.18 if name=='deep_crouch_rise_walk_stop_sit' else .10
    nominal_height=.47
    initial_q=np.tile([0.,-.55,1.1],4)
    if standing:
        model=json.loads(Path(args.model_path).with_name('cad_model.json').read_text())
        crouch=json.loads(Path(args.model_path).with_name('crouch.json').read_text())
        nominal_height=model['nominal_root_height_m']+.006
        if name!='standing_hold':
            key='deep_crouch' if drop>.1 else 'standard_crouch'
            initial_q=np.asarray(crouch[key]['q_canonical_rad'])
    ground=0. if standing else ground_height(spec,0.,0.)
    spawn_height=ground+nominal_height-(drop if standing and name!='standing_hold' else 0.)
    robot=scene['robot']
    ids,found=robot.find_joints(list(physical_names),preserve_order=True)
    if tuple(found)!=physical_names:raise RuntimeError('Physical joint mapping changed')
    root=robot.data.default_root_state.clone()
    root[:,:3]=torch.tensor([[lane,0.,spawn_height]],device=sim.device)
    joint=robot.data.default_joint_pos.clone()
    joint[0,ids]=torch.tensor(initial_q[to_policy]*sign_np,device=sim.device,dtype=torch.float32)
    robot.write_root_pose_to_sim(root[:,:7]);robot.write_root_velocity_to_sim(root[:,7:])
    robot.write_joint_state_to_sim(joint,torch.zeros_like(joint))
    robot.set_joint_position_target(joint);robot.set_joint_effort_target(torch.zeros_like(joint))
    robot.write_joint_stiffness_to_sim(300.);robot.write_joint_damping_to_sim(8.)
    caps=torch.tensor([[24.8 if 'roll' in n else 48. for n in robot.joint_names]],device=sim.device)
    robot.write_joint_effort_limit_to_sim(caps)
    for actuator in robot.actuators.values():
        actuator.stiffness[:]=300.;actuator.damping[:]=8.
    scene.reset();scene.write_data_to_sim();scene.update(.005)
    if not torch.allclose(robot.data.joint_pos,joint,atol=1e-6):raise RuntimeError('Initial joints disagree with requested pose')
    if not torch.allclose(robot.data.root_link_pos_w,root[:,:3],atol=1e-5):raise RuntimeError('Initial root disagrees with requested pose')
    metadata=dict(base_com_offset=offset,initial_joint_canonical_rad=initial_q.tolist(),
        initial_root_world_m=root[0,:3].tolist(),physical_joint_names=physical_names,physical_sign=sign_np.tolist(),
        canonical_joint_order=CANONICAL,pair_order=pair_order,actual_mass_kg=float(robot.root_physx_view.get_masses().sum()))
    write_json(folder/'metadata.json',metadata)
    sim.set_camera_view(eye=(lane+2.,-2.,1.5+ground),target=(lane,0.,.3+ground))
    for _ in range(3):sim.render()
    process,log=start_controller(name,'/v15cad/'+name,offset,folder,drop)
    bridge=None;video=None;records=[];depth_samples=[];failure=None
    print('CASE START '+name,flush=True)
    try:
        bridge=V15CadBridge(scene,sim.device,namespace='/v15cad/'+name)
        from v14_video import RolloutVideos
        video=RolloutVideos(sim,robot,str(folder/'video'),[0],view='side',fps=25.)
        for step in range(round(duration/.005)):
            if process.poll() is not None:raise RuntimeError('ROS2 controller exited: '+str(process.returncode))
            bridge.apply_actions()
            data=robot.data
            def canonical_value(value):return (value[0,ids]*bridge.sign)[canonical].cpu().numpy().copy()
            q=canonical_value(data.joint_pos);dq=canonical_value(data.joint_vel)
            target=canonical_value(data.joint_pos_target);ff=canonical_value(data.joint_effort_target)
            drive_caps=data.joint_effort_limits[0,ids][canonical].cpu().numpy().copy()
            status=bridge.history[-1]
            kp,kd=status['stiffness_nm_rad'],status['damping_nms_rad']
            tau=estimated_torque(q,dq,target,kp,kd,drive_caps,ff)
            if step%4==0:
                depth=scene['depth_camera'].data.output['distance_to_image_plane'][0,:,:,0]
                valid=depth[torch.isfinite(depth)]
                depth_samples.append(dict(stamp_ns=step*5_000_000,finite_fraction=float(torch.isfinite(depth).float().mean()),
                    min_m=float(valid.min()) if valid.numel() else None,max_m=float(valid.max()) if valid.numel() else None))
            if step%8==0:video.capture(torch.ones(1,dtype=torch.bool,device=sim.device))
            scene.write_data_to_sim();physics_index=sim.current_time_step_index
            sim.step(render=False)
            if (step+1)%4==0:sim.render()
            if sim.current_time_step_index!=physics_index+1:raise RuntimeError('Expected one PhysX step per command')
            scene.update(.005);data=robot.data
            gravity=quat_apply_inverse(data.root_link_quat_w,torch.tensor([[0.,0.,-1.]],device=sim.device))[0]
            pairs=[];net=[]
            for i in range(len(bodies)):
                sensor=scene[f'pair_{i}']
                pairs.append(torch.linalg.vector_norm(sensor.data.force_matrix_w[0,0],dim=-1).cpu().numpy().copy())
                net.append(sensor.data.net_forces_w[0,0].cpu().numpy().copy())
            position=data.root_link_pos_w[0].cpu().numpy().copy();position[0]-=lane
            local_ground=0. if standing else ground_height(spec,position[0],position[1])
            support=scene['contact_forces'].data.net_forces_w[0].cpu().numpy().copy()
            records.append(dict(command_stamp_ns=step*5_000_000,state_stamp_ns=(step+1)*5_000_000,
                position=position,quaternion=data.root_link_quat_w[0].cpu().numpy().copy(),
                velocity=data.root_lin_vel_b[0].cpu().numpy().copy(),angular_velocity=data.root_ang_vel_b[0].cpu().numpy().copy(),
                gravity=gravity.cpu().numpy().copy(),joint_position=canonical_value(data.joint_pos),
                joint_velocity=canonical_value(data.joint_vel),pre_joint_velocity=dq,estimated_torque=tau,
                position_target=target,feedforward_torque=ff,drive_limits=drive_caps,
                command=np.asarray(status['command']),kp=kp,kd=kd,
                ground_height_m=local_ground,nonadjacent_pair_force_n=np.concatenate(pairs),body_net_force=np.stack(net),
                shank_support_force=support,bounded_torque=np.asarray(status['bounded_torque'])))
            if not torch.isfinite(data.root_state_w).all() or not np.isfinite(records[-1]['joint_position']).all():
                raise RuntimeError('Nonfinite physics state')
            if float(gravity[2])>-.5:raise RuntimeError('Trunk tilt exceeded 60 degrees')
            if position[2]-local_ground<.2:raise RuntimeError('Trunk below 0.20m above local ground')
    except Exception as error:
        failure=repr(error);traceback.print_exc()
    finally:
        if video is not None:video.close()
        process.terminate()
        try:process.wait(timeout=10.)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()
        if bridge is not None:bridge._node.destroy_node()
    arrays={k:np.stack([r[k] for r in records]) for k in records[0]} if records else {}
    np.savez_compressed(folder/'telemetry.npz',**arrays)
    history=bridge.history if bridge is not None else []
    write_json(folder/'status.json',history);write_json(folder/'depth.json',depth_samples)
    completed=failure is None and len(records)==round(duration/.005)
    result=dict(schema=2,case=name,seed=args.seed,seconds_requested=duration,seconds_recorded=len(records)*.005,
        completed=completed,failure=failure,reset_during_trial=False,metadata=metadata,
        drive_gain_checks=bridge.drive_gain_checks if bridge else [],drive_limit_checks=bridge.drive_limit_checks if bridge else [],
        transport_retry_count=bridge.retry_count if bridge else None,
        torque_semantics='Pre-step implicit PD clipping plus separately applied feedforward estimate; not a measured PhysX drive moment.',
        collision_scope='All 66 nonadjacent real rigid-body pairs, 132 directed filters; original 67 CAD convex hulls. Joint-connected pairs are excluded by PhysX.',
        cad_source_unchanged=True,video=video.metadata() if video else [])
    if records:
        if len(records)<2:
            verdict=dict(motion_success=False,reasons=['insufficient_post_physics_samples'],metrics={})
        elif standing:
            verdict=standing_verdict(name,arrays,history,completed,nominal_height,nominal_height-drop)
        else:
            verdict=evaluate_case(spec,time_s=arrays['state_stamp_ns']*1e-9,positions=arrays['position'],
                quaternions=arrays['quaternion'],body_velocity=arrays['velocity'],gravity=arrays['gravity'],alive_at_end=completed)
        result['motion_verdict']=verdict;result['motion_success']=verdict['motion_success']
        result.update(torque_summary(arrays['estimated_torque'],arrays['pre_joint_velocity']))
        result['max_nonadjacent_contact_force_n']=float(arrays['nonadjacent_pair_force_n'].max())
        result['self_contact_frames']=int((arrays['nonadjacent_pair_force_n'].max(axis=1)>1.).sum())
        result['collision_checks_pass']=result['max_nonadjacent_contact_force_n']<=1.
        result['pd_vs_status_max_error_nm']=float(abs(arrays['estimated_torque']-arrays['bounded_torque']).max())
        result['torque_transport_integrity_pass']=result['pd_vs_status_max_error_nm']<=1e-3
        result['torso_min_ground_clearance_m']=float(np.min(arrays['position'][:,2]-arrays['ground_height_m']))
        result['max_abs_tilt_deg']=float(np.degrees(np.arccos(np.clip(-arrays['gravity'][:,2],-1.,1.))).max())
        result['compute_ms_p50_p99']=np.quantile([s['compute_ms'] for s in history],[.5,.99]).tolist()
        pair_labels=[p['body']+' <-> '+target for p in pair_order for target in p['targets']]
        peaks=arrays['nonadjacent_pair_force_n'].max(axis=0)
        result['contact_pairs_over_1n']={label:float(force) for label,force in zip(pair_labels,peaks) if force>1.}
    else:
        result.update(motion_success=False,collision_checks_pass=False,torque_transport_integrity_pass=False)
    result['task_success']=bool(completed and result['motion_success'] and result['collision_checks_pass']
        and result['torque_transport_integrity_pass'] and not result.get('simulator_peak_exceeded',True))
    write_json(folder/'result.json',result)
    print('CASE END '+name+' '+json.dumps(strict({k:result.get(k) for k in ['task_success','seconds_recorded','failure','motion_verdict','max_nonadjacent_contact_force_n']})),flush=True)
    return result


def main():
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    names=args.cases.split(',')
    if len(set(names))!=len(names):raise ValueError('Duplicate cases')
    if any(n in stand_names for n in names) and not args.model_path:raise ValueError('CAD standing model required')
    for n in names:
        if n not in stand_names:get_case(n)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,render_interval=4,device=args.device))
    cfg=CadSceneCfg(num_envs=1,env_spacing=12.,replicate_physics=False)
    pair_order=[]
    for i,body in enumerate(bodies):
        others=[n for n in bodies if n!=body and frozenset((body,n)) not in adjacent]
        pair_order.append(dict(body=body,targets=others))
        setattr(cfg,f'pair_{i}',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+body,update_period=0.,history_length=1,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+n for n in others]))
    scene=InteractiveScene(cfg)
    for i,name in enumerate(names):
        if name in stand_names:continue
        for cube in terrain_cuboids(name):
            cuboid=sim_utils.CuboidCfg(size=cube.size,collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7,restitution=0.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.45,.50)))
            cuboid.func('/World/terrain_'+name+'/'+cube.name,cuboid,translation=(cube.center[0]+i*30.,cube.center[1],cube.center[2]))
    offset=list(UsdPhysics.MassAPI(sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/base')).GetCenterOfMassAttr().Get())
    sim.reset();faulthandler.cancel_dump_traceback_later()
    results=[]
    for i,name in enumerate(names):
        try:result=trial(sim,scene,name,i*30.,offset,pair_order)
        except Exception as error:
            traceback.print_exc();result=dict(case=name,task_success=False,completed=False,failure=repr(error),seconds_recorded=0.)
            folder=output/name;folder.mkdir(parents=True,exist_ok=True);write_json(folder/'result.json',result)
        results.append(result)
        write_json(output/'suite.json',dict(schema=2,cases=results,all_pass=all(r['task_success'] for r in results)))
    return 0 if all(r['task_success'] for r in results) else 2


exit_code=1
try:exit_code=main()
except Exception as error:
    traceback.print_exc()
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    write_json(Path(args.output_dir)/'startup_failure.json',dict(failure=repr(error)))
sys.stdout.flush();sys.stderr.flush();os._exit(exit_code)
