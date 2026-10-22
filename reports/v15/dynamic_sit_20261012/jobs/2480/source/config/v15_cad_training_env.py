"""V15 fine-tuning on the original CAD mass, joints and self-collision meshes."""
import json
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from v14_ros2_control import contract
from v15_env_cfg import GuideV15LocomotionEnvCfg
from v15_cad_contract import joint_mapping
from v15_cad_training_actions import V15CADJointPositionActionCfg, canonical_joint_pos_rel, canonical_joint_vel
import v15_cad_training_mdp as cad_mdp
from v15_cad_training_terrain import CAD_STEP_HEIGHTS, cad_lane_terrain, rough_tile_specs, rough_surface_height_torch

FEET = ('FL_shank', 'FR_shank', 'BL_shank', 'BR_shank')
MODEL_METADATA = Path(__file__).with_name('model')/'cad_model.json'
if not MODEL_METADATA.exists():
    MODEL_METADATA = Path(__file__).parent/'experiments/v15_cad_20261011/model/cad_model.json'


def empty_scan(env):
    # Frozen depth student replaces these 187 entries in the existing wrapper.
    return torch.zeros((env.num_envs,187),device=env.device)


def _remap(value):
    if isinstance(value, str):
        return value.replace('RL_foot','BL_shank').replace('RR_foot','BR_shank').replace('_foot','_shank')\
            .replace('_HAA','_hip_roll').replace('_HFE','_hip_pitch').replace('_KFE','_knee')
    if isinstance(value, SceneEntityCfg):
        for field in ('body_names','joint_names'):
            setattr(value,field,_remap(getattr(value,field)))
    elif isinstance(value,dict):
        return {key:_remap(item) for key,item in value.items()}
    elif isinstance(value,(tuple,list)):
        return type(value)(_remap(item) for item in value)
    return value


def _terrain_state(env):
    terrain=env.scene.terrain
    if terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs,device=env.device),torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    heights=torch.tensor(CAD_STEP_HEIGHTS,device=env.device)[terrain.terrain_levels]
    heights=torch.where(_terrain_kind_mask(env,('up','down')),heights,0.)
    return heights,_terrain_kind_mask(env,('down',))


def _terrain_kind_mask(env,kinds):
    terrain=env.scene.terrain
    generator=terrain.cfg.terrain_generator
    if generator is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    rows=list(generator.sub_terrains.values())
    return cad_mdp.terrain_kind_mask(terrain.terrain_types,generator.num_cols,
        [row.kind for row in rows],[row.proportion for row in rows],kinds)


def ground_at(env, positions):
    height,down=_terrain_state(env)
    extra=positions.ndim-2
    height=height.reshape(-1,*([1]*extra));down=down.reshape(-1,*([1]*extra))
    # Downward start platforms are four rises above the shared zero plane.
    origin=env.scene.env_origins.reshape(env.num_envs,*([1]*extra),3)
    relative=positions-origin
    steps=((relative[...,1]-2.)/.3).floor().add(1.).clamp(0.,4.)
    z=torch.where(down,4.-steps,steps)*height
    ground=origin[...,2]-torch.where(down,4.*height,0.)+torch.where(relative[...,0].abs()<=1.,z,0.)
    terrain=env.scene.terrain
    generator=terrain.cfg.terrain_generator
    if generator is not None and 'rough' in generator.sub_terrains:
        rough=_terrain_kind_mask(env,('rough',))
        if not hasattr(env,'_cad_rough_boxes'):
            env._cad_rough_boxes=torch.as_tensor(rough_tile_specs(),device=env.device,dtype=positions.dtype)
        rough_z=origin[...,2]+rough_surface_height_torch(relative[...,:2],env._cad_rough_boxes)
        ground=torch.where(rough.reshape(-1,*([1]*extra)),rough_z,ground)
    return ground


def foot_kinematics(env):
    asset=env.scene['robot']
    if not hasattr(env,'_cad_foot_geometry'):
        ids,_=asset.find_bodies(list(FEET),preserve_order=True)
        metadata=json.loads(MODEL_METADATA.read_text())
        local=torch.tensor([metadata['foot_frames'][leg+'_foot']['center_body_m']
                            for leg in ('FL','FR','RL','RR')],device=env.device)
        env._cad_foot_geometry=(ids,local)
    ids,local=env._cad_foot_geometry
    data=asset.data
    offset=quat_apply(data.body_link_quat_w[:,ids],local.unsqueeze(0).expand(env.num_envs,-1,-1))
    position=data.body_link_pos_w[:,ids]+offset
    velocity=data.body_com_lin_vel_w[:,ids]+torch.cross(data.body_ang_vel_w[:,ids],
        position-data.body_com_pos_w[:,ids],dim=-1)
    return position,velocity


def cad_foot_clearance(env):
    position,velocity=foot_kinematics(env)
    clearance=position[...,2]-ground_at(env,position)-.02215
    return ((clearance-.07).square()*torch.tanh(2*torch.linalg.vector_norm(velocity[...,:2],dim=-1))).sum(-1)


def cad_foot_slip(env):
    _,velocity=foot_kinematics(env)
    sensor=env.scene['contact_forces']
    ids,_=sensor.find_bodies(list(FEET),preserve_order=True)
    contact=torch.linalg.vector_norm(sensor.data.net_forces_w[:,ids],dim=-1)>5.
    return (contact*velocity[...,:2].square().sum(-1)).sum(-1)


def cad_low_height(env):
    root=env.scene['robot'].data.root_link_pos_w
    return root[:,2]-ground_at(env,root)<.20


def cad_lane_cost(env):
    terrain=env.scene.terrain
    if terrain.cfg.terrain_generator is None:return torch.zeros(env.num_envs,device=env.device)
    relative=env.scene['robot'].data.root_link_pos_w-env.scene.env_origins
    lane=_terrain_kind_mask(env,('rough','up','down'))
    return lane*relative[:,0].square()


def cad_tile_exit(env):
    terrain=env.scene.terrain
    if terrain.cfg.terrain_generator is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    relative=env.scene['robot'].data.root_link_pos_w-env.scene.env_origins
    lane=_terrain_kind_mask(env,('rough','up','down'))
    return torch.where(lane,(relative[:,0].abs()>.7)|(relative[:,1]<-.9),
                       (relative[:,0].abs()>2.6)|(relative[:,1].abs()>8.))


def cad_terrain_levels(env,env_ids):
    terrain=env.scene.terrain
    ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if ids.numel() and bool((env.episode_length_buf[ids]>0).any()):
        relative=env.scene['robot'].data.root_link_pos_w[ids]-env.scene.env_origins[ids]
        stairs=_terrain_kind_mask(env,('up','down'))[ids]
        passed=stairs & ~env.termination_manager.terminated[ids] & (relative[:,1]>=3.9) & (relative[:,0].abs()<=.7)
        terrain.update_env_origins(ids,passed,stairs & ~passed)
    return terrain.terrain_levels.float().mean()


@configclass
class CadLaneTerrainCfg(SubTerrainBaseCfg):
    function=cad_lane_terrain
    kind:str='flat'


@configclass
class CadCurriculumCfg:
    terrain_levels=CurriculumTermCfg(func=cad_terrain_levels)


@configclass
class GuideV15CadEnvCfg(GuideV15LocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        names,sign=joint_mapping(contract.JOINT_NAMES)
        self.scene.robot=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
            spawn=sim_utils.UsdFileCfg(usd_path='',activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.47),
                joint_pos={name:float(s*contract.joint_default(policy)) for name,s,policy in zip(names,sign,contract.JOINT_NAMES)},
                joint_vel={'.*':0.}),soft_joint_pos_limit_factor=1.,
            actuators={
                'haa':ImplicitActuatorCfg(joint_names_expr=['.*_hip_roll'],stiffness=300.,damping=8.,
                    effort_limit_sim=24.8,armature=.02,friction=.3),
                'pitch_knee':ImplicitActuatorCfg(joint_names_expr=['.*_hip_pitch','.*_knee'],stiffness=300.,damping=8.,
                    effort_limit_sim=48.,armature=.02,friction=.3)})
        self.scene.replicate_physics=False
        self.scene.env_spacing=8.
        self.scene.height_scanner=None
        self.scene.depth_camera.prim_path='{ENV_REGEX_NS}/Robot/base/front_depth'
        self.scene.contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*_shank',
            update_period=.005,history_length=3,track_air_time=True)
        for i,row in enumerate(cad_mdp.cad_self_contact_filters()):
            setattr(self.scene,f'pair_{i}',ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+row['body'],
                update_period=.005,history_length=1,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+name for name in row['targets']]))
        self.actions.joint_pos=V15CADJointPositionActionCfg()
        self.commands.base_velocity=cad_mdp.CadVelocityCommandCfg()
        self.observations.policy.joint_pos.func=canonical_joint_pos_rel
        self.observations.policy.joint_pos.params={}
        self.observations.policy.joint_vel.func=canonical_joint_vel
        self.observations.policy.joint_vel.params={}
        self.observations.policy.height_scan.func=empty_scan
        self.observations.policy.height_scan.params={}
        self.observations.policy.enable_corruption=False
        self.events.v14_ground_filter=None
        self.events.physics_material=None
        self.events.add_base_mass=None
        self.events.push_robot=None
        self.events.reset_base.params['pose_range']={'x':(-.1,.1),'y':(-.1,.1),'yaw':(-3.14,3.14)}
        self.events.reset_base.params['velocity_range']={name:(-.05,.05) for name in ('x','y','z','roll','pitch','yaw')}
        self.events.reset_robot_joints.params['position_range']=(.97,1.03)
        material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7,restitution=0.)
        self.scene.terrain.physics_material=material
        self.sim.physics_material=material
        for reward in vars(self.rewards).values():
            if isinstance(reward,RewardTermCfg):reward.params=_remap(reward.params)
        self.rewards.base_linear_velocity=RewardTermCfg(func=cad_mdp.track_planar_velocity_exp,weight=8.)
        self.rewards.base_angular_velocity=RewardTermCfg(func=cad_mdp.track_yaw_velocity_exp,weight=5.)
        self.rewards.gait.weight=2.
        self.rewards.air_time.weight=1.
        self.rewards.foot_clearance=RewardTermCfg(func=cad_foot_clearance,weight=-.25)
        self.rewards.foot_slip=RewardTermCfg(func=cad_foot_slip,weight=-.5)
        self.rewards.zero_command_motion=RewardTermCfg(func=cad_mdp.zero_command_velocity_l2,weight=-3.)
        self.rewards.rated_motor_load=RewardTermCfg(func=cad_mdp.rated_torque_l2,weight=-.5)
        self.rewards.excess_rated_motor_load=RewardTermCfg(func=cad_mdp.excess_rated_torque_l2,weight=-1.)
        self.rewards.self_contact_count=RewardTermCfg(func=cad_mdp.self_contact_count,weight=-2.)
        self.rewards.self_contact_force=RewardTermCfg(func=cad_mdp.self_contact_force_l2,weight=-.01)
        self.rewards.stair_lane=RewardTermCfg(func=cad_lane_cost,weight=-2.)
        self.terminations.low_height=TerminationTermCfg(func=cad_low_height)
        self.terminations.tile_exit=TerminationTermCfg(func=cad_tile_exit)
        self.scene.terrain.terrain_type='plane'
        self.scene.terrain.terrain_generator=None
        self.curriculum=None


def enable_stair_curriculum(cfg):
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(size=(6.,18.),num_rows=len(CAD_STEP_HEIGHTS),
        num_cols=16,curriculum=True,border_width=5.,sub_terrains={
            'flat':CadLaneTerrainCfg(kind='flat',proportion=.5),
            'stairs_up':CadLaneTerrainCfg(kind='up',proportion=.25),
            'stairs_down':CadLaneTerrainCfg(kind='down',proportion=.25)})
    cfg.scene.terrain.max_init_terrain_level=0
    cfg.commands.base_velocity.force_forward_terrain_fraction_start=.5
    cfg.events.reset_base.params['pose_range']['yaw']=(0.,0.)
    cfg.curriculum=CadCurriculumCfg()


def enable_targeted_finetune(cfg):
    """Follow-up profile after measured command collapse and rough regression."""
    generator=cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.sub_terrains={
            'flat':CadLaneTerrainCfg(kind='flat',proportion=.4),
            'rough':CadLaneTerrainCfg(kind='rough',proportion=.1),
            'stairs_up':CadLaneTerrainCfg(kind='up',proportion=.25),
            'stairs_down':CadLaneTerrainCfg(kind='down',proportion=.25)}
        cfg.commands.base_velocity.force_forward_terrain_fraction_start=.4
    cfg.commands.base_velocity.command_probabilities=(.10,.10,.30,.25,.20,.05)
    cfg.commands.base_velocity.resampling_time_range=(8.,14.)
    cfg.rewards.directed_motion_progress=RewardTermCfg(func=cad_mdp.directed_motion_progress_reward,weight=4.)
    cfg.rewards.base_linear_velocity.params={'std':.20}
    cfg.rewards.base_angular_velocity.params={'std':.10}
    cfg.rewards.stair_lane.weight=-4.
    cfg.rewards.self_contact_count.weight=-5.
    cfg.rewards.self_contact_force.weight=-.025


def enable_rms_finetune(cfg):
    """Optimize non-stair motor load while preserving commanded motion."""
    enable_targeted_finetune(cfg)
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(size=(6.,18.),num_rows=1,
        num_cols=cfg.scene.num_envs,curriculum=True,border_width=5.,sub_terrains={
            'flat':CadLaneTerrainCfg(kind='flat',proportion=.9),
            'rough':CadLaneTerrainCfg(kind='rough',proportion=.1)})
    cfg.scene.terrain.max_init_terrain_level=0
    cfg.curriculum=None
    cfg.commands.base_velocity.force_forward_terrain_fraction_start=.9
    cfg.commands.base_velocity.command_probabilities=(.10,.20,.25,.10,.30,.05)
    cfg.commands.base_velocity.resampling_time_range=(14.,18.)
    cfg.events.reset_base.params['pose_range']['yaw']=(0.,0.)
    cfg.rewards.directed_motion_progress.weight=6.
    cfg.rewards.base_angular_velocity.weight=8.
    cfg.rewards.rated_motor_load.weight=-.75
    cfg.rewards.rms_motor_load=RewardTermCfg(func=cad_mdp.excess_rms_motor_load,weight=-12.,
        params={'target_fraction':.90})
