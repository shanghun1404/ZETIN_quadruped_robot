"""V15-R: a separate blind fall-recovery policy on the V15 plant.

Selected by the ROS2 policy node when the base tilts past a threshold; the
locomotion policy stays unchanged. Same actuation path as V15-L: PhysX joint-drive
PD (gains U(0.8, 1.2)), V13 25 Hz LPF + 0.99 torque guard, 0-3 step target delay.

* Terrain: flat plane. The V15 recovery USD adds trunk colliders and does not
  filter the V14 leg capsules from the ground, so a fallen robot rests on its body.
* Reset: 80% uniform random orientation dropped from ``DROP_ROOT_HEIGHT_M`` with
  joints uniform within ``JOINT_RANGE_RAD`` of the stand pose; 20% standing upright.
* The robot_v2 MJCF declares no joint ranges, so simulated joints are unlimited.
  ``JOINT_RANGE_RAD`` is an ASSUMED working range (not a measured hardware limit)
  used for reset sampling and a soft penalty; replace it with the real limits.
* Actor (42): base angular velocity, projected gravity, joint position offset,
  joint velocity, last action -- no linear velocity, command or depth.
  Critic adds base linear velocity and root height.
* Rewards: uprightness, standing height and stand pose (gated by uprightness),
  action rate, joint power, joint velocity and assumed-range penalties. No early
  termination; 6 s episodes.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from guide_robot_v2_cfg import GUIDE_ROBOT_V2_CFG, GUIDE_V2_BASE_BODY
from v14_ros2_bridge import implicit_joint_drives
from v15_env_cfg import V15_GAIN_SCALE_RANGE, V15_MAX_ACTION_DELAY_STEPS, V15DelayedLowPassJointPositionActionCfg
from v15_math import STAND_ROOT_HEIGHT_M, sample_recovery_orientations, standing_gate, upright_score


RECOVERY_ACTION_SCALE = 0.5  # rad per action unit; rolling over needs larger joint travel than walking
FALLEN_FRACTION = 0.8
DROP_ROOT_HEIGHT_M = 0.8
ROBOT = SceneEntityCfg("robot")
# ASSUMPTION, not hardware data: half-range around the stand pose (HAA 0, HFE -0.55, KFE 1.10).
JOINT_RANGE_RAD = {"HAA": 0.5, "HFE": 1.2, "KFE": 1.0}


def joint_half_range(asset) -> torch.Tensor:
    return torch.tensor([JOINT_RANGE_RAD[name.rsplit("_", 1)[1]] for name in asset.joint_names], device=asset.device)


def root_height(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def root_height_observation(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return root_height(env, asset_cfg)[:, None]


def reset_fallen_or_standing(env, env_ids, fallen_fraction: float, drop_height_m: float,
                             asset_cfg: SceneEntityCfg = ROBOT) -> None:
    asset = env.scene[asset_cfg.name]
    count = len(env_ids)
    fallen, quat = sample_recovery_orientations(count, fallen_fraction, device=env.device)
    position = env.scene.env_origins[env_ids].clone()
    position[:, 2] += torch.where(fallen, drop_height_m, STAND_ROOT_HEIGHT_M + 0.02)
    asset.write_root_pose_to_sim(torch.cat((position, quat), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(count, 6, device=env.device), env_ids=env_ids)
    default = asset.data.default_joint_pos[env_ids]
    random = default + (2.0 * torch.rand_like(default) - 1.0) * joint_half_range(asset)
    joint_pos = torch.where(fallen[:, None], random, default)
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


def upright(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return upright_score(env.scene[asset_cfg.name].data.projected_gravity_b).square()


def standing_height(env, std_m: float, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    gate = standing_gate(env.scene[asset_cfg.name].data.projected_gravity_b)
    return gate * torch.exp(-((root_height(env, asset_cfg) - STAND_ROOT_HEIGHT_M) / std_m).square())


def stand_pose(env, std_rad2: float, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    offset = (asset.data.joint_pos - asset.data.default_joint_pos).square().sum(-1)
    return standing_gate(asset.data.projected_gravity_b) * torch.exp(-offset / std_rad2)


def joint_range_excess(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    """Squared travel beyond the assumed working range; the asset itself has no joint limits."""
    asset = env.scene[asset_cfg.name]
    offset = (asset.data.joint_pos - asset.data.default_joint_pos).abs()
    return (offset - joint_half_range(asset)).clamp_min(0.0).square().sum(-1)


def joint_power(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return (asset.data.applied_torque * asset.data.joint_vel).abs().sum(-1)


@configclass
class V15RecoverySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = GUIDE_ROBOT_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class V15RecoveryObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        root_height = ObsTerm(func=root_height_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class V15RecoveryActionsCfg:
    joint_pos = V15DelayedLowPassJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=RECOVERY_ACTION_SCALE, use_default_offset=True,
        cutoff_hz=25.0, effort_guard_fraction=0.99, max_delay_steps=V15_MAX_ACTION_DELAY_STEPS,
    )


@configclass
class V15RecoveryEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.4, 1.2),
                "dynamic_friction_range": (0.4, 1.0), "restitution_range": (0.0, 0.2), "num_buckets": 64},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=GUIDE_V2_BASE_BODY),
                "mass_distribution_params": (-2.0, 4.0), "operation": "add"},
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": V15_GAIN_SCALE_RANGE,
                "damping_distribution_params": V15_GAIN_SCALE_RANGE,
                "operation": "scale", "distribution": "uniform"},
    )
    reset_robot = EventTerm(
        func=reset_fallen_or_standing, mode="reset",
        params={"fallen_fraction": FALLEN_FRACTION, "drop_height_m": DROP_ROOT_HEIGHT_M},
    )


@configclass
class V15RecoveryRewardsCfg:
    upright = RewTerm(func=upright, weight=1.0)
    standing_height = RewTerm(func=standing_height, weight=1.0, params={"std_m": 0.08})
    stand_pose = RewTerm(func=stand_pose, weight=0.5, params={"std_rad2": 0.5})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_power = RewTerm(func=joint_power, weight=-2.0e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_range = RewTerm(func=joint_range_excess, weight=-1.0)


@configclass
class V15RecoveryTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class GuideV15RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    scene: V15RecoverySceneCfg = V15RecoverySceneCfg(num_envs=4096, env_spacing=3.0)
    observations: V15RecoveryObservationsCfg = V15RecoveryObservationsCfg()
    actions: V15RecoveryActionsCfg = V15RecoveryActionsCfg()
    events: V15RecoveryEventsCfg = V15RecoveryEventsCfg()
    rewards: V15RecoveryRewardsCfg = V15RecoveryRewardsCfg()
    terminations: V15RecoveryTerminationsCfg = V15RecoveryTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        implicit_joint_drives(self.scene.robot)
