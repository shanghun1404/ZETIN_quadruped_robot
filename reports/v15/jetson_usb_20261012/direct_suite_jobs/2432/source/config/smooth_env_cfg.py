from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.managers as mg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
)

from guide_frame import guide_forward_speed_range_to_base
from guide_robot_cfg import GUIDE_ROBOT_CFG
from smooth_actions import SmoothJointPositionActionCfg
from stance_calibration import TARGET_HEIGHT_M
import smooth_mdp


FINAL_REWARD_WEIGHTS = {
    "signed_forward_progress": 2.00,
    "track_lin_vel_xy_exp": 2.00,
    "track_ang_vel_z_exp": 1.00,
    "wrong_direction": -1.00,
    "support_deficit": -4.00,
    "flight": -8.00,
    "lateral_velocity": -0.75,
    "yaw_rate": -0.50,
    "heading_hold": -0.75,
    "base_height_centered": -0.50,
    "vertical_velocity": -0.50,
    "vertical_acceleration": -0.20,
    "roll_pitch_rate": -0.75,
    "flat_orientation": -0.75,
    "foot_impact": -0.25,
    "foot_slip": -0.20,
    "swing_clearance": -0.10,
    "filtered_target_rate": -0.08,
    "filtered_target_jerk": -0.04,
    "joint_power": -0.03,
    "undesired_contacts": -0.35,
}

RAMPED_TERMS = (
    "foot_impact",
    "foot_slip",
    "swing_clearance",
    "filtered_target_rate",
    "filtered_target_jerk",
    "joint_power",
)


@configclass
class PhaseFreeSceneCfg(MySceneCfg):
    forward_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(-0.8, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class PhaseFreeObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=smooth_mdp.guide_base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=smooth_mdp.guide_projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        filtered_action = ObsTerm(func=smooth_mdp.filtered_action_observation)
        foot_contacts = ObsTerm(
            func=smooth_mdp.foot_contact_flags,
            params={
                "threshold": 5.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TerrainCfg(ObsGroup):
        height_scan = ObsTerm(
            func=smooth_mdp.guide_height_scan,
            params={"sensor_cfg": SceneEntityCfg("forward_height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CommandCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=smooth_mdp.guide_velocity_command,
            params={"command_name": "base_velocity"},
        )
        heading_error = ObsTerm(func=smooth_mdp.commanded_heading_features)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=smooth_mdp.guide_base_lin_vel)
        base_ang_vel = ObsTerm(func=smooth_mdp.guide_base_ang_vel)
        projected_gravity = ObsTerm(func=smooth_mdp.guide_projected_gravity)
        velocity_commands = ObsTerm(
            func=smooth_mdp.guide_velocity_command,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        filtered_action = ObsTerm(func=smooth_mdp.filtered_action_observation)
        height_scan = ObsTerm(
            func=smooth_mdp.guide_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        foot_contacts = ObsTerm(
            func=smooth_mdp.foot_contact_flags,
            params={
                "threshold": 5.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
        foot_forces = ObsTerm(
            func=smooth_mdp.foot_force_normalized,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        local_height = ObsTerm(func=smooth_mdp.local_base_height_observation)
        heading_error = ObsTerm(func=smooth_mdp.commanded_heading_features)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    terrain: TerrainCfg = TerrainCfg()
    command: CommandCfg = CommandCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GroundedGuideEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene: PhaseFreeSceneCfg = PhaseFreeSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PhaseFreeObservationsCfg = PhaseFreeObservationsCfg()
    curriculum_step_offset: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GUIDE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.forward_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.forward_height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = 0

        initial_semantic_speed = (0.03, 0.06)
        self.commands.base_velocity.ranges.lin_vel_x = guide_forward_speed_range_to_base(initial_semantic_speed)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.radians(10.0), math.radians(10.0))
        self.actions.joint_pos = SmoothJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=True,
            alpha_range=(0.20, 0.35),
        )

        ordered_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        ordered_sensor = SceneEntityCfg(
            "contact_forces", body_names=ordered_feet, preserve_order=True
        )
        ordered_asset = SceneEntityCfg("robot", body_names=ordered_feet, preserve_order=True)
        foot_sensor = SceneEntityCfg("contact_forces", body_names=".*_foot")
        foot_asset = SceneEntityCfg("robot", body_names=".*_foot")

        self.terminations.base_contact = mg.TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names="base_link|.*_hip"
                ),
                "threshold": 1.0,
            },
        )
        self.terminations.low_height = mg.TerminationTermCfg(
            func=smooth_mdp.base_height_below,
            params={"minimum_height": 0.15, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.terminations.support_loss = None
        self.terminations.joint_pos_out_of_limit_body = None

        self.rewards.signed_forward_progress = mg.RewardTermCfg(
            func=smooth_mdp.signed_forward_progress,
            weight=2.00,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_exp.weight = 2.00
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.12
        self.rewards.track_ang_vel_z_exp.weight = 1.00
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.10
        self.rewards.wrong_direction = mg.RewardTermCfg(
            func=smooth_mdp.wrong_direction,
            weight=-1.00,
        )
        self.rewards.support_deficit = mg.RewardTermCfg(
            func=smooth_mdp.support_deficit,
            weight=-4.00,
            params={"threshold": 5.0, "sensor_cfg": ordered_sensor},
        )
        self.rewards.flight = mg.RewardTermCfg(
            func=smooth_mdp.flight,
            weight=-8.00,
            params={"threshold": 5.0, "sensor_cfg": ordered_sensor},
        )
        self.rewards.lateral_velocity = mg.RewardTermCfg(
            func=smooth_mdp.lateral_velocity_normalized,
            weight=-0.75,
            params={"scale": 0.12},
        )
        self.rewards.yaw_rate = mg.RewardTermCfg(
            func=smooth_mdp.yaw_rate_normalized,
            weight=-0.50,
            params={"scale": 0.10},
        )
        self.rewards.heading_hold = mg.RewardTermCfg(
            func=smooth_mdp.heading_hold_penalty,
            weight=-0.75,
            params={"scale": math.radians(8.0)},
        )
        self.rewards.base_height_centered = mg.RewardTermCfg(
            func=smooth_mdp.base_height_centered,
            weight=-0.50,
            params={"target": TARGET_HEIGHT_M, "scale": 0.03},
        )
        self.rewards.lin_vel_z_l2 = mg.RewardTermCfg(
            func=smooth_mdp.vertical_velocity_normalized,
            weight=-0.50,
            params={"scale": 0.20},
        )
        self.rewards.vertical_acceleration = mg.RewardTermCfg(
            func=smooth_mdp.vertical_acceleration_normalized,
            weight=-0.20,
            params={"scale": 2.5},
        )
        self.rewards.ang_vel_xy_l2 = mg.RewardTermCfg(
            func=smooth_mdp.roll_pitch_rate_normalized,
            weight=-0.75,
            params={"scale": 0.60},
        )
        self.rewards.flat_orientation_l2.weight = -0.75
        self.rewards.foot_impact = mg.RewardTermCfg(
            func=smooth_mdp.foot_impact_normalized,
            weight=-0.0625,
            params={"sensor_cfg": foot_sensor},
        )
        self.rewards.foot_slip = mg.RewardTermCfg(
            func=smooth_mdp.foot_slip_normalized,
            weight=-0.050,
            params={"threshold": 5.0, "sensor_cfg": foot_sensor, "asset_cfg": foot_asset},
        )
        self.rewards.swing_clearance = mg.RewardTermCfg(
            func=smooth_mdp.swing_clearance_normalized,
            weight=-0.025,
            params={
                "asset_cfg": ordered_asset,
                "contact_sensor_cfg": ordered_sensor,
                "height_sensor_cfg": SceneEntityCfg("height_scanner"),
                "threshold": 5.0,
                "low": 0.03,
                "high": 0.05,
                "margin": 0.03,
            },
        )
        self.rewards.filtered_target_rate = mg.RewardTermCfg(
            func=smooth_mdp.filtered_target_rate, weight=-0.020
        )
        self.rewards.filtered_target_jerk = mg.RewardTermCfg(
            func=smooth_mdp.filtered_target_jerk, weight=-0.010
        )
        self.rewards.joint_power = mg.RewardTermCfg(
            func=smooth_mdp.joint_power_normalized,
            weight=-0.0075,
            params={"effort_limit": 53.0, "velocity_limit": 50.0},
        )
        self.rewards.undesired_contacts = mg.RewardTermCfg(
            func=smooth_mdp.thigh_calf_contact_fraction,
            weight=-0.35,
            params={
                "threshold": 10.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*_thigh|.*_calf"
                ),
            },
        )
        self.rewards.dof_pos_limits = mg.RewardTermCfg(
            func=smooth_mdp.joint_limit_normalized, weight=-0.20
        )
        self.rewards.termination = mg.RewardTermCfg(func=base_mdp.is_terminated, weight=-25.0)
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.feet_air_time = None

        for table_name in RAMPED_TERMS:
            term_name = table_name
            final_weight = FINAL_REWARD_WEIGHTS[table_name]
            setattr(
                self.curriculum,
                f"ramp_{term_name}",
                mg.CurriculumTermCfg(
                    func=smooth_mdp.ramp_reward_weight,
                    params={
                        "term_name": term_name,
                        "initial_weight": 0.25 * final_weight,
                        "final_weight": final_weight,
                        "start_step": 16_000,
                        "end_step": 80_000,
                    },
                ),
            )
        self.curriculum.terrain_levels = None
        self.curriculum.velocity_commands = mg.CurriculumTermCfg(
            func=smooth_mdp.velocity_command_curriculum_v5
        )

        self.events.physics_material.params["static_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.9)
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
