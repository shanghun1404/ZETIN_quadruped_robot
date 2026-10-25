from __future__ import annotations

from guide_frame import guide_forward_speed_range_to_base


def _zero_range(keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    return {key: (0.0, 0.0) for key in keys}


def configure_deterministic_evaluation(env_cfg, num_envs: int, semantic_speed: float) -> None:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if semantic_speed <= 0.0:
        raise ValueError("semantic_speed must be positive")
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = 60.0
    env_cfg.seed = 42
    env_cfg.curriculum = None
    for group_name in ("proprio", "terrain", "command", "critic"):
        getattr(env_cfg.observations, group_name).enable_corruption = False
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (60.0, 60.0)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = guide_forward_speed_range_to_base((semantic_speed, semantic_speed))
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)
    env_cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.7, 0.7)
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    zero_ranges = _zero_range(("x", "y", "z", "roll", "pitch", "yaw"))
    env_cfg.events.reset_base.params["pose_range"] = zero_ranges
    env_cfg.events.reset_base.params["velocity_range"] = zero_ranges
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.actions.joint_pos.alpha_range = (0.275, 0.275)
    env_cfg.terminations.support_loss = None
