from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from guide_frame import (
    base_to_guide_observation,
    guide_forward_speed_range_to_base,
    rotate_height_scan_180,
)
from smooth_math import (
    bounded_square_error,
    flight_indicator,
    heading_error_features,
    hybrid_signed_progress_ratio,
    normalize_foot_forces,
    quaternion_yaw_wxyz,
    smoothstep,
    support_deficit_ratio,
    swing_clearance_penalty,
    target_jerk_penalty,
    target_rate_penalty,
    vertical_acceleration_step,
    wrapped_angle_error,
    wrong_direction_ratio,
)
from v5_schedule import semantic_speed_range


def _action_term(env):
    return env.action_manager.get_term("joint_pos")


def filtered_action_observation(env) -> torch.Tensor:
    return _action_term(env).filtered_residual


def guide_base_lin_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return base_to_guide_observation(env.scene[asset_cfg.name].data.root_lin_vel_b)


def guide_base_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return base_to_guide_observation(env.scene[asset_cfg.name].data.root_ang_vel_b)


def guide_projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return base_to_guide_observation(env.scene[asset_cfg.name].data.projected_gravity_b)


def guide_velocity_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    return base_to_guide_observation(env.command_manager.get_command(command_name))


def guide_height_scan(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    raw = base_mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    return rotate_height_scan_180(raw)


def central_ground_height(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner")) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w[..., 2].reshape(-1, 17, 11)
    center = hits[:, 7:10, 4:7].reshape(hits.shape[0], -1)
    fallback = env.scene["robot"].data.root_pos_w[:, 2:3] - 0.40
    center = torch.where(torch.isfinite(center), center, fallback)
    return torch.median(center, dim=1).values


def local_base_height_observation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    height = env.scene[asset_cfg.name].data.root_pos_w[:, 2] - central_ground_height(env, sensor_cfg)
    return height.unsqueeze(-1)


def base_height_centered(
    env,
    target: float = 0.40,
    scale: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    height = local_base_height_observation(env, asset_cfg, sensor_cfg).squeeze(-1)
    return bounded_square_error(height - target, scale)


def lateral_velocity_normalized(
    env,
    scale: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return bounded_square_error(guide_base_lin_vel(env, asset_cfg)[:, 1], scale)


def yaw_rate_normalized(
    env,
    scale: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return bounded_square_error(guide_base_ang_vel(env, asset_cfg)[:, 2], scale)


def base_height_below(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    return local_base_height_observation(env, asset_cfg, sensor_cfg).squeeze(-1) < minimum_height


def _contact_forces(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return torch.max(torch.linalg.vector_norm(history, dim=-1), dim=1).values


def foot_contact_flags(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return (_contact_forces(env, sensor_cfg) > threshold).float()


def signed_forward_progress(env, command_name: str = "base_velocity") -> torch.Tensor:
    linear_velocity = guide_base_lin_vel(env)
    angular_velocity = guide_base_ang_vel(env)
    forward_command = guide_velocity_command(env, command_name)[:, 0]
    yaw = quaternion_yaw_wxyz(env.scene["robot"].data.root_quat_w)
    heading_error = wrapped_angle_error(torch.zeros_like(yaw), yaw)
    roll_pitch_rate = torch.linalg.vector_norm(angular_velocity[:, :2], dim=-1)
    return hybrid_signed_progress_ratio(
        linear_velocity[:, 0],
        forward_command,
        linear_velocity[:, 1],
        roll_pitch_rate,
        heading_error,
    )


def wrong_direction(env) -> torch.Tensor:
    return wrong_direction_ratio(guide_base_lin_vel(env)[:, 0])


def support_deficit(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_count = foot_contact_flags(env, threshold, sensor_cfg).sum(dim=-1)
    return support_deficit_ratio(contact_count)


def flight(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_count = foot_contact_flags(env, threshold, sensor_cfg).sum(dim=-1)
    return flight_indicator(contact_count)


def swing_clearance_normalized(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    threshold: float = 5.0,
    low: float = 0.03,
    high: float = 0.05,
    margin: float = 0.03,
) -> torch.Tensor:
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = foot_z - central_ground_height(env, height_sensor_cfg).unsqueeze(-1)
    actual_contacts = foot_contact_flags(env, threshold, contact_sensor_cfg)
    return swing_clearance_penalty(clearance, actual_contacts, low, high, margin)


def foot_force_normalized(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    forces = _contact_forces(env, sensor_cfg)
    asset: Articulation = env.scene["robot"]
    return normalize_foot_forces(forces, asset.data.default_mass)


def thigh_calf_contact_fraction(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return (_contact_forces(env, sensor_cfg) > threshold).float().mean(dim=1)


def foot_slip_normalized(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact = foot_contact_flags(env, threshold, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return (contact * torch.clamp(speed / 0.5, 0.0, 1.0).square()).mean(dim=1)


def foot_impact_normalized(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    normalized = foot_force_normalized(env, sensor_cfg) * 4.0
    return torch.clamp(normalized - 2.0, 0.0, 1.0).square().mean(dim=1)


def filtered_target_rate(env) -> torch.Tensor:
    term = _action_term(env)
    return target_rate_penalty(term.current_targets, term.previous_targets)


def filtered_target_jerk(env) -> torch.Tensor:
    term = _action_term(env)
    return target_jerk_penalty(term.current_targets, term.previous_targets, term.older_targets)


def joint_power_normalized(
    env,
    effort_limit: float,
    velocity_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / effort_limit
    velocity = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) / velocity_limit
    return torch.clamp(torque * velocity, 0.0, 1.0).mean(dim=1)


def vertical_velocity_normalized(
    env,
    scale: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    value = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 2]
    return bounded_square_error(value, scale)


def roll_pitch_rate_normalized(
    env,
    scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    value = env.scene[asset_cfg.name].data.root_ang_vel_b[:, :2] / scale
    return torch.clamp(value.square().sum(dim=1), 0.0, 4.0)


def joint_limit_normalized(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    span = torch.clamp(limits[..., 1] - limits[..., 0], min=1.0e-6)
    violation = torch.clamp(limits[..., 0] - position, min=0.0)
    violation += torch.clamp(position - limits[..., 1], min=0.0)
    return torch.clamp(violation / (0.05 * span), 0.0, 1.0).mean(dim=1)


def _env_ids(env, env_ids: Sequence[int] | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


class heading_hold_penalty(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._target = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._target[_env_ids(self._env, env_ids)] = 0.0

    def __call__(
        self,
        env,
        scale: float = math.radians(10.0),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        yaw = quaternion_yaw_wxyz(env.scene[asset_cfg.name].data.root_quat_w)
        return bounded_square_error(wrapped_angle_error(yaw, self._target), scale)


class commanded_heading_features(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._target = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._target[_env_ids(self._env, env_ids)] = 0.0

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        yaw = quaternion_yaw_wxyz(env.scene[asset_cfg.name].data.root_quat_w)
        return heading_error_features(yaw, self._target)


class vertical_acceleration_normalized(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self._previous[ids] = 0.0
        self._initialized[ids] = False

    def __call__(
        self,
        env,
        scale: float = 2.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        current = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 2]
        penalty, self._previous, self._initialized = vertical_acceleration_step(
            current,
            self._previous,
            self._initialized,
            env.step_dt,
            scale,
        )
        return penalty


class ramp_reward_weight(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        term_name: str,
        initial_weight: float,
        final_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        del env_ids, term_name
        progress = smoothstep(effective_schedule_step(env), start_step, end_step)
        self._term_cfg.weight = initial_weight + progress * (final_weight - initial_weight)
        env.reward_manager.set_term_cfg(self.cfg.params["term_name"], self._term_cfg)
        return self._term_cfg.weight


class velocity_command_curriculum_v5(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = env.command_manager.get_term("base_velocity")

    def __call__(self, env, env_ids: Sequence[int]) -> float:
        del env_ids
        semantic_range = semantic_speed_range(effective_schedule_step(env))
        self._command.cfg.ranges.lin_vel_x = guide_forward_speed_range_to_base(semantic_range)
        return semantic_range[1]


def effective_schedule_step(env) -> int:
    return env.common_step_counter + int(env.cfg.curriculum_step_offset)
