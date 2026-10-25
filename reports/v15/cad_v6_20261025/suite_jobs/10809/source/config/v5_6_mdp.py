from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import smooth_mdp
from v5_6_math import (
    all_feet_stance_penalty,
    contact_transition_burst_penalty,
    foot_usage_imbalance_penalty,
)


def _moving_command(
    env,
    command_name: str,
    moving_threshold: float,
) -> torch.Tensor:
    if moving_threshold < 0.0:
        raise ValueError("moving_threshold must be non-negative")
    command = smooth_mdp.guide_velocity_command(env, command_name)[:, 0]
    return torch.abs(command) >= moving_threshold


def _transition_flags(
    env,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    first_air = sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    return first_contact, first_air


def all_feet_stance(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.05,
) -> torch.Tensor:
    contacts = smooth_mdp.foot_contact_flags(env, threshold, sensor_cfg)
    moving = _moving_command(env, command_name, moving_threshold)
    return all_feet_stance_penalty(contacts, moving)


def contact_transition_burst(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.05,
    max_per_direction: int = 2,
) -> torch.Tensor:
    first_contact, first_air = _transition_flags(env, sensor_cfg)
    moving = _moving_command(env, command_name, moving_threshold)
    return contact_transition_burst_penalty(
        first_contact,
        first_air,
        moving,
        max_per_direction=max_per_direction,
    )


class foot_usage_imbalance(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._duty = torch.ones((env.num_envs, 4), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = smooth_mdp._env_ids(self._env, env_ids)
        self._duty[ids] = 1.0

    def __call__(
        self,
        env,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        moving_threshold: float = 0.05,
        window_s: float = 1.0,
    ) -> torch.Tensor:
        if window_s <= 0.0:
            raise ValueError("window_s must be positive")
        contacts = smooth_mdp.foot_contact_flags(env, threshold, sensor_cfg)
        alpha = 1.0 - math.exp(-env.step_dt / window_s)
        self._duty.add_(alpha * (contacts - self._duty))
        moving = _moving_command(env, command_name, moving_threshold)
        return foot_usage_imbalance_penalty(self._duty, moving)
