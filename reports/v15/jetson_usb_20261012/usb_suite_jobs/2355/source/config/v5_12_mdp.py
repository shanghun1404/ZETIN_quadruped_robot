"""V5.12 reward terms that penalize the two measured V5.11 gait cheats."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import smooth_mdp
from v5_12_math import pair_synchrony_penalty, worst_foot_contact_deficit


def _moving_mask(env, command_name: str, moving_threshold: float) -> torch.Tensor:
    """True where the robot is being asked to travel, on whichever planar axis.

    This read column 0 alone until V10.2. That is the x channel -- the guide
    transform negates x and y without swapping them -- so when V10.1 moved the
    forward command to base +y and pinned ``lin_vel_x`` to zero, the mask went
    False everywhere and the duty floor it gates returned zero for an entire
    3000-iteration run. The term stayed registered at -3.0 the whole time; a
    masked-off penalty reports as satisfied, not as missing.

    The planar magnitude is axis-independent. Every configuration older than
    V10.1 pins the lateral band to exactly zero (``smooth_env_cfg.py`` line 192,
    ``v8_env_cfg.py`` line 151), so there the norm of ``(x, 0)`` is ``|x|`` and
    this is numerically the same mask those results were measured under.
    """
    command = smooth_mdp.guide_velocity_command(env, command_name)
    planar_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    return planar_speed >= moving_threshold


class front_rear_pair_synchrony(ManagerTermBase):
    """Penalty for the left/right feet of a pair lifting or landing together.

    Reads the contact sensor's first-contact / first-air event flags, which are
    the same signals the evaluator uses to build its footfall statistics, so the
    quantity trained against is exactly the quantity reported.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        moving_threshold: float = 0.05,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        first_air = sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
        moving = _moving_mask(env, command_name, moving_threshold)
        return pair_synchrony_penalty(first_contact, first_air, moving)


class abandoned_foot_penalty(ManagerTermBase):
    """Bounded penalty for the least-used foot falling below a contact floor.

    Maintains the same per-foot contact-duty exponential moving average as the
    V5.7/V5.8 terms, but applies it as an additive bounded penalty on the worst
    foot rather than as a multiplicative gate on the forward reward. The gate
    form starved the forward payoff and produced the stiff, over-planted gait;
    the penalty form leaves the reference forward reward intact.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._duty = torch.full((env.num_envs, 4), 0.5, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = smooth_mdp._env_ids(self._env, env_ids)
        self._duty[ids] = 0.5

    def __call__(
        self,
        env,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        moving_threshold: float = 0.05,
        window_s: float = 1.0,
        contact_floor: float = 0.35,
    ) -> torch.Tensor:
        if window_s <= 0.0:
            raise ValueError("window_s must be positive")

        contacts = smooth_mdp.foot_contact_flags(env, threshold, sensor_cfg)
        alpha = 1.0 - math.exp(-env.step_dt / window_s)
        self._duty.add_(alpha * (contacts - self._duty))

        moving = _moving_mask(env, command_name, moving_threshold)
        return worst_foot_contact_deficit(
            self._duty, moving, contact_floor=contact_floor
        )
