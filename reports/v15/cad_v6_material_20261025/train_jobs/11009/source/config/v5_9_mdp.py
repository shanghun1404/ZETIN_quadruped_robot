from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import smooth_mdp
from v5_9_math import usage_gate_factor


class usage_gated_forward_velocity(ManagerTermBase):
    """Forward velocity tracking multiplied by a symmetric per-foot usage gate.

    Identical bookkeeping to V5.8 (a per-foot 1.0-second EMA of the contact
    flag), but the gate now closes on either the contact or the swing side, so a
    leg that is abandoned *or* permanently planted shrinks the forward reward.
    When every foot walks within the healthy duty band the gate is ``1.0`` and
    the reward equals the stock ``track_lin_vel_xy_exp`` value.
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
        std: float,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        moving_threshold: float = 0.05,
        window_s: float = 1.0,
        contact_floor: float = 0.35,
        swing_floor: float = 0.10,
        gain: float = 1.5,
        min_gate: float = 0.0,
    ) -> torch.Tensor:
        if moving_threshold < 0.0:
            raise ValueError("moving_threshold must be non-negative")
        if window_s <= 0.0:
            raise ValueError("window_s must be positive")

        contacts = smooth_mdp.foot_contact_flags(env, threshold, sensor_cfg)
        alpha = 1.0 - math.exp(-env.step_dt / window_s)
        self._duty.add_(alpha * (contacts - self._duty))

        command = smooth_mdp.guide_velocity_command(env, command_name)[:, 0]
        moving = torch.abs(command) >= moving_threshold
        gate = usage_gate_factor(
            self._duty,
            moving,
            contact_floor=contact_floor,
            swing_floor=swing_floor,
            gain=gain,
            min_gate=min_gate,
        )
        tracking = base_mdp.track_lin_vel_xy_exp(
            env, std=std, command_name=command_name
        )
        return tracking * gate
