from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import smooth_mdp
from v5_8_math import usage_gate_factor


class usage_gated_forward_velocity(ManagerTermBase):
    """Forward linear-velocity tracking multiplied by a per-foot usage gate.

    The term maintains a per-foot exponential moving average of the binary
    contact flag (identical bookkeeping to the V5.7 ``per_foot_usage_floor``
    term) and multiplies the stock ``track_lin_vel_xy_exp`` reward by
    :func:`v5_8_math.usage_gate_factor`. When every foot clears ``contact_floor``
    the gate is ``1.0`` and the reward equals the stock forward-tracking reward,
    so V5.8 reduces to V5.6/V5.7 on healthy four-leg gaits. When a leg is
    abandoned the gate shrinks the forward reward, removing the incentive that
    let the three-leg cheat survive the additive V5.7 floor penalty.
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
            gain=gain,
            min_gate=min_gate,
        )
        tracking = base_mdp.track_lin_vel_xy_exp(
            env, std=std, command_name=command_name
        )
        return tracking * gate
