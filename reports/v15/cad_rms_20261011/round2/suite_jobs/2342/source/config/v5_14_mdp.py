"""V5.14 reward term: penalize the most over-planted foot."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import smooth_mdp
from v5_14_math import worst_foot_swing_deficit


class overplanted_foot_penalty(ManagerTermBase):
    """Bounded penalty for a foot that almost never leaves the ground.

    Shares the exact contact-duty exponential moving average bookkeeping used by
    ``v5_12_mdp.abandoned_foot_penalty``; this term is its mirror image, pressing
    on the over-planted forefeet rather than the abandoned rear foot.
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
        swing_floor: float = 0.25,
    ) -> torch.Tensor:
        if window_s <= 0.0:
            raise ValueError("window_s must be positive")
        contacts = smooth_mdp.foot_contact_flags(env, threshold, sensor_cfg)
        alpha = 1.0 - math.exp(-env.step_dt / window_s)
        self._duty.add_(alpha * (contacts - self._duty))
        command = smooth_mdp.guide_velocity_command(env, command_name)[:, 0]
        moving = torch.abs(command) >= moving_threshold
        return worst_foot_swing_deficit(self._duty, moving, swing_floor=swing_floor)
