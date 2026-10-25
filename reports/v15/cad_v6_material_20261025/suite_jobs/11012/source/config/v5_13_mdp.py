"""V5.13 reward term: continuous in-phase penalty on the leg pairs."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import smooth_mdp
from v5_13_math import pair_in_phase_penalty


class pair_in_phase(ManagerTermBase):
    """Penalize the left/right legs of a pair swinging in the same direction.

    Reads HFE joint velocities directly, so the penalty is dense (every step)
    and cannot be dodged by offsetting footfall timing -- the loophole that made
    the V5.12 event-based ``pair_synchrony`` term ineffective.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._env = env

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        moving_threshold: float = 0.05,
        vel_scale: float = 4.0,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
        command = smooth_mdp.guide_velocity_command(env, command_name)[:, 0]
        moving = torch.abs(command) >= moving_threshold
        return pair_in_phase_penalty(velocity, moving, vel_scale=vel_scale)
