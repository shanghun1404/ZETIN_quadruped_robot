from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from smooth_math import ema_step


class SmoothJointPositionAction(JointPositionAction):
    cfg: "SmoothJointPositionActionCfg"

    def __init__(self, cfg: "SmoothJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._current_targets = default.clone()
        self._previous_targets = default.clone()
        self._older_targets = default.clone()
        self._alpha = torch.empty((self.num_envs, 1), device=self.device)
        self._sample_alpha(torch.arange(self.num_envs, device=self.device))

    def _sample_alpha(self, env_ids: torch.Tensor) -> None:
        low, high = self.cfg.alpha_range
        self._alpha[env_ids] = torch.empty_like(self._alpha[env_ids]).uniform_(low, high)

    @property
    def current_targets(self) -> torch.Tensor:
        return self._current_targets

    @property
    def previous_targets(self) -> torch.Tensor:
        return self._previous_targets

    @property
    def older_targets(self) -> torch.Tensor:
        return self._older_targets

    @property
    def filtered_residual(self) -> torch.Tensor:
        return (self._current_targets - self._offset) / self._scale

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        raw_targets = self._processed_actions.clone()
        self._older_targets.copy_(self._previous_targets)
        self._previous_targets.copy_(self._current_targets)
        filtered = ema_step(raw_targets, self._previous_targets, self._alpha)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._current_targets.copy_(torch.clamp(filtered, limits[..., 0], limits[..., 1]))
        self._processed_actions.copy_(self._current_targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        current = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._current_targets[ids] = current
        self._previous_targets[ids] = current
        self._older_targets[ids] = current
        self._sample_alpha(ids)


@configclass
class SmoothJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = SmoothJointPositionAction
    alpha_range: tuple[float, float] = (0.20, 0.35)
