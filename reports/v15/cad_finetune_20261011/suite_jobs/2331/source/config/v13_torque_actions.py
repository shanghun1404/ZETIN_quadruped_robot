"""A first-order low-pass filter on V13 position targets at the physics rate."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from guide_robot_v2_cfg import HAA_EFFORT_LIMIT_NM, HFE_KFE_EFFORT_LIMIT_NM
from v13_torque_mdp import clip_position_target_to_motor_effort, low_pass_alpha, low_pass_step


class V13LowPassJointPositionAction(JointPositionAction):
    """Filter policy position targets and reset only the environments that restart."""

    cfg: "V13LowPassJointPositionActionCfg"

    def __init__(self, cfg: "V13LowPassJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._alpha = low_pass_alpha(cutoff_hz=cfg.cutoff_hz, control_dt_s=env.cfg.sim.dt)
        self._filtered_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._raw_targets = self._filtered_targets.clone()
        self._effort_limits = None
        if cfg.effort_guard_fraction is not None:
            actuators = env.cfg.scene.robot.actuators
            hip = actuators["hip_abduction"]
            knee = actuators["hip_knee"]
            for motor, effort in ((hip, HAA_EFFORT_LIMIT_NM), (knee, HFE_KFE_EFFORT_LIMIT_NM)):
                if (motor.stiffness, motor.damping, motor.velocity_limit, motor.effort_limit,
                    motor.saturation_effort) != (300.0, 8.0, 50.0, effort, effort):
                    raise ValueError("effort guard requires the reviewed robot_v2 DCMotor settings")
            self._effort_limits = torch.tensor(
                [HAA_EFFORT_LIMIT_NM if name.endswith("_HAA") else HFE_KFE_EFFORT_LIMIT_NM
                 for name in self._joint_names],
                device=self.device,
                dtype=self._filtered_targets.dtype,
            )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._raw_targets.copy_(self._processed_actions)

    def apply_actions(self) -> None:
        self._filtered_targets.copy_(low_pass_step(self._raw_targets, self._filtered_targets, self._alpha))
        self._processed_actions.copy_(self._filtered_targets)
        if self._effort_limits is not None:
            self._processed_actions.copy_(clip_position_target_to_motor_effort(
                self._processed_actions,
                self._asset.data.joint_pos[:, self._joint_ids],
                self._asset.data.joint_vel[:, self._joint_ids],
                self._effort_limits,
                stiffness=300.0,
                damping=8.0,
                velocity_limit=50.0,
                fraction=self.cfg.effort_guard_fraction,
            ))
        super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._filtered_targets[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._raw_targets[ids] = self._filtered_targets[ids]


@configclass
class V13LowPassJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = V13LowPassJointPositionAction
    cutoff_hz: float = 25.0
    effort_guard_fraction: float | None = None
