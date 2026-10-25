"""V15 actor actions on the unchanged CAD articulation, matching ROS2 WALK.

All public action tensors use ``contract.JOINT_NAMES`` and policy signs. Only
the articulation boundary uses CAD joint names and mirrored physical signs.
The LPF state remains unguarded, just as in ``JointCommandFilter``. WALK's
second torque guard is applied after that filter's target guard.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from v14_ros2_control import contract
from v13_torque_mdp import clip_position_target_to_motor_effort, low_pass_alpha, low_pass_step
from v15_cad_contract import joint_mapping
from v15_math import TargetDelayLine


class V15CADJointPositionAction(ActionTerm):
    """Delay, filter and guard policy targets before applying signed CAD targets."""

    cfg: "V15CADJointPositionActionCfg"

    def __init__(self, cfg: "V15CADJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        if not isinstance(cfg.max_delay_steps, int) or not 0 <= cfg.max_delay_steps <= 10:
            raise ValueError("max_delay_steps must be an integer in [0, 10]")
        if not math.isfinite(cfg.scale) or cfg.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not math.isfinite(cfg.rms_time_constant_s) or cfg.rms_time_constant_s <= 0:
            raise ValueError("rms_time_constant_s must be finite and positive")
        for fraction in (cfg.effort_guard_fraction, cfg.walk_guard_fraction):
            if fraction is not None and (not math.isfinite(fraction) or not 0 < fraction <= 1):
                raise ValueError("guard fractions must be None or in (0, 1]")
        if cfg.clip is not None:
            raise ValueError("CAD actions preserve the actor's unclipped raw action contract")
        if not math.isclose(env.cfg.sim.dt, contract.PHYSICS_DT_S, rel_tol=0, abs_tol=1e-12):
            raise ValueError("CAD WALK requires the ROS2 0.005 s physics step")

        names, signs = joint_mapping(contract.JOINT_NAMES)
        self._joint_ids, found = self._asset.find_joints(list(names), preserve_order=True)
        if tuple(found) != names:
            raise ValueError("CAD joint mapping does not cover the articulation in policy order")
        self._joint_names = list(contract.JOINT_NAMES)
        dtype = self._asset.data.joint_pos.dtype
        self._sign = torch.as_tensor(signs, device=self.device, dtype=dtype)
        self._default_policy = torch.tensor(
            [contract.joint_default(name) for name in contract.JOINT_NAMES], device=self.device, dtype=dtype,
        )
        self._effort_limits = torch.tensor(
            [contract.joint_effort_limit(name) for name in contract.JOINT_NAMES], device=self.device, dtype=dtype,
        )
        self._alpha = low_pass_alpha(cutoff_hz=cfg.cutoff_hz, control_dt_s=env.cfg.sim.dt)
        self._rms_alpha = 1.0 - math.exp(-env.cfg.sim.dt / cfg.rms_time_constant_s)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device, dtype=dtype)
        self._raw_targets = self.joint_pos_policy.clone()
        self._filtered_targets = self._raw_targets.clone()
        self._processed_actions = self._raw_targets.clone()
        self._estimated_torque_policy = torch.zeros_like(self._raw_actions)
        self._torque_squared_ewma_policy = torch.zeros_like(self._raw_actions)
        self._delay_line = TargetDelayLine(self._raw_targets, cfg.max_delay_steps)
        self._delay_line.reset(self._delay_line.index, self._raw_targets)

    @property
    def action_dim(self) -> int:
        return len(contract.JOINT_NAMES)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def raw_targets(self) -> torch.Tensor:
        """Undelayed ``q_default + scale * action``, in policy coordinates."""
        return self._raw_targets

    @property
    def processed_actions(self) -> torch.Tensor:
        """Last guarded position command, in policy coordinates."""
        return self._processed_actions

    @property
    def joint_pos_policy(self) -> torch.Tensor:
        return self._asset.data.joint_pos[:, self._joint_ids] * self._sign

    @property
    def joint_vel_policy(self) -> torch.Tensor:
        return self._asset.data.joint_vel[:, self._joint_ids] * self._sign

    @property
    def joint_pos_rel_policy(self) -> torch.Tensor:
        return self.joint_pos_policy - self._default_policy

    @property
    def estimated_torque_policy(self) -> torch.Tensor:
        """Last pre-step implicit PD estimate, clipped to actual drive caps.

        This is a cached estimate from ``apply_actions``, with randomized drive
        gains and the state used to form the command. It is not a measured
        solver torque and is not recomputed from the later post-step state.
        """
        return self._estimated_torque_policy

    @property
    def estimated_torque_rms_policy(self) -> torch.Tensor:
        """Training-only RMS in Nm from the pre-step estimate at every physics tick.

        This zero-initialized EWMA tracks squared torque independently for each
        environment and policy joint. It does not alter the position command.
        """
        return self._torque_squared_ewma_policy.sqrt()

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape or not bool(torch.isfinite(actions).all()):
            raise ValueError("expected finite [environment, 12] policy actions")
        self._raw_actions.copy_(actions)
        self._raw_targets.copy_(self._default_policy + self.cfg.scale * actions)

    def apply_actions(self) -> None:
        delayed = self._delay_line.push(self._raw_targets)
        self._filtered_targets.copy_(low_pass_step(delayed, self._filtered_targets, self._alpha))
        position = self.joint_pos_policy
        velocity = self.joint_vel_policy
        target = self._filtered_targets
        if self.cfg.effort_guard_fraction is not None:
            target = clip_position_target_to_motor_effort(
                target, position, velocity, self._effort_limits,
                stiffness=contract.PD_STIFFNESS, damping=contract.PD_DAMPING,
                velocity_limit=contract.MOTOR_VELOCITY_LIMIT, fraction=self.cfg.effort_guard_fraction,
            )
        if self.cfg.walk_guard_fraction is not None:
            # hybrid_motor_command(weight=1) -> torque_position_target: unlike
            # the preceding midpoint guard, this scales the envelope endpoints.
            requested = contract.PD_STIFFNESS * (target - position) - contract.PD_DAMPING * velocity
            upper = self.cfg.walk_guard_fraction * self._effort_limits * (
                1 - velocity / contract.MOTOR_VELOCITY_LIMIT
            ).clamp(0, 1)
            lower = self.cfg.walk_guard_fraction * self._effort_limits * (
                -1 - velocity / contract.MOTOR_VELOCITY_LIMIT
            ).clamp(-1, 0)
            bounded = torch.minimum(torch.maximum(requested, lower), upper)
            target = position + (bounded + contract.PD_DAMPING * velocity) / contract.PD_STIFFNESS
        self._processed_actions.copy_(target)
        physical_target = target * self._sign
        self._asset.set_joint_position_target(physical_target, joint_ids=self._joint_ids)

        # ImplicitActuator's reported applied effort uses the same pre-step PD
        # estimate. Read the actual drive buffers, not the nominal guard gains.
        data = self._asset.data
        physical_position = data.joint_pos[:, self._joint_ids]
        physical_velocity = data.joint_vel[:, self._joint_ids]
        stiffness = data.joint_stiffness[:, self._joint_ids]
        damping = data.joint_damping[:, self._joint_ids]
        caps = data.joint_effort_limits[:, self._joint_ids]
        requested_physical = stiffness * (physical_target - physical_position) - damping * physical_velocity
        clipped_physical = torch.minimum(torch.maximum(requested_physical, -caps), caps)
        self._estimated_torque_policy.copy_(clipped_physical * self._sign)
        self._torque_squared_ewma_policy.mul_(1.0 - self._rms_alpha).addcmul_(
            self._estimated_torque_policy, self._estimated_torque_policy, value=self._rms_alpha,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            ids = self._delay_line.index
        elif isinstance(env_ids, slice):
            ids = self._delay_line.index[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        measured = self.joint_pos_policy[ids]
        self._raw_actions[ids] = 0
        self._raw_targets[ids] = measured
        self._filtered_targets[ids] = measured
        self._processed_actions[ids] = measured
        self._estimated_torque_policy[ids] = 0
        self._torque_squared_ewma_policy[ids] = 0
        self._delay_line.reset(ids, measured)


@configclass
class V15CADJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = V15CADJointPositionAction
    asset_name: str = "robot"
    scale: float = 0.2
    cutoff_hz: float = 25.0
    effort_guard_fraction: float | None = 0.99
    walk_guard_fraction: float | None = 0.99
    max_delay_steps: int = 3
    rms_time_constant_s: float = 0.5


def canonical_joint_pos_rel(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Actor joint positions relative to reviewed policy defaults, in policy order."""
    return env.action_manager.get_term(action_name).joint_pos_rel_policy


def canonical_joint_vel(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Actor joint velocities with the reviewed order and mirrored-axis signs."""
    return env.action_manager.get_term(action_name).joint_vel_policy
