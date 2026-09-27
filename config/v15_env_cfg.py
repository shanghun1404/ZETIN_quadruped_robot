"""V15 locomotion: adapt the V14 5700 actor to PhysX joint-drive PD.

V14's ROS2 A/B showed the 5700 actor falls on flat ground once PD moves from
IsaacLab's explicit DCMotor into the PhysX joint drive (reports/v14/ros2). V15
fine-tunes it on that plant, the one the ROS2 structure drives, with:

* PhysX implicit PD (same 300/8 gains and 24.8/48 N m limits), gains scaled
  per environment at startup by U(0.8, 1.2);
* the V13 25 Hz target LPF and 0.99 torque-envelope guard, computed with the
  nominal gains as the ROS2 joint command node does;
* a policy-target delay of 0-3 physics steps (0-15 ms), resampled per episode;
* V14 stair-focused terrain curriculum, V14 capsules and the lane heading
  command with the original yaw reward (V14's strong yaw reward hurt stairs).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from guide_robot_v2_cfg import HAA_EFFORT_LIMIT_NM, HFE_KFE_EFFORT_LIMIT_NM
from v13_torque_actions import V13LowPassJointPositionAction, V13LowPassJointPositionActionCfg
from v13_torque_mdp import low_pass_alpha
from v14_heading_command import make_lane_command_cfg
from v14_leg_env_cfg import GuideV14StairEnvCfg
from v14_ros2_bridge import implicit_joint_drives
from v15_math import TargetDelayLine


V15_GAIN_SCALE_RANGE = (0.8, 1.2)
V15_MAX_ACTION_DELAY_STEPS = 3


class V15DelayedLowPassJointPositionAction(V13LowPassJointPositionAction):
    """V13 LPF + guard on PhysX drives, fed by a per-environment delayed policy target."""

    cfg: "V15DelayedLowPassJointPositionActionCfg"

    def __init__(self, cfg: "V15DelayedLowPassJointPositionActionCfg", env) -> None:
        # V13's constructor accepts only the explicit DCMotor; replicate it for PhysX drives.
        JointPositionAction.__init__(self, cfg, env)
        for name, motor in env.cfg.scene.robot.actuators.items():
            if (motor.stiffness, motor.damping) != (300.0, 8.0) or motor.class_type.__name__ != "ImplicitActuator":
                raise ValueError(f"V15 requires reviewed PhysX joint drives, got {name}")
        if cfg.effort_guard_fraction is None or not 0.0 <= cfg.max_delay_steps <= 10:
            raise ValueError("V15 keeps the torque guard and a bounded target delay")
        self._alpha = low_pass_alpha(cutoff_hz=cfg.cutoff_hz, control_dt_s=env.cfg.sim.dt)
        self._filtered_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._raw_targets = self._filtered_targets.clone()
        self._effort_limits = torch.tensor(
            [HAA_EFFORT_LIMIT_NM if name.endswith("_HAA") else HFE_KFE_EFFORT_LIMIT_NM
             for name in self._joint_names],
            device=self.device, dtype=self._filtered_targets.dtype,
        )
        self._delay_line = TargetDelayLine(self._raw_targets, cfg.max_delay_steps)

    def apply_actions(self) -> None:
        current = self._raw_targets.clone()
        # V13.apply_actions filters self._raw_targets; hand it the delayed target for this step.
        self._raw_targets.copy_(self._delay_line.push(current))
        super().apply_actions()
        self._raw_targets.copy_(current)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = self._delay_line.index if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        self._delay_line.reset(ids, self._raw_targets[ids])


@configclass
class V15DelayedLowPassJointPositionActionCfg(V13LowPassJointPositionActionCfg):
    class_type: type[ActionTerm] = V15DelayedLowPassJointPositionAction
    effort_guard_fraction: float | None = 0.99
    max_delay_steps: int = V15_MAX_ACTION_DELAY_STEPS


def use_physx_joint_drives(env_cfg, *, max_delay_steps: int, guard_fraction: float = 0.99) -> None:
    """Swap the V13 explicit-DCMotor action for PhysX drives + V15 delayed LPF/guard."""
    implicit_joint_drives(env_cfg.scene.robot)
    env_cfg.actions.joint_pos = V15DelayedLowPassJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True,
        cutoff_hz=25.0, effort_guard_fraction=guard_fraction, max_delay_steps=max_delay_steps,
    )


@configclass
class GuideV15LocomotionEnvCfg(GuideV14StairEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_physx_joint_drives(self, max_delay_steps=V15_MAX_ACTION_DELAY_STEPS)
        # Implicit gains go through CPU tensors, so IsaacLab advises startup-only sampling.
        self.events.v15_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": V15_GAIN_SCALE_RANGE,
                "damping_distribution_params": V15_GAIN_SCALE_RANGE,
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        self.commands.base_velocity = make_lane_command_cfg(self.commands.base_velocity)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
