"""200 Hz joint command stage: V13 25 Hz target LPF and 0.99 torque-envelope guard.

Mirrors ``V13LowPassJointPositionAction.apply_actions`` with the same math
(``v13_torque_mdp``), but outputs a position target for the simulator's joint
drive instead of calling IsaacLab's actuator model.
"""

from __future__ import annotations

from collections import deque

import torch

from v13_torque_mdp import clip_position_target_to_motor_effort, low_pass_alpha, low_pass_step

from v14_ros2_control import contract


class JointCommandFilter:
    def __init__(self, *, cutoff_hz: float = 25.0, guard_fraction: float | None = 0.99,
                 names=contract.JOINT_NAMES):
        self.names = tuple(names)
        self.alpha = low_pass_alpha(cutoff_hz=cutoff_hz, control_dt_s=contract.PHYSICS_DT_S)
        self.guard_fraction = guard_fraction
        self.effort_limits = torch.tensor([contract.joint_effort_limit(n) for n in self.names])
        self.filtered = None
        self.raw = None

    def reset(self, position: torch.Tensor) -> None:
        self.filtered = position.clone()
        self.raw = position.clone()

    def set_target(self, target: torch.Tensor) -> None:
        if not bool(torch.isfinite(target).all()):
            raise ValueError("non-finite joint target")
        self.raw = target.clone()

    def step(self, position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Advance one physics step; the filter state stays unguarded, as in V13."""
        self.filtered = low_pass_step(self.raw, self.filtered, self.alpha)
        command = self.filtered.clone()
        if self.guard_fraction is not None:
            command = clip_position_target_to_motor_effort(
                command[None], position[None], velocity[None], self.effort_limits,
                stiffness=contract.PD_STIFFNESS, damping=contract.PD_DAMPING,
                velocity_limit=contract.MOTOR_VELOCITY_LIMIT, fraction=self.guard_fraction,
            )[0]
        return command


class LockstepJointCommander:
    """Order joint states and policy targets by simulation stamp.

    On a policy tick the joint state waits for the target carrying the same stamp,
    so the simulator always receives the command IsaacLab would have applied.
    """

    def __init__(self, command_filter: JointCommandFilter):
        self.filter = command_filter
        self.pending = deque()
        self.targets = {}
        self.initialized = False

    def add_state(self, stamp: int, position, velocity) -> None:
        self.pending.append((stamp, torch.tensor(position), torch.tensor(velocity)))

    def add_target(self, stamp: int, target) -> None:
        if not contract.is_policy_tick(stamp):
            raise ValueError(f"policy target stamp {stamp} ns is not a policy tick")
        self.targets[stamp] = torch.tensor(target)

    def drain(self) -> list[tuple[int, torch.Tensor]]:
        """Return (stamp, command) for every state that can be answered now."""
        ready = []
        while self.pending:
            stamp, position, velocity = self.pending[0]
            if not self.initialized:
                self.filter.reset(position)
                self.initialized = True
            if contract.is_policy_tick(stamp):
                if stamp not in self.targets:
                    break
                self.filter.set_target(self.targets.pop(stamp))
            self.pending.popleft()
            ready.append((stamp, self.filter.step(position, velocity)))
        return ready
