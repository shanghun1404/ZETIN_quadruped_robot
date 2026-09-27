"""Simulator-independent V15 tensors: policy-target delay line and recovery scoring."""

from __future__ import annotations

import math

import torch


# guide_robot_v2_cfg.SETTLED_ROOT_FRAME_Z_M: root frame height standing on flat ground.
STAND_ROOT_HEIGHT_M = 0.445
RECOVERED_TILT_RAD = math.radians(20.0)
RECOVERED_HEIGHT_RANGE_M = (0.35, 0.52)


class TargetDelayLine:
    """Per-environment delay of 0..max_steps physics steps on a [env, joint] target."""

    def __init__(self, initial: torch.Tensor, max_steps: int):
        if initial.ndim != 2 or not 0 <= max_steps <= 10:
            raise ValueError("expected [env, joint] targets and a delay of 0-10 steps")
        self.max_steps = max_steps
        # history[k] is the target pushed k steps ago.
        self.history = initial.unsqueeze(0).repeat(max_steps + 1, 1, 1)
        self.delay = torch.zeros(initial.shape[0], dtype=torch.long, device=initial.device)
        self.index = torch.arange(initial.shape[0], device=initial.device)

    def push(self, target: torch.Tensor) -> torch.Tensor:
        """Record this step's target and return each environment's delayed target."""
        self.history = torch.roll(self.history, 1, dims=0)
        self.history[0] = target
        return self.history[self.delay, self.index]

    def reset(self, ids: torch.Tensor, target: torch.Tensor, generator: torch.Generator | None = None) -> None:
        """Fill the reset environments' history with ``target`` and draw new delays."""
        self.history[:, ids] = target
        self.delay[ids] = torch.randint(0, self.max_steps + 1, (len(ids),), generator=generator,
                                        device=self.delay.device)


def upright_score(projected_gravity: torch.Tensor) -> torch.Tensor:
    """1 when the base is upright, 0 when upside down: (1 - g_z) / 2 of the base-frame gravity."""
    return ((1.0 - projected_gravity[..., 2]) / 2.0).clamp(0.0, 1.0)


def standing_gate(projected_gravity: torch.Tensor) -> torch.Tensor:
    """Ramp from 0 at ~53 deg tilt (score 0.8) to 1 when fully upright; gates standing rewards."""
    return ((upright_score(projected_gravity) - 0.8) / 0.2).clamp(0.0, 1.0)


def tilt_rad(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.acos((-projected_gravity[..., 2]).clamp(-1.0, 1.0))


def is_recovered(projected_gravity: torch.Tensor, root_height_m: torch.Tensor) -> torch.Tensor:
    low, high = RECOVERED_HEIGHT_RANGE_M
    return (tilt_rad(projected_gravity) < RECOVERED_TILT_RAD) & (root_height_m > low) & (root_height_m < high)


def sample_recovery_orientations(count: int, fallen_fraction: float, *, device=None,
                                 generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (fallen mask, wxyz quaternions): uniform SO(3) when fallen, else upright with random yaw."""
    if not 0.0 <= fallen_fraction <= 1.0:
        raise ValueError("fallen_fraction must be in [0, 1]")
    fallen = torch.rand(count, generator=generator, device=device) < fallen_fraction
    random = torch.randn(count, 4, generator=generator, device=device)
    random = random / torch.linalg.vector_norm(random, dim=-1, keepdim=True).clamp_min(1e-9)
    yaw = (torch.rand(count, generator=generator, device=device) * 2.0 - 1.0) * math.pi
    upright = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
    return fallen, torch.where(fallen[:, None], random, upright)


def non_physical_state(joint_vel: torch.Tensor, root_lin_vel: torch.Tensor, root_ang_vel: torch.Tensor,
                       *, max_joint_vel: float = 100.0, max_lin_vel: float = 10.0,
                       max_ang_vel: float = 50.0) -> torch.Tensor:
    """Environments whose physics diverged: non-finite or far beyond the robot's reachable speeds."""
    finite = torch.isfinite(joint_vel).all(-1) & torch.isfinite(root_lin_vel).all(-1) & torch.isfinite(root_ang_vel).all(-1)
    fast = ((joint_vel.abs() > max_joint_vel).any(-1) | (torch.linalg.vector_norm(root_lin_vel, dim=-1) > max_lin_vel)
            | (torch.linalg.vector_norm(root_ang_vel, dim=-1) > max_ang_vel))
    return ~finite | fast


def make_finite_rollout(base_cls):
    """Subclass an RSL-RL VecEnv wrapper to zero non-finite rewards/observations.

    One diverged robot otherwise poisons the whole PPO update (V15-R job 315 crashed at
    iteration ~2860). The ``physics_diverged`` termination resets that robot on the same step.
    """

    class FiniteRollout(base_cls):
        nonfinite_steps = 0

        def _clean(self, observations):
            for key, value in observations.items():
                if not bool(torch.isfinite(value).all()):
                    self.nonfinite_steps += 1
                    observations[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return observations

        def get_observations(self):
            return self._clean(super().get_observations())

        def step(self, actions):
            observations, rewards, dones, extras = super().step(actions)
            if not bool(torch.isfinite(rewards).all()):
                self.nonfinite_steps += 1
                rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            extras.setdefault("log", {})["Diagnostics/nonfinite_steps"] = float(self.nonfinite_steps)
            return self._clean(observations), rewards, dones, extras

    FiniteRollout.__name__ = f"Finite{base_cls.__name__}"
    return FiniteRollout
