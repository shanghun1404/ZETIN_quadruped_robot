"""Ramped push, motor-delay and sensor-noise randomization for V15 CAD fine-tuning.

All act on the unchanged CAD plant and the ROS2 WALK action path:

* push: a planar root-velocity change with a uniform random heading, applied as
  an interval event every 8-12 s;
* command delay: the per-episode policy-target delay of the CAD action, drawn
  from a wider range, plus an occasional missed policy update that keeps the
  previous target (Jetson plan arrival gap reached 53 ms against 20 ms);
* sensor noise: uniform additive noise on the 36 proprioceptive actor inputs
  after the frozen depth student composes the 235 inputs. Commands, last
  actions and the depth-derived scan are unchanged.

The ``motor`` profile adds the motor side:

* delay jitter: each physics step the command delay may grow by up to
  ``delay_jitter_steps`` above the episode delay, but an older target never
  replaces a newer one (commands arrive in order);
* feedback delay: joint position/velocity seen by the actor and by the WALK
  torque guards are 0-``feedback_delay_steps`` physics steps old, redrawn every
  policy step (200 Hz feedback period plus CAN reply time);
* motor response: the drive torque follows the PD torque through a first-order
  lag whose time constant is drawn per episode. The lagged torque is turned
  back into the implicit drive's position target, as the WALK guard already
  does for bounded torque.

One ``RobustSettings`` on the environment holds the current values. Training
ramps it from ``ROBUST_START`` to the profile's end settings; evaluation pins it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import torch

from v14_ros2_control import contract
from v13_torque_mdp import clip_position_target_to_motor_effort, low_pass_step
from v15_cad_training_actions import V15CADJointPositionAction, V15CADJointPositionActionCfg

try:
    from isaaclab.utils import configclass
except ImportError:  # CPU tests without IsaacLab
    configclass = dataclasses.dataclass


@dataclasses.dataclass(frozen=True)
class RobustSettings:
    push_speed_mps: tuple[float, float] = (0.0, 0.0)
    delay_steps: tuple[int, int] = (0, 3)
    hold_probability: float = 0.0
    noise_scale: float = 0.0
    delay_jitter_steps: int = 0
    feedback_delay_steps: tuple[int, int] = (0, 0)
    motor_lag_s: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        low, high = self.push_speed_mps
        if not 0.0 <= low <= high or not math.isfinite(high):
            raise ValueError("push speed range must be finite and 0 <= low <= high")
        for name in ("delay_steps", "feedback_delay_steps"):
            low, high = getattr(self, name)
            if not (isinstance(low, int) and isinstance(high, int) and 0 <= low <= high <= 10):
                raise ValueError(f"{name} must be integers with 0 <= low <= high <= 10")
        if not isinstance(self.delay_jitter_steps, int) or not 0 <= self.delay_jitter_steps <= 10:
            raise ValueError("delay jitter must be an integer in [0, 10]")
        if not 0.0 <= self.hold_probability < 1.0:
            raise ValueError("hold probability must be in [0, 1)")
        if not 0.0 <= self.noise_scale or not math.isfinite(self.noise_scale):
            raise ValueError("noise scale must be finite and nonnegative")
        low, high = self.motor_lag_s
        if not 0.0 <= low <= high <= 0.1:
            raise ValueError("motor lag must satisfy 0 <= low <= high <= 0.1 s")


# Start matches the adopted model's training (0-15 ms delay) with light push and noise.
ROBUST_START = RobustSettings(push_speed_mps=(0.0, 0.3), delay_steps=(0, 3), hold_probability=0.0, noise_scale=0.25)
# 30 ms covers the Jetson stress test's AI-plan p99 (11.2 ms) + command compute p99 (2.1 ms)
# + 250 kbit/s CAN last reply (4.7 ms) with margin; those runs used virtual inputs, motors off.
ROBUST_END = RobustSettings(push_speed_mps=(0.0, 1.0), delay_steps=(0, 6), hold_probability=0.05, noise_scale=1.0)
# Same 30 ms command-delay ceiling, split into a 0-20 ms episode delay and 0-10 ms per-step jitter.
# Feedback is up to one 5 ms feedback period plus the ~4.7 ms CAN reply; the 0-10 ms motor
# torque lag is an estimate (the AK manuals give no response time and none was measured).
MOTOR_END = dataclasses.replace(ROBUST_END, delay_steps=(0, 4), delay_jitter_steps=2,
                                feedback_delay_steps=(0, 2), motor_lag_s=(0.0, 0.010))
PROFILES = {"v1": ROBUST_END, "motor": MOTOR_END}
PUSH_INTERVAL_S = (8.0, 12.0)

# Full-scale uniform half-widths for the first 48 actor inputs:
# base lin vel, base ang vel, projected gravity, command, joint pos, joint vel, last action.
PROPRIOCEPTIVE_NOISE = (
    (0.15,) * 3 + (0.2,) * 3 + (0.05,) * 3 + (0.0,) * 3
    + (0.02,) * 12 + (1.0,) * 12 + (0.0,) * 12
)


def ramp_level(step: int, ramp_steps: int) -> float:
    """Linear 0 -> 1 over ``ramp_steps`` policy steps; 1 when the ramp is empty."""
    if ramp_steps <= 0:
        return 1.0
    return min(max(step / ramp_steps, 0.0), 1.0)


def interpolate_settings(level: float, start: RobustSettings = ROBUST_START,
                         end: RobustSettings = ROBUST_END) -> RobustSettings:
    if not 0.0 <= level <= 1.0:
        raise ValueError("level must be in [0, 1]")

    def mix(a, b):
        return a + (b - a) * level

    def mix_steps(a, b):
        return int(round(mix(a, b)))

    return RobustSettings(
        push_speed_mps=tuple(mix(a, b) for a, b in zip(start.push_speed_mps, end.push_speed_mps)),
        delay_steps=tuple(mix_steps(a, b) for a, b in zip(start.delay_steps, end.delay_steps)),
        hold_probability=mix(start.hold_probability, end.hold_probability),
        noise_scale=mix(start.noise_scale, end.noise_scale),
        delay_jitter_steps=mix_steps(start.delay_jitter_steps, end.delay_jitter_steps),
        feedback_delay_steps=tuple(mix_steps(a, b) for a, b in zip(start.feedback_delay_steps, end.feedback_delay_steps)),
        motor_lag_s=tuple(mix(a, b) for a, b in zip(start.motor_lag_s, end.motor_lag_s)),
    )


def robust_settings(env) -> RobustSettings:
    settings = getattr(env, "v15_robust_settings", None)
    if not isinstance(settings, RobustSettings):
        raise RuntimeError("robust randomization requires env.v15_robust_settings")
    return settings


def corrupt_proprioception(policy_obs: torch.Tensor, scale: float,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """Add uniform noise to the proprioceptive part of [batch, 235] actor inputs."""
    if policy_obs.ndim != 2 or policy_obs.shape[1] != 235:
        raise ValueError("expected [batch, 235] actor observations")
    if scale == 0.0:
        return policy_obs
    amplitude = torch.tensor(PROPRIOCEPTIVE_NOISE, device=policy_obs.device, dtype=policy_obs.dtype) * scale
    unit = torch.rand((policy_obs.shape[0], 48), generator=generator, device=policy_obs.device,
                      dtype=policy_obs.dtype)
    return torch.cat((policy_obs[:, :48] + (2.0 * unit - 1.0) * amplitude, policy_obs[:, 48:]), dim=1)


def planar_push_velocity(count: int, speed_range: tuple[float, float], *, device=None,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """[count, 2] world-frame velocity changes with uniform heading and speed."""
    low, high = speed_range
    speed = low + (high - low) * torch.rand(count, generator=generator, device=device)
    heading = 2.0 * math.pi * torch.rand(count, generator=generator, device=device)
    return torch.stack((speed * torch.cos(heading), speed * torch.sin(heading)), dim=-1)


def push_planar(env, env_ids: torch.Tensor) -> None:
    """Interval event: add a random planar velocity change to the robot base."""
    speed_range = robust_settings(env).push_speed_mps
    if speed_range[1] <= 0.0 or len(env_ids) == 0:
        return
    asset = env.scene["robot"]
    velocity = asset.data.root_vel_w[env_ids].clone()
    velocity[:, :2] += planar_push_velocity(len(env_ids), speed_range, device=velocity.device).to(velocity.dtype)
    asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def robust_ramp(env, env_ids, ramp_steps: int, offset_steps: int = 0, profile: str = "v1") -> float:
    """Curriculum term: update the environment's settings from the global step counter.

    ``offset_steps`` continues a ramp across resumed training segments.
    """
    level = ramp_level(env.common_step_counter + offset_steps, ramp_steps)
    env.v15_robust_settings = interpolate_settings(level, ROBUST_START, PROFILES[profile])
    return level


def next_command_delay(previous: torch.Tensor, episode: torch.Tensor, jitter_steps: int, cap: int) -> torch.Tensor:
    """Per-step delay in ``[episode, episode + jitter]`` that never revisits an older target.

    Growing by more than one step per physics step would apply an older command after a
    newer one, which an in-order link cannot do.
    """
    wanted = (episode + torch.randint(0, jitter_steps + 1, episode.shape, device=episode.device)).clamp(max=cap)
    return torch.minimum(wanted, previous + 1)


def lag_alpha(time_constant_s: torch.Tensor, dt_s: float) -> torch.Tensor:
    """First-order lag gain per physics step; 1 (no lag) for a zero time constant."""
    safe = time_constant_s.clamp_min(1e-9)
    return torch.where(time_constant_s > 0, 1.0 - torch.exp(-dt_s / safe), torch.ones_like(time_constant_s))


class V15CADRobustJointPositionAction(V15CADJointPositionAction):
    """CAD WALK action whose delays, missed updates and motor response follow the settings.

    With zero jitter, feedback delay and motor lag, every command equals the parent's.
    """

    cfg: "V15CADRobustJointPositionActionCfg"

    def __init__(self, cfg: "V15CADRobustJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        if not isinstance(cfg.max_feedback_delay_steps, int) or not 0 <= cfg.max_feedback_delay_steps <= 10:
            raise ValueError("max_feedback_delay_steps must be an integer in [0, 10]")
        size = cfg.max_feedback_delay_steps + 1
        # history[k] is the measured state k physics steps before the current one.
        self._position_history = self.joint_pos_policy.unsqueeze(0).repeat(size, 1, 1)
        self._velocity_history = self.joint_vel_policy.unsqueeze(0).repeat(size, 1, 1)
        self._feedback_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_delay = self._delay_line.delay.clone()
        self._motor_alpha = torch.ones(self.num_envs, 1, device=self.device, dtype=self._raw_targets.dtype)
        self._lagged_torque_physical = torch.zeros_like(self._raw_targets)

    def observed_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy-frame (position, velocity) as the actor sees them after this policy step."""
        # The post-step state is not in the history yet, so history[k-1] is k steps old.
        position = torch.cat((self.joint_pos_policy.unsqueeze(0), self._position_history[:-1]))
        velocity = torch.cat((self.joint_vel_policy.unsqueeze(0), self._velocity_history[:-1]))
        index = self._delay_line.index
        return position[self._feedback_delay, index], velocity[self._feedback_delay, index]

    def process_actions(self, actions: torch.Tensor) -> None:
        previous = self._raw_targets.clone()
        super().process_actions(actions)
        settings = robust_settings(self._env)
        if settings.hold_probability > 0.0:
            # The 200 Hz command loop keeps the last plan when a new one is late.
            hold = torch.rand(self.num_envs, device=self.device) < settings.hold_probability
            self._raw_targets[hold] = previous[hold]
        low, high = settings.feedback_delay_steps
        if high > self.cfg.max_feedback_delay_steps:
            raise ValueError("feedback delay settings exceed the action's history")
        if high > 0:
            self._feedback_delay = torch.randint(low, high + 1, (self.num_envs,), device=self.device)
        else:
            self._feedback_delay.zero_()

    def apply_actions(self) -> None:
        settings = robust_settings(self._env)
        if settings.delay_jitter_steps > 0:
            self._delay_line.delay = next_command_delay(
                self._delay_line.delay, self._episode_delay, settings.delay_jitter_steps, self.cfg.max_delay_steps)
        self._position_history = torch.roll(self._position_history, 1, dims=0)
        self._velocity_history = torch.roll(self._velocity_history, 1, dims=0)
        self._position_history[0] = self.joint_pos_policy
        self._velocity_history[0] = self.joint_vel_policy
        index = self._delay_line.index
        # The Jetson-side filter and guards only know the last feedback that arrived.
        position = self._position_history[self._feedback_delay, index]
        velocity = self._velocity_history[self._feedback_delay, index]

        # Parent WALK path (delay -> LPF -> two guards) with delayed feedback.
        delayed = self._delay_line.push(self._raw_targets)
        self._filtered_targets.copy_(low_pass_step(delayed, self._filtered_targets, self._alpha))
        target = self._filtered_targets
        if self.cfg.effort_guard_fraction is not None:
            target = clip_position_target_to_motor_effort(
                target, position, velocity, self._effort_limits,
                stiffness=contract.PD_STIFFNESS, damping=contract.PD_DAMPING,
                velocity_limit=contract.MOTOR_VELOCITY_LIMIT, fraction=self.cfg.effort_guard_fraction,
            )
        if self.cfg.walk_guard_fraction is not None:
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

        # The motor's own PD uses its fresh encoder; its torque output lags the PD torque.
        data = self._asset.data
        physical_position = data.joint_pos[:, self._joint_ids]
        physical_velocity = data.joint_vel[:, self._joint_ids]
        stiffness = data.joint_stiffness[:, self._joint_ids]
        damping = data.joint_damping[:, self._joint_ids]
        caps = data.joint_effort_limits[:, self._joint_ids]
        requested_physical = stiffness * (physical_target - physical_position) - damping * physical_velocity
        clipped_physical = torch.minimum(torch.maximum(requested_physical, -caps), caps)
        self._lagged_torque_physical.add_(self._motor_alpha * (clipped_physical - self._lagged_torque_physical))
        lagged_target = physical_position + (self._lagged_torque_physical + damping * physical_velocity) / stiffness
        no_lag = self._motor_alpha == 1.0
        self._asset.set_joint_position_target(torch.where(no_lag, physical_target, lagged_target),
                                              joint_ids=self._joint_ids)
        self._estimated_torque_policy.copy_(torch.where(no_lag, clipped_physical, self._lagged_torque_physical)
                                            * self._sign)
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
        settings = robust_settings(self._env)
        low, high = settings.delay_steps
        if high > self.cfg.max_delay_steps:
            raise ValueError("delay settings exceed the action's delay buffer")
        self._delay_line.delay[ids] = torch.randint(low, high + 1, (len(ids),), device=self._delay_line.delay.device)
        self._episode_delay[ids] = self._delay_line.delay[ids]
        self._position_history[:, ids] = self.joint_pos_policy[ids]
        self._velocity_history[:, ids] = self.joint_vel_policy[ids]
        self._feedback_delay[ids] = 0
        self._lagged_torque_physical[ids] = 0
        low, high = settings.motor_lag_s
        if high > 0.0:
            time_constant = low + (high - low) * torch.rand(len(ids), 1, device=self.device, dtype=self._motor_alpha.dtype)
            self._motor_alpha[ids] = lag_alpha(time_constant, self._env.cfg.sim.dt)
        else:
            self._motor_alpha[ids] = 1.0


@configclass
class V15CADRobustJointPositionActionCfg(V15CADJointPositionActionCfg):
    class_type: type = V15CADRobustJointPositionAction
    max_delay_steps: int = ROBUST_END.delay_steps[1]
    max_feedback_delay_steps: int = MOTOR_END.feedback_delay_steps[1]


def observed_joint_pos_rel(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Actor joint positions relative to policy defaults, as old as the drawn feedback delay."""
    term = env.action_manager.get_term(action_name)
    return term.observed_joint_state()[0] - term._default_policy


def observed_joint_vel(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Actor joint velocities, as old as the drawn feedback delay."""
    return env.action_manager.get_term(action_name).observed_joint_state()[1]


def enable_robust_finetune(cfg, *, ramp_steps: int, offset_steps: int = 0, profile: str = "v1") -> None:
    """Swap in the robust action, observations, push event and ramp; noise is added by the wrapper."""
    from isaaclab.managers import CurriculumTermCfg, EventTermCfg

    if ramp_steps < 0 or offset_steps < 0:
        raise ValueError("ramp_steps and offset_steps must be nonnegative")
    if profile not in PROFILES:
        raise ValueError(f"unknown robust profile {profile}")
    action = cfg.actions.joint_pos
    cfg.actions.joint_pos = V15CADRobustJointPositionActionCfg(
        asset_name=action.asset_name, scale=action.scale, cutoff_hz=action.cutoff_hz,
        effort_guard_fraction=action.effort_guard_fraction, walk_guard_fraction=action.walk_guard_fraction,
        rms_time_constant_s=action.rms_time_constant_s,
    )
    cfg.observations.policy.joint_pos.func = observed_joint_pos_rel
    cfg.observations.policy.joint_vel.func = observed_joint_vel
    cfg.events.push_robot = EventTermCfg(func=push_planar, mode="interval", interval_range_s=PUSH_INTERVAL_S)
    if cfg.curriculum is None:
        @configclass
        class RobustCurriculumCfg:
            pass

        cfg.curriculum = RobustCurriculumCfg()
    cfg.curriculum.robust_ramp = CurriculumTermCfg(
        func=robust_ramp, params={"ramp_steps": ramp_steps, "offset_steps": offset_steps, "profile": profile})


def make_noisy_proprioception(base_cls):
    """Subclass the V13 depth wrapper to corrupt its composed actor inputs."""

    class NoisyProprioception(base_cls):
        def _with_depth(self, observations):
            observations = super()._with_depth(observations)
            scale = robust_settings(self.unwrapped).noise_scale
            with torch.inference_mode():
                observations["policy"] = corrupt_proprioception(observations["policy"], scale)
            return observations

    NoisyProprioception.__name__ = f"Noisy{base_cls.__name__}"
    return NoisyProprioception


def settings_record(settings: RobustSettings) -> dict:
    return dataclasses.asdict(settings)
