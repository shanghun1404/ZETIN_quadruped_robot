"""Balanced native-CAD commands and rewards for the frozen V12 student.

The CAD base frame is +X right, +Y forward. Numeric helpers run on CPU without
starting Isaac Sim; the command term uses IsaacLab after AppLauncher starts.
Torque costs use the action's clipped implicit-PD plus feedforward estimate,
which is also the estimate checked by the unchanged CAD acceptance suite.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch


COMMAND_CATEGORIES = ("zero", "forward", "reverse", "lateral", "yaw", "combined")
COMMAND_PROBABILITIES = (.20, .30, .15, .15, .10, .10)
CAD_BODY_NAMES = ("base",) + tuple(
    f"{leg}_{part}" for leg in ("FL", "FR", "BL", "BR") for part in ("hip", "thigh", "shank")
)
PAIR_SENSOR_NAMES = tuple(f"pair_{i}" for i in range(len(CAD_BODY_NAMES)))


def commands_for_categories(
    categories: torch.Tensor,
    random_values: torch.Tensor,
    *,
    forward_range: tuple[float, float] = (.25, .50),
    reverse_range: tuple[float, float] = (-.40, -.20),
    lateral_magnitude_range: tuple[float, float] = (.15, .30),
    yaw_magnitude_range: tuple[float, float] = (.15, .40),
    combined_forward_magnitude_range: tuple[float, float] = (.20, .50),
    force_forward_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert six named motion categories to [right, forward, yaw] targets."""
    if categories.ndim != 1 or random_values.shape != (len(categories), 6):
        raise ValueError("categories and random values must be [env] and [env, 6]")
    if bool(((categories < 0) | (categories >= 6)).any()):
        raise ValueError("unknown CAD command category")
    if bool(((random_values < 0.) | (random_values > 1.)).any()):
        raise ValueError("command random values must lie in [0, 1]")
    ranges = (forward_range, reverse_range, lateral_magnitude_range, yaw_magnitude_range,
              combined_forward_magnitude_range)
    if any(len(r) != 2 or not all(math.isfinite(v) for v in r) or r[0] > r[1] for r in ranges):
        raise ValueError("command ranges must be finite ordered pairs")
    if forward_range[0] <= 0. or reverse_range[1] >= 0. or any(r[0] <= 0. for r in ranges[2:]):
        raise ValueError("forward/magnitudes must be positive and reverse must be negative")
    if force_forward_mask is not None and force_forward_mask.shape != categories.shape:
        raise ValueError("forward mask must have one value per environment")

    def uniform(column, interval):
        return interval[0] + random_values[:, column] * (interval[1] - interval[0])

    def sign(column):
        return torch.where(random_values[:, column] < .5, -1., 1.)

    out = random_values.new_zeros((len(categories), 3))
    forward = categories == 1
    reverse = categories == 2
    lateral = categories == 3
    yaw = categories == 4
    combined = categories == 5
    out[:, 0] = torch.where(lateral | combined, sign(1) * uniform(0, lateral_magnitude_range), 0.)
    out[:, 1] = torch.where(forward, uniform(2, forward_range), 0.)
    out[:, 1] = torch.where(reverse, uniform(2, reverse_range), out[:, 1])
    out[:, 1] = torch.where(combined, sign(3) * uniform(2, combined_forward_magnitude_range), out[:, 1])
    out[:, 2] = torch.where(yaw | combined, sign(5) * uniform(4, yaw_magnitude_range), 0.)
    if force_forward_mask is not None:
        out[force_forward_mask] = 0.
        out[force_forward_mask, 1] = uniform(2, forward_range)[force_forward_mask]
    return out


def sample_cad_velocity_commands(
    num_envs: int,
    *,
    device,
    generator: torch.Generator | None = None,
    probabilities: Sequence[float] = COMMAND_PROBABILITIES,
    **ranges,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw categories independently for each segment; lateral/yaw signs balance."""
    if len(probabilities) != 6 or any(not math.isfinite(p) or p < 0. for p in probabilities) \
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-6):
        raise ValueError("six command probabilities must be nonnegative and sum to one")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    categories = torch.multinomial(torch.tensor(probabilities, device=device), num_envs,
                                   replacement=True, generator=generator)
    draws = torch.rand(num_envs, 6, device=device, generator=generator)
    return commands_for_categories(categories, draws, **ranges), categories


def terrain_forward_mask(terrain_types: torch.Tensor, *, num_cols: int, fraction_start: float | None):
    """Select stair columns after the flat terrain fraction, independent of rows."""
    if fraction_start is None:
        return torch.zeros_like(terrain_types, dtype=torch.bool)
    if num_cols < 1 or not math.isfinite(fraction_start) or not 0. <= fraction_start < 1.:
        raise ValueError("terrain column count and forward fraction must be valid")
    return terrain_types >= num_cols * fraction_start


def _positive_std(std: float) -> None:
    if not math.isfinite(std) or std <= 0.:
        raise ValueError("standard deviation must be finite and positive")


def planar_velocity_reward(command: torch.Tensor, velocity: torch.Tensor, *, std: float = .25):
    """Symmetric tracking of both native CAD planar axes, including zero targets."""
    _positive_std(std)
    return torch.exp(-torch.square(command[:, :2] - velocity[:, :2]).sum(-1) / std ** 2)


def yaw_velocity_reward(command: torch.Tensor, yaw_velocity: torch.Tensor, *, std: float = .25):
    _positive_std(std)
    return torch.exp(-torch.square(command[:, 2] - yaw_velocity) / std ** 2)


def directed_motion_progress(command: torch.Tensor, velocity: torch.Tensor, yaw_velocity: torch.Tensor):
    """Mean signed target fraction over commanded axes, bounded to [-1, 1].

    Each active native [right, forward, yaw] component receives one at its
    requested speed, zero at rest and minus one for opposite target speed.
    Clipping prevents overspeed from earning additional progress. Inactive
    axes receive no progress reward; the tracking terms still constrain them.
    """
    if command.ndim != 2 or command.shape[1] != 3 or velocity.ndim != 2 \
            or velocity.shape[0] != command.shape[0] or velocity.shape[1] < 2 \
            or yaw_velocity.shape != (command.shape[0],):
        raise ValueError("progress needs [env, 3] commands, planar velocities and [env] yaw rates")
    measured = torch.cat((velocity[:, :2], yaw_velocity.unsqueeze(-1)), dim=-1)
    active = command.abs() > 1e-6
    denominator = torch.where(active, command, torch.ones_like(command))
    components = (measured / denominator).clamp(-1., 1.) * active
    return components.sum(-1) / active.sum(-1).clamp_min(1)


def zero_command_motion_cost(
    command: torch.Tensor, velocity: torch.Tensor, yaw_velocity: torch.Tensor, *,
    planar_std: float = .25, yaw_std: float = .25, command_tolerance: float = 1e-6,
):
    """Explicit drift cost only for a full zero command; pure yaw is a motion task."""
    _positive_std(planar_std)
    _positive_std(yaw_std)
    standing = command.abs().amax(-1) <= command_tolerance
    return standing * (velocity[:, :2].square().sum(-1) / planar_std ** 2
                       + yaw_velocity.square() / yaw_std ** 2)


def _rated_ratio(torque: torch.Tensor, rated: torch.Tensor):
    if torque.ndim != 2 or rated.shape != (torque.shape[1],):
        raise ValueError("torque and rated limits must be [env, joint] and [joint]")
    if not bool(torch.isfinite(rated).all()) or bool((rated <= 0.).any()):
        raise ValueError("rated torque limits must be finite and positive")
    return torque / rated


def rated_load_squared(torque: torch.Tensor, rated: torch.Tensor):
    """Mean squared rated fraction; minimizing its time average reduces RMS load."""
    return _rated_ratio(torque, rated).square().mean(-1)


def excess_rated_load_squared(torque: torch.Tensor, rated: torch.Tensor):
    return (_rated_ratio(torque, rated).abs() - 1.).clamp_min(0.).square().mean(-1)


def policy_rated_torque_limits(joint_names: Sequence[str], *, device="cpu", dtype=torch.float32):
    rated = []
    for name in joint_names:
        if name.endswith("_HAA"):
            rated.append(8.3)
        elif name.endswith(("_HFE", "_KFE")):
            rated.append(18.)
        else:
            raise ValueError(f"unknown policy joint rated torque: {name}")
    return torch.tensor(rated, device=device, dtype=dtype)


def cad_self_contact_filters():
    """Same 13 sensor/filter order as evaluate_v15_cad_suite: 66 unique pairs."""
    adjacent = {frozenset(("base", f"{leg}_hip")) for leg in ("FL", "FR", "BL", "BR")}
    adjacent.update(frozenset((f"{leg}_{a}", f"{leg}_{b}"))
                    for leg in ("FL", "FR", "BL", "BR")
                    for a, b in (("hip", "thigh"), ("thigh", "shank")))
    return tuple(dict(body=body, targets=[other for other in CAD_BODY_NAMES
                 if other != body and frozenset((body, other)) not in adjacent]) for body in CAD_BODY_NAMES)


@lru_cache(maxsize=8)
def _unique_pair_columns(pairs: tuple[tuple[str, str], ...]):
    unique = {}
    indices = []
    for body, target in pairs:
        if body == target:
            raise ValueError("self-contact filter cannot target its own rigid body")
        pair = tuple(sorted((body, target)))
        if pair not in unique:
            unique[pair] = len(unique)
        indices.append(unique[pair])
    return tuple(indices), len(unique)


def unique_self_contact_force_norms(directed_force: torch.Tensor, pairs: Sequence[tuple[str, str]]):
    """Fold reciprocal filtered forces by maximum norm so each pair counts once."""
    if directed_force.ndim != 3 or directed_force.shape[1:] != (len(pairs), 3):
        raise ValueError("filtered pair forces must be [env, directed pair, 3]")
    columns, count = _unique_pair_columns(tuple(pairs))
    index = torch.tensor(columns, device=directed_force.device).expand(directed_force.shape[0], -1)
    norms = torch.linalg.vector_norm(directed_force, dim=-1)
    return norms.new_zeros((norms.shape[0], count)).scatter_reduce_(1, index, norms, reduce="amax")


def _scene_self_contact_norms(env, sensor_names=PAIR_SENSOR_NAMES):
    filters = cad_self_contact_filters()
    if len(sensor_names) != len(filters):
        raise ValueError("all 13 CAD filtered pair sensors are required")
    force, pairs = [], []
    for name, row in zip(sensor_names, filters):
        matrix = env.scene[name].data.force_matrix_w
        if matrix is None or matrix.ndim != 4 or matrix.shape[1:] != (1, len(row["targets"]), 3):
            raise ValueError(f"{name} needs the reviewed filtered CAD force matrix")
        force.append(matrix[:, 0])
        pairs.extend((row["body"], target) for target in row["targets"])
    return unique_self_contact_force_norms(torch.cat(force, dim=1), pairs)


def track_planar_velocity_exp(env, command_name="base_velocity", std=.25, asset_name="robot"):
    return planar_velocity_reward(env.command_manager.get_command(command_name),
                                  env.scene[asset_name].data.root_lin_vel_b, std=std)


def track_yaw_velocity_exp(env, command_name="base_velocity", std=.25, asset_name="robot"):
    return yaw_velocity_reward(env.command_manager.get_command(command_name),
                               env.scene[asset_name].data.root_ang_vel_b[:, 2], std=std)


def directed_motion_progress_reward(env, command_name="base_velocity", asset_name="robot"):
    data = env.scene[asset_name].data
    return directed_motion_progress(env.command_manager.get_command(command_name),
                                    data.root_lin_vel_b, data.root_ang_vel_b[:, 2])


def zero_command_velocity_l2(env, command_name="base_velocity", asset_name="robot", planar_std=.25, yaw_std=.25):
    data = env.scene[asset_name].data
    return zero_command_motion_cost(env.command_manager.get_command(command_name), data.root_lin_vel_b,
                                    data.root_ang_vel_b[:, 2], planar_std=planar_std, yaw_std=yaw_std)


def _action_torque_and_rated(env, action_name):
    from v14_ros2_control import contract
    torque = env.action_manager.get_term(action_name).estimated_torque_policy
    return torque, policy_rated_torque_limits(contract.JOINT_NAMES, device=torque.device, dtype=torque.dtype)


def rated_torque_l2(env, action_name="joint_pos"):
    return rated_load_squared(*_action_torque_and_rated(env, action_name))


def excess_rated_torque_l2(env, action_name="joint_pos"):
    return excess_rated_load_squared(*_action_torque_and_rated(env, action_name))


def self_contact_count(env, sensor_names=PAIR_SENSOR_NAMES, threshold=1.):
    return (_scene_self_contact_norms(env, sensor_names) > threshold).to(torch.float32).sum(-1)


def self_contact_force_l2(env, sensor_names=PAIR_SENSOR_NAMES, threshold=1., reference_force=20.):
    if not math.isfinite(reference_force) or reference_force <= 0.:
        raise ValueError("reference contact force must be positive")
    return ((_scene_self_contact_norms(env, sensor_names) - threshold).clamp_min(0.) / reference_force).square().sum(-1)


# Importing IsaacLab before AppLauncher can fail on omni.*. Keep the numeric
# objective importable for CPU tests; constructing the command still fails loudly.
try:
    from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
    from isaaclab.utils import configclass
    _ISAAC_IMPORT_ERROR = None
except ImportError as error:
    UniformVelocityCommand = object
    UniformVelocityCommandCfg = object
    _ISAAC_IMPORT_ERROR = error

    def configclass(cls):
        return cls


class CadVelocityCommand(UniformVelocityCommand):
    """Categorical velocity segments with no heading/yaw mutation between draws."""

    def __init__(self, cfg, env):
        if _ISAAC_IMPORT_ERROR is not None:
            raise RuntimeError("Start Isaac AppLauncher before importing CAD command configuration") from _ISAAC_IMPORT_ERROR
        if cfg.heading_command or cfg.rel_heading_envs != 0.:
            raise ValueError("CAD command segments must preserve sampled yaw, with heading disabled")
        super().__init__(cfg, env)
        self.motion_category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids]
        if ids.numel() == 0:
            return
        force_forward = None
        fraction = self.cfg.force_forward_terrain_fraction_start
        if fraction is not None:
            terrain = self._env.scene.terrain
            generator = terrain.cfg.terrain_generator
            if generator is None or terrain.terrain_types is None:
                raise ValueError("forward terrain restriction requires generated terrain columns")
            force_forward = terrain_forward_mask(terrain.terrain_types[ids], num_cols=generator.num_cols,
                                                 fraction_start=fraction)
        command, category = sample_cad_velocity_commands(
            ids.numel(), device=self.device, probabilities=self.cfg.command_probabilities,
            forward_range=self.cfg.forward_range, reverse_range=self.cfg.reverse_range,
            lateral_magnitude_range=self.cfg.lateral_magnitude_range,
            yaw_magnitude_range=self.cfg.yaw_magnitude_range,
            combined_forward_magnitude_range=self.cfg.combined_forward_magnitude_range,
            force_forward_mask=force_forward,
        )
        if force_forward is not None:
            category[force_forward] = 1
        self.vel_command_b[ids] = command
        self.motion_category[ids] = category
        self.is_standing_env[ids] = category == 0
        self.is_heading_env[ids] = False

    def _update_command(self):
        # Zero, planar and yaw commands are complete targets fixed by resampling.
        pass


@configclass
class CadVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CadVelocityCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (2., 5.)
    heading_command: bool = False
    rel_heading_envs: float = 0.
    rel_standing_envs: float = .20
    debug_vis: bool = False
    command_probabilities: tuple[float, ...] = COMMAND_PROBABILITIES
    forward_range: tuple[float, float] = (.25, .50)
    reverse_range: tuple[float, float] = (-.40, -.20)
    lateral_magnitude_range: tuple[float, float] = (.15, .30)
    yaw_magnitude_range: tuple[float, float] = (.15, .40)
    combined_forward_magnitude_range: tuple[float, float] = (.20, .50)
    force_forward_terrain_fraction_start: float | None = None
    if _ISAAC_IMPORT_ERROR is None:
        ranges = UniformVelocityCommandCfg.Ranges(lin_vel_x=(-.30, .30), lin_vel_y=(-.40, .50), ang_vel_z=(-.40, .40))
