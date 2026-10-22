"""Deterministic exact-mode velocity commands for the V10 flat WASD demo."""

from __future__ import annotations

from typing import Sequence

import torch

try:  # Pure sampler tests run without IsaacLab on the laptop/container.
    from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
    from isaaclab.utils import configclass
except ModuleNotFoundError:  # pragma: no cover - real runtime supplies these classes.
    class UniformVelocityCommand:  # type: ignore[no-redef]
        """Import-only placeholder for the Isaac command superclass."""

    class UniformVelocityCommandCfg:  # type: ignore[no-redef]
        """Import-only placeholder for the Isaac command config."""

    def configclass(cls):  # type: ignore[no-redef]
        return cls


COMMAND_TABLE: dict[str, tuple[float, float, float]] = {
    "stop": (0.0, 0.0, 0.0),
    "forward": (0.0, 0.5, 0.0),
    "reverse": (0.0, -0.5, 0.0),
    "yaw_left": (0.0, 0.0, 0.4),
    "yaw_right": (0.0, 0.0, -0.4),
    "lateral_left": (-0.3, 0.0, 0.0),
    "lateral_right": (0.3, 0.0, 0.0),
}
MODE_COUNTS: dict[str, int] = {
    "stop": 50,
    "forward": 25,
    "reverse": 25,
    "yaw_left": 25,
    "yaw_right": 25,
    "lateral_left": 25,
    "lateral_right": 25,
}
CYCLE_SIZE = sum(MODE_COUNTS.values())


def _cycle(seed: int, cycle_index: int) -> tuple[torch.Tensor, tuple[str, ...]]:
    modes = tuple(
        name
        for name, count in MODE_COUNTS.items()
        for _ in range(count)
    )
    values = torch.tensor(
        [COMMAND_TABLE[name] for name in modes],
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(seed) + 1_000_003 * int(cycle_index)) % (2**63 - 1))
    order = torch.randperm(CYCLE_SIZE, generator=generator)
    return values[order], tuple(modes[index] for index in order.tolist())


def sample_wasd_batch(
    count: int,
    *,
    seed: int,
    cycle_offset: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return exact WASD commands from independently shuffled 200-slot cycles."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    if type(cycle_offset) is not int:
        raise ValueError("cycle_offset must be an integer")

    batches: list[torch.Tensor] = []
    sampled_modes: list[str] = []
    remaining = count
    cycle_index = cycle_offset
    while remaining:
        commands, modes = _cycle(seed, cycle_index)
        take = min(remaining, CYCLE_SIZE)
        batches.append(commands[:take])
        sampled_modes.extend(modes[:take])
        remaining -= take
        cycle_index += 1

    if not batches:
        return torch.empty((0, 3), dtype=torch.float32, device=device), ()
    return torch.cat(batches, dim=0).to(device=device), tuple(sampled_modes)


class WASDVelocityCommand(UniformVelocityCommand):
    """IsaacLab velocity-command term backed by exact categorical commands."""

    def __init__(self, cfg: "WASDVelocityCommandCfg", env):
        super().__init__(cfg, env)
        configured_seed = getattr(cfg, "seed", 0)
        environment_seed = getattr(getattr(env, "cfg", object()), "seed", None)
        self._seed = int(configured_seed if environment_seed is None else environment_seed)
        self._cycle_offset = int(getattr(cfg, "cycle_offset", 0))

    def _resample_command(self, env_ids: Sequence[int]):
        commands, modes = sample_wasd_batch(
            len(env_ids),
            seed=self._seed,
            cycle_offset=self._cycle_offset,
            device=str(self.device),
        )
        self.vel_command_b[env_ids] = commands
        self.is_standing_env[env_ids] = torch.tensor(
            [mode == "stop" for mode in modes],
            dtype=torch.bool,
            device=self.device,
        )
        self._last_modes = modes
        self._cycle_offset += (len(env_ids) + CYCLE_SIZE - 1) // CYCLE_SIZE


@configclass
class WASDVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`WASDVelocityCommand`."""

    class_type: type = WASDVelocityCommand
    seed: int = 0
    cycle_offset: int = 0
