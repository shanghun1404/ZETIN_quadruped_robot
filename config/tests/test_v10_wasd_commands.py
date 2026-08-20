"""Behavioral contracts for the deterministic V10 flat-WASD sampler."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from v10_wasd_commands import (
    COMMAND_TABLE,
    WASDVelocityCommand,
    sample_wasd_batch,
)


EXPECTED_COMMANDS = {
    "stop": (0.0, 0.0, 0.0),
    "forward": (0.0, 0.5, 0.0),
    "reverse": (0.0, -0.5, 0.0),
    "yaw_left": (0.0, 0.0, 0.4),
    "yaw_right": (0.0, 0.0, -0.4),
    "lateral_left": (-0.3, 0.0, 0.0),
    "lateral_right": (0.3, 0.0, 0.0),
}


def test_two_complete_cycles_have_exact_commands_and_pinned_proportions():
    commands, modes = sample_wasd_batch(400, seed=42)

    assert COMMAND_TABLE == EXPECTED_COMMANDS
    assert Counter(modes) == {
        "stop": 100,
        "forward": 50,
        "reverse": 50,
        "yaw_left": 50,
        "yaw_right": 50,
        "lateral_left": 50,
        "lateral_right": 50,
    }
    for expected in EXPECTED_COMMANDS.values():
        row = torch.tensor(expected, dtype=torch.float32)
        assert bool(torch.all(commands == row, dim=1).any())
    assert torch.unique(commands, dim=0).shape[0] == len(EXPECTED_COMMANDS)
    assert commands.shape == (400, 3)
    assert torch.isfinite(commands).all()


def test_seed_and_cycle_offset_control_order_without_changing_composition():
    first, first_modes = sample_wasd_batch(200, seed=17, cycle_offset=3)
    same, same_modes = sample_wasd_batch(200, seed=17, cycle_offset=3)
    next_cycle, next_modes = sample_wasd_batch(200, seed=17, cycle_offset=4)

    assert torch.equal(first, same)
    assert first_modes == same_modes
    assert Counter(first_modes) == Counter(next_modes)
    assert not torch.equal(first, next_cycle) or first_modes != next_modes


def test_partial_batch_is_the_exact_prefix_of_the_same_seeded_cycle():
    complete, complete_modes = sample_wasd_batch(200, seed=99)
    partial, partial_modes = sample_wasd_batch(37, seed=99)

    assert torch.equal(partial, complete[:37])
    assert partial_modes == complete_modes[:37]


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_sampler_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="non-negative integer"):
        sample_wasd_batch(count, seed=42)


def test_sampler_returns_a_typed_empty_batch():
    commands, modes = sample_wasd_batch(0, seed=42)

    assert commands.shape == (0, 3)
    assert commands.dtype == torch.float32
    assert modes == ()


def test_isaac_adapter_updates_commands_standing_mask_and_cycle_state():
    adapter = WASDVelocityCommand.__new__(WASDVelocityCommand)
    adapter.device = "cpu"
    adapter.vel_command_b = torch.full((200, 3), float("nan"))
    adapter.is_standing_env = torch.zeros(200, dtype=torch.bool)
    adapter._seed = 42
    adapter._cycle_offset = 5
    env_ids = torch.arange(200)

    adapter._resample_command(env_ids)

    assert torch.isfinite(adapter.vel_command_b).all()
    assert int(adapter.is_standing_env.sum()) == 50
    assert torch.equal(
        adapter.vel_command_b[adapter.is_standing_env],
        torch.zeros((50, 3)),
    )
    assert adapter._cycle_offset == 6
