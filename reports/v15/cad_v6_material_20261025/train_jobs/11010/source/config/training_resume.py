"""Helpers for exact PPO checkpoint continuation."""

from __future__ import annotations

from typing import Any


def validate_initial_terrain_level(level: int, num_rows: int) -> int:
    """Validate an exact zero-based terrain row used to resume curriculum."""
    if not 0 <= level < num_rows:
        raise ValueError(f"initial terrain level {level} is outside 0..{num_rows - 1}")
    return level


def load_training_checkpoint(
    runner: Any,
    checkpoint: str,
    map_location: str,
) -> Any:
    """Load policy and Adam state, including the adaptive PPO learning rate."""
    infos = runner.load(
        checkpoint,
        load_optimizer=True,
        map_location=map_location,
    )
    optimizer_groups = runner.alg.optimizer.param_groups
    if not optimizer_groups:
        raise ValueError("checkpoint optimizer has no parameter groups")
    learning_rates = {float(group["lr"]) for group in optimizer_groups}
    if len(learning_rates) != 1:
        raise ValueError(
            f"checkpoint optimizer has inconsistent learning rates: {learning_rates}"
        )
    runner.alg.learning_rate = learning_rates.pop()
    return infos
