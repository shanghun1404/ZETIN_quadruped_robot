"""Pure command-line helpers shared by simulator evaluation entry points."""

from __future__ import annotations

import argparse
from typing import Any


def add_seed_argument(parser: argparse.ArgumentParser) -> None:
    """Add the reproducible evaluator seed without importing IsaacLab."""
    parser.add_argument("--seed", type=int, default=42)


def add_fixed_terrain_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add an optional deterministic generated-terrain difficulty row."""
    parser.add_argument("--terrain-level", type=int, default=None)


def validate_fixed_terrain_level(level: int, num_rows: int) -> int:
    """Return a valid zero-based terrain row or fail with its exact range."""
    if not 0 <= level < num_rows:
        raise ValueError(f"terrain level {level} is outside 0..{num_rows - 1}")
    return level


def evaluation_metadata(checkpoint: str, seed: int) -> dict[str, object]:
    """Return the identity fields required by strict multi-seed aggregation."""
    return {"checkpoint": checkpoint, "seed": seed}


def load_inference_checkpoint(runner: Any, checkpoint: str) -> None:
    """Load model weights only; evaluators never need optimizer state."""
    runner.load(checkpoint, load_optimizer=False)
