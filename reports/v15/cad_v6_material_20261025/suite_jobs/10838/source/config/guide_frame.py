from __future__ import annotations

import torch


def _rotate_planar_180(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0 or values.shape[-1] < 2:
        raise ValueError("planar values must have at least two components")
    result = values.clone()
    result[..., :2] = -result[..., :2]
    return result


def guide_to_base_command(values: torch.Tensor) -> torch.Tensor:
    return _rotate_planar_180(values)


def base_to_guide_observation(values: torch.Tensor) -> torch.Tensor:
    return _rotate_planar_180(values)


def guide_forward_speed_range_to_base(
    speed_range: tuple[float, float],
) -> tuple[float, float]:
    low, high = speed_range
    if low < 0.0 or high < low:
        raise ValueError("guide speed range must be ordered and non-negative")
    return (-high, -low)


def rotate_height_scan_180(
    scan: torch.Tensor,
    rows: int = 17,
    cols: int = 11,
) -> torch.Tensor:
    expected = rows * cols
    if scan.ndim == 0 or scan.shape[-1] != expected:
        actual = None if scan.ndim == 0 else scan.shape[-1]
        raise ValueError(f"height scan width must be {expected}, got {actual}")
    grid = scan.reshape(*scan.shape[:-1], rows, cols)
    return torch.flip(grid, dims=(-2, -1)).reshape_as(scan)
