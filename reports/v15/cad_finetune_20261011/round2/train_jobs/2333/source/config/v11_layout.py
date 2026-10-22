"""Simulator-free geometry and validation helpers for V11 terrain sensing."""

from __future__ import annotations

import torch


def scanner_layout(forward_axis: str) -> dict[str, object]:
    """Return the 187-ray footprint mounted ahead along base +y."""
    if forward_axis != "+y":
        raise ValueError(f"unsupported forward axis: {forward_axis}")
    return {
        "offset": (0.0, 0.8, 20.0),
        "size": (1.0, 1.6),
        "resolution": 0.1,
        "rows": 11,
        "columns": 17,
        "points": 187,
    }


def terrain_spawn_root_z(clearance_m: float, feet_below_root_m: float) -> float:
    """Place the feet above a terrain tile origin using the corrected sign."""
    return clearance_m + feet_below_root_m


def scan_quality(scan: torch.Tensor, clip: float) -> dict[str, float]:
    """Report finite and clipped fractions without hiding non-finite samples."""
    finite = torch.isfinite(scan)
    count = scan.numel()
    return {
        "finite_fraction": float(finite.sum() / count),
        "low_saturation_fraction": float(((scan <= -clip) & finite).sum() / count),
        "high_saturation_fraction": float(((scan >= clip) & finite).sum() / count),
    }
