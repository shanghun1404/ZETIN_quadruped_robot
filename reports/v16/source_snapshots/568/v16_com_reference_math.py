"""Simulator-free NumPy/Torch minimum-jerk references for world XY COM goals.

The straight waypoint segment uses h(s)=10s^3-15s^4+6s^5. Its peak
normalized speed is 15/8 and peak acceleration is 10/sqrt(3), giving
Euclidean speed/acceleration limits independent of segment direction.
These are reference limits; tracking, contact feasibility and torque limits
remain the caller's responsibility. New segments start and finish at rest.
"""
from __future__ import annotations

import math
import numpy as np


def _array(value, like=None):
    if hasattr(like if like is not None else value, "new_tensor"):
        import torch
        anchor = value if like is None else like
        if anchor.dtype not in (torch.float32, torch.float64):
            raise ValueError("Torch waypoints require float32 or float64")
        return torch.as_tensor(value, dtype=anchor.dtype, device=anchor.device)
    result = np.asarray(value, dtype=None if like is None else like.dtype)
    if result.dtype.kind in "iu":
        result = result.astype(np.float64)
    if result.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Waypoints require real float32 or float64 values")
    return result


def _backend(value):
    if hasattr(value, "new_tensor"):
        import torch
        return torch
    return np


def _validate(value, name, *, positive=False, nonnegative=False):
    xp = _backend(value)
    valid = xp.isfinite(value)
    if positive:
        valid = valid & (value > 0)
    if nonnegative:
        valid = valid & (value >= 0)
    if not bool(valid.all()):
        qualifier = "finite positive" if positive else "finite nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must contain {qualifier} values")


def _waypoints(start_xy, end_xy):
    start = _array(start_xy)
    end = _array(end_xy, start)
    if start.shape[-1:] != (2,) or end.shape[-1:] != (2,):
        raise ValueError("Waypoints must have shape (...,2)")
    _validate(start, "start_xy")
    _validate(end, "end_xy")
    return start, end


def minimum_jerk_duration(start_xy, end_xy, *, vmax=0.20, amax=0.6, min_duration=0.2):
    """Return T=max(1.875*distance/vmax, sqrt(10*distance/(sqrt(3)*amax)), floor).

    Waypoints (...,2) and positive bounds broadcast over batch dimensions.
    Outputs retain the start waypoint's NumPy/Torch backend, float dtype and
    Torch device, with shape (...). Units are metres and seconds. A zero
    distance still gets ``min_duration``. Inputs are never modified.
    """
    start, end = _waypoints(start_xy, end_xy)
    xp = _backend(start)
    speed, acceleration, floor = (_array(value, start) for value in (vmax, amax, min_duration))
    for value, name in ((speed, "vmax"), (acceleration, "amax"), (floor, "min_duration")):
        _validate(value, name, positive=True)
    delta = end - start
    squared_distance = (delta * delta).sum(dim=-1) if hasattr(start, "new_tensor") else (delta * delta).sum(axis=-1)
    distance = xp.sqrt(squared_distance)
    duration = xp.maximum(xp.maximum(1.875 * distance / speed,
                                    xp.sqrt((10 / math.sqrt(3)) * distance / acceleration)), floor)
    _validate(duration, "computed duration", positive=True)
    return _array(duration, start)


def minimum_jerk_reference(start_xy, end_xy, elapsed, duration):
    """Return (position, velocity, acceleration), each broadcast shape (...,2).

    Elapsed time and duration are scalar or broadcastable batch arrays (...),
    without a trailing coordinate axis. Elapsed must be finite and >=0;
    duration must be finite and >0. Phase is clamped to [0,1]: after the end
    the reference holds the final waypoint with zero velocity/acceleration.
    The start waypoint determines backend, dtype and device. No input mutates.
    """
    start, end = _waypoints(start_xy, end_xy)
    elapsed, duration = _array(elapsed, start), _array(duration, start)
    _validate(elapsed, "elapsed", nonnegative=True)
    _validate(duration, "duration", positive=True)
    xp = _backend(start)
    phase = (elapsed / duration).clamp(0., 1.) if hasattr(start, "new_tensor") else np.clip(elapsed / duration, 0., 1.)
    delta = end - start
    blend = phase**3 * (10 + phase * (-15 + 6 * phase))
    speed = 30 * phase**2 * (1 - phase)**2 / duration
    acceleration = 60 * phase * (1 - phase) * (1 - 2 * phase) / duration**2
    position = start + blend[..., None] * delta
    # Exact endpoint selection avoids cancellation changing a latched goal.
    position = xp.where((phase >= 1)[..., None], end, position)
    return tuple(_array(value, start) for value in
                 (position, speed[..., None] * delta, acceleration[..., None] * delta))
