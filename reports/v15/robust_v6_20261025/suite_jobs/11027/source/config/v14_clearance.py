"""World-space clearance between the V14 leg collision capsules."""

from __future__ import annotations

import numpy as np

from v14_usd_plan import CAPSULE_CENTER_LINK_M, CAPSULE_HEIGHT_M, CAPSULE_RADIUS_M, LEGS, PROFILE_PARTS


def _dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sum(first * second, axis=-1)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    length_squared = _dot(direction, direction)
    numerator = _dot(point - start, direction)
    fraction = np.divide(
        numerator,
        length_squared,
        out=np.zeros_like(numerator),
        where=length_squared > 1e-12,
    )
    closest = start + np.clip(fraction, 0.0, 1.0)[..., None] * direction
    return np.linalg.norm(point - closest, axis=-1)


def segment_distance_m(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray:
    """Return the exact minimum distance of two finite 3D segments."""
    p0, p1, q0, q1 = np.broadcast_arrays(
        *[np.asarray(point, dtype=np.float64) for point in
          (first_start, first_end, second_start, second_end)]
    )
    if p0.shape[-1] != 3 or not all(np.isfinite(point).all() for point in (p0, p1, q0, q1)):
        raise ValueError("segment endpoints must be finite XYZ vectors")
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a, b, c = _dot(u, u), _dot(u, v), _dot(v, v)
    d, e = _dot(u, w), _dot(v, w)
    denominator = a * c - b * b
    s = np.divide(b * e - c * d, denominator, out=np.zeros_like(a), where=denominator > 1e-12)
    t = np.divide(a * e - b * d, denominator, out=np.zeros_like(a), where=denominator > 1e-12)
    interior = (denominator > 1e-12) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    interior_distance = np.where(
        interior,
        np.linalg.norm(w + s[..., None] * u - t[..., None] * v, axis=-1),
        np.inf,
    )
    return np.minimum.reduce((
        interior_distance,
        _point_segment_distance(p0, q0, q1),
        _point_segment_distance(p1, q0, q1),
        _point_segment_distance(q0, p0, p1),
        _point_segment_distance(q1, p0, p1),
    ))


def capsule_gap_m(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    first_radius_m: float,
    second_radius_m: float,
) -> np.ndarray:
    if not np.isfinite(first_radius_m + second_radius_m) or min(first_radius_m, second_radius_m) < 0:
        raise ValueError("capsule radii must be finite and non-negative")
    return segment_distance_m(first_start, first_end, second_start, second_end) - (
        first_radius_m + second_radius_m
    )


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("body quaternions must be finite and nonzero")
    q = quaternion / norm
    doubled_cross = 2.0 * np.cross(q[..., 1:], vector)
    return vector + q[..., :1] * doubled_cross + np.cross(q[..., 1:], doubled_cross)


def interleg_clearances_m(
    body_positions_world_m: np.ndarray,
    body_quaternions_wxyz: np.ndarray,
    body_names: tuple[str, ...] | list[str],
    *,
    radius_m: float = CAPSULE_RADIUS_M,
) -> np.ndarray:
    """Return [environment, 24 cross-leg capsule-pair] signed gaps in meters."""
    positions = np.asarray(body_positions_world_m, dtype=np.float64)
    quaternions = np.asarray(body_quaternions_wxyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or quaternions.shape != positions.shape[:-1] + (4,):
        raise ValueError("body poses must have [env, body, XYZ/quaternion] shape")
    if positions.shape[1] != len(body_names) or len(set(body_names)) != len(body_names):
        raise ValueError("body names must uniquely match the pose tensor")
    if not np.isfinite(positions).all():
        raise ValueError("body positions must be finite")

    names = [f"{leg}_{part}_profile" for leg in LEGS for part in PROFILE_PARTS]
    lookup = {name: index for index, name in enumerate(body_names)}
    if any(name not in lookup for name in names):
        raise ValueError("a V14 thigh/calf profile body is missing")
    indices = [lookup[name] for name in names]
    centers = np.asarray(CAPSULE_CENTER_LINK_M)
    half = np.array((CAPSULE_HEIGHT_M * 0.5, 0.0, 0.0))
    selected_positions = positions[:, indices]
    selected_quaternions = quaternions[:, indices]
    starts = selected_positions + _rotate_wxyz(selected_quaternions, centers - half)
    ends = selected_positions + _rotate_wxyz(selected_quaternions, centers + half)

    first, second = zip(*(
        (left, right)
        for left in range(8)
        for right in range(left + 1, 8)
        if left // 2 != right // 2
    ))
    return capsule_gap_m(
        starts[:, first], ends[:, first], starts[:, second], ends[:, second],
        radius_m, radius_m,
    )


def summarize_interleg_clearance(gaps_m: np.ndarray, alive: np.ndarray) -> dict[str, float | int]:
    """Summarize per-control-step gaps while excluding completed environments."""
    gaps = np.asarray(gaps_m, dtype=np.float64)
    mask = np.asarray(alive, dtype=bool)
    if gaps.ndim != 3 or gaps.shape[-1] != 24 or mask.shape != gaps.shape[:2]:
        raise ValueError("gaps must be [step, env, 24] and alive [step, env]")
    if not np.isfinite(gaps).all() or not mask.any():
        raise ValueError("gaps must be finite and at least one environment must be alive")
    minimum_by_frame = gaps.min(axis=-1)[mask]
    return {
        "measured_env_frames": int(mask.sum()),
        "minimum_gap_m": float(minimum_by_frame.min()),
        "negative_gap_frame_fraction": float(np.mean(minimum_by_frame < 0.0)),
        "gap_below_minus_2mm_frame_fraction": float(np.mean(minimum_by_frame < -0.002)),
    }
