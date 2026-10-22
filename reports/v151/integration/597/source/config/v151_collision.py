"""Exact geometric V14 profile capsule clearance to the source trunk box.

This checks geometry independently of collider presence or collision filters.
Gap = segment-to-solid-box distance minus capsule radius. Negative means overlap;
once the centerline intersects the box, gap saturates at -radius and is NOT a
measurement of penetration depth. Actual PhysX contact must be recorded separately.
"""

from __future__ import annotations

import numpy as np

from v14_clearance import _rotate_wxyz
from v14_usd_plan import collision_capsules


TRUNK_CENTER_BASE_M = (0.0, 0.0, 0.075)
TRUNK_SIZE_M = (0.4, 0.52, 0.15)
PROFILE_NAMES = tuple(spec.link_name for spec in collision_capsules())


def segment_aabb_distance_m(start, end, half_extents):
    """Exact segment distance to a solid origin-centered axis-aligned box.

    Inputs broadcast over leading dimensions. Every box-plane crossing splits
    [0,1] into intervals where squared distance is a quadratic. Minimize each
    quadratic analytically, including interval boundaries; no point sampling.
    """
    p, q, half = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64)
                                     for x in (start, end, half_extents)])
    if p.ndim < 1 or p.shape[-1] != 3 or not all(np.isfinite(x).all() for x in (p, q, half)):
        raise ValueError("segment endpoints and box extents must be finite XYZ vectors")
    if (half <= 0).any():
        raise ValueError("box half extents must be positive")
    direction = q - p
    crossings = []
    for face in (-half, half):
        crossing = np.divide(face - p, direction, out=np.zeros_like(p), where=direction != 0)
        crossings.append(np.clip(crossing, 0., 1.))
    boundaries = np.sort(np.concatenate((np.zeros(p.shape[:-1] + (1,)),
                                         *crossings, np.ones(p.shape[:-1] + (1,))), axis=-1), axis=-1)
    low, high = boundaries[..., :-1], boundaries[..., 1:]
    middle = (low + high) * .5
    at_middle = p[..., None, :] + middle[..., :, None] * direction[..., None, :]
    box_half = half[..., None, :]
    outside = np.abs(at_middle) > box_half
    # On each interval: distance-vector components are a*t+b outside, zero inside.
    a = np.where(outside, direction[..., None, :], 0.)
    b = np.where(outside, p[..., None, :] - np.sign(at_middle) * box_half, 0.)
    aa = np.sum(a * a, axis=-1)
    stationary = np.divide(-np.sum(a * b, axis=-1), aa, out=low.copy(), where=aa != 0)
    fraction = np.clip(stationary, low, high)
    closest = p[..., None, :] + fraction[..., :, None] * direction[..., None, :]
    residual = np.maximum(np.abs(closest) - box_half, 0.)
    return np.sqrt(np.min(np.sum(residual * residual, axis=-1), axis=-1))


def capsule_trunk_gaps_m(body_positions_world_m, body_quaternions_wxyz, body_names):
    """Return [env,8] signed gaps in PROFILE_NAMES order from actor body poses.

    The body's ``base`` actor frame defines the trunk; body poses must not be
    center-of-mass poses. Capsule centers, X axes, heights and radii come from
    v14_usd_plan, regardless of which collision pairs PhysX enables.
    """
    positions = np.asarray(body_positions_world_m, dtype=np.float64)
    quaternions = np.asarray(body_quaternions_wxyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or quaternions.shape != positions.shape[:-1] + (4,):
        raise ValueError("body poses must be [env,body,XYZ/quaternion]")
    if len(body_names) != positions.shape[1] or len(set(body_names)) != len(body_names):
        raise ValueError("body names must uniquely match pose tensors")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("body poses must be finite")
    lookup = {name: i for i, name in enumerate(body_names)}
    if any(name not in lookup for name in ("base", *PROFILE_NAMES)):
        raise ValueError("base or a V14 profile body is missing")
    specs = collision_capsules()
    ids = [lookup[spec.link_name] for spec in specs]
    centers = np.asarray([spec.center_link_m for spec in specs])
    half_segments = np.array([[spec.height_m / 2., 0., 0.] for spec in specs])
    base_pos = positions[:, lookup["base"], None, :]
    inverse_base_quat = quaternions[:, lookup["base"], None, :].copy()
    inverse_base_quat[..., 1:] *= -1.
    endpoints = []
    for local in (centers - half_segments, centers + half_segments):
        world = positions[:, ids] + _rotate_wxyz(quaternions[:, ids], local)
        endpoints.append(_rotate_wxyz(inverse_base_quat, world - base_pos) - TRUNK_CENTER_BASE_M)
    distance = segment_aabb_distance_m(*endpoints, np.asarray(TRUNK_SIZE_M) / 2.)
    return distance - np.asarray([spec.radius_m for spec in specs])


def summarize_trunk_clearance(gaps_m, alive):
    """Summarize live [step,env,8] samples without asserting physical contact."""
    gaps = np.asarray(gaps_m, dtype=np.float64)
    mask = np.asarray(alive, dtype=bool)
    if gaps.ndim != 3 or gaps.shape[-1] != len(PROFILE_NAMES) or mask.shape != gaps.shape[:2]:
        raise ValueError("gaps must be [step,env,8] and alive [step,env]")
    if not mask.any() or not np.isfinite(gaps[mask]).all():
        raise ValueError("at least one finite live frame is required")
    live = gaps[mask]
    minimum = live.min(axis=-1)
    worst_profile = int(np.unravel_index(np.argmin(live), live.shape)[-1])
    return {
        "measured_env_frames": int(mask.sum()),
        "minimum_gap_m": float(minimum.min()),
        "minimum_gap_profile": PROFILE_NAMES[worst_profile],
        "negative_gap_frame_fraction": float(np.mean(minimum < 0.)),
        "gap_below_minus_2mm_frame_fraction": float(np.mean(minimum < -.002)),
        "per_profile_minimum_gap_m": dict(zip(PROFILE_NAMES, map(float, live.min(axis=0)))),
        "penetration_depth_measured": False,
        "actual_contact_measured": False,
        "geometry_independent_of_collision_filters": True,
        "gap_semantics": "distance(segment,solid trunk box)-radius; saturates at -radius on centerline intersection",
    }
