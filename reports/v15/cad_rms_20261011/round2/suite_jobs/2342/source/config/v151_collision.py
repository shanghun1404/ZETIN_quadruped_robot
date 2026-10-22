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
LEGS = ('FL', 'FR', 'RL', 'RR')
SAME_LEG_THIGH_FOOT_PAIRS = tuple(f'{leg}_thigh_profile--{leg}_foot' for leg in LEGS)


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


def foot_mesh_box_from_points(points, mesh_to_foot):
    """Enclose every mesh vertex in an AABB expressed in the foot actor frame.

    The USD/Gf matrix uses row vectors and includes all intervening transforms.
    Its box also contains the mesh's convex hull. Box overlap is conservative
    evidence of possible interference, not proof of a convex-hull collision.
    """
    vertices = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(mesh_to_foot, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('mesh points must be a nonempty [vertex,3] array')
    if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.isfinite(vertices).all():
        raise ValueError('mesh geometry and transform must be finite')
    if not np.allclose(matrix[:,3], [0.,0.,0.,1.], atol=1e-12, rtol=0):
        raise ValueError('expected an affine USD row-vector transform')
    local = vertices @ matrix[:3,:3] + matrix[3,:3]
    lower, upper = local.min(axis=0), local.max(axis=0)
    if np.any(upper <= lower):
        raise ValueError('foot collider bounds must have positive volume')
    return dict(center_local_m=((lower+upper)/2).tolist(),
                half_extents_m=((upper-lower)/2).tolist())


def foot_proxy_boxes_from_stage(stage, robot_prim_path):
    """Inspect the actual composed V14 foot collision meshes, including instances.

    Deliberately reject a changed collider type/approximation instead of silently
    applying a stale CAD bound. This routine reads USD only; it changes no filters.
    """
    from pxr import UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    boxes = []
    for leg in LEGS:
        body = stage.GetPrimAtPath(f'{robot_prim_path}/base/{leg}_foot')
        mesh = stage.GetPrimAtPath(f'{robot_prim_path}/base/{leg}_foot/collisions/foot/foot')
        if not body.IsValid() or not mesh.IsValid() or mesh.GetTypeName() != 'Mesh':
            raise ValueError(f'{leg}: expected V14 foot collision mesh')
        if not mesh.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get():
            raise ValueError(f'{leg}: foot collider is not enabled')
        approximation = UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()
        if approximation != 'convexHull':
            raise ValueError(f'{leg}: unreviewed foot collision approximation {approximation}')
        transform, resets = cache.ComputeRelativeTransform(mesh, body)
        if resets:
            raise ValueError(f'{leg}: unexpected reset transform inside foot collider')
        vertices = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
        box = foot_mesh_box_from_points(vertices, np.asarray(transform))
        thigh = stage.GetPrimAtPath(f'{robot_prim_path}/base/{leg}_thigh_profile')
        filtered_targets = [str(path) for path in UsdPhysics.FilteredPairsAPI(thigh).GetFilteredPairsRel().GetTargets()]
        box.update(leg=leg, collider_path=str(mesh.GetPath()), mesh_point_count=len(vertices),
                   mesh_to_foot_transform=np.asarray(transform).tolist(),
                   mesh_points_bounds=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                   collision_approximation=approximation, proxy='foot_actor_frame_AABB_of_convex_hull_vertices',
                   actual_contact_measured=False, physical_pair_filtered=str(body.GetPath()) in filtered_targets,
                   interpretation='Positive gap separates modeled geometry; negative gap is possible proxy overlap, not confirmed mesh collision. Contact offsets and solver tolerances excluded.')
        boxes.append(box)
    return boxes


def same_leg_thigh_foot_gaps_m(body_positions_world_m, body_quaternions_wxyz, body_names, foot_boxes):
    """Return [env,4] thigh-capsule/own-foot-box gaps in FL,FR,RL,RR order.

    These pairs are physically filtered in the existing V14 asset. Transform the
    capsule into each foot actor frame so the off-origin CAD foot mesh is placed
    correctly. Negative values screen possible overlap; they are not measured
    contact or penetration depth and must not join the cross-leg adoption gate.
    """
    positions = np.asarray(body_positions_world_m, dtype=np.float64)
    quats = np.asarray(body_quaternions_wxyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or quats.shape != positions.shape[:-1]+(4,):
        raise ValueError('body poses must be [env,body,XYZ/quaternion]')
    if len(body_names) != positions.shape[1] or len(set(body_names)) != len(body_names):
        raise ValueError('body names must uniquely match pose tensors')
    if not np.isfinite(positions).all() or not np.isfinite(quats).all() or len(foot_boxes) != 4:
        raise ValueError('finite poses and four foot boxes required')
    lookup = {name:i for i,name in enumerate(body_names)}
    wanted = [f'{leg}_{part}' for leg in LEGS for part in ('thigh_profile','foot')]
    if any(name not in lookup for name in wanted):
        raise ValueError('a same-leg thigh or foot body is missing')
    thigh_ids = [lookup[f'{leg}_thigh_profile'] for leg in LEGS]
    foot_ids = [lookup[f'{leg}_foot'] for leg in LEGS]
    specs = [spec for spec in collision_capsules() if spec.link_name.endswith('thigh_profile')]
    centers = np.asarray([spec.center_link_m for spec in specs])
    half_segments = np.array([[spec.height_m/2.,0.,0.] for spec in specs])
    box_centers = np.asarray([box['center_local_m'] for box in foot_boxes])
    box_half = np.asarray([box['half_extents_m'] for box in foot_boxes])
    if box_centers.shape != (4,3) or box_half.shape != (4,3) or not np.isfinite(box_centers).all():
        raise ValueError('four finite XYZ foot boxes required')
    inverse = quats[:,foot_ids].copy(); inverse[...,1:] *= -1.
    endpoints = []
    for local in (centers-half_segments, centers+half_segments):
        world = positions[:,thigh_ids]+_rotate_wxyz(quats[:,thigh_ids],local)
        endpoints.append(_rotate_wxyz(inverse,world-positions[:,foot_ids])-box_centers)
    return segment_aabb_distance_m(*endpoints,box_half)-np.asarray([spec.radius_m for spec in specs])
