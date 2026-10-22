"""NumPy/Torch contact and support-triangle algebra, without simulator imports.

XY points are world-aligned and must share a coordinate origin. The caller
supplies the whole-robot COM, measured contact status and the chosen physical
foot points. No function here establishes friction feasibility or gait success.

Primary references:
https://underactuated.mit.edu/humanoids.html (whole COM and contact convex hull)
https://arxiv.org/html/1909.06586 (contact-first weighted acceleration projection)
https://jzus.zju.edu.cn/opentxt.php?doi=10.1631/jzus.A2200310 (walk COM preparation)
"""
from __future__ import annotations

import math
import numpy as np


def _float_array(value, like=None):
    if like is not None and hasattr(like, "new_tensor"):
        import torch
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if like is None and hasattr(value, "new_tensor"):
        import torch
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError("Torch inputs must use float32 or float64")
        return value
    result = np.asarray(value, dtype=None if like is None else like.dtype)
    if result.dtype.kind != "f":
        result = result.astype(float)
    if result.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("NumPy inputs must use float32 or float64")
    return result


def _xp(value):
    if hasattr(value, "new_tensor"):
        import torch
        return torch
    return np


def _sum(value, axis=-1):
    return value.sum(dim=axis) if hasattr(value, "new_tensor") else value.sum(axis=axis)


def _min(value, axis=-1):
    return value.amin(dim=axis) if hasattr(value, "new_tensor") else value.min(axis=axis)


def _max(value, axis=-1):
    return value.amax(dim=axis) if hasattr(value, "new_tensor") else value.max(axis=axis)


def _all(value, axis=None):
    if hasattr(value, "new_tensor"):
        return value.all() if axis is None else value.all(dim=axis)
    return value.all(axis=axis)


def _stack(values, axis, like):
    xp = _xp(like)
    return xp.stack(values, dim=axis) if hasattr(like, "new_tensor") else xp.stack(values, axis=axis)


def _transpose(value):
    return value.transpose(-1, -2) if hasattr(value, "new_tensor") else value.swapaxes(-1, -2)


def _eye(size, like):
    xp = _xp(like)
    return xp.eye(size, dtype=like.dtype, device=like.device) if hasattr(like, "new_tensor") else xp.eye(size, dtype=like.dtype)


def _positive_scalar(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")


def contact_support_masks(normal_force, *, presence_min_n=1., load_min_n=20.):
    """Return (contact present, stably loaded) Boolean masks in input shape.

    Normal forces are in newtons. Strictly greater thresholds preserve the
    runtime's existing stable-load gate. A newly landed foot can be present
    before it is loaded, allowing the force allocator to build that load.
    Nonfinite forces fail both gates. Inputs are not modified; NumPy/Torch
    outputs retain their backend and, for Torch, their device.
    """
    _positive_scalar(presence_min_n, "presence_min_n")
    _positive_scalar(load_min_n, "load_min_n")
    if presence_min_n >= load_min_n:
        raise ValueError("presence_min_n must be less than load_min_n")
    force = _float_array(normal_force)
    finite = _xp(force).isfinite(force)
    return finite & (force > presence_min_n), finite & (force > load_min_n)


def _triangle_geometry(triangle_xy, length_tolerance, relative_area_tolerance):
    _positive_scalar(length_tolerance, "length_tolerance")
    _positive_scalar(relative_area_tolerance, "relative_area_tolerance")
    vertices = _float_array(triangle_xy)
    if vertices.shape[-2:] != (3, 2):
        raise ValueError("Expected triangle shape (...,3,2)")
    xp = _xp(vertices)
    finite = _all(_all(xp.isfinite(vertices), -1), -1)
    vertices = xp.where(xp.isfinite(vertices), vertices, 0.)
    edges = _stack((vertices[..., 1, :] - vertices[..., 0, :],
                    vertices[..., 2, :] - vertices[..., 1, :],
                    vertices[..., 0, :] - vertices[..., 2, :]), -2, vertices)
    lengths = xp.sqrt(_sum(edges * edges))
    first, second = edges[..., 0, :], -edges[..., 2, :]
    twice_area = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    valid = (finite & (_min(lengths) > length_tolerance)
             & (xp.abs(twice_area) > relative_area_tolerance * _max(lengths)**2))
    return vertices, edges, lengths, twice_area, valid


def triangle_signed_margin(triangle_xy, point_xy, *, length_tolerance=1e-8,
                           relative_area_tolerance=1e-8):
    """Return (minimum inward edge distance, valid), in the input XY units.

    Shapes (...,3,2) and (...,2) broadcast. Positive margin means inside,
    zero means on an edge, negative means outside. ``valid`` characterizes
    finite nondegenerate geometry/point, not whether the point is inside.
    Invalid samples return -inf, preventing a positive-margin liftoff gate.
    Outside margin is signed half-plane distance, not Euclidean distance to
    a corner. Area tolerance is relative to the square of the longest edge.
    """
    vertices, edges, lengths, area, valid = _triangle_geometry(
        triangle_xy, length_tolerance, relative_area_tolerance)
    point = _float_array(point_xy, vertices)
    if point.shape[-1:] != (2,):
        raise ValueError("Expected point shape (...,2)")
    xp = _xp(vertices)
    valid = valid & _all(xp.isfinite(point), -1)
    point = xp.where(xp.isfinite(point), point, 0.)
    safe_length = xp.where(lengths > length_tolerance, lengths, 1.)
    orientation = xp.where(area >= 0., xp.ones_like(area), -xp.ones_like(area))
    normals = _stack((-edges[..., 1], edges[..., 0]), -1, vertices)
    normals = normals * orientation[..., None, None] / safe_length[..., None]
    distance = _sum((point[..., None, :] - vertices) * normals)
    margin = _min(distance)
    return xp.where(valid, margin, xp.full_like(margin, -math.inf)), valid


def triangle_com_target(triangle_xy, *, inset=0., point_xy=None, length_tolerance=1e-8,
                        relative_area_tolerance=1e-8):
    """Return (safe inset target, valid), with finite fallback 0.

    ``inset`` is a nonnegative scalar edge clearance in XY units. All three
    edges are moved inward by that distance. The inset triangle is a homothety
    about the incenter with scale 1-inset/inradius, so its centroid varies
    continuously from the original centroid to the incenter. With point_xy=None
    return that centroid. With a desired point (...,2), preserve it when inside
    the inset triangle; otherwise return its Euclidean projection onto the
    nearest inset segment/corner, minimizing COM displacement. An inset larger
    than the inradius is infeasible. Invalid triangles/points have a zero
    target and valid=False; callers MUST keep four-contact preparation active.
    This target supplies geometry only; motion timing and COM tracking belong
    to the runtime planner.
    """
    if not isinstance(inset, (int, float)) or not math.isfinite(inset) or inset < 0:
        raise ValueError("inset must be a finite nonnegative scalar")
    vertices, _, lengths, area, valid = _triangle_geometry(
        triangle_xy, length_tolerance, relative_area_tolerance)
    xp = _xp(vertices)
    perimeter = _sum(lengths)
    safe_perimeter = xp.where(valid, perimeter, xp.ones_like(perimeter))
    # Opposite-edge lengths weight vertices 0,1,2 respectively.
    opposite = _stack((lengths[..., 1], lengths[..., 2], lengths[..., 0]), -1, vertices)
    incenter = _sum(vertices * opposite[..., None], -2) / safe_perimeter[..., None]
    radius = xp.abs(area) / safe_perimeter
    valid = valid & (inset <= radius)
    safe_radius = xp.where(valid, radius, xp.ones_like(radius))
    centroid = _sum(vertices, -2) / 3.
    scale = xp.ones_like(safe_radius) - _float_array(inset, vertices) / safe_radius
    target = incenter + scale[..., None] * (centroid - incenter)
    if point_xy is not None:
        point = _float_array(point_xy, vertices)
        if point.shape[-1:] != (2,):
            raise ValueError("Expected point_xy shape (...,2)")
        valid = valid & _all(xp.isfinite(point), -1)
        point = xp.where(xp.isfinite(point), point, 0.)
        inset_vertices = incenter[..., None, :] + scale[..., None, None] * (vertices - incenter[..., None, :])
        edges = _stack((inset_vertices[..., 1, :] - inset_vertices[..., 0, :],
                        inset_vertices[..., 2, :] - inset_vertices[..., 1, :],
                        inset_vertices[..., 0, :] - inset_vertices[..., 2, :]), -2, vertices)
        length_squared = _sum(edges * edges)
        safe_length_squared = xp.where(length_squared > 0., length_squared, xp.ones_like(length_squared))
        fraction = _sum((point[..., None, :] - inset_vertices) * edges) / safe_length_squared
        fraction = xp.clip(fraction, 0., 1.)
        candidates = inset_vertices + fraction[..., None] * edges
        distance_squared = _sum((candidates - point[..., None, :])**2)
        if hasattr(vertices, "new_tensor"):
            indices = distance_squared.argmin(dim=-1)
            index = indices[..., None, None].expand(indices.shape + (1, 2))
            closest = xp.gather(candidates, -2, index)[..., 0, :]
        else:
            indices = distance_squared.argmin(axis=-1)
            index = np.broadcast_to(indices[..., None, None], indices.shape + (1, 2))
            closest = np.take_along_axis(candidates, index, axis=-2)[..., 0, :]
        margin, _ = triangle_signed_margin(vertices, point, length_tolerance=length_tolerance,
                                           relative_area_tolerance=relative_area_tolerance)
        target = xp.where((margin >= inset)[..., None], point, closest)
    return xp.where(valid[..., None], target, 0.), valid


def contact_acceleration_projection(desired_acceleration, foot_jacobian, stance,
                                    *, contact_target=None, weight_matrix=None,
                                    regularization=1e-6):
    """Return (weighted projected acceleration, masked contact residual).

    Inputs: a0 (...,n), J (...,feet,3,n), stance (...,feet), b (...,feet,3).
    ``b`` defaults to zero. To hold stationary stance points, the caller supplies
    b=-Jdot*v+Kp*(anchor-point)-Kd*(J*v), using the SAME physical point as J.
    W=weight_matrix (...,n,n) is symmetric positive definite, or defaults to I.

    The unregularized solution minimizes (a-a0)^T W (a-a0) subject to Js*a=b.
    Let G=Js W^-1 Js^T, s_i=sqrt(G_ii) (zero rows use s_i=1), S=diag(s).
    Solve (S^-1 G S^-1 + eps*I) z = S^-1*(b-Js*a0), then
    a=a0+W^-1 Js^T S^-1 z. Thus epsilon is dimensionless and invariant to
    independent changes in contact-row units. Effective epsilon is
    max(regularization, 8*row_count*finfo(dtype).eps), retaining a representable
    regularizer for float32 redundant rows. Constraint satisfaction is soft;
    residual=Js*a-b must be checked by the caller.

    Disabled J/b rows are selected to zero BEFORE multiplication; disabled NaNs
    cannot contaminate valid peers. Returned residual has shape (...,feet,3)
    and disabled rows exactly zero. No input is mutated. This projection does
    not enforce floating-base dynamics, friction cones, or torque limits.
    """
    _positive_scalar(regularization, "regularization")
    acceleration = _float_array(desired_acceleration)
    if acceleration.ndim < 1:
        raise ValueError("Expected acceleration shape (...,n)")
    n = acceleration.shape[-1]
    J = _float_array(foot_jacobian, acceleration)
    if J.ndim < 3 or J.shape[-2:] != (3, n):
        raise ValueError("Expected Jacobian shape (...,feet,3,n)")
    feet = J.shape[-3]
    if feet < 1 or n < 1:
        raise ValueError("Contact and acceleration dimensions must be nonempty")
    xp = _xp(acceleration)
    if hasattr(acceleration, "new_tensor"):
        mask = xp.as_tensor(stance, dtype=xp.bool, device=acceleration.device)
    else:
        mask = np.asarray(stance, dtype=bool)
    if mask.shape[-1:] != (feet,):
        raise ValueError("Expected stance shape (...,feet)")
    J = xp.where(mask[..., None, None], J, 0.)
    target = xp.zeros_like(J[..., 0]) if contact_target is None else _float_array(contact_target, acceleration)
    if target.shape[-2:] != (feet, 3):
        raise ValueError("Expected contact_target shape (...,feet,3)")
    target = xp.where(mask[..., None], target, 0.)
    for value in (acceleration, J, target):
        if not bool(_all(xp.isfinite(value))):
            raise ValueError("Nonfinite acceleration or active contact data")
    rows = feet * 3
    J = J.reshape(J.shape[:-3] + (rows, n))
    target = target.reshape(target.shape[:-2] + (rows,))
    JT = _transpose(J)
    if weight_matrix is None:
        weighted_JT = JT
    else:
        weight = _float_array(weight_matrix, acceleration)
        if weight.shape[-2:] != (n, n) or not bool(_all(xp.isfinite(weight))):
            raise ValueError("Expected finite weight_matrix shape (...,n,n)")
        tolerance = 16 * xp.finfo(weight.dtype).eps
        if not xp.allclose(weight, _transpose(weight), rtol=tolerance, atol=tolerance):
            raise ValueError("weight_matrix must be symmetric positive definite")
        try:
            xp.linalg.cholesky(weight)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            raise ValueError("weight_matrix must be symmetric positive definite") from error
        weighted_JT = xp.linalg.solve(weight, JT)
    gram = J @ weighted_JT
    diagonal = gram.diagonal(dim1=-2, dim2=-1) if hasattr(gram, "new_tensor") else np.diagonal(gram, axis1=-2, axis2=-1)
    scales = xp.sqrt(xp.where(diagonal > 0., diagonal, 1.))
    normalized_gram = gram / scales[..., :, None] / scales[..., None, :]
    epsilon = max(regularization, 8 * rows * xp.finfo(acceleration.dtype).eps)
    error = target - (J @ acceleration[..., None])[..., 0]
    multiplier = xp.linalg.solve(normalized_gram + epsilon * _eye(rows, acceleration),
                                 (error / scales)[..., None])[..., 0] / scales
    projected = acceleration + (weighted_JT @ multiplier[..., None])[..., 0]
    residual = (J @ projected[..., None])[..., 0] - target
    return projected, residual.reshape(residual.shape[:-1] + (feet, 3))


def quarter_crawl_stance(phase, *, duty_factor=.8):
    """Return canonical FL,FR,RL,RR stance flags, never scheduling two swings.

    ``phase`` is a dimensionless cycle count, not wall time. Duty broadcasts
    over phase and MUST lie in [.75,1]. Offsets are [0,.5,.75,.25]. A duty >.75
    creates four-support gaps but does not guarantee enough COM preparation
    time. The caller must pause its phase clock for preparation/touchdown gates.
    This steady-cycle helper does not infer measured contacts or implement a
    sequential gait state machine.
    """
    phase = _float_array(phase)
    duty = _float_array(duty_factor, phase)
    xp = _xp(phase)
    if not bool(_all(xp.isfinite(phase))) or not bool(_all(xp.isfinite(duty) & (duty >= .75) & (duty <= 1.))):
        raise ValueError("Finite phase and duty_factor in [.75,1] are required")
    offsets = _float_array([0., .5, .75, .25], phase)
    leg_phase = xp.remainder(phase[..., None] - offsets, 1.)
    stance = leg_phase < duty[..., None]
    if not bool(_all(_sum(stance) >= 3)):
        raise ValueError("Numerical phase lost the three-support crawl invariant")
    return stance


def prepared_crawl_transition(stage, elapsed, ready_elapsed, *, dt, ready,
                              all_contacts, stance_contacts_ok, swing_seconds,
                              prepare_min_seconds=.1, settle_seconds=.1):
    """Advance independent PREPARE(0), SWING(1), SETTLE(2) state machines.

    Return (stage,elapsed,ready_elapsed,liftoff,touchdown,advance_leg). Arrays
    broadcast over environments; output stage is int64, clocks retain elapsed's
    float32/float64 dtype/device, and event flags are Boolean one-tick pulses.
    Inputs are never modified. At most ONE transition happens in a call.

    PREPARE counts total time plus consecutive ready AND all_contacts time.
    Both must reach prepare_min_seconds before liftoff. SWING time advances
    only while stance_contacts_ok, and reaching swing_seconds enters SETTLE.
    ``touchdown`` labels this trajectory-end transition, not measured contact.
    SETTLE requires consecutive all_contacts for settle_seconds before advancing
    the leg. A contact/readiness interruption resets that consecutive dwell.
    Every stage transition resets both clocks to zero.

    The caller defines ready (e.g. whole COM margin/velocity), contact hysteresis,
    leg order, point anchors and swing references. Flags describe the current
    sampled interval; this helper makes no claim about intersample contact.
    Boundary comparisons use 8*dtype_epsilon*max(|clock|,|threshold|) seconds to
    avoid an extra tick from float32 roundoff at an exact duration boundary.
    """
    for value, name in ((dt, "dt"), (prepare_min_seconds, "prepare_min_seconds"),
                        (settle_seconds, "settle_seconds")):
        _positive_scalar(value, name)
    clock = _float_array(elapsed)
    dwell = _float_array(ready_elapsed, clock)
    stages = _float_array(stage, clock)
    duration = _float_array(swing_seconds, clock)
    xp = _xp(clock)
    for value in (clock, dwell):
        if not bool(_all(xp.isfinite(value) & (value >= 0.))):
            raise ValueError("Elapsed clocks must be finite and nonnegative")
    if not bool(_all(xp.isfinite(stages) & (stages == xp.floor(stages)) & (stages >= 0.) & (stages <= 2.))):
        raise ValueError("stage must contain only PREPARE=0, SWING=1, SETTLE=2")
    if not bool(_all(xp.isfinite(duration) & (duration > 0.))):
        raise ValueError("swing_seconds must be finite and positive")

    def flag(value):
        data = xp.as_tensor(value, device=clock.device) if hasattr(clock, "new_tensor") else np.asarray(value)
        if not bool(_all((data == 0) | (data == 1))):
            raise ValueError("Condition flags must be Boolean or binary 0/1")
        return data.to(dtype=xp.bool) if hasattr(clock, "new_tensor") else data.astype(bool)

    ready_flag, contact_flag, stance_flag = (flag(value) for value in
                                           (ready, all_contacts, stance_contacts_ok))
    dt_array = _float_array(dt, clock)
    prepare_duration = _float_array(prepare_min_seconds, clock)
    settle_duration = _float_array(settle_seconds, clock)
    zero = xp.zeros_like(clock)
    is_prepare, is_swing, is_settle = (stages == value for value in (0, 1, 2))
    next_clock = xp.where(is_swing & ~stance_flag, clock, clock + dt_array)
    prepare_dwell = xp.where(ready_flag & contact_flag, dwell + dt_array, zero)
    settle_dwell = xp.where(contact_flag, dwell + dt_array, zero)
    next_dwell = xp.where(is_prepare, prepare_dwell, xp.where(is_settle, settle_dwell, zero))

    def reached(value, threshold):
        tolerance = 8 * xp.finfo(clock.dtype).eps * xp.maximum(xp.abs(value), xp.abs(threshold))
        return value + tolerance >= threshold

    liftoff = (is_prepare & ready_flag & contact_flag & reached(next_clock, prepare_duration)
               & reached(next_dwell, prepare_duration))
    touchdown = is_swing & stance_flag & reached(next_clock, duration)
    advance_leg = (is_settle & contact_flag & reached(next_clock, settle_duration)
                   & reached(next_dwell, settle_duration))
    next_stage = xp.where(liftoff, 1, xp.where(touchdown, 2, xp.where(advance_leg, 0, stages)))
    transitioned = liftoff | touchdown | advance_leg
    next_clock = xp.where(transitioned, zero, next_clock)
    next_dwell = xp.where(transitioned, zero, next_dwell)
    next_stage = next_stage.to(dtype=xp.int64) if hasattr(clock, "new_tensor") else next_stage.astype(np.int64)
    return next_stage, next_clock, next_dwell, liftoff, touchdown, advance_leg
