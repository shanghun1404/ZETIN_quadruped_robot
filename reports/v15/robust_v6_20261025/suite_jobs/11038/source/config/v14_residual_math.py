"""Tensor geometry and physical-unit limits for the V14 residual policy."""

import math

import torch

from v14_usd_plan import CAPSULE_CENTER_LINK_M, CAPSULE_HEIGHT_M, CAPSULE_RADIUS_M


def bounded_delta(raw, max_rad):
    if not math.isfinite(max_rad) or not 0 < max_rad <= .1:
        raise ValueError("residual bound must be finite and in (0, .1] rad")
    return max_rad * torch.tanh(raw)


def compose_action(base, delta_rad, action_scale):
    if base.shape != delta_rad.shape or not math.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("matching action shapes and positive scale required")
    return base + delta_rad / action_scale


def _dot(a, b):
    return (a*b).sum(-1)


def _point_segment(point, start, end):
    direction = end-start
    t = (_dot(point-start, direction) / _dot(direction, direction).clamp_min(1e-12)).clamp(0, 1)
    return torch.linalg.vector_norm(point-start-t[..., None]*direction, dim=-1)


def _rotate(q, vector):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    vector = vector.expand_as(q[..., 1:])
    cross = 2*torch.cross(q[..., 1:], vector, dim=-1)
    return vector + q[..., :1]*cross + torch.cross(q[..., 1:], cross, dim=-1)


def capsule_gaps(positions, quaternions):
    """Signed surface gaps for 24 cross-leg pairs; order FL/FR/RL/RR, thigh/calf."""
    if positions.ndim != 3 or positions.shape[1:] != (8, 3) or quaternions.shape != positions.shape[:-1]+(4,):
        raise ValueError("expected [N,8,3] positions and [N,8,4] wxyz quaternions")
    center = positions.new_tensor(CAPSULE_CENTER_LINK_M)
    half = positions.new_tensor((CAPSULE_HEIGHT_M/2, 0., 0.))
    starts = positions + _rotate(quaternions, center-half)
    ends = positions + _rotate(quaternions, center+half)
    first, second = zip(*((i, j) for i in range(8) for j in range(i+1, 8) if i//2 != j//2))
    p0, p1, q0, q1 = starts[:, first], ends[:, first], starts[:, second], ends[:, second]
    u, v, w = p1-p0, q1-q0, p0-q0
    a, b, c, d, e = _dot(u,u), _dot(u,v), _dot(v,v), _dot(u,w), _dot(v,w)
    denominator = a*c-b*b
    s = (b*e-c*d)/denominator.clamp_min(1e-12)
    t = (a*e-b*d)/denominator.clamp_min(1e-12)
    interior = (denominator > 1e-12) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    inside_distance = torch.where(interior, torch.linalg.vector_norm(w+s[...,None]*u-t[...,None]*v, dim=-1), torch.inf)
    distance = torch.stack((inside_distance, _point_segment(p0,q0,q1), _point_segment(p1,q0,q1),
                            _point_segment(q0,p0,p1), _point_segment(q1,p0,p1))).amin(0)
    return distance - 2*CAPSULE_RADIUS_M


def clearance_cost(gaps):
    return ((.02-gaps)/.02).clamp(0,3).square().amax(-1)
