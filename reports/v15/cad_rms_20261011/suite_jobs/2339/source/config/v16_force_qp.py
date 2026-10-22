"""Batched NumPy/Torch torque-constrained contact-force allocation.

This module contains algebra only, and imports no simulator or optimizer.
The caller supplies one consistent world contact-point Jacobian for both
the desired acceleration and its force wrench. Forces use canonical
FL, FR, RL, RR order, with XYZ per foot, and upward Z on flat ground.

Primary references:
https://osqp.org/docs/solver/index.html (box-split ADMM and residuals)
https://arxiv.org/html/1909.06586 (floating-base dynamics and force constraints)
This is a small fixed-iteration ADMM implementation, not the OSQP library;
it provides numerical violations, not a certificate of infeasibility.
"""
from __future__ import annotations

import math
import numpy as np


def _array(value, like=None):
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


def _xp(like):
    if hasattr(like, "new_tensor"):
        import torch
        return torch
    return np


def _transpose(value):
    return value.transpose(-1,-2) if hasattr(value,"new_tensor") else value.swapaxes(-1,-2)


def _max(value, axis=-1):
    return value.amax(dim=axis) if hasattr(value,"new_tensor") else value.max(axis=axis)


def _sum(value, axis=-1):
    return value.sum(dim=axis) if hasattr(value,"new_tensor") else value.sum(axis=axis)


def _cat(values, axis, like):
    xp = _xp(like)
    return xp.cat(values, dim=axis) if hasattr(like,"new_tensor") else xp.concatenate(values, axis=axis)


def _eye(size, like):
    xp = _xp(like)
    return xp.eye(size,dtype=like.dtype,device=like.device) if hasattr(like,"new_tensor") else xp.eye(size,dtype=like.dtype)


def _broadcast(value, shape):
    try:
        return _xp(value).broadcast_to(value, shape)
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"Input shape {tuple(value.shape)} cannot broadcast to {shape}") from error


def _require(condition, message):
    if not bool(condition.all()):
        raise ValueError(message)


def _positive(value, name, allow_zero=False):
    if (not isinstance(value,(int,float)) or isinstance(value,bool)
            or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} scalar")


def _matvec(matrix, vector):
    return (matrix @ vector[...,None])[...,0]


def _cholesky_solve(factor, vector):
    if hasattr(factor,"new_tensor"):
        import torch
        return torch.cholesky_solve(vector[...,None], factor)[...,0]
    # Forward/back substitution across the batch reuses the single Cholesky
    # factor. np.linalg.solve on the triangular factor would refactor it every
    # iteration; no SciPy or offline solver is needed here.
    forward = np.empty_like(vector)
    result = np.empty_like(vector)
    for i in range(12):
        forward[...,i] = (vector[...,i] - np.sum(factor[...,i,:i]*forward[...,:i],axis=-1)) / factor[...,i,i]
    for i in range(11,-1,-1):
        result[...,i] = (forward[...,i] - np.sum(factor[...,i+1:,i]*result[...,i+1:],axis=-1)) / factor[...,i,i]
    return result


def solve_force_qp(A, Jjoint_transpose, dynjoint, wantedwrench, stance, *,
                   limits, tau_lower, tau_upper, weight, mu=.6,
                   max_normal_force=500., angular_length=.30,
                   force_regularization=.01, iterations=80, rho=.3,
                   tolerance=1e-3, warm_start=None):
    """Return ``(forces, diagnostics, state)`` in the input dtype/device.

    A (...,6,12) maps forces [N] to the base wrench [N,Nm]; its first three
    rows are translation, last three rotation. Jjoint_transpose (...,12,12)
    maps the SAME point forces to joint torques. dynjoint (...,12) is
    (M_eff*a+h)[6:], wantedwrench (...,6) its unactuated base part. stance
    (...,4) enables point contacts. limits, tau_lower, tau_upper are (...,12)
    in Nm, positive limits and ordered finite bounds required. weight (...)
    is mass*9.81 in N. Batch axes broadcast; scalar and per-joint limits work.

    For lambda=weight*u, T=weight*Jjoint_transpose/limits[..., :,None], minimize
    .5*||T*u-dynjoint/limits||^2 + .5*force_regularization*||u||^2.
    Constraints enforce base equality, 0<=Fz<=max_normal_force and
    |Fx|,|Fy|<=mu/sqrt(2)*Fz, zero disabled forces, and
    tau_lower<=dynjoint-Jjoint_transpose*lambda<=tau_upper.
    No force or torque clipping is applied afterward. Disabled force columns
    are zeroed BEFORE all products, excluding disabled NaNs from the batch.

    Base wrench normalization divides force by weight and moment by
    weight*angular_length; torque rows divide by limits. For ADMM conditioning
    each nonzero constraint row is then divided by its Euclidean norm and
    the objective by max(1,max(diag(H))); these do not change the optimizer.
    One 12x12 batched Cholesky is factored per call; iterations are fixed.
    state x is normalized force, z/y are auxiliary/scaled dual variables of
    the equilibrated 46-row constraints. Optional warm_start={x,z,y} is read
    without mutation. Reset warm state if the problem changes discontinuously.

    diagnostics['feasible'] means returned forces satisfy the un-equilibrated
    normalized constraints within tolerance. It does not prove physical
    feasibility, optimality or stability. 'converged' additionally requires
    equilibrated primal/dual residuals <=tolerance. A false flag can reflect
    finite iteration error or an infeasible request; neither is certified.
    Evaluate the returned diagnostics BEFORE applying the returned forces.
    """
    for name,value in (("max_normal_force",max_normal_force), ("angular_length",angular_length),
                       ("force_regularization",force_regularization), ("rho",rho),
                       ("tolerance",tolerance)):
        _positive(value,name)
    _positive(mu,"mu",allow_zero=True)
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    A = _array(A)
    xp = _xp(A)
    B, d, b = (_array(value,A) for value in (Jjoint_transpose,dynjoint,wantedwrench))
    enabled = _array(stance,A)
    limit, lower_tau, upper_tau = (_array(value,A) for value in (limits,tau_lower,tau_upper))
    body_weight = _array(weight,A)
    if A.shape[-2:] != (6,12) or B.shape[-2:] != (12,12):
        raise ValueError("A and Jjoint_transpose require (...,6,12) and (...,12,12)")
    if d.shape[-1:] != (12,) or b.shape[-1:] != (6,) or enabled.shape[-1:] != (4,):
        raise ValueError("dynjoint/wantedwrench/stance require (...,12)/(...,6)/(...,4)")
    # scalar limits/bounds broadcast to all joints; other tails must be 12.
    for value in (limit, lower_tau, upper_tau):
        if value.ndim > 0 and value.shape[-1] != 12:
            raise ValueError("Joint limits/bounds must be scalar or (...,12)")
    batch_shapes = [A.shape[:-2],B.shape[:-2],d.shape[:-1],b.shape[:-1],enabled.shape[:-1],body_weight.shape]
    batch_shapes += [value.shape[:-1] if value.ndim else () for value in (limit,lower_tau,upper_tau)]
    try:
        batch = tuple(np.broadcast_shapes(*batch_shapes))
    except ValueError as error:
        raise ValueError("Input batch axes do not broadcast") from error
    A, B = _broadcast(A,batch+(6,12)), _broadcast(B,batch+(12,12))
    d, b, enabled = _broadcast(d,batch+(12,)),_broadcast(b,batch+(6,)),_broadcast(enabled,batch+(4,))
    limit, lower_tau, upper_tau = (_broadcast(value,batch+(12,)) for value in (limit,lower_tau,upper_tau))
    body_weight = _broadcast(body_weight,batch)
    _require(xp.isfinite(enabled)&((enabled==0)|(enabled==1)),"stance must be binary")
    enabled = enabled > 0
    variable_mask = _broadcast(enabled[...,None],batch+(4,3)).reshape(batch+(12,))
    A, B = xp.where(variable_mask[...,None,:], A, 0.), xp.where(variable_mask[...,None,:],B,0.)
    for name,value in (("active A",A),("active Jjoint_transpose",B),("dynjoint",d),("wantedwrench",b),
                       ("limits",limit),("tau_lower",lower_tau),("tau_upper",upper_tau),("weight",body_weight)):
        _require(xp.isfinite(value),f"{name} must be finite")
    _require(limit>0,"limits must be positive")
    _require(body_weight>0,"weight must be positive in N")
    _require(lower_tau<=upper_tau,"tau_lower must not exceed tau_upper")
    T = body_weight[...,None,None]*B/limit[..., :,None]
    target = d/limit
    H = _transpose(T)@T + force_regularization*_eye(12,A)
    q = -_matvec(_transpose(T),target)
    wrench_scale = _array([1.,1.,1.,angular_length,angular_length,angular_length],A)
    Cbase = A/wrench_scale[..., :,None]
    bbase = b/(body_weight[...,None]*wrench_scale)
    force_identity = _broadcast(_eye(12,A),batch+(12,12))
    lower_force = xp.full_like(d,-math.inf).reshape(batch+(4,3))
    upper_force = xp.full_like(d, math.inf).reshape(batch+(4,3))
    lower_force[...,2], upper_force[...,2] = 0., max_normal_force/body_weight[...,None]
    lower_force = xp.where(enabled[...,None],lower_force,0.).reshape(batch+(12,))
    upper_force = xp.where(enabled[...,None],upper_force,0.).reshape(batch+(12,))
    friction = np.zeros((16,12))
    friction_coefficient = mu/math.sqrt(2.)
    for i in range(4):
        friction[4*i:4*i+4,3*i:3*i+3] = [[1,0,-friction_coefficient],[-1,0,-friction_coefficient],
                                         [0,1,-friction_coefficient],[0,-1,-friction_coefficient]]
    Cfriction = _broadcast(_array(friction,A),batch+(16,12))
    C = _cat((Cbase,force_identity,Cfriction,T),-2,A)
    C = xp.where(variable_mask[...,None,:],C,0.)
    lower = _cat((bbase,lower_force,xp.full_like(Cfriction[...,0],-math.inf),(d-upper_tau)/limit),-1,A)
    upper = _cat((bbase,upper_force,xp.zeros_like(Cfriction[...,0]),(d-lower_tau)/limit),-1,A)
    row_norm = xp.sqrt(_sum(C*C))
    row_norm = xp.where(row_norm>0,row_norm,1.)
    C, lower, upper = C/row_norm[...,None], lower/row_norm, upper/row_norm
    # Objective rescaling is per problem, never a change in relative costs.
    diagonal = H.diagonal(dim1=-2,dim2=-1) if hasattr(H,"new_tensor") else np.diagonal(H,axis1=-2,axis2=-1)
    objective_scale = xp.maximum(_max(diagonal),xp.ones_like(body_weight))
    H, q = H/objective_scale[...,None,None], q/objective_scale[...,None]
    factor = xp.linalg.cholesky(H+rho*(_transpose(C)@C))
    if warm_start is None:
        x = xp.zeros_like(d)
        z = xp.minimum(xp.maximum(_matvec(C,x),lower),upper)
        y = xp.zeros_like(z)
    else:
        if not isinstance(warm_start,dict) or any(key not in warm_start for key in ("x","z","y")):
            raise ValueError("warm_start requires x, z, y")
        x, z, y = (_array(warm_start[key],A) for key in ("x","z","y"))
        for value,tail in ((x,12),(z,46),(y,46)):
            if value.shape[-1:] != (tail,):
                raise ValueError("Invalid warm_start shape")
        x, z, y = _broadcast(x,batch+(12,)),_broadcast(z,batch+(46,)),_broadcast(y,batch+(46,))
        x = xp.where(variable_mask,x,0.)
        zero_row = _sum(C*C)==0
        z, y = xp.where(zero_row,0.,z),xp.where(zero_row,0.,y)
        for value in (x,z,y):
            _require(xp.isfinite(value),"Active warm_start values must be finite")
        z = xp.minimum(xp.maximum(z,lower),upper)
    for _ in range(iterations):
        x = _cholesky_solve(factor,-q+rho*_matvec(_transpose(C),z-y))
        x = xp.where(variable_mask,x,0.)
        Cx = _matvec(C,x)
        z = xp.minimum(xp.maximum(Cx+y,lower),upper)
        y = y+Cx-z
    forces = (body_weight[...,None]*x).reshape(batch+(4,3))
    force_vector = forces.reshape(batch+(12,))
    actual_wrench = _matvec(A,force_vector)
    torque = d-_matvec(B,force_vector)
    base_error = (actual_wrench-b)/(body_weight[...,None]*wrench_scale)
    torque_error = xp.maximum(xp.maximum(lower_tau-torque,torque-upper_tau),xp.zeros_like(torque))
    normal_error = xp.maximum(xp.maximum(-forces[...,2],forces[...,2]-max_normal_force),xp.zeros_like(forces[...,2]))
    tangent_error = xp.maximum(xp.abs(forces[...,:2])-friction_coefficient*forces[...,2,None],xp.zeros_like(forces[...,:2]))
    disabled_error = xp.where(enabled[...,None],0.,xp.abs(forces))
    base_violation = _max(xp.abs(base_error))
    torque_violation = _max(torque_error/limit)
    normal_violation = _max(normal_error)/body_weight
    friction_violation = _max(_max(tangent_error))/body_weight
    disabled_violation = _max(_max(disabled_error))/body_weight
    violation = xp.maximum(xp.maximum(base_violation,torque_violation),
                           xp.maximum(normal_violation,xp.maximum(friction_violation,disabled_violation)))
    primal_residual = _max(xp.abs(Cx-z))
    dual_residual = _max(xp.abs(_matvec(H,x)+q+rho*_matvec(_transpose(C),y)))
    finite = xp.isfinite(violation)&xp.isfinite(primal_residual)&xp.isfinite(dual_residual)
    diagnostics = dict(feasible=finite&(violation<=tolerance),
                       converged=finite&(violation<=tolerance)&(primal_residual<=tolerance)&(dual_residual<=tolerance),
                       max_constraint_violation=violation, base_wrench_violation=base_violation,
                       base_wrench_error_normalized=base_error, base_wrench_error=actual_wrench-b,
                       torque_violation_nm=_max(torque_error), torque=torque,
                       normal_force_violation_n=_max(normal_error),
                       friction_violation_n=_max(_max(tangent_error)),
                       disabled_force_violation_n=_max(_max(disabled_error)),
                       primal_residual=primal_residual, dual_residual=dual_residual)
    return forces, diagnostics, dict(x=x,z=z,y=y)
