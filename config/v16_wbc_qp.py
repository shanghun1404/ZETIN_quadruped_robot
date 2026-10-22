"""NumPy/Torch whole-body acceleration and point-force QP, without simulation.

Thirty variables jointly adjust generalized acceleration and four contact
forces. A single consistent world point Jacobian supplies contact kinematics,
base wrenches and joint torques. Coordinates are world root-COM translation,
world root angular velocity, then FL/FR/RL/RR joints; forces are XYZ per foot.

Primary references:
https://arxiv.org/html/1909.06586#S4.SS2 (acceleration/force relaxation)
https://groups.csail.mit.edu/robotics-center/public_papers/Kuindersma13.pdf
https://osqp.org/docs/solver/index.html (box-split ADMM and residuals)
This fixed-iteration implementation is not OSQP and returns no certificate
of infeasibility or guarantee about actual contact physics or gait stability.
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


def _xp(value):
    if hasattr(value, "new_tensor"):
        import torch
        return torch
    return np


def _transpose(value):
    return value.transpose(-1,-2) if hasattr(value,"new_tensor") else value.swapaxes(-1,-2)


def _sum(value, axis=-1):
    return value.sum(dim=axis) if hasattr(value,"new_tensor") else value.sum(axis=axis)


def _max(value, axis=-1):
    return value.amax(dim=axis) if hasattr(value,"new_tensor") else value.max(axis=axis)


def _cat(values, axis, like):
    return _xp(like).cat(values,dim=axis) if hasattr(like,"new_tensor") else np.concatenate(values,axis=axis)


def _eye(size, like):
    xp = _xp(like)
    return xp.eye(size,dtype=like.dtype,device=like.device) if hasattr(like,"new_tensor") else xp.eye(size,dtype=like.dtype)


def _zeros(shape, like):
    xp = _xp(like)
    return xp.zeros(shape,dtype=like.dtype,device=like.device) if hasattr(like,"new_tensor") else xp.zeros(shape,dtype=like.dtype)


def _broadcast(value, shape):
    try:
        return _xp(value).broadcast_to(value,shape)
    except (ValueError,RuntimeError) as error:
        raise ValueError(f"Input shape {tuple(value.shape)} cannot broadcast to {shape}") from error


def _require(condition, message):
    if not bool(condition.all()):
        raise ValueError(message)


def _positive(value, name, allow_zero=False):
    if (not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value)
            or (value<0 if allow_zero else value<=0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} scalar")


def _matvec(matrix, vector):
    return (matrix@vector[...,None])[...,0]


def _cholesky_solve(factor, vector):
    if hasattr(factor,"new_tensor"):
        import torch
        return torch.cholesky_solve(vector[...,None],factor)[...,0]
    forward,result = np.empty_like(vector),np.empty_like(vector)
    for i in range(30):
        forward[...,i] = (vector[...,i]-np.sum(factor[...,i,:i]*forward[...,:i],axis=-1))/factor[...,i,i]
    for i in range(29,-1,-1):
        result[...,i] = (forward[...,i]-np.sum(factor[...,i+1:,i]*result[...,i+1:],axis=-1))/factor[...,i,i]
    return result


def solve_wbc_qp(M, h, Jcontact, desired_acceleration, stance, *, contact_target,
                 limits, tau_lower, tau_upper, weight, acceleration_scale=None,
                 acceleration_weights=None, torque_regularization=.01,
                 force_regularization=.01, contact_acceleration_scale=10.,
                 mu=.6, max_normal_force=500., angular_length=.30,
                 iterations=320, rho=10., over_relaxation=1., tolerance=1e-3, warm_start=None,
                 backend='auto'):
    """Return (a, forces, torque, diagnostics, state), without post-clipping.

    M (...,18,18) must be the caller's consistent effective generalized inertia
    (including the chosen rotor convention), h (...,18) its modeled bias.
    Jcontact (...,4,3,18) uses the SAME world points for both kinematics and
    forces. contact_target (...,4,3) is Ja in m/s², e.g. anchor feedback minus
    Jdot*v. stance (...,4) is binary. All batch axes broadcast. limits and
    caller-computed velocity/slew tau bounds are scalar or (...,12), in Nm.
    weight (...) is total mass*9.81 in N, not a dimensionless objective weight.

    x=[a/s,u], lambda=weight*u. Default s=[3 linear,10 angular,50 joint]
    in m/s² or rad/s²; default positive tracking weights are [100]*6+[1]*12.
    Scale/weights accept scalars or (...,18); weights are DIAGONAL, not matrices.
    Minimize .5*sum(w*(a/s-a_des/s)^2) +
    .5*torque_regularization*||tau/limits||² +
    .5*force_regularization*||lambda/weight||².

    Hard constraints: base M_b*a+h_b=J_b^T*lambda; active J*a=contact_target;
    0<=Fz<=max_normal_force; |Fx|,|Fy|<=mu/sqrt(2)*Fz;
    tau_lower<=M_j*a+h_j-J_j^T*lambda<=tau_upper; disabled forces exactly zero.
    Disabled J and target rows are selected to zero before any arithmetic,
    so inactive NaNs cannot poison valid contacts. No acceleration bound is
    imposed; returned accelerations, especially free fall, require caller gates.

    Base rows divide by weight [N] and weight*angular_length [Nm], contact rows
    by contact_acceleration_scale [m/s²], torque rows by limits [Nm]. ADMM then
    equilibrates nonzero rows by their Euclidean norm and rescales the whole
    objective by max(1,max(diag(H))), preserving the minimizer. One 30x30 batched
    Cholesky per call is reused for fixed iterations. Warm state x30/z58/y58 is
    not mutated. z/y are a heuristic initial guess as row normalization changes,
    not carried physical dual variables. Callers should reset the state on
    contact/point, normalization scale, weight, or objective changes.
    Standard ADMM relaxation defaults to 1; its valid range is strictly (0,2).
    The 320-iteration default covers the hand-derived roll/contact test; 80
    iterations did not satisfy its tolerance. Runtime cost is not benchmarked.

    Diagnostics use the ACTUALLY RETURNED a/lambda/tau, not box auxiliary z.
    feasible means finite and max normalized violation<=tolerance. converged
    also checks equilibrated primal/KKT residuals. False is not a certificate
    of infeasibility; true is not a certificate about physical contact or gait.
    """
    if backend not in ('auto', 'numpy', 'cpp'):
        raise ValueError('backend must be auto, numpy or cpp')
    for name,value in (("force_regularization",force_regularization),("contact_acceleration_scale",contact_acceleration_scale),
                       ("max_normal_force",max_normal_force),("angular_length",angular_length),
                       ("rho",rho),("tolerance",tolerance)):
        _positive(value,name)
    _positive(torque_regularization,"torque_regularization",allow_zero=True)
    _positive(mu,"mu",allow_zero=True)
    _positive(over_relaxation,"over_relaxation")
    if over_relaxation>=2:
        raise ValueError("over_relaxation must be less than 2")
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<=0:
        raise ValueError("iterations must be a positive integer")
    M = _array(M)
    xp = _xp(M)
    h,J,a_des,enabled,target = (_array(value,M) for value in (h,Jcontact,desired_acceleration,stance,contact_target))
    limit,lower_tau,upper_tau,body_weight = (_array(value,M) for value in (limits,tau_lower,tau_upper,weight))
    scale = _array([3.]*3+[10.]*3+[50.]*12 if acceleration_scale is None else acceleration_scale,M)
    track = _array([100.]*6+[1.]*12 if acceleration_weights is None else acceleration_weights,M)
    if M.shape[-2:]!=(18,18) or J.shape[-3:]!=(4,3,18):
        raise ValueError("M/Jcontact require (...,18,18)/(...,4,3,18)")
    if h.shape[-1:]!=(18,) or a_des.shape[-1:]!=(18,) or enabled.shape[-1:]!=(4,) or target.shape[-2:]!=(4,3):
        raise ValueError("h/desired_acceleration/stance/contact_target require (...,18)/(...,18)/(...,4)/(...,4,3)")
    for value,tail,name in ((limit,12,"limits"),(lower_tau,12,"tau_lower"),(upper_tau,12,"tau_upper"),
                            (scale,18,"acceleration_scale"),(track,18,"acceleration_weights")):
        if value.ndim and value.shape[-1]!=tail:
            raise ValueError(f"{name} must be scalar or (...,{tail})")
    shapes = [M.shape[:-2],J.shape[:-3],h.shape[:-1],a_des.shape[:-1],enabled.shape[:-1],target.shape[:-2],body_weight.shape]
    shapes += [value.shape[:-1] if value.ndim else () for value in (limit,lower_tau,upper_tau,scale,track)]
    try:
        batch = tuple(np.broadcast_shapes(*shapes))
    except ValueError as error:
        raise ValueError("Input batch axes do not broadcast") from error
    M,J = _broadcast(M,batch+(18,18)),_broadcast(J,batch+(4,3,18))
    h,a_des = _broadcast(h,batch+(18,)),_broadcast(a_des,batch+(18,))
    enabled,target = _broadcast(enabled,batch+(4,)),_broadcast(target,batch+(4,3))
    limit,lower_tau,upper_tau = (_broadcast(value,batch+(12,)) for value in (limit,lower_tau,upper_tau))
    scale,track = (_broadcast(value,batch+(18,)) for value in (scale,track))
    body_weight = _broadcast(body_weight,batch)
    _require(xp.isfinite(enabled)&((enabled==0)|(enabled==1)),"stance must be binary")
    enabled = enabled>0
    J = xp.where(enabled[...,None,None],J,0.)
    target = xp.where(enabled[...,None],target,0.)
    for name,value in (("M",M),("h",h),("active Jcontact",J),("active contact_target",target),
                       ("desired_acceleration",a_des),("limits",limit),("tau_lower",lower_tau),
                       ("tau_upper",upper_tau),("weight",body_weight),("acceleration_scale",scale),("acceleration_weights",track)):
        _require(xp.isfinite(value),f"{name} must be finite")
    _require((limit>0),"limits must be positive in Nm")
    _require(body_weight>0,"weight must be positive in N")
    _require((scale>0)&(track>0),"acceleration scales and tracking weights must be positive")
    _require(lower_tau<=upper_tau,"tau_lower must not exceed tau_upper")
    Jflat = J.reshape(batch+(12,18))
    Jbase = _transpose(Jflat[...,:6])
    Jjoint = _transpose(Jflat[...,6:])
    scaled_M = M*scale[...,None,:]
    Ttau = _cat((scaled_M[...,6:,:],-body_weight[...,None,None]*Jjoint),-1,M)/limit[..., :,None]
    tau_bias = h[...,6:]/limit
    tracking_diagonal = _cat((track,force_regularization*xp.ones_like(h[...,:12])),-1,M)
    H = _eye(30,M)*tracking_diagonal[...,None,:] + torque_regularization*(_transpose(Ttau)@Ttau)
    q = _cat((-track*a_des/scale,xp.zeros_like(h[...,:12])),-1,M)
    q = q+torque_regularization*_matvec(_transpose(Ttau),tau_bias)
    wrench_scale = body_weight[...,None]*_array([1.]*3+[angular_length]*3,M)
    Cbase = _cat((scaled_M[...,:6,:],-body_weight[...,None,None]*Jbase),-1,M)/wrench_scale[..., :,None]
    bbase = -h[...,:6]/wrench_scale
    Ccontact = _cat((Jflat*scale[...,None,:],_zeros(batch+(12,12),M)),-1,M)/contact_acceleration_scale
    bcontact = target.reshape(batch+(12,))/contact_acceleration_scale
    Cforce = _cat((_zeros(batch+(12,18),M),_broadcast(_eye(12,M),batch+(12,12))),-1,M)
    lower_force = xp.full_like(target,-math.inf)
    upper_force = xp.full_like(target,math.inf)
    lower_force[...,2],upper_force[...,2] = 0.,max_normal_force/body_weight[...,None]
    lower_force = xp.where(enabled[...,None],lower_force,0.).reshape(batch+(12,))
    upper_force = xp.where(enabled[...,None],upper_force,0.).reshape(batch+(12,))
    friction = np.zeros((16,30))
    coefficient = mu/math.sqrt(2.)
    for foot in range(4):
        friction[4*foot:4*foot+4,18+3*foot:21+3*foot] = [[1,0,-coefficient],[-1,0,-coefficient],
                                                                  [0,1,-coefficient],[0,-1,-coefficient]]
    Cfriction = _broadcast(_array(friction,M),batch+(16,30))
    C = _cat((Cbase,Ccontact,Cforce,Cfriction,Ttau),-2,M)
    force_mask = _broadcast(enabled[...,None],batch+(4,3)).reshape(batch+(12,))
    variable_mask = _cat((xp.ones_like(h)>0,force_mask),-1,M)
    C = xp.where(variable_mask[...,None,:],C,0.)
    lower = _cat((bbase,bcontact,lower_force,xp.full_like(Cfriction[...,0],-math.inf),(lower_tau-h[...,6:])/limit),-1,M)
    upper = _cat((bbase,bcontact,upper_force,xp.zeros_like(Cfriction[...,0]),(upper_tau-h[...,6:])/limit),-1,M)
    row_norm = xp.sqrt(_sum(C*C))
    row_norm = xp.where(row_norm>0,row_norm,1.)
    C,lower,upper = C/row_norm[...,None],lower/row_norm,upper/row_norm
    diagonal = H.diagonal(dim1=-2,dim2=-1) if hasattr(H,"new_tensor") else np.diagonal(H,axis1=-2,axis2=-1)
    objective_scale = xp.maximum(_max(diagonal),xp.ones_like(body_weight))
    H,q = H/objective_scale[...,None,None],q/objective_scale[...,None]
    factor = xp.linalg.cholesky(H+rho*(_transpose(C)@C))
    if warm_start is None:
        x = _zeros(batch+(30,),M)
        z = xp.minimum(xp.maximum(_matvec(C,x),lower),upper)
        y = xp.zeros_like(z)
    else:
        if not isinstance(warm_start,dict) or any(key not in warm_start for key in ("x","z","y")):
            raise ValueError("warm_start requires x, z, y")
        x,z,y = (_array(warm_start[key],M) for key in ("x","z","y"))
        for value,tail in ((x,30),(z,58),(y,58)):
            if value.shape[-1:]!=(tail,):
                raise ValueError("Invalid warm_start shape")
        x,z,y = _broadcast(x,batch+(30,)),_broadcast(z,batch+(58,)),_broadcast(y,batch+(58,))
        x = xp.where(variable_mask,x,0.)
        zero_row = _sum(C*C)==0
        z,y = xp.where(zero_row,0.,z),xp.where(zero_row,0.,y)
        for value in (x,z,y):
            _require(xp.isfinite(value),"Active warm_start values must be finite")
        z = xp.minimum(xp.maximum(z,lower),upper)
    CT = _transpose(C)
    native = None
    supported = not hasattr(M, 'new_tensor') and x.shape == (30,) and M.dtype == np.float64
    if backend == 'cpp' and not supported:
        raise ValueError('C++ backend requires unbatched float64 NumPy inputs')
    if backend != 'numpy' and supported:
        try:
            from _v16_wbc_native import iterate
            native = iterate
        except ImportError:
            if backend == 'cpp':
                raise ImportError('Build the C++ WBC extension before selecting backend=cpp') from None
    if native is not None:
        x, z, y = native(factor, C, q, lower, upper, variable_mask, x, z, y,
                         rho, over_relaxation, iterations)
        Cx = _matvec(C, x)
    else:
        for _ in range(iterations):
            x = _cholesky_solve(factor,-q+rho*_matvec(CT,z-y))
            x = xp.where(variable_mask,x,0.)
            Cx = _matvec(C,x)
            relaxed = over_relaxation*Cx+(1.-over_relaxation)*z
            z = xp.minimum(xp.maximum(relaxed+y,lower),upper)
            y = y+relaxed-z
    acceleration = x[...,:18]*scale
    forces = (body_weight[...,None]*x[...,18:]).reshape(batch+(4,3))
    dynamic_force = _matvec(M,acceleration)+h
    contact_force = _matvec(_transpose(Jflat),forces.reshape(batch+(12,)))
    torque = dynamic_force[...,6:]-contact_force[...,6:]
    base_error = dynamic_force[...,:6]-contact_force[...,:6]
    base_error_normalized = base_error/wrench_scale
    contact_error = (_matvec(Jflat,acceleration).reshape(batch+(4,3))-target)
    contact_error = xp.where(enabled[...,None],contact_error,0.)
    torque_error = xp.maximum(xp.maximum(lower_tau-torque,torque-upper_tau),xp.zeros_like(torque))
    normal_error = xp.maximum(xp.maximum(-forces[...,2],forces[...,2]-max_normal_force),xp.zeros_like(forces[...,2]))
    tangent_error = xp.maximum(xp.abs(forces[...,:2])-coefficient*forces[...,2,None],xp.zeros_like(forces[...,:2]))
    disabled_error = xp.where(enabled[...,None],0.,xp.abs(forces))
    base_violation = _max(xp.abs(base_error_normalized))
    contact_violation_physical = _max(_max(xp.abs(contact_error)))
    contact_violation = contact_violation_physical/contact_acceleration_scale
    torque_violation = _max(torque_error/limit)
    normal_violation = _max(normal_error)/body_weight
    friction_violation = _max(_max(tangent_error))/body_weight
    disabled_violation = _max(_max(disabled_error))/body_weight
    violation = _max(_cat(tuple(v[...,None] for v in (base_violation,contact_violation,torque_violation,
                                                   normal_violation,friction_violation,disabled_violation)),-1,M))
    primal_residual = _max(xp.abs(Cx-z))
    dual_residual = _max(xp.abs(_matvec(H,x)+q+rho*_matvec(CT,y)))
    finite = (xp.isfinite(violation)&xp.isfinite(primal_residual)&xp.isfinite(dual_residual)
              &(_sum(xp.isfinite(x))==30)&(_sum(xp.isfinite(torque))==12))
    change = (acceleration-a_des)/scale
    diagnostics = dict(finite=finite,feasible=finite&(violation<=tolerance),
        converged=finite&(violation<=tolerance)&(primal_residual<=tolerance)&(dual_residual<=tolerance),
        max_constraint_violation=violation,base_wrench_violation=base_violation,
        base_wrench_error=base_error,base_wrench_error_normalized=base_error_normalized,
        contact_acceleration_error=contact_error,contact_acceleration_violation=contact_violation,
        contact_acceleration_violation_m_s2=contact_violation_physical,
        torque_violation_nm=_max(torque_error),normal_force_violation_n=_max(normal_error),
        friction_violation_n=_max(_max(tangent_error)),disabled_force_violation_n=_max(_max(disabled_error)),
        acceleration_tracking_error_normalized=change,
        acceleration_change_norm_normalized=xp.sqrt(_sum(change*change)),
        primal_residual=primal_residual,dual_residual=dual_residual)
    return acceleration,forces,torque,diagnostics,dict(x=x,z=z,y=y)
