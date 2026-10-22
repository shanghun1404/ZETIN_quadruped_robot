"""CPU-only anchored-foot posture study; not a physics/contact validation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from v151_ros2_control.standing import StandingController
from v151_collision import capsule_trunk_gaps_m
from v14_clearance import interleg_clearances_m


def angular_limits(usd_path):
    from pxr import Usd, UsdPhysics
    stage=Usd.Stage.Open(str(usd_path))
    limits={}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint=UsdPhysics.RevoluteJoint(prim)
            lower,upper=joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()
            limits[prim.GetName()]=dict(lower_deg=float(lower) if np.isfinite(lower) else None,
                                       upper_deg=float(upper) if np.isfinite(upper) else None)
    return limits


def search(model_path,usd_path):
    controller=StandingController(model_path)
    nominal=controller.nominal_joint_position.copy()
    quat=[1.,0.,0.,0.]
    feet=controller.contact_positions(nominal,[0.,0.,controller.nominal_height],quat)
    ratings=np.tile([8.3,18.,18.],4)
    cases={(h,y) for h in (.364,.394,.424,.454,.464,.484,.504) for y in (-.06,-.03,0.,.03,.06)}
    cases.update((.364,y) for y in (.07,.08,.09,.10,.11,.12))
    cases.update(((.394,.09),(.394,.12),(.424,.09)))
    rows=[]
    for height,y in sorted(cases):
        position=[0.,y,height];base=controller._base(position,quat);q=nominal.copy()
        for _ in range(100):
            error=(feet-controller.contact_positions(q,position,quat)).ravel()
            jac=controller._jacobians(q,base)[:,:,6:].reshape(12,12)
            if np.max(np.abs(error))<1e-8: break
            q+=np.clip(np.linalg.solve(jac.T@jac+1e-8*np.eye(12),jac.T@error),-.12,.12)
        else:
            rows.append(dict(height_m=height,forward_shift_m=y,ik_converged=False));continue
        bias=controller.model.bias(q,np.zeros(18),include_damping=False,**base)
        contact=controller._jacobians(q,base).reshape(12,18)
        joint_map=contact[:,6:].T
        # Fixed posture, zero acceleration: base equilibrium is exact. Minimize
        # maximum |tau/rated| over supported friction-constrained point forces.
        inequality=np.r_[np.c_[-joint_map,-ratings],np.c_[joint_map,-ratings]]
        rhs=np.r_[-bias[6:],bias[6:]]
        friction=[]
        for foot in range(4):
            for axis in (0,1):
                for sign in (-1.,1.):
                    row=np.zeros(13);row[foot*3+axis]=sign;row[foot*3+2]=-.6/np.sqrt(2)
                    friction.append(row)
        lp=linprog(np.r_[np.zeros(12),1.],A_ub=np.r_[inequality,friction],
            b_ub=np.r_[rhs,np.zeros(len(friction))],A_eq=np.c_[contact[:,:6].T,np.zeros(6)],
            b_eq=bias[:6],bounds=[(None,None),(None,None),(5.,500.)]*4+[(0.,None)],method='highs')
        controller.initialize(q,position,quat)
        tau,diag=controller.compute(q,np.zeros(12),position,quat,np.zeros(3),np.zeros(3),
                                   height=height,joint_reference=q)
        state=controller.model.configuration(q,**base);names=controller.model.body_names
        poses=[controller.model._frame_pose(state,name)[1] for name in names]
        body_positions=np.array([pose.translation for pose in poses])[None]
        body_quats=np.array([controller.model.pin.Quaternion(pose.rotation).coeffs()[[3,0,1,2]] for pose in poses])[None]
        com=controller.model.pin.centerOfMass(controller.model.model,controller.model.model.createData(),state)
        rows.append(dict(height_m=height,forward_shift_m=y,ik_converged=True,
            ik_max_contact_error_m=float(np.max(np.abs(error))),joint_position_rad=q.tolist(),
            joint_jacobian_condition=float(np.linalg.cond(jac)),
            wbc_torque_nm=tau.tolist(),wbc_max_rated_fraction=float(np.max(np.abs(tau)/ratings)),
            wbc_feasible=bool(diag['feasible']),wbc_forces_world_N=diag['forces_world'].tolist(),
            static_minimax_rated_fraction=float(lp.fun) if lp.success else None,
            static_minimax_torque_nm=(bias[6:]-joint_map@lp.x[:12]).tolist() if lp.success else None,
            center_of_mass_world_m=com.tolist(),
            minimum_interleg_capsule_gap_m=float(interleg_clearances_m(body_positions,body_quats,names).min()),
            minimum_trunk_proxy_gap_m=float(capsule_trunk_gaps_m(body_positions,body_quats,names).min())))
    limits=angular_limits(usd_path)
    return dict(physics_simulation_performed=False,scope='static anchored-foot theoretical search',
        model_path=str(model_path),model_sha256=controller.model.source_sha256,
        usd_path=str(usd_path),usd_sha256=hashlib.sha256(Path(usd_path).read_bytes()).hexdigest(),
        mass_kg=controller.model.total_mass,nominal_actor_height_m=controller.nominal_height,
        joint_order=list(controller.model.joint_names),nominal_contact_anchors_world_m=feet.tolist(),
        usd_angular_limits=limits,
        hardware_angular_limits_verified=False,
        static_force_constraints='zero acceleration,base force/moment equilibrium,5<=Fz<=500N,|Fx|,|Fy|<=0.6/sqrt(2)*Fz',
        limitations=['USD joints are unbounded where limit is null; no hardware angular-limit assurance.',
            'Fixed projected foot material points approximate actual moving contact.',
            'WBC feasibility and static torque are model results, not actual contact/stability proof.',
            'No transition trajectory, motor thermal state or body-ground collision validation.',
            'Negative trunk-proxy gaps do not measure penetration depth or actual contact.'],poses=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--usd',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(search(args.model,args.usd),indent=2,allow_nan=False)+'\n')
