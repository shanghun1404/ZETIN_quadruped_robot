"""CPU posture screening, preserving all four material contact anchors.

Run from repository root with
PYTHONPATH=config:ros2/v151_ros2_control:/usr/lib/python3/dist-packages
and /tmp/v151-model-env/bin/python. No physics simulation is performed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from v151_ros2_control.standing import StandingController
from v151_collision import capsule_trunk_gaps_m, same_leg_thigh_foot_gaps_m
from v14_clearance import interleg_clearances_m

OUT = Path(__file__).parent
MODEL = Path('reports/v16/physical_model/v16_model_source_quadruped.xml')
SOURCE = Path('reports/v151/integration/601')
RATINGS = np.tile([8.3, 18., 18.], 4)


def contexts(controller):
    status = json.loads((SOURCE / 'result.status.json').read_text())
    recorded = np.load(SOURCE / 'result.npz')
    final = status[-1]['wbc_vectors']
    stand = next(r['wbc_vectors'] for r in reversed(status) if r['mode'] == 'STAND')
    origin = np.asarray(final['base_position_reference_world'])
    direction = (np.asarray(stand['base_position_reference_world']) - origin)[:2] / .03
    yaw = np.arctan2(-direction[0], direction[1])
    quat = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
    q = np.asarray(final['joint_reference'])
    anchors = controller.contact_positions(q, origin, quat)
    measured = controller.contact_positions(recorded['joint_position'][-1], recorded['position'][-1],
                                            recorded['quaternion'][-1])
    nominal = controller.contact_positions(controller.nominal_joint_position,
        [0., 0., controller.nominal_height], [1., 0., 0., 0.])
    return [dict(name='nominal', anchors=nominal, origin=np.zeros(3), quat=np.array([1., 0., 0., 0.]),
                 initial_q=controller.nominal_joint_position, source='model nominal material points'),
            dict(name='601_final', anchors=anchors, origin=np.r_[origin[:2], 0.], quat=quat, initial_q=q,
                 source='final recorded joint reference FK at final body target and latched upright yaw',
                 reconstructed_yaw_rad=float(yaw),
                 forward_direction_norm=float(np.linalg.norm(direction)),
                 final_measured_material_point_max_error_m=float(np.max(np.abs(anchors-measured))),
                 recorded_anchor_error_m=status[-1]['wbc']['anchor_error_m'])]


def evaluate(controller, context, height, shift, boxes):
    quat = context['quat']
    rot = controller.model.pin.Quaternion(quat[[1,2,3,0]]).matrix()
    position = context['origin'] + rot @ [0., shift, height]
    base = controller._base(position, quat)
    q = context['initial_q'].copy()
    row = dict(context=context['name'], height_m=float(height), forward_shift_m=float(shift))
    for iteration in range(100):
        error = (context['anchors'] - controller.contact_positions(q, position, quat)).ravel()
        if np.max(np.abs(error)) < 1e-8:
            break
        jac = controller._jacobians(q, base)[:, :, 6:].reshape(12,12)
        q += np.clip(np.linalg.solve(jac.T@jac + 1e-8*np.eye(12), jac.T@error), -.12, .12)
    row.update(ik_converged=bool(np.max(np.abs(error)) < 1e-8),
               ik_max_error_m=float(np.max(np.abs(error))), ik_iterations=iteration)
    if not row['ik_converged']:
        return row
    bias = controller.model.bias(q, np.zeros(18), include_damping=False, **base)
    contact = controller._jacobians(q, base).reshape(12,18)
    joint_map = contact[:,6:].T
    inequality = np.r_[np.c_[-joint_map, -RATINGS], np.c_[joint_map, -RATINGS]]
    rhs = np.r_[-bias[6:], bias[6:]]
    friction=[]
    for foot in range(4):
        for axis in (0,1):
            for sign in (-1.,1.):
                constraint=np.zeros(13)
                constraint[3*foot+axis]=sign
                constraint[3*foot+2]=-.6/np.sqrt(2)
                friction.append(constraint)
    lp=linprog(np.r_[np.zeros(12),1.],A_ub=np.r_[inequality,friction],
        b_ub=np.r_[rhs,np.zeros(len(friction))],A_eq=np.c_[contact[:,:6].T,np.zeros(6)],
        b_eq=bias[:6],bounds=[(None,None),(None,None),(5.,500.)]*4+[(0.,None)],method='highs')
    controller.initialize(q,position,quat)
    tau,diag=controller.compute(q,np.zeros(12),position,quat,np.zeros(3),np.zeros(3),
                               height=height,joint_reference=q)
    state=controller.model.configuration(q,**base)
    names=controller.model.body_names
    poses=[controller.model._frame_pose(state,name)[1] for name in names]
    body_pos=np.array([pose.translation for pose in poses])[None]
    body_quat=np.array([controller.model.pin.Quaternion(pose.rotation).coeffs()[[3,0,1,2]] for pose in poses])[None]
    gaps=dict(interleg=float(interleg_clearances_m(body_pos,body_quat,names).min()),
              trunk_proxy=float(capsule_trunk_gaps_m(body_pos,body_quat,names).min()),
              same_leg_thigh_foot=float(same_leg_thigh_foot_gaps_m(body_pos,body_quat,names,boxes).min()))
    forces=np.asarray(diag['forces_world'])
    row.update(joint_position_rad=q.tolist(), ik_jacobian_condition=float(np.linalg.cond(contact[:,6:])),
        wbc_torque_nm=tau.tolist(),wbc_max_rated_fraction=float(np.max(np.abs(tau)/RATINGS)),
        wbc_feasible=bool(diag['feasible']),wbc_min_normal_force_N=float(forces[:,2].min()),
        wbc_forces_world_N=forces.tolist(),wbc_acceleration_norm=float(np.linalg.norm(diag['acceleration'])),
        wbc_friction_pyramid_margin_N=float(np.min(.6/np.sqrt(2)*forces[:,2,None]-np.abs(forces[:,:2]))),
        minimum_gaps_m=gaps,lp_feasible=bool(lp.success),
        lp_max_rated_fraction=float(lp.fun) if lp.success else None,
        lp_torque_nm=(bias[6:]-joint_map@lp.x[:12]).tolist() if lp.success else None,
        lp_forces_world_N=lp.x[:12].reshape(4,3).tolist() if lp.success else None,
        lp_base_equilibrium_error=float(np.max(np.abs(contact[:,:6].T@lp.x[:12]-bias[:6]))) if lp.success else None,
        clearance_and_support_screen_pass=bool(diag['feasible'] and forces[:,2].min()>5.
             and gaps['interleg']>=0. and gaps['same_leg_thigh_foot']>=0.))
    return row


def main():
    controller=StandingController(MODEL)
    contexts_all=contexts(controller)
    boxes=json.loads((SOURCE/'result.json').read_text())['same_leg_thigh_foot_geometry']
    rows=[]
    for context in contexts_all:
        for height in np.linspace(controller.nominal_height-.10,controller.nominal_height,6):
            shifts=sorted(set([0.]+[round(v,6) for v in np.arange(.03,.1501,.005)]))
            for shift in shifts:
                rows.append(evaluate(controller,context,height,shift,boxes))
        print('completed',context['name'],flush=True)
    best=[]
    for context in contexts_all:
        for height in sorted(set(r['height_m'] for r in rows)):
            candidates=[r for r in rows if r['context']==context['name'] and r['height_m']==height
                        and r.get('clearance_and_support_screen_pass')
                        and (height<controller.nominal_height-.0001 or r['forward_shift_m']<=.09)]
            if candidates:
                best.append(min(candidates,key=lambda r:r['wbc_max_rated_fraction']))
    result=dict(scope='CPU anchored IK, static minimax LP, current standing WBC at rest',
        physics_simulation_performed=False,crouch_depth_m=.10,mass_kg=controller.model.total_mass,
        model_sha256=controller.model.source_sha256,
        source_npz_sha256=hashlib.sha256((SOURCE/'result.npz').read_bytes()).hexdigest(),
        joint_order=list(controller.model.joint_names),rated_torque_nm=RATINGS.tolist(),
        contexts=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in contexts_all],
        lp_constraints='a=0, exact base equilibrium, 5<=Fz<=500N, |Fx|,|Fy|<=.6/sqrt(2)*Fz',
        wbc_constraints='production standing.compute, zero velocities, exact pose/reference/anchor agreement',
        clearances='interleg and same-leg thigh-foot nonnegative; trunk proxy reported separately (601 baseline overlaps)',
        limitations=['No contact dynamics, transition accelerations, slipping, thermal or hardware angle validation.',
          'Fixed material contact points approximate finite foot contact patches.',
          'LP is a theoretical torque lower bound, not the current WBC force allocation.',
          'WBC optimizes accelerations and force regularization, so its near-static torque need not equal LP.',
          'Trunk-proxy overlap is not measured penetration and is not gated in existing601 acceptance.',
          '601 supports reconstructed from recorded upright qref; measured point mismatch reported separately.'],
        best_by_context_and_height=best,rows=rows)
    (OUT/'search.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in best:
        print(r['context'],round(r['height_m'],3),r['forward_shift_m'],
              round(r['wbc_max_rated_fraction'],4),round(r['lp_max_rated_fraction'],4),r['minimum_gaps_m'],flush=True)


if __name__=='__main__':
    main()
