"""Real-model numerical checks; requires the optional Pinocchio model runtime."""
from pathlib import Path
import sys
import os

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'config'), str(ROOT / 'ros2/v151_ros2_control')]


def controller():
    from v151_ros2_control.standing import StandingController
    default=ROOT / 'reports/v16/physical_model/v16_model_source_quadruped.xml'
    model=Path(os.environ.get('V151_MODEL',default))
    if not model.exists():
        model=next((p / 'model/quadruped.xml' for p in ROOT.parents if (p / 'model/quadruped.xml').exists()),model)
    return StandingController(model)


def test_crouch_preserves_four_support_points_and_requested_drop():
    # Catches wrong joint order, root reference, and IK sign.
    ctl = controller()
    q, height = ctl.crouch_pose(.10)
    assert height == pytest.approx(ctl.nominal_height - .10)
    feet = ctl.contact_positions(q, [0., 0., height], [1., 0., 0., 0.])
    nominal = ctl.contact_positions(ctl.nominal_joint_position, [0., 0., ctl.nominal_height], [1., 0., 0., 0.])
    np.testing.assert_allclose(feet, nominal, atol=2e-6)
    assert np.max(np.abs(q - ctl.nominal_joint_position)) > .2


def test_optimized_shift_schedule_preserves_feet_at_same_crouch_depth():
    from v151_ros2_control.motion import posture_forward_offset
    ctl=controller();q,height=ctl.crouch_pose(.10)
    base=np.array([0.,0.,height]);quat=[1.,0.,0.,0.]
    ctl.initialize(q,base,quat)
    anchors=ctl.contact_anchors.copy()
    # Entire reference path, including initial smooth startup and both endpoints.
    for fraction,elapsed in [(0.,t) for t in np.linspace(0.,1.,6)]+[(f,5.) for f in np.linspace(0.,1.,21)]:
        shift=posture_forward_offset(fraction,.09,.15,elapsed)
        qref,goal=ctl.contact_consistent_reference(q,base,quat,height+.10*fraction,
            np.ones(4,dtype=bool),center_body=1.,forward_offset_m=shift)
        actual=ctl.contact_positions(qref,goal+[0.,shift,0.],quat)
        np.testing.assert_allclose(actual,anchors,atol=2e-7)
        assert goal[2]==pytest.approx(height+.10*fraction)
    assert ctl.nominal_height-height==pytest.approx(.10)


def test_static_stand_supports_weight_without_contact_acceleration():
    # Catches gravity/contact sign mistakes and a joint-order mismatch.
    ctl = controller()
    tau, d = ctl.compute(ctl.nominal_joint_position, np.zeros(12),
                         [0., 0., ctl.nominal_height], [1., 0., 0., 0.], np.zeros(3), np.zeros(3))
    assert tau.shape == (12,)
    assert np.isfinite(tau).all()
    assert d['finite'] and d['feasible']
    assert np.max(np.abs(d['contact_acceleration_error'])) < .01
    assert abs(d['forces_world'][:, 2].sum() - ctl.model.total_mass * 9.81) < 2.
    assert np.max(np.abs(d['acceleration'][:6])) < .1
    assert np.all(np.abs(tau) <= np.tile([22.32, 43.2, 43.2], 4) + 1e-10)


def test_crouch_hold_then_rise_is_feasible_without_reinitializing_anchors():
    ctl = controller()
    q, h = ctl.crouch_pose(.10)
    _, hold = ctl.compute(q, np.zeros(12), [0., 0., h], [1., 0., 0., 0.], np.zeros(3), np.zeros(3), height=h, joint_reference=q)
    anchors = ctl.contact_anchors.copy()
    _, rise = ctl.compute(q, np.zeros(12), [0., 0., h], [1., 0., 0., 0.], np.zeros(3), np.zeros(3), height=h+.02, joint_reference=q)
    assert hold['feasible'] and rise['feasible']
    assert rise['acceleration'][2] > hold['acceleration'][2] + .1
    np.testing.assert_array_equal(ctl.contact_anchors, anchors)


def test_yaw_and_translation_do_not_change_joint_torque():
    # Catches use of body velocities as world velocities and fixed yaw=0 target.
    a, b = controller(), controller()
    qa, h = a.crouch_pose(.06)
    qz = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    va, wa = [.04, -.02, .01], [.01, -.02, .03]
    ta, da = a.compute(qa, np.linspace(-.01,.01,12), [0., 0., h], [1., 0., 0., 0.], va, wa, height=h, joint_reference=qa)
    tb, db = b.compute(qa, np.linspace(-.01,.01,12), [2., -1., h], qz, va, wa, height=h, joint_reference=qa)
    np.testing.assert_allclose(ta, tb, atol=1e-5)
    np.testing.assert_allclose(da['acceleration'], db['acceleration'], atol=1e-5)


@pytest.mark.parametrize('bad', [np.full(12, np.nan), np.zeros(11)])
def test_invalid_joint_state_is_rejected(bad):
    ctl = controller()
    with pytest.raises(ValueError):
        ctl.compute(bad, np.zeros(12), [0.,0.,.4], [1.,0.,0.,0.], np.zeros(3), np.zeros(3))


def test_nonzero_velocity_obeys_external_rigid_body_equations():
    # Catches accidental MJCF viscous damping, missing rotor inertia, or
    # using mismatched contact points for force and acceleration.
    ctl = controller()
    q, h = ctl.crouch_pose(.04)
    dq = np.linspace(-.08, .09, 12)
    linear, angular = [.04, -.03, .01], [.02, -.01, .03]
    quat = np.array([.995, .03, -.06, .05]); quat /= np.linalg.norm(quat)
    _, d = ctl.compute(q, dq, [0., 0., h], quat, linear, angular, height=h, joint_reference=q)
    model = ctl.model
    config = model.configuration(q, base_position=[0.,0.,h], base_quaternion_xyzw=quat[[1,2,3,0]])
    velocity = np.r_[linear, angular, dq]
    native_v = np.empty(18); native_a = np.empty(18)
    native_v[model.canonical_v_indices] = velocity
    native_a[model.canonical_v_indices] = d['acceleration']
    rigid = np.asarray(model.pin.rnea(model.model, model.data, config, native_v, native_a))[model.canonical_v_indices]
    rigid[6:] += .02 * d['acceleration'][6:]
    contact = np.zeros(18)
    for name, local, force in zip(ctl.foot_names, ctl.contact_local, d['forces_world']):
        frame = model.model.getFrameId(name)
        model.pin.forwardKinematics(model.model, model.data, config)
        model.pin.updateFramePlacements(model.model, model.data)
        rotation = model.data.oMf[frame].rotation.copy()
        spatial = np.asarray(model.pin.computeFrameJacobian(model.model, model.data, config, frame, model.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        # Independent wrench translation: moment at foot origin is r × F.
        wrench = np.r_[force, np.cross(rotation @ local, force)]
        contact += (spatial.T @ wrench)[model.canonical_v_indices]
    np.testing.assert_allclose(rigid[6:] - contact[6:], d['requested_torque'], atol=1e-9)
    np.testing.assert_allclose(rigid[:6] - contact[:6], d['base_wrench_error'], atol=1e-9)


def test_speed_envelope_blocks_further_acceleration_above_no_load_speed():
    ctl = controller()
    dq = np.tile([60., -60., 60.], 4)
    torque, d = ctl.compute(ctl.nominal_joint_position, dq, [0.,0.,ctl.nominal_height],
                            [1.,0.,0.,0.], np.zeros(3), np.zeros(3))
    assert np.all(torque[dq > 0] <= 0.)
    assert np.all(torque[dq < 0] >= 0.)
    assert np.isfinite(torque).all()
    # Infeasibility must remain observable, even when the motor guard clips.
    assert not d['feasible']


def test_reinitialization_latches_new_support_after_walking():
    ctl = controller()
    q = ctl.nominal_joint_position
    ctl.initialize(q, [0.,0.,.464], [1.,0.,0.,0.])
    first = ctl.contact_anchors.copy()
    ctl.initialize(q, [0.,.5,.464], [1.,0.,0.,0.])
    np.testing.assert_allclose(ctl.contact_anchors - first, np.tile([0.,.5,0.], (4,1)), atol=1e-12)
    _, diagnostics = ctl.compute(q, np.zeros(12), [0.,.5,.464], [1.,0.,0.,0.], np.zeros(3), np.zeros(3))
    assert diagnostics['feasible']
    assert diagnostics['anchor_error_m'] < 1e-12


def test_body_twist_contact_bias_matches_geometric_second_derivative():
    # Check convective acceleration independently through point trajectories;
    # a world/body mix-up can pass static equilibrium tests.
    ctl = controller()
    q, h = ctl.crouch_pose(.05)
    quat = np.array([.99, .07, -.05, .03]); quat /= np.linalg.norm(quat)
    base = dict(base_position=np.array([.1,-.1,h]), base_quaternion_xyzw=quat[[1,2,3,0]])
    velocity = np.r_[[.2,-.1,.04,.4,-.3,.2], np.linspace(-.2,.3,12)]
    actual = ctl._jacobian_rate_velocity(q, velocity, base)
    expected = []
    epsilon = 1e-4
    for name, local in zip(ctl.foot_names, ctl.contact_local):
        center = ctl.model.frame_position(q, name, local, **base)
        plus = ctl.model.point_after_integration(q, epsilon*velocity, name, local, **base)
        minus = ctl.model.point_after_integration(q, -epsilon*velocity, name, local, **base)
        expected.append((plus - 2*center + minus) / epsilon**2)
    np.testing.assert_allclose(actual, expected, atol=3e-7)


def test_rise_preserves_v16_unscaled_base_to_joint_tracking_metric():
    # The accepted V16 configuration weights physical base acceleration
    # errors 1000 times joint errors; normalized variables require [3.6,40,1].
    # Compare the controller's consumer-visible torque to the original QP
    # on an independently specified static 2 cm upward target.
    from v16_wbc_qp import solve_wbc_qp
    ctl = controller()
    q, h = ctl.nominal_joint_position, ctl.nominal_height
    base = dict(base_position=[0.,0.,h], base_quaternion_xyzw=[0.,0.,0.,1.])
    jac = np.array([ctl.model.point_jacobian(q, name, local, **base)
                    for name, local in zip(ctl.foot_names, ctl.contact_local)])
    _, _, expected, _, _ = solve_wbc_qp(
        ctl.model.mass_matrix(q, **base), ctl.model.gravity(q, **base), jac,
        np.r_[[0.,0.,.661154,0.,0.,0.], np.zeros(12)], np.ones(4),
        contact_target=np.zeros((4,3)), limits=np.tile([24.8,48.,48.],4),
        tau_lower=-np.tile([22.32,43.2,43.2],4), tau_upper=np.tile([22.32,43.2,43.2],4),
        weight=ctl.model.total_mass*9.81, acceleration_weights=[3.6]*3+[40.]*3+[1.]*12)
    actual, _ = ctl.compute(q, np.zeros(12), [0.,0.,h], [1.,0.,0.,0.], np.zeros(3), np.zeros(3), height=h+.02)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_measured_stance_disables_swing_contact_force_and_acceleration_constraint():
    ctl = controller()
    q = ctl.nominal_joint_position.copy()
    q[1] -= .08
    _, d = ctl.compute(q, np.zeros(12), [0.,0.,.464], [1.,0.,0.,0.],
                       np.zeros(3), np.zeros(3), stance_mask=[False,True,True,True])
    np.testing.assert_array_equal(d['forces_world'][0], np.zeros(3))
    np.testing.assert_array_equal(d['contact_acceleration_error'][0], np.zeros(3))
    assert d['support_count'] == 3
    assert d['finite']
    assert d['feasible']


def test_touchdown_relatches_only_the_entering_foot_at_its_actual_position():
    ctl = controller()
    q = ctl.nominal_joint_position.copy()
    inputs = (np.zeros(12), [0.,0.,.464], [1.,0.,0.,0.], np.zeros(3), np.zeros(3))
    ctl.compute(q, *inputs)
    anchors = ctl.contact_anchors.copy()
    q[1] -= .06
    ctl.compute(q, *inputs, stance_mask=[False,True,True,True])
    np.testing.assert_array_equal(ctl.contact_anchors, anchors)
    _, d = ctl.compute(q, *inputs, stance_mask=[True,True,True,True])
    actual = ctl.contact_positions(q, inputs[1], inputs[2])
    np.testing.assert_allclose(ctl.contact_anchors[0], actual[0], atol=1e-12)
    np.testing.assert_array_equal(ctl.contact_anchors[1:], anchors[1:])
    assert np.linalg.norm(ctl.contact_anchors[0] - anchors[0]) > .01
    assert d['anchor_error_m'] < 1e-12
    assert d['support_count'] == 4


def test_default_stance_matches_explicit_four_feet():
    a, b = controller(), controller()
    state = (a.nominal_joint_position, np.zeros(12), [0.,0.,.464],
             [1.,0.,0.,0.], [.03,0.,0.], np.zeros(3))
    ta, da = a.compute(*state)
    tb, db = b.compute(*state, stance_mask=[True]*4)
    np.testing.assert_array_equal(ta, tb)
    np.testing.assert_array_equal(da['forces_world'], db['forces_world'])


def test_no_support_is_reported_without_inventing_ground_forces():
    ctl = controller()
    _, d = ctl.compute(ctl.nominal_joint_position, np.zeros(12), [0.,0.,.464],
                       [1.,0.,0.,0.], np.zeros(3), np.zeros(3), stance_mask=[False]*4)
    np.testing.assert_array_equal(d['forces_world'], np.zeros((4,3)))
    assert d['support_count'] == 0
    assert d['finite'] and d['feasible']
    assert d['acceleration'][2] < -8.


@pytest.mark.parametrize('mask', [[True]*12, [1,1,1,.5], [1,1,1,np.nan]])
def test_invalid_stance_mask_is_rejected(mask):
    ctl = controller()
    with pytest.raises(ValueError):
        ctl.compute(ctl.nominal_joint_position, np.zeros(12), [0.,0.,.464],
                    [1.,0.,0.,0.], np.zeros(3), np.zeros(3), stance_mask=mask)


@pytest.mark.parametrize('drop', [0., .10])
def test_forward_shift_ik_preserves_nominal_contacts(drop):
    ctl = controller()
    q, height = ctl.crouch_pose(drop, forward_shift_m=.03)
    feet = ctl.contact_positions(q, [0., .03, height], [1.,0.,0.,0.])
    nominal = ctl.contact_positions(ctl.nominal_joint_position, [0.,0.,ctl.nominal_height], [1.,0.,0.,0.])
    np.testing.assert_allclose(feet, nominal, atol=2e-6)
    assert height == pytest.approx(.464-drop)
    assert q[1] < ctl.nominal_joint_position[1]-.025


def test_zero_forward_options_preserve_existing_pose_and_torques():
    a,b = controller(),controller()
    qa,ha = a.crouch_pose(.10)
    qb,hb = b.crouch_pose(.10, forward_shift_m=0.)
    np.testing.assert_array_equal(qa,qb)
    assert ha==hb
    inputs = (qa,np.zeros(12),[0.,0.,ha],[1.,0.,0.,0.],np.zeros(3),np.zeros(3))
    ta,da = a.compute(*inputs,height=ha,joint_reference=qa)
    tb,db = b.compute(*inputs,height=ha,joint_reference=qa,forward_offset_m=0.)
    np.testing.assert_array_equal(ta,tb)
    np.testing.assert_array_equal(da['acceleration'],db['acceleration'])


def test_forward_offset_follows_latched_yaw_and_does_not_accumulate():
    a,b = controller(),controller()
    q = a.nominal_joint_position
    quat90 = [np.sqrt(.5),0.,0.,np.sqrt(.5)]
    a.initialize(q,[0.,0.,.464],[1.,0.,0.,0.])
    b.initialize(q,[2.,-1.,.464],quat90)
    for _ in range(2):
        ta,da = a.compute(q,np.zeros(12),[0.,0.,.464],[1.,0.,0.,0.],np.zeros(3),np.zeros(3),forward_offset_m=.03)
        tb,db = b.compute(q,np.zeros(12),[2.,-1.,.464],quat90,np.zeros(3),np.zeros(3),forward_offset_m=.03)
        np.testing.assert_allclose(db['base_position_reference_world'],[1.97,-1.,.464],atol=1e-12)
        np.testing.assert_allclose(ta,tb,atol=1e-8)
        assert da['acceleration'][1] > .05


def test_shifted_static_stand_reduces_rear_knee_torque_with_feasible_wbc():
    baseline,shifted = controller(),controller()
    q,h = shifted.crouch_pose(0.,forward_shift_m=.03)
    shifted.initialize(shifted.nominal_joint_position,[0.,0.,h],[1.,0.,0.,0.])
    for _ in range(3):
        before,old = baseline.compute(baseline.nominal_joint_position,np.zeros(12),[0.,0.,h],
                                      [1.,0.,0.,0.],np.zeros(3),np.zeros(3))
        after,d = shifted.compute(q,np.zeros(12),[0.,.03,h],[1.,0.,0.,0.],np.zeros(3),np.zeros(3),
                                  joint_reference=q,forward_offset_m=.03)
    assert d['finite'] and d['feasible']
    assert d['anchor_error_m'] < 2e-6
    assert np.all(abs(after[[8,11]]) < abs(before[[8,11]])-1.)
    assert np.all(abs(after[[8,11]]) < 18.)
    assert np.max(abs(d['acceleration'][:6])) < .1


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_forward_displacement_rejects_nonfinite_values(value):
    ctl = controller()
    with pytest.raises(ValueError):
        ctl.crouch_pose(0.,forward_shift_m=value)
    with pytest.raises(ValueError):
        ctl.compute(ctl.nominal_joint_position,np.zeros(12),[0.,0.,.464],[1.,0.,0.,0.],
                    np.zeros(3),np.zeros(3),forward_offset_m=value)


def test_explicit_position_reference_overrides_latched_xyz_and_adds_offset_once():
    ctl=controller();q=ctl.nominal_joint_position
    _,d=ctl.compute(q,np.zeros(12),[0.,0.,.464],[1.,0.,0.,0.],np.zeros(3),np.zeros(3),
                    position_reference_world=[.02,.10,.48],forward_offset_m=.03)
    np.testing.assert_allclose(d['base_position_reference_world'],[.02,.13,.48],atol=1e-12)


def test_contact_consistent_reference_keeps_irregular_stance_at_new_height():
    ctl=controller();q=ctl.nominal_joint_position+np.tile([.05,.12,-.1],4)
    position=[.2,.4,.47];quat=[1.,0.,0.,0.];mask=[True,False,True,True]
    ctl.initialize(q,position,quat);anchors=ctl.contact_anchors.copy()
    ref,goal=ctl.contact_consistent_reference(q,position,quat,.464,mask)
    np.testing.assert_allclose(goal,[.2,.4,.464],atol=1e-12)
    feet=ctl.contact_positions(ref,goal,quat)
    np.testing.assert_allclose(feet[np.array(mask)],anchors[np.array(mask)],atol=2e-6)
    assert ctl.contact_positions(ref,position,quat)[1,2] <= np.median(anchors[np.array(mask),2])-.005+1e-7


def test_contact_reference_center_uses_foot_layout_and_shift_only_once():
    ctl=controller();q,h=ctl.crouch_pose(0.,forward_shift_m=.03)
    quat90=[np.sqrt(.5),0.,0.,np.sqrt(.5)]
    ctl.initialize(ctl.nominal_joint_position,[2.,-1.,h],quat90)
    # The robot has moved off the center, but unchanged stored foot anchors
    # define the reference center independently of the current XY.
    ref,goal=ctl.contact_consistent_reference(q,[1.97,-1.,h],quat90,h,[True]*4,
                                              center_body=True,forward_offset_m=.03)
    np.testing.assert_allclose(goal,[2.,-1.,h],atol=1e-12)
    shifted=goal+[-.03,0.,0.]
    np.testing.assert_allclose(ctl.contact_positions(ref,shifted,quat90),ctl.contact_anchors,atol=2e-6)
    np.testing.assert_allclose(ref,q,atol=2e-6)


def test_reference_touchdown_uses_current_foot_before_ik():
    ctl=controller();q=ctl.nominal_joint_position.copy();pos=[0.,0.,.464];quat=[1.,0.,0.,0.]
    ctl.compute(q,np.zeros(12),pos,quat,np.zeros(3),np.zeros(3),stance_mask=[0,1,1,1])
    q[1]-=.05;actual=ctl.contact_positions(q,pos,quat)
    ref,goal=ctl.contact_consistent_reference(q,pos,quat,.464,[1]*4)
    np.testing.assert_allclose(ctl.contact_anchors[0],actual[0],atol=1e-12)
    np.testing.assert_allclose(ctl.contact_positions(ref,goal,quat)[0],actual[0],atol=2e-6)


def test_contact_reference_rejects_unreachable_height():
    ctl=controller();q=ctl.nominal_joint_position
    with pytest.raises(ValueError,match='IK'):
        ctl.contact_consistent_reference(q,[0.,0.,.464],[1.,0.,0.,0.],2.,[1]*4)


def test_position_override_rejects_nonfinite():
    ctl=controller()
    with pytest.raises(ValueError):
        ctl.compute(ctl.nominal_joint_position,np.zeros(12),[0.,0.,.464],[1.,0.,0.,0.],np.zeros(3),np.zeros(3),
                    position_reference_world=[0.,float('nan'),.464])


def test_initial_swing_touchdown_before_first_compute_relatches_anchor():
    ctl=controller();q=ctl.nominal_joint_position.copy();pos=[0.,0.,.464];quat=[1.,0.,0.,0.]
    ctl.initialize(q,pos,quat,stance_mask=[0,1,1,1])
    q[1]-=.05;actual=ctl.contact_positions(q,pos,quat)
    ref,goal=ctl.contact_consistent_reference(q,pos,quat,.464,[1]*4)
    np.testing.assert_allclose(ctl.contact_anchors[0],actual[0],atol=1e-12)
    _,diag=ctl.compute(q,np.zeros(12),pos,quat,np.zeros(3),np.zeros(3),
                       stance_mask=[1]*4,joint_reference=ref,position_reference_world=goal)
    assert diag['anchor_error_m'] < 1e-12


def test_initialize_rejects_nonbinary_stance():
    ctl=controller()
    with pytest.raises(ValueError,match='binary'):
        ctl.initialize(ctl.nominal_joint_position,[0.,0.,.464],[1.,0.,0.,0.],stance_mask=[0,1,2,1])


def test_contact_reference_center_fraction_interpolates_current_and_center():
    ctl=controller();q=ctl.nominal_joint_position;quat=[1.,0.,0.,0.]
    ctl.initialize(q,[0.,0.,.464],quat)
    goals=[]
    for fraction in (0.,.5,1.):
        ref,goal=ctl.contact_consistent_reference(q,[.02,.04,.464],quat,.464,[1]*4,center_body=fraction)
        goals.append(goal)
        np.testing.assert_allclose(ctl.contact_positions(ref,goal,quat),ctl.contact_anchors,atol=2e-6)
    np.testing.assert_allclose(goals[0],[.02,.04,.464],atol=1e-12)
    np.testing.assert_allclose(goals[1],[.01,.02,.464],atol=1e-12)
    np.testing.assert_allclose(goals[2],[0.,0.,.464],atol=1e-12)


@pytest.mark.parametrize('fraction',[-.1,1.1,float('nan'),float('inf')])
def test_contact_reference_rejects_invalid_center_fraction(fraction):
    ctl=controller()
    with pytest.raises(ValueError,match='center_body'):
        ctl.contact_consistent_reference(ctl.nominal_joint_position,[0.,0.,.464],[1.,0.,0.,0.],
                                         .464,[1]*4,center_body=fraction)


def test_air_foot_landing_uses_measured_tilt_and_preserves_stance_anchors():
    ctl=controller();q=ctl.crouch_pose(0.,.03)[0];pos=[.2,.4,.462]
    angle=np.deg2rad(.85);quat=[np.cos(angle/2),np.sin(angle/2),0.,0.];mask=np.array([1,1,1,0],bool)
    ctl.initialize(q,pos,quat,stance_mask=mask);anchors=ctl.contact_anchors.copy()
    ref,goal=ctl.contact_consistent_reference(q,pos,quat,.464,mask,center_body=1.,
                                             forward_offset_m=.03,landing_depth_m=.005)
    actual_pose_feet=ctl.contact_positions(ref,pos,quat)
    plane=np.median(anchors[mask,2])
    assert actual_pose_feet[3,2] == pytest.approx(plane-.005,abs=1e-7)
    nominal_xy=ctl.contact_positions(q,pos,quat)[3,:2]
    np.testing.assert_allclose(actual_pose_feet[3,:2],nominal_xy,atol=1e-7)
    desired_pose_feet=ctl.contact_positions(ref,goal+[0.,.03,0.],[1.,0.,0.,0.])
    np.testing.assert_allclose(desired_pose_feet[mask],anchors[mask],atol=1e-7)
    np.testing.assert_array_equal(ctl.contact_anchors,anchors)


def test_air_foot_landing_retains_last_support_plane_when_support_unknown():
    ctl=controller();q=ctl.nominal_joint_position;quat=[1.,0.,0.,0.]
    ctl.initialize(q,[0.,0.,.464],quat,stance_mask=[1,1,1,0])
    ref,goal=ctl.contact_consistent_reference(q,[0.,0.,.48],quat,.464,[0]*4,landing_depth_m=.005)
    np.testing.assert_allclose(ctl.contact_positions(ref,[0.,0.,.48],quat)[:,2],-.005,atol=1e-7)


def test_air_foot_landing_rejects_unreachable_measured_pose():
    ctl=controller();q=ctl.nominal_joint_position;quat=[1.,0.,0.,0.]
    ctl.initialize(q,[0.,0.,.464],quat,stance_mask=[1,1,1,0])
    with pytest.raises(ValueError,match='Landing IK unreachable'):
        ctl.contact_consistent_reference(q,[0.,0.,2.],quat,.464,[0]*4,landing_depth_m=.005)


@pytest.mark.parametrize('depth',[-.001,.021,float('nan'),float('inf')])
def test_landing_depth_is_finite_and_bounded(depth):
    ctl=controller()
    with pytest.raises(ValueError,match='landing_depth_m'):
        ctl.contact_consistent_reference(ctl.nominal_joint_position,[0.,0.,.464],[1.,0.,0.,0.],
                                         .464,[1]*4,landing_depth_m=depth)
