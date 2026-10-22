"""Real-model numerical checks; requires the optional Pinocchio model runtime."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'config'), str(ROOT / 'ros2/v151_ros2_control')]


def controller():
    from v151_ros2_control.standing import StandingController
    return StandingController(ROOT / 'reports/v16/physical_model/v16_model_source_quadruped.xml')


def test_crouch_preserves_four_support_points_and_requested_drop():
    # Catches wrong joint order, root reference, and IK sign.
    ctl = controller()
    q, height = ctl.crouch_pose(.10)
    assert height == pytest.approx(ctl.nominal_height - .10)
    feet = ctl.contact_positions(q, [0., 0., height], [1., 0., 0., 0.])
    nominal = ctl.contact_positions(ctl.nominal_joint_position, [0., 0., ctl.nominal_height], [1., 0., 0., 0.])
    np.testing.assert_allclose(feet, nominal, atol=2e-6)
    assert np.max(np.abs(q - ctl.nominal_joint_position)) > .2


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
