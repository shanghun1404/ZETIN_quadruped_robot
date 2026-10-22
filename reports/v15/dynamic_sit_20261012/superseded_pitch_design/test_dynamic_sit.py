"""Posture geometry and readiness checks; physical validation is separate."""
from pathlib import Path
import sys
import importlib.util
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'config'), str(ROOT / 'ros2/v151_ros2_control')]


def posture():
    from v151_ros2_control import posture
    return posture


def test_sitting_shifts_rearward_before_finishing_lowering_and_pitch():
    p = posture()
    stand = p.dynamic_sit_reference(1., .467, .14)
    seat = p.dynamic_sit_reference(0., .467, .14)
    assert stand == dict(height=.467, pitch_rad=0., forward_offset_m=0.)
    assert seat['height'] == pytest.approx(.327)
    assert seat['pitch_rad'] == pytest.approx(.4)
    assert seat['forward_offset_m'] == pytest.approx(-.04)
    mid = p.dynamic_sit_reference(.5, .467, .14)
    assert abs(mid['forward_offset_m']) > .02
    assert mid['pitch_rad'] == pytest.approx(.2)
    samples = [p.dynamic_sit_reference(f, .467, .14) for f in np.linspace(1, 0, 101)]
    assert np.all(np.diff([s['height'] for s in samples]) < 0)
    assert np.all(np.diff([s['pitch_rad'] for s in samples]) >= 0)


def test_seated_readiness_requires_target_pitch_actual_support_and_rest():
    p = posture()
    pitch = .4
    gravity = [0., -np.sin(pitch), -np.cos(pitch)]
    args = dict(gravity=gravity, velocity=np.zeros(3), angular_velocity=np.zeros(3),
                support_force=np.full(4, 55.), height=.327, target_height=.327, pitch_rad=pitch)
    assert p.posture_ready(**args)
    assert not p.posture_ready(**dict(args, gravity=[0., 0., -1.]))
    assert not p.posture_ready(**dict(args, support_force=[55., 55., 0., 0.]))
    assert not p.posture_ready(**dict(args, velocity=[0., .2, 0.]))
    assert not p.posture_ready(**dict(args, height=.40))
    assert not p.posture_ready(**dict(args, gravity=[0., np.nan, -1.]))


@pytest.mark.skipif(importlib.util.find_spec('pinocchio') is None, reason='CAD IK runs on ws1 CPU')
def test_pitched_contact_reference_preserves_feet_and_folds_rear_legs_more():
    from v151_ros2_control.cad_standing import CadStandingController
    ctl = CadStandingController(ROOT / 'config/experiments/v15_cad_20261011/model/quadruped_cad.xml')
    q = ctl.nominal_joint_position.copy()
    base = np.array([0., 0., ctl.nominal_height])
    ctl.initialize(q, base, [1., 0., 0., 0.])
    anchors = ctl.contact_anchors.copy()
    for fraction in np.linspace(1., 0., 21):
        ref = posture().dynamic_sit_reference(fraction, ctl.nominal_height, .14)
        q, goal = ctl.contact_consistent_reference(q, base, [1., 0., 0., 0.], ref['height'],
            np.ones(4, dtype=bool), center_body=True, forward_offset_m=ref['forward_offset_m'],
            pitch_reference_rad=ref['pitch_rad'])
        quat = [np.cos(ref['pitch_rad']/2), np.sin(ref['pitch_rad']/2), 0., 0.]
        actual = ctl.contact_positions(q, goal+[0., ref['forward_offset_m'], 0.], quat)
        np.testing.assert_allclose(actual, anchors, atol=2e-7)
        assert np.all(np.abs(q.reshape(4, 3)) <= [np.pi/4, 2*np.pi/3, 7*np.pi/9])
    assert q[[8, 11]].mean() - q[[2, 5]].mean() > 1.
    # Static hold starts at the seated reference, without a spurious one-tick
    # reference velocity from the nominal standing configuration.
    ctl.initialize(q, goal+[0., ref['forward_offset_m'], 0.], quat)
    tau, diag = ctl.compute(q, np.zeros(12), goal+[0., ref['forward_offset_m'], 0.], quat,
        np.zeros(3), np.zeros(3), height=ref['height'], joint_reference=q,
        position_reference_world=goal, forward_offset_m=ref['forward_offset_m'],
        pitch_reference_rad=ref['pitch_rad'])
    assert diag['finite'] and diag['feasible']
    assert np.linalg.norm(diag['acceleration'][3:6]) < .15
    assert np.all(abs(tau) <= .9*ctl.limits+1e-8)
