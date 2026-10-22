"""Checks the CAD dynamics boundary against independently audited USD data."""
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'config'), str(ROOT / 'ros2/v151_ros2_control')]
AUDIT = ROOT / 'reports/v15/usd_20261011/audit/usd_cpu_audit.json'


def derivation():
    path = ROOT / 'config/derive_v15_cad_model.py'
    assert path.exists(), 'The CAD model derivation has not been implemented'
    spec = importlib.util.spec_from_file_location('derive_v15_cad_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_model_retains_each_authored_mass_and_full_inertia(tmp_path):
    # Catches inertia rescaling, a 45 kg source reuse, and invented foot masses.
    paths = derivation().generate_model(AUDIT, tmp_path)
    root = ET.parse(paths['mjcf']).getroot()
    bodies = root.findall('.//body')
    source = json.loads(AUDIT.read_text())['new']['bodies']
    assert len(bodies) == 13
    assert sum(float(b.find('inertial').get('mass')) for b in bodies) == pytest.approx(23.09449815750122, abs=1e-12)
    for body in bodies:
        inertial, authored = body.find('inertial'), source['/quadruped/' + body.get('name')]
        assert float(inertial.get('mass')) == authored['mass_kg']
        np.testing.assert_allclose(np.fromstring(inertial.get('pos'), sep=' '), authored['center_of_mass_local_m'], atol=1e-15)
        xx, yy, zz, xy, xz, yz = np.fromstring(inertial.get('fullinertia'), sep=' ')
        tensor = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        np.testing.assert_allclose(tensor, authored['inertia_tensor_local_kg_m2'], atol=1e-14)
    assert not root.findall('.//geom'), 'The dynamics-only MJCF must not invent simulation colliders'


def test_nonidentity_cad_link_frames_are_rejected_instead_of_guessed(tmp_path):
    # Common-frame anchors and COMs are valid only for the audited identity links.
    audit = json.loads(AUDIT.read_text())
    audit['new']['bodies']['/quadruped/FL_thigh']['world_transform'][0][3] = .01
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='identity'):
        derivation().generate_model(bad, tmp_path / 'bad_model')


@pytest.mark.skipif(importlib.util.find_spec('pinocchio') is None, reason='Pinocchio checks run on the CPU workstation')
def test_canonical_foot_frames_match_independent_usd_nominal_fk(tmp_path):
    # Catches raw-sign errors and baked common-frame anchors interpreted as local offsets.
    paths = derivation().generate_model(AUDIT, tmp_path)
    from v151_ros2_control.cad_standing import CadStandingController
    ctl = CadStandingController(paths['mjcf'])
    expected = [
        [-.16988098740607477, .3266422341961812, -.43873935455585256],
        [.1698190020938836, .326909111661024, -.43858931399062073],
        [-.169880990406076, -.30336176705553003, -.4387393635153982],
        [.16981899479391474, -.30309889239715687, -.4385893273588514],
    ]
    actual = [ctl.model.frame_position(ctl.nominal_joint_position, name,
        base_position=[0, 0, 0], base_quaternion_xyzw=[0, 0, 0, 1]) for name in ctl.foot_names]
    np.testing.assert_allclose(actual, expected, atol=2e-10)
    assert ctl.model.body_count == 13
    assert ctl.model.total_mass == pytest.approx(23.09449815750122, abs=1e-12)


@pytest.mark.skipif(importlib.util.find_spec('pinocchio') is None, reason='Pinocchio checks run on the CPU workstation')
@pytest.mark.parametrize('drop', [.10, .18])
def test_cad_static_support_and_crouch_use_real_mass_and_preserve_contacts(tmp_path, drop):
    paths = derivation().generate_model(AUDIT, tmp_path)
    from v151_ros2_control.cad_standing import CadStandingController
    ctl = CadStandingController(paths['mjcf'])
    anchors = ctl.contact_positions(ctl.nominal_joint_position, [0, 0, ctl.nominal_height], [1, 0, 0, 0])
    q, height = ctl.crouch_pose(drop)
    np.testing.assert_allclose(ctl.contact_positions(q, [0, 0, height], [1, 0, 0, 0]), anchors, atol=2e-7)
    for position, z in [(ctl.nominal_joint_position, ctl.nominal_height), (q, height)]:
        ctl.initialize(position, [0, 0, z], [1, 0, 0, 0])
        tau, diag = ctl.compute(position, np.zeros(12), [0, 0, z], [1, 0, 0, 0],
            np.zeros(3), np.zeros(3), height=z, joint_reference=position)
        assert diag['finite'] and diag['feasible']
        assert np.max(np.abs(diag['acceleration'][:6])) < .1
        assert abs(diag['forces_world'][:, 2].sum() - 226.557026925087) < 2.
        assert np.all(np.abs(tau) <= np.tile([22.32, 43.2, 43.2], 4) + 1e-8)
        assert np.all(np.linalg.norm(diag['forces_world'][:, :2], axis=1) <= .3 * diag['forces_world'][:, 2] + 1e-4)


@pytest.mark.skipif(importlib.util.find_spec('pinocchio') is None or importlib.util.find_spec('pxr') is None,
    reason='CPU USD and Pinocchio geometry checks run on the workstation')
def test_authored_hull_screen_detects_crossing_feet_and_accepts_crouch(tmp_path):
    # Catches an empty-result screen and failure to transform the CAD hulls.
    module = derivation()
    paths = module.generate_model(AUDIT, tmp_path, ROOT / 'assets/quadruped_20261011/quadruped_robot.usda')
    from v151_ros2_control.cad_standing import CADStandingController
    ctl = CADStandingController(paths['mjcf'])
    deep, _ = ctl.crouch_pose(.18)
    assert module.collision_screen(ctl, paths['collision_hulls'], deep)['candidate_count'] == 0
    crossing = ctl.nominal_joint_position.copy()
    crossing[::3] = [-.3, .3, -.3, .3]
    intersections = module.collision_screen(ctl, paths['collision_hulls'], crossing)['overlap_candidates']
    assert len(intersections) >= 2
    assert max(item['interior_radius_m'] for item in intersections) > .005


@pytest.mark.skipif(importlib.util.find_spec('pinocchio') is None, reason='Pinocchio checks run on the CPU workstation')
def test_recorded_handover_lands_at_plane_but_rejects_five_mm_penetration(tmp_path):
    # Catches restoring the below-plane CAD default, a relaxed IK gate, or
    # a nominal-pose fallback that fails to reach the recorded landing points.
    fixture = json.loads((Path(__file__).parent / 'fixtures/v15_cad_landing_2320.json').read_text())
    paths = derivation().generate_model(AUDIT, tmp_path)
    from v151_ros2_control.cad_standing import CadStandingController

    def measured_controller():
        ctl = CadStandingController(paths['mjcf'])
        saved = fixture['latched_state']
        ctl.contact_anchors = np.asarray(saved['contact_anchors'], float)
        ctl._previous_stance = np.asarray(saved['previous_stance'], bool)
        ctl._target_rotation = np.asarray(saved['target_rotation'], float)
        ctl._support_plane_z = float(saved['support_plane_z'])
        return ctl

    ctl = measured_controller()
    with pytest.raises(ValueError, match=r'Landing IK unreachable for FR_foot: residual 0\.00205351 m'):
        ctl.contact_consistent_reference(**fixture['inputs'], landing_depth_m=.005)

    ctl = measured_controller()
    qref, goal = ctl.contact_consistent_reference(**fixture['inputs'])
    inputs = fixture['inputs']
    unsupported = ~np.asarray(inputs['stance_mask'], bool)
    actual = ctl.contact_positions(qref, inputs['base_pos'], inputs['quat_wxyz'])
    expected = np.asarray(fixture['expected_landing_targets_world_m'])
    np.testing.assert_allclose(actual[unsupported], expected[unsupported], atol=1e-7, rtol=0)
    assert goal[2] == pytest.approx(inputs['height'], abs=1e-12)
