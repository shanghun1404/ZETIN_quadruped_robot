"""Derive the 2026-10-11 CAD dynamics from its read-only USD audit.

The output MJCF is solely an explicit-inertia interchange for Pinocchio.
Isaac must still load the source USD with its original collision meshes.
Optional USD geometry checks use CPU pxr/scipy and never SimulationApp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

JOINT_NAMES = tuple(f'{leg}_{joint}' for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('HAA', 'HFE', 'KFE'))
NOMINAL = np.tile([0., -.55, 1.10], 4)
LIMITS = np.tile([24.8, 48., 48.], 4)


def _numbers(values):
    return ' '.join(format(float(value), '.17g') for value in values)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def generate_model(audit_path, output, usd_path=None):
    audit_path, output = Path(audit_path), Path(output)
    audit = json.loads(audit_path.read_text())
    source = audit['new']
    if source['meters_per_unit'] != 1 or source['up_axis'] != 'Z':
        raise ValueError('The CAD model must use meters and Z up')
    if source['rigid_body_count'] != 13 or source['revolute_joint_count'] != 12 or source['fixed_joint_count']:
        raise ValueError('Expected the audited 13-body, 12-hinge CAD model')
    for body in source['bodies'].values():
        if not np.allclose(body['world_transform'], np.eye(4), rtol=0, atol=1e-12):
            raise ValueError('CAD derivation requires identity zero-pose body frames')
        inertia = np.asarray(body['inertia_tensor_local_kg_m2'])
        eig = np.linalg.eigvalsh(inertia)
        if body['mass_kg'] <= 0 or eig.min() <= 0 or eig.max() > eig.sum() - eig.max() + 1e-9:
            raise ValueError('Authored CAD mass or inertia is nonphysical')
    if usd_path and hashlib.sha256(Path(usd_path).read_bytes()).hexdigest() != source['root_layer_sha256']:
        raise ValueError('USD root layer differs from the audited source')
    mapping = audit['comparison']['mapping']
    if tuple(item['canonical'] for item in mapping) != JOINT_NAMES:
        raise ValueError('Audit mapping must explicitly name all canonical joints in order')
    joints_by_name = {item['name']: item for item in source['joints'].values()}
    by_child = {}
    for index, item in enumerate(mapping):
        joint = joints_by_name[item['native']]
        if joint['collision_enabled']:
            raise ValueError('CAD collision screen requires the audited adjacent-pair collision disable flags')
        if joint['authored_zero_position_error_m'] > 1e-8 or joint['authored_zero_rotation_error_rad'] > 1e-6:
            raise ValueError('Joint zero frames do not coincide')
        axis = np.asarray(joint['axis_world']) * item['native_sign']
        axis /= np.linalg.norm(axis)
        lo, hi = np.deg2rad([joint['lower_limit_deg'], joint['upper_limit_deg']]) * item['native_sign']
        by_child[joint['body1']] = dict(canonical=item['canonical'], native=item['native'],
            sign=item['native_sign'], axis=axis, anchor=joint['zero_pivot_world_m'],
            limits=sorted([lo, hi]), parent=joint['body0'], index=index)
    nominal_feet = audit['comparison']['new_stand_fk']['feet']
    nominal_height = max(-float(foot['mesh_vertex_min_root_m'][2]) for foot in nominal_feet.values())
    root = ET.Element('mujoco', model='quadruped_cad_20261011_dynamics_only')
    ET.SubElement(root, 'compiler', angle='radian', coordinate='local', inertiafromgeom='false')
    ET.SubElement(root, 'option', gravity='0 0 -9.81', timestep='0.005')
    world = ET.SubElement(root, 'worldbody')
    traversal = []

    def append_body(path, parent):
        record = source['bodies'][path]
        attrs = dict(name=record['name'], pos='0 0 0')
        if path == source['root_bodies'][0]:
            attrs['pos'] = _numbers([0, 0, nominal_height])
        body = ET.SubElement(parent, 'body', **attrs)
        if path in by_child:
            joint = by_child[path]
            ET.SubElement(body, 'joint', name=joint['canonical'], type='hinge',
                pos=_numbers(joint['anchor']), axis=_numbers(joint['axis']),
                range=_numbers(joint['limits']), damping='0', armature='0.02')
            traversal.append(joint['canonical'])
        else:
            ET.SubElement(body, 'freejoint', name='root')
        inertia = np.asarray(record['inertia_tensor_local_kg_m2'])
        ET.SubElement(body, 'inertial', mass=format(record['mass_kg'], '.17g'),
            pos=_numbers(record['center_of_mass_local_m']),
            fullinertia=_numbers(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
        for child, joint in sorted(by_child.items(), key=lambda pair: pair[1]['index']):
            if joint['parent'] == path:
                append_body(child, body)

    append_body(source['root_bodies'][0], world)
    actuator = ET.SubElement(root, 'actuator')
    for name, limit in zip(JOINT_NAMES, LIMITS):
        ET.SubElement(actuator, 'motor', name=name, joint=name, gear='1 0 0 0 0 0',
            ctrllimited='true', ctrlrange=_numbers([-limit, limit]))
    values = dict(zip(JOINT_NAMES, NOMINAL))
    ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', name='stand',
        qpos=_numbers([0, 0, nominal_height, 1, 0, 0, 0] + [values[name] for name in traversal]))
    output.mkdir(parents=True, exist_ok=True)
    mjcf = output / 'quadruped_cad.xml'
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(mjcf, encoding='unicode')
    metadata = dict(schema=1, source_usd_root_sha256=source['root_layer_sha256'],
        source_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        source_total_mass_kg=source['total_mass_kg'], physical_body_count=13,
        canonical_joint_names=list(JOINT_NAMES), canonical_nominal_q=NOMINAL.tolist(),
        nominal_root_height_m=nominal_height, base_com_local_m=source['bodies']['/quadruped/base']['center_of_mass_local_m'],
        effort_limits_Nm=LIMITS.tolist(), rotor_armature_kg_m2=.02, viscous_damping=0.,
        runtime_position_pd=dict(stiffness_Nm_per_rad=300., damping_Nm_s_per_rad=8.),
        runtime_joint_friction=.3, wbc_friction_coefficient=.3,
        dynamics_mjcf_sha256=hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        model_scope='CAD-authored inertias and anchors, dynamics only; source USD remains collision authority',
        contact_scope='massless kinematic foot frames at tennis mesh AABB centers; fixed material points projected to nominal ground',
        joint_mapping=mapping, foot_frames={})
    for canonical, native in [('FL', 'FL'), ('FR', 'FR'), ('RL', 'BL'), ('RR', 'BR')]:
        foot = source['feet'][native]
        metadata['foot_frames'][canonical + '_foot'] = dict(parent_body=foot['body'].split('/')[-1],
            center_body_m=foot['center_root_m'], collider=foot['collider'],
            audited_nominal_center_root_m=nominal_feet[native]['center_root_m'],
            audited_nominal_min_root_m=nominal_feet[native]['mesh_vertex_min_root_m'])
    meta = output / 'cad_model.json'
    _write_json(meta, metadata)
    paths = dict(mjcf=str(mjcf), metadata=str(meta))
    if usd_path:
        paths['collision_hulls'] = extract_collision_hulls(usd_path, source, output / 'collision_hulls.npz')
    return paths


def extract_collision_hulls(usd_path, source, output):
    """Preserve each authored convexHull mesh in its actual owner's frame."""
    from pxr import Usd, UsdGeom
    from scipy.spatial import ConvexHull
    stage = Usd.Stage.Open(str(usd_path))
    transforms = UsdGeom.XformCache()
    arrays, records = {}, []
    for collider in source['colliders']:
        if not collider['enabled']:
            continue
        if collider['type'] != 'Mesh' or collider['approximation'] != 'convexHull':
            raise ValueError('CPU collision screen currently requires authored convexHull meshes')
        prim = stage.GetPrimAtPath(collider['path'])
        vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        mesh_world = np.asarray(transforms.GetLocalToWorldTransform(prim), float).T
        body_world = np.asarray(source['bodies'][collider['owner']]['world_transform'])
        transform = np.linalg.inv(body_world) @ mesh_world
        points = (transform @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]
        hull = ConvexHull(points)
        index = len(records)
        arrays[f'vertices_{index}'] = points[hull.vertices]
        # Round removes repeated coplanar triangle planes without moving vertices.
        arrays[f'planes_{index}'] = np.unique(np.round(hull.equations, 12), axis=0)
        records.append(dict(path=collider['path'], owner=collider['owner'].split('/')[-1], approximation='convexHull'))
    arrays['records_json'] = np.asarray(json.dumps(records))
    arrays['usd_sha256'] = np.asarray(hashlib.sha256(Path(usd_path).read_bytes()).hexdigest())
    np.savez_compressed(output, **arrays)
    return str(output)


def collision_screen(controller, hull_path, joint_position):
    """Screen interior overlap using authored convex hulls and a CPU LP.

    Directly connected bodies are excluded exactly as the USD's joint flags.
    A positive LP interior radius is a collision candidate; no contact offset,
    mesh cooking tolerance, or dynamic collision response is inferred.
    """
    from scipy.optimize import linprog
    hulls = np.load(hull_path, allow_pickle=False)
    records = json.loads(str(hulls['records_json']))
    model = controller.model
    q = model.configuration(joint_position, base_position=[0, 0, 0], base_quaternion_xyzw=[0, 0, 0, 1])
    model.pin.forwardKinematics(model.model, model.data, q)
    model.pin.updateFramePlacements(model.model, model.data)
    shapes = []
    for index, record in enumerate(records):
        pose = model.data.oMf[model.model.getFrameId(record['owner'])]
        points = hulls[f'vertices_{index}'] @ pose.rotation.T + pose.translation
        equations = hulls[f'planes_{index}']
        normal = equations[:, :3] @ pose.rotation.T
        offset = equations[:, 3] - normal @ pose.translation
        shapes.append((points.min(0), points.max(0), normal, offset))
    adjacent = {frozenset(('base', leg + '_hip')) for leg in ('FL', 'FR', 'BL', 'BR')}
    adjacent |= {frozenset((leg + '_hip', leg + '_thigh')) for leg in ('FL', 'FR', 'BL', 'BR')}
    adjacent |= {frozenset((leg + '_thigh', leg + '_shank')) for leg in ('FL', 'FR', 'BL', 'BR')}
    overlap = []
    for a, first in enumerate(records):
        for b in range(a + 1, len(records)):
            second = records[b]
            owners = frozenset((first['owner'], second['owner']))
            if len(owners) == 1 or owners in adjacent:
                continue
            lo_a, hi_a, normal_a, offset_a = shapes[a]
            lo_b, hi_b, normal_b, offset_b = shapes[b]
            if np.any(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) < 1e-7):
                continue
            normal = np.vstack([normal_a, normal_b])
            rhs = -np.r_[offset_a, offset_b]
            result = linprog([0, 0, 0, -1], A_ub=np.c_[normal, np.ones(len(normal))],
                b_ub=rhs, bounds=[(None, None)] * 4, method='highs')
            if not result.success:
                raise RuntimeError(f'Convex-hull intersection LP failed: {result.message}')
            if result.x[3] > 1e-6:
                overlap.append(dict(first=first['path'], second=second['path'],
                    interior_radius_m=float(result.x[3])))
    return dict(overlap_candidates=overlap, candidate_count=len(overlap),
        scope='CPU authored convexHull interior intersection; adjacent joint pairs excluded by USD collisionEnabled=false')


def _pose_geometry(controller, hull_path, joint_position, root_height):
    hulls = np.load(hull_path, allow_pickle=False)
    records = json.loads(str(hulls['records_json']))
    model = controller.model
    config = model.configuration(joint_position, base_position=[0, 0, root_height], base_quaternion_xyzw=[0, 0, 0, 1])
    model.pin.forwardKinematics(model.model, model.data, config)
    model.pin.updateFramePlacements(model.model, model.data)
    foot_by_path = {item['collider']: name for name, item in controller.metadata['foot_frames'].items()}
    feet, other_min_z = {}, np.inf
    for index, record in enumerate(records):
        pose = model.data.oMf[model.model.getFrameId(record['owner'])]
        vertices = hulls[f'vertices_{index}'] @ pose.rotation.T + pose.translation
        if record['path'] in foot_by_path:
            name = foot_by_path[record['path']]
            feet[name] = dict(center_world_m=model.frame_position(joint_position, name,
                base_position=[0, 0, root_height], base_quaternion_xyzw=[0, 0, 0, 1]).tolist(),
                hull_min_world_m=vertices.min(0).tolist(), hull_max_world_m=vertices.max(0).tolist(),
                material_contact_world_m=model.frame_position(joint_position, name,
                    controller.contact_local[controller.foot_names.index(name)],
                    base_position=[0, 0, root_height], base_quaternion_xyzw=[0, 0, 0, 1]).tolist())
        else:
            other_min_z = min(other_min_z, float(vertices[:, 2].min()))
    return dict(feet=feet, lowest_foot_hull_world_z_m=min(item['hull_min_world_m'][2] for item in feet.values()),
        lowest_nonfoot_hull_world_z_m=other_min_z,
        contact_note='Fixed material contact points preserve kinematics; actual tennis hull lowest points move with shank rotation. Spawn clearance is applied by the caller.')


def verify_model(paths, output, crouch_drop=.10):
    from v151_ros2_control.cad_standing import CadStandingController
    controller = CadStandingController(paths['mjcf'])
    model, nominal = controller.model, controller.nominal_joint_position
    metadata = json.loads(Path(paths['metadata']).read_text())
    base0 = dict(base_position=[0, 0, 0], base_quaternion_xyzw=[0, 0, 0, 1])
    nominal_fk = np.array([model.frame_position(nominal, name, **base0) for name in controller.foot_names])
    audited_fk = np.array([metadata['foot_frames'][name]['audited_nominal_center_root_m'] for name in controller.foot_names])
    mass = model.mass_matrix(nominal, **base0)
    rng = np.random.default_rng(20261011)
    velocity, acceleration = rng.normal(0, .1, (2, model.model.nv))
    effort = model.inverse_dynamics(nominal, velocity, acceleration, include_damping=False, **base0)
    recovered = model.forward_dynamics(nominal, velocity, effort, include_damping=False, **base0)
    checks = dict(scope='CPU CAD kinematics, rigid-body dynamics, static WBC and convex-hull screen; no PhysX stepping',
        model_manifest=model.manifest(),
        nominal_foot_fk_max_abs_error_m=float(np.max(np.abs(nominal_fk - audited_fk))),
        mass_matrix_min_eigenvalue=float(np.linalg.eigvalsh(mass).min()),
        inverse_forward_dynamics_max_abs_error=float(np.max(np.abs(recovered - acceleration))), poses={})
    epsilon = 1e-6
    max_jac_error = 0.
    for name in controller.foot_names:
        analytic = model.point_jacobian(nominal, name, **base0)
        finite = np.column_stack([(model.point_after_integration(nominal, np.eye(18)[j] * epsilon, name, **base0) -
            model.point_after_integration(nominal, -np.eye(18)[j] * epsilon, name, **base0)) / (2 * epsilon) for j in range(18)])
        max_jac_error = max(max_jac_error, float(np.max(np.abs(analytic - finite))))
    checks['foot_point_jacobian_fd_max_abs_error'] = max_jac_error
    qc, hc = controller.crouch_pose(crouch_drop)
    qdeep, hdeep = controller.crouch_pose(.18)
    for label, q, h in [('stand', nominal, controller.nominal_height), ('standard_crouch', qc, hc), ('deep_crouch', qdeep, hdeep)]:
        controller.initialize(q, [0, 0, h], [1, 0, 0, 0])
        torque, diagnostics = controller.compute(q, np.zeros(12), [0, 0, h], [1, 0, 0, 0],
            np.zeros(3), np.zeros(3), height=h, joint_reference=q)
        pose = dict(canonical_joint_position_rad=q.tolist(), root_height_m=h,
            finite=bool(diagnostics['finite']), numerical_wbc_feasible=bool(diagnostics['feasible']),
            max_base_acceleration=float(np.max(np.abs(diagnostics['acceleration'][:6]))),
            forces_world_N=diagnostics['forces_world'].tolist(), torque_Nm=torque.tolist(),
            weight_N=float(model.total_mass * 9.81), force_sum_z_N=float(diagnostics['forces_world'][:, 2].sum()),
            max_contact_acceleration_error=float(np.max(np.abs(diagnostics['contact_acceleration_error']))))
        if 'collision_hulls' in paths:
            pose['collision_screen'] = collision_screen(controller, paths['collision_hulls'], q)
            pose['physical_geometry'] = _pose_geometry(controller, paths['collision_hulls'], q, h)
        checks['poses'][label] = pose
    # Screen the whole sit/rise reference path, not just the endpoints.
    if 'collision_hulls' in paths:
        trajectory = []
        for fraction in np.linspace(0, 1, 13):
            qref, height = controller.crouch_pose(.18 * (1 - fraction))
            screen = collision_screen(controller, paths['collision_hulls'], qref)
            trajectory.append(dict(rise_fraction=float(fraction), root_height_m=height, **screen))
        checks['sit_rise_collision_path'] = trajectory
    crouch = dict(definition='upright all-feet crouch; root drops 0.10m for standard crouch and 0.18m for deep crouch, with four nominal material foot points held. The trunk remains elevated; this is not a trunk-on-ground seated posture.',
        height_drop_m=crouch_drop, height_m=hc, root_height_m=hc,
        q_canonical_rad=qc.tolist(), q=qc.tolist(), stand_root_height_m=controller.nominal_height,
        stand_q_canonical_rad=nominal.tolist(), transition_duration_s=3.,
        source_model_sha256=model.source_sha256, physics_validation='not yet performed',
        recommended_spawn_clearance_m=.006,
        cpu_static_support=checks['poses']['standard_crouch'],
        standard_crouch=dict(height_drop_m=crouch_drop, root_height_m=hc, q_canonical_rad=qc.tolist(),
            cpu_static_support=checks['poses']['standard_crouch']),
        deep_crouch=dict(height_drop_m=.18, root_height_m=hdeep, q_canonical_rad=qdeep.tolist(),
            cpu_static_support=checks['poses']['deep_crouch']))
    _write_json(Path(output) / 'verification.json', checks)
    _write_json(Path(output) / 'crouch.json', crouch)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--usd')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--crouch-drop', type=float, default=.10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'config'), str(root / 'ros2/v151_ros2_control')]
    paths = generate_model(args.audit, args.output, args.usd)
    if args.verify:
        checks = verify_model(paths, args.output, args.crouch_drop)
        print(json.dumps(dict(paths=paths, mass_kg=checks['model_manifest']['total_mass_kg'],
            nominal_fk_error_m=checks['nominal_foot_fk_max_abs_error_m'],
            poses={name: dict(feasible=value['numerical_wbc_feasible'],
                collision_candidates=value.get('collision_screen', {}).get('candidate_count')) for name, value in checks['poses'].items()})))
    else:
        print(json.dumps(paths))


if __name__ == '__main__':
    main()
