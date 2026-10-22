"""Replay recorded joint poses through CAD hull geometry, without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


CASES = [('2318', 'stairs_up_10cm'), ('2318', 'stairs_down_10cm'), ('2319', 'stairs_up_157mm')]


def transformed_hulls(model, hulls, records, joints):
    model.pin.forwardKinematics(model.model, model.data, model.configuration(joints,
        base_position=[0, 0, 0], base_quaternion_xyzw=[0, 0, 0, 1]))
    model.pin.updateFramePlacements(model.model, model.data)
    result = []
    for index, record in enumerate(records):
        pose = model.data.oMf[model.model.getFrameId(record['owner'])]
        vertices = hulls[f'vertices_{index}'] @ pose.rotation.T + pose.translation
        equations = hulls[f'planes_{index}']
        normal = equations[:, :3] @ pose.rotation.T
        offset = equations[:, 3] - normal @ pose.translation
        result.append((vertices, normal, offset))
    return result


def closest_base_shank_pairs(model, hulls, records, joints):
    """Return signed intersection radii for every base/FL_shank mesh pair.

    Positive radius proves authored-hull interior intersection. A negative
    radius records the shared plane-slack LP value, not Euclidean distance.
    """
    from scipy.optimize import linprog
    transformed = transformed_hulls(model, hulls, records, joints)
    candidates = []
    for first, first_record in enumerate(records):
        if first_record['owner'] != 'base':
            continue
        for second, second_record in enumerate(records):
            if second_record['owner'] != 'FL_shank':
                continue
            _, normal_a, offset_a = transformed[first]
            _, normal_b, offset_b = transformed[second]
            normal = np.vstack([normal_a, normal_b])
            rhs = -np.r_[offset_a, offset_b]
            fit = linprog([0, 0, 0, -1], A_ub=np.c_[normal, np.ones(len(normal))], b_ub=rhs,
                bounds=[(None, None)] * 4, method='highs')
            if not fit.success:
                raise RuntimeError(f'Intersection LP failed: {fit.message}')
            candidates.append(dict(first=first_record['path'], second=second_record['path'],
                max_common_interior_radius_m=float(fit.x[3]), witness_point_root_m=fit.x[:3].tolist()))
    candidates.sort(key=lambda item: item['max_common_interior_radius_m'], reverse=True)
    return candidates[:5]


def analyze(root, runs, output):
    root, runs, output = Path(root), Path(runs), Path(output)
    sys.path[:0] = [str(root / 'config')]
    from derive_v15_cad_model import collision_screen
    from v16_physical_model import PhysicalModel
    model_dir = root / 'config/experiments/v15_cad_20261011/model'
    model_path, hull_path = model_dir / 'quadruped_cad.xml', model_dir / 'collision_hulls.npz'
    metadata = json.loads((model_dir / 'cad_model.json').read_text())
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata['dynamics_mjcf_sha256']
    hulls = np.load(hull_path, allow_pickle=False)
    assert str(hulls['usd_sha256']) == metadata['source_usd_root_sha256']
    hull_records = json.loads(str(hulls['records_json']))
    model = PhysicalModel(model_path, floating_base=True)
    view = SimpleNamespace(model=model)
    nominal = np.asarray(metadata['canonical_nominal_q'])
    nominal_screen = collision_screen(view, hull_path, nominal)
    crossing = nominal.copy(); crossing[::3] = [-.3, .3, -.3, .3]
    crossing_screen = collision_screen(view, hull_path, crossing)
    assert nominal_screen['candidate_count'] == 0
    assert crossing_screen['candidate_count'] >= 2
    report = dict(schema=1,
        scope='CPU replay of actual recorded canonical joint configurations through authored CAD convex hulls; no PhysX stepping',
        model_sha256=metadata['dynamics_mjcf_sha256'], source_usd_sha256=metadata['source_usd_root_sha256'],
        collision_hulls_sha256=hashlib.sha256(hull_path.read_bytes()).hexdigest(),
        method='derive_v15_cad_model.collision_screen; each physical body placed from recorded joint positions. Self-intersection is invariant to rigid root pose, so the root is identity.',
        intersection_metric='Positive max_common_interior_radius_m proves overlapping interiors. It is an inscribed common-ball lower bound, not a penetration depth. Negative values are shared plane slack and are not Euclidean clearance.',
        controls=dict(nominal_candidates=nominal_screen['candidate_count'], inward_haa_candidates=crossing_screen['candidate_count']),
        references=[dict(url='https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html',
            supports='PhysX can generate contact before shape interiors intersect, according to the sum of contact offsets. The actual runtime offsets were not recorded in these trials.')],
        cases=[],
        torque_interpretation=dict(
            estimated_torque='The telemetry estimate is pre-step clipped implicit drive PD plus separately applied feedforward. It is neither a measured PhysX drive moment nor a joint reaction wrench.',
            structural_contact_load='Nonadjacent body contact force is a separate structural collision load. Its reaction and joint-constraint moments are absent from the PD+feedforward estimate.',
            current='The ideal current derived from estimated drive torque excludes collision reaction loads. A force norm alone cannot determine the moment about each motor axis or actual motor current.',
            duplicate_pair='base -> FL_shank and FL_shank -> base are two directed reports of the same interaction; do not add the two peak force norms.',
            temporal_alignment='joint_position is post-step; estimated_torque is calculated before that step. The predecessor joint_position is the pre-step pose for the peak force step.'),
        limits='Isaac cooked-hull differences, contact offsets, and solver correction within the 5 ms step are not reconstructed. Absence of post-step authored-hull intersection cannot dismiss the recorded contact.')
    for job, name in CASES:
        folder = runs / job / 'results' / name
        data = np.load(folder / 'telemetry.npz', allow_pickle=False)
        recorded = json.loads((folder / 'metadata.json').read_text())
        assert tuple(recorded['canonical_joint_order']) == model.joint_names
        labels = [entry['body'] + ' <-> ' + target for entry in recorded['pair_order'] for target in entry['targets']]
        forces = data['nonadjacent_pair_force_n']
        assert len(labels) == forces.shape[1]
        peak = int(np.argmax(forces.max(axis=1)))
        pair = int(np.argmax(forces[peak]))
        assert labels[pair] == 'base <-> FL_shank'
        result = json.loads((folder / 'result.json').read_text())
        case = dict(job_id=job, case=name, telemetry_sha256=hashlib.sha256((folder / 'telemetry.npz').read_bytes()).hexdigest(),
            peak_frame_index=peak, peak_state_time_s=float(data['state_stamp_ns'][peak] * 1e-9),
            peak_force_N=float(forces[peak, pair]), peak_pair=labels[pair],
            self_contact_frames=int(np.count_nonzero(forces.max(axis=1) > 1)),
            frames=[], reported_estimated_peak_torque_nm=result['torque_peak_nm'])
        for index in [peak - 1, peak]:
            joints = data['joint_position'][index].astype(float)
            screen = collision_screen(view, hull_path, joints)
            pair_candidates = closest_base_shank_pairs(model, hulls, hull_records, joints)
            base_shank_overlap = [item for item in screen['overlap_candidates']
                if '/base/' in item['first'] and '/FL_shank/' in item['second']]
            case['frames'].append(dict(role='predecessor' if index == peak - 1 else 'peak_post_step',
                index=index, state_time_s=float(data['state_stamp_ns'][index] * 1e-9),
                canonical_joint_position_rad=joints.tolist(), canonical_joint_velocity_rad_s=data['joint_velocity'][index].tolist(),
                estimated_torque_pre_step_nm=data['estimated_torque'][index].tolist(),
                feedforward_torque_pre_step_nm=data['feedforward_torque'][index].tolist(),
                measured_base_fl_shank_force_N=float(forces[index, pair]),
                authored_hull_intersections=screen['overlap_candidates'],
                authored_base_fl_shank_intersections=base_shank_overlap,
                closest_base_fl_shank_mesh_pairs=pair_candidates))
        report['cases'].append(case)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps([dict(case=item['case'], peak_force_N=item['peak_force_N'],
        peak_time_s=item['peak_state_time_s'],
        frames=[dict(role=frame['role'], overlaps=frame['authored_base_fl_shank_intersections'],
            closest=frame['closest_base_fl_shank_mesh_pairs'][0]) for frame in item['frames']]) for item in report['cases']], indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Project root containing config/model')
    parser.add_argument('--runs', required=True, help='Root of JOB/results/CASE recordings')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.root, args.runs, args.output)
