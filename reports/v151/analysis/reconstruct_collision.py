"""Offline FK reconstruction of missing collision geometry; no physics simulation.

Validate the model on recorded 597 foot COMs, then cross-check each run's
existing cross-leg/trunk gap channels before publishing separate same-leg gaps.
The original recordings and their success/failure verdicts are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'config'))
from v16_physical_model import PhysicalModel
from v14_clearance import interleg_clearances_m
from v151_collision import (LEGS, PROFILE_NAMES, SAME_LEG_THIGH_FOOT_PAIRS,
                            capsule_trunk_gaps_m, same_leg_thigh_foot_gaps_m)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_summary(reconstructed, measured):
    reconstructed, measured = np.asarray(reconstructed), np.asarray(measured)
    if reconstructed.shape != measured.shape or not np.isfinite(measured).all():
        raise ValueError('Validation measurement has mismatched shape or nonfinite values')
    error = reconstructed - measured
    if not np.isfinite(error).all():
        raise ValueError('Reconstruction has nonfinite values')
    worst = np.unravel_index(np.abs(error).argmax(), error.shape)
    return dict(max_absolute_error_m=float(np.abs(error).max()),
                rms_error_m=float(np.sqrt(np.mean(error ** 2))),
                worst_array_index=list(map(int, worst)))


def reconstruct(model, input_path, foot_boxes):
    source = np.load(input_path, allow_pickle=False)
    metadata = json.loads(input_path.with_suffix('.json').read_text())
    if metadata['joint_order'] != list(model.joint_names):
        raise ValueError(f'{input_path}: joint ordering does not match the model')
    position, quaternion, joints = (source[key] for key in ('position', 'quaternion', 'joint_position'))
    count = len(position)
    if not count or position.shape != (count, 3) or quaternion.shape != (count, 4) or joints.shape != (count, 12):
        raise ValueError('Expected nonempty synchronized single-robot pose and joint arrays')
    if not all(np.isfinite(value).all() for value in (position, quaternion, joints)):
        raise ValueError('Source pose or joints are nonfinite')
    if np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1.)) > 1e-5:
        raise ValueError('Source quaternions are not unit length')
    names = ('base', *PROFILE_NAMES, *(f'{leg}_foot' for leg in LEGS))
    frame_ids = [model.model.getFrameId(name) for name in names]
    if any(i >= len(model.model.frames) for i in frame_ids):
        raise ValueError('Model is missing a required collision body frame')
    positions = np.empty((count, len(names), 3))
    quaternions = np.empty((count, len(names), 4))
    foot_com = np.empty((count, 4, 3))
    records = {record['name']: record for record in model.body_records}
    for i in range(count):
        q = model.configuration(joints[i], base_position=position[i],
                                base_quaternion_xyzw=quaternion[i, [1, 2, 3, 0]])
        model.pin.forwardKinematics(model.model, model.data, q)
        model.pin.updateFramePlacements(model.model, model.data)
        for body, frame in enumerate(frame_ids):
            pose = model.data.oMf[frame]
            positions[i, body] = pose.translation
            quaternions[i, body] = model.pin.Quaternion(pose.rotation).coeffs()[[3, 0, 1, 2]]
        for leg, name in enumerate(names[-4:]):
            pose = model.data.oMf[frame_ids[-4 + leg]]
            foot_com[i, leg] = pose.translation + pose.rotation @ records[name]['com_body']
    cross = interleg_clearances_m(positions, quaternions, names)
    trunk = capsule_trunk_gaps_m(positions, quaternions, names)
    validation = dict(cross_leg_gap=error_summary(cross, source['gaps']),
                      trunk_gap=error_summary(trunk, source['trunk_gaps']))
    if 'foot_com_position' in source:
        validation['foot_com_position'] = error_summary(foot_com, source['foot_com_position'])
        validation['foot_com_position']['max_euclidean_error_m'] = float(
            np.linalg.norm(foot_com - source['foot_com_position'], axis=-1).max())
    gaps = same_leg_thigh_foot_gaps_m(positions, quaternions, names, foot_boxes)
    arrays = dict(sample_index=np.arange(count), inferred_time_s=np.arange(count) * .005,
                  reconstructed_same_leg_thigh_foot_gap_m=gaps,
                  reconstructed_foot_com_position_world_m=foot_com,
                  reconstructed_body_position_world_m=positions,
                  reconstructed_body_quaternion_wxyz=quaternions,
                  reconstructed_cross_leg_gap_m=cross, reconstructed_trunk_gap_m=trunk)
    if 'foot_com_position' in source:
        arrays['foot_com_error_m'] = foot_com - source['foot_com_position']
    return arrays, validation, names, metadata


def require_validation(validation, tolerance_m, require_com=False):
    if require_com and 'foot_com_position' not in validation:
        raise ValueError('Reference run lacks measured foot COM positions')
    failed = {name: result for name, result in validation.items()
              if result['max_absolute_error_m'] > tolerance_m}
    if failed:
        raise ValueError(f'Reconstruction validation exceeds {tolerance_m} m: {failed}')


def run(args):
    model = PhysicalModel(args.model, floating_base=True)
    evidence = json.loads(args.foot_evidence.read_text())
    boxes = evidence['feet']
    inputs = [args.reference, *args.runs]
    outputs = []
    # Validate all source runs before creating any result artifacts.
    for index, source_path in enumerate(inputs):
        arrays, validation, names, original = reconstruct(model, source_path, boxes)
        require_validation(validation, args.tolerance_m, require_com=index == 0)
        print(f'{source_path.parent.name}: ' + ', '.join(
            f'{key}={value["max_absolute_error_m"]:.3g} m' for key, value in validation.items()), flush=True)
        outputs.append((source_path, arrays, validation, names, original))
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for source_path, arrays, validation, names, original in outputs:
        gaps = arrays['reconstructed_same_leg_thigh_foot_gap_m']
        worst_frame, worst_pair = np.unravel_index(gaps.argmin(), gaps.shape)
        label = '_'.join(source_path.relative_to(ROOT / 'reports/v151').parts[:-1])
        destination = args.output / label
        np.savez_compressed(destination.with_suffix('.npz'), **arrays)
        summary = dict(schema=1, scope='offline model reconstruction; not measured PhysX body poses',
            original_input=str(source_path), original_npz_sha256=sha256(source_path),
            original_result_sha256=sha256(source_path.with_suffix('.json')),
            original_completed=original.get('completed'), original_failure=original.get('failure'),
            original_verdict_modified=False, new_physics_simulation_performed=False,
            model_path=str(args.model), model_sha256=model.source_sha256,
            foot_evidence_path=str(args.foot_evidence), foot_evidence_sha256=sha256(args.foot_evidence),
            foot_usd_source=evidence['source_usd'], foot_usd_source_sha256=evidence['source_sha256'],
            validation_reference=str(args.reference), validation_tolerance_m=args.tolerance_m,
            validation=validation, reconstruction_validated=True,
            direct_foot_com_validation_available='foot_com_position' in validation,
            body_order=list(names), joint_order=list(model.joint_names), pair_order=list(SAME_LEG_THIGH_FOOT_PAIRS),
            samples=len(gaps), sample_period_s=.005,
            time_semantics='inferred sample_index * 0.005 s; source NPZ has no explicit timestamps',
            frame_scope='all recorded samples, including terminal/falling trajectory; no stable-window exclusion',
            minimum_same_leg_thigh_foot_gap_m=float(gaps.min()),
            per_pair_minimum_gap_m=dict(zip(SAME_LEG_THIGH_FOOT_PAIRS, map(float, gaps.min(axis=0)))),
            proxy_overlap_sample_fraction=float(np.mean(gaps.min(axis=1) < 0)),
            minimum_sample_index=int(worst_frame), minimum_inferred_time_s=float(worst_frame * .005),
            minimum_pair=SAME_LEG_THIGH_FOOT_PAIRS[worst_pair],
            reconstructed_geometry_clear=bool(np.all(gaps >= 0)),
            actual_same_leg_contact_measured=False, same_leg_physical_pairs_filtered=True,
            limitations=[
                'Forward kinematics reconstructs rigid body poses from logged actor root pose and joint angles; it cannot recover independent PhysX constraint drift.',
                'Runs without foot COM telemetry transfer the COM validation from 597; their existing cross-leg and trunk gaps are separately validated on every sample.',
                'Thigh capsule versus enclosing foot convex-hull AABB is conservative. Negative gap is possible proxy overlap, not confirmed mesh collision or penetration depth.',
                'Only the four same-leg thigh-profile to foot pairs are covered here; contact offsets, CAD-wide collisions and body-ground contact are excluded.',
                'This adds retrospective geometry evidence, not a new physical validation or change to original task success.'],
            implementation_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
                Path(__file__), ROOT/'config/v16_physical_model.py', ROOT/'config/v151_collision.py',
                ROOT/'config/v14_clearance.py', ROOT/'config/v14_usd_plan.py')})
        destination.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        summaries.append(summary)
        print(f'{label}: minimum {gaps.min():.9f} m; overlap fraction {summary["proxy_overlap_sample_fraction"]:.6f}', flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'reports/v16/physical_model/v16_model_source_quadruped.xml')
    parser.add_argument('--foot-evidence', type=Path, default=ROOT/'reports/v151/geometry/foot_collider_evidence.json')
    parser.add_argument('--reference', type=Path, default=ROOT/'reports/v151/integration/597/result.npz')
    parser.add_argument('--runs', type=Path, nargs='*', default=sorted((ROOT/'reports/v151/comparison/594').glob('*/result.npz')))
    parser.add_argument('--output', type=Path, default=ROOT/'reports/v151/analysis/reconstructed_collision')
    parser.add_argument('--tolerance-m', type=float, default=1e-5)
    options = parser.parse_args()
    if not np.isfinite(options.tolerance_m) or options.tolerance_m <= 0:
        parser.error('--tolerance-m must be positive and finite')
    run(options)
