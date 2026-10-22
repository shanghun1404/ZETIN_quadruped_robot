"""Replay the recorded 2320 standing handover, preserving controller state."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def replay(source, runs, output, fixture_output=None):
    source, runs, output = Path(source), Path(runs), Path(output)
    sys.path[:0] = [str(source / 'config'), str(source / 'ros2/v151_ros2_control')]
    from v151_ros2_control.cad_standing import CadStandingController
    from v151_ros2_control.motion import MotionSupervisor
    folder = runs / '2320/results/sit_rise_walk_stop_sit'
    data = np.load(folder / 'telemetry.npz', allow_pickle=False)
    history = json.loads((folder / 'status.json').read_text())
    metadata = json.loads((folder / 'metadata.json').read_text())
    config = json.loads((folder / 'case_config.json').read_text())
    lane = np.asarray(config['lane_origin_world_m'])
    status_by_stamp = {item['stamp_ns']: item for item in history}
    state_by_stamp = {int(stamp): index for index, stamp in enumerate(data['state_stamp_ns'])}
    matched = []
    for item in history:
        stamp = item['stamp_ns']
        if stamp % 20_000_000 or not 17.30 <= stamp * 1e-9 <= 17.90:
            continue
        if 'stance_mask' not in item.get('wbc_vectors', {}):
            continue
        raw = np.linalg.norm(data['shank_support_force'][state_by_stamp[stamp]], axis=1) > 5
        matched.append((raw, np.asarray(item['wbc_vectors']['stance_mask'])))
    permutations = [order for order in itertools.permutations(range(4))
        if all(np.array_equal(raw[list(order)], expected) for raw, expected in matched)]
    assert len(permutations) == 1, 'Contact sensor order must be identified from the recorded masks'
    support_order = permutations[0]
    ctl = CadStandingController(source / 'config/model/quadruped_cad.xml', dt=.02)
    base_com = np.asarray(metadata['base_com_offset'])
    stand_height = metadata['initial_root_world_m'][2] + .10
    def measured(stamp):
        index = state_by_stamp[stamp]
        root = data['position'][index].astype(float) + lane
        quat = data['quaternion'][index].astype(float)
        angular = data['angular_velocity'][index].astype(float)
        # Telemetry root_lin_vel_b is at the base COM. The ROS bridge supplies
        # root_link_lin_vel_w rotated to body axes, hence this origin conversion.
        linear = data['velocity'][index].astype(float) - np.cross(angular, base_com)
        support = np.linalg.norm(data['shank_support_force'][index], axis=1)[list(support_order)] > 5
        return data['joint_position'][index].astype(float), data['joint_velocity'][index].astype(float), root, quat, linear, angular, support
    start_stamp = 17_300_000_000
    q, dq, root, quat, linear, angular, support = measured(start_stamp)
    ctl.initialize(q, root, quat, stance_mask=support)
    handover_height = float(root[2])
    supervisor = MotionSupervisor(initial_mode='WALK')
    supervisor.mode = 'BLEND_STAND'
    supervisor.entered = 17.30
    supervisor.previous = 17.30
    supervisor.requested = 'stand'
    supervisor.last_command = np.asarray(status_by_stamp[start_stamp]['command'])
    supervisor.issued_command = supervisor.last_command.copy()
    replay_checks = []
    failed_stamp = int(data['state_stamp_ns'][-1])
    for stamp in range(start_stamp + 20_000_000, failed_stamp, 20_000_000):
        q, dq, root, quat, linear, angular, support = measured(stamp)
        rotation = ctl.model.pin.Quaternion(ctl._base(root, quat)['base_quaternion_xyzw']).matrix()
        gravity = rotation.T @ [0, 0, -1]
        stable = (gravity[2] < -.97 and np.linalg.norm(linear) < .12 and np.linalg.norm(angular) < .35
            and support.sum() == 4 and abs(root[2] - stand_height) < .035)
        state = supervisor.step(stamp * 1e-9, stable=stable,
            support_ready=gravity[2] < -.97 and support.sum() >= 2, seated_ready=False)
        recorded = status_by_stamp[stamp]
        assert state['mode'] == recorded['mode']
        assert abs(state['walk_weight'] - recorded['walk_weight']) < 1e-12
        height = state['walk_weight'] * handover_height + (1 - state['walk_weight']) * stand_height
        qref, goal = ctl.contact_consistent_reference(q, root, quat, height, support,
            center_body=1 - state['walk_weight'], forward_offset_m=0.)
        torque, diag = ctl.compute(q, dq, root, quat, linear, angular, height=height,
            joint_reference=qref, stance_mask=support, position_reference_world=goal)
        replay_checks.append(dict(stamp_ns=stamp,
            reference_max_abs_error_rad=float(np.max(np.abs(qref - recorded['wbc_vectors']['joint_reference']))),
            body_goal_max_abs_error_m=float(np.max(np.abs(goal - recorded['wbc_vectors']['base_position_reference_world']))),
            support_mask_matches=bool(np.array_equal(support, recorded['wbc_vectors']['stance_mask'])),
            wbc_acceleration_max_abs_error=float(np.max(np.abs(diag['acceleration'] - recorded['wbc_vectors']['acceleration'])))))
    q, dq, root, quat, linear, angular, support = measured(failed_stamp)
    state = supervisor.step(failed_stamp * 1e-9, stable=False, support_ready=False, seated_ready=False)
    height = state['walk_weight'] * handover_height + (1 - state['walk_weight']) * stand_height
    saved = {name: copy.deepcopy(getattr(ctl, name)) for name in ['contact_anchors', '_support_plane_z', '_initial_position',
        '_target_rotation', '_previous_reference', '_previous_stance', '_previous', '_warm']}
    report = dict(scope='CPU replay of recorded BLEND_STAND inputs and inherited CAD standing controller state; no simulation',
        source_snapshot=str(source), failed_command_stamp_ns=failed_stamp,
        raw_contact_sensor_to_canonical_indices=list(support_order), permutation_matched_policy_ticks=len(matched),
        replayed_policy_ticks=len(replay_checks), replay_checks=replay_checks,
        supervisor_output=state, actor_origin_world_m=root.tolist(), quat_wxyz=quat.tolist(),
        canonical_joint_position_rad=q.tolist(), stance_mask_canonical=support.tolist(),
        latched_contact_anchors_world_m=ctl.contact_anchors.tolist(),
        support_plane_z_before_call_m=float(ctl._support_plane_z), target_root_height_m=height,
        handover_root_height_m=handover_height, stand_target_root_height_m=stand_height, comparisons=[])
    for depth in [.005, 0.]:
        for name, value in saved.items():
            setattr(ctl, name, copy.deepcopy(value))
        item = dict(landing_depth_m=depth)
        try:
            qref, goal = ctl.contact_consistent_reference(q, root, quat, height, support,
                center_body=1 - state['walk_weight'], forward_offset_m=0., landing_depth_m=depth)
            target = ctl.contact_positions(ctl.crouch_pose(0.)[0], root, quat)
            target[:, 2] = ctl._support_plane_z - depth
            actual = ctl.contact_positions(qref, root, quat)
            item.update(ik_success=True, canonical_joint_reference_rad=qref.tolist(), body_goal_world_m=goal.tolist(),
                support_plane_z_after_call_m=float(ctl._support_plane_z), landing_targets_world_m=target.tolist(),
                unsupported_foot_max_abs_residual_m=float(np.max(np.abs(actual[~support] - target[~support]))))
            torque, diag = ctl.compute(q, dq, root, quat, linear, angular, height=height,
                joint_reference=qref, stance_mask=support, position_reference_world=goal)
            item.update(wbc_finite=bool(diag['finite']), wbc_feasible=bool(diag['feasible']), torque_Nm=torque.tolist())
        except ValueError as error:
            item.update(ik_success=False, error=str(error), support_plane_z_after_call_m=float(ctl._support_plane_z))
        report['comparisons'].append(item)
    if fixture_output:
        fixture = dict(source=dict(job_id='2320', case='sit_rise_walk_stop_sit', command_stamp_ns=failed_stamp,
                model_sha256=ctl.model.source_sha256,
                definition='Measured actor-origin pose and canonical joints, with the latched contact state immediately before the failed landing IK call.'),
            inputs=dict(joint_pos=q.tolist(), base_pos=root.tolist(), quat_wxyz=quat.tolist(), height=height,
                stance_mask=support.tolist(), center_body=1 - state['walk_weight']),
            latched_state=dict(contact_anchors=saved['contact_anchors'].tolist(),
                previous_stance=saved['_previous_stance'].tolist(), target_rotation=saved['_target_rotation'].tolist(),
                support_plane_z=saved['_support_plane_z']),
            expected_landing_targets_world_m=report['comparisons'][1]['landing_targets_world_m'],
            expected_explicit_depth_failure=dict(landing_depth_m=.005, leg='FR_foot', residual_m=.00205351))
        fixture_path = Path(fixture_output)
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(failed_time_s=failed_stamp * 1e-9, support_order=support_order,
        reference_replay_max_abs_error=max(item['reference_max_abs_error_rad'] for item in replay_checks),
        body_goal_replay_max_abs_error=max(item['body_goal_max_abs_error_m'] for item in replay_checks),
        support_plane_z_m=report['support_plane_z_before_call_m'], comparisons=report['comparisons']), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture-output', help='Export only the measured input and latched state needed by IK')
    args = parser.parse_args()
    replay(args.source, args.runs, args.output, args.fixture_output)
