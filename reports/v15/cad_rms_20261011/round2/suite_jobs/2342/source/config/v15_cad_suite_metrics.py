"""Pure telemetry calculations for the CAD ROS2 acceptance suite."""
import numpy as np

RATED = np.tile([8.3, 18., 18.], 4)
PLANT_PEAK = np.tile([24.8, 48., 48.], 4)
DATA_SHEET_PEAK = np.tile([24.8, 53., 53.], 4)
OUTPUT_KT = np.tile([1.23, 1.44, 1.44], 4)


def estimated_torque(q, dq, target, kp, kd, drive_limits, feedforward):
    """PhysX implicit PD is limited separately from external feedforward."""
    return np.clip(kp * (target - q) - kd * dq, -drive_limits, drive_limits) + feedforward


def torque_summary(torque, velocity):
    torque = np.asarray(torque)
    rms = np.sqrt(np.mean(torque ** 2, axis=0))
    peak = np.max(abs(torque), axis=0)
    speed = np.clip(np.asarray(velocity) / 50., -1., 1.)
    low = PLANT_PEAK * np.maximum(-1. - speed, -1.)
    high = PLANT_PEAK * np.minimum(1. - speed, 1.)
    return dict(torque_rms_nm=rms.tolist(), torque_peak_nm=peak.tolist(),
                ideal_current_rms_a=(rms / OUTPUT_KT).tolist(),
                ideal_current_peak_a=(peak / OUTPUT_KT).tolist(),
                rated_torque_nm=RATED.tolist(), simulator_peak_torque_nm=PLANT_PEAK.tolist(),
                motor_datasheet_peak_torque_nm=DATA_SHEET_PEAK.tolist(),
                rated_exceed_fraction=np.mean(abs(torque) > RATED, axis=0).tolist(),
                rms_exceeds_rated=(rms > RATED).tolist(),
                simulator_peak_exceeded=bool(np.any(peak > PLANT_PEAK + 1e-4)),
                motor_datasheet_peak_exceeded=bool(np.any(peak > DATA_SHEET_PEAK + 1e-4)),
                velocity_guard_exceed_frames=int(np.any((torque < low - 1e-4) | (torque > high + 1e-4), axis=1).sum()),
                current_semantics='Ideal motor phase current |tau|/(Kt*gear), efficiency=1; not battery current or a measured drive moment.')


def standing_verdict(name, records, history, completed, stand_height, sit_height):
    reasons = []
    if not completed:
        reasons.append('did_not_complete_requested_duration')
    modes = list(dict.fromkeys(s['mode'] for s in history))
    if not records:
        return dict(motion_success=False, reasons=reasons + ['no_physics_samples'], modes_seen=modes)
    t = records['state_stamp_ns'] * 1e-9
    position = records['position']
    gravity = records['gravity']
    velocity = records['velocity']
    if name == 'standing_hold':
        mask = t >= 2.
        if not mask.any():
            return dict(motion_success=False, reasons=reasons + ['no_settled_standing_samples'], modes_seen=modes)
        drift = float(np.linalg.norm(position[:, :2] - position[0, :2], axis=1).max())
        height_error = float(np.max(abs(position[mask, 2] - stand_height))) if mask.any() else float('inf')
        if drift > .10:
            reasons.append('standing_drift_exceeds_0.10m')
        if height_error > .035:
            reasons.append('standing_height_error_exceeds_0.035m')
        if np.max(np.linalg.norm(velocity[mask, :2], axis=1)) > .12:
            reasons.append('standing_speed_exceeds_0.12m_s')
        if np.max(gravity[mask, 2]) > -.97:
            reasons.append('standing_tilt_exceeds_stability_gate')
        metrics = dict(drift_xy_m=drift, max_height_error_m=height_error)
    else:
        expected = ['SIT', 'RISE', 'STAND', 'BLEND_WALK', 'WALK', 'BRAKE', 'BLEND_STAND', 'LOWER', 'SETTLE_SIT']
        missing = [m for m in expected if m not in modes]
        if missing:
            reasons.append('missing_modes:' + ','.join(missing))
        if not history or history[-1]['mode'] != 'SIT':
            reasons.append('did_not_finish_in_SIT')
        stand_mask = (t >= 5.) & (t < 7.)
        if not stand_mask.any() or np.max(abs(position[stand_mask, 2] - stand_height)) > .035:
            reasons.append('rise_did_not_reach_and_hold_stand_height')
        walk_mask = (t >= 9.) & (t < 17.)
        progress = float(position[walk_mask][-1, 1] - position[walk_mask][0, 1]) if walk_mask.any() else 0.
        if progress < 1.6:
            reasons.append('integrated_walk_progress_below_1.6m')
        end_height = float(position[-1, 2])
        if abs(end_height - sit_height) > .035:
            reasons.append('final_sit_height_error_exceeds_0.035m')
        stop_mask = (t >= 21.) & (t < 23.)
        if not stop_mask.any() or np.max(np.linalg.norm(velocity[stop_mask, :2], axis=1)) > .10:
            reasons.append('integrated_stop_speed_exceeds_0.10m_s')
        metrics = dict(walking_progress_y_m=progress, final_sit_height_m=end_height,
                       stand_height_target_m=stand_height, sit_height_target_m=sit_height)
    return dict(motion_success=not reasons, reasons=reasons, modes_seen=modes, metrics=metrics)
