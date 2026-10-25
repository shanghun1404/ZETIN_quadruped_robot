"""Direction and terrain-height metrics, excluding every post-reset sample."""

import numpy as np


def summarize_path(*, positions, quaternions, ground_height, valid, alive_at_end,
                   scenario, step_height_m=0.0, full_rise_m=0.0):
    if scenario in ("stairs_up", "stairs_down") and not (step_height_m > 0 and full_rise_m > 0):
        raise ValueError("stair success requires positive, known step height and flight rise")
    positions, quaternions = np.asarray(positions), np.asarray(quaternions)
    ground_height, valid = np.asarray(ground_height), np.asarray(valid, dtype=bool)
    alive_at_end = np.asarray(alive_at_end, dtype=bool)
    if (positions.shape != (*valid.shape, 3) or quaternions.shape != (*valid.shape, 4)
            or ground_height.shape != valid.shape or alive_at_end.shape != (valid.shape[1],)):
        raise ValueError("trajectory shapes must be [time, env, channels]")
    if not all(np.isfinite(a[valid]).all() for a in (positions, quaternions, ground_height)):
        raise ValueError("valid trajectory samples must be finite")
    rows = []
    for env_id in range(valid.shape[1]):
        mask = valid[:, env_id]
        if not mask.any():
            raise ValueError("each environment needs an initial sample")
        p, q, ground = positions[mask, env_id], quaternions[mask, env_id], ground_height[mask, env_id]
        # Project the robot's physical +y forward axis onto the world floor.
        w, x, y, z = q.T
        forward = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z)), axis=-1)
        forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
        right = np.array([forward[0, 1], -forward[0, 0]])
        delta = p[:, :2] - p[0, :2]
        progress, lateral = delta @ forward[0], delta @ right
        angle = np.arctan2(forward[:, 1], forward[:, 0])
        error = np.arctan2(np.sin(angle-angle[0]), np.cos(angle-angle[0]))
        path_length = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
        ground_change = float(ground[-1] - ground[0])
        travel = -ground_change if scenario == "stairs_down" else ground_change
        corridor = bool(np.abs(lateral).max() <= 0.6 and np.abs(error).max() <= np.pi/3)
        stairs = scenario in ("stairs_up", "stairs_down")
        passed = bool(stairs and alive_at_end[env_id] and corridor and progress[-1] >= 2.1
                      and travel >= 2*step_height_m - 0.02)
        full = bool(passed and progress[-1] >= 3.6 and travel >= full_rise_m - 0.02)
        rows.append(dict(
            env_id=env_id, final_directed_progress_m=float(progress[-1]),
            max_abs_cross_track_m=float(np.abs(lateral).max()),
            final_cross_track_m=float(lateral[-1]),
            max_abs_heading_error_deg=float(np.degrees(np.abs(error).max())),
            final_heading_error_deg=float(np.degrees(error[-1])),
            path_length_m=float(path_length), final_ground_change_m=ground_change,
            corridor_kept=corridor, directed_stair_success=passed, full_flight_success=full,
        ))
    return dict(
        per_environment=rows,
        directed_stair_success_count=sum(r["directed_stair_success"] for r in rows),
        full_flight_success_count=sum(r["full_flight_success"] for r in rows),
        corridor_kept_count=sum(r["corridor_kept"] for r in rows),
        mean_final_directed_progress_m=float(np.mean([r["final_directed_progress_m"] for r in rows])),
        mean_max_abs_cross_track_m=float(np.mean([r["max_abs_cross_track_m"] for r in rows])),
        mean_max_abs_heading_error_deg=float(np.mean([r["max_abs_heading_error_deg"] for r in rows])),
        criteria=dict(min_forward_m=2.1, max_cross_track_m=0.6, max_heading_error_deg=60,
                      ground_height_tolerance_m=0.02, full_forward_m=3.6),
    )
