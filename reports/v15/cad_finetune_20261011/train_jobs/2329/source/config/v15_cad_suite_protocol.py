"""CPU-only motion contracts for the unchanged V15 CAD ROS2 suite.

The real CAD body frame has +X right, +Y forward, and +Z up. Commands
are ``(vx_right, vy_forward, yaw_rate)``; positions are in world XYZ and
quaternions are scalar-first WXYZ. Geometry is created separately from the
CAD asset and its height is known analytically, without camera estimates.
"""
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np


_EDGE_TOLERANCE_M = 1e-9
_SETTLE_S = 2.0


@dataclass(frozen=True)
class CommandSegment:
    start_s: float
    end_s: float
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class TerrainSpec:
    kind: str = "flat"
    length_m: float = 10.0
    width_m: float = 2.0
    initial_support_back_m: float = 2.0
    start_flat_m: float = 2.0
    step_width_m: float = 0.3
    step_height_m: float = 0.0
    riser_count: int = 4
    top_plateau_m: float = 3.0
    rough_end_y_m: float = 6.0
    rough_seed: int = 42
    rough_max_height_m: float = 0.035

    @property
    def full_rise_m(self):
        return self.riser_count * self.step_height_m

    @property
    def last_riser_y_m(self):
        return self.start_flat_m + (self.riser_count - 1) * self.step_width_m

    @property
    def required_finish_y_m(self):
        return self.last_riser_y_m + 1.0


@dataclass(frozen=True)
class SuiteCase:
    name: str
    description: str
    duration_s: float
    terrain: TerrainSpec
    commands: tuple[CommandSegment, ...]


@dataclass(frozen=True)
class Cuboid:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


def _constant_case(name, description, duration, velocity, terrain=TerrainSpec()):
    return SuiteCase(name, description, duration, terrain,
                     (CommandSegment(0.0, duration, velocity),))


_base_cases = (
    _constant_case("flat_14s", "Original 14 s +Y flat reference", 14.0, (0.0, 0.4, 0.0)),
    SuiteCase("stop_restart", "Walk, stop, restart, and stop again", 24.0, TerrainSpec(), (
        CommandSegment(0.0, 8.0, (0.0, 0.4, 0.0)),
        CommandSegment(8.0, 12.0, (0.0, 0.0, 0.0)),
        CommandSegment(12.0, 20.0, (0.0, 0.4, 0.0)),
        CommandSegment(20.0, 24.0, (0.0, 0.0, 0.0)),
    )),
    _constant_case("reverse", "Commanded reverse in physical body -Y", 14.0, (0.0, -0.3, 0.0)),
    _constant_case("lateral", "Commanded lateral travel in physical body +X", 14.0, (0.2, 0.0, 0.0)),
    _constant_case("yaw", "Positive yaw with no commanded translation", 14.0, (0.0, 0.0, 0.25)),
)
_stair_cases = tuple(
    _constant_case(f"stairs_{direction}_{label}",
                   f"Four {height * 100:g} cm risers {direction}, +Y corridor", 24.0,
                   (0.0, 0.4, 0.0), TerrainSpec(kind=f"stairs_{direction}", step_height_m=height))
    for height, label in ((0.05, "05cm"), (0.10, "10cm"), (0.157, "157mm"))
    for direction in ("up", "down")
)
SUITE_CASES = _base_cases + _stair_cases + (
    _constant_case("rough", "Seeded 1–3.5 cm rough cuboid tiles, +Y corridor", 24.0,
                   (0.0, 0.35, 0.0), TerrainSpec(kind="rough")),
)
CASES = MappingProxyType({case.name: case for case in SUITE_CASES})


def get_case(case):
    """Accept an immutable record or resolve a suite case by its stable name."""
    if isinstance(case, SuiteCase):
        return case
    try:
        return CASES[case]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown CAD suite case: {case!r}") from error


def command_at(case, time_s):
    """Zero-order hold; commands stop at and after the case duration."""
    case = get_case(case)
    time_s = float(time_s)
    if not np.isfinite(time_s):
        raise ValueError("command time must be finite")
    for segment in case.commands:
        if segment.start_s <= time_s < segment.end_s:
            return segment.velocity
    return (0.0, 0.0, 0.0)


def _block(name, x0, x1, y0, y1, height):
    return Cuboid(name, ((x0 + x1) / 2.0, (y0 + y1) / 2.0, height / 2.0),
                  (x1 - x0, y1 - y0, height))


def terrain_cuboids(case):
    """Static cube specs above the existing z=0 plane; never CAD edits.

    The ascent has a 3 m top plateau and a level continuation to the end
    of the 10 m corridor so the 24 s test cannot encounter a second cliff.
    The descent's initial platform extends behind the spawn for rear feet.
    """
    terrain = get_case(case).terrain
    half_width = terrain.width_m / 2.0
    cubes = []
    if terrain.kind == "flat":
        return ()
    if terrain.kind in ("stairs_up", "stairs_down"):
        down = terrain.kind == "stairs_down"
        if down:
            cubes.append(_block("initial_platform", -half_width, half_width,
                                -terrain.initial_support_back_m, terrain.start_flat_m,
                                terrain.full_rise_m))
        for i in range(terrain.riser_count):
            height = (terrain.riser_count - i - 1 if down else i + 1) * terrain.step_height_m
            if height > 0.0:
                start = terrain.start_flat_m + i * terrain.step_width_m
                cubes.append(_block(f"tread_{i + 1}", -half_width, half_width,
                                    start, start + terrain.step_width_m, height))
        if not down:
            start = terrain.start_flat_m + terrain.riser_count * terrain.step_width_m
            plateau_end = min(start + terrain.top_plateau_m, terrain.length_m)
            cubes.append(_block("top_plateau", -half_width, half_width, start,
                                plateau_end, terrain.full_rise_m))
            if plateau_end < terrain.length_m:
                cubes.append(_block("top_continuation", -half_width, half_width,
                                    plateau_end, terrain.length_m, terrain.full_rise_m))
        return tuple(cubes)
    if terrain.kind == "rough":
        rng = np.random.default_rng(terrain.rough_seed)
        xs = np.linspace(-half_width, half_width, 5)
        ys = np.linspace(terrain.start_flat_m, terrain.rough_end_y_m, 11)
        for row in range(10):
            for col in range(4):
                height = float(rng.uniform(0.01, terrain.rough_max_height_m))
                cubes.append(_block(f"rough_{row}_{col}", xs[col], xs[col + 1],
                                    ys[row], ys[row + 1], height))
        return tuple(cubes)
    raise ValueError(f"unsupported terrain kind: {terrain.kind}")


def ground_height(case, x, y):
    """Known local ground, broadcasting coordinates; exact risers use +Y side.

    Coordinates within 1 nm of a riser are treated as lying on the new
    tread, preventing binary floating point from moving a nominal boundary.
    """
    case = get_case(case)
    terrain = case.terrain
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("ground coordinates must be finite")
    height = np.zeros_like(x)
    if terrain.kind in ("stairs_up", "stairs_down"):
        boundaries = terrain.start_flat_m + np.arange(terrain.riser_count) * terrain.step_width_m
        climbed = np.searchsorted(boundaries - _EDGE_TOLERANCE_M, y, side="right")
        level = terrain.riser_count - climbed if terrain.kind == "stairs_down" else climbed
        inside = ((np.abs(x) <= terrain.width_m / 2.0 + _EDGE_TOLERANCE_M)
                  & (y >= -terrain.initial_support_back_m - _EDGE_TOLERANCE_M)
                  & (y <= terrain.length_m + _EDGE_TOLERANCE_M))
        height = np.where(inside, level * terrain.step_height_m, 0.0)
    elif terrain.kind == "rough":
        for cube in terrain_cuboids(case):
            inside = ((np.abs(x - cube.center[0]) <= cube.size[0] / 2.0 + _EDGE_TOLERANCE_M)
                      & (np.abs(y - cube.center[1]) <= cube.size[1] / 2.0 + _EDGE_TOLERANCE_M))
            height = np.maximum(height, np.where(inside, cube.size[2], 0.0))
    elif terrain.kind != "flat":
        raise ValueError(f"unsupported terrain kind: {terrain.kind}")
    return float(height) if height.ndim == 0 else height


def case_config(case):
    """JSON-ready metadata; mutating the returned dict cannot alter the suite."""
    case = get_case(case)
    config = asdict(case)
    config["command_frame"] = "physical body (vx_right, vy_forward, yaw_rate)"
    config["quaternion_order"] = "wxyz"
    terrain = config["terrain"]
    terrain.update(full_rise_m=case.terrain.full_rise_m,
                   last_riser_y_m=case.terrain.last_riser_y_m,
                   required_finish_y_m=case.terrain.required_finish_y_m,
                   spawn_ground_height_m=ground_height(case, 0.0, 0.0),
                   cuboids=[asdict(cube) for cube in terrain_cuboids(case)])
    config["criteria"] = dict(settle_s=_SETTLE_S, max_velocity_mae_m_s=0.15,
                               min_torso_height_above_ground_m=0.20,
                               max_tilt_deg=60.0, max_corridor_abs_x_m=0.70,
                               ground_change_tolerance_m=0.02,
                               max_settled_stop_speed_m_s=0.10,
                               max_settled_stop_drift_m=0.20,
                               max_yaw_rate_mae_rad_s=0.15,
                               max_final_yaw_error_rad=0.60)
    if case.name == "flat_14s":
        config["criteria"]["min_progress_y_m"] = 0.3 * max(0.0, case.duration_s - 2.0)
    return config


def _validate_trajectory(time_s, positions, quaternions, body_velocity, gravity):
    time = np.asarray(time_s, dtype=float)
    positions = np.asarray(positions, dtype=float)
    quaternion = np.asarray(quaternions, dtype=float)
    velocity = np.asarray(body_velocity, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("trajectory time must contain at least two increasing samples")
    if positions.shape != (len(time), 3) or velocity.shape != positions.shape or quaternion.shape != (len(time), 4):
        raise ValueError("trajectory shapes must be time [N], position/velocity [N,3], quaternion [N,4]")
    arrays = (time, positions, quaternion, velocity)
    if gravity is not None:
        gravity = np.asarray(gravity, dtype=float)
        if gravity.shape != positions.shape:
            raise ValueError("gravity shape must be [N,3]")
        arrays += (gravity,)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("trajectory samples must be finite")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("quaternion norm must be positive")
    return time, positions, quaternion / norms[:, None], velocity, gravity


def evaluate_case(case, *, time_s, positions, quaternions, body_velocity,
                  gravity=None, alive_at_end=True):
    """Return motion verdicts only; the runner separately gates CAD contacts/torques.

    All trajectory samples must precede any reset. Complete recording and
    survival are mandatory; a partial trace never inherits a smaller target.
    """
    case = get_case(case)
    time, position, quaternion, velocity, gravity = _validate_trajectory(
        time_s, positions, quaternions, body_velocity, gravity)
    ground = ground_height(case, position[:, 0], position[:, 1])
    torso_height = position[:, 2] - ground
    w, x, y, z = quaternion.T
    upright = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    if gravity is not None:
        norm = np.linalg.norm(gravity, axis=1)
        if np.any(norm < 1e-8):
            raise ValueError("gravity norm must be positive")
        upright = np.minimum(upright, -gravity[:, 2] / norm)
    tilt_deg = np.degrees(np.arccos(np.clip(upright, -1.0, 1.0)))
    yaw = np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    dt = float(np.median(np.diff(time)))
    recorded_complete = bool(time[0] <= 1.5 * dt and time[-1] >= case.duration_s - 1.5 * dt)
    metrics = dict(progress_y_m=float(position[-1, 1] - position[0, 1]),
                   progress_x_m=float(position[-1, 0] - position[0, 0]),
                   final_y_m=float(position[-1, 1]),
                   max_abs_x_m=float(np.abs(position[:, 0]).max()),
                   ground_height_change_m=float(ground[-1] - ground[0]),
                   torso_min_height_above_ground_m=float(torso_height.min()),
                   max_tilt_deg=float(tilt_deg.max()),
                   actual_yaw_change_rad=float(yaw[-1] - yaw[0]),
                   seconds_recorded=float(time[-1]), recording_complete=recorded_complete)
    gates = dict(recording_complete=recorded_complete, survived=bool(alive_at_end),
                 torso_height=bool(torso_height.min() >= 0.20 - 1e-9),
                 tilt=bool(tilt_deg.max() <= 60.0 + 1e-9))
    reason_codes = dict(recording_complete="incomplete_recording", survived="did_not_survive",
                        torso_height="low_torso", tilt="excessive_tilt")

    settled = (time >= _SETTLE_S) & (time <= case.duration_s)
    if case.name == "flat_14s":
        # Original CAD telemetry is recorded after each 5 ms step; [400:]
        # starts at 2.005 s and excludes the post-state sample at 2.000 s.
        settled = (time > _SETTLE_S) & (time <= case.duration_s)
        mae = float(np.mean(np.abs(velocity[settled, 1] - 0.4))) if settled.any() else float("inf")
        metrics.update(command_mae_m_s=mae, required_progress_y_m=0.3 * (case.duration_s - 2.0))
        gates.update(progress=bool(metrics["progress_y_m"] >= metrics["required_progress_y_m"] - 1e-9),
                     velocity_tracking=bool(mae <= 0.15 + 1e-9))
        reason_codes.update(progress="insufficient_progress", velocity_tracking="velocity_tracking_error")
    elif case.name == "stop_restart":
        segments = []
        for i, segment in enumerate(case.commands):
            mask = (time >= segment.start_s + _SETTLE_S) & (time < segment.end_s)
            start = min(int(np.searchsorted(time, segment.start_s)), len(time) - 1)
            end = min(int(np.searchsorted(time, segment.end_s)), len(time) - 1)
            travel = float(position[end, 1] - position[start, 1])
            row = dict(start_s=segment.start_s, end_s=segment.end_s, progress_y_m=travel)
            if segment.velocity[1] != 0.0:
                mae = float(np.mean(np.abs(velocity[mask, :2] - segment.velocity[:2]))) if mask.any() else float("inf")
                required = 0.3 * (segment.end_s - segment.start_s - _SETTLE_S)
                gates[f"walk_progress_{i}"] = bool(travel >= required - 1e-9)
                gates[f"walk_velocity_{i}"] = bool(mae <= 0.15 + 1e-9)
                reason_codes[f"walk_progress_{i}"] = "initial_walk_progress_missing" if i == 0 else "restart_progress_missing"
                reason_codes[f"walk_velocity_{i}"] = "velocity_tracking_error"
                row.update(command_mae_m_s=mae, required_progress_y_m=required)
            else:
                speed = float(np.linalg.norm(velocity[mask, :2], axis=1).max()) if mask.any() else float("inf")
                samples = position[mask, :2]
                drift = float(np.linalg.norm(samples - samples[0], axis=1).max()) if mask.any() else float("inf")
                gates[f"stop_speed_{i}"] = bool(speed <= 0.10 + 1e-9)
                gates[f"stop_drift_{i}"] = bool(drift <= 0.20 + 1e-9)
                reason_codes[f"stop_speed_{i}"] = "stop_speed_exceeded"
                reason_codes[f"stop_drift_{i}"] = "stop_drift_exceeded"
                row.update(max_settled_speed_m_s=speed, max_settled_drift_m=drift)
            segments.append(row)
        metrics["segments"] = segments
    elif case.terrain.kind in ("stairs_up", "stairs_down", "rough"):
        terrain = case.terrain
        gates["corridor"] = bool(metrics["max_abs_x_m"] <= 0.70 + 1e-9)
        reason_codes["corridor"] = "outside_corridor"
        if terrain.kind in ("stairs_up", "stairs_down"):
            required = terrain.required_finish_y_m
            expected_change = terrain.full_rise_m * (-1.0 if terrain.kind == "stairs_down" else 1.0)
            gates["stair_traversal"] = bool(position[-1, 1] >= required - 1e-9)
            gates["ground_change"] = bool(abs(metrics["ground_height_change_m"] - expected_change) <= 0.02 + 1e-9)
            reason_codes.update(stair_traversal="insufficient_stair_traversal", ground_change="wrong_ground_change")
            metrics.update(required_finish_y_m=required, expected_ground_change_m=expected_change)
        else:
            required = terrain.rough_end_y_m + 0.5
            gates["rough_traversal"] = bool(position[-1, 1] >= required - 1e-9)
            reason_codes["rough_traversal"] = "insufficient_rough_traversal"
            metrics["required_finish_y_m"] = required
    else:
        command = case.commands[0].velocity
        mask = (time >= _SETTLE_S) & (time < case.duration_s)
        mae = float(np.mean(np.abs(velocity[mask, :2] - command[:2]))) if mask.any() else float("inf")
        metrics["command_mae_m_s"] = mae
        gates["velocity_tracking"] = bool(mae <= 0.15 + 1e-9)
        reason_codes["velocity_tracking"] = "velocity_tracking_error"
        if case.name == "yaw":
            expected_yaw = sum(segment.velocity[2] * max(0.0, min(time[-1], segment.end_s)
                               - max(time[0], segment.start_s)) for segment in case.commands)
            rates = np.gradient(yaw, time)
            rate_mae = float(np.mean(np.abs(rates[mask] - command[2]))) if mask.any() else float("inf")
            error = abs(metrics["actual_yaw_change_rad"] - expected_yaw)
            gates["yaw_tracking"] = bool(error <= 0.60 + 1e-9 and rate_mae <= 0.15 + 1e-9)
            reason_codes["yaw_tracking"] = "yaw_tracking_error"
            metrics.update(expected_yaw_change_rad=float(expected_yaw), yaw_error_rad=float(error),
                           yaw_rate_mae_rad_s=rate_mae)
        else:
            axis = 1 if case.name == "reverse" else 0
            directed = float((position[-1, axis] - position[0, axis]) * np.sign(command[axis]))
            required = 0.7 * abs(command[axis]) * max(0.0, case.duration_s - _SETTLE_S)
            gates["directed_progress"] = bool(directed >= required - 1e-9)
            reason_codes["directed_progress"] = "insufficient_directed_progress"
            metrics.update(directed_progress_m=directed, required_directed_progress_m=required)

    reasons = list(dict.fromkeys(reason_codes[key] for key, passed in gates.items() if not passed))
    # Missing settled windows fail their gates but have no numeric metric.
    # Keep early-failure reports valid with json.dumps(..., allow_nan=False).
    for row in [metrics] + metrics.get("segments", []):
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    return dict(case=case.name, motion_success=bool(all(gates.values())),
                reasons=reasons, metrics=metrics, gates=gates, skipped=False)
