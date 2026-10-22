"""CPU contracts for the actual +Y CAD terrain and motion gate."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from v15_cad_suite_protocol import (
    CASES, case_config, command_at, evaluate_case, get_case, ground_height,
    terrain_cuboids,
)


def trajectory(case_name, *, y_end=0.0, x_end=0.0, yaw_rate=0.0):
    case = get_case(case_name)
    time = np.arange(0.0, case.duration_s + 0.05, 0.05)
    positions = np.zeros((len(time), 3))
    positions[:, 0] = np.linspace(0.0, x_end, len(time))
    positions[:, 1] = np.linspace(0.0, y_end, len(time))
    positions[:, 2] = ground_height(case, positions[:, 0], positions[:, 1]) + 0.4
    quaternion = np.zeros((len(time), 4))
    quaternion[:, 0] = np.cos(yaw_rate * time / 2.0)
    quaternion[:, 3] = np.sin(yaw_rate * time / 2.0)
    velocity = np.asarray([command_at(case, t)[:2] + (0.0,) for t in time])
    return dict(time_s=time, positions=positions, quaternions=quaternion,
                body_velocity=velocity)


def test_cases_are_immutable_and_configuration_is_json_serializable():
    case = get_case("stairs_up_10cm")
    with pytest.raises(FrozenInstanceError):
        case.duration_s = 1.0
    with pytest.raises(TypeError):
        CASES["flat_14s"] = case
    config = json.loads(json.dumps(case_config(case)))
    assert config["terrain"]["full_rise_m"] == pytest.approx(0.4)
    assert config["terrain"]["last_riser_y_m"] == pytest.approx(2.9)
    assert config["terrain"]["required_finish_y_m"] == pytest.approx(3.9)


def test_stop_restart_command_changes_at_exact_segment_boundaries():
    for t, expected in [(0.0, (0.0, .4, 0.0)), (7.999, (0.0, .4, 0.0)),
                        (8.0, (0.0, 0.0, 0.0)), (12.0, (0.0, .4, 0.0)),
                        (20.0, (0.0, 0.0, 0.0)), (24.0, (0.0, 0.0, 0.0))]:
        assert command_at("stop_restart", t) == expected
    assert command_at("flat_14s", -1.0) == (0.0, 0.0, 0.0)
    assert command_at("flat_14s", 14.0) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("case_name, expected", [
    ("stairs_up_10cm", [0.0, .1, .1, .2, .2, .3, .4, .4]),
    ("stairs_down_10cm", [.4, .3, .3, .2, .2, .1, 0.0, 0.0]),
])
def test_ground_height_uses_four_exact_risers_and_float_tolerant_edges(case_name, expected):
    y = np.array([1.999999, 2.0, 2.0 - 1e-12, 2.3, 2.3 - 1e-12, 2.6, 2.9, 4.0])
    np.testing.assert_allclose(ground_height(case_name, np.zeros(len(y)), y), expected)
    assert ground_height(case_name, 1.01, 2.4) == 0.0


@pytest.mark.parametrize("case_name", ["stairs_up_05cm", "stairs_up_10cm",
                                       "stairs_up_157mm", "stairs_down_05cm",
                                       "stairs_down_10cm", "stairs_down_157mm"])
def test_ground_height_matches_spawned_cuboid_tops(case_name):
    for cube in terrain_cuboids(case_name):
        x, y, z = cube.center
        assert ground_height(case_name, x, y) == pytest.approx(z + cube.size[2] / 2.0)
    case = get_case(case_name)
    expected_initial = case.terrain.full_rise_m if "down" in case_name else 0.0
    assert ground_height(case, 0.0, 0.0) == pytest.approx(expected_initial)


def test_up_platform_continues_to_the_end_of_the_test_corridor():
    assert ground_height("stairs_up_10cm", 0.0, 9.9) == pytest.approx(.4)
    assert ground_height("stairs_up_10cm", 0.0, 10.01) == 0.0


@pytest.mark.parametrize("case_name", ["stairs_up_10cm", "stairs_down_10cm"])
def test_surviving_at_stair_base_is_not_success(case_name):
    result = evaluate_case(case_name, **trajectory(case_name, y_end=1.8))
    assert not result["motion_success"]
    assert "insufficient_stair_traversal" in result["reasons"]
    assert "wrong_ground_change" in result["reasons"]


@pytest.mark.parametrize("case_name, change", [
    ("stairs_up_10cm", .4), ("stairs_down_10cm", -.4),
    ("stairs_up_157mm", .628), ("stairs_down_157mm", -.628),
])
def test_traversing_the_whole_stair_flight_passes(case_name, change):
    result = evaluate_case(case_name, **trajectory(case_name, y_end=4.1))
    assert result["motion_success"], result
    assert result["metrics"]["ground_height_change_m"] == pytest.approx(change)
    assert result["skipped"] is False


def test_stair_traversal_outside_corridor_fails_even_if_it_returns():
    frames = trajectory("stairs_up_10cm", y_end=4.1)
    frames["positions"][len(frames["positions"]) // 2, 0] = .71
    result = evaluate_case("stairs_up_10cm", **frames)
    assert "outside_corridor" in result["reasons"]
    assert not result["motion_success"]


def test_stair_torso_height_is_relative_to_known_local_ground():
    frames = trajectory("stairs_up_10cm", y_end=4.1)
    frames["positions"][-1, 2] = .59
    result = evaluate_case("stairs_up_10cm", **frames)
    assert "low_torso" in result["reasons"]
    assert result["metrics"]["torso_min_height_above_ground_m"] == pytest.approx(.19)


def test_flat_preserves_existing_progress_gate_and_command_mae_gate():
    frames = trajectory("flat_14s", y_end=3.59)
    assert not evaluate_case("flat_14s", **frames)["motion_success"]
    frames["positions"][:, 1] = np.linspace(0.0, 3.6, len(frames["positions"]))
    frames["body_velocity"][:, 1] = .551
    result = evaluate_case("flat_14s", **frames)
    assert not result["motion_success"]
    assert "velocity_tracking_error" in result["reasons"]
    frames["body_velocity"][:, 1] = .4
    assert evaluate_case("flat_14s", **frames)["motion_success"]


def test_flat_excludes_the_two_second_sample_from_the_historical_mae_gate():
    # A zero-error sample at exactly 2.000s would dilute a failing MAE below
    # .15. The original [400:] slice starts at post-state 2.005s instead.
    time = np.arange(1, 2801, dtype=np.int64) * 5_000_000 * 1e-9
    positions = np.zeros((len(time), 3))
    positions[:, 1] = np.linspace(0.0, 5.6, len(time))
    positions[:, 2] = .4
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (len(time), 1))
    velocity = np.zeros_like(positions)
    velocity[:, 1] = .55005
    velocity[399, 1] = .4
    result = evaluate_case("flat_14s", time_s=time, positions=positions,
                           quaternions=quaternion, body_velocity=velocity)
    assert not result["motion_success"]
    assert result["metrics"]["command_mae_m_s"] == pytest.approx(.15005)
    assert "velocity_tracking_error" in result["reasons"]


def test_stop_restart_rejects_drift_after_two_second_settle():
    frames = trajectory("stop_restart")
    time = frames["time_s"]
    positions = frames["positions"]
    positions[:, 1] = .4 * np.minimum(time, 8.0) + .4 * np.clip(time - 12.0, 0.0, 8.0)
    assert evaluate_case("stop_restart", **frames)["motion_success"]
    frames["body_velocity"][(time >= 10.0) & (time < 12.0), 1] = .11
    result = evaluate_case("stop_restart", **frames)
    assert not result["motion_success"]
    assert "stop_speed_exceeded" in result["reasons"]


def test_stop_restart_requires_motion_in_both_walk_segments():
    frames = trajectory("stop_restart")
    time = frames["time_s"]
    frames["positions"][:, 1] = .4 * np.minimum(time, 8.0)
    result = evaluate_case("stop_restart", **frames)
    assert not result["motion_success"]
    assert "restart_progress_missing" in result["reasons"]


def test_stop_drift_uses_position_as_well_as_reported_body_velocity():
    frames = trajectory("stop_restart")
    time = frames["time_s"]
    frames["positions"][:, 1] = (.4 * np.minimum(time, 8.0)
                                + .4 * np.clip(time - 12.0, 0.0, 8.0)
                                + .2 * np.clip(time - 10.0, 0.0, 2.0))
    result = evaluate_case("stop_restart", **frames)
    assert not result["motion_success"]
    assert "stop_drift_exceeded" in result["reasons"]


def test_stop_restart_reports_velocity_error_without_claiming_missing_progress():
    frames = trajectory("stop_restart")
    time = frames["time_s"]
    frames["positions"][:, 1] = .4 * np.minimum(time, 8.0) + .4 * np.clip(time - 12.0, 0.0, 8.0)
    frames["body_velocity"][(time >= 2.0) & (time < 8.0), 1] = .75
    result = evaluate_case("stop_restart", **frames)
    assert not result["motion_success"]
    assert "velocity_tracking_error" in result["reasons"]
    assert "initial_walk_progress_missing" not in result["reasons"]


def test_early_failure_metrics_can_be_written_as_strict_json():
    frames = trajectory("stop_restart")
    frames = {key: value[:10] for key, value in frames.items()}
    result = evaluate_case("stop_restart", **frames, alive_at_end=False)
    assert not result["motion_success"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case_name, kwargs", [
    ("reverse", dict(y_end=-4.2)), ("lateral", dict(x_end=2.8)),
    ("yaw", dict(yaw_rate=.25)),
])
def test_direction_profiles_require_actual_commanded_motion(case_name, kwargs):
    assert evaluate_case(case_name, **trajectory(case_name, **kwargs))["motion_success"]
    result = evaluate_case(case_name, **trajectory(case_name))
    assert not result["motion_success"]


def test_yaw_unwraps_beyond_pi_and_rejects_wrong_turn_direction():
    frames = trajectory("yaw", yaw_rate=-.25)
    result = evaluate_case("yaw", **frames)
    assert not result["motion_success"]
    assert "yaw_tracking_error" in result["reasons"]
    assert result["metrics"]["actual_yaw_change_rad"] == pytest.approx(-3.5)


def test_rough_tiles_are_seeded_and_verdict_requires_crossing_them():
    cubes = terrain_cuboids("rough")
    assert cubes == terrain_cuboids("rough")
    assert len(cubes) > 10
    assert all(0.0 < cube.size[2] <= .04 for cube in cubes)
    assert not evaluate_case("rough", **trajectory("rough", y_end=3.0))["motion_success"]
    assert evaluate_case("rough", **trajectory("rough", y_end=7.0))["motion_success"]


def test_incomplete_recording_and_fallen_robot_cannot_pass_motion():
    frames = trajectory("flat_14s", y_end=5.6)
    frames = {key: value[:100] for key, value in frames.items()}
    assert "incomplete_recording" in evaluate_case("flat_14s", **frames)["reasons"]
    frames = trajectory("flat_14s", y_end=5.6)
    result = evaluate_case("flat_14s", **frames, alive_at_end=False)
    assert not result["motion_success"]
    assert "did_not_survive" in result["reasons"]


def test_tilt_above_sixty_degrees_is_rejected():
    frames = trajectory("flat_14s", y_end=5.6)
    angle = np.radians(61.0)
    frames["quaternions"][-1] = (np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0)
    assert "excessive_tilt" in evaluate_case("flat_14s", **frames)["reasons"]


def test_malformed_or_nonfinite_trajectory_is_rejected():
    frames = trajectory("flat_14s", y_end=5.6)
    frames["time_s"][2] = frames["time_s"][1]
    with pytest.raises(ValueError, match="time"):
        evaluate_case("flat_14s", **frames)
    frames = trajectory("flat_14s", y_end=5.6)
    frames["body_velocity"][3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_case("flat_14s", **frames)
