import pytest

from v11_video_geometry import (
    stair_descent_passed,
    stair_rollout_passed,
    stair_step_height_m,
    tracking_camera_pose,
)


def test_level_five_is_a_physical_15_7_cm_step():
    height = stair_step_height_m((0.05, 0.20), levels=8, level=5)

    assert height == pytest.approx(0.15714285714285717)


def test_stair_height_rejects_a_level_outside_the_generator():
    with pytest.raises(ValueError, match="level must be between 0 and 7"):
        stair_step_height_m((0.05, 0.20), levels=8, level=8)


def test_side_camera_tracks_a_robot_whose_forward_axis_is_positive_y():
    eye, target = tracking_camera_pose((1.0, 2.0, 0.5), "side")

    assert eye == (-1.6, 2.0, 1.5)
    assert target == (1.0, 2.2, 0.55)


def test_rear_quarter_camera_stays_behind_positive_y_motion():
    eye, target = tracking_camera_pose((1.0, 2.0, 0.5), "rear_quarter")

    assert eye == pytest.approx((-0.8, -0.2, 1.9))
    assert target == pytest.approx((1.0, 2.4, 0.55))


def test_camera_rejects_an_unknown_view():
    with pytest.raises(ValueError, match="unsupported view"):
        tracking_camera_pose((0.0, 0.0, 0.0), "front")


def test_rollout_pass_requires_survival_approach_and_two_steps_of_height():
    assert stair_rollout_passed(
        alive=True,
        final_progress_m=2.1,
        peak_height_after_approach_m=0.32,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    assert not stair_rollout_passed(
        alive=False,
        final_progress_m=3.0,
        peak_height_after_approach_m=0.40,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )


def test_descent_pass_requires_survival_approach_and_two_steps_of_drop():
    assert stair_descent_passed(
        alive=True,
        final_progress_m=2.1,
        max_descent_after_approach_m=0.32,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    assert not stair_descent_passed(
        alive=False,
        final_progress_m=3.0,
        max_descent_after_approach_m=0.40,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    assert not stair_descent_passed(
        alive=True,
        final_progress_m=2.09,
        max_descent_after_approach_m=0.40,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    assert not stair_descent_passed(
        alive=True,
        final_progress_m=3.0,
        max_descent_after_approach_m=0.313,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    assert not stair_rollout_passed(
        alive=True,
        final_progress_m=2.09,
        peak_height_after_approach_m=0.40,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
    assert not stair_rollout_passed(
        alive=True,
        final_progress_m=3.0,
        peak_height_after_approach_m=0.313,
        approach_distance_m=2.1,
        step_height_m=0.157,
    )
