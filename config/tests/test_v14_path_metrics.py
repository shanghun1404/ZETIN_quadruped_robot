import numpy as np

from v14_path_metrics import summarize_path


def rollout(x=0.0, y=2.5, ground=0.4, body=0.4, yaw=0.0, alive=True):
    positions = np.array([[[0., 0., 0.5]], [[x, y, 0.5 + body]]])
    quats = np.array([[[1., 0., 0., 0.]], [[np.cos(yaw/2), 0., 0., np.sin(yaw/2)]]])
    return dict(positions=positions, quaternions=quats,
                ground_height=np.array([[0.], [ground]]), valid=np.ones((2, 1), bool),
                alive_at_end=np.array([alive]), scenario="stairs_up", step_height_m=0.157,
                full_rise_m=1.099)


def test_forward_climb_passes_but_not_full_flight():
    r = summarize_path(**rollout())
    assert r["directed_stair_success_count"] == 1
    assert r["full_flight_success_count"] == 0
    assert r["per_environment"][0]["final_directed_progress_m"] == 2.5


def test_sideways_escape_and_returning_down_do_not_pass():
    assert summarize_path(**rollout(x=2., y=0.2))["directed_stair_success_count"] == 0
    assert summarize_path(**rollout(ground=0.))["directed_stair_success_count"] == 0
    assert summarize_path(**rollout(yaw=np.pi/2))["directed_stair_success_count"] == 0


def test_body_bounce_cannot_replace_ground_ascent():
    assert summarize_path(**rollout(ground=0., body=1.5))["directed_stair_success_count"] == 0


def test_failed_episode_ignores_post_reset_teleport():
    data = rollout(alive=False)
    data["positions"] = np.concatenate([data["positions"], [[[100., 100., 100.]]]])
    data["quaternions"] = np.concatenate([data["quaternions"], [[[1., 0., 0., 0.]]]])
    data["ground_height"] = np.concatenate([data["ground_height"], [[100.]]])
    data["valid"] = np.array([[True], [True], [False]])
    r = summarize_path(**data)
    assert r["per_environment"][0]["final_directed_progress_m"] == 2.5
    assert r["directed_stair_success_count"] == 0


def test_descent_and_complete_flight_use_signed_ground_change():
    data = rollout(y=3.8, ground=-1.10, body=-1.1)
    data["scenario"] = "stairs_down"
    r = summarize_path(**data)
    assert r["directed_stair_success_count"] == 1
    assert r["full_flight_success_count"] == 1


def test_nonfinite_valid_terrain_is_not_silently_accepted():
    import pytest
    data = rollout(ground=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        summarize_path(**data)


def test_stair_success_requires_known_positive_geometry():
    import pytest
    data = rollout(y=3.8, ground=0.)
    data["step_height_m"] = data["full_rise_m"] = 0.
    with pytest.raises(ValueError, match="positive"):
        summarize_path(**data)
