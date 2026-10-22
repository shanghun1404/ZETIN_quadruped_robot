"""Guard, digital poles and selection failures, independent hand-derived cases."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_torque_envelope_allows_braking_but_not_overspeed_acceleration():
    from v16_controller_math import limit_torque
    result = limit_torque(np.array([[100., -100., 100., -100.]]),
                          np.array([[0., 0., 50., -50.]]), np.full(4, 24.8),
                          np.zeros((1, 4)), dt=.02, fraction=.9, slew_nm_s=10000.)
    np.testing.assert_allclose(result, [[22.32, -22.32, 0., 0.]])
    braking = limit_torque(np.array([[-100., 100.]]), np.array([[70., -70.]]),
                           np.array([24.8, 24.8]), np.zeros((1, 2)), dt=.02,
                           fraction=.9, slew_nm_s=10000.)
    np.testing.assert_allclose(braking, [[-22.32, 22.32]])


def test_new_velocity_envelope_has_priority_over_slew_memory():
    from v16_controller_math import limit_torque
    result = limit_torque(np.array([[20.]]), np.array([[50.]]), np.array([24.8]),
                          np.array([[20.]]), dt=.02, fraction=.9, slew_nm_s=1.)
    np.testing.assert_allclose(result, [[0.]])


def test_invalid_torque_cannot_pass_safety_guard():
    from v16_controller_math import limit_torque
    with pytest.raises(ValueError):
        limit_torque(np.array([[np.nan]]), np.zeros((1, 1)), np.ones(1),
                     np.zeros((1, 1)), dt=.02)


def test_discrete_feedback_has_prescribed_continuous_poles_after_zoh():
    from v16_controller_math import digital_acceleration_gains
    kp, kd = digital_acceleration_gains(10., 1., .02)
    A = np.array([[1., .02], [0., 1.]])
    B = np.array([[.0002], [.02]])
    closed = A - B @ np.array([[kp, kd]])
    np.testing.assert_allclose(np.linalg.eigvals(closed), np.exp(-.2), atol=1e-7)


def test_fall_and_stationary_low_torque_cannot_win_walk_selection():
    from v16_controller_math import candidate_score
    good = candidate_score(1., 1.5, .4, .05, .02, 20., .4, 4.)
    stationary = candidate_score(1., 0., 0., 0., 0., 0., .4, 4.)
    fallen = candidate_score(.3, 3., .4, 0., 0., 0., .4, 4.)
    assert good > stationary > fallen


def test_command_tracking_beats_overspeed_with_equal_effort():
    from v16_controller_math import candidate_score
    tracking = candidate_score(1., 1., .25, .1, .1, 10., .25, 4.)
    overspeed = candidate_score(1., 1.25, .3125, .1, .1, 10., .25, 4.)
    assert tracking > overspeed


def test_cem_update_keeps_every_parameter_in_bounds():
    from v16_controller_math import cem_update
    mean, std = cem_update(np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
                           np.array([-4., -3., 3., 4.]), np.array([0., 0.]),
                           np.array([2., 2.]), elite_fraction=.5)
    np.testing.assert_allclose(mean, [2., 2.])
    assert np.all(std >= .02)


def test_late_fall_cannot_win_even_with_99_percent_survival():
    from v16_controller_math import candidate_score
    good = candidate_score(1., 1., .25, .1, .1, 10., .25, 4., ended_alive=True)
    late_fall = candidate_score(.995, 2., .25, 0., 0., 0., .25, 4., ended_alive=False)
    assert late_fall < good


def test_swing_reference_has_continuous_endpoints_and_zero_endpoint_velocity():
    from v16_controller_math import swing_reference
    start, end = np.array([[1., 2., .03]]), np.array([[1.2, 2., .03]])
    np.testing.assert_allclose(swing_reference(start, end, np.array([0.]), np.array([.06])), start)
    np.testing.assert_allclose(swing_reference(start, end, np.array([1.]), np.array([.06])), end)
    np.testing.assert_allclose(swing_reference(start, end, np.array([.5]), np.array([.06])), [[1.1, 2., .09]])
    np.testing.assert_allclose((swing_reference(start, end, np.array([1e-6]), np.array([.06])) - start) / 1e-6,
                               0., atol=1e-5)


def test_speed_envelope_clipping_is_recorded_below_static_peak():
    from v16_controller_math import guard_activity
    active = guard_activity(np.array([[100.]]), np.array([[20.088]]))
    np.testing.assert_array_equal(active, [[True]])


def test_failed_nonfinite_metric_is_zero_while_other_environments_are_preserved():
    from v16_controller_math import mask_failed_metric
    values = np.array([np.nan, 3., 7.])
    result = mask_failed_metric(values, np.array([False, True, True]))
    np.testing.assert_array_equal(result, [0., 3., 7.])
    assert np.isnan(values[0])  # callers' evidence is not silently overwritten


def test_cem_resume_restores_random_stream_and_rejects_changed_objective():
    from v16_controller_math import resume_cem, PARAM_NAMES, DEFAULT
    rng = np.random.default_rng(51)
    rng.normal(size=7)
    state = {"schema": 2, "generation": 3, "parameter_names": list(PARAM_NAMES),
             "mean": DEFAULT.tolist(),
             "std": [.05] * 12, "best_parameters": DEFAULT.tolist(),
             "best_score": 10., "training_seed": 51, "rollout_seconds": 6., "control_dt_s": .02,
             "speed_command_m_s": .25, "effort_fraction": .90, "controller": "model",
             "walk_axis_world": "y",
             "rng_state": rng.bit_generator.state, "history": [{"generation": 3}]}
    restored = resume_cem(state, seed=51, seconds=6., control_dt=.02)
    np.testing.assert_array_equal(restored["rng"].normal(size=5), rng.normal(size=5))
    assert restored["next_generation"] == 4
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=12., control_dt=.02)


def test_resume_rejects_changed_speed_effort_or_controller():
    from v16_controller_math import resume_cem, PARAM_NAMES, DEFAULT
    state = {"schema": 2, "generation": 3, "parameter_names": list(PARAM_NAMES),
             "mean": DEFAULT.tolist(), "std": (.1 * np.ones(12)).tolist(),
             "best_parameters": DEFAULT.tolist(), "best_score": 10., "training_seed": 51,
             "rollout_seconds": 6., "control_dt_s": .02, "speed_command_m_s": .25,
             "effort_fraction": .90, "controller": "model",
             "walk_axis_world": "y",
             "rng_state": np.random.default_rng(51).bit_generator.state, "history": []}
    for changed in ({"speed": .4}, {"effort_fraction": .95}, {"controller": "pd"}, {"walk_axis": "x"}):
        with pytest.raises(ValueError):
            resume_cem(state, seed=51, seconds=6., control_dt=.02, **changed)


def test_physx_joint_order_maps_states_and_generalized_dynamics_by_name():
    from v16_controller_math import joint_permutation
    actual = ['FL_HAA', 'FR_HAA', 'FL_HFE', 'FR_HFE', 'RR_HAA', 'RL_HAA',
              'RR_HFE', 'RL_HFE', 'FL_KFE', 'FR_KFE', 'RR_KFE', 'RL_KFE']
    joint_ids, generalized_ids = joint_permutation(actual)
    assert joint_ids == [0, 2, 8, 1, 3, 9, 5, 7, 11, 4, 6, 10]
    assert generalized_ids == [0, 1, 2, 3, 4, 5, 6, 8, 14, 7, 9, 15, 11, 13, 17, 10, 12, 16]
    with pytest.raises(ValueError):
        joint_permutation(actual[:-1] + ['FL_HAA'])


def test_armature_changes_joint_acceleration_and_preserves_base_blocks():
    from v16_controller_math import add_joint_armature
    rigid = np.diag(np.arange(1., 19.))
    rotor = np.full(12, .02)
    effective = add_joint_armature(rigid, rotor)
    np.testing.assert_array_equal(effective[:6], rigid[:6])
    np.testing.assert_allclose(np.diag(effective)[6:], np.arange(7., 19.) + .02)
    np.testing.assert_allclose(np.linalg.solve(effective, np.ones(18))[6:], 1. / (np.arange(7., 19.) + .02))
    np.testing.assert_array_equal(np.diag(rigid), np.arange(1., 19.))


def test_lateral_translation_cannot_count_as_forward_walking():
    from v16_controller_math import forward_metrics, candidate_score
    position, velocity, start = np.array([[1., 0., .45]]), np.array([[.5, 0., 0.]]), np.zeros((1, 2))
    progress, speed = forward_metrics(position, velocity, start, axis=1)
    assert candidate_score(1., progress[0], speed[0], 0., 0., 0., .25, 4.) < 0
    np.testing.assert_allclose(forward_metrics(position, velocity, start, axis=0), [[1.], [.5]])


def test_cem_resume_rejects_a_changed_contact_acceleration_controller():
    from v16_controller_math import resume_cem, PARAM_NAMES, DEFAULT
    state = {"schema": 2, "generation": 0, "parameter_names": list(PARAM_NAMES),
             "mean": DEFAULT.tolist(), "std": [.1] * 12, "best_parameters": DEFAULT.tolist(),
             "best_score": 0., "training_seed": 51, "rollout_seconds": 6., "control_dt_s": .02,
             "speed_command_m_s": .25, "effort_fraction": .9, "controller": "model",
             "walk_axis_world": "y", "contact_projection": False,
             "rng_state": np.random.default_rng(51).bit_generator.state, "history": []}
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, contact_projection=True)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, gait_mode="prepared_crawl")
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, stance_point="ground")
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, joint_friction=0.)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, force_allocator="qp")
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, joint_dynamic_friction=0.)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, joint_viscous_friction=0.)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, controller_code_sha256={"controller.py": "new-code"})
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, base_acceleration_weight=1000.)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, smooth_com_reference=True)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, wbc_iterations=80)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, candidate_repeats=4)
    with pytest.raises(ValueError):
        resume_cem(state, seed=51, seconds=6., control_dt=.02, wbc_tracking_metric="unscaled")


def test_contact_point_jacobian_predicts_velocity_at_same_rigid_point():
    from v16_controller_math import rigid_point_jacobian
    linear = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)
    angular = np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=1)
    point = rigid_point_jacobian(linear, angular, np.array([.1, 0., -.2]))
    velocity = np.array([1., 2., 3., 0., 2., 0.])
    np.testing.assert_allclose(point @ velocity, [.6, 2., 2.8])
    np.testing.assert_array_equal(linear[:, 3:], np.zeros((3, 3)))
    # A stationary ground point allows its COM to move as the foot rotates.
    np.testing.assert_allclose(point @ np.array([.4, 0., .2, 0., 2., 0.]), 0., atol=1e-12)


def test_wbc_normalization_preserves_the_declared_acceleration_cost_ratio():
    from v16_controller_math import wbc_acceleration_weights
    scale = np.array([3.] * 3 + [10.] * 3 + [50.] * 12)
    error = np.linspace(-2., 3., 18)
    weights = wbc_acceleration_weights(1000., "unscaled")
    expected = (1000. * np.sum(error[:6] ** 2) + np.sum(error[6:] ** 2)) / 2500.
    assert np.sum(weights * (error / scale) ** 2) == pytest.approx(expected)
    np.testing.assert_allclose(weights, [3.6] * 3 + [40.] * 3 + [1.] * 12)
    np.testing.assert_array_equal(wbc_acceleration_weights(1000., "normalized"), [1000.] * 6 + [1.] * 12)
    with pytest.raises(ValueError):
        wbc_acceleration_weights(0., "unscaled")
