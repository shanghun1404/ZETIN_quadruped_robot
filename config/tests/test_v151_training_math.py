"""Hand-calculated load fixtures catch peak-normalization and L2-norm regressions."""

import numpy as np
import pytest


def test_squared_rated_load_has_quadratic_cost_and_sign_symmetry():
    from v151_training_math import rated_squared_load

    # Equal rated fraction across different motors must contribute equally.
    torque = np.array([[8.3, 18.0, -18.0], [4.15, -9.0, 9.0], [16.6, 36.0, -36.0]])
    ratings = np.array([8.3, 18.0, 18.0])
    np.testing.assert_allclose(rated_squared_load(torque, ratings), [1.0, 0.25, 4.0])


def test_load_integrates_sustained_demand_without_joints_canceling():
    from v151_training_math import rated_squared_load

    ratings = np.array([8.3, 18.0])
    costs = rated_squared_load(np.array([[8.3, -18.0], [0.0, 0.0], [0.0, 0.0]]), ratings)
    # Three equal-duration samples: a short full-load pulse costs 1/3 of sustained load.
    assert costs.mean() == pytest.approx(1.0 / 3.0)
    assert rated_squared_load(np.array([[8.3, 0.0]]), ratings)[0] == pytest.approx(0.5)


def test_ratings_follow_joint_names_instead_of_assuming_joint_order():
    from v151_training_math import joint_ratings_nm

    assert joint_ratings_nm(["RR_KFE", "FL_HAA", "RL_HFE"]) == (18.0, 8.3, 18.0)
    with pytest.raises(ValueError):
        joint_ratings_nm(["FL_UNKNOWN"])


def test_rejects_mismatched_motor_dimensions():
    from v151_training_math import rated_squared_load

    with pytest.raises(ValueError):
        rated_squared_load(np.zeros((2, 12)), np.ones(3))
