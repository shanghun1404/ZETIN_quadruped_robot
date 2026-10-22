"""Force allocation algebra only; no simulator or offline optimizer dependency."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


def qp_module():
    path = Path(__file__).resolve().parents[1] / "v16_force_qp.py"
    if not path.exists():
        raise AssertionError("Required torque-constrained force allocator is missing")
    spec = importlib.util.spec_from_file_location("v16_force_qp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point_wrench_map(points):
    """Independent wrench definition: each point contributes [f; r cross f]."""
    matrix = np.zeros((6, 12))
    for i, (x, y, z) in enumerate(points):
        matrix[:3, 3*i:3*i+3] = np.eye(3)
        matrix[3:, 3*i:3*i+3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
    return matrix


def fixture(dtype=np.float64):
    A = point_wrench_map([[-.3,0,0], [.3,0,0], [0,.3,0], [0,-.3,0]]).astype(dtype)
    return dict(A=A, Jjoint_transpose=np.zeros((12,12), dtype=dtype),
                dynjoint=np.zeros(12, dtype=dtype),
                wantedwrench=np.array([0,0,100,0,0,0], dtype=dtype),
                stance=np.array([True,True,False,False]), limits=np.full(12, 10., dtype=dtype),
                tau_lower=np.full(12, -10., dtype=dtype), tau_upper=np.full(12, 10., dtype=dtype),
                weight=100.)


class ForceQP(unittest.TestCase):
    def test_symmetric_vertical_load_has_hand_checked_minimum_norm_solution(self):
        data = fixture()
        forces, diagnostics, state = qp_module().solve_force_qp(**data)
        np.testing.assert_allclose(forces, [[0,0,50], [0,0,50], [0,0,0], [0,0,0]], atol=.02)
        self.assertTrue(diagnostics["feasible"])
        self.assertLess(diagnostics["base_wrench_violation"], 1e-3)
        self.assertEqual(state["x"].shape, (12,))
        self.assertEqual(state["z"].shape, (46,))
        self.assertEqual(state["y"].shape, (46,))

    def test_internal_tangential_force_can_avoid_known_joint_torque_saturation(self):
        data = fixture()
        # Vertical equilibrium fixes fz=50 each. Least-force allocation fx=0
        # needs tau0=80-50=30 Nm. Opposite fx=+15,-15 cancels the base wrench
        # and reduces tau0=80-50-2*15 to zero, within +/-10 Nm.
        data["dynjoint"][0] = 80.
        data["Jjoint_transpose"][0, 0] = 2.
        data["Jjoint_transpose"][0, 2] = 1.
        forces, diagnostics, _ = qp_module().solve_force_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        self.assertGreater(forces[0,0], 10.)
        np.testing.assert_allclose(forces[0,0], -forces[1,0], atol=.02)
        np.testing.assert_allclose(forces[:2,2], [50.,50.], atol=.03)
        self.assertLess(abs(diagnostics["torque"][0]), .05)
        self.assertLess(diagnostics["torque_violation_nm"], .02)

    def test_velocity_and_slew_torque_bounds_are_applied_with_correct_sign(self):
        data = fixture()
        data["dynjoint"][0] = 80.
        data["Jjoint_transpose"][0, [0,2]] = [2.,1.]
        data["tau_lower"][0], data["tau_upper"][0] = 4., 6.
        forces, diagnostics, _ = qp_module().solve_force_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        self.assertGreaterEqual(diagnostics["torque"][0], 3.98)
        self.assertLessEqual(diagnostics["torque"][0], 6.02)
        self.assertAlmostEqual(forces[0,0], 13., delta=.04)

    def test_infeasible_base_wrench_is_reported_without_fake_success(self):
        data = fixture()
        data["stance"][:] = False
        forces, diagnostics, _ = qp_module().solve_force_qp(**data)
        np.testing.assert_array_equal(forces, np.zeros((4,3)))
        self.assertFalse(diagnostics["feasible"])
        self.assertAlmostEqual(diagnostics["base_wrench_violation"], 1.)
        self.assertTrue(np.isfinite(forces).all())

    def test_infeasible_torque_and_friction_constraints_have_separate_diagnostics(self):
        data = fixture()
        data["dynjoint"][0] = 30.  # J=0 makes this torque uncontrollable.
        _, diagnostics, _ = qp_module().solve_force_qp(**data)
        self.assertFalse(diagnostics["feasible"])
        self.assertAlmostEqual(diagnostics["torque_violation_nm"], 20.)
        data = fixture()
        data["wantedwrench"][0] = 80.  # > mu/sqrt(2)*100 = 42.426...
        _, diagnostics, _ = qp_module().solve_force_qp(**data)
        self.assertFalse(diagnostics["feasible"])
        self.assertGreater(diagnostics["max_constraint_violation"], .05)

    def test_disabled_nan_columns_do_not_poison_valid_batch_peer_or_mutate_inputs(self):
        data = fixture(np.float32)
        data["A"][:, 6:] = np.nan
        data["Jjoint_transpose"][:, 6:] = np.nan
        data["A"] = np.stack([data["A"], fixture(np.float32)["A"]])
        data["Jjoint_transpose"] = np.stack([data["Jjoint_transpose"], np.zeros((12,12),np.float32)])
        data["stance"] = np.array([[1,1,0,0], [0,0,0,0]], dtype=bool)
        data["wantedwrench"] = np.array([[0,0,100,0,0,0], [0,0,0,0,0,0]], np.float32)
        copies = {key: value.copy() for key,value in data.items() if isinstance(value,np.ndarray)}
        forces, diagnostics, _ = qp_module().solve_force_qp(**data)
        self.assertEqual(forces.shape, (2,4,3))
        self.assertEqual(forces.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(forces).all())
        np.testing.assert_array_equal(forces[:,2:], np.zeros((2,2,3)))
        np.testing.assert_array_equal(forces[1], np.zeros((4,3)))
        np.testing.assert_array_equal(diagnostics["feasible"], [True,True])
        for key, original in copies.items():
            np.testing.assert_array_equal(data[key], original)

    def test_angular_wrench_is_scaled_by_length_and_uses_actual_moment_arm(self):
        data = fixture()
        data["wantedwrench"][4] = 6.  # .3*fz_FL - .3*fz_FR = 6 Nm.
        forces, diagnostics, _ = qp_module().solve_force_qp(**data)
        np.testing.assert_allclose(forces[:2,2], [60.,40.], atol=.03)
        self.assertTrue(diagnostics["feasible"])

    def test_normal_force_cap_and_conservative_friction_pyramid_are_enforced(self):
        data = fixture()
        data["wantedwrench"][0:2] = [30.,30.]
        forces, diagnostics, _ = qp_module().solve_force_qp(**data, max_normal_force=55.)
        self.assertTrue(diagnostics["feasible"])
        np.testing.assert_allclose(forces[:2], [[15,15,50],[15,15,50]], atol=.03)
        self.assertLess(diagnostics["friction_violation_n"], .02)
        self.assertLess(diagnostics["normal_force_violation_n"], .02)
        data["wantedwrench"][2] = 120.
        _, diagnostics, _ = qp_module().solve_force_qp(**data, max_normal_force=55.)
        self.assertFalse(diagnostics["feasible"])

    def test_warm_start_is_nonmutating_and_not_confused_after_contact_change(self):
        data = fixture()
        cold, cold_diagnostics, state = qp_module().solve_force_qp(**data, iterations=20)
        snapshot = {key:value.copy() for key,value in state.items()}
        warm, warm_diagnostics, _ = qp_module().solve_force_qp(**data, iterations=20, warm_start=state)
        self.assertLessEqual(warm_diagnostics["max_constraint_violation"],
                             cold_diagnostics["max_constraint_violation"] + 1e-6)
        self.assertLess(abs(warm[0,2]-50), abs(cold[0,2]-50)+1e-6)
        for key, before in snapshot.items():
            np.testing.assert_array_equal(state[key], before)
        data["stance"][:] = False
        data["wantedwrench"][:] = 0.
        forces, diagnostics, _ = qp_module().solve_force_qp(**data, warm_start=state)
        np.testing.assert_array_equal(forces, np.zeros((4,3)))
        self.assertTrue(diagnostics["feasible"])

    def test_normalization_respects_force_and_torque_units_under_global_scaling(self):
        data = fixture()
        data["dynjoint"][0] = 80.
        data["Jjoint_transpose"][0,[0,2]] = [2.,1.]
        first, _, _ = qp_module().solve_force_qp(**data)
        for key in ("dynjoint", "wantedwrench", "limits", "tau_lower", "tau_upper"):
            data[key] = data[key] * 2.
        data["weight"] = 200.
        second, diagnostics, _ = qp_module().solve_force_qp(**data, max_normal_force=1000.)
        np.testing.assert_allclose(second, 2*first, atol=.03)
        self.assertTrue(diagnostics["feasible"])

    def test_per_environment_weight_and_joint_limits_broadcast_without_peer_mix(self):
        data = fixture(np.float32)
        data["weight"] = np.array([100.,200.], np.float32)
        data["wantedwrench"] = np.array([[0,0,100,0,0,0], [0,0,200,0,0,0]], np.float32)
        data["limits"] = np.array([[10.]*12,[20.]*12], np.float32)
        data["tau_lower"] = -data["limits"]
        data["tau_upper"] = data["limits"]
        forces, diagnostics, _ = qp_module().solve_force_qp(**data)
        np.testing.assert_allclose(forces[:, :2, 2], [[50.,50.],[100.,100.]], atol=.04)
        np.testing.assert_array_equal(diagnostics["feasible"], [True,True])

    def test_one_iteration_exposes_actual_returned_force_error_instead_of_projected_z(self):
        data = fixture()
        forces, diagnostics, _ = qp_module().solve_force_qp(**data, iterations=1)
        # The auxiliary box variable satisfies equality by projection, but
        # only the actually returned force enters the robot wrench.
        actual_vertical_error = abs(forces[:,2].sum()-100.)/100.
        self.assertGreater(actual_vertical_error, .01)
        self.assertAlmostEqual(diagnostics["base_wrench_violation"], actual_vertical_error)
        self.assertFalse(diagnostics["feasible"])
        self.assertFalse(diagnostics["converged"])

    def test_warm_state_survives_swap_to_different_canonical_support_legs(self):
        data = fixture()
        _, _, state = qp_module().solve_force_qp(**data)
        data["stance"] = np.array([False,False,True,True])
        forces, diagnostics, _ = qp_module().solve_force_qp(**data, warm_start=state)
        np.testing.assert_array_equal(forces[:2], np.zeros((2,3)))
        np.testing.assert_allclose(forces[2:,2], [50.,50.], atol=.03)
        self.assertTrue(diagnostics["feasible"])

    def test_bad_units_shapes_active_nan_and_options_fail_before_solve(self):
        for update in ({"weight":0.}, {"limits":np.zeros(12)}, {"stance":[2,1,0,0]},
                       {"A":np.zeros((12,6))}, {"tau_lower":np.full(12,11.)},
                       {"wantedwrench":np.full(6,np.nan)}):
            data = fixture()
            data.update(update)
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                qp_module().solve_force_qp(**data)
        for option in ({"rho":0.}, {"iterations":0}, {"mu":-1.},
                       {"angular_length":0.}, {"force_regularization":0.}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                qp_module().solve_force_qp(**fixture(), **option)
        data = fixture()
        data["A"][0,0] = np.nan
        with self.assertRaises(ValueError):
            qp_module().solve_force_qp(**data)

    def test_torch_dtype_device_parity_and_nonmutating_warm_start(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable locally; run this pure algebra case on HPC")
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            data = fixture(np.float32)
            data["dynjoint"][0] = 80.
            data["Jjoint_transpose"][0,[0,2]] = [2.,1.]
            expected, _, _ = qp_module().solve_force_qp(**data)
            tensors = {key:torch.as_tensor(value,device=device) if isinstance(value,np.ndarray) else value
                       for key,value in data.items()}
            forces, diagnostics, state = qp_module().solve_force_qp(**tensors)
            self.assertEqual(forces.dtype, torch.float32)
            self.assertEqual(str(forces.device), device if device == "cpu" else "cuda:0")
            np.testing.assert_allclose(forces.cpu().numpy(), expected, atol=.04)
            self.assertTrue(bool(diagnostics["feasible"]))
            original = {key:value.clone() for key,value in state.items()}
            qp_module().solve_force_qp(**tensors, warm_start=state)
            for key,value in original.items():
                torch.testing.assert_close(state[key], value, rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
