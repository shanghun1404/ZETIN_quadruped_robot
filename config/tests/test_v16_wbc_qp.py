"""Whole-body acceleration/force algebra; no simulator or offline optimizer."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


def wbc_module():
    path = Path(__file__).resolve().parents[1] / "v16_wbc_qp.py"
    if not path.exists():
        raise AssertionError("Required joint acceleration/force optimizer is missing")
    spec = importlib.util.spec_from_file_location("v16_wbc_qp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(dtype=np.float64, lever=0.):
    # Synthetic floating mass, four colocated translational contacts. With
    # lever=0 contacts fix base translation. With lever=.25 each independent
    # joint triplet can compensate the base acceleration: a_b+.25*a_j=0.
    mass = 40.
    M = np.diag([mass]*3 + [2.]*3 + [.02]*12).astype(dtype)
    h = np.zeros(18, dtype=dtype)
    h[2] = mass * 9.81
    J = np.zeros((4, 3, 18), dtype=dtype)
    for foot in range(4):
        J[foot, :, :3] = np.eye(3)
        J[foot, :, 6+3*foot:9+3*foot] = lever * np.eye(3)
    return dict(M=M, h=h, Jcontact=J, desired_acceleration=np.zeros(18, dtype=dtype),
                stance=np.ones(4, dtype=bool), contact_target=np.zeros((4,3), dtype=dtype),
                limits=np.full(12, 48., dtype=dtype), tau_lower=np.full(12, -43.2, dtype=dtype),
                tau_upper=np.full(12, 43.2, dtype=dtype), weight=mass*9.81)


class WholeBodyQP(unittest.TestCase):
    def test_fixed_root_mass_equilibrium_has_equal_hand_checked_support(self):
        a, forces, tau, diagnostics, state = wbc_module().solve_wbc_qp(**fixture())
        np.testing.assert_allclose(a, 0., atol=.003)
        np.testing.assert_allclose(forces, [[0.,0.,98.1]]*4, atol=.08)
        np.testing.assert_allclose(tau, 0., atol=.003)
        self.assertTrue(diagnostics["finite"])
        self.assertTrue(diagnostics["feasible"])
        self.assertTrue(diagnostics["converged"])
        self.assertEqual(state["x"].shape, (30,))
        self.assertEqual(state["z"].shape, (58,))
        self.assertEqual(state["y"].shape, (58,))

    def test_roll_moment_uses_world_point_levers_and_correct_contact_force_sign(self):
        data=fixture()
        # Canonical FL/FR are at y=+.2, RL/RR at y=-.2. A 4 Nm
        # required roll moment needs front/rear totals differing by 20 N.
        for foot,(x,y,z) in enumerate(((-.3,.2,-.5),(.3,.2,-.5),(-.3,-.2,-.5),(.3,-.2,-.5))):
            data["Jcontact"][foot,:,3:6]=[[0.,z,-y],[-z,0.,x],[y,-x,0.]]
        data["h"][3]=4.
        a,forces,_,diagnostics,_=wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        np.testing.assert_allclose(a[:6],0.,atol=.003)
        np.testing.assert_allclose(forces[:,2],[103.1,103.1,93.1,93.1],atol=.08)
        self.assertLess(abs(diagnostics["base_wrench_error"][3]),.02)

    def test_nonzero_contact_target_acceleration_has_correct_equality_sign(self):
        data=fixture()
        data["contact_target"][:]=[.1,.2,-.2]
        a,forces,_,diagnostics,_=wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        np.testing.assert_allclose(a[:3],[.1,.2,-.2],atol=.003)
        np.testing.assert_allclose(forces.sum(axis=0),[4.,8.,384.4],atol=.08)

    def test_body_tracking_priority_can_dominate_conflicting_joint_tracking(self):
        data=fixture(lever=.25)
        data["desired_acceleration"][1]=1.
        high,_,_,high_diagnostics,_=wbc_module().solve_wbc_qp(**data)
        low,_,_,low_diagnostics,_=wbc_module().solve_wbc_qp(**data,
            acceleration_weights=np.array([.001]*6+[1.]*12))
        self.assertTrue(high_diagnostics["feasible"] and low_diagnostics["feasible"])
        self.assertGreater(high[1],.9)
        self.assertLess(low[1],.5)

    def test_joint_bias_and_reaction_torque_use_the_full_eliminated_equation(self):
        data=fixture(lever=.25)
        data["h"][6:]=np.linspace(-3.,3.,12)
        a,forces,tau,diagnostics,_=wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        expected=(data["M"]@a+data["h"])[6:]-np.einsum("fij,fi->j",data["Jcontact"][...,6:],forces)
        np.testing.assert_allclose(tau,expected,atol=1e-12)

    def test_impossible_requested_cone_wrench_is_resolved_by_acceleration_relaxation(self):
        data = fixture(lever=.25)
        data["desired_acceleration"][1] = 10.
        # Holding a_y=10,a_z=0 needs Fy=400 N > (.6/sqrt2)*392.4 N.
        self.assertGreater(40*10, .6/np.sqrt(2)*data["weight"])
        a, forces, tau, diagnostics, _ = wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        self.assertLess(a[1], 6.)
        self.assertGreater(a[2], .5)
        self.assertLessEqual(abs(forces[:,1].sum()), .6/np.sqrt(2)*forces[:,2].sum()+.1)
        np.testing.assert_allclose(a[:3] + .25*a[6:].reshape(4,3), 0., atol=.01)
        self.assertLess(diagnostics["contact_acceleration_violation_m_s2"], .01)
        self.assertLess(diagnostics["torque_violation_nm"], .05)

    def test_velocity_slew_bounds_restrict_actual_torque_with_correct_sign(self):
        data = fixture(lever=.25)
        data["desired_acceleration"][1] = 10.
        data["tau_lower"][1::3] = -10.
        data["tau_upper"][1::3] = 10.
        a, forces, tau, diagnostics, _ = wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        self.assertLessEqual(a[1], 4.05)
        self.assertGreater(a[1], 2.)
        np.testing.assert_allclose(tau, (data["M"]@a+data["h"])[6:]
                                   - np.einsum("fij,fi->j",data["Jcontact"][...,6:],forces), atol=1e-10)
        self.assertLessEqual(abs(tau[1::3]).max(), 10.05)

    def test_asymmetric_torque_interval_is_not_replaced_by_absolute_limit(self):
        data = fixture(lever=.25)
        data["desired_acceleration"][1] = 10.
        data["tau_lower"][1::3], data["tau_upper"][1::3] = -6., -4.
        a, forces, tau, diagnostics, _ = wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        self.assertGreaterEqual(tau[1::3].min(), -6.05)
        self.assertLessEqual(tau[1::3].max(), -3.95)
        self.assertLess(a[1], 2.5)

    def test_no_contacts_allow_ballistic_fall_instead_of_fake_static_support(self):
        data = fixture()
        data["stance"][:] = False
        a, forces, _, diagnostics, _ = wbc_module().solve_wbc_qp(**data)
        self.assertTrue(diagnostics["feasible"])
        np.testing.assert_array_equal(forces, np.zeros((4,3)))
        np.testing.assert_allclose(a[:3], [0.,0.,-9.81], atol=.003)

    def test_inconsistent_active_contact_constraint_is_reported_without_certificate(self):
        data = fixture()
        data["Jcontact"][:] = 0.
        data["contact_target"][0,0] = 1.
        a, forces, tau, diagnostics, _ = wbc_module().solve_wbc_qp(**data)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(forces).all() and np.isfinite(tau).all())
        self.assertFalse(diagnostics["feasible"])
        self.assertFalse(diagnostics["converged"])
        self.assertAlmostEqual(diagnostics["contact_acceleration_violation_m_s2"], 1.)

    def test_disabled_nan_contact_rows_and_targets_do_not_poison_batch(self):
        data = fixture(np.float32)
        data["Jcontact"][2:] = np.nan
        data["contact_target"][2:] = np.nan
        data["stance"][2:] = False
        data["M"] = np.stack((data["M"],data["M"]))
        data["h"] = np.stack((data["h"],np.zeros(18,np.float32)))
        data["stance"] = np.array([[1,1,0,0],[0,0,0,0]],bool)
        originals = {k:v.copy() for k,v in data.items() if isinstance(v,np.ndarray)}
        a, forces, tau, diagnostics, _ = wbc_module().solve_wbc_qp(**data)
        self.assertEqual(a.shape,(2,18))
        self.assertEqual(forces.shape,(2,4,3))
        self.assertEqual(a.dtype,np.dtype("float32"))
        self.assertEqual(tau.dtype,np.dtype("float32"))
        np.testing.assert_array_equal(forces[:,2:],0.)
        np.testing.assert_array_equal(forces[1],0.)
        np.testing.assert_array_equal(diagnostics["contact_acceleration_error"][...,2:,:],0.)
        np.testing.assert_array_equal(diagnostics["feasible"],[True,True])
        for key,original in originals.items():
            np.testing.assert_array_equal(data[key],original)

    def test_warm_state_is_nonmutating_and_zeroes_disabled_forces_after_contact_change(self):
        data = fixture()
        _,_,_,cold,state = wbc_module().solve_wbc_qp(**data,iterations=20)
        originals = {k:v.copy() for k,v in state.items()}
        _,_,_,warm,_ = wbc_module().solve_wbc_qp(**data,iterations=20,warm_start=state)
        self.assertLessEqual(warm["max_constraint_violation"],cold["max_constraint_violation"]+1e-6)
        for key,original in originals.items():
            np.testing.assert_array_equal(state[key],original)
        data["stance"][[0,2]] = False
        _,forces,_,diagnostics,_ = wbc_module().solve_wbc_qp(**data,warm_start=state)
        np.testing.assert_array_equal(forces[[0,2]],0.)
        self.assertTrue(diagnostics["feasible"])

    def test_global_force_torque_unit_scaling_preserves_acceleration(self):
        data = fixture(lever=.25)
        data["desired_acceleration"][1] = 3.
        a1,f1,tau1,_,_ = wbc_module().solve_wbc_qp(**data)
        for key in ("M","h","limits","tau_lower","tau_upper"):
            data[key] *= 2.
        data["weight"] *= 2.
        a2,f2,tau2,diagnostics,_ = wbc_module().solve_wbc_qp(**data,max_normal_force=1000.)
        np.testing.assert_allclose(a2,a1,atol=1e-9)
        np.testing.assert_allclose(f2,2*f1,atol=1e-8)
        np.testing.assert_allclose(tau2,2*tau1,atol=1e-8)
        self.assertTrue(diagnostics["feasible"])

    def test_one_iteration_reports_actual_returned_errors_not_box_variable(self):
        data = fixture()
        a,forces,tau,diagnostics,_ = wbc_module().solve_wbc_qp(**data,iterations=1)
        base = (data["M"]@a+data["h"])[:6]-np.einsum("fij,fi->j",data["Jcontact"][...,:6],forces)
        np.testing.assert_allclose(diagnostics["base_wrench_error"],base,atol=1e-12)
        self.assertFalse(diagnostics["feasible"])
        self.assertFalse(diagnostics["converged"])

    def test_invalid_shapes_units_options_and_active_nan_fail_before_solving(self):
        for update in ({"M":np.zeros((12,12))},{"h":np.zeros(12)},
                       {"Jcontact":np.zeros((4,6,18))},{"contact_target":np.zeros((4,2))},
                       {"weight":0.},{"stance":[2,1,1,1]},{"limits":np.zeros(12)},
                       {"tau_lower":np.full(12,50.)},{"desired_acceleration":np.full(18,np.nan)}):
            data = fixture(); data.update(update)
            with self.subTest(update=list(update)),self.assertRaises(ValueError):
                wbc_module().solve_wbc_qp(**data)
        for option in ({"rho":0.},{"iterations":0},{"mu":-1.},{"angular_length":0.},
                       {"over_relaxation":0.},{"over_relaxation":2.},
                       {"acceleration_scale":np.zeros(18)}, {"acceleration_weights":np.zeros(18)},
                       {"force_regularization":0.},{"torque_regularization":-1.},
                       {"contact_acceleration_scale":0.},{"warm_start":{"x":np.zeros(12),"z":np.zeros(58),"y":np.zeros(58)}}):
            with self.subTest(option=list(option)),self.assertRaises(ValueError):
                wbc_module().solve_wbc_qp(**fixture(),**option)
        data=fixture();data["Jcontact"][0,0,0]=np.nan
        with self.assertRaises(ValueError):
            wbc_module().solve_wbc_qp(**data)

    def test_numpy_torch_cpu_parity_dtype_device_and_nonmutating_warm_state(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch unavailable locally; verify this algebra case on HPC CPU")
        data=fixture(np.float32,lever=.25)
        data["desired_acceleration"][1]=10.
        expected=wbc_module().solve_wbc_qp(**data)
        tensors={k:torch.as_tensor(v) if isinstance(v,np.ndarray) else v for k,v in data.items()}
        a,forces,tau,diagnostics,state=wbc_module().solve_wbc_qp(**tensors)
        for result,reference in zip((a,forces,tau),expected[:3]):
            self.assertEqual(result.dtype,torch.float32)
            self.assertEqual(str(result.device),"cpu")
            np.testing.assert_allclose(result.numpy(),reference,atol=.04,rtol=2e-4)
        self.assertTrue(bool(diagnostics["feasible"]))
        original={k:v.clone() for k,v in state.items()}
        wbc_module().solve_wbc_qp(**tensors,warm_start=state)
        for key,value in original.items():
            torch.testing.assert_close(state[key],value,rtol=0.,atol=0.)


if __name__ == "__main__":
    unittest.main()
