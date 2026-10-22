"""Independent algebra fixtures and HPC rigid-body checks for the V16 model.

Run real-model checks with V16_MJCF pointing at the source on the HPC.
Physics tests (including small fixtures) are submitted to Slurm, not run locally.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

CONFIG = Path(__file__).resolve().parents[1]


def load_module(name):
    path = CONFIG / (name + ".py")
    if not path.exists():
        raise AssertionError(f"Required V16 implementation is missing: {path.name}")
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FIXTURE = '''<mujoco><compiler angle="radian"/><default><joint damping="1.5" armature="0.02"/></default>
<worldbody><body name="base" pos="1 2 3"><freejoint name="root"/>
<inertial mass="2" pos="0 0 0" diaginertia="1 1 1"/>
<body name="rotor" pos="0 0 1" euler="0 0 1.5707963267948966">
<joint name="hinge" axis="0 1 0" pos="1 0 0"/>
<inertial mass="1" pos="2 0 0" fullinertia="0.1 0.2 0.3 0.01 0.02 0.03"/>
<body name="fixed" pos="0 2 0"><inertial mass="3" pos="0 0 0" diaginertia="0.4 0.5 0.6"/></body>
</body></body></worldbody><actuator><motor name="hinge" joint="hinge" ctrlrange="-5 5"/></actuator>
</mujoco>'''


@unittest.skipUnless(os.environ.get("V16_MJCF") or os.environ.get("V16_PHYSICS_TESTS"), "physical fixtures require HPC execution")
class SmallPhysicsFixture(unittest.TestCase):
    def setUp(self):
        self.module = load_module("v16_physical_model")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "fixture.xml"
        self.path.write_text(FIXTURE)

    def build(self, floating=False):
        return self.module.build_model(self.path, floating_base=floating, joint_names=("hinge",))

    def test_offset_axis_and_fixed_member_kinematics(self):
        # Catches dropping joint.pos, wrong axis frame, or dropping fixed offsets.
        model = self.build()
        np.testing.assert_allclose(model.frame_position([0.0], "fixed"), [-1, 2, 4], atol=1e-12)
        np.testing.assert_allclose(model.frame_position([np.pi / 2], "fixed"), [-1, 3, 5], atol=1e-12)
        jac = model.point_jacobian([0.37], "fixed", [0.2, -0.3, 0.4])
        eps = 1e-6
        fd = (model.frame_position([0.37 + eps], "fixed", [0.2, -0.3, 0.4]) -
              model.frame_position([0.37 - eps], "fixed", [0.2, -0.3, 0.4])) / (2 * eps)
        np.testing.assert_allclose(jac[:, 0], fd, rtol=1e-7, atol=1e-8)

    def test_fixed_mass_is_aggregated_once_and_armature_is_explicit(self):
        # Hinge about y through x=1: rotor 0.2+1, fixed 0.5+3; plus 0.02 armature.
        model = self.build()
        self.assertAlmostEqual(model.total_mass, 6.0)
        np.testing.assert_allclose(model.mass_matrix([0]), [[4.72]], atol=1e-12)
        np.testing.assert_allclose(model.mass_matrix([0], include_armature=False), [[4.7]], atol=1e-12)
        np.testing.assert_allclose(model.source_damping, [1.5])
        np.testing.assert_allclose(model.effort_limits, [5.0])

    def test_floating_base_has_no_virtual_actuator(self):
        model = self.build(True)
        self.assertEqual((model.model.nq, model.model.nv), (8, 7))
        np.testing.assert_array_equal(model.actuation_matrix, [[0, 0, 0, 0, 0, 0, 1]])
        np.testing.assert_allclose(model.frame_position([0], "fixed"), [-1, 2, 4], atol=1e-12)

    def test_intrinsic_euler_is_not_extrinsic_xyz(self):
        # Rx(pi/2) Ry(pi/2) maps z to x. Extrinsic xyz maps it to -y.
        elem = __import__("xml.etree.ElementTree", fromlist=["Element"]).Element("body", euler="1.5707963267948966 1.5707963267948966 0")
        rotation = self.module.mjcf_rotation(elem, angle="radian", eulerseq="xyz")
        np.testing.assert_allclose(rotation @ [0, 0, 1], [1, 0, 0], atol=1e-12)

    def test_rnea_aba_with_rotor_armature_and_damping(self):
        model = self.build(True)
        velocity = np.array([0.1, -0.2, 0.3, 0.04, -0.05, 0.06, 0.7])
        accel = np.array([-0.1, 0.5, 0.2, -0.2, 0.3, 0.1, -0.4])
        effort = model.inverse_dynamics([0.3], velocity, accel)
        np.testing.assert_allclose(model.forward_dynamics([0.3], velocity, effort), accel, atol=1e-10)


class DiscreteAlgebra(unittest.TestCase):
    def test_zoh_double_integrator_and_nonzero_initial_conditions(self):
        module = load_module("derive_v16_discrete")
        A = np.array([[0., 1.], [0., 0.]])
        B = np.array([[0.], [2.]])
        Ad, Bd, dd = module.exact_zoh(A, B, 0.2, np.array([0., 3.]))
        np.testing.assert_allclose(Ad, [[1, 0.2], [0, 1]], atol=1e-14)
        np.testing.assert_allclose(Bd, [[0.04], [0.4]], atol=1e-14)
        np.testing.assert_allclose(dd, [0.06, 0.6], atol=1e-14)
        response = module.sampled_response(Ad, Bd, np.array([[1.], [1.]]), np.array([4., -2.]), dd)
        np.testing.assert_allclose(response, [[4, -2], [3.7, -1], [3.6, 0]], atol=1e-14)

    def test_siso_channels_keep_cross_coupling_and_match_state_space(self):
        module = load_module("derive_v16_discrete")
        Ad = np.array([[0.7, 0.2], [0.1, 0.6]])
        Bd = np.array([[0.3, 0.4], [0.5, 0.6]])
        C = np.eye(2)
        channels = module.siso_channels(Ad, Bd, C)
        # Hand elimination: (1-1.3B+.4B^2)y1=(.3B-.08B^2)u1 +(.4B-.12B^2)u2.
        np.testing.assert_allclose(channels[0][0]["denominator_B"], [1, -1.3, 0.4], atol=1e-14)
        np.testing.assert_allclose(channels[0][0]["numerator_B"], [0, 0.3, -0.08], atol=1e-14)
        np.testing.assert_allclose(channels[0][1]["numerator_B"], [0, 0.4, -0.12], atol=1e-14)
        for omega in (0.1, 1., 2.):
            delay = np.exp(-1j * omega)
            target = C @ np.linalg.solve(np.eye(2) - Ad * delay, Bd * delay)
            np.testing.assert_allclose(module.evaluate_channels(channels, delay), target, atol=1e-13)

    def test_difference_equations_reproduce_mimo_input_response(self):
        # Catches missing one-sample input delay and missing other-input coupling.
        module = load_module("derive_v16_discrete")
        self.assertTrue(callable(getattr(module, "difference_response", None)), "Difference-equation implementation is missing")
        Ad = np.array([[0.7, 0.2], [0.1, 0.6]])
        Bd = np.array([[0.3, 0.4], [0.5, 0.6]])
        channels = module.siso_channels(Ad, Bd, np.eye(2))
        inputs = np.array([[1., 2.], [-1., 0.], [0., 1.], [2., -3.], [0., 0.]])
        target = module.sampled_response(Ad, Bd, inputs, [0., 0.])[:-1]
        response = np.column_stack([sum(module.difference_response(channels[i][j], inputs[:, j]) for j in range(2)) for i in range(2)])
        np.testing.assert_allclose(response, target, atol=1e-13)

    def test_near_unit_poles_use_stable_modal_frequency_evaluation(self):
        # Catches cancellation from evaluating nearly repeated B polynomials.
        module = load_module("derive_v16_discrete")
        Ad = np.diag([.999999, .999998])
        Bd = np.ones((2, 1))
        C = np.array([[1., -.5]])
        delay = np.exp(-1j*1e-5)
        target = delay/(1-.999999*delay)-.5*delay/(1-.999998*delay)
        channels = module.siso_channels(Ad, Bd, C)
        np.testing.assert_allclose(module.evaluate_channels(channels, delay), [[target]], rtol=1e-12, atol=1e-9)


@unittest.skipUnless(os.environ.get("V16_MJCF"), "real source requires HPC V16_MJCF")
class RobotModel(unittest.TestCase):
    def setUp(self):
        self.module = load_module("v16_physical_model")
        self.fixed = self.module.build_model(os.environ["V16_MJCF"])
        self.free = self.module.build_model(os.environ["V16_MJCF"], floating_base=True)
        self.q = np.tile([0., -.55, 1.10], 4)

    def test_joint_order_dimensions_total_mass_and_keyframe(self):
        self.assertEqual((self.fixed.model.nq, self.fixed.model.nv), (12, 12))
        self.assertEqual((self.free.model.nq, self.free.model.nv), (19, 18))
        self.assertEqual(self.fixed.body_count, 54)
        self.assertEqual(self.fixed.joint_names, ("FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE", "RL_HAA", "RL_HFE", "RL_KFE", "RR_HAA", "RR_HFE", "RR_KFE"))
        self.assertEqual(len(set(self.fixed.body_names)), 54)
        np.testing.assert_allclose(self.fixed.keyframes["stand"]["joint_position"], self.q)
        np.testing.assert_allclose(self.fixed.keyframes["stand"]["base_position"], [0, 0, .464])
        self.assertAlmostEqual(self.fixed.total_mass, self.fixed.source_total_mass, places=10)
        np.testing.assert_allclose(self.fixed.effort_limits, np.tile([24.8, 48, 48], 4))
        np.testing.assert_array_equal(self.free.actuation_matrix[:, :6], np.zeros((12, 6)))

    def test_mass_spd_and_inverse_forward_roundtrip(self):
        rng = np.random.default_rng(16)
        for model in (self.fixed, self.free):
            q = self.q + rng.normal(0, .1, 12)
            M = model.mass_matrix(q)
            np.testing.assert_allclose(M, M.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(M).min(), 0)
            velocity = rng.normal(0, .1, model.model.nv)
            accel = rng.normal(0, .2, model.model.nv)
            torque = model.inverse_dynamics(q, velocity, accel)
            np.testing.assert_allclose(model.forward_dynamics(q, velocity, torque), accel, atol=1e-9)
            np.testing.assert_allclose(M @ accel + model.bias(q, velocity), torque, atol=1e-9)

    def test_velocity_constructor_composes_with_canonical_dynamics(self):
        # Catches returning Pin traversal velocities to an API accepting canonical order.
        joints = np.arange(12, dtype=float) * .01
        constructed = self.free.velocity(joints, base_twist=[.1,.2,.3,.4,.5,.6])
        target = np.r_[[.1,.2,.3,.4,.5,.6],joints]
        np.testing.assert_array_equal(constructed,target)
        np.testing.assert_allclose(self.free.bias(self.q,constructed),self.free.bias(self.q,target),atol=1e-12)

    def test_foot_point_jacobians_include_base_and_match_finite_difference(self):
        for model in (self.fixed, self.free):
            for prefix in ("FL", "FR", "RL", "RR"):
                J = model.point_jacobian(self.q, prefix + "_foot", [.01, -.02, .03])
                fd = np.empty_like(J)
                for col in range(model.model.nv):
                    delta = np.zeros(model.model.nv)
                    delta[col] = 1e-6
                    fd[:, col] = (model.point_after_integration(self.q, delta, prefix + "_foot", [.01, -.02, .03]) - model.point_after_integration(self.q, -delta, prefix + "_foot", [.01, -.02, .03])) / 2e-6
                np.testing.assert_allclose(J, fd, atol=2e-8, rtol=2e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
