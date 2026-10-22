"""Contact/triangle algebra only: these tests do not construct a physics model."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


def crawl_module():
    path = Path(__file__).resolve().parents[1] / "v16_crawl_math.py"
    if not path.exists():
        raise AssertionError("Required crawl mathematics implementation is missing")
    spec = importlib.util.spec_from_file_location("v16_crawl_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRIANGLE = np.array([[0., 0.], [4., 0.], [0., 3.]])


class TriangleGeometry(unittest.TestCase):
    def test_signed_margin_is_distance_and_independent_of_vertex_winding(self):
        module = crawl_module()
        points = np.array([[1., 1.], [4., 1.], [0., 0.]])
        for vertices in (TRIANGLE, TRIANGLE[::-1]):
            margin, valid = module.triangle_signed_margin(vertices, points)
            np.testing.assert_allclose(margin, [1., -.8, 0.], atol=1e-12)
            np.testing.assert_array_equal(valid, [True, True, True])

    def test_inset_centroid_moves_continuously_toward_incenter(self):
        module = crawl_module()
        # The 3-4-5 triangle has centroid (4/3,1), incenter (1,1), radius 1.
        for inset, expected in ((0., [4/3, 1.]), (.5, [7/6, 1.]), (1., [1., 1.])):
            for vertices in (TRIANGLE, TRIANGLE[::-1]):
                target, valid = module.triangle_com_target(vertices, inset=inset)
                self.assertTrue(valid)
                np.testing.assert_allclose(target, expected, atol=1e-12)

    def test_degenerate_and_nonfinite_triangles_fail_closed_without_poisoning_peer(self):
        module = crawl_module()
        vertices = np.array([TRIANGLE, [[0,0],[1,0],[2,0]],
                             [[0,0],[1,0],[1,1e-12]], [[0,0],[np.nan,1],[1,0]]])
        margin, valid = module.triangle_signed_margin(vertices, [1., 1.])
        np.testing.assert_array_equal(valid, [True, False, False, False])
        self.assertAlmostEqual(margin[0], 1., places=12)
        self.assertTrue(np.isneginf(margin[1:]).all())
        targets, target_valid = module.triangle_com_target(vertices, inset=.1)
        np.testing.assert_array_equal(target_valid, valid)
        self.assertTrue(np.isfinite(targets).all())
        np.testing.assert_array_equal(targets[1:], np.zeros((3, 2)))

    def test_impossible_inset_and_nonfinite_com_never_pass_liftoff_gate(self):
        module = crawl_module()
        target, valid = module.triangle_com_target(TRIANGLE, inset=1.01)
        self.assertFalse(valid)
        self.assertTrue(np.isfinite(target).all())
        margin, valid = module.triangle_signed_margin(TRIANGLE, [np.nan, 1.])
        self.assertFalse(valid)
        self.assertTrue(np.isneginf(margin))
        with self.assertRaises(ValueError):
            module.triangle_com_target(TRIANGLE, inset=-.1)

    def test_world_rotation_translation_and_float32_preserve_geometry(self):
        module = crawl_module()
        rotation = np.array([[0., -1.], [1., 0.]])
        vertices = (TRIANGLE @ rotation.T + [100., -200.]).astype(np.float32)
        point = np.array([99., -199.], dtype=np.float32)
        margin, valid = module.triangle_signed_margin(vertices, point)
        self.assertTrue(valid)
        self.assertEqual(margin.dtype, np.dtype("float32"))
        np.testing.assert_allclose(margin, 1., atol=1e-6)
        target, valid = module.triangle_com_target(vertices, inset=.5)
        self.assertTrue(valid)
        np.testing.assert_allclose(target, [99., -200+7/6], atol=2e-5)

    def test_safe_requested_com_is_preserved_instead_of_recentering(self):
        module = crawl_module()
        points = np.array([[1., 1.], [.7, .7]])
        target, valid = module.triangle_com_target(TRIANGLE, inset=.5, point_xy=points)
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_allclose(target, points, atol=1e-12)

    def test_requested_com_uses_nearest_inset_edge_or_corner(self):
        module = crawl_module()
        # At inset .5 the vertices are (.5,.5),(2.5,.5),(.5,2).
        points = np.array([[4., 1.], [2., 2.], [-1., 1.]])
        target, valid = module.triangle_com_target(TRIANGLE, inset=.5, point_xy=points)
        np.testing.assert_array_equal(valid, [True, True, True])
        np.testing.assert_allclose(target, [[2.5, .5], [1.46, 1.28], [.5, 1.]], atol=1e-12)
        target, valid = module.triangle_com_target(TRIANGLE, inset=1., point_xy=[100., 100.])
        self.assertTrue(valid)
        np.testing.assert_allclose(target, [1., 1.], atol=1e-12)
        target, valid = module.triangle_com_target(TRIANGLE, inset=.5, point_xy=[np.nan, 1.])
        self.assertFalse(valid)
        np.testing.assert_array_equal(target, [0., 0.])


class ContactProjection(unittest.TestCase):
    def test_weighted_projection_minimizes_acceleration_change(self):
        module = crawl_module()
        jacobian = np.array([[[1., 1.], [0., 0.], [0., 0.]]])
        target = np.array([[1., 0., 0.]])
        projected, residual = module.contact_acceleration_projection(
            [0., 0.], jacobian, [True], contact_target=target,
            weight_matrix=np.diag([1., 4.]), regularization=1e-10)
        # Minimize a1^2+4*a2^2 subject to a1+a2=1.
        np.testing.assert_allclose(projected, [.8, .2], atol=1e-9)
        np.testing.assert_allclose(residual, np.zeros((1, 3)), atol=1e-9)

    def test_full_spd_weight_retains_off_diagonal_coupling(self):
        module = crawl_module()
        jacobian = np.array([[[1., 0.], [0., 0.], [0., 0.]]])
        projected, _ = module.contact_acceleration_projection(
            [0., 0.], jacobian, [True], contact_target=[[1., 0., 0.]],
            weight_matrix=[[2., 1.], [1., 2.]], regularization=1e-10)
        np.testing.assert_allclose(projected, [1., -.5], atol=1e-9)

    def test_disabled_nan_rows_are_zero_and_do_not_change_unconstrained_acceleration(self):
        module = crawl_module()
        jacobian = np.array([[[1., 0.], [0., 0.], [0., 0.]],
                             [[np.nan, np.nan]] * 3])
        target = np.array([[1., 0., 0.], [np.nan] * 3])
        original = jacobian.copy()
        projected, residual = module.contact_acceleration_projection(
            [3., 4.], jacobian, [True, False], contact_target=target, regularization=1e-10)
        np.testing.assert_allclose(projected, [1., 4.], atol=1e-9)
        np.testing.assert_array_equal(residual[1], [0., 0., 0.])
        np.testing.assert_array_equal(jacobian, original)
        projected, residual = module.contact_acceleration_projection(
            [3., 4.], jacobian, [False, False], contact_target=target)
        np.testing.assert_array_equal(projected, [3., 4.])
        np.testing.assert_array_equal(residual, np.zeros((2, 3)))

    def test_regularization_is_invariant_to_contact_row_units(self):
        module = crawl_module()
        jacobian = np.array([[[1., 1., 0.], [0., 1., 1.], [0., 0., 0.]]])
        target = np.array([[1., 2., 0.]])
        scales = np.array([1000., .001, 1.])
        for J, b in ((jacobian, target), (jacobian * scales[None, :, None], target * scales)):
            projected, _ = module.contact_acceleration_projection(
                np.zeros(3), J, [True], contact_target=b, regularization=1e-10)
            np.testing.assert_allclose(projected, [0., 1., 1.], atol=1e-8)

    def test_redundant_constraints_and_batched_masks_remain_finite(self):
        module = crawl_module()
        jacobian = np.array([[[1., 0.], [1., 0.], [0., 0.]]])
        projected, residual = module.contact_acceleration_projection(
            np.zeros((2, 2)), jacobian, [[True], [False]],
            contact_target=[[1., 3., 0.]], regularization=1e-8)
        np.testing.assert_allclose(projected, [[2., 0.], [0., 0.]], atol=2e-8)
        np.testing.assert_allclose(residual, [[[1., -1., 0.]], [[0., 0., 0.]]], atol=2e-8)

    def test_invalid_weight_and_zero_regularization_fail_fast(self):
        module = crawl_module()
        jacobian = np.array([np.eye(3)])
        for weight in (np.diag([1., -1., 1.]), [[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]):
            with self.assertRaises(ValueError):
                module.contact_acceleration_projection(np.zeros(3), jacobian, [True], weight_matrix=weight)
        with self.assertRaises(ValueError):
            module.contact_acceleration_projection(np.zeros(3), jacobian, [True], regularization=0.)

    def test_float32_redundant_rows_keep_a_representable_regularizer(self):
        module = crawl_module()
        jacobian = np.array([[[1., 0.], [1., 0.], [0., 0.]]], dtype=np.float32)
        projected, residual = module.contact_acceleration_projection(
            np.zeros(2, dtype=np.float32), jacobian, [True],
            contact_target=[[1., 1., 0.]], regularization=1e-12)
        self.assertEqual(projected.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(projected).all() and np.isfinite(residual).all())
        np.testing.assert_allclose(projected, [1., 0.], atol=1e-4)


class CrawlSchedule(unittest.TestCase):
    def test_quarter_crawl_never_schedules_two_swing_legs(self):
        module = crawl_module()
        stance = module.quarter_crawl_stance(np.linspace(0., 2., 801), duty_factor=.8)
        self.assertEqual(stance.shape, (801, 4))
        self.assertTrue((stance.sum(axis=-1) >= 3).all())
        self.assertTrue((stance.sum(axis=-1) == 4).any())
        np.testing.assert_array_equal(module.quarter_crawl_stance([.875, 1.125], duty_factor=.8),
                                      [[False, True, True, True], [True, True, True, False]])
        for duty in (.55, .70):
            with self.assertRaises(ValueError):
                module.quarter_crawl_stance([0., .875], duty_factor=duty)

    def test_per_environment_duty_and_minimum_boundary(self):
        module = crawl_module()
        np.testing.assert_array_equal(module.quarter_crawl_stance([.875, .875], duty_factor=[.8, .9]),
                                      [[False, True, True, True], [True, True, True, True]])
        stance = module.quarter_crawl_stance(np.linspace(0., 1., 1001), duty_factor=.75)
        self.assertTrue((stance.sum(axis=-1) >= 3).all())


class PreparedCrawlTransitions(unittest.TestCase):
    def transition(self, module, stage, elapsed, ready_elapsed, **updates):
        options = dict(dt=.02, ready=True, all_contacts=True,
                       stance_contacts_ok=True, swing_seconds=.1)
        options.update(updates)
        return module.prepared_crawl_transition(stage, elapsed, ready_elapsed, **options)

    def test_batch_transitions_have_separate_clocks_and_one_tick_event_pulses(self):
        module = crawl_module()
        stage = np.array([0, 0, 1, 1, 2, 2])
        elapsed = np.array([.08, .08, .06, .06, .08, .08])
        ready_elapsed = np.array([.08, .08, 0., 0., .08, .08])
        result = self.transition(module, stage, elapsed, ready_elapsed,
                     ready=[True, False, False, False, False, False],
                     all_contacts=[True, True, False, False, True, False],
                     stance_contacts_ok=[True, True, True, False, True, True], swing_seconds=.08)
        np.testing.assert_array_equal(result[0], [1, 0, 2, 1, 0, 2])
        np.testing.assert_allclose(result[1], [0., .1, 0., .06, 0., .1], atol=1e-12)
        np.testing.assert_array_equal(result[2], np.zeros(6))
        np.testing.assert_array_equal(result[3], [True, False, False, False, False, False])
        np.testing.assert_array_equal(result[4], [False, False, True, False, False, False])
        np.testing.assert_array_equal(result[5], [False, False, False, False, True, False])
        np.testing.assert_array_equal(stage, [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(elapsed, [.08, .08, .06, .06, .08, .08])
        np.testing.assert_array_equal(ready_elapsed, [.08, .08, 0., 0., .08, .08])

    def test_prepare_requires_continuous_ready_and_all_contacts(self):
        module = crawl_module()
        state = (np.array(0), np.array(0., dtype=np.float32), np.array(0., dtype=np.float32))
        for index, ready in enumerate([True, True, False, True, True, True, True, True]):
            result = self.transition(module, *state, ready=ready)
            self.assertEqual(bool(result[3]), index == 7)
            state = result[:3]
        self.assertEqual(int(state[0]), 1)
        result = self.transition(module, 0, .2, .08, all_contacts=False)
        self.assertEqual(int(result[0]), 0)
        self.assertEqual(float(result[2]), 0.)
        self.assertFalse(result[3])

    def test_swing_clock_pauses_when_three_supports_are_not_measured(self):
        module = crawl_module()
        result = self.transition(module, 1, .05, 0., stance_contacts_ok=False)
        self.assertEqual(int(result[0]), 1)
        self.assertEqual(float(result[1]), .05)
        self.assertFalse(result[4])
        for _ in range(3):
            result = self.transition(module, *result[:3])
        self.assertEqual(int(result[0]), 2)
        self.assertTrue(result[4])
        next_result = self.transition(module, *result[:3], all_contacts=False)
        self.assertFalse(next_result[4])

    def test_settle_contact_interruption_resets_continuous_dwell(self):
        module = crawl_module()
        state = (2, 0., 0.)
        for index, contacts in enumerate([True, False, True, True]):
            result = self.transition(module, *state, dt=.05, all_contacts=contacts)
            self.assertEqual(bool(result[5]), index == 3)
            state = result[:3]
        self.assertEqual(int(state[0]), 0)
        self.assertEqual(float(state[1]), 0.)
        self.assertEqual(float(state[2]), 0.)

    def test_invalid_stage_clock_or_duration_fails_fast(self):
        module = crawl_module()
        for stage, elapsed, dwell, update in ((3, 0., 0., {}), (.5, 0., 0., {}),
                 (0, -.1, 0., {}), (0, 0., np.nan, {}), (0, 0., 0., {"dt": 0.}),
                 (0, 0., 0., {"swing_seconds": 0.}),
                 (0, 0., 0., {"prepare_min_seconds": 0.}),
                 (0, 0., 0., {"settle_seconds": -.1})):
            with self.assertRaises(ValueError):
                self.transition(module, stage, elapsed, dwell, **update)


class ContactSupportMasks(unittest.TestCase):
    def test_touchdown_presence_allows_loading_before_stable_load_gate(self):
        force = np.array([[3., 30., 0., np.nan], [1.,20.,21.,-1.],
                          [np.inf,-np.inf,1.1,20.1]], dtype=np.float32)
        original = force.copy()
        presence, loaded = crawl_module().contact_support_masks(force)
        # A landed 3 N foot enters force allocation without claiming stable
        # 20 N support. Strict boundaries preserve the existing Fz>20 gate.
        np.testing.assert_array_equal(presence, [[1,1,0,0],[0,1,1,0],[0,0,1,1]])
        np.testing.assert_array_equal(loaded, [[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.assertEqual(presence.dtype, np.dtype(bool))
        self.assertEqual(loaded.dtype, np.dtype(bool))
        np.testing.assert_array_equal(force, original)

    def test_invalid_or_unordered_thresholds_fail_before_contact_classification(self):
        for presence_min, load_min in ((0.,20.),(-1.,20.),(1.,0.),(20.,20.),
                                       (21.,20.),(np.nan,20.),(1.,np.inf)):
            with self.subTest(thresholds=(presence_min,load_min)), self.assertRaises(ValueError):
                crawl_module().contact_support_masks([3.,30.], presence_min_n=presence_min,
                                                     load_min_n=load_min)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch parity requires a Torch installation")
class TorchParity(unittest.TestCase):
    def test_contact_presence_and_loaded_masks_match_numpy_without_mutation(self):
        import torch
        force = torch.tensor([[3.,30.,0.,float("nan")],
                              [float("inf"),-1.,1.,20.]], dtype=torch.float32)
        original = force.clone()
        presence, loaded = crawl_module().contact_support_masks(force)
        torch.testing.assert_close(presence, torch.tensor([[True,True,False,False],
                                                         [False,False,False,True]]))
        torch.testing.assert_close(loaded, torch.tensor([[False,True,False,False],
                                                       [False,False,False,False]]))
        self.assertEqual(presence.device, force.device)
        self.assertEqual(loaded.device, force.device)
        torch.testing.assert_close(force, original, equal_nan=True)

    def test_tensor_backend_matches_numpy_and_preserves_dtype_device(self):
        import torch
        module = crawl_module()
        triangle = torch.tensor(TRIANGLE, dtype=torch.float64)
        margin, valid = module.triangle_signed_margin(triangle, torch.tensor([1., 1.], dtype=torch.float64))
        target, target_valid = module.triangle_com_target(triangle, inset=.5)
        torch.testing.assert_close(margin, torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close(target, torch.tensor([7/6, 1.], dtype=torch.float64))
        closest, closest_valid = module.triangle_com_target(triangle, inset=.5,
                    point_xy=torch.tensor([[1., 1.], [2., 2.]], dtype=torch.float64))
        torch.testing.assert_close(closest, torch.tensor([[1., 1.], [1.46, 1.28]], dtype=torch.float64))
        self.assertTrue(bool(closest_valid.all()))
        self.assertTrue(bool(valid & target_valid))
        jacobian = torch.tensor([[[1., 1.], [0., 0.], [0., 0.]], [[float("nan")]*2]*3], dtype=torch.float64)
        acceleration = torch.zeros(2, dtype=torch.float64)
        projected, residual = module.contact_acceleration_projection(
            acceleration, jacobian, torch.tensor([True, False]),
            contact_target=[[1., 0., 0.], [float("nan")]*3],
            weight_matrix=[[1., 0.], [0., 4.]], regularization=1e-10)
        torch.testing.assert_close(projected, torch.tensor([.8, .2], dtype=torch.float64), rtol=0., atol=1e-9)
        torch.testing.assert_close(residual[1], torch.zeros(3, dtype=torch.float64))
        self.assertEqual(projected.dtype, acceleration.dtype)
        self.assertEqual(projected.device, acceleration.device)
        stance = module.quarter_crawl_stance(torch.tensor([.875, 1.125], dtype=torch.float32), duty_factor=.8)
        torch.testing.assert_close(stance, torch.tensor([[False, True, True, True], [True, True, True, False]]))

    def test_prepared_transitions_keep_tensor_device_and_float32_clocks(self):
        import torch
        module = crawl_module()
        result = module.prepared_crawl_transition(torch.tensor([0, 1, 2]),
                    torch.tensor([.08, .08, .08], dtype=torch.float32),
                    torch.tensor([.08, 0., .08], dtype=torch.float32),
                    dt=.02, ready=torch.tensor([True, False, False]),
                    all_contacts=torch.tensor([True, False, True]),
                    stance_contacts_ok=torch.tensor([True, True, True]), swing_seconds=.1)
        torch.testing.assert_close(result[0], torch.tensor([1, 2, 0]))
        torch.testing.assert_close(result[1], torch.zeros(3, dtype=torch.float32))
        for item in result:
            self.assertEqual(item.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
