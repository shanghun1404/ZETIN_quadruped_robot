"""Analytic and sampled contracts for simulator-free COM waypoint references."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "v16_com_reference_math.py"


def helper():
    assert MODULE.is_file(), "Minimum-jerk COM helper is not implemented"
    spec = importlib.util.spec_from_file_location("v16_com_reference_math", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_midpoint_analytic_position_peak_velocity_and_zero_acceleration(dtype):
    # A wrong polynomial or time scaling breaks these hand-derived values.
    m = helper()
    start, end = np.array([-.2, .3], dtype=dtype), np.array([.1, -.1], dtype=dtype)
    p, v, a = m.minimum_jerk_reference(start, end, 1., 2.)
    np.testing.assert_allclose(p, [-.05, .1], atol=4e-8)
    np.testing.assert_allclose(v, [.28125, -.375], atol=4e-8)
    np.testing.assert_array_equal(a, [0., 0.])
    assert all(value.shape == (2,) and value.dtype == dtype for value in (p, v, a))


def test_endpoints_and_after_end_hold_have_zero_velocity_and_acceleration():
    m = helper()
    start, end = np.array([1., -2.]), np.array([2., 3.])
    p, v, a = m.minimum_jerk_reference(start, end, np.array([0., 2., 3.]), 2.)
    np.testing.assert_array_equal(p, [start, end, end])
    np.testing.assert_array_equal(v, np.zeros((3, 2)))
    np.testing.assert_array_equal(a, np.zeros((3, 2)))


def test_duration_uses_euclidean_distance_and_each_active_limit():
    m = helper()
    start = np.zeros(2)
    # Distance .4; the independent peak speed formula gives T=7.5 s.
    assert m.minimum_jerk_duration(start, [.24, .32], vmax=.1, amax=100.) == pytest.approx(7.5)
    # Distance .03; the independent peak acceleration formula gives this T.
    assert m.minimum_jerk_duration(start, [0., .03], vmax=100., amax=.6) == pytest.approx(.537284965911771)
    assert m.minimum_jerk_duration(start, start) == pytest.approx(.2)


@pytest.mark.parametrize("displacement", [[.003, .004], [.072, .096], [.6, -.8]])
def test_full_sampled_trace_respects_euclidean_speed_and_acceleration_bounds(displacement):
    m = helper()
    start, end = np.array([.4, -.2]), np.array([.4, -.2]) + displacement
    duration = m.minimum_jerk_duration(start, end)
    times = np.linspace(0., float(duration), 10001)
    p, v, a = m.minimum_jerk_reference(start, end, times, duration)
    assert np.linalg.norm(v, axis=-1).max() <= .20 + 1e-12
    assert np.linalg.norm(a, axis=-1).max() <= .6 + 1e-12
    # Derivatives of sampled positions/velocities catch inconsistent outputs.
    dt = times[1] - times[0]
    np.testing.assert_allclose((p[2:] - p[:-2]) / (2 * dt), v[1:-1], atol=1e-7)
    np.testing.assert_allclose((v[2:] - v[:-2]) / (2 * dt), a[1:-1], atol=1e-6)


def test_zero_distance_is_constant_and_batch_inputs_are_not_mutated():
    m = helper()
    start = np.array([[.2, -.4], [1., 2.]], dtype=np.float32)
    end, elapsed, duration = start.copy(), np.array([.1, 5.], dtype=np.float32), np.array([.2, .3], dtype=np.float32)
    inputs = [start, end, elapsed, duration]
    originals = [x.copy() for x in inputs]
    p, v, a = m.minimum_jerk_reference(*inputs)
    np.testing.assert_array_equal(p, start)
    np.testing.assert_array_equal(v, np.zeros_like(start))
    np.testing.assert_array_equal(a, np.zeros_like(start))
    for value, original in zip(inputs, originals):
        np.testing.assert_array_equal(value, original)
    times = m.minimum_jerk_duration(start, end)
    assert times.shape == (2,) and times.dtype == np.float32


@pytest.mark.parametrize("elapsed,duration", [(-.1, 1.), (np.nan, 1.), (np.inf, 1.), (0., 0.), (0., -.1), (0., np.inf), (0., np.nan)])
def test_invalid_time_values_are_rejected(elapsed, duration):
    with pytest.raises(ValueError):
        helper().minimum_jerk_reference(np.zeros(2), np.ones(2), elapsed, duration)


@pytest.mark.parametrize("name,value", [("vmax", 0.), ("vmax", np.nan), ("amax", -.1), ("amax", np.inf), ("min_duration", 0.)])
def test_invalid_duration_bounds_are_rejected(name, value):
    with pytest.raises(ValueError):
        helper().minimum_jerk_duration(np.zeros(2), np.ones(2), **{name: value})


@pytest.mark.parametrize("start,end", [([np.nan, 0.], [0., 0.]), ([0., 0.], [np.inf, 0.]), ([0., 0., 0.], [1., 1., 1.])])
def test_invalid_waypoints_are_rejected(start, end):
    m = helper()
    with pytest.raises(ValueError):
        m.minimum_jerk_duration(start, end)
    with pytest.raises(ValueError):
        m.minimum_jerk_reference(start, end, .1, 1.)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_batch_parity_dtype_device_and_no_input_mutation(dtype_name):
    torch = pytest.importorskip("torch")
    m = helper()
    dtype = getattr(torch, dtype_name)
    start = torch.tensor([[.1, -.2], [.5, .3]], dtype=dtype)
    end = torch.tensor([[.2, .05], [.4, -.1]], dtype=dtype)
    elapsed = torch.tensor([.25, 1.], dtype=dtype)
    duration = m.minimum_jerk_duration(start, end)
    inputs = [start, end, elapsed, duration]
    originals = [x.clone() for x in inputs]
    result = m.minimum_jerk_reference(*inputs)
    expected = m.minimum_jerk_reference(*(x.numpy() for x in inputs))
    np.testing.assert_allclose(duration.numpy(), m.minimum_jerk_duration(start.numpy(), end.numpy()), rtol=1e-6)
    assert duration.shape == (2,) and duration.dtype == dtype and duration.device == start.device
    for actual, want in zip(result, expected):
        assert actual.shape == (2, 2) and actual.dtype == dtype and actual.device == start.device
        np.testing.assert_allclose(actual.numpy(), want, rtol=1e-6, atol=1e-7)
    for value, original in zip(inputs, originals):
        assert torch.equal(value, original)
