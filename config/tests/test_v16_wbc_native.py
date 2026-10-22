"""The native deployment path must retain the reference QP and warm state."""
import importlib.util
import numpy as np
import pytest

from test_v16_wbc_qp import fixture, wbc_module


def test_native_backend_is_built_for_this_python():
    assert importlib.util.find_spec('_v16_wbc_native') is not None, 'Build the C++ WBC extension for the deployment interpreter'


@pytest.mark.parametrize('contacts', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('asymmetric', [False, True])
def test_cpp_preserves_reference_outputs_and_warm_state(contacts, asymmetric):
    data = fixture(lever=.25)
    data['stance'][:] = False
    data['stance'][:contacts] = True
    data['desired_acceleration'] = np.linspace(-.1, .2, 18)
    data['h'][6:] = np.linspace(-2., 3., 12)
    if asymmetric:
        data['tau_lower'][1::3], data['tau_upper'][1::3] = -6., -4.
    solve = wbc_module().solve_wbc_qp
    reference = solve(**data, backend='numpy')
    native = solve(**data, backend='cpp')
    for actual, expected in zip(native[:3], reference[:3]):
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    for key in reference[3]:
        np.testing.assert_allclose(native[3][key], reference[3][key], atol=2e-8, rtol=2e-8)
    for key in ('x', 'z', 'y'):
        np.testing.assert_allclose(native[4][key], reference[4][key], atol=2e-8, rtol=2e-8)
    saved = {key: value.copy() for key, value in native[4].items()}
    again = solve(**data, backend='cpp', warm_start=native[4])
    expected = solve(**data, backend='numpy', warm_start=reference[4])
    for actual, wanted in zip(again[:3], expected[:3]):
        np.testing.assert_allclose(actual, wanted, atol=2e-8, rtol=2e-8)
    for key in saved:
        np.testing.assert_array_equal(native[4][key], saved[key])


def test_explicit_cpp_rejects_unsupported_batch_and_float32():
    solve = wbc_module().solve_wbc_qp
    data = fixture(dtype=np.float32)
    with pytest.raises(ValueError, match='unbatched float64'):
        solve(**data, backend='cpp')
    data = fixture()
    data['M'] = np.stack([data['M'], data['M']])
    with pytest.raises(ValueError, match='unbatched float64'):
        solve(**data, backend='cpp')


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match='backend'):
        wbc_module().solve_wbc_qp(**fixture(), backend='misspelled')
