"""Pin the training stairs to the physical acceptance geometry."""
import numpy as np
import pytest

from v15_cad_training_terrain import CAD_STEP_HEIGHTS, cad_terrain_layout, surface_height, promotion_flags


@pytest.mark.parametrize('kind', ['up', 'down'])
def test_training_stairs_require_four_risers_on_the_forward_y_lane(kind):
    for row, height in enumerate(CAD_STEP_HEIGHTS):
        layout = cad_terrain_layout(kind, (row + .25) / len(CAD_STEP_HEIGHTS))
        points = np.asarray([[0., y] for y in [0., 1.999, 2.001, 2.301, 2.601, 2.901, 4., 10.]])
        levels = surface_height(points, np.full(len(points), height), np.full(len(points), kind == 'down'))
        expected = np.asarray([0, 0, 1, 2, 3, 4, 4, 4]) * height
        if kind == 'down': expected = 4 * height - expected
        np.testing.assert_allclose(levels, expected)
        assert layout['origin'][2] == pytest.approx(4 * height if kind == 'down' else 0.)
        assert layout['terrain'].step_width_m == .3
        assert layout['terrain'].riser_count == 4
        assert layout['terrain'].width_m == 2.


def test_side_bypass_and_fall_never_promote_stair_curriculum():
    position = np.array([[0., 4.1], [1.2, 5.], [0., 4.1], [0., .8]])
    up, down = promotion_flags(position, np.array([False, False, True, False]), np.ones(4, bool))
    np.testing.assert_array_equal(up, [True, False, False, False])
    np.testing.assert_array_equal(down, [False, True, True, True])


def test_flat_terrain_keeps_omnidirectional_origins_and_levels():
    layout = cad_terrain_layout('flat', .8)
    np.testing.assert_allclose(layout['origin'], [3., 9., 0.])
    assert not layout['blocks']
    up, down = promotion_flags(np.array([[0., 4.], [3., 8.]]), np.array([False, True]), np.zeros(2, bool))
    assert not up.any() and not down.any()
