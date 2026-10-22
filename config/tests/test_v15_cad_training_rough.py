"""Known training rough surfaces; seed73 differs from the fixed seed42 test."""
import numpy as np
from v15_cad_training_terrain import cad_terrain_layout, rough_tile_specs


def test_rough_layout_contains_forty_tiles_and_flat_spawn_support():
    layout = cad_terrain_layout('rough', .5)
    blocks = layout['blocks']
    assert len(blocks) == 40
    assert layout['terrain'].rough_seed == 73
    np.testing.assert_allclose(layout['origin'], [3., 2., 0.])
    assert all(.01 <= block.size[2] <= .035 for block in blocks)
    assert all(block.center[1] - block.size[1]/2. >= 2. for block in blocks)


def test_training_rough_tiles_keep_dimensions_and_use_different_heights_from_holdout():
    train = rough_tile_specs(73)
    holdout = rough_tile_specs(42)
    assert train.shape == (40, 5)
    np.testing.assert_allclose(train[:, :4], holdout[:, :4])
    assert np.max(abs(train[:, 4] - holdout[:, 4])) > .01
