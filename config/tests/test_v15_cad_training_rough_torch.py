"""Training height observations must match the actual authored rough boxes."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from v15_cad_training_terrain import rough_tile_specs, rough_surface_height_torch


def test_each_tile_center_has_its_authored_height_and_outside_is_zero():
    boxes = torch.tensor(rough_tile_specs(), dtype=torch.float64)
    heights = rough_surface_height_torch(boxes[:, :2], boxes)
    torch.testing.assert_close(heights, boxes[:, 4])
    xy = torch.tensor([[0., 0.], [0., 1.99], [1.1, 3.], [0., 6.01]], dtype=torch.float64)
    torch.testing.assert_close(rough_surface_height_torch(xy, boxes), torch.zeros(4, dtype=torch.float64))


def test_broadcast_heights_match_independent_box_geometry_with_shared_edges():
    boxes_np = rough_tile_specs()
    rng = np.random.default_rng(79)
    xy_np = rng.uniform([-1.1, 1.9], [1.1, 6.1], (3, 4, 2))
    # Shared edge: both touching surfaces exist, so choose the highest top.
    xy_np[0, 0] = [boxes_np[0, 0] + boxes_np[0, 2]/2, boxes_np[0, 1]]
    expected = np.zeros((3, 4))
    for i in range(3):
        for j in range(4):
            for cx, cy, sx, sy, height in boxes_np:
                if abs(xy_np[i,j,0]-cx) <= sx/2+1e-7 and abs(xy_np[i,j,1]-cy) <= sy/2+1e-7:
                    expected[i,j] = max(expected[i,j], height)
    actual = rough_surface_height_torch(torch.tensor(xy_np, dtype=torch.float64))
    np.testing.assert_allclose(actual.numpy(), expected)
