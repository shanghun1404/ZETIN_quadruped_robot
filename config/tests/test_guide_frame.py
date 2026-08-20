import pytest
import torch

from guide_frame import (
    base_to_guide_observation,
    guide_forward_speed_range_to_base,
    guide_to_base_command,
    rotate_height_scan_180,
)


def test_guide_command_maps_positive_forward_to_raw_negative_x():
    semantic = torch.tensor([[0.40, 0.10, -0.20]])
    raw = guide_to_base_command(semantic)
    torch.testing.assert_close(raw, torch.tensor([[-0.40, -0.10, -0.20]]))


def test_raw_command_maps_back_exactly_once_and_preserves_yaw():
    raw = torch.tensor([[-0.40, -0.10, -0.20]])
    semantic = base_to_guide_observation(raw)
    torch.testing.assert_close(semantic, torch.tensor([[0.40, 0.10, -0.20]]))
    torch.testing.assert_close(guide_to_base_command(semantic), raw)


def test_semantic_speed_range_maps_through_the_same_frame_contract():
    assert guide_forward_speed_range_to_base((0.10, 0.25)) == (-0.25, -0.10)
    with pytest.raises(ValueError, match="non-negative"):
        guide_forward_speed_range_to_base((-0.10, 0.25))


def test_height_grid_rotates_legacy_minus_x_obstacle_to_guide_forward_row():
    raw = torch.zeros(1, 17, 11)
    raw[0, 16, 5] = 0.75
    rotated = rotate_height_scan_180(raw.reshape(1, -1))
    assert rotated.reshape(1, 17, 11)[0, 0, 5].item() == pytest.approx(0.75)


def test_height_rotation_is_an_involution_and_validates_width():
    scan = torch.arange(187, dtype=torch.float32).reshape(1, -1)
    torch.testing.assert_close(rotate_height_scan_180(rotate_height_scan_180(scan)), scan)
    with pytest.raises(ValueError, match="187"):
        rotate_height_scan_180(torch.zeros(2, 186))
