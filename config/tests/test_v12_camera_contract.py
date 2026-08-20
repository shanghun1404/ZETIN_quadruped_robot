from types import SimpleNamespace

import pytest
import torch

from v12_camera import camera_depth_nchw, camera_spec


def test_camera_spec_matches_the_robot_forward_depth_contract_without_launching_sim():
    spec = camera_spec(0.02)

    assert spec["prim_path"] == "{ENV_REGEX_NS}/Robot/base/base/front_depth"
    assert spec["update_period"] == pytest.approx(0.02)
    assert (spec["height"], spec["width"]) == (84, 84)
    assert spec["data_types"] == ["distance_to_image_plane"]
    assert spec["clipping_range"] == (0.1, 5.0)
    assert spec["position"] == (0.0, 0.30, 0.10)
    assert spec["rotation"] == pytest.approx((0.573576, -0.819152, 0.0, 0.0))
    assert spec["convention"] == "ros"


def test_camera_depth_conversion_moves_channel_first_and_normalizes_metric_depth():
    metric_depth = torch.tensor(
        [
            [[[0.1], [5.0]], [[float("nan")], [2.55]]],
            [[[0.0], [6.0]], [[float("inf")], [float("-inf")]]],
        ]
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(output={"distance_to_image_plane": metric_depth})
    )

    normalized = camera_depth_nchw(sensor, expected_height=2, expected_width=2)

    assert normalized.shape == (2, 1, 2, 2)
    assert torch.equal(
        normalized,
        torch.tensor(
            [
                [[[0.0, 1.0], [1.0, 0.5]]],
                [[[0.0, 1.0], [1.0, 1.0]]],
            ]
        ),
    )


def test_camera_depth_conversion_rejects_an_unexpected_sensor_shape():
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={"distance_to_image_plane": torch.zeros((3, 84, 84))}
        )
    )

    with pytest.raises(ValueError, match="camera depth must have shape"):
        camera_depth_nchw(sensor)
