import pytest
import torch

from v11_layout import scanner_layout, scan_quality, terrain_spawn_root_z


def test_scanner_layout_points_along_robot_v2_positive_y_axis():
    result = scanner_layout("+y")

    assert result == {
        "offset": (0.0, 0.8, 20.0),
        "size": (1.0, 1.6),
        "resolution": 0.1,
        "rows": 11,
        "columns": 17,
        "points": 187,
    }


def test_scanner_layout_rejects_an_unverified_axis():
    with pytest.raises(ValueError, match="unsupported forward axis"):
        scanner_layout("-x")


def test_terrain_spawn_adds_clearance_to_feet_below_root():
    assert terrain_spawn_root_z(0.32, 0.445) == pytest.approx(0.765)


def test_scan_quality_reports_nonfinite_and_clip_saturation():
    scan = torch.tensor([[0.0, -1.0, 1.0, float("inf")]])

    result = scan_quality(scan, clip=1.0)

    assert result["finite_fraction"] == pytest.approx(0.75)
    assert result["low_saturation_fraction"] == pytest.approx(0.25)
    assert result["high_saturation_fraction"] == pytest.approx(0.25)
