"""Depth-camera configuration and tensor conversion for V12."""

from __future__ import annotations

import torch

from v12_vision_student import normalize_depth


CAMERA_HEIGHT = 84
CAMERA_WIDTH = 84
CAMERA_NEAR_M = 0.1
CAMERA_FAR_M = 5.0
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/base/base/front_depth"
CAMERA_POSITION_M = (0.0, 0.30, 0.10)
CAMERA_ROTATION_ROS_WXYZ = (0.573576, -0.819152, 0.0, 0.0)


def camera_spec(step_dt: float) -> dict[str, object]:
    """Return the simulator-independent literal V12 camera contract."""
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    return {
        "prim_path": CAMERA_PRIM_PATH,
        "update_period": step_dt,
        "height": CAMERA_HEIGHT,
        "width": CAMERA_WIDTH,
        "data_types": ["distance_to_image_plane"],
        "clipping_range": (CAMERA_NEAR_M, CAMERA_FAR_M),
        "position": CAMERA_POSITION_M,
        "rotation": CAMERA_ROTATION_ROS_WXYZ,
        "convention": "ros",
    }


def make_v12_depth_camera_cfg(step_dt: float):
    """Create the V12 tiled depth camera after Isaac Sim is launched."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    spec = camera_spec(step_dt)
    return TiledCameraCfg(
        prim_path=spec["prim_path"],
        update_period=spec["update_period"],
        height=spec["height"],
        width=spec["width"],
        data_types=spec["data_types"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=spec["clipping_range"],
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=spec["position"],
            rot=spec["rotation"],
            convention=spec["convention"],
        ),
    )


def camera_depth_nchw(
    sensor,
    *,
    expected_height: int = CAMERA_HEIGHT,
    expected_width: int = CAMERA_WIDTH,
) -> torch.Tensor:
    """Read metric tiled-camera depth and return normalized NCHW depth."""

    metric_depth = sensor.data.output["distance_to_image_plane"]
    expected_tail = (expected_height, expected_width, 1)
    if metric_depth.ndim != 4 or tuple(metric_depth.shape[1:]) != expected_tail:
        raise ValueError(
            "camera depth must have shape "
            f"[batch, {expected_height}, {expected_width}, 1]"
        )
    normalized = normalize_depth(
        metric_depth,
        near_m=CAMERA_NEAR_M,
        far_m=CAMERA_FAR_M,
    )
    return normalized.permute(0, 3, 1, 2).contiguous()
