"""V14 stage-two training emphasizes upward stairs without dropping other terrain."""

import os

import pytest


def test_stair_focus_terrain_mix() -> None:
    if not os.environ.get("V14_ISAAC_TEST"):
        pytest.skip("set V14_ISAAC_TEST=1 on the HPC Isaac environment")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    from v14_leg_env_cfg import GuideV14StairEnvCfg

    cfg = GuideV14StairEnvCfg()
    terrain = cfg.scene.terrain.terrain_generator.sub_terrains
    proportions = {name: item.proportion for name, item in terrain.items()}
    assert sum(proportions.values()) == pytest.approx(1.0)
    assert proportions["pyramid_stairs_inv"] == pytest.approx(0.45)
    assert proportions["pyramid_stairs"] == pytest.approx(0.25)
    assert cfg.actions.joint_pos.effort_guard_fraction == pytest.approx(0.99)
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
