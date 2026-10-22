"""Stage 1 V11: V10.4 blind terrain with a frame-free forward height scan."""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import v10_mdp
from guide_robot_v2_cfg import FEET_BELOW_ROOT_FRAME_M, FEET_SPAWN_CLEARANCE_M
from v8_env_cfg import V8ObservationsCfg
from v10_4_terrain_env_cfg import (
    GuideV2BlindTerrainEnvCfg,
    V10_4_REWARD_WEIGHTS,
)
from v11_layout import scanner_layout, terrain_spawn_root_z


SCANNER_LAYOUT = scanner_layout("+y")
SCAN_POINTS = int(SCANNER_LAYOUT["points"])
TERRAIN_SPAWN_ROOT_FRAME_Z_M = terrain_spawn_root_z(
    FEET_SPAWN_CLEARANCE_M, FEET_BELOW_ROOT_FRAME_M
)
V11_REWARD_WEIGHTS = dict(V10_4_REWARD_WEIGHTS)


@configclass
class V11ObservationsCfg(V8ObservationsCfg):
    @configclass
    class PolicyCfg(V8ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=v10_mdp.relative_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class GuideV2ScannerTerrainEnvCfg(GuideV2BlindTerrainEnvCfg):
    """V10.4 terrain dynamics plus 187 frame-independent relief samples."""

    observations: V11ObservationsCfg = V11ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.pos = (
            0.0,
            0.0,
            TERRAIN_SPAWN_ROOT_FRAME_Z_M,
        )
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/base",
            offset=RayCasterCfg.OffsetCfg(pos=SCANNER_LAYOUT["offset"]),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=float(SCANNER_LAYOUT["resolution"]),
                size=list(SCANNER_LAYOUT["size"]),
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
