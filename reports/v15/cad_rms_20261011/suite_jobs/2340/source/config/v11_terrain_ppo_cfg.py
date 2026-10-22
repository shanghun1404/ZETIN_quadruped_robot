"""V11 PPO configuration: V10.4 settings with a new experiment identity."""

from __future__ import annotations

import copy

from v10_4_terrain_ppo_cfg import V10_4_TERRAIN_PPO_CFG


V11_TERRAIN_PPO_CFG = copy.deepcopy(V10_4_TERRAIN_PPO_CFG)
V11_TERRAIN_PPO_CFG.experiment_name = "guide_v11_scanner_terrain_robot_v2"
V11_TERRAIN_PPO_CFG.run_name = "robot_v2_scanner_terrain_bootstrap"
