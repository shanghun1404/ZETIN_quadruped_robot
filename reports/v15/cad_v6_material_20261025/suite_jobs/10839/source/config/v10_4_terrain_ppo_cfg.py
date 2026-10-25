"""V10.4 PPO settings -- V10.3-flat's agent config, unchanged apart from logging."""

from __future__ import annotations

import copy

from v10_3_flat_ppo_cfg import V10_3_FLAT_PPO_CFG


V10_4_TERRAIN_PPO_CFG = copy.deepcopy(V10_3_FLAT_PPO_CFG)
V10_4_TERRAIN_PPO_CFG.experiment_name = "guide_v10_4_terrain_robot_v2"
V10_4_TERRAIN_PPO_CFG.run_name = "robot_v2_blind_terrain"
