"""V10 flat WASD PPO identity with the established 48-to-12 network."""

from __future__ import annotations

import copy

from v10_3_flat_ppo_cfg import V10_3_FLAT_PPO_CFG


V10_WASD_PPO_CFG = copy.deepcopy(V10_3_FLAT_PPO_CFG)
V10_WASD_PPO_CFG.experiment_name = "guide_v10_wasd_flat_robot_v2"
V10_WASD_PPO_CFG.run_name = "robot_v2_flat_wasd_demo"
