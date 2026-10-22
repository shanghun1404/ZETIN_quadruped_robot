"""V10-flat PPO settings -- V8.1's agent config, unchanged apart from where it logs."""

from __future__ import annotations

import copy

from v8_1_ppo_cfg import V8_1_PPO_CFG


V10_FLAT_PPO_CFG = copy.deepcopy(V8_1_PPO_CFG)
V10_FLAT_PPO_CFG.experiment_name = "guide_v10_flat_robot_v2"
V10_FLAT_PPO_CFG.run_name = "robot_v2_flat"
