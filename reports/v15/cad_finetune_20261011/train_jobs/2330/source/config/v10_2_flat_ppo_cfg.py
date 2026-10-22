"""V10.2-flat PPO settings -- V10.1-flat's agent config, unchanged apart from logging."""

from __future__ import annotations

import copy

from v10_1_flat_ppo_cfg import V10_1_FLAT_PPO_CFG


V10_2_FLAT_PPO_CFG = copy.deepcopy(V10_1_FLAT_PPO_CFG)
V10_2_FLAT_PPO_CFG.experiment_name = "guide_v10_2_flat_robot_v2"
V10_2_FLAT_PPO_CFG.run_name = "robot_v2_flat_duty_live"
