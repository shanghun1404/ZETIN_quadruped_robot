"""V10 PPO settings -- V9.4's agent config, unchanged apart from where it logs."""

from __future__ import annotations

import copy

from v9_4_ppo_cfg import V9_4_PPO_CFG


V10_PPO_CFG = copy.deepcopy(V9_4_PPO_CFG)
V10_PPO_CFG.experiment_name = "guide_v10_robot_v2"
V10_PPO_CFG.run_name = "robot_v2_terrain"
