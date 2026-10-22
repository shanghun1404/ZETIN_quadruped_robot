"""V10.2 PPO settings -- V10's agent config, unchanged apart from where it logs."""

from __future__ import annotations

import copy

from v10_ppo_cfg import V10_PPO_CFG


V10_2_PPO_CFG = copy.deepcopy(V10_PPO_CFG)
V10_2_PPO_CFG.experiment_name = "guide_v10_2_robot_v2"
V10_2_PPO_CFG.run_name = "robot_v2_v10_2"
