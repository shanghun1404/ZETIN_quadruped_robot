"""V9.4 PPO settings -- V9.3's agent config, unchanged apart from where it logs."""

from __future__ import annotations

import copy

from v9_3_ppo_cfg import V9_3_PPO_CFG


V9_4_PPO_CFG = copy.deepcopy(V9_3_PPO_CFG)
V9_4_PPO_CFG.experiment_name = "guide_v9_4_terrain"
V9_4_PPO_CFG.run_name = "terrain_trot_travelgated"
