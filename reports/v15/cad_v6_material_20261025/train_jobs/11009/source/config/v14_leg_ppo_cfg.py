"""Separate V14 PPO checkpoints from the V11/V13 runs."""

from __future__ import annotations

import copy

from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG


V14_LEG_PPO_CFG = copy.deepcopy(V11_TERRAIN_PPO_CFG)
V14_LEG_PPO_CFG.experiment_name = "guide_v14_interleg_collision"
V14_LEG_PPO_CFG.run_name = "v11_actor_v12_depth_guard_099"
V14_LEG_PPO_CFG.save_interval = 50
