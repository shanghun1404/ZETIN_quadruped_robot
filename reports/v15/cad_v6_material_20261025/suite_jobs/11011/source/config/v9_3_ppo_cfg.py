"""V9.3 PPO settings -- V9.2's agent config, unchanged apart from where it logs.

Deep-copying rather than restating keeps the network, exploration noise,
optimizer schedule and rollout length identical to V9.2, so any difference in
the result is attributable to the velocity kernel width alone.
"""

from __future__ import annotations

import copy

from v9_2_ppo_cfg import V9_2_PPO_CFG


V9_3_PPO_CFG = copy.deepcopy(V9_2_PPO_CFG)
V9_3_PPO_CFG.experiment_name = "guide_v9_3_terrain"
V9_3_PPO_CFG.run_name = "terrain_trot_sharpvel"
