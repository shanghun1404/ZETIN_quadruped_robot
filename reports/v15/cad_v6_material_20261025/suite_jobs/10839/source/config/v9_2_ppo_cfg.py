"""V9.2 PPO settings -- V9.1's agent config, unchanged apart from where it logs.

Deep-copying rather than restating keeps the network, exploration noise,
optimizer schedule and rollout length identical to V9.1, so any difference in
the result is attributable to the yaw command band alone.
"""

from __future__ import annotations

import copy

from v9_1_ppo_cfg import V9_1_PPO_CFG


V9_2_PPO_CFG = copy.deepcopy(V9_1_PPO_CFG)
V9_2_PPO_CFG.experiment_name = "guide_v9_2_terrain"
V9_2_PPO_CFG.run_name = "terrain_trot_turnband"
