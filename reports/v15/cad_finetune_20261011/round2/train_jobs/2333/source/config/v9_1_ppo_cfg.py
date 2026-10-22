"""V9.1 PPO settings -- V9's agent config, unchanged apart from where it logs.

V9.1 is a single termination-set experiment. Deep-copying ``V9_PPO_CFG`` rather
than restating its fields makes that structural: the network, the exploration
noise, the optimizer schedule and the rollout length cannot drift from V9 by
accident, so any difference in the result is attributable to the termination
change alone. Same reasoning as ``v8_1_ppo_cfg``.
"""

from __future__ import annotations

import copy

from v9_ppo_cfg import V9_PPO_CFG


V9_1_PPO_CFG = copy.deepcopy(V9_PPO_CFG)
V9_1_PPO_CFG.experiment_name = "guide_v9_1_terrain"
V9_1_PPO_CFG.run_name = "terrain_trot_survivable"
