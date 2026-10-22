"""V13 keeps V11 PPO and network settings, with a separate experiment name."""

from __future__ import annotations

import copy

from v11_terrain_ppo_cfg import V11_TERRAIN_PPO_CFG


V13_TORQUE_PPO_CFG = copy.deepcopy(V11_TERRAIN_PPO_CFG)
V13_TORQUE_PPO_CFG.experiment_name = "guide_v13_v12_depth_torque"
V13_TORQUE_PPO_CFG.run_name = "v12_frozen_depth_actor_torque"
