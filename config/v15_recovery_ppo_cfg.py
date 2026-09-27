"""V15-R PPO: V8 network/algorithm, asymmetric critic with privileged height and velocity."""

from __future__ import annotations

import copy

from v8_ppo_cfg import V8_PPO_CFG


V15_RECOVERY_PPO_CFG = copy.deepcopy(V8_PPO_CFG)
V15_RECOVERY_PPO_CFG.experiment_name = "guide_v15_recovery"
V15_RECOVERY_PPO_CFG.run_name = "blind_fall_recovery"
V15_RECOVERY_PPO_CFG.obs_groups = {"policy": ["policy"], "critic": ["policy", "privileged"]}
V15_RECOVERY_PPO_CFG.save_interval = 100
V15_RECOVERY_PPO_CFG.algorithm.entropy_coef = 0.005
