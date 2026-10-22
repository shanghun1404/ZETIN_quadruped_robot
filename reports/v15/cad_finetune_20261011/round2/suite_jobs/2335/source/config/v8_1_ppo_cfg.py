"""V8.1 PPO settings -- V8's agent config, unchanged apart from where it logs.

V8.1 is a single reward-term experiment. Deep-copying ``V8_PPO_CFG`` rather than
restating its fields makes that structural: the network, the exploration noise,
the optimizer schedule and the rollout length cannot drift from V8 by accident,
so any difference in the result is attributable to the duty floor alone.

Only ``experiment_name`` and ``run_name`` change, and only so the runs land in
separate log directories and stay comparable side by side in TensorBoard.
"""

from __future__ import annotations

import copy

from v8_ppo_cfg import V8_PPO_CFG


V8_1_PPO_CFG = copy.deepcopy(V8_PPO_CFG)
V8_1_PPO_CFG.experiment_name = "guide_v8_1_duty_floor"
V8_1_PPO_CFG.run_name = "duty_floor_trot"
