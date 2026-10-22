import torch

from isaaclab.managers import SceneEntityCfg

import smooth_mdp


def guide_lateral_velocity_raw_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.square(smooth_mdp.guide_base_lin_vel(env, asset_cfg)[:, 1])
