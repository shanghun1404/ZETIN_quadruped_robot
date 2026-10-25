"""Fixed evaluation geometry, independent of random row difficulty."""

import copy

from v11_video_geometry import stair_step_height_m


def fixed_eval_generator(source, scenario, level, num_envs):
    cfg = copy.deepcopy(source)
    if not 0 <= level < cfg.num_rows or num_envs < 1:
        raise ValueError("invalid fixed evaluation level or environment count")
    difficulty = level / (cfg.num_rows - 1)
    cfg.difficulty_range = (difficulty, difficulty)
    cfg.num_rows, cfg.num_cols = 1, num_envs
    cfg.curriculum = False
    metadata = {"difficulty": difficulty, "unique_tiles": num_envs}
    if scenario in ("stairs_up", "stairs_down"):
        if len(cfg.sub_terrains) != 1:
            raise ValueError("fixed stairs require a single stair type")
        stair = next(iter(cfg.sub_terrains.values()))
        height = stair_step_height_m(stair.step_height_range, source.num_rows, level)
        stair.step_height_range = (height, height)
        steps = int(min((size - 2*stair.border_width - stair.platform_width)
                        // (2*stair.step_width) + 1 for size in cfg.size))
        metadata.update(step_height_m=height, full_rise_m=(steps+1)*height)
    return cfg, metadata


def ground_below_robot(base_env):
    """Use the nearest height-scanner ray, whose footprint includes the root XY."""
    import torch

    hits = base_env.scene["height_scanner"].data.ray_hits_w
    root = base_env.scene["robot"].data.root_pos_w
    nearest = (hits[:, :, :2] - root[:, None, :2]).square().sum(-1).argmin(-1)
    return hits[torch.arange(base_env.num_envs, device=base_env.device), nearest, 2]
