"""V15.1 rated motor load objective, independent of Isaac and joint ordering."""

from __future__ import annotations


def joint_ratings_nm(joint_names):
    """Continuous torque ratings supplied for the V15.1 experiment."""
    by_type = {"HAA": 8.3, "HFE": 18.0, "KFE": 18.0}
    try:
        return tuple(by_type[name.rsplit("_", 1)[-1]] for name in joint_names)
    except KeyError as error:
        raise ValueError(f"Unknown motor joint type: {error.args[0]}") from error


def rated_squared_load(torque_nm, ratings_nm):
    """Mean squared rated fraction per environment (NumPy or Torch arrays).

    Integrating this cost over time penalizes sustained load, including demand
    below rated torque. Ratings must be positive; production uses joint_ratings_nm.
    No clipping: exceeding rated torque remains increasingly expensive.
    """
    if torque_nm.ndim != 2 or ratings_nm.ndim != 1 or torque_nm.shape[-1] != ratings_nm.shape[0]:
        raise ValueError("torque must be [environment, joint] and ratings [joint]")
    return ((torque_nm / ratings_nm) ** 2).mean(axis=-1)


def estimated_rated_motor_load(env, asset_cfg):
    """Isaac reward adapter: clipped implicit PD estimates, not solver torque."""
    import torch

    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    names = asset.joint_names[ids] if isinstance(ids, slice) else [asset.joint_names[i] for i in ids]
    torque = asset.data.applied_torque[:, ids]
    ratings = torch.tensor(joint_ratings_nm(names), dtype=torque.dtype, device=torque.device)
    return rated_squared_load(torque, ratings)
