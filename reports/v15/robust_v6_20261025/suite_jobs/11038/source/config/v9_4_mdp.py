"""V9.4 reward term: pay for the trot in proportion to how far the robot travels.

Why this exists
---------------
V9.3 sharpened the linear-velocity kernel and the robot still did not travel
(0.158-0.224 m/s against a commanded 0.500). The reward breakdown at iteration
4999 explains why -- a robot that stands still and cycles its legs collects
almost the whole positive table::

    term                     measured   what a *stationary* robot scores
    base_angular_velocity      4.543     4.524   (yaw cmd 0.2, std 2.0)
    gait                       6.745     ~7      (contact timing only)
    air_time                   0.872     ~0.9
    base_linear_velocity       2.226     1.506   (cmd 0.6, std 0.5)
                             -------   -------
    positive total            14.690     ~13.6   = 93%

``GaitReward`` is the largest single term and its own gate is what makes it free::

    # isaaclab_tasks/.../spot/mdp/rewards.py
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold),
        sync_reward * async_reward, 0.0,
    )

The gate is an **or**: merely being *commanded* to move unlocks the full reward,
whatever the body actually does. Six pairwise contact-timing comparisons are then
satisfied perfectly by marching on the spot.

The change
----------
``SpeedRampedGaitReward`` subclasses the maintained ``GaitReward`` -- inheriting
all six pairwise comparisons and any upstream fix to them -- and scales its
output by how fast the body is actually moving::

    ramp  = clamp(body_speed / velocity_threshold, 0, 1)
    value = ramp * stock_gait_reward

Why a ramp and not an ``and``
-----------------------------
Replacing the ``or`` with an ``and`` was the obvious edit and is the wrong one.
V9.3's robot travels at 0.22 m/s against a 0.5 m/s threshold, so a hard gate
would zero 6.7 per second at once and leave **no gradient at all** until the
policy first exceeded 0.5 m/s by chance. That is the same trap that ruled out a
sharper velocity kernel in V9.3: it improves the payoff gap by destroying the
signal that teaches the way there.

The ramp is monotone in speed, so at the measured 0.22 m/s the policy keeps
0.22/0.5 = 44% of the gait reward and every increment of speed pays more. It
loses about 3.8 per second at the current behaviour -- three times what V9.3's
kernel change was worth -- while pointing uphill the whole way.

Above ``velocity_threshold`` the ramp saturates at 1.0, so a robot that already
walks at the commanded speed sees exactly the published Spot reward and nothing
about V8.1's validated flat-ground gait changes.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp


class SpeedRampedGaitReward(spot_mdp.GaitReward):
    """Stock Spot gait reward, scaled by actual travel speed.

    Inherits the six pairwise contact-timing comparisons rather than
    reimplementing them, so upstream corrections to the gait definition still
    apply. Only the gate changes.
    """

    def __call__(
        self,
        env,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg,
        sensor_cfg,
    ) -> torch.Tensor:
        reward = super().__call__(
            env,
            std=std,
            max_err=max_err,
            velocity_threshold=velocity_threshold,
            synced_feet_pair_names=synced_feet_pair_names,
            asset_cfg=asset_cfg,
            sensor_cfg=sensor_cfg,
        )
        return reward * speed_ramp(self.asset.data.root_lin_vel_b[:, :2], velocity_threshold)


def speed_ramp(planar_velocity: torch.Tensor, velocity_threshold: float) -> torch.Tensor:
    """Fraction of the gait reward earned at a given body speed, in ``[0, 1]``.

    Separated from the term so the shape can be tested without a simulator.
    Linear in speed up to ``velocity_threshold`` and saturating at 1.0 above it:
    a robot already walking at the commanded speed is scored exactly as the
    published recipe scores it.
    """
    if velocity_threshold <= 0.0:
        raise ValueError("velocity_threshold must be positive")
    speed = torch.linalg.norm(planar_velocity, dim=1)
    return torch.clamp(speed / velocity_threshold, min=0.0, max=1.0)
