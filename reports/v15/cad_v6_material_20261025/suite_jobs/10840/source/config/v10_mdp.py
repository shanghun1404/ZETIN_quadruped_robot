"""V10 observation term: terrain relief that does not depend on the body frame.

Why the stock term cannot be used on this asset
------------------------------------------------
``mdp.height_scan`` returns ``sensor.data.pos_w.z - hit_z - offset``. It measures
the ground relative to the *sensor's parent body frame*. The MJCF->USD conversion
of this robot leaves body frames displaced from their geometry by a large
constant -- a foot frame can read above the trunk -- so that difference is not a
clearance and no fixed ``offset`` corrects it.

**Corrected in V10.3:** this previously justified the point with "at rest the
root frame reads z = -0.419". That figure had its sign inverted (the settled root
is +0.445) and it referred to the articulation root, which is *not* one of the
displaced frames. The saturation measurements below stand; only the explanation
of the root value was wrong.

That was measured, not assumed. With the stock ``offset = 0.5`` the scan would
sit near -0.92. Calibrating ``offset`` to the settled frame value (-0.4192) was
tried and **still saturated**::

    scan mean            -0.9688
    fraction at -1 clip   0.9688
    std                   0.1741

96.9% of the 187 scan points were pinned at the clip, so the policy received an
almost constant vector and could not see stairs, boxes or slopes at all. The
matching training run confirmed it: terrain level fell to 0.000 and mean episode
length to 7.8 steps of 1000.

What this returns instead
-------------------------
Terrain relief relative to the ground beneath the robot::

    relief = median(hit_z) - hit_z

The median over the scan's own hits is the local ground level, so the output is a
*difference between two ray hits* and the body frame never enters it. Flat ground
reads 0 everywhere, a step 0.15 m up ahead reads -0.15, a drop reads +0.15. The
[-1, 1] clip then spans a metre of relief in each direction, which covers every
sub-terrain the generator produces.

The median is used rather than the mean because a large obstacle filling half the
grid would drag a mean away from the ground the robot is actually standing on.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def relative_height_scan(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ground relief around the robot, in metres, independent of the body frame."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w[..., 2]

    # A ray that hits nothing returns inf; substitute the local ground so a single
    # miss at a terrain border cannot poison the median or the returned vector.
    finite = torch.isfinite(hits)
    safe = torch.where(finite, hits, torch.zeros_like(hits))
    counts = finite.sum(dim=1, keepdim=True).clamp(min=1)
    fallback = (safe * finite).sum(dim=1, keepdim=True) / counts
    hits = torch.where(finite, hits, fallback.expand_as(hits))

    ground = hits.median(dim=1, keepdim=True).values
    return ground - hits
