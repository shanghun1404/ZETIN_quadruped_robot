"""Training-only terrain assembly using the unchanged acceptance stair geometry."""
import numpy as np
from v15_cad_suite_protocol import SuiteCase, TerrainSpec, terrain_cuboids

CAD_STEP_HEIGHTS = (.02, .035, .05, .075, .10, .125, .157)


def cad_terrain_layout(kind, difficulty, size=(6., 18.)):
    if kind not in ('flat', 'rough', 'up', 'down') or not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError('Expected flat/rough/up/down and finite difficulty in [0,1]')
    row = min(int(difficulty * len(CAD_STEP_HEIGHTS)), len(CAD_STEP_HEIGHTS) - 1)
    height = CAD_STEP_HEIGHTS[row]
    terrain = TerrainSpec(kind=kind if kind in ('flat', 'rough') else 'stairs_' + kind,
        length_m=size[1]-4., step_height_m=height if kind in ('up', 'down') else 0., rough_seed=73)
    case = SuiteCase('training', 'Four actual risers, +Y lane', 20., terrain, ())
    origin = np.array([size[0] / 2., size[1] / 2. if kind == 'flat' else 2.,
                       4 * height if kind == 'down' else 0.])
    return dict(terrain=terrain, blocks=terrain_cuboids(case), origin=origin, step_height_m=height)


def rough_tile_specs(seed=73):
    """Return exact authored training tile centers, XY extents and top heights."""
    case = SuiteCase('training_rough', 'Training rough ground', 20.,
                     TerrainSpec(kind='rough', rough_seed=seed), ())
    return np.asarray([(b.center[0], b.center[1], b.size[0], b.size[1], b.size[2])
                       for b in terrain_cuboids(case)])


def rough_surface_height_torch(xy, boxes=None):
    """Maximum tile top at XY, including shared edges; outside tiles is zero."""
    import torch
    if boxes is None:
        boxes = torch.as_tensor(rough_tile_specs(), device=xy.device, dtype=xy.dtype)
    inside = (xy.unsqueeze(-2) - boxes[:, :2]).abs() <= boxes[:, 2:4] / 2. + 1e-7
    return torch.where(inside.all(-1), boxes[:, 4], 0.).amax(-1)


def surface_height(xy, height, descending):
    xy, height, descending = np.asarray(xy), np.asarray(height), np.asarray(descending, bool)
    levels = np.clip(np.floor((xy[..., 1] - 2.) / .3) + 1., 0., 4.)
    value = np.where(descending, 4. - levels, levels) * height
    return np.where(np.abs(xy[..., 0]) <= 1., value, 0.)


def promotion_flags(relative_xy, terminated, stairs):
    xy, terminated, stairs = np.asarray(relative_xy), np.asarray(terminated, bool), np.asarray(stairs, bool)
    passed = stairs & ~terminated & (xy[:, 1] >= 3.9) & (np.abs(xy[:, 0]) <= .7)
    return passed, stairs & ~passed


def cad_lane_terrain(difficulty, cfg):
    # Called by Isaac's terrain generator only after SimulationApp initialization.
    import trimesh
    layout = cad_terrain_layout(cfg.kind, difficulty, cfg.size)
    base = trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], .05))
    base.apply_translation([cfg.size[0]/2., cfg.size[1]/2., -.025])
    meshes = [base]
    for block in layout['blocks']:
        mesh = trimesh.creation.box(extents=block.size)
        mesh.apply_translation(np.asarray(block.center) + [layout['origin'][0], 2., 0.])
        meshes.append(mesh)
    return meshes, layout['origin']
