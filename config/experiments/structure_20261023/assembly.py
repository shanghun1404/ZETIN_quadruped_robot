"""World-frame meshes (mm) of one leg and the body parts it loads."""
import numpy as np
from geom import read_stl, usd_placements, transform, sanitize, PRINT, MESHES

_cache = {}


def mesh(component, prefer_print=True):
    if component in _cache: return _cache[component]
    files = {sanitize(p.stem): p for p in MESHES.glob('*.stl')}
    p = PRINT / f'{component}.stl'
    if not (prefer_print and p.exists()):
        p = files[component]
    _cache[component] = read_stl(p)
    return _cache[component]


def leg_parts(leg='FL', with_fasteners=False):
    """[(link, component, material, V_world_mm, F)]; base parts limited to the leg's side."""
    pl = usd_placements()
    out = []
    for link in (f'{leg}_hip', f'{leg}_thigh', f'{leg}_shank', 'base'):
        for comp, mat, M in pl[link]:
            if not with_fasteners and (comp.startswith('M3') or comp.startswith('M4') or comp.startswith('dowel')):
                continue
            V, F = mesh(comp)
            W = transform(V / 1000.0, M) * 1000.0
            out.append((link, comp, mat, W, F))
    return out
