"""Per-part linear FEA of the FL leg load path and the front body shell.

For each part: body-side contact faces clamped, foot-side contact faces loaded
with the foot force (+ its moment) for unit world forces Fx, Fy, Fz in the
nominal stance posture. Results are stored per unit load so design cases can
be superposed.

Run: PYTHONPATH=../fea_pkgs:. python3 analyze_parts.py <part> [<part> ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from assembly import leg_parts
from fea_core import Part, TriSurface, contact_facets, surface_samples, tet_mesh

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

# Bambu PLA Basic TDS V2.0 (100 % infill, annealed): E_XY 2680 MPa. Poisson's ratio is not listed: 0.35 assumed.
PLA_E, PLA_NU = 2680.0, 0.35
# Aluminium 6061 (design log §11 material for the 2020 profiles): handbook E = 68.9 GPa, nu = 0.33.
AL_E, AL_NU = 68900.0, 0.33
# Print settings (bambu/profiles/process_legs_*.json, process_body_*.json): walls 0.42 + n-1 x 0.45 mm.
PRINT = {
    'legs': dict(skin=0.42 + 3 * 0.45, infill=0.30),
    'body': dict(skin=0.42 + 2 * 0.45, infill=0.20),
}
# Gibson-Ashby bending-dominated estimate for the gyroid core: E_core = E_s * rho^2 (assumption).
def core_modulus(rho):
    return PLA_E * rho ** 2


# Nominal stance (cad_model canonical_nominal_q mapped to FL physical joints): hip pitch +0.55 rad
# about (-1,0,0) at (-100, 315, 75) mm, knee +1.1 rad about (+1,0,0) at (-170, 315, -215) mm.
HIP = np.array([-100.0, 315.0, 75.0])
KNEE = np.array([-170.0, 315.0, -215.0])
Q_PITCH, Q_KNEE = 0.55, 1.10


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


R_THIGH = rot_x(-Q_PITCH)           # rotation +0.55 about (-1, 0, 0)
R_SHANK = R_THIGH @ rot_x(Q_KNEE)    # then +1.1 about (+1, 0, 0)


def to_world(link, x):
    """Zero-pose point(s) of ``link`` to world in the nominal stance."""
    x = np.atleast_2d(x)
    if link.endswith('thigh'):
        return (x - HIP) @ R_THIGH.T + HIP
    if link.endswith('shank'):
        knee_world = (KNEE - HIP) @ R_THIGH.T + HIP
        return (x - KNEE) @ R_SHANK.T + knee_world
    return x


def link_rotation(link):
    return R_THIGH if link.endswith('thigh') else R_SHANK if link.endswith('shank') else np.eye(3)


def to_link(link, x_world):
    R = link_rotation(link)
    x_world = np.atleast_2d(x_world)
    if link.endswith('thigh'):
        return (x_world - HIP) @ R + HIP
    if link.endswith('shank'):
        knee_world = (KNEE - HIP) @ R_THIGH.T + HIP
        return (x_world - knee_world) @ R + KNEE
    return x_world


PARTS = {
    # name: (link, component, instance selector, group, fixed neighbours, loaded neighbours, maxvol mm^3, tol mm)
    'knee': ('FL_shank', 'knee', None, 'legs', [('FL_shank', 'AK10_9_KV60_V3_0_3D', None)], [('FL_shank', 'profile', None)], 3.0, 0.35),
    'com_pro_knee': ('FL_thigh', 'com_pro', 'lower', 'legs', [('FL_thigh', 'profile', None)], [('FL_shank', 'AK10_9_KV60_V3_0_3D', None)], 3.0, 0.35),
    'com_pro_hub': ('FL_thigh', 'com_pro', 'upper', 'legs', [('FL_thigh', 'fake_cover', None)], [('FL_thigh', 'profile', None)], 3.0, 0.35),
    'fake_cover': ('FL_thigh', 'fake_cover', None, 'legs', [('FL_thigh', 'fake_motor', None)], [('FL_thigh', 'com_pro', 'upper')], 3.0, 0.35),
    'fake_motor': ('FL_thigh', 'fake_motor', None, 'legs', [('FL_hip', 'AK10_9_KV60_V3_0_3D', None)], [('FL_thigh', 'fake_cover', None)], 4.0, 0.35),
    'combine_1': ('FL_hip', 'combine_1', None, 'legs', [('base', 'ak70_10', 'FL')], [('FL_hip', 'AK10_9_KV60_V3_0_3D', None)], 4.0, 0.35),
    'body_shell_F': ('base', 'body_shell_F', None, 'body',
                     [('base', n, None) for n in ('body_bulkhead', 'body_shell_B', 'body_bottom_A', 'body_bottom_B', 'body_top_A', 'body_top_B')],
                     [('base', 'ak70_10', 'FL')], 25.0, 0.35),
    'profile_shank': ('FL_shank', 'profile', None, 'aluminium', [('FL_shank', 'knee', None)], [('FL_shank', 'tennis', None)], 6.0, 0.35),
    'profile_thigh': ('FL_thigh', 'profile', None, 'aluminium', [('FL_thigh', 'com_pro', 'upper')], [('FL_thigh', 'com_pro', 'lower')], 6.0, 0.35),
}


def select(parts, link, comp, which):
    hits = [(W, F) for l, c, m, W, F in parts if l == link and c == comp]
    if which == 'upper':
        hits = [max(hits, key=lambda h: h[0][:, 2].mean())]
    elif which == 'lower':
        hits = [min(hits, key=lambda h: h[0][:, 2].mean())]
    elif which == 'FL':
        hits = [h for h in hits if h[0][:, 0].mean() < 0 and h[0][:, 1].mean() > 0]
    assert len(hits) == 1, (link, comp, which, len(hits))
    return hits[0]


def foot_contact_world(parts):
    W, _ = select(parts, 'FL_shank', 'tennis', None)
    centre0 = 0.5 * (W.min(axis=0) + W.max(axis=0))
    centre0[2] = W[:, 2].min() + 20.0  # ball radius 20 mm
    return to_world('FL_shank', centre0)[0] - np.array([0.0, 0.0, 20.0])


def mixed_skin(mesh, surf_pts, skin):
    """Fraction of each tet within ``skin`` mm of the surface (centroid + 4 interior points)."""
    tree = cKDTree(surf_pts)
    P = mesh.p[:, mesh.t]  # (3, 4, nel)
    a, b = 0.5854101966249685, 0.1381966011250105
    pts = [P.mean(axis=1)]
    for i in range(4):
        w = np.full(4, b); w[i] = a
        pts.append(np.einsum('k,dkn->dn', w, P))
    inside = np.stack([tree.query(q.T)[0] < skin for q in pts])
    return inside.mean(axis=0)


VARIANTS = {
    'solid': '100 % infill print (whole part solid PLA)',
    'al': 'whole part machined from aluminium 6061',
    'al_endwall': 'aluminium 6061 disk replacing the 9 mm hub end wall (x -139.7..-130.7)',
    'brace': 'knee cup socket +x face tied to the motor rear face by a stiff aluminium brace (upper bound)',
}


def bolt_region_facets(mesh, V, F, fixed_s, loaded_s, reach=12.0):
    from holes import holes
    bf = mesh.boundary_facets()
    cen = mesh.p[:, mesh.facets[:, bf]].mean(axis=1).T
    out = {'fixed': [], 'loaded': []}
    log = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for centre, r, (a0, a1) in holes(V, F, axis):
            line = np.zeros((int((a1 - a0 + 2 * reach) / 0.5) + 1, 3))
            line[:, axis] = np.linspace(a0 - reach, a1 + reach, len(line))
            line[:, other[0]], line[:, other[1]] = centre
            hit = None
            for key, surfs in (('fixed', fixed_s), ('loaded', loaded_s)):
                if any(np.min(s.distance(line, 2.5)) < 2.5 for s in surfs):  # passes threaded or clearance holes of the neighbour
                    hit = key
                    break
            if hit is None:
                continue
            d = np.linalg.norm(cen[:, other] - centre, axis=1)
            sel = (d < r + 2.0) & (cen[:, axis] > a0 - 2.5) & (cen[:, axis] < a1 + 2.5)
            out[hit].append(bf[sel])
            log.append(dict(axis=axis, centre=[float(c) for c in centre], dia=float(2 * r), to=hit, facets=int(sel.sum())))
    return {k: (np.concatenate(v) if v else np.zeros(0, int)) for k, v in out.items()}, log


def analyse(name):
    base, _, variant = name.partition('__')
    coarse = variant == 'coarse'
    if variant and variant not in VARIANTS and not coarse:
        raise ValueError(f'unknown variant {variant}')
    link, comp, which, group, fixed_n, load_n, maxvol, tol = PARTS[base]
    parts = leg_parts('FL')
    V, F = select(parts, link, comp, which)
    t0 = time.time()
    minratio = float(__import__('os').environ.get('MINRATIO', '2.5'))
    if coarse:
        maxvol *= 2.5
    mesh = tet_mesh(V, F, maxvol, minratio=minratio)
    max_tets = int(__import__('os').environ.get('MAX_TETS', '130000'))
    print(f'{name}: {mesh.t.shape[1]} tets (maxvol {maxvol})', flush=True)
    if mesh.t.shape[1] > max_tets:
        raise RuntimeError(f'{name}: {mesh.t.shape[1]} tets exceeds MAX_TETS={max_tets}; raise maxvol')
    surf = surface_samples(V, F, 0.4)
    if group == 'aluminium':
        E_el = np.full(mesh.t.shape[1], AL_E)
        nu = AL_NU
        phi = np.ones(mesh.t.shape[1])
        E_skin, E_core = AL_E, AL_E
    else:
        cfg = PRINT[group]
        phi = mixed_skin(mesh, surf, cfg['skin'])
        E_skin, E_core = PLA_E, core_modulus(cfg['infill'])
        if variant == 'solid':
            phi = np.ones_like(phi)
        E_el = phi * E_skin + (1 - phi) * E_core  # Voigt mix within each tet
        nu = PLA_NU
        if variant == 'al':
            E_el = np.full_like(E_el, AL_E); phi = np.ones_like(phi); E_skin = E_core = AL_E; group = 'aluminium'
        elif variant == 'al_endwall':
            cx = mesh.p[0, mesh.t].mean(axis=0)
            wall = (cx > -139.7) & (cx < -130.7)
            E_el = np.where(wall, AL_E, E_el)
    part = Part(mesh, E_el, nu)

    fixed_s = [TriSurface(*select(parts, l, c, w)) for l, c, w in fixed_n]
    loaded_s = [TriSurface(*select(parts, l, c, w)) for l, c, w in load_n]
    fixed = np.concatenate([contact_facets(mesh, t, tol) for t in fixed_s])
    loaded = np.concatenate([contact_facets(mesh, t, tol) for t in loaded_s])
    # Bolts are not modelled: surfaces within (hole radius + 2 mm) of a bolt axis that runs into a
    # neighbour (hole wall, counterbore wall and head seat) join that neighbour's interface.
    hf, hole_log = bolt_region_facets(mesh, V, F, fixed_s, loaded_s)
    fixed = np.union1d(fixed, hf['fixed'])
    loaded = np.union1d(loaded, hf['loaded'])
    if variant == 'brace':
        bf = mesh.boundary_facets()
        pts = mesh.p[:, mesh.facets[:, bf]]
        normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0], axis=0)
        flat_x = np.all(np.abs(pts[0] + 155.0) < 0.2, axis=0) & (np.abs(normal[0]) > 0.99 * np.linalg.norm(normal, axis=0))
        low = pts[2].mean(axis=0) < -262.0
        fixed = np.union1d(fixed, bf[flat_x & low])
    loaded = np.setdiff1d(loaded, fixed)
    if len(fixed) == 0 or len(loaded) == 0:
        raise RuntimeError(f'{name}: empty interface (fixed {len(fixed)}, loaded {len(loaded)})')
    from fea_core import facet_area
    foot_w = foot_contact_world(parts)
    foot_l = to_link(link, foot_w)[0]
    R = link_rotation(link)
    loads, info = [], []
    for k in range(3):
        F_world = np.eye(3)[k]
        F_link = R.T @ F_world
        f, c, area = part.load_vector(loaded, F_link, np.zeros(3))
        moment = np.cross(foot_l - c, F_link)
        f, c, area = part.load_vector(loaded, F_link, moment)
        loads.append(f)
        info.append(dict(centroid=c.tolist(), area=float(area), moment_Nmm_per_N=moment.tolist()))
    us = part.solve(loads, fixed)
    S_all, foot_disp, U = [], [], []
    for u in us:
        U.append(part.vertex_displacement(u))
        S, vm, pr = part.stresses(u)
        S_all.append(S)
        a, theta, cfit = part.rigid_fit(u, loaded)
        d_link = a + np.cross(theta, foot_l - cfit)
        foot_disp.append(R @ d_link)  # world frame, mm per N
    vol = np.abs(np.linalg.det(np.stack([mesh.p[:, mesh.t[i]] - mesh.p[:, mesh.t[0]] for i in (1, 2, 3)], axis=0).transpose(2, 0, 1))) / 6
    np.savez_compressed(OUT / f'{name}.npz', p=mesh.p, t=mesh.t, S=np.stack(S_all), phi=phi, E=E_el, vol=vol,
                        fixed=fixed, loaded=loaded, facets=mesh.facets, f2t=mesh.f2t, foot_disp=np.array(foot_disp), U=np.stack(U),
                        V=V, F=F)
    summary = dict(part=name, variant=VARIANTS.get(variant, 'as printed (legs 4 walls 30 % gyroid, body 3 walls 20 % gyroid)'), link=link, component=comp, group=group, tets=int(mesh.t.shape[1]), dofs=int(part.basis.N),
                   maxvol_mm3=maxvol, minratio=minratio,
                   cg_iterations=part.iterations, volume_mm3=float(vol.sum()),
                   fixed_area_mm2=float(facet_area(mesh, fixed).sum()), loaded_area_mm2=float(facet_area(mesh, loaded).sum()),
                   bolt_holes=hole_log,
                   E_skin=E_skin, E_core=E_core, mean_skin_fraction=float((phi * vol).sum() / vol.sum()),
                   foot_world=foot_w.tolist(), foot_link=foot_l.tolist(), loads=info,
                   foot_compliance_mm_per_N=np.array(foot_disp).tolist(), seconds=time.time() - t0)
    (OUT / f'{name}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ('part', 'tets', 'dofs', 'cg_iterations', 'fixed_area_mm2', 'loaded_area_mm2', 'mean_skin_fraction', 'seconds')}), flush=True)
    return summary


if __name__ == '__main__':
    for name in sys.argv[1:]:
        analyse(name)
