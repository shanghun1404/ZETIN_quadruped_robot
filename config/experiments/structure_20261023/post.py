"""Superpose unit-load results into design cases, check strength, draw figures.

Run: PYTHONPATH=../fea_pkgs:. python3 post.py <part> [...]
"""
import json
import sys
from pathlib import Path

# The system matplotlib's mpl_toolkits package would shadow the newer one in fea_pkgs.
sys.path = [p for p in sys.path if 'dist-packages' not in p]
sys.modules.pop('mpl_toolkits', None)  # registered at start-up by Debian's matplotlib .pth file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent / 'results'
# Foot loads (world N) from the round2 nine-case ROS2/Isaac suite of model_15699 (shank_support_force):
# stance p99 220 N, p99.9 380 N, single-sample peak 1045 N, largest horizontal 197 N (rough case).
CASES = {
    'D1 stance p99 (Fz 220 N)': [(0, 0, 220.0)],
    'D2 severe p99.9 (Fz 380 N)': [(0, 0, 380.0)],
    'D3 severe + horizontal (Fz 380 N, Fxy 197 N)': [(sx * 197.0, 0, 380.0) for sx in (-1, 1)] + [(0, sy * 197.0, 380.0) for sy in (-1, 1)],
    'D4 impact peak (Fz 1045 N)': [(0, 0, 1045.0)],
}
# Bambu PLA Basic TDS V2.0: tensile strength XY 39 MPa, Z 35 MPa. Aluminium 6061-T6 yield 276 MPa (handbook).
STRENGTH = {'pla': 35.0, 'aluminium': 276.0}


def evaluate(name):
    d = np.load(OUT / f'{name}.npz')
    meta = json.loads((OUT / f'{name}.json').read_text())
    S_unit, E, vol, phi = d['S'], d['E'], d['vol'], d['phi']
    group = meta['group']
    scale = (meta['E_skin'] / E)[:, None, None]
    skin_present = phi > 0.0
    if group != 'aluminium':
        skin_present &= E < 10000.0  # PLA elements only (an aluminium insert is reported separately)
    rows = {}
    for case, forces in CASES.items():
        worst = None
        for F in forces:
            S = np.tensordot(np.asarray(F), S_unit, axes=(0, 0)) * scale  # stress in the solid skin material
            pr = np.linalg.eigvalsh(S)
            dev = S - np.eye(3)[None] * (np.trace(S, axis1=1, axis2=2) / 3)[:, None, None]
            vm = np.sqrt(1.5 * np.einsum('nij,nij->n', dev, dev))
            s1 = pr[:, 2]
            sel = skin_present
            order = np.argsort(s1[sel])
            cum = np.cumsum(vol[sel][order]) / vol[sel].sum()
            s1_p995 = s1[sel][order][np.searchsorted(cum, 0.995)]
            disp = np.asarray(F) @ np.asarray(meta['foot_compliance_mm_per_N'])
            res = dict(force=list(F), s1_max=float(s1[sel].max()), s1_p995=float(s1_p995), vm_max=float(vm[sel].max()),
                       s3_min=float(pr[sel, 0].min()), foot_disp_mm=disp.tolist(), argmax=int(np.flatnonzero(sel)[np.argmax(s1[sel])]))
            if worst is None or res['s1_max'] > worst['s1_max']:
                worst, worst_vm = res, vm
        strength = STRENGTH['aluminium' if group == 'aluminium' else 'pla']
        worst['fos_max'] = strength / worst['s1_max']
        worst['fos_p995'] = strength / worst['s1_p995']
        rows[case] = worst
    meta['cases'] = rows
    (OUT / f'{name}.json').write_text(json.dumps(meta, indent=1))
    return meta


def boundary_colour(name, case):
    d = np.load(OUT / f'{name}.npz')
    meta = json.loads((OUT / f'{name}.json').read_text())
    F = np.asarray(meta['cases'][case]['force'])
    S = np.tensordot(F, d['S'], axes=(0, 0)) * (meta['E_skin'] / d['E'])[:, None, None]
    s1 = np.linalg.eigvalsh(S)[:, 2]
    facets, f2t, p = d['facets'], d['f2t'], d['p']
    bnd = np.flatnonzero(f2t[1] == -1)
    tri = p[:, facets[:, bnd]].transpose(2, 1, 0)
    return tri, s1[f2t[0, bnd]], bnd, d


def figure(name, case, views=((20, -60), (20, 120), (-15, 30)), out=None):
    meta = json.loads((OUT / f'{name}.json').read_text())
    tri, val, bnd, d = boundary_colour(name, case)
    fixed, loaded = set(d['fixed'].tolist()), set(d['loaded'].tolist())
    # Common scale for PLA parts: colour saturates at the PLA Basic Z tensile strength.
    vmax = 35.0 if meta['group'] != 'aluminium' else max(5.0, float(np.percentile(val, 99.8)))
    cmap = plt.get_cmap('turbo')
    fig = plt.figure(figsize=(15, 4.8), facecolor='#fcfcfb')
    for j, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, len(views) + 1, j + 1, projection='3d')
        colours = cmap(np.clip(val / vmax, 0, 1))
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colours, edgecolor='none'))
        lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(el, az); ax.set_axis_off()
        imax = meta['cases'][case]['argmax']
        cen = d['p'][:, d['t'][:, imax]].mean(axis=1)
        ax.scatter(*cen, color='k', s=40, marker='x')
    ax = fig.add_subplot(1, len(views) + 1, len(views) + 1, projection='3d')
    colours = np.array([[0.2, 0.4, 1.0, 1] if f in fixed else [1.0, 0.25, 0.2, 1] if f in loaded else [0.85, 0.85, 0.85, 0.35] for f in bnd])
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colours, edgecolor='none'))
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(*views[0]); ax.set_axis_off(); ax.set_title('blue: clamped (body side)\nred: loaded (foot side)', fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=fig.axes[:len(views)], shrink=0.6, label='max principal stress in skin [MPa]')
    fig.suptitle(f'{name}: {case}   (x = element with the highest skin stress)', fontsize=11)
    path = Path(out) if out else OUT / f'fig_{name}.png'
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path


if __name__ == '__main__':
    for name in sys.argv[1:]:
        meta = evaluate(name)
        for case, r in meta['cases'].items():
            print(f"{name:14s} {case:44s} s1max {r['s1_max']:7.2f} p99.5 {r['s1_p995']:6.2f} vm {r['vm_max']:7.2f} MPa  FoS {r['fos_max']:5.2f}/{r['fos_p995']:5.2f}  foot {np.round(r['foot_disp_mm'],3)} mm")
        print(figure(name, 'D2 severe p99.9 (Fz 380 N)'))
