"""Summary charts for the structural report (static PNG).

Run: PYTHONPATH=../fea_pkgs:. python3 charts.py <out_dir>
"""
import json
import sys
from pathlib import Path

sys.path = [p for p in sys.path if 'dist-packages' not in p]
sys.modules.pop('mpl_toolkits', None)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parent / 'results'
SERIES, TEXT, MUTED, SURFACE = '#2a78d6', '#0b0b0b', '#52514e', '#fcfcfb'
LABELS = {
    'knee': 'knee cup (knee)',
    'com_pro_knee': 'thigh clamp, knee side (com_pro)',
    'com_pro_hub': 'thigh clamp, hub side (com_pro)',
    'fake_cover': 'hub cover (fake_cover)',
    'fake_motor': 'hub (fake_motor)',
    'combine_1': 'hip roll ring (combine_1)',
    'body_shell_F': 'front body shell (body_shell_F)',
}
CASE = {'D1': 'D1 stance p99 (Fz 220 N)', 'D2': 'D2 severe p99.9 (Fz 380 N)',
        'D3': 'D3 severe + horizontal (Fz 380 N, Fxy 197 N)'}


def load(name):
    return json.loads((RES / f'{name}.json').read_text())


def hbar(ax, names, values, unit, ref=None, ref_label=None):
    order = np.argsort(values)
    y = np.arange(len(names))
    ax.barh(y, np.asarray(values)[order], color=SERIES, height=0.6)
    ax.set_yticks(y, [LABELS[names[i]] for i in order], color=TEXT, fontsize=9)
    for yi, v in zip(y, np.asarray(values)[order]):
        ax.annotate(f'{v:.1f}', (v, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=9, color=TEXT)
    if ref is not None:
        ax.axvline(ref, color=MUTED, ls='--', lw=1.2)
        ax.text(ref, -0.85, f' {ref_label}', color=MUTED, fontsize=8, va='center')
        ax.set_ylim(-1.2, len(names) - 0.5)
    ax.set_xlabel(unit, color=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='x', color='#e6e5e1', lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(LABELS)
    for key, title, fname in (('D2', 'Skin stress at 380 N vertical foot load (p99.9 of the gait suite)', 'stress_D2.png'),
                              ('D3', 'Skin stress at 380 N vertical + 197 N horizontal foot load', 'stress_D3.png')):
        vals = [load(n)['cases'][CASE[key]]['s1_p995'] for n in names]
        fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE)
        hbar(ax, names, vals, 'max principal stress in the printed skin, 99.5th percentile by volume [MPa]',
             ref=35.0, ref_label='PLA Basic Z tensile strength 35 MPa')
        ax.set_title(title, color=TEXT, fontsize=10, loc='left', pad=10)
        fig.tight_layout(); fig.savefig(out / fname, dpi=130); plt.close(fig)
    vals = [float(np.linalg.norm(load(n)['cases'][CASE['D1']]['foot_disp_mm'])) for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE)
    hbar(ax, names, vals, 'foot displacement caused by this part alone [mm] (linear, upper bound)')
    ax.set_title('Foot deflection contribution at 220 N vertical foot load (stance p99)', color=TEXT, fontsize=10, loc='left')
    fig.tight_layout(); fig.savefig(out / 'compliance_D1.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else RES / 'charts')
