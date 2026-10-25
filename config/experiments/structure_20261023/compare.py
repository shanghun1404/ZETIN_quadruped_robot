"""Compare as-printed parts with print-setting and aluminium variants.

Run: PYTHONPATH=../fea_pkgs:. python3 compare.py
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / 'results'
CASES = {'D1': 'D1 stance p99 (Fz 220 N)', 'D2': 'D2 severe p99.9 (Fz 380 N)',
         'D3': 'D3 severe + horizontal (Fz 380 N, Fxy 197 N)'}
PAIRS = [('knee', 'knee__solid'), ('knee', 'knee__brace'), ('knee', 'knee__coarse'),
         ('com_pro_knee', 'com_pro_knee__solid'), ('com_pro_hub', 'com_pro_hub__solid'),
         ('fake_cover', 'fake_cover__solid'), ('fake_cover', 'fake_cover__al'),
         ('fake_motor', 'fake_motor__solid'), ('fake_motor', 'fake_motor__al_endwall'),
         ('combine_1', 'combine_1__solid'), ('combine_1', 'combine_1__al'), ('com_pro_knee', 'com_pro_knee__al')]


def row(name):
    m = json.loads((RES / f'{name}.json').read_text())
    c = m['cases']
    return dict(name=name, variant=m.get('variant'), tets=m['tets'],
                **{f's{k}': c[v]['s1_p995'] for k, v in CASES.items()},
                **{f'smax{k}': c[v]['s1_max'] for k, v in CASES.items()},
                foot1=float(np.linalg.norm(c[CASES['D1']]['foot_disp_mm'])),
                foot3=float(np.linalg.norm(c[CASES['D3']]['foot_disp_mm'])))


def main():
    out = []
    for base, var in PAIRS:
        if not (RES / f'{var}.json').exists():
            continue
        b, v = row(base), row(var)
        out.append(dict(base=b, variant=v,
                        stress_ratio_D2=v['sD2'] / b['sD2'], stress_ratio_D3=v['sD3'] / b['sD3'],
                        foot_ratio_D1=v['foot1'] / b['foot1']))
        print(f"{var:26s} p99.5 D2 {b['sD2']:6.1f} -> {v['sD2']:6.1f} MPa ({v['sD2'] / b['sD2']:.2f}x)  "
              f"D3 {b['sD3']:6.1f} -> {v['sD3']:6.1f} ({v['sD3'] / b['sD3']:.2f}x)  foot D1 {b['foot1']:6.2f} -> {v['foot1']:6.2f} mm ({v['foot1'] / b['foot1']:.2f}x)")
    (RES / 'compare.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
