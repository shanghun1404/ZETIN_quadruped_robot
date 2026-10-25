"""Section loads and bolt loads at the bolted joints of the FL leg load path (stance posture).

Rigid-flange estimate for n bolts on a circle of radius r:
  tension per bolt  = 2 M_b / (n r) + max(N, 0) / n     (N > 0 separates the joint)
  shear per bolt    = V / n + T / (n r)                  (friction and dowels ignored)
"""
import json
from pathlib import Path

import numpy as np

from analyze_parts import HIP, KNEE, R_SHANK, R_THIGH, foot_contact_world
from assembly import leg_parts
from post import CASES

# (name, link, centre in zero pose [mm], joint axis (unit, zero pose; points from body side to foot side),
#  n, bolt, PCD [mm], what holds the thread, source)
JOINTS = [
    ('J1 AK70-10 housing -> body shell end plate', 'base', (-100, 250, 75), (0, -1, 0), 8, 'M3', 83, 'motor thread; PLA plate 8 mm clamped', 'log §6, mesh PCD 83'),
    ('J2 AK70-10 output -> combine_1', 'base', (-100, 256, 75), (0, 1, 0), 6, 'M3', 25, 'motor thread; PLA boss clamped', 'log §1/§3'),
    ('J4 AK10-9 hip output -> fake_motor end wall', 'thigh', (-130.7, 315, 75), (-1, 0, 0), 6, 'M4', 24, 'motor thread; PLA end wall 9 mm clamped (+2 dowels)', 'log §3, mesh PCD 24'),
    ('J5 fake_motor rim -> fake_cover', 'thigh', (-195.7, 315, 75), (-1, 0, 0), 6, 'M3', 90, 'M3 heat-set inserts in PLA', 'log §3, mesh PCD 90'),
    ('J6 fake_cover -> com_pro (hub side)', 'thigh', (-200.7, 315, 75), (-1, 0, 0), 6, 'M4', 24, 'M4 heat-set inserts in PLA disk 5 mm', 'log §3, mesh PCD 24'),
    ('J9 com_pro (knee side) -> knee motor output', 'thigh', (-200.7, 315, -215), (1, 0, 0), 6, 'M4', 24, 'motor thread; PLA flange 7.5 mm clamped (+dowels)', 'log §3, mesh PCD 24'),
    ('J10 knee motor housing -> knee cup backplate', 'shank', (-190.5, 315, -215), (-1, 0, 0), 6, 'M4', 90, 'motor thread; PLA backplate 5 mm clamped', 'log §3, mesh PCD 90'),
]


def posed(link, x, axis):
    x, axis = np.asarray(x, float), np.asarray(axis, float)
    if link == 'thigh':
        return (x - HIP) @ R_THIGH.T + HIP, R_THIGH @ axis
    if link == 'shank':
        knee_w = (KNEE - HIP) @ R_THIGH.T + HIP
        return (x - KNEE) @ R_SHANK.T + knee_w, R_SHANK @ axis
    return x, axis


def section(F, foot, c, a):
    """Loads transmitted from the foot side to the body side across a plane at c with normal a."""
    F = np.asarray(F, float)
    M = np.cross(foot - c, F) / 1000.0  # N m
    N = F @ a                    # + : pulls the foot side away from the body side
    V = np.linalg.norm(F - N * a)
    T = M @ a
    Mb = np.linalg.norm(M - T * a)
    return N, V, T, Mb


def main():
    foot = foot_contact_world(leg_parts('FL'))
    rows = []
    for name, link, c0, a0, n, bolt, pcd, hold, src in JOINTS:
        c, a = posed(link, c0, a0)
        r = pcd / 2000.0  # m
        for case, forces in CASES.items():
            worst = None
            for F in forces:
                N, V, T, Mb = section(F, foot, c, a)
                tension = 2 * Mb / (n * r) + max(N, 0.0) / n
                shear = V / n + abs(T) / (n * r)
                row = dict(joint=name, case=case, force=list(F), N=N, V=V, T_Nm=T, Mb_Nm=Mb,
                           bolt_tension_N=tension, bolt_shear_N=shear, n=n, bolt=bolt, pcd=pcd, holds=hold, source=src)
                if worst is None or tension > worst['bolt_tension_N']:
                    worst = row
            rows.append(worst)
    out = Path(__file__).resolve().parent / 'results' / 'joints.json'
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r['case'].startswith(('D1', 'D2', 'D3')):
            print(f"{r['joint']:46s} {r['case'][:2]}  Mb {r['Mb_Nm']:6.1f} Nm  T {r['T_Nm']:6.1f} Nm  N {r['N']:7.1f} N  "
                  f"bolt tension {r['bolt_tension_N']:7.0f} N  shear {r['bolt_shear_N']:6.0f} N  ({r['n']}x{r['bolt']} PCD{r['pcd']})")


if __name__ == '__main__':
    main()
