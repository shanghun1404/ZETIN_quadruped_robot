"""Markdown tables for the report, generated from results/*.json."""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / 'results'
C = {'D1': 'D1 stance p99 (Fz 220 N)', 'D2': 'D2 severe p99.9 (Fz 380 N)',
     'D3': 'D3 severe + horizontal (Fz 380 N, Fxy 197 N)', 'D4': 'D4 impact peak (Fz 1045 N)'}
NAMES = {'com_pro_knee': '허벅지 클램프, 무릎 쪽 (com_pro)', 'combine_1': '엉덩이 롤 링 (combine_1)',
         'knee': '무릎 컵 (knee)', 'fake_motor': '허브 (fake_motor)', 'com_pro_hub': '허벅지 클램프, 허브 쪽 (com_pro)',
         'fake_cover': '허브 커버 (fake_cover)', 'body_shell_F': '앞 몸체 셸 (body_shell_F)',
         'profile_thigh': '허벅지 2020 알루미늄', 'profile_shank': '정강이 2020 알루미늄'}


def load(n):
    return json.loads((RES / f'{n}.json').read_text())


def verdict(s2, s3, strength):
    if s2 >= strength:
        return '**불합격** (평상 하중대에서 강도 초과)'
    if s3 >= strength:
        return '**주의** (수평력이 더해지면 초과)'
    if s3 >= strength / 2:
        return '여유 작음 (안전율 < 2)'
    return '양호'


def parts_table():
    rows = ['| 부품 | 스킨 응력 p99.5 [MPa] D1 / D2 / D3 | 최대값 D2 [MPa] | 안전율 D2 (p99.5) | 발끝 처짐 기여 D1 [mm] | 판정 |',
            '|---|---|---|---|---|---|']
    for n in NAMES:
        m = load(n); c = m['cases']
        strength = 276.0 if m['group'] == 'aluminium' else 35.0
        s = [c[C[k]]['s1_p995'] for k in ('D1', 'D2', 'D3')]
        foot = np.linalg.norm(c[C['D1']]['foot_disp_mm'])
        rows.append(f"| {NAMES[n]} | {s[0]:.1f} / {s[1]:.1f} / {s[2]:.1f} | {c[C['D2']]['s1_max']:.0f} | {strength / s[1]:.2f} | {foot:.1f} | {verdict(s[1], s[2], strength)} |")
    return '\n'.join(rows)


def mesh_table():
    rows = ['| 부품 | 사면체 | 자유도 | 고정면 [mm²] | 하중면 [mm²] | 스킨 체적비 | 계산 시간 [s] |', '|---|---|---|---|---|---|---|']
    for n in NAMES:
        m = load(n)
        rows.append(f"| {NAMES[n]} | {m['tets']:,} | {m['dofs']:,} | {m['fixed_area_mm2']:.0f} | {m['loaded_area_mm2']:.0f} | {m['mean_skin_fraction']:.2f} | {m['seconds']:.0f} |")
    return '\n'.join(rows)


VARIANT_KO = {'solid': '인필 100 % 출력', 'brace': '알루미늄 무릎 브래킷 (강체 연결 상한)', 'coarse': '메시 2.5배 거칠게 (민감도 확인)',
              'al': '알루미늄 6061 가공품으로 교체', 'al_endwall': '허브 끝벽을 알루미늄 판으로 (관과 일체 가정, 상한)'}


def variant_table():
    rows = ['| 부품 | 변경 | 스킨 응력 p99.5 D2 [MPa] | D3 [MPa] | 발끝 처짐 D1 [mm] | D2 안전율 |', '|---|---|---|---|---|---|']
    for r in json.loads((RES / 'compare.json').read_text()):
        b, v = r['base'], r['variant']
        key = v['name'].split('__')[1]
        strength = 276.0 if key == 'al' else 35.0
        rows.append(f"| {NAMES[b['name']]} | {VARIANT_KO[key]} | {b['sD2']:.1f} → **{v['sD2']:.1f}** ({v['sD2'] / b['sD2']:.2f}×) | {b['sD3']:.1f} → **{v['sD3']:.1f}** | {b['foot1']:.2f} → **{v['foot1']:.2f}** ({v['foot1'] / b['foot1']:.2f}×) | {35.0 / b['sD2']:.2f} → **{strength / v['sD2']:.2f}** |")
    return '\n'.join(rows)


JOINT_KO = {
    'J1': ('AK70-10 하우징 → 몸체 셸 끝판', '모터 나사, PLA 판 8 mm 체결'),
    'J2': ('AK70-10 출력 → 엉덩이 롤 링', '모터 나사, PLA 보스 체결'),
    'J4': ('엉덩이 AK10-9 출력 → 허브 끝벽', '모터 나사, PLA 끝벽 9 mm 체결 (+핀 2)'),
    'J5': ('허브 테두리 → 허브 커버', 'PLA 안의 M3 히트셋 인서트'),
    'J6': ('허브 커버 → 허브 쪽 클램프', 'PLA 원판(5 mm) 안의 M4 히트셋 인서트'),
    'J9': ('무릎 쪽 클램프 → 무릎 모터 출력', '모터 나사, PLA 플랜지 7.5 mm 체결 (+핀)'),
    'J10': ('무릎 모터 하우징 → 무릎 컵 백플레이트', '모터 나사, PLA 판 5 mm 체결'),
}


def joints_table():
    rows = ['| 체결부 | 볼트 | 나사가 물리는 곳 | D1 볼트당 인장/전단 [N] | D2 | D3 (수평력 포함) |', '|---|---|---|---|---|---|']
    js = json.loads((RES / 'joints.json').read_text())
    for name in dict.fromkeys(j['joint'] for j in js):
        rs = {j['case'][:2]: j for j in js if j['joint'] == name}
        j = rs['D1']
        tag = name.split()[0]
        rows.append(f"| {tag} {JOINT_KO[tag][0]} | {j['n']}×{j['bolt']} PCD {j['pcd']} | {JOINT_KO[tag][1]} | "
                    + ' | '.join(f"{rs[k]['bolt_tension_N']:.0f} / {rs[k]['bolt_shear_N']:.0f}" for k in ('D1', 'D2', 'D3')) + ' |')
    return '\n'.join(rows)


if __name__ == '__main__':
    import sys
    print({'parts': parts_table, 'mesh': mesh_table, 'variants': variant_table, 'joints': joints_table}[sys.argv[1]]())
