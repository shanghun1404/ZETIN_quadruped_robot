"""Build the V6 metal-leg (al7075, tc4) walking and motor-RMS report from archived ROS2/Isaac suite runs.

usage: python3 -I build_report.py
suite_jobs/<job>/ and train_jobs/<job>/ are copies of the ws1 job folders (v15_cad_20261011/v6_material_*_jobs).
Each variant runs two policies: the V6 fine-tuned model_16099 (train job 10806) and its torque-optimization
fine-tune on that variant (round2 recipe, 400 updates). The V6 baseline is read from ../cad_v6_20261025.
Writes report.html, summary.json, motor_rms.csv and <case>/telemetry.png next to each archived result.
Controller split: Kp=0 steps are the stand/sit WBC (feedforward torque), Kp=300 steps the walking policy, the rest blend both.
"""
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'usd_20261011'))
from build_finetune_report import check_source  # noqa: E402
from build_suite_report import LABELS, NAMES, check_trace, controller_failure, plot  # noqa: E402

CASES = [name for name in LABELS if not name.startswith('stairs_')]
BASE_SHA256 = 'c87d6b1d797a2df76f22d610cb25bb33c33c6b8ef89a6394acbcc226ead885d5'
V6 = dict(label='v6 기본 (알루미늄 다리)', usd='quadruped_v6_20261025', jobs=['10809', '10810'], root=HERE.parent / 'cad_v6_20261025')
VARIANTS = {
    'al7075': dict(label='AL7075 다리', usd_sha256='ffbe907f091544d96ad33125bf949d3443f51885265a37101cd78743e39bae2b',
                   mass_kg=29.5759640, base_jobs=['10837', '10838'], train='11010', tuned_jobs=['11013', '11014']),
    'tc4': dict(label='TC4(Ti-6Al-4V) 다리', usd_sha256='780d946745daeb272cadf880ed31bbd57367c8deb9d1a220749e4e30d4765645',
                mass_kg=33.6362480, base_jobs=['10839', '10840'], train='11009', tuned_jobs=['11011', '11012']),
}
GROUPS = {'HAA': [0, 3, 6, 9], 'HFE': [1, 4, 7, 10], 'KFE': [2, 5, 8, 11]}
JOINT_KO = dict(HAA='고관절 roll', HFE='고관절 pitch', KFE='무릎')
REGIMES = dict(wbc='기립·앉기 WBC (Kp=0)', blend='전환', policy='보행 정책 (Kp=300)')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def foot_loads(folders):
    fz, peaks = [], []
    for name, folder in folders.items():
        force = np.load(folder / 'telemetry.npz')['shank_support_force'][..., 2]
        fz.append(force.ravel())
        t, foot = np.unravel_index(force.argmax(), force.shape)
        peaks.append(dict(case=name, newton=float(force.max()), time_s=round(t * .005, 3), foot=['FL', 'FR', 'RL', 'RR'][foot]))
    fz = np.concatenate(fz)
    contact = fz[fz > 5.]
    peaks.sort(key=lambda p: -p['newton'])
    return dict(p99=float(np.percentile(fz, 99)), p999=float(np.percentile(fz, 99.9)), peak=float(fz.max()),
                contact_p999=float(np.percentile(contact, 99.9)), samples=int(fz.size), top_peaks=peaks[:3])


def regime_rms(folder, rated):
    """Knee torque RMS / rated within each controller regime (telemetry RMS equals result.json RMS over the whole case)."""
    raw = np.load(folder / 'telemetry.npz')
    kp, tau = raw['kp'], raw['estimated_torque'][:, GROUPS['KFE']]
    masks = dict(wbc=kp == 0., policy=kp >= 300. - 1e-6)
    masks['blend'] = ~(masks['wbc'] | masks['policy'])
    out = {}
    for key, mask in masks.items():
        if mask.any():
            ratio = np.sqrt((tau[mask] ** 2).mean(0)) / np.asarray(rated)[GROUPS['KFE']]
            out[key] = dict(seconds=float(mask.sum() * .005), knee_ratio_max=float(ratio.max()),
                            joint=NAMES[GROUPS['KFE'][int(ratio.argmax())]])
    return out


def self_contact(folder, result):
    """Non-adjacent body contact over 1 N: pairs, peak force, time span and controller regime of the contact samples."""
    raw = np.load(folder / 'telemetry.npz')
    hit = raw['nonadjacent_pair_force_n'].max(1) > 1.
    t = np.flatnonzero(hit) * .005
    regimes = sorted({'WBC' if k == 0. else '보행' if k >= 300. - 1e-6 else '전환' for k in raw['kp'][hit]})
    pairs = sorted({' ↔ '.join(sorted(pair.split(' <-> '))) for pair in result['contact_pairs_over_1n']})
    return (f'자가 접촉 >1 N: {", ".join(pairs)} 최대 {result["max_nonadjacent_contact_force_n"]:.0f} N, '
            f'{t[0]:.1f}–{t[-1]:.1f} s, {"/".join(regimes)} 구간')


def policy_record(label, root, jobs, actor_sha256, usd, usd_sha256=None):
    results, folders, traces, sources = {}, {}, {}, {}
    for job in jobs:
        run = root / 'suite_jobs' / job
        sources[job] = check_source(run)
        assert (run / 'checkpoints.sha256').read_text().split()[0] == actor_sha256, job
        if usd_sha256:
            assert sha256(run / f'source/assets/{usd}/quadruped_robot.usda') == usd_sha256, job
        for result in json.loads((run / 'results/suite.json').read_text())['cases']:
            name = result['case']
            assert name in CASES and name not in results and result['seed'] == 42
            folder = run / 'results' / name
            results[name], folders[name], traces[name] = result, folder, check_trace(folder, result)
            plot(folder)
    assert set(results) == set(CASES), sorted(results)
    assert all(record['passed'] for record in sources.values()), sources
    assert all(record['pass_check'] for record in traces.values()), traces
    ratios = {n: (np.asarray(results[n]['torque_rms_nm']) / np.asarray(results[n]['rated_torque_nm'])).tolist() for n in CASES}
    worst = {}
    for group, index in GROUPS.items():
        name, joint = max(((n, i) for n in CASES for i in index), key=lambda item: ratios[item[0]][item[1]])
        worst[group] = dict(ratio=ratios[name][joint], case=name, joint=NAMES[joint], rms_nm=results[name]['torque_rms_nm'][joint],
                            rated_nm=results[name]['rated_torque_nm'][joint])
    regimes = {n: regime_rms(folders[n], results[n]['rated_torque_nm']) for n in CASES}
    regime_worst = {key: max((r[key]['knee_ratio_max'], n) for n, r in regimes.items() if key in r)
                    for key in REGIMES if any(key in r for r in regimes.values())}
    metrics = {n: results[n]['motion_verdict'].get('metrics') or {} for n in CASES}
    return dict(label=label, checkpoint_sha256=actor_sha256, jobs=jobs, results=results, folders=folders,
                mass_kg=results['flat_14s']['metadata']['actual_mass_kg'],
                passed=[n for n in CASES if results[n]['task_success']],
                rated_exceed={n: [NAMES[i] for i, flag in enumerate(results[n]['rms_exceeds_rated']) if flag] for n in CASES},
                datasheet_peak_exceeded=[n for n in CASES if results[n]['motor_datasheet_peak_exceeded']],
                ratios=ratios, worst=worst, regimes=regimes, regime_worst=regime_worst, loads=foot_loads(folders),
                yaw_error_rad=metrics['yaw'].get('yaw_error_rad'), rough_max_abs_x_m=metrics['rough'].get('max_abs_x_m'),
                source_integrity=sources, telemetry_integrity=traces)


def training_record(job, usd_sha256):
    training = HERE / 'train_jobs' / job
    completed = json.loads((training / 'results/completed.json').read_text())
    contract = json.loads((training / 'results/training_contract.json').read_text())
    checkpoint = training / 'results' / Path(completed['checkpoint']).name
    assert completed['training_completed'] and contract['usd_sha256'] == usd_sha256
    assert sha256(checkpoint) == completed['checkpoint_sha256']
    assert contract['actor_sha256'] == contract['retention_teacher_sha256'] == BASE_SHA256
    source = check_source(training)
    assert source['passed'], source
    return dict(job=job, checkpoint=checkpoint.relative_to(HERE).as_posix(), completed=completed, source_integrity=source,
                recipe={k: contract.get(k) for k in ('profile', 'terrain', 'num_envs', 'training_seed', 'learning_rate', 'retention_weight',
                                                     'iterations_requested', 'starting_iteration', 'actor_sha256', 'usd_sha256',
                                                     'mass_min_max_kg', 'collider_count')})


def cell(ratio):
    style = ' style="background:#f8d7da"' if ratio > 1 else ' style="background:#fff3cd"' if ratio >= .9 else ''
    return f'<td{style}>{ratio:.2f}</td>'


def ok(record):
    return len(record['passed']) == 9, not any(record['rated_exceed'].values()) and not record['datasheet_peak_exceeded']


def main():
    v6 = policy_record(V6['label'] + ' · model_16099', V6['root'], V6['jobs'], BASE_SHA256, V6['usd'])
    records, trainings = {}, {}
    for key, spec in VARIANTS.items():
        usd = f'quadruped_v6_{key}_20261025'
        trainings[key] = training_record(spec['train'], spec['usd_sha256'])
        records[key] = dict(
            base=policy_record(f'{spec["label"]} · model_16099', HERE, spec['base_jobs'], BASE_SHA256, usd, spec['usd_sha256']),
            tuned=policy_record(f'{spec["label"]} · 토크 최적화 {Path(trainings[key]["checkpoint"]).stem}', HERE, spec['tuned_jobs'],
                                trainings[key]['completed']['checkpoint_sha256'], usd, spec['usd_sha256']))
        for record in records[key].values():
            assert abs(record['mass_kg'] - spec['mass_kg']) < 1e-3, (key, record['mass_kg'])

    with (HERE / 'motor_rms.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['variant', 'policy', 'case', 'joint', 'rated_torque_nm', 'torque_rms_nm', 'rms_over_rated', 'torque_peak_nm',
                         'ideal_current_rms_a', 'rms_exceeds_rated', 'seconds_recorded'])
        for key, pair in records.items():
            for which, record in pair.items():
                for name in CASES:
                    r = record['results'][name]
                    for i, joint in enumerate(NAMES):
                        writer.writerow([key, which, name, joint, r['rated_torque_nm'][i], r['torque_rms_nm'][i], record['ratios'][name][i],
                                         r['torque_peak_nm'][i], r['ideal_current_rms_a'][i], r['rms_exceeds_rated'][i], r['seconds_recorded']])

    strip = lambda record: {k: v for k, v in record.items() if k != 'folders'}
    verdict = {key: {which: dict(zip(('all_cases_walk', 'all_rms_within_rated'), ok(record))) for which, record in pair.items()}
               for key, pair in records.items()}
    summary = dict(schema=1, robot='V6 round-body CAD 2026-10-25, metal-leg mass variants (only the 12 leg links differ)',
                   evaluator='round2 suite job 2343 snapshot (evaluate_v15_cad_suite.py, ROS2 controller), seed 42',
                   verdict=verdict, trainings=trainings, v6_reference=strip(v6),
                   variants={key: {which: strip(record) for which, record in pair.items()} for key, pair in records.items()},
                   adoption='Simulation result only; the Jetson bundle keeps model_15699',
                   controller_split='Kp=0 steps: stand/sit WBC torque (MuJoCo model of the 2026-10-11 CAD); Kp=300: walking policy')
    write(HERE / 'summary.json', summary)

    def pass_text(record, name):
        r = record['results'][name]
        reasons = list(r['motion_verdict'].get('reasons') or []) + ([r['failure']] if r.get('failure') else [])
        reasons += [self_contact(record['folders'][name], r)] if r.get('collision_checks_pass') is False else []
        error = controller_failure(record['folders'][name])
        reasons += [error] if error else []
        return ('통과' if r['task_success'] else '미통과') + (f' ({html.escape("; ".join(reasons))})' if reasons else '')

    ordered = [v6] + [record for pair in records.values() for record in pair.values()]
    overview = ''
    for record in ordered:
        walk, rms = ok(record)
        rw = record['regime_worst']
        split = ' / '.join(f'{rw[k][0]:.2f}' if k in rw else '-' for k in ('policy', 'blend', 'wbc'))
        overview += (f'<tr><td>{html.escape(record["label"])}</td><td>{record["mass_kg"]:.2f}</td><td>{len(record["passed"])}/9</td>'
                     f'{cell(record["worst"]["KFE"]["ratio"])}<td>{record["worst"]["HAA"]["ratio"]:.2f}</td><td>{split}</td>'
                     f'<td>{sum(bool(v) for v in record["rated_exceed"].values())}</td><td>{len(record["datasheet_peak_exceeded"])}</td>'
                     f'<td>{record["yaw_error_rad"]:.3f}</td><td>{record["rough_max_abs_x_m"]:.3f}</td>'
                     f'<td>{record["loads"]["p999"]:.1f}</td><td>{record["loads"]["peak"]:.0f}</td></tr>')
    motor_tables = ''
    for record in ordered[1:]:
        head = ''.join(f'<th>{j}</th>' for j in NAMES)
        body = ''.join(f'<tr><td>{LABELS[n]}</td>' + ''.join(cell(x) for x in record['ratios'][n]) + '</tr>' for n in CASES)
        worst = ' · '.join(f'{JOINT_KO[g]} {w["ratio"]:.2f} ({LABELS[w["case"]]} {w["joint"]}, {w["rms_nm"]:.1f}/{w["rated_nm"]:.1f} N·m)'
                           for g, w in record['worst'].items())
        regime_rows = ''.join(
            f'<tr><td>{LABELS[n]}</td>' + ''.join(
                cell(record['regimes'][n][k]['knee_ratio_max']) if k in record['regimes'][n] else '<td>-</td>' for k in ('policy', 'blend', 'wbc'))
            + '<td>' + ' / '.join(f'{record["regimes"][n][k]["seconds"]:.1f}' if k in record['regimes'][n] else '0' for k in ('policy', 'blend', 'wbc'))
            + '</td></tr>' for n in CASES)
        motor_tables += (f'<h3>{html.escape(record["label"])}</h3><p>관절 종류별 최대 RMS/정격: {worst}</p>'
                         f'<div class="wide"><table><tr><th>시험</th>{head}</tr>{body}</table></div>'
                         f'<p>무릎 RMS/정격 최대, 제어 구간별:</p><div class="wide"><table><tr><th>시험</th><th>{REGIMES["policy"]}</th>'
                         f'<th>{REGIMES["blend"]}</th><th>{REGIMES["wbc"]}</th><th>구간 길이 s (보행/전환/WBC)</th></tr>{regime_rows}</table></div>')
    train_rows = ''.join(
        f'<li>{VARIANTS[key]["label"]}: job {t["job"]}, {t["recipe"]["starting_iteration"]}→{t["completed"]["last_iteration"]} '
        f'({t["recipe"]["iterations_requested"]}회, {t["recipe"]["profile"]}, {t["recipe"]["num_envs"]}환경, lr {t["recipe"]["learning_rate"]}, '
        f'보존 가중치 {t["recipe"]["retention_weight"]}), <a href="{t["checkpoint"]}">체크포인트</a> SHA {t["completed"]["checkpoint_sha256"][:8]}…</li>'
        for key, t in trainings.items())
    sections = ''
    for n in CASES:
        columns = ''
        for record in ordered[1:]:
            rel = record['folders'][n].relative_to(HERE).as_posix()
            columns += (f'<div><h3>{html.escape(record["label"])} — {pass_text(record, n)}</h3>'
                        f'<video controls preload="none" src="{rel}/video/env_00.mp4"></video>'
                        f'<p><a href="{rel}/result.json">원본 결과</a> · <a href="{rel}/telemetry.npz">시계열</a></p>'
                        f'<img loading="lazy" src="{rel}/telemetry.png" alt="{LABELS[n]} 시계열"></div>')
        sections += f'<section><h2>{LABELS[n]}</h2><div class="pair">{columns}</div></section>'
    conclusion = ' '.join(
        f'{VARIANTS[key]["label"]}: model_16099 {len(pair["base"]["passed"])}/9 통과, 무릎 RMS/정격 최대 {pair["base"]["worst"]["KFE"]["ratio"]:.2f} → '
        f'토크 최적화 후 {len(pair["tuned"]["passed"])}/9 통과, {pair["tuned"]["worst"]["KFE"]["ratio"]:.2f}.'
        for key, pair in records.items())
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6 금속 다리 보행</title>
<style>body{{font:16px/1.6 system-ui;max-width:1280px;margin:30px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:6px}}th{{background:#edf3fa}}video,img{{max-width:100%;display:block;margin:10px 0}}video{{width:100%;background:#000}}
section{{border-top:1px solid #ddd}}.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
.wide{{overflow-x:auto}}h3{{font-size:16px}}</style>
<h1>V6 금속 다리 변형 · ROS2 / Isaac Sim 보행·모터 RMS 검증</h1>
<p><b>결과</b>: {conclusion} 모든 결과는 시뮬레이션이며 Jetson 번들 모델은 model_15699를 유지한다.</p>
<p>로봇: v6 CAD(2026-10-25)에서 다리 12개 링크의 질량·무게중심·관성만 바꾼 변형
(AL7075 {VARIANTS["al7075"]["mass_kg"]:.2f} kg, TC4 {VARIANTS["tc4"]["mass_kg"]:.2f} kg; v6 기본 {v6["mass_kg"]:.2f} kg).
시험: round2 suite job 2343 소스 그대로(ROS2 제어기 포함), seed 42, 각 조건 1회 실행.</p>
<p>토크 최적화 학습(round2 방식, 시작·보존 기준 모두 v6 model_16099):</p><ul>{train_rows}</ul>
<p><a href="summary.json">전체 결과</a> · <a href="motor_rms.csv">모터·시험별 RMS</a> · <a href="build_report.py">생성 스크립트</a> ·
<a href="../cad_v6_20261025/report.html">v6 기본 보고서</a></p>
<h2>요약</h2><div class="wide"><table><tr><th>로봇 · 정책</th><th>질량 kg</th><th>통과</th><th>무릎 RMS/정격 최대</th><th>고관절 roll RMS/정격 최대</th>
<th>무릎 RMS/정격 최대 (보행 / 전환 / WBC)</th><th>정격 초과 시험 수</th><th>사양 최대 토크 초과</th><th>회전 오차 rad (≤0.6)</th>
<th>요철 옆 이동 m (≤0.70)</th><th>발 Fz p99.9 N</th><th>발 Fz 최대 N</th></tr>{overview}</table></div>
<p>정격: 고관절 roll(HAA) 8.3 N·m, 고관절 pitch(HFE)·무릎(KFE) 18 N·m. RMS는 각 시험의 전체 기록구간 값이다.
0.9 이상은 노란색, 1 초과는 빨간색이다. 제어 구간은 기록된 PD 게인으로 나눈다: Kp=0은 기립·앉기 WBC가 토크를 직접 내는 구간,
Kp=300은 보행 정책 구간, 그 사이는 두 제어기를 섞는 전환 구간이다. 토크 최적화 학습은 보행 정책만 바꾸므로 WBC 구간 토크는 바뀌지 않는다.</p>
<h2>모터별 토크 RMS / 정격</h2>{motor_tables}
<h2>주의</h2><ul><li>각 조건을 한 번씩만 실행했다. 요철 지형은 프로토콜의 고정 seed(42)로 만들어지며 <code>--seed</code>로 바뀌지 않는다.</li>
<li>기립·앉기 WBC는 2026-10-11 CAD 기준 MuJoCo 모델(23.1 kg)을 그대로 쓴다. 금속 변형의 실제 질량과 다르다.</li>
<li>발 하중은 shank_support_force의 수직 성분(접촉 수직력만)이고, 최대값은 착지 순간 한 샘플(5 ms) 값이다.</li>
<li>전류는 효율 1을 가정한 추정 상전류이며 배터리 전류나 실측값이 아니다.</li>
<li>토크 최적화 학습 job 11009·11010은 대기 중 <code>scontrol update MinMemoryNode=20480</code>으로 메모리 요청을 20G로 낮춰 실행했다(sacct MaxRSS 18089204 KiB·19846816 KiB, 약 17.3·18.9 GiB). 복사된 launcher.sbatch에는 원래 요청값 24G가 남아 있다.</li></ul>
{sections}</html>'''
    (HERE / 'report.html').write_text(content)
    print(json.dumps(verdict))


if __name__ == '__main__':
    main()
