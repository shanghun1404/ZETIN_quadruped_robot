"""Build the V6 round-body CAD walking report from archived ROS2/Isaac suite runs of two policies.

usage: python3 -I build_report.py [--compare "label=results_dir[,results_dir]" ...]
suite_jobs/<job>/ and train_jobs/10806/ are copies of the ws1 job folders (v15_cad_20261011/v6_*_jobs).
Writes report.html, summary.json, motor_rms.csv and <case>/telemetry.png next to each archived result.
Foot loads: Fz of shank_support_force (contact normal only, no friction), all samples of all feet and cases
(swing zeros included, the method that reproduces the 220/380/1045 N design loads) plus contact-only (Fz > 5 N).
"""
import argparse
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
USD_SHA256 = 'ab30c2dc43f8427304ca86b74d2a774d12eec92997a1ad5a1886ce19f93ba5f8'
POLICIES = {
    'model_15699': dict(label='원래 모델 model_15699', jobs=['10813', '10808'],
                        sha256='9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6'),
    'model_16099_v6': dict(label='v6 추가 학습 model_16099', jobs=['10809', '10810'],
                           sha256='c87d6b1d797a2df76f22d610cb25bb33c33c6b8ef89a6394acbcc226ead885d5'),
}
GROUPS = {'HAA': [0, 3, 6, 9], 'HFE': [1, 4, 7, 10], 'KFE': [2, 5, 8, 11]}
JOINT_KO = dict(HAA='고관절 roll', HFE='고관절 pitch', KFE='무릎')


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
        peaks.append(dict(case=name, newton=float(force.max()), time_s=round(t * .005, 3), foot=['FL', 'FR', 'RL', 'RR'][foot],
                          samples_over_800n=int((force > 800).sum()), samples_over_500n=int((force > 500).sum())))
    fz = np.concatenate(fz)
    contact = fz[fz > 5.]
    peaks.sort(key=lambda p: -p['newton'])
    return dict(p99=float(np.percentile(fz, 99)), p999=float(np.percentile(fz, 99.9)), peak=float(fz.max()),
                contact_p99=float(np.percentile(contact, 99)), contact_p999=float(np.percentile(contact, 99.9)),
                samples=int(fz.size), top_peaks=peaks[:3])


def saturation(folders):
    tau, cap = [], []
    for folder in folders.values():
        raw = np.load(folder / 'telemetry.npz')
        tau.append(np.abs(raw['estimated_torque'])); cap.append(raw['drive_limits'])
    tau, cap = np.concatenate(tau), np.concatenate(cap)
    return {g: float((tau[:, i] >= .99 * cap[:, i] - 1e-3).mean()) for g, i in GROUPS.items()}


def policy_record(key, spec):
    results, folders, traces, sources = {}, {}, {}, {}
    for job in spec['jobs']:
        run = HERE / 'suite_jobs' / job
        sources[job] = check_source(run)
        hashes = (run / 'checkpoints.sha256').read_text().split()
        assert hashes[0] == spec['sha256'], (job, hashes[0])
        assert sha256(run / 'source/assets/quadruped_v6_20261025/quadruped_robot.usda') == USD_SHA256
        for result in json.loads((run / 'results/suite.json').read_text())['cases']:
            name = result['case']
            assert name in CASES and name not in results and result['seed'] == 42
            folder = run / 'results' / name
            results[name], folders[name], traces[name] = result, folder, check_trace(folder, result)
            plot(folder)
    assert set(results) == set(CASES), sorted(results)
    assert all(record['passed'] for record in sources.values()), sources
    assert all(record['pass_check'] for record in traces.values()), traces
    ratios = {name: (np.asarray(results[name]['torque_rms_nm']) / np.asarray(results[name]['rated_torque_nm'])).tolist() for name in CASES}
    worst = {}
    for group, index in GROUPS.items():
        name, joint = max(((n, i) for n in CASES for i in index), key=lambda item: ratios[item[0]][item[1]])
        worst[group] = dict(ratio=ratios[name][joint], case=name, joint=NAMES[joint], rms_nm=results[name]['torque_rms_nm'][joint],
                            rated_nm=results[name]['rated_torque_nm'][joint], peak_nm=max(results[n]['torque_peak_nm'][i] for n in CASES for i in index))
    metrics = {name: results[name]['motion_verdict'].get('metrics', {}) for name in CASES}
    return dict(label=spec['label'], checkpoint_sha256=spec['sha256'], jobs=spec['jobs'], results=results, folders=folders,
                passed=[n for n in CASES if results[n]['task_success']],
                rated_exceed={n: [NAMES[i] for i, flag in enumerate(results[n]['rms_exceeds_rated']) if flag] for n in CASES},
                datasheet_peak_exceeded=[n for n in CASES if results[n]['motor_datasheet_peak_exceeded']],
                ratios=ratios, worst=worst, loads=foot_loads(folders), saturation=saturation(folders),
                yaw_error_rad=metrics['yaw'].get('yaw_error_rad'), rough_max_abs_x_m=metrics['rough'].get('max_abs_x_m'),
                source_integrity=sources, telemetry_integrity=traces)


def compare_record(spec):
    label, dirs = spec.split('=', 1)
    folders, results = {}, {}
    for root in map(Path, dirs.split(',')):
        for path in sorted(root.glob('*/result.json')):
            folders[path.parent.name], results[path.parent.name] = path.parent, json.loads(path.read_text())
    knee = max(float(np.max(np.asarray(r['torque_rms_nm'])[GROUPS['KFE']] / np.asarray(r['rated_torque_nm'])[GROUPS['KFE']])) for r in results.values())
    metric = lambda case, key: (results[case]['motion_verdict'].get('metrics') or {}).get(key) if case in results else None
    return dict(label=label, sources=dirs.split(','), passed=sum(bool(r['task_success']) for r in results.values()), cases=len(results),
                failed=[n for n, r in results.items() if not r['task_success']], knee_rms_ratio_max=knee,
                yaw_error_rad=metric('yaw', 'yaw_error_rad'), rough_max_abs_x_m=metric('rough', 'max_abs_x_m'),
                loads=foot_loads(folders), mass_kg=next(iter(results.values()))['metadata']['actual_mass_kg'])


def cell(ratio):
    style = ' style="background:#f8d7da"' if ratio > 1 else ' style="background:#fff3cd"' if ratio >= .9 else ''
    return f'<td{style}>{ratio:.2f}</td>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='append', default=[])
    args = parser.parse_args()
    training = HERE / 'train_jobs/10806'
    completed = json.loads((training / 'results/completed.json').read_text())
    contract = json.loads((training / 'results/training_contract.json').read_text())
    assert completed['training_completed'] and contract['usd_sha256'] == USD_SHA256
    assert sha256(training / 'results/model_16099.pt') == completed['checkpoint_sha256'] == POLICIES['model_16099_v6']['sha256']
    assert contract['actor_sha256'] == contract['retention_teacher_sha256'] == POLICIES['model_15699']['sha256']
    training_source = check_source(training)
    assert training_source['passed'], training_source
    records = {key: policy_record(key, spec) for key, spec in POLICIES.items()}
    comparisons = [compare_record(spec) for spec in args.compare]

    with (HERE / 'motor_rms.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['policy', 'case', 'joint', 'rated_torque_nm', 'torque_rms_nm', 'rms_over_rated', 'torque_peak_nm',
                         'ideal_current_rms_a', 'rms_exceeds_rated', 'seconds_recorded'])
        for key, record in records.items():
            for name in CASES:
                r = record['results'][name]
                for i, joint in enumerate(NAMES):
                    writer.writerow([key, name, joint, r['rated_torque_nm'][i], r['torque_rms_nm'][i], record['ratios'][name][i],
                                     r['torque_peak_nm'][i], r['ideal_current_rms_a'][i], r['rms_exceeds_rated'][i], r['seconds_recorded']])

    walk_ok = all(len(record['passed']) == 9 for record in records.values())
    rms_ok = all(not any(record['rated_exceed'].values()) and not record['datasheet_peak_exceeded'] for record in records.values())
    summary = dict(schema=1, robot='V6 round-body CAD 2026-10-25', usd_sha256=USD_SHA256,
                   mass_kg=records['model_15699']['results']['flat_14s']['metadata']['actual_mass_kg'],
                   evaluator='round2 suite job 2343 snapshot (evaluate_v15_cad_suite.py, ROS2 controller), seed 42',
                   all_cases_walk=walk_ok, all_rms_within_rated=rms_ok,
                   next_step='videos and report' if walk_ok and rms_ok else 'torque optimization' if walk_ok else 'walking failure',
                   training_job_10806=dict(completed=completed, recipe={k: contract.get(k) for k in (
                       'profile', 'terrain', 'num_envs', 'training_seed', 'learning_rate', 'retention_weight', 'iterations_requested',
                       'starting_iteration', 'actor_sha256', 'usd_sha256', 'mass_min_max_kg', 'collider_count')}, source_integrity=training_source),
                   policies={key: {k: v for k, v in record.items() if k not in ('folders',)} for key, record in records.items()},
                   comparisons=comparisons,
                   adoption='Simulation result only; the Jetson bundle keeps model_15699',
                   load_semantics='shank_support_force is the contact normal force only; peaks are single 5 ms touchdown samples')
    write(HERE / 'summary.json', summary)

    def pass_text(record, name):
        r = record['results'][name]
        reasons = list(r['motion_verdict'].get('reasons') or []) + ([r['failure']] if r.get('failure') else [])
        error = controller_failure(record['folders'][name])
        reasons += [error] if error else []
        return ('통과' if r['task_success'] else '미통과') + (f' ({html.escape("; ".join(reasons))})' if reasons else '')

    a, b = records['model_15699'], records['model_16099_v6']
    case_rows = ''.join(
        f'<tr><td>{LABELS[n]}</td><td>{pass_text(a, n)}</td><td>{pass_text(b, n)}</td>'
        f'<td>{max(a["ratios"][n]):.2f} / {max(b["ratios"][n]):.2f}</td><td>{a["results"][n]["seconds_recorded"]:.1f}</td>'
        f'<td>{a["results"][n]["max_nonadjacent_contact_force_n"]:.1f} / {b["results"][n]["max_nonadjacent_contact_force_n"]:.1f}</td></tr>'
        for n in CASES)
    motor_tables = ''
    for record in records.values():
        head = ''.join(f'<th>{j}</th>' for j in NAMES)
        body = ''.join(f'<tr><td>{LABELS[n]}</td>' + ''.join(cell(x) for x in record['ratios'][n]) + '</tr>' for n in CASES)
        worst = ' · '.join(f'{JOINT_KO[g]} {w["ratio"]:.2f} ({LABELS[w["case"]]} {w["joint"]}, {w["rms_nm"]:.1f}/{w["rated_nm"]:.1f} N·m)'
                           for g, w in record['worst'].items())
        motor_tables += (f'<h3>{record["label"]}</h3><p>관절 종류별 최대 RMS/정격: {worst}</p>'
                         f'<div class="wide"><table><tr><th>시험</th>{head}</tr>{body}</table></div>')
    load_rows = ''
    for c in comparisons:
        l = c['loads']
        load_rows += (f'<tr><td>{html.escape(c["label"])}</td><td>{c["mass_kg"]:.2f}</td><td>{c["passed"]}/{c["cases"]}</td>'
                      f'<td>{l["p99"]:.1f}</td><td>{l["p999"]:.1f}</td><td>{l["peak"]:.0f}</td><td>{l["contact_p999"]:.1f}</td>'
                      f'<td>{c["knee_rms_ratio_max"]:.2f}</td><td>{c["yaw_error_rad"]:.3f}</td><td>{c["rough_max_abs_x_m"]:.3f}</td></tr>')
    for record in records.values():
        l = record['loads']
        load_rows += (f'<tr><td><b>v6 · {record["label"]}</b></td><td>{summary["mass_kg"]:.2f}</td><td>{len(record["passed"])}/9</td>'
                      f'<td>{l["p99"]:.1f}</td><td>{l["p999"]:.1f}</td><td>{l["peak"]:.0f}</td><td>{l["contact_p999"]:.1f}</td>'
                      f'<td>{record["worst"]["KFE"]["ratio"]:.2f}</td><td>{record["yaw_error_rad"]:.3f}</td><td>{record["rough_max_abs_x_m"]:.3f}</td></tr>')
    peak_rows = ''.join(f'<li>{record["label"]}: ' + ', '.join(
        f'{p["newton"]:.0f} N ({LABELS[p["case"]]}, {p["foot"]}, {p["time_s"]:.2f} s, 800 N 초과 {p["samples_over_800n"]}샘플)'
        for p in record['loads']['top_peaks']) + '</li>' for record in records.values())
    sections = ''
    for n in CASES:
        columns = ''
        for record in records.values():
            rel = record['folders'][n].relative_to(HERE).as_posix()
            columns += (f'<div><h3>{record["label"]} — {pass_text(record, n)}</h3>'
                        f'<video controls preload="none" src="{rel}/video/env_00.mp4"></video>'
                        f'<p><a href="{rel}/result.json">원본 결과</a> · <a href="{rel}/telemetry.npz">시계열</a></p>'
                        f'<img loading="lazy" src="{rel}/telemetry.png" alt="{LABELS[n]} 시계열"></div>')
        sections += f'<section><h2>{LABELS[n]}</h2><div class="pair">{columns}</div></section>'
    sat = ' · '.join(f'{record["label"]} {record["saturation"]["KFE"] * 100:.2f}%' for record in records.values())
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6 보행 토크 검증</title>
<style>body{{font:16px/1.6 system-ui;max-width:1280px;margin:30px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:6px}}th{{background:#edf3fa}}video,img{{max-width:100%;display:block;margin:10px 0}}video{{width:100%;background:#000}}
section{{border-top:1px solid #ddd}}.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.wide{{overflow-x:auto}}h3{{font-size:16px}}</style>
<h1>V6 몸통 라운드 CAD · ROS2 / Isaac Sim 보행·모터 RMS 검증</h1>
<p><b>결론</b>: 두 정책 모두 9개 시험 통과({len(a["passed"])}/9, {len(b["passed"])}/9).
두 정책 각각 108개 관절·시험 조합 모두 토크 RMS가 정격 이하이고, 모터 사양 최대 토크를 넘은 시험도 없다.
가장 높은 RMS/정격은 무릎 {a["worst"]["KFE"]["ratio"]:.2f}({a["label"]}, {LABELS[a["worst"]["KFE"]["case"]]})와
{b["worst"]["KFE"]["ratio"]:.2f}({b["label"]}, {LABELS[b["worst"]["KFE"]["case"]]})다.
Jetson 번들 모델은 model_15699를 유지한다. 모든 결과는 시뮬레이션이며 실기 검증이 아니다.</p>
<p>로봇: <code>assets/quadruped_v6_20261025</code> (USD SHA {USD_SHA256[:8]}…, {summary["mass_kg"]:.2f} kg, 충돌 형상 75개).
시험: round2 suite job 2343 소스 그대로(ROS2 제어기 포함), seed 42, 각 조건 1회 실행.
v6 추가 학습: job 10806, model_15699에서 round2 방식 400회(rms_retain, 128환경, lr 3e-5, 보존 가중치 5).</p>
<p><a href="summary.json">전체 결과</a> · <a href="motor_rms.csv">216개 모터·시험별 RMS</a> ·
<a href="train_jobs/10806/results/model_16099.pt">v6 추가 학습 체크포인트</a> · <a href="build_report.py">생성 스크립트</a></p>
<h2>시험별 결과</h2><table><tr><th>시험</th><th>{a["label"]}</th><th>{b["label"]}</th><th>RMS/정격 최대 (원래 / 추가 학습)</th>
<th>기록 s</th><th>자가 접촉 N</th></tr>{case_rows}</table>
<h2>모터별 토크 RMS / 정격</h2><p>정격: 고관절 roll(HAA) 8.3 N·m, 고관절 pitch(HFE)·무릎(KFE) 18 N·m. RMS는 각 시험의 전체 기록구간 값이다.
0.9 이상은 노란색, 1 초과는 빨간색으로 표시한다. RL/RR은 뒷다리(BL/BR)다. 무릎이 시뮬레이션 토크 상한(48 N·m)에 닿은 시간 비율: {sat}.</p>
{motor_tables}
<h2>발 수직 하중과 이전 로봇 비교</h2><table><tr><th>로봇 · 정책</th><th>질량 kg</th><th>통과</th><th>Fz p99 N</th><th>Fz p99.9 N</th>
<th>Fz 최대 N</th><th>접지 중 p99.9 N</th><th>무릎 RMS/정격 최대</th><th>회전 오차 rad (≤0.6)</th><th>요철 옆 이동 m (≤0.70)</th></tr>{load_rows}</table>
<p>하중은 shank_support_force의 수직 성분(접촉 수직력만, 마찰력 없음)이다. p99·p99.9는 9개 시험·4발의 전체 샘플(공중 구간 0 포함)로,
구조 설계 하중 220/380/1045 N을 재현하는 계산법이다. 최대값은 모든 실행에서 발이 닿는 순간 한 샘플(5 ms)에만 튀는 값이라 실행마다 크게 달라진다:</p>
<ul>{peak_rows}</ul>
<h2>주의</h2><ul><li>각 조건을 한 번씩만 실행했다. 요철 지형은 프로토콜의 고정 seed(42)로 만들어지며 <code>--seed</code>로 바뀌지 않는다.</li>
<li>기립·앉기 제어기는 2026-10-11 CAD 기준 MuJoCo 모델(23.1 kg)을 그대로 쓴다.</li>
<li>전류는 효율 1을 가정한 추정 상전류이며 배터리 전류나 실측값이 아니다.</li></ul>
{sections}</html>'''
    (HERE / 'report.html').write_text(content)
    print(json.dumps(dict(all_cases_walk=walk_ok, all_rms_within_rated=rms_ok,
                          worst_knee={k: round(r['worst']['KFE']['ratio'], 3) for k, r in records.items()})))


if __name__ == '__main__':
    main()
