"""Compare a CAD fine-tuning candidate with the preserved fixed-suite baseline."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np

from build_suite_report import LABELS, NAMES, check_trace, controller_failure, plot


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_source(run):
    records = []
    for line in (run / 'source.sha256').read_text().splitlines():
        expected, remote = line.split(maxsplit=1)
        relative = remote.split('/source/', 1)[1]
        local = run / 'source' / relative
        actual = hashlib.sha256(local.read_bytes()).hexdigest() if local.is_file() else None
        records.append(dict(file=relative, matches=actual == expected))
    return dict(files=len(records), passed=all(item['matches'] for item in records),
                failures=[item['file'] for item in records if not item['matches']])


def plot_training(directory, job):
    path = directory / 'verification' / f'{job}_training_events.json'
    record = json.loads(path.read_text())
    assert record['all_scalar_values_finite']
    scalars = record['scalars']
    start = json.loads((directory/'train_jobs'/str(job)/'results/training_contract.json').read_text())['starting_iteration']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    third_tag = 'Curriculum/terrain_levels' if 'Curriculum/terrain_levels' in scalars else 'Episode_Reward/rms_motor_load'
    for axis, tag, factor, label in (
        (axes[0], 'Train/mean_reward', 1., 'Episode reward'),
        (axes[1], 'Train/mean_episode_length', .02, 'Episode length (s)'),
        (axes[2], third_tag, 1., 'Mean terrain level' if third_tag.startswith('Curriculum') else 'RMS load reward')):
        values = np.asarray(scalars[tag]['events'])
        axis.plot(values[:, 0] - start, values[:, 1] * factor, alpha=.3, label='Recorded')
        window = min(50, len(values))
        smoothed = np.convolve(values[:, 1], np.ones(window) / window, mode='valid')
        axis.plot(values[window-1:, 0] - start, smoothed * factor, label=f'{window}-update mean')
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        axis.legend()
    axes[-1].set_xlabel('Fine-tuning updates (zero-based)')
    fig.savefig(directory / 'training_curve.png', dpi=140)
    plt.close(fig)
    selected = ('Train/mean_reward', 'Train/mean_episode_length',
                'Curriculum/terrain_levels', 'Diagnostics/nonfinite_steps',
                'Episode_Termination/physics_diverged', 'Episode_Reward/rated_motor_load',
                'Episode_Reward/excess_rated_motor_load', 'Episode_Reward/self_contact_count')
    selected += ('Episode_Reward/rms_motor_load',)
    selected += ('Loss/policy_retention', 'Loss/policy_retention_eligible_fraction')
    return {tag: {key: value for key, value in scalars[tag].items() if key != 'events'}
            for tag in selected if tag in scalars}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--training-job', required=True)
    parser.add_argument('--suite-jobs', required=True)
    args = parser.parse_args()
    directory = args.directory
    baseline = json.loads(args.baseline.read_text())
    training = directory / 'train_jobs' / args.training_job
    completed = json.loads((training / 'results/completed.json').read_text())
    contract = json.loads((training / 'results/training_contract.json').read_text())
    preflight = json.loads((training / 'results/preflight.json').read_text())
    training_metrics = plot_training(directory, args.training_job)
    candidate = training / 'results' / Path(completed['checkpoint']).name
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert completed.get('training_completed') is True
    assert candidate_hash == completed['checkpoint_sha256']
    assert preflight['passed'] and preflight['nonfinite_steps'] == 0
    jobs = args.suite_jobs.split(',')
    results, sources, integrity, origins = {}, {}, {}, {}
    sources['training/' + args.training_job] = check_source(training)
    for job in jobs:
        run = directory / 'suite_jobs' / job
        sources['evaluation/' + job] = check_source(run)
        baseline_source = args.baseline.parent / 'runs' / '2321' / 'source/config'
        for protected in ('evaluate_v15_cad_suite.py', 'v15_cad_suite_protocol.py',
                          'v15_cad_suite_metrics.py', 'v15_cad_contract.py'):
            assert (run / 'source/config' / protected).read_bytes() == (baseline_source / protected).read_bytes(), protected
        checkpoint_manifest = (run / 'checkpoints.sha256').read_text().splitlines()
        assert checkpoint_manifest[0].split()[0] == candidate_hash
        assert checkpoint_manifest[1].split()[0] == contract['student_sha256']
        usd = run / 'source/assets/quadruped_20261011/quadruped_robot.usda'
        assert hashlib.sha256(usd.read_bytes()).hexdigest() == contract['usd_sha256']
        for result in json.loads((run / 'results/suite.json').read_text())['cases']:
            name = result['case']
            assert name not in results, f'Duplicate candidate case: {name}'
            folder = run / 'results' / name
            assert result.get('seed') == 42
            baseline_result = args.baseline.parent / baseline['case_attempts'][name][-1]['result_path']
            previous_config = json.loads((baseline_result.parent / 'case_config.json').read_text())
            current_config = json.loads((folder / 'case_config.json').read_text())
            previous_config.pop('lane_origin_world_m', None)
            current_config.pop('lane_origin_world_m', None)
            assert current_config == previous_config, f'Changed case criteria: {name}'
            results[name] = result
            integrity[name] = check_trace(folder, result)
            origins[name] = folder
    assert all(item['passed'] for item in sources.values())
    expected = baseline['expected_cases']
    assert list(LABELS) == expected
    assert set(results) == set(expected), 'Missing fixed-suite candidate cases'
    passed = [name for name in expected if results[name].get('task_success')
              and integrity[name]['pass_check']]
    rows, details, motors, comparisons = [], [], [], {}
    for name in expected:
        result = results[name]
        previous = baseline['results'][name]
        folder = origins[name]
        relative = folder.relative_to(directory).as_posix()
        plot(folder)
        reasons = list(result.get('motion_verdict', {}).get('reasons', []))
        if result.get('failure'):
            reasons.append(result['failure'])
        error = controller_failure(folder)
        if error:
            reasons.append(error)
        if result.get('collision_checks_pass') is False:
            reasons.append('자가 접촉 >1 N')
        if result.get('simulator_peak_exceeded'):
            reasons.append('구동 토크 피크 초과')
        if not integrity[name]['pass_check']:
            reasons.append('시계열 검증 실패')
        currents = result.get('ideal_current_rms_a', [])
        previous_currents = previous.get('ideal_current_rms_a', [])
        contact = result.get('max_nonadjacent_contact_force_n')
        comparisons[name] = dict(
            baseline_pass=name in baseline['passed_cases'], candidate_pass=name in passed,
            baseline_seconds_recorded=previous['seconds_recorded'], candidate_seconds_recorded=result['seconds_recorded'],
            baseline_metrics=previous.get('motion_verdict', {}).get('metrics'),
            candidate_metrics=result.get('motion_verdict', {}).get('metrics'),
            baseline_max_current_rms_a=max(previous_currents) if previous_currents else None,
            candidate_max_current_rms_a=max(currents) if currents else None,
            baseline_max_self_contact_n=previous.get('max_nonadjacent_contact_force_n'),
            candidate_max_self_contact_n=contact, reasons=reasons)
        fmt = lambda value: f'{value:.3f}' if isinstance(value, (int, float)) else '—'
        rows.append('<tr><td><a href="#' + name + '">' + LABELS[name] + '</a></td><td>'
            + ('통과' if name in baseline['passed_cases'] else '미통과') + '</td><td>'
            + ('통과' if name in passed else '미통과') + '</td><td>'
            + fmt(previous['seconds_recorded']) + ' → ' + fmt(result['seconds_recorded']) + '</td><td>'
            + fmt(max(previous_currents) if previous_currents else None) + ' → '
            + fmt(max(currents) if currents else None) + '</td><td>' + fmt(contact)
            + '</td><td>' + html.escape('; '.join(reasons)) + '</td></tr>')
        for index, joint in enumerate(NAMES):
            if not currents:
                break
            motors.append(dict(case=name, joint=joint,
                baseline_seconds_recorded=previous['seconds_recorded'], candidate_seconds_recorded=result['seconds_recorded'],
                baseline_current_rms_a=previous_currents[index],
                candidate_current_rms_a=currents[index],
                candidate_current_peak_a=result['ideal_current_peak_a'][index],
                baseline_torque_rms_nm=previous['torque_rms_nm'][index],
                candidate_torque_rms_nm=result['torque_rms_nm'][index],
                candidate_torque_peak_nm=result['torque_peak_nm'][index],
                candidate_rms_exceeds_rated=result['rms_exceeds_rated'][index],
                candidate_rated_exceed_fraction=result['rated_exceed_fraction'][index]))
        video = folder / 'video/env_00.mp4'
        visual = f'<video controls preload="none" src="{relative}/video/env_00.mp4"></video>' if video.exists() else ''
        if (folder / 'telemetry.png').exists():
            visual += f'<img loading="lazy" src="{relative}/telemetry.png" alt="{LABELS[name]} 시계열">'
        details.append(f'<section id="{name}"><h2>{LABELS[name]}</h2><p>'
            + html.escape('; '.join(reasons) or '기존 기준 통과')
            + f'</p><p><a href="{relative}/result.json">결과</a> · '
            + f'<a href="{relative}/telemetry.npz">시계열</a> · '
            + f'<a href="{relative}/controller.log">ROS2 로그</a></p>' + visual + '</section>')
    with (directory / 'motor_comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(motors[0]))
        writer.writeheader()
        writer.writerows(motors)
    report = dict(schema=1, training_job=args.training_job, evaluation_jobs=jobs,
        training_completed=completed, training_contract=contract, preflight=preflight,
        training_metrics=training_metrics,
        baseline_summary_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        baseline_passed_cases=baseline['passed_cases'], candidate_passed_cases=passed,
        candidate_failed_cases=[name for name in expected if name not in passed],
        expected_cases=expected, comparisons=comparisons, results=results,
        telemetry_integrity=integrity, source_integrity=sources,
        candidate_all_fixed_cases_pass=bool(len(passed) == len(expected)),
        candidate_rated_rms_within_all_cases=bool(all(
            not any(result.get('rms_exceeds_rated', [True])) for result in results.values())),
        adoption='Evaluation candidate; task criteria and rated-RMS risks reported separately',
        current_semantics='Ideal motor phase current estimate; efficiency=1; not battery current or measured solver torque',
        scope='One seed42 per fixed case; rated RMS is reported separately from the task-success gate')
    write(directory / 'summary.json', report)
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>V15 CAD 파인튜닝 비교</title><style>
body{{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#17202a}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #dce1e5;padding:8px;text-align:left}}
th{{background:#eef3f8}}video,img{{max-width:100%;display:block;margin:16px 0}}section{{border-top:1px solid #ccd4dc;padding:20px 0}}
</style><h1>V15 CAD 파인튜닝 · ROS2 검증 비교</h1>
<p>기존 {len(baseline['passed_cases'])}/15 → 후보 {len(passed)}/15. 학습 job {args.training_job}, 실제 PPO 업데이트 {completed['updates']:,}회.
후보 {candidate.name}, SHA256 {candidate_hash}.</p>
<p>원본 CAD 질량 23.0945 kg, 13개 강체와 67개 충돌 형상을 유지했다. 기존 체크포인트와 시험 기준은 보존했다.
평지·명령 방향·정지·실제 4단 계단·요철·기립·기립과 보행 통합을 외부 ROS2 프로세스로 시험했다.</p>
<p>전류는 HAA 1.23, HFE/KFE 1.44 Nm/A를 적용한 이상적인 모터 상전류 추정이며 배터리 전류와 다르다.
정격 RMS 초과 여부는 관절별 CSV에 별도 기록했다. 접촉 반력과 구동 토크도 별도 기록했다.
각 RMS는 실제 기록구간 전체의 값이며, 조기 실패한 시험은 단축된 구간이다. 비교표와 CSV에 기록 시간을 표시했다.
미통과 후보의 토크 감소는 동작을 달성한 토크 최적화로 판단하지 않는다.</p>
<p><a href="summary.json">전체 비교 JSON</a> · <a href="motor_comparison.csv">관절별 비교 CSV</a> ·
<a href="train_jobs/{args.training_job}/results/preflight.json">원시 상태를 확인한 학습 전 점검</a></p>
<p>학습 보상과 교육 단계는 ROS2 시험의 통과 판정을 대신하지 않는다.</p>
<img src="training_curve.png" alt="실제 PPO 학습 기록">
<table><tr><th>시험</th><th>기존</th><th>후보</th><th>기록 시간 s 기존→후보</th><th>최대 관절 RMS A 기존→후보</th><th>자가 접촉 N</th><th>미통과 사유</th></tr>
{''.join(rows)}</table>{''.join(details)}</html>'''
    (directory / 'report.html').write_text(content)
    print(json.dumps(dict(passed=len(passed), tested=len(results), motor_rows=len(motors),
                          candidate_sha256=candidate_hash, source_integrity=True)))


if __name__ == '__main__':
    main()
