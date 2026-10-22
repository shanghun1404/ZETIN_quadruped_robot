"""Compare all unchanged non-stair cases with the preceding CAD candidate."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from build_finetune_report import check_source, plot_training
from build_suite_report import LABELS, NAMES, check_trace, controller_failure, plot

CASES = [name for name in LABELS if not name.startswith('stairs_')]


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def baseline_folder(directory, record, name):
    matches = [directory/'suite_jobs'/str(job)/'results'/name for job in record['evaluation_jobs']
               if (directory/'suite_jobs'/str(job)/'results'/name/'result.json').is_file()]
    assert len(matches) == 1
    return matches[0]


def compare_tree(before, after):
    def hashes(directory):
        return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    original, candidate = hashes(before), hashes(after)
    assert original and original == candidate, (before,after)
    return dict(files=len(original),unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--parent-summary', type=Path, default=None)
    parser.add_argument('--training-job', required=True)
    parser.add_argument('--suite-jobs', required=True)
    args = parser.parse_args()
    root = args.directory
    baseline = json.loads(args.baseline.read_text())
    parent_summary = args.parent_summary or args.baseline
    parent = json.loads(parent_summary.read_text())
    training = root/'train_jobs'/args.training_job
    completed = json.loads((training/'results/completed.json').read_text())
    contract = json.loads((training/'results/training_contract.json').read_text())
    preflight = json.loads((training/'results/preflight.json').read_text())
    assert completed['training_completed'] and preflight['passed'] and preflight['nonfinite_steps'] == 0
    assert contract['actor_sha256'] == parent['training_completed']['checkpoint_sha256']
    if contract.get('retention_teacher_sha256'):
        assert contract['retention_teacher_sha256'] == baseline['training_completed']['checkpoint_sha256']
    assert contract['usd_sha256'] == baseline['training_contract']['usd_sha256']
    assert contract['student_sha256'] == baseline['training_contract']['student_sha256']
    candidate = training/'results'/Path(completed['checkpoint']).name
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == completed['checkpoint_sha256']
    sources = {'training':check_source(training)}
    parent_source = args.baseline.parent/'train_jobs'/str(baseline['training_job'])/'source'
    physical_identity = {'training':{name:compare_tree(parent_source/name,training/'source'/name)
                         for name in ('assets','ros2','config/model')}}
    results, folders, traces = {}, {}, {}
    jobs = args.suite_jobs.split(',')
    for job in jobs:
        run = root/'suite_jobs'/job
        sources[job] = check_source(run)
        physical_identity[job] = {name:compare_tree(parent_source/name,run/'source'/name)
                                 for name in ('assets','ros2','config/model')}
        hashes = (run/'checkpoints.sha256').read_text().splitlines()
        assert hashes[0].split()[0] == completed['checkpoint_sha256']
        assert hashes[1].split()[0] == contract['student_sha256']
        for result in json.loads((run/'results/suite.json').read_text())['cases']:
            name = result['case']
            assert name in CASES and name not in results and result['seed'] == 42
            folder = run/'results'/name
            previous_folder = baseline_folder(args.baseline.parent, baseline, name)
            for protected in ('evaluate_v15_cad_suite.py','v15_cad_suite_protocol.py',
                              'v15_cad_suite_metrics.py','v15_cad_contract.py'):
                original = previous_folder.parents[1]/'source/config'/protected
                assert original.read_bytes() == (run/'source/config'/protected).read_bytes()
            old_cfg = json.loads((previous_folder/'case_config.json').read_text())
            new_cfg = json.loads((folder/'case_config.json').read_text())
            for cfg in (old_cfg,new_cfg): cfg.pop('lane_origin_world_m',None)
            assert old_cfg == new_cfg, name
            results[name],folders[name],traces[name] = result,folder,check_trace(folder,result)
    assert set(results) == set(CASES)
    assert all(record['passed'] for record in sources.values())
    assert all(record['pass_check'] for record in traces.values())
    passed = [name for name in CASES if results[name]['task_success']]
    rated_fail = [name for name in CASES if any(results[name]['rms_exceeds_rated'])]
    comparisons, motors, rows, details = {}, [], [], []
    for name in CASES:
        result,previous = results[name],baseline['results'][name]
        old_hot = [NAMES[i] for i,flag in enumerate(previous['rms_exceeds_rated']) if flag]
        new_hot = [NAMES[i] for i,flag in enumerate(result['rms_exceeds_rated']) if flag]
        reasons = list(result['motion_verdict']['reasons'])
        if result.get('failure'):
            reasons.append(result['failure'])
        error = controller_failure(folders[name])
        if error:
            reasons.append(error)
        comparisons[name] = dict(baseline_pass=previous['task_success'],candidate_pass=result['task_success'],
            baseline_rated_exceed_joints=old_hot,candidate_rated_exceed_joints=new_hot,
            baseline_seconds_recorded=previous['seconds_recorded'],candidate_seconds_recorded=result['seconds_recorded'],
            candidate_failure_reasons=reasons)
        current = result['ideal_current_rms_a']
        rows.append(f'<tr><td>{LABELS[name]}</td><td>{"통과" if previous["task_success"] else "미통과"}</td>'
                    f'<td>{"통과" if result["task_success"] else "미통과"}</td>'
                    f'<td>{len(old_hot)} → {len(new_hot)}</td><td>{result["seconds_recorded"]:.3f}</td>'
                    f'<td>{min(current):.2f}–{max(current):.2f}</td>'
                    f'<td>{result["max_nonadjacent_contact_force_n"]:.1f}</td>'
                    f'<td>{html.escape("; ".join(reasons))}</td></tr>')
        for i,joint in enumerate(NAMES):
            rated = 8.3 if joint.endswith('HAA') else 18.
            motors.append(dict(case=name,joint=joint,rated_torque_nm=rated,
                baseline_seconds_recorded=previous['seconds_recorded'],candidate_seconds_recorded=result['seconds_recorded'],
                baseline_torque_rms_nm=previous['torque_rms_nm'][i],candidate_torque_rms_nm=result['torque_rms_nm'][i],
                candidate_torque_peak_nm=result['torque_peak_nm'][i],baseline_current_rms_a=previous['ideal_current_rms_a'][i],
                candidate_current_rms_a=current[i],candidate_rms_exceeds_rated=result['rms_exceeds_rated'][i]))
        folder = folders[name]
        plot(folder)
        relative = folder.relative_to(root).as_posix()
        details.append(f'<section><h2>{LABELS[name]}</h2><p>정격 RMS 초과: '
            +html.escape(', '.join(new_hot) or '없음')+'</p>'
            +f'<p><a href="{relative}/result.json">원본 결과</a> · <a href="{relative}/telemetry.npz">시계열</a></p>'
            +f'<video controls preload="none" src="{relative}/video/env_00.mp4"></video>'
            +f'<img loading="lazy" src="{relative}/telemetry.png" alt="{LABELS[name]} 시계열"></section>')
    with (root/'motor_comparison.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(motors[0]));writer.writeheader();writer.writerows(motors)
    training_metrics = plot_training(root,args.training_job)
    for tag in ('Diagnostics/nonfinite_steps','Episode_Termination/physics_diverged'):
        assert training_metrics[tag]['count'] == completed['updates']
        assert training_metrics[tag]['maximum'] == 0.
    report = dict(schema=1,completed=True,training_job=args.training_job,evaluation_jobs=jobs,
        training_completed=completed,training_contract=contract,training_metrics=training_metrics,preflight=preflight,
        expected_cases=CASES,candidate_passed_cases=passed,candidate_failed_cases=[n for n in CASES if n not in passed],
        rated_rms_exceed_cases=rated_fail,all_motion_cases_pass=len(passed)==9,
        all_rated_rms_within=not rated_fail,optimization_goal_met=len(passed)==9 and not rated_fail,
        results=results,comparisons=comparisons,source_integrity=sources,telemetry_integrity=traces,
        physical_identity=physical_identity,
        baseline_summary_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        parent_summary_sha256=hashlib.sha256(parent_summary.read_bytes()).hexdigest(),
        adoption='Simulation candidate only; hardware deployment is not authorized',
        scope='Nine original non-stair cases, seed42; stairs excluded by user',
        current_semantics='Ideal motor phase current, efficiency=1; not battery current or solver measured torque')
    write(root/'summary.json',report)
    retention_note = (f'<p>후속 학습에서는 이전 통과 후보를 동결한 교사로 사용해 전진·정지 명령의 '
                      f'actor 평균 출력 보존 손실(가중치 {contract["retention_weight"]})을 PPO 손실에 추가했다.</p>'
                      if contract.get('retention_teacher_sha256') else '')
    diagnosis_note = ('<h2>정지 전환 실패 분석</h2><p>보행 부하 감소 뒤에도 두 발 착지를 회복하지 못해 기립 제어로 '
                      '넘어간 후 붕괴했다. 아래는 동일한 시험의 이전 통과 후보와 실패 후보 기록이다.</p>'
                      '<img src="transition_diagnosis.png" alt="기립 전환 실패 시계열">'
                      if (root/'transition_diagnosis.png').is_file() else '')
    previous_note = ('<p><a href="../report.html">보존된 첫 RMS 후보 결과와 전환 실패 기록</a></p>'
                     if (root.parent/'report.html').is_file() else '')
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V15 비계단 RMS 최적화</title>
<style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:8px}}th{{background:#edf3fa}}video,img{{max-width:100%;display:block;margin:18px 0}}section{{border-top:1px solid #ddd}}</style>
<h1>V15 비계단 RMS 최적화 · ROS2 / Isaac Sim 검증</h1>
<p>동작 통과 {len(passed)}/9, 정격 RMS 초과가 남은 시험 {len(rated_fail)}/9.
추가 PPO 업데이트 {completed['updates']}회, 후보 {candidate.name}.</p>
<p>후진·회전 비중과 관절별 부하 보상을 조정했다. 원본 CAD·모터 한도·주행 명령과 시험 기준은 유지했다.
학습에서는 200Hz 토크 제곱 EWMA(.5s)를 사용했고, 아래 RMS는 각 시험의 실제 전체 기록구간 통계다.
조기 실패하면 기록구간이 단축된다. 추정 상전류이며 배터리 전류나 실측 전류가 아니다.
동작과 정격 부하 기준을 함께 충족해야 최적화 목표 달성으로 판정한다. 계단은 사용자의 요청으로 이번 범위에서 제외했다.</p>
<p><a href="summary.json">전체 결과</a> · <a href="motor_comparison.csv">108개 관절별 비교</a> ·
<a href="train_jobs/{args.training_job}/results/{candidate.name}">최종 후보 체크포인트</a></p>
{retention_note}{previous_note}{diagnosis_note}
<img src="training_curve.png" alt="실제 추가 학습 기록"><table><tr><th>시험</th><th>기존</th><th>후보</th>
<th>정격 RMS 초과 수 기존→후보</th><th>기록 s</th><th>관절별 RMS A 범위</th><th>자가 접촉 N</th><th>미통과 사유</th></tr>
{''.join(rows)}</table>{''.join(details)}</html>'''
    (root/'report.html').write_text(content)
    print(json.dumps(dict(passed=len(passed),tested=9,rated_exceed_cases=rated_fail,goal_met=report['optimization_goal_met'])))


if __name__ == '__main__':
    main()
