"""Preserve both measured fine-tuning stages beside the original fixed suite."""
import argparse
import csv
import html
import json
from pathlib import Path

from build_suite_report import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory
    baseline = json.loads(args.baseline.read_text())
    first = json.loads((directory/'summary.json').read_text())
    second = json.loads((directory/'round2/summary.json').read_text())
    expected = baseline['expected_cases']
    assert expected == first['expected_cases'] == second['expected_cases']
    assert second['training_contract']['actor_sha256'] == first['training_completed']['checkpoint_sha256']
    assert first['training_contract']['usd_sha256'] == second['training_contract']['usd_sha256']
    assert all(item['passed'] for record in (first, second) for item in record['source_integrity'].values())
    rows, table = [], []
    for name in expected:
        row = dict(case=name, baseline_pass=name in baseline['passed_cases'],
                   round1_pass=name in first['candidate_passed_cases'],
                   round2_pass=name in second['candidate_passed_cases'])
        cells = []
        for label, record in (('baseline',baseline), ('round1',first), ('round2',second)):
            result = record['results'][name]
            currents = result['ideal_current_rms_a']
            row[label+'_seconds_recorded'] = result['seconds_recorded']
            row[label+'_phase_rms_min_a'] = min(currents)
            row[label+'_phase_rms_max_a'] = max(currents)
            row[label+'_rated_rms_exceed_joints'] = sum(result['rms_exceeds_rated'])
            row[label+'_self_contact_max_n'] = result['max_nonadjacent_contact_force_n']
            cells.append(f'<td>{"통과" if row[label+"_pass"] else "미통과"}</td>'
                         f'<td>{result["seconds_recorded"]:.3f}</td>'
                         f'<td>{min(currents):.2f}–{max(currents):.2f}</td>'
                         f'<td>{row[label+"_rated_rms_exceed_joints"]}</td>'
                         f'<td>{row[label+"_self_contact_max_n"]:.1f}</td>')
        reasons = second['comparisons'][name]['reasons']
        row['round2_failure_reasons'] = '; '.join(reasons)
        rows.append(row)
        table.append('<tr><td>'+LABELS[name]+'</td>'+''.join(cells)
                     +'<td>'+html.escape(row['round2_failure_reasons'])+'</td></tr>')
    with (directory/'stages_comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    count = [len(baseline['passed_cases']),len(first['candidate_passed_cases']),len(second['candidate_passed_cases'])]
    record = dict(schema=1, completed=True, total_ppo_updates=sum(
        r['training_completed']['updates'] for r in (first,second)), pass_counts=count,
        adopted=False, original_actor_preserved=True,
        stages=[dict(training_job=r['training_job'], checkpoint=r['training_completed'],
                     passed=r['candidate_passed_cases'], failed=r['candidate_failed_cases']) for r in (first,second)],
        all_fixed_cases_pass=second['candidate_all_fixed_cases_pass'],
        measured_scope='One unchanged seed42 ROS2/Isaac case per stage; no real hardware test',
        current_semantics=second['current_semantics'], comparison=rows)
    (directory/'overview.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>V15 CAD 추가 훈련 결과</title><style>
body{{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#17202a}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #dce1e5;padding:7px}}
th{{background:#edf3fa}}img{{max-width:100%}}
</style><h1>V15 · 원본 CAD에서 추가 훈련 및 ROS2 검증</h1>
<p>기존 {count[0]}/15 → 1차 {count[1]}/15 → 2차 {count[2]}/15.
총 실제 PPO 업데이트 {record['total_ppo_updates']:,}회. 원본 actor와 CAD는 보존했다.</p>
<p>1차는 2,000회 CAD 적응 학습, 2차는 1,000회 명령 유지·방향 진행·요철 보완 학습이다.
두 후보를 같은 외부 ROS2/Isaac Sim 시험으로 비교했다. 학습 보상과 시험 통과 기준은 별개다.</p>
<p>자가 접촉은 원본 인접 관절 필터를 유지한 66개 비인접 강체 쌍의 접촉 반력이다.
상전류는 토크 상수 1.23/1.44 Nm/A와 효율1을 사용한 추정이다. 배터리 전류와 실측 전류가 아니다.
정격 RMS 초과 관절 수도 동작 성공과 별도로 표시했다. 실패한 동작의 낮은 전류는 최적화 성공으로 판단하지 않는다.</p>
<p>각 RMS는 실제 기록구간 전체의 통계다. 조기 실패한 시험은 기록 시간이 짧으며 동일한 시간창 비교가 아니다.
표와 CSV의 기록 시간(s)을 함께 확인해야 한다.</p>
<p><a href="report.html">1차 상세·영상·모터별 CSV</a> ·
<a href="round2/report.html">2차 상세·영상·모터별 CSV</a> ·
<a href="stages_comparison.csv">3단계 비교 CSV</a> · <a href="overview.json">전체 요약 JSON</a></p>
<table><tr><th rowspan="2">시험</th><th colspan="5">기존</th><th colspan="5">1차</th>
<th colspan="5">2차</th><th rowspan="2">2차 미통과 사유</th></tr><tr>
{('<th>판정</th><th>기록 s</th><th>관절 RMS A 범위</th><th>정격 RMS 초과 수</th><th>자가 접촉 N</th>')*3}</tr>
{''.join(table)}</table><h2>실제 학습 기록</h2>
<p>1차</p><img src="training_curve.png" alt="1차 학습 기록">
<p>2차</p><img src="round2/training_curve.png" alt="2차 학습 기록">
<p>단일 seed42 검증이다. 미통과 항목을 유지했으며 실기체 준비 완료 판정이나 자동 채택은 하지 않았다.</p></html>'''
    (directory/'overview.html').write_text(content)
    print(json.dumps(dict(pass_counts=count, total_updates=record['total_ppo_updates'])))


if __name__ == '__main__':
    main()
