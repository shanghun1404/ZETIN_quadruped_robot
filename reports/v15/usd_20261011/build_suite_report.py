"""Build a standalone report from actual archived CAD acceptance runs."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np

LABELS=dict(flat_14s='평지 전진',stop_restart='정지·재출발',reverse='후진',lateral='측면 이동',yaw='제자리 회전',
    stairs_up_05cm='5 cm 계단 상승',stairs_down_05cm='5 cm 계단 하강',stairs_up_10cm='10 cm 계단 상승',
    stairs_down_10cm='10 cm 계단 하강',stairs_up_157mm='15.7 cm 계단 상승',stairs_down_157mm='15.7 cm 계단 하강',
    rough='1–3.5 cm 요철',standing_hold='기립 유지',sit_rise_walk_stop_sit='10 cm 낮추기·기립·보행·정지',
    deep_crouch_rise_walk_stop_sit='18 cm 낮추기·기립·보행·정지')
NAMES=[leg+'_'+joint for leg in ['FL','FR','RL','RR'] for joint in ['HAA','HFE','KFE']]


def controller_failure(folder):
    path=folder/'controller.log'
    if not path.exists():return None
    errors=[line.strip() for line in path.read_text(errors='replace').splitlines()
        if line.startswith(('ValueError:', 'RuntimeError:', 'AssertionError:'))]
    return errors[-1] if errors else None


def check_trace(folder,result):
    path=folder/'telemetry.npz'
    if not path.exists():return dict(pass_check=False,reason='no_telemetry')
    raw=np.load(path)
    if not raw.files:return dict(pass_check=False,reason='empty_telemetry')
    stamps=raw['command_stamp_ns'];states=raw['state_stamp_ns'];n=len(stamps)
    history=json.loads((folder/'status.json').read_text())
    force=raw['nonadjacent_pair_force_n']
    integrity=dict(samples=n,seconds=n*.005,starts_at_zero=bool(stamps[0]==0),
        command_spacing_5ms=bool(np.all(np.diff(stamps)==5_000_000)),
        state_is_one_step_after_command=bool(np.all(states==stamps+5_000_000)),
        matching_status_stamp=bool(np.array_equal(stamps,[s['stamp_ns'] for s in history[:n]])),
        finite_recorded_state=bool(all(np.isfinite(raw[k]).all() for k in ['position','quaternion','velocity','joint_position','estimated_torque','nonadjacent_pair_force_n'])),
        nonadjacent_directed_pairs=int(force.shape[1]),
        torque_vs_status_max_nm=float(abs(raw['estimated_torque']-raw['bounded_torque']).max()),
        recorded_duration_agrees=bool(abs(n*.005-result['seconds_recorded'])<1e-8))
    integrity['pass_check']=bool(all(integrity[k] for k in ['starts_at_zero','command_spacing_5ms','state_is_one_step_after_command',
        'matching_status_stamp','finite_recorded_state','recorded_duration_agrees']) and force.shape[1]==132 and integrity['torque_vs_status_max_nm']<=1e-3)
    return integrity


def plot(folder):
    path=folder/'telemetry.npz'
    if not path.exists():return False
    raw=np.load(path)
    if not raw.files:return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=raw['state_stamp_ns']*1e-9
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
    axes[0].plot(t,raw['position'][:,1],label='World Y progress')
    axes[0].plot(t,raw['position'][:,2],label='Trunk Z')
    axes[0].plot(t,raw['ground_height_m'],label='Ground Z')
    axes[0].set_ylabel('Position (m)');axes[0].legend(ncol=3)
    for j,label in [(0,'right X'),(1,'forward Y')]:
        axes[1].plot(t,raw['velocity'][:,j],label='body COM '+label)
        axes[1].plot(t,raw['command'][:,j],ls='--',label='command '+label)
    if folder.name=='yaw':
        yaw_axis=axes[1].twinx()
        yaw_axis.plot(t,raw['angular_velocity'][:,2],color='tab:purple',label='yaw rate')
        yaw_axis.plot(t,raw['command'][:,2],color='tab:purple',ls='--',label='yaw command')
        yaw_axis.set_ylabel('Yaw rate (rad/s)');yaw_axis.legend(loc='upper right')
    axes[1].set_ylabel('Planar speed (m/s)');axes[1].legend(ncol=2,loc='upper left')
    axes[2].plot(t,raw['nonadjacent_pair_force_n'].max(axis=1),label='Nonadjacent maximum')
    axes[2].axhline(1.,color='k',ls='--',label='1 N gate')
    axes[2].set_ylabel('Self contact (N)');axes[2].legend()
    for j,label in enumerate(['HAA','HFE','KFE']):
        axes[3].plot(t,abs(raw['estimated_torque'][:,j::3]).max(axis=1),label=label+' max over4legs')
    axes[3].axhline(24.8,color='gray',ls='--',label='HAA plant cap')
    axes[3].axhline(48.,color='k',ls='--',label='HFE/KFE plant cap')
    axes[3].set_ylabel('Estimated torque (N m)');axes[3].set_xlabel('Time (s)');axes[3].legend(ncol=2)
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(folder/'telemetry.png',dpi=130);plt.close(fig)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--jobs',required=True)
    args=p.parse_args();directory=args.directory
    output=directory/'full_suite_20261011';output.mkdir(exist_ok=True)
    results={};origins={};integrity={};attempts=[];case_attempts={};archive_integrity={}
    for job in args.jobs.split(','):
        run=output/'runs'/job
        summary_path=run/'results/suite.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            for result in summary['cases']:
                name=result['case'];folder=run/'results'/name
                results[name]=result;origins[name]=(job,folder)
                trace=check_trace(folder,result)
                archive_integrity[job+'/'+name]=trace
                controller=run/'source/ros2/v151_ros2_control/v151_ros2_control/cad_standing.py'
                case_attempts.setdefault(name,[]).append(dict(job=job,
                    result_path=(folder/'result.json').relative_to(output).as_posix(),
                    task_success=result.get('task_success',False),seconds_recorded=result.get('seconds_recorded'),
                    failure=result.get('failure'),controller_failure=controller_failure(folder),
                    controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest() if controller.exists() else None,
                    telemetry_integrity=trace))
        if (run/'results/startup_failure.json').exists():attempts.append(dict(job=job,startup=json.loads((run/'results/startup_failure.json').read_text())))
        elif not summary_path.exists():attempts.append(dict(job=job,
            physical_cases_recorded=False,reason='앱 시작 중 취소; 물리 시험 결과 없음'))
    rows=[];details=[];csv_rows=[]
    for name in LABELS:
        if name not in results:continue
        result=results[name];job,folder=origins[name]
        integrity[name]=check_trace(folder,result)
        has_plot=plot(folder)
        reasons=list(result.get('motion_verdict',{}).get('reasons',[]))
        if result.get('failure'):reasons=[result['failure']]+reasons
        actual_error=controller_failure(folder)
        if actual_error:reasons.insert(0,actual_error)
        if result.get('collision_checks_pass') is False:reasons.append('자가 접촉 >1 N')
        if result.get('simulator_peak_exceeded'):reasons.append('구동 토크 피크 한도 초과')
        if not result.get('torque_transport_integrity_pass',False):reasons.append('토크 명령 검증 실패')
        if not integrity[name]['pass_check']:reasons.append('시계열 검증 실패')
        status='통과' if result.get('task_success') and integrity[name]['pass_check'] else '미통과'
        def number(key):
            value=result.get(key)
            return f'{value:.3f}' if isinstance(value,(int,float)) else '—'
        current=result.get('ideal_current_rms_a',[])
        current_text=f'{min(current):.2f}–{max(current):.2f}' if current else '—'
        requested=result.get('seconds_requested',0.)
        rows.append(f'<tr><td><a href="#{name}">{LABELS[name]}</a></td><td>{status}</td><td>{number("seconds_recorded")} / {requested:g}</td><td>{current_text}</td><td>{number("max_nonadjacent_contact_force_n")}</td><td>{html.escape("; ".join(reasons))}</td></tr>')
        motor_rows=[]
        for i,joint in enumerate(NAMES):
            if not current:break
            record=dict(case=name,job=job,joint=joint,seconds_recorded=result['seconds_recorded'],
                torque_rms_nm=result['torque_rms_nm'][i],torque_peak_nm=result['torque_peak_nm'][i],
                ideal_phase_current_rms_a=current[i],ideal_phase_current_peak_a=result['ideal_current_peak_a'][i],
                rated_exceed_percent=100*result['rated_exceed_fraction'][i],
                rms_exceeds_rated=result['rms_exceeds_rated'][i],simulator_peak_nm=result['simulator_peak_torque_nm'][i])
            csv_rows.append(record)
            motor_rows.append('<tr><td>'+joint+'</td>'+''.join(f'<td>{record[k]:.2f}</td>' for k in
                ['torque_rms_nm','torque_peak_nm','ideal_phase_current_rms_a','ideal_phase_current_peak_a','rated_exceed_percent'])
                +'<td>'+('초과' if record['rms_exceeds_rated'] else '이내')+'</td></tr>')
        relative=folder.relative_to(output).as_posix()
        history=''
        if len(case_attempts[name])>1:
            history='<details><summary>최초 시험과 재시험 이력</summary><ul>'+''.join(
                '<li><a href="'+item['result_path']+'">작업 '+item['job']+'</a>: '
                +('통과' if item['task_success'] else '미통과')
                +f', {item["seconds_recorded"]:.3f} s; '
                +html.escape(item['controller_failure'] or item['failure'] or '완료')+'</li>'
                for item in case_attempts[name])+'</ul></details>'
        video=f'<video controls preload="none" src="{relative}/video/env_00.mp4"></video>' if (folder/'video/env_00.mp4').exists() else ''
        image=f'<img loading="lazy" src="{relative}/telemetry.png" alt="{LABELS[name]} 시계열">' if has_plot else ''
        details.append(f'<section id="{name}"><h2>{LABELS[name]} — {status}</h2><p>작업 {job}; '+html.escape('; '.join(reasons) if reasons else '모션·비인접 충돌·토크 명령 및 피크 한도 검증 통과.')
            +f'</p><p><a href="{relative}/result.json">결과 JSON</a> · <a href="{relative}/telemetry.npz">원본 시계열</a> · '
            +f'<a href="{relative}/case_config.json">시험 조건</a> · <a href="{relative}/controller.log">ROS2 로그</a></p>'
            +history+video+image+'<details><summary>관절별 토크·추정 전류</summary><table><tr><th>관절</th><th>RMS Nm</th><th>피크 Nm</th><th>RMS A</th><th>피크 A</th><th>정격 초과 시간 %</th><th>RMS/정격</th></tr>'
            +''.join(motor_rows)+'</table></details></section>')
    if csv_rows:
        with (output/'motor_metrics.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    passed=[name for name,result in results.items() if result.get('task_success') and integrity.get(name,{}).get('pass_check')]
    report=dict(schema=2,expected_cases=list(LABELS),tested_cases=list(results),passed_cases=passed,
        failed_cases=[n for n in results if n not in passed],missing_cases=[n for n in LABELS if n not in results],
        results=results,archive_jobs=args.jobs.split(','),startup_attempts=attempts,telemetry_integrity=integrity,
        case_attempts=case_attempts,all_attempt_telemetry_integrity=archive_integrity,
        current_assumptions=dict(haa_output_kt_nm_per_a=1.23,hfe_kfe_output_kt_nm_per_a=1.44,efficiency=1.,type='ideal phase current estimate, not battery input'),
        scope='single deterministic seed42, nominal temperature and voltage; CPU collision screen plus physical66-pair contact monitoring')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'telemetry_integrity.json').write_text(json.dumps(integrity,indent=2,allow_nan=False))
    (output/'all_attempt_telemetry_integrity.json').write_text(json.dumps(archive_integrity,indent=2,allow_nan=False))
    content=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>새 CAD USD V15 전체 시험</title>
    <style>body{{font:16px/1.6 system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #dce1e5;padding:8px;text-align:left}}th{{background:#f1f5f9}}video,img{{max-width:100%;display:block;margin:16px 0}}section{{border-top:1px solid #ccd4dc;padding:20px 0}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#1260a8}}</style>
    <h1>새 CAD USD · V15 ROS2 전체 시험</h1><p>통과 {len(passed)} / 실행 {len(results)} / 예정 {len(LABELS)}. 원본 CAD 23.0945 kg, 13 rigid body, 12 관절, 67 convexHull을 유지했다. 채택 V15 actor job339 model_11699와 기존 depth student를 ROS2 외부 프로세스에서 실행했다.</p>
    <p>실제 계단 4단을 지나 최종 Y≥3.9 m와 상승·하강 지면 변화를 확인했다. 계단 아래에서 버틴 실행은 통과하지 않는다. 정지·속도·기립 기준은 각 시험 조건 파일에 고정했다. 각 독립 시험은 한 번 초기화하며, 시험 도중에는 리셋하지 않는다.</p>
    <p>10 cm·18 cm 낮은 자세는 네 발로 지지하는 웅크림이다. 몸통을 지면에 붙이는 앉기나 넘어진 상태의 복구 기능은 이번 동작에 포함되지 않는다.</p>
    <p>각 항목은 가장 최근 실행의 결과다. 재시험한 항목에는 최초 실패의 원본 결과와 로그를 함께 보존했다. 평지는 이전 job2280의 속도 MAE 0.1499 m/s와 이번 job2321의 0.1522 m/s가 기준 0.15 m/s 양쪽에 있어 여유가 없다. 요철의 통과 기준은 주행 구간 완주·방향·자세·충돌이며 속도 추종 정확도는 포함하지 않는다. 요철에서는 명령 0.35 m/s에 초기2 s 이후 평균 전진 속도0.538 m/s로 과속했다.</p>
    <table><tr><th>시험</th><th>결과</th><th>기록 / 요청 s</th><th>관절 RMS A 범위</th><th>최대 자가 접촉 N</th><th>미통과 이유</th></tr>{''.join(rows)}</table>
    <h2>측정 해석</h2><p>토크는 각 물리 단계 직전의 implicit PD 제한과 별도 feedforward를 합산한 추정치다. 실제 PhysX 구동 모멘트 측정값이 아니다. 전류는 |τ|/(Kt×감속비), 효율 1 가정으로 HAA 1.23·HFE/KFE 1.44 Nm/A를 사용한 모터 상전류 추정치이며 배터리 전류와 다르다.</p>
    <p>시뮬레이터 토크 피크 한도는 HAA24.8·HFE/KFE48 Nm이고, 비교에 사용한 모터 피크는24.8·53 Nm다. 정격은8.3·18 Nm다. 피크 한도 이내여도 정격을 자주 넘거나 RMS가 정격을 넘으면 지속 사용의 여유가 부족할 수 있다. 전압·발열·실제 감속기 효율과 하드웨어 지연은 이 시험에서 측정하지 않았다.</p>
    <p>자가 충돌은 원본67 convexHull을 활성화한 상태로 13개 실제 바디의 비인접66쌍을 양방향으로 감시했다. 직접 연결 관절12쌍은 원본 PhysX 필터에 따라 제외된다. CPU 원본 hull 검사에서는 의도적으로 발을 겹친 대조 자세를 검출했다. 한 개 시드42의 시험 결과이며 모든 실환경에서의 안정성을 보증하지 않는다.</p>
    <h2>실패 원인 확인과 수정</h2><p>최초 통합 동작 job2320은 보행 후 기립 복귀에서 착지 IK 오류로 ROS2 노드가 종료됐다. 앵커와 제어기 상태를 유지한 30틱 CPU 재생에서 앞오른발 잔차 2.05351 mm를 재현했다. 원래 참조가 발을 지면 아래 5 mm로 요구했으며 새 CAD에서는 도달할 수 없었다. 새 CAD 제어기의 기본 착지 깊이만 0 mm로 수정했고, 같은 입력에서 잔차 2.46×10⁻¹⁰ m와 WBC feasible을 확인했다. IK 허용오차·토크 한도·시험 통과 기준은 유지했다. 물리 재시험 결과는 해당 항목의 최신 실행에 기록했다. <a href="diagnostics/cad_landing_ik_replay.json">재현 수치</a> · <a href="diagnostics/replay_cad_landing_ik_cpu.py">재현 코드</a></p>
    <p>수정 후 job2322의10 cm 통합 동작은 기립·보행을 수행하고 기존 IK 오류를 지나갔지만, 복귀에서 자세가 무너져18.960 s에 몸통 높이0.197 m로 중단됐다. 몸통↔앞오른쪽 하퇴654.07 N, 뒷다리 하퇴끼리583.59 N 접촉도 검출됐다. 이 수정은 정지 전환을 안정화한 것으로 채택할 수 없다. CAD 기립 통합은 시험용 ROS2 제어기에서 검증한 후보이며 실제 장치에 배포하지 않았다.</p>
    <p>같은 재시험의18 cm 통합 동작도 기립·보행 후18.440 s에 지지 발 FL의 IK 잔차147.192 mm로 중단됐다. 두 통합 동작 모두 다시 앉기까지 완료하지 못했다. 전진 정책의0속도 명령에서도 정지·재출발 시험은 정지 구간 속도가1.1 m/s를 넘었다. 기립 유지 통과만으로 보행에서 기립으로 복귀하는 전환을 통과했다고 볼 수 없다.</p>
    <p>10 cm 상승·하강 및 15.7 cm 상승에서 몸통과 앞왼쪽 하퇴 사이 접촉을 검출했다. 최대 반력은 1032.72 N이며 모터의 능동 구동 토크와 별개다. 원본 CAD hull의 가까운 메시 쌍은 body_shell_F_1 ↔ knee_4다. 10 cm 시험에서는 형상 내부 겹침을 확인했다. 15.7 cm 피크 전후 저장 자세에서는 내부 겹침이 확인되지 않아 접촉 오프셋·cooking·물리 단계 내부 보정 중 원인을 확정하지 않았다. PhysX는 형상 관통 전에 접촉을 생성할 수 있다. <a href="https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html">PhysX 공식 문서</a> · <a href="diagnostics/cad_contacts.json">접촉 분석 수치</a> · <a href="diagnostics/analyze_cad_contacts_cpu.py">분석 코드</a></p>
    <p><a href="summary.json">전체 결과 JSON</a> · <a href="motor_metrics.csv">15개 시험 관절별 CSV</a> · <a href="all_attempt_telemetry_integrity.json">재시험 포함 시계열 검증</a> · <a href="archive_integrity.json">소스·체크포인트 해시 확인</a> · <a href="../verification/full_suite_cpu_final.log">CPU·ROS2 회귀 시험 로그</a> · <a href="../report.html">기존 평지 job2280</a></p>
    {''.join(details)}</html>'''
    (output/'report.html').write_text(content)
    print(json.dumps({k:report[k] for k in ['tested_cases','passed_cases','failed_cases','missing_cases']},ensure_ascii=False))


if __name__=='__main__':main()
