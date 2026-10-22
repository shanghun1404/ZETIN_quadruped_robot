"""Read existing JSON evidence into Korean V15.1 HTML/JSON; no gate decisions.

python3 reports/v151/analysis/summary_report.py
Missing expected integration/comparison results remain incomplete. Inputs are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import os
from pathlib import Path

JOINTS=tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))


def build_summary(root):
    root=Path(root).resolve();sources={}
    def read(path):
        path=Path(path)
        if not path.exists(): return None
        data=path.read_bytes()
        sources[os.path.relpath(path,root)]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def unique(pattern):
        found=list(root.glob(pattern))
        if len(found)>1: raise ValueError(f'Ambiguous evidence: {pattern}')
        return read(found[0]) if found else None

    old=read(root/'comparison/594/comparison_summary.json')
    targeted=read(root/'comparison/600/comparison_summary.json')
    incident=read(root/'comparison/594/collision_incident.json')
    reconstructed=read(root/'analysis/reconstructed_collision/summary.json')
    motors=read(root.parent/'motor_audit_20260930/specs/motors.json')
    static=read(root/'static_pose_search.json')
    planned=(read(root/'training/evaluation_plan.json') or {}).get('integration',{})
    integration={}
    jobs={'591','593','597','599'} | set(planned)
    jobs.update(p.stem.removeprefix('integration_') for p in (root/'analysis/integration_reports').glob('integration_*.json') if p.stem.removeprefix('integration_').isdigit())
    jobs.update(p.name for p in (root/'integration').glob('*') if p.is_dir() and p.name.isdigit())
    for job in sorted(jobs,key=int):
        detailed=read(root/f'analysis/integration_reports/integration_{job}.json')
        metadata=read(root/f'integration/{job}/result.json') or read(root/f'integration/{job}/startup_failure.json')
        data=detailed or metadata
        if data is None:
            expected=planned.get(job,{})
            sequence=expected.get('demo_sequence_required')
            integration[job]=dict(status='미완료',task_success=None,completed=None,scenario=expected.get('scenario'),demo_sequence_required=sequence,run_kind='full_sequence' if sequence is True else ('standalone' if sequence is False else 'unknown'))
            continue
        sequence=(metadata or {}).get('demo_sequence_required',data.get('demo_sequence_required'))
        scenario=(metadata or {}).get('scenario',data.get('scenario'))
        run_kind='full_sequence' if sequence is True else ('standalone' if sequence is False else 'unknown')
        raw_task=(metadata or {}).get('task_success',data.get('task_success'))
        task=data.get('task_success')
        stair_posthoc=read(root/f'analysis/stair_{job}.json') if 'stairs' in (scenario or '') else None
        verdict_source='raw_evaluator'
        if stair_posthoc is not None:
            directed=stair_posthoc.get('posthoc_directed_stair_task_success')
            full=stair_posthoc.get('posthoc_full_flight_task_success')
            task=(directed and full) if isinstance(directed,bool) and isinstance(full,bool) else None
            verdict_source='posthoc_directed_and_full_flight'
        status='성공' if task is True else ('실패' if task is False or data.get('failure') else '판정 미기록')
        stages=data.get('modes',{})
        if not isinstance(stages,dict): stages={}
        timeline=data.get('timeline',[])
        first_stand=next((dict(segment) for segment in timeline if segment.get('mode')=='STAND'),None)
        if first_stand is not None:
            path=root/f'integration/{job}/result.npz'
            if path.exists() and all('samples' in segment for segment in timeline):
                import numpy as np
                sources[os.path.relpath(path,root)]=hashlib.sha256(path.read_bytes()).hexdigest()
                start=0
                for segment in timeline:
                    if segment.get('mode')=='STAND': break
                    start+=segment['samples']
                stop=start+first_stand['samples']
                with np.load(path) as raw:
                    keys=('drive_kp','drive_kd','feedforward','estimated_torque')
                    if all(key in raw for key in keys):
                        kp=float(np.max(np.abs(raw['drive_kp'][start:stop])))
                        kd=float(np.max(np.abs(raw['drive_kd'][start:stop])))
                        error=float(np.max(np.abs(raw['estimated_torque'][start:stop]-raw['feedforward'][start:stop])))
                        first_stand['native_effort_verification']=dict(sample_start=start,sample_stop_exclusive=stop,maximum_drive_kp=kp,maximum_drive_kd=kd,maximum_torque_minus_feedforward_nm=error,verified=kp==0 and kd==0 and error<1e-6)
        integration[job]=dict(status=status,task_success=task,completed=data.get('completed'),
            demo_sequence_required=sequence,scenario=scenario,run_kind=run_kind,
            raw_task_success=raw_task,verdict_source=verdict_source,stair_posthoc=stair_posthoc,
            failure=data.get('failure'),seconds=data.get('seconds_used',data.get('seconds_recorded')),
            stand_forward_shift_m=data.get('stand_forward_shift_m'),
            torque_semantics=data.get('torque_semantics'),stage_summaries=stages,
            timeline=timeline,first_stand=first_stand,actuation_checks=data.get('actuation_checks'),
            same_leg_thigh_foot_evidence=data.get('same_leg_thigh_foot_evidence'),
            detailed_report_available=detailed is not None)

    latest_job=max(jobs,key=int)
    def latest_matching(predicate):
        matches=[job for job,run in integration.items() if predicate(run)]
        if not matches: return dict(job=None,status='미완료')
        job=max(matches,key=int)
        return dict(job=job,status=integration[job]['status'])
    latest_full_sequence=latest_matching(lambda run:run.get('demo_sequence_required') is True)
    latest_stairs=latest_matching(lambda run:run.get('demo_sequence_required') is False and 'stairs' in (run.get('scenario') or ''))
    standing_comparisons={}
    baseline=(integration.get('593',{}).get('first_stand') or {})
    baseline_joints={j['joint']:j for j in baseline.get('joints',[])}
    for job,run in integration.items():
        if int(job)<=593 or not run.get('first_stand'): continue
        segment=run['first_stand'];rows=[]
        for joint in segment.get('joints',[]):
            previous=baseline_joints.get(joint['joint'])
            if previous is None: continue
            old_rms=previous['torque_rms_nm'];new_rms=joint['torque_rms_nm']
            rows.append(dict(joint=joint['joint'],baseline=previous,candidate=joint,rms_change_percent=100*(new_rms/old_rms-1) if old_rms else None))
        standing_comparisons[job]=dict(baseline_job='593',candidate_job=job,baseline_start_s=baseline.get('start_s'),baseline_end_s=baseline.get('end_s'),candidate_start_s=segment.get('start_s'),candidate_end_s=segment.get('end_s'),both_native_effort_verified=bool(baseline.get('native_effort_verification',{}).get('verified') and segment.get('native_effort_verification',{}).get('verified')),full_task_success=run.get('task_success'),joints=rows)

    training={}
    for job,comparison in (('589',old),('598',targeted)):
        completed=unique(f'training/{job}/**/run_completed.json')
        contract=unique(f'training/{job}/**/finetune_actual_contract.json')
        trained=bool(completed and completed.get('training_completed'))
        if comparison is None: evaluation_status='미완료'
        elif comparison.get('all_gates_pass'): evaluation_status='seed42 후보 통과 · 채택 미확정'
        else: evaluation_status='후보 전체 탈락'
        training[job]=dict(training_status='완료' if trained else '미완료',evaluation_status=evaluation_status,
            checkpoints=(completed or {}).get('pilot_checkpoints',[]),
            nonfinite_steps=(completed or {}).get('nonfinite_steps'),
            rated_load_weight=(contract or {}).get('actual_rated_load_weight'),
            actual_targeted_scope=(contract or {}).get('actual_targeted_scope'),
            source_contract_available=contract is not None)

    # Display a selector's existing output; do not choose/re-rank or imply adoption.
    selection=(targeted or {}).get('selected') if (targeted or {}).get('all_gates_pass') else None
    displayed=dict(label='baseline339',scope='현재 기준 정책339',terrains={})
    if selection:
        item=targeted['candidates'][selection]
        displayed=dict(label=selection,scope='seed42 통과 후보 · 채택 미확정',terrains={
            'flat':item['flat']['candidate'],'rough':item['rough']['candidate']})
    else:
        for terrain in ('flat','rough'):
            values=(old or {}).get('runs',{}).get(f'baseline339_{terrain}_seed42')
            if values is not None: displayed['terrains'][terrain]=values
    partial=[]
    if targeted is None:
        for path in sorted((root/'comparison/600').glob('*_flat_gate.json')):
            partial.append(read(path))
    targeted_outcomes={}
    for label,item in (targeted or {}).get('candidates',{}).items():
        flat=item.get('flat',{});gate=flat.get('gate',{});metrics=flat.get('candidate',{})
        rms_ratio=gate.get('mean_joint_rms_ratio');power_ratio=gate.get('absolute_power_ratio')
        targeted_outcomes[label]=dict(status=item.get('status'),failed_gates=[name for name,passed in gate.get('gates',{}).items() if passed is False],mean_joint_rms_change_percent=100*(rms_ratio-1) if rms_ratio is not None else None,absolute_power_change_percent=100*(power_ratio-1) if power_ratio is not None else None,command_speed_mae_m_s=metrics.get('command_speed_mae_m_s'),baseline_command_speed_mae_m_s=flat.get('baseline',{}).get('command_speed_mae_m_s'),rough_evaluated=isinstance(item.get('rough'),dict))
    static_cases=[]
    for pose in (static or {}).get('poses',[]):
        if (pose['height_m'],pose['forward_shift_m']) in ((.464,0.),(.464,.03),(.364,0.),(.364,.12),(.394,.12),(.424,.09)):
            static_cases.append(pose)
    optimization=read(root/'optimization_20261001/summary.json')
    release=read(root/'release_manifest.json')
    return dict(schema=1,generated_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        torque_optimization=optimization,release=release,
        adoption_confirmed=False,display_policy=displayed,training=training,integration=integration,
        latest_integration=dict(job=latest_job,status=integration[latest_job]['status']),
        latest_full_sequence=latest_full_sequence,latest_stairs=latest_stairs,standing_torque_comparisons=standing_comparisons,
        targeted_outcomes=targeted_outcomes,initial_comparison=old,targeted_comparison=dict(status='미완료' if targeted is None else '완료',
            existing_selection=selection,data=targeted,partial_flat_results=partial),
        collision_incident=incident,reconstructed_collision=reconstructed or [],motor_specs=motors,
        static_posture=dict(physics_simulation_performed=(static or {}).get('physics_simulation_performed'),
            model_sha256=(static or {}).get('model_sha256'),cases=static_cases,
            hardware_angular_limits_verified=(static or {}).get('hardware_angular_limits_verified'),
            integration599_status=integration['599']['status']),
        limitations=['전류는 |토크|/(Kt×감속비), 효율1 가정의 이상적 환산이며 배터리 전류가 아니다.',
            '601은 lockstep 평지 시퀀스 검증이다. 정책 틱 계산 P99 약 82.82 ms로 50 Hz 실시간 실기 운전은 검증되지 않았다.',
            '601 최종 crouch의 네 무릎 RMS 20.44/24.24/28.44/25.49 Nm는 모두 정격 18 Nm를 넘는다. 낮은 자세를 장시간 유지할 열 안전성은 미검증이다.',
            '보행 토크는 순간 implicit PD 추정값이다. 영게인 기립은 feedforward 명령을 사용하며 PhysX 구동력 실측이 아니다.',
            '정격 8.3/18 Nm와 시뮬레이션 상한 24.8/48 Nm를 구분한다. 열 안전성·지속 운전은 미검증이다.',
            '실제 접촉 채널과 기하학적 캡슐/몸통 프록시는 별도 증거다. 필터된 쌍의 무충돌을 증명하지 않는다.',
            '현재 비교는 seed42 평지·거친 지형이다. 생략된 거친 지형은 통과가 아니며 계단 검증도 아니다.',
            '이 보고서는 기존 JSON 판정을 표시하며 후보 선택이나 채택 게이트를 변경하지 않는다.'],source_sha256=sources)


def render_html(summary,root):
    root=Path(root);esc=html.escape
    def number(value,digits=2): return '—' if value is None else f'{value:.{digits}f}'
    def link(path,label):
        return f'<a href="{esc(path,quote=True)}">{esc(label)}</a>' if (root/path).exists() else esc(label)+' (미완료)'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    blocks=['<h1>V15.1 ROS2 검증 기록</h1>',f'<p class="muted">생성: {esc(summary["generated_at"])} · '+link('summary.json','기계 판독용 요약')+' · '+link('release_manifest.json','검증 설정과 소스 해시')+'</p>']
    optimization=summary.get('torque_optimization')
    if optimization:
        blocks.append('<h2>추가 토크 최적화 — 601 기준</h2><p>'+link('optimization_20261001/report.html','전체 관절·단계 비교와 고정 채택 기준')+' · '+link('optimization_20261001/attempts.json','통신 실패를 포함한 실행 기록')+'</p>')
        blocks.append(table(['후보','최적화 기준','미달 항목'],[
            [esc(c['name']),'통과' if c['comparison']['accepted'] else '탈락',esc(', '.join(c['comparison']['failed_gates']) or '없음')]
            for c in optimization['candidates']]))
        release=summary.get('release') or {}
        blocks.append('<p>현재 검증 설정: '+esc(str(release.get('parameters',{})))+' · 근거 '+esc(release.get('validated_flat_run','미기록'))+'. 동작 성공과 토크 최적화 채택은 별도로 판정합니다.</p>')
        selected_run=Path(release.get('validated_flat_run','')).parent.name
        chart=f'optimization_20261001/knees_{selected_run}.png'
        if selected_run.isdigit() and (root/chart).exists():
            blocks.append('<img class="incident" src="'+esc(chart,quote=True)+'" alt="채택 후보와 601의 단계별 무릎 RMS 비교">')
    pending=[job for job,run in summary['integration'].items() if run['status']=='미완료']
    if summary['targeted_comparison']['status']=='미완료': pending.append('600')
    blocks.append('<p class="notice">'+('작업 '+', '.join(pending)+' 결과는 미완료입니다. ' if pending else '')+
        '학습 완료와 동작 성공은 별도 판정입니다. 현재 표시 정책: '+esc(summary['display_policy']['scope'])+'.</p>')
    blocks.append('<h2>학습과 후보 비교</h2>')
    blocks.append(table(['학습 작업','학습','비교 판정','체크포인트 업데이트','정격 부하 계수'],[
        [job,t['training_status'],t['evaluation_status'],', '.join(str(c['updates']) for c in t['checkpoints']),number(t['rated_load_weight'],3)]
        for job,t in summary['training'].items()]))
    rows=[]
    old=summary.get('initial_comparison') or {}
    for key,metrics in old.get('runs',{}).items():
        label,terrain=key.rsplit('_seed',1)[0].rsplit('_',1)
        gate=old.get('comparisons',{}).get(label,{}).get(f'{terrain}_seed42',{})
        verdict='기준' if label=='baseline339' else ('통과' if gate.get('all_gates_pass') else '탈락')
        rows.append([link(f'comparison/594/{key}/result.json',label)+' · '+link(f'comparison/594/{key}/result.npz','NPZ'),terrain,verdict,
            number(metrics.get('progress_world_y_m')),number(metrics.get('command_speed_mae_m_s'),3),
            number(metrics.get('mean_joint_torque_rms_nm')),number(metrics.get('mean_absolute_power_W')),
            number(metrics.get('maximum_crossleg_contact_force_N')),
            number(None if metrics.get('minimum_interleg_gap_m') is None else metrics['minimum_interleg_gap_m']*1000)])
    new=summary['targeted_comparison']['data']
    if new:
        for label,item in new.get('candidates',{}).items():
            for terrain in ('flat','rough'):
                record=item.get(terrain)
                if not isinstance(record,dict):
                    rows.append([esc(label),terrain,'생략: 평지 탈락','—','—','—','—','—','—']);continue
                m=record.get('candidate',{});gate=record.get('gate',{})
                rows.append([link(f'comparison/600/{label}_{terrain}_seed42/result.json',label)+' · '+link(f'comparison/600/{label}_{terrain}_seed42/result.npz','NPZ'),terrain,
                    '통과' if gate.get('all_gates_pass') else '탈락',number(m.get('progress_world_y_m')),
                    number(m.get('command_speed_mae_m_s'),3),number(m.get('mean_joint_torque_rms_nm')),
                    number(m.get('mean_absolute_power_W')),number(m.get('maximum_crossleg_contact_force_N')),
                    number(None if m.get('minimum_interleg_gap_m') is None else m['minimum_interleg_gap_m']*1000)])
    else:
        for record in summary['targeted_comparison']['partial_flat_results']:
            m=record['candidate'];gate=record['gate'];label=record['label']
            rows.append([link(f'comparison/600/{label}_flat_gate.json',label),'flat',
                '평지 통과·전체 미완료' if gate.get('all_gates_pass') else '평지 탈락',
                number(m.get('progress_world_y_m')),number(m.get('command_speed_mae_m_s'),3),
                number(m.get('mean_joint_torque_rms_nm')),number(m.get('mean_absolute_power_W')),
                number(m.get('maximum_crossleg_contact_force_N')),
                number(None if m.get('minimum_interleg_gap_m') is None else m['minimum_interleg_gap_m']*1000)])
    blocks.append(table(['정책','지형','기존 판정','+y 진행 m','속도 MAE m/s','평균 관절 RMS Nm','절대 파워 W','접촉 최대 N','캡슐 간격 mm'],rows))
    if summary['targeted_outcomes']:
        blocks.append('<h3>598 학습 → 600 평가: 평지 탈락 사유</h3>')
        blocks.append(table(['후보','평균 RMS 변화 %','절대 파워 변화 %','속도 MAE m/s','기준 MAE m/s','실패 게이트','거친 지형 평가'],[
            [esc(label),number(item['mean_joint_rms_change_percent']),number(item['absolute_power_change_percent']),number(item['command_speed_mae_m_s'],6),number(item['baseline_command_speed_mae_m_s'],6),esc(', '.join(item['failed_gates']) or '없음'),'실행' if item['rough_evaluated'] else '생략'] for label,item in summary['targeted_outcomes'].items()]))
        blocks.append('<p>속도 MAE 허용 증가는 고정 기준 +0.02 m/s입니다. 토크·파워 감소만으로 채택하지 않습니다. 평지 탈락 후보의 거친 지형과 추가 시드 평가는 생략되었으며, 검증 통과를 뜻하지 않습니다. '+link('training/outcomes.json','학습 결과 문서 JSON')+'</p>')
    blocks.append('<p>'+link('comparison/594/comparison_summary.json','589 후보의 전체 비교 JSON')+' · '+link('comparison/600/comparison_summary.json','598 후보의 전체 비교 JSON')+'</p>')

    blocks.append('<h2>확인된 자기충돌</h2>')
    incident=summary.get('collision_incident')
    if incident:
        blocks.append('<p class="danger">'+esc(incident.get('candidate',''))+' / '+esc(incident.get('trial',''))+
            ': 접촉 최대 <strong>'+number(incident.get('max_force_N'))+' N</strong>, 최소 캡슐 간격 <strong>'+
            number(incident.get('minimum_capsule_gap_m',0)*1000)+' mm</strong>. 1 N 초과 최초 시점 '+
            number(incident.get('first_contact_over_1N_s'))+' s. 무충돌로 판정할 수 없습니다.</p>')
        blocks.append('<p>최소 간격 쌍: '+esc(' ↔ '.join(incident.get('minimum_pair',[])))+
            '. 접촉력 기록은 몸체별 상대쌍 최대값이므로 정확한 힘 상대쌍은 미기록입니다.</p>')
        if (root/'comparison/594/collision_incident.png').exists():
            blocks.append('<a href="comparison/594/collision_incident.png"><img class="incident" src="comparison/594/collision_incident.png" alt="후보500 거친 지형 충돌 시점과 접촉력·간격"></a>')
        blocks.append('<p>'+link('comparison/594/collision_incident.json','충돌 사건 JSON')+' · '+link('comparison/594/candidate500_rough_seed42/result.npz','실패 전 원시 기록')+'</p>')
    else: blocks.append('<p>충돌 사건 JSON 없음: 미완료.</p>')
    if summary['reconstructed_collision']:
        blocks.append('<h3>같은 다리 허벅지–발: 기록에서 복원한 기하학 프록시</h3><p>실제 접촉 측정이 아닙니다. 이 쌍들은 물리 접촉이 필터되어 있으며, 양의 복원 간격이 다른 다리 사이의 실제 충돌을 무효화하지 않습니다.</p>')
        reconstructed_rows=[]
        for item in summary['reconstructed_collision']:
            source=Path(item.get('original_input',''))
            label=source.parent.name
            if source.parent.parent.name=='integration': label='통합 '+label
            validation=item.get('validation',{})
            errors=[v['max_absolute_error_m'] for v in validation.values() if 'max_absolute_error_m' in v]
            reconstructed_rows.append([esc(label),number(item.get('minimum_same_leg_thigh_foot_gap_m',0)*1000),
                '복원 검증됨' if item.get('reconstruction_validated') else '미검증',
                number(max(errors)*1e6 if errors else None,3),'아니오' if item.get('actual_same_leg_contact_measured') is False else '미기록'])
        blocks.append(table(['기록','최소 프록시 간격 mm','모델 복원','검증 최대 좌표/간격 오차 µm','실제 같은 다리 접촉 측정'],reconstructed_rows))
        blocks.append('<p>'+link('analysis/reconstructed_collision/summary.json','복원 검증·쌍별 간격 전체 JSON')+'</p>')

    blocks.append('<h2>전체 동작 시퀀스와 단독 보행 평가</h2>')
    blocks.append(table(['작업','평가 범위','과제 판정','전체 시간 완료','기록 s','실패 또는 제한','자료'],[
        [job,esc({'full_sequence':'전체 시퀀스','standalone':'단독 보행'}.get(run.get('run_kind'),'범위 미기록'))+' · '+esc(run.get('scenario') or '—'),run['status'],'예' if run.get('completed') is True else ('아니오' if run.get('completed') is False else '미완료'),
         number(run.get('seconds')),esc(run.get('failure') or ('계단 상승·통과 사후 기준 미달' if run.get('stair_posthoc') and run['status']=='실패' else ('요구 상태 전환 미완료' if run['status']=='실패' else '—'))),
         link(f'analysis/integration_reports/integration_{job}.html','상세 보고서')+' · '+link(f'integration/{job}/result.json','원본 JSON')]
        for job,run in summary['integration'].items()]))

    for job,run in summary['integration'].items():
        stair=run.get('stair_posthoc')
        if stair is None: continue
        rows=stair.get('path',{}).get('per_environment',[])
        progress=rows[0].get('final_directed_progress_m') if rows else None
        blocks.append('<h3>계단 '+esc(job)+': 실제 상승·통과 사후 판정 '+esc(run['status'])+'</h3>')
        blocks.append('<p class="notice">원시 task_success='+esc(str(run.get('raw_task_success'))).lower()+'는 시간 완료와 충돌 검사만 반영합니다. 계단 상승 성공을 뜻하지 않습니다. 원본은 보존했고, 별도의 기존 방향성·전체 계단 통과 기준으로 판정했습니다.</p>')
        blocks.append(table(['방향성 계단 과제','전체 계단 통과','진행 거리 m','최종 지형 상승 m','전체 시간 완료'],[[
            '성공' if stair.get('posthoc_directed_stair_task_success') else '실패',
            '성공' if stair.get('posthoc_full_flight_task_success') else '실패',number(progress,3),
            number(stair.get('final_geometry_derived_ground_rise_m'),3),'예' if run.get('completed') else '아니오']]))
        blocks.append('<p>지형 상승량은 설정된 메시와 몸통 XY에서 복원한 기하학 값이며 발 접촉 높이 실측은 아닙니다. '+link(f'analysis/stair_{job}.json','사후 계단 분석 JSON')+'</p>')

    collision_rows=[]
    for job,run in summary['integration'].items():
        whole=run.get('stage_summaries',{}).get('ALL',{})
        collision=whole.get('collision',{});same=whole.get('same_leg_thigh_foot_proxy',{})
        if not collision: continue
        gap=collision.get('minimum_crossleg_capsule_gap_m');same_gap=same.get('minimum_gap_m',same.get('minimum_same_leg_thigh_foot_gap_m'))
        collision_rows.append([job,number(collision.get('maximum_crossleg_contact_force_n')),number(gap*1000 if gap is not None else None),number(same_gap*1000 if same_gap is not None else None),'기하학 프록시 · 실제 접촉 미측정'])
    blocks.append(table(['통합 작업','다른 다리 실제 접촉 최대 N','다른 다리 최소 프록시 mm','같은 다리 허벅지–발 최소 mm','같은 다리 측정 범위'],collision_rows))

    for key,label in (('latest_full_sequence','최신 전체 시퀀스'),('latest_stairs','최신 단독 계단 보행')):
        latest=summary[key]
        blocks.append('<p>'+label+' '+esc(latest['job'] or '기록 없음')+': <strong>'+esc(latest['status'])+'</strong>.</p>')
    latest=summary['latest_integration']
    blocks.append('<p class="muted">최근 기록 작업 번호: '+esc(latest['job'])+' (서로 다른 평가 범위를 통합 판정하지 않음).</p>')
    blocks.append('<h3>첫 번째 연속 STAND: 물리 시뮬레이션 중 기립 토크</h3><p>후반의 복귀 STAND를 섞지 않고 초기 기립 구간만 비교합니다. 이 구간의 토크 감소와 전체 동작 성공은 별도 결과입니다. 전류는 이상적 모터 전류 환산입니다.</p>')
    for job,comparison in summary['standing_torque_comparisons'].items():
        if not comparison['joints']: continue
        title='593 → '+job+' · '+number(comparison['candidate_start_s'])+'–'+number(comparison['candidate_end_s'])+' s'
        verified=comparison['both_native_effort_verified']
        content='<p>'+('두 기록 모두 구동 PD 게인0, 기록 토크=feedforward 명령을 원시 배열에서 확인했습니다. 물리 시뮬레이션에서 사용된 native effort 명령이며 구동력 센서 실측은 아닙니다.' if verified else 'native effort 원시 배열 확인이 완료되지 않았습니다.')+' 전체 과제: '+esc(summary['integration'][job]['status'])+'.</p>'
        rows=[]
        for item in comparison['joints']:
            a=item['baseline'];b=item['candidate']
            rows.append([esc(item['joint']),number(a.get('torque_rms_nm')),number(b.get('torque_rms_nm')),number(item.get('rms_change_percent'),1),number(b.get('torque_peak_nm')),number(b.get('ideal_current_rms_a')),number(b.get('ideal_current_peak_a')),number(b.get('rated_torque_nm'),1)])
        if job==summary['latest_full_sequence']['job'] and verified:
            rear=[item for item in comparison['joints'] if item['joint'] in ('RL_KFE','RR_KFE')]
            knees=[item['candidate']['torque_rms_nm'] for item in comparison['joints'] if item['joint'].endswith('KFE')]
            content+='<p class="notice">초기 기립에서 뒤 무릎 RMS 감소: '+', '.join(esc(item['joint'])+' '+number(-item['rms_change_percent'],1)+'%' for item in rear)+'. 무릎 최대 RMS '+number(max(knees) if knees else None)+' Nm. 이 구간의 개선은 확인되었지만 전체 동작 과제는 '+esc(summary['integration'][job]['status'])+'입니다.</p>'
        content+=table(['관절','593 RMS Nm',job+' RMS Nm','RMS 변화 %',job+' 최대 Nm',job+' 전류 RMS A',job+' 전류 최대 A','정격 Nm'],rows)
        if job==summary['latest_full_sequence']['job']: blocks.append('<h3>'+esc(title)+'</h3>'+content)
        else: blocks.append('<details><summary>'+esc(title)+'</summary>'+content+'</details>')

    blocks.append('<h2>관절별 토크·이상적 전류</h2><p>'+esc(summary['display_policy']['scope'])+
        ' · 전류=|토크|/(Kt×감속비), 효율1. 배터리 전류가 아닙니다.</p>')
    specs=summary.get('motor_specs') or {}
    for terrain,m in summary['display_policy']['terrains'].items():
        rms=m.get('torque_rms_nm',[]);peak=m.get('torque_peak_nm',[]);joint_rows=[]
        for i,name in enumerate(JOINTS):
            spec=specs.get('HAA' if name.endswith('HAA') else 'HFE_KFE',{})
            factor=spec.get('motor_kt_nm_per_a',0)*spec.get('gear_ratio',0)
            rated=spec.get('rated_torque_nm');cap=spec.get('sim_limit_nm')
            torque=rms[i] if i<len(rms) else None;pmax=peak[i] if i<len(peak) else None
            display=number(torque)
            if torque is not None and rated is not None and torque>rated: display='<strong class="danger">'+display+'</strong>'
            joint_rows.append([name,number(rated,1),number(cap,1),display,number(pmax),
                number(torque/factor if torque is not None and factor else None),number(pmax/factor if pmax is not None and factor else None)])
        blocks.append('<h3>'+('평지' if terrain=='flat' else '거친 지형')+'</h3>'+table(
            ['관절','정격 Nm','시뮬 상한 Nm','토크 RMS Nm','토크 최대 Nm','전류 RMS A','전류 최대 A'],joint_rows))
    blocks.append('<p class="muted">빨간 RMS는 정격 토크 초과입니다. 시뮬레이션 상한 이하는 연속 운전 또는 열 안전성의 증거가 아닙니다.</p>')

    blocks.append('<h2>기립 자세의 정적 모델 탐색</h2>')
    blocks.append('<p>네 발 접촉점을 고정한 동역학 계산입니다. 모델 예측을 실제 토크 개선으로 단정하지 않습니다. 실제 적용 결과는 위의 초기 STAND 비교 및 최신 전체 시퀀스 '+esc(summary['latest_full_sequence']['job'] or '미기록')+' ('+esc(summary['latest_full_sequence']['status'])+')에서 별도로 확인합니다.</p>')
    blocks.append(table(['몸통 높이 m','앞쪽 이동 m','최대 정격 비율','뒤 왼쪽 KFE Nm','뒤 오른쪽 KFE Nm','최소 다리 간격 mm'],[
        [number(p['height_m'],3),number(p['forward_shift_m'],3),number(p.get('wbc_max_rated_fraction'),3),
         number(p.get('wbc_torque_nm',[None]*12)[8]),number(p.get('wbc_torque_nm',[None]*12)[11]),
         number(p.get('minimum_interleg_capsule_gap_m',0)*1000)] for p in summary['static_posture']['cases']]))
    blocks.append('<p>'+link('static_pose_search.json','44개 자세 수치')+' · '+link('analysis/static_pose_search.py','재현 스크립트')+
        '. 모델의 관절 각도 한도는 무한대이며 하드웨어 각도 한도는 확인되지 않았습니다.</p>')
    blocks.append('<h2>측정 범위와 연결 자료</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in summary['limitations'])+'</ul>')
    blocks.append('<p>'+link('../../ros2/v151_ros2_control/README.md','ROS2 제어·메시지 계약')+' · '+link('../../docs/v151_plan.md','개발·검증 기준')+' · '+link('training/targeted_preparation.json','다음 학습 계약')+' · '+link('analysis/summary_report.py','보고서 생성기')+'</p>')
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V15.1 ROS2 검증 기록</title><style>body{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#172033;background:#f8fafc;line-height:1.6}h1,h2,h3{line-height:1.3}h2{margin-top:36px;border-bottom:2px solid #dbe3ed;padding-bottom:8px}a{color:#075eb5}.muted{color:#526071;font-size:.9rem}.notice{background:#e9f1fc;padding:16px;border-left:4px solid #3979b5}.danger{color:#a11919}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:.88rem}th,td{border-bottom:1px solid #dbe3ed;padding:8px 10px;text-align:right;white-space:nowrap}th{background:#e8edf4}td:first-child,th:first-child{text-align:left}td:last-child{white-space:normal}img.incident{max-width:100%;background:white}li{margin:6px 0}</style><body>'+''.join(blocks)+'</body></html>'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();summary=build_summary(args.root)
    outcome_path=args.root/'training/outcomes.json'
    outcome_path.parent.mkdir(parents=True,exist_ok=True)
    outcome_path.write_text(json.dumps(dict(generated_at=summary['generated_at'],training=summary['training'],targeted_candidates=summary['targeted_outcomes'],adoption_confirmed=summary['adoption_confirmed'],display_policy=summary['display_policy']['label'],comparison600_status=summary['targeted_comparison']['status'],latest_full_sequence=summary['latest_full_sequence'],latest_stairs=summary['latest_stairs'],source_sha256=summary['source_sha256']),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.root/'report.html').write_text(render_html(summary,args.root))
    print(json.dumps(dict(report=str(args.root/'report.html'),display_policy=summary['display_policy']['label'],
        latest_full_sequence=summary['latest_full_sequence'],latest_stairs=summary['latest_stairs'],comparison600=summary['targeted_comparison']['status']),ensure_ascii=False))
