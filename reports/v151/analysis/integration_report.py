"""Offline V15.1 integration evidence; requires NumPy, optional matplotlib.

python3 integration_report.py --sourcepath ../integration/593 --outputdir report
No simulator, ROS, or network calls. Input artifacts are never modified.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import numpy as np

CANONICAL=tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))
# Getter evidence uses the bridge's reviewed policy/PhysX joint order.
POLICY=('FL_HAA','FR_HAA','FL_HFE','FR_HFE','RR_HAA','RL_HAA','RR_HFE','RL_HFE','FL_KFE','FR_KFE','RR_KFE','RL_KFE')
RATED=np.tile([8.3,18.,18.],4)
LIMITS=np.tile([24.8,48.,48.],4)
NM_PER_A=np.tile([.123*10,.16*9,.16*9],4)
DT=.005
SAME_LEG_PAIRS=tuple(f'{leg}_thigh_profile--{leg}_foot' for leg in ('FL','FR','RL','RR'))


def longest(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return float(lengths.max(initial=0)*DT)


def checks(arrays, metadata, statuses):
    result={}
    for kind in ('drive_gain_getters','drive_limit_getters','drive_gain_arrays','drive_budget_arrays'):
        result[kind]=dict(available=False,passed=None)
    gains=metadata.get('drive_gain_checks',[])
    by_stamp={s['stamp_ns']:s for s in statuses}
    if gains:
        errors=[]
        for row in gains:
            state=by_stamp.get(row['stamp_ns'],{})
            if 'stiffness_nm_rad' not in state or 'damping_nms_rad' not in state:
                errors.append(float('inf'));continue
            errors.extend([float(np.max(np.abs(np.asarray(row[k])-state[v]))) for k,v in [('kp','stiffness_nm_rad'),('kd','damping_nms_rad')]])
        worst=max(errors)
        result['drive_gain_getters']=dict(available=True,count=len(gains),passed=bool(worst<1e-4),max_error=worst if np.isfinite(worst) else None)
    caps=metadata.get('drive_limit_checks',[])
    if caps:
        errors=[];budget=[]
        limits=np.array([24.8 if n.endswith('HAA') else 48. for n in POLICY])
        for row in caps:
            actual=np.asarray(row['drive_limit'],float).reshape(12)
            ff=np.asarray(row['feedforward'],float).reshape(12)
            errors.append(float(np.max(abs(actual-np.maximum(0.,limits-abs(ff))))))
            budget.append(float(np.max(actual+abs(ff)-limits)))
        result['drive_limit_getters']=dict(available=True,count=len(caps),passed=bool(max(errors)<1e-4 and max(budget)<1e-4),max_error_nm=max(errors),max_budget_excess_nm=max(0.,max(budget)),joint_order=list(POLICY))
    if all(k in arrays for k in ('drive_kp','drive_kd')) and all('stiffness_nm_rad' in s and 'damping_nms_rad' in s for s in statuses):
        error=max(float(np.max(abs(arrays[k]-np.array([s[v] for s in statuses])[:,None]))) for k,v in [('drive_kp','stiffness_nm_rad'),('drive_kd','damping_nms_rad')])
        result['drive_gain_arrays']=dict(available=True,passed=bool(error<1e-4),max_error=error,semantics='IsaacLab state buffers; getter evidence listed separately')
    if all(k in arrays for k in ('drive_limit','feedforward')):
        excess=arrays['drive_limit']+abs(arrays['feedforward'])-LIMITS
        worst=float(excess.max());minimum=float(arrays['drive_limit'].min())
        result['drive_budget_arrays']=dict(available=True,passed=bool(worst<1e-4 and minimum>=-1e-6),max_budget_excess_nm=max(0.,worst),minimum_drive_limit_nm=minimum,semantics='IsaacLab state buffers; getter evidence listed separately')
    result['command_schema_versions']=sorted(set(s.get('command_schema',1) for s in statuses))
    return result


def interpret_stair_task(result, source):
    """Keep the survival flag separate from hash-matched posthoc stair evidence."""
    if 'stair' not in str(result['scenario']).lower():
        return
    path=source.parent.parent/'analysis'/f'stair_{source.name}.json'
    assessment=dict(available=path.is_file(),source=str(path),source_sha256=None,
        provenance_matches=False,limitations=[],directed_stair_task_success=None,
        full_flight_task_success=None)
    result['stair_assessment']=assessment
    result['task_success']=None
    result['task_verdict']='unverified'
    result['task_interpretation']='계단 완주 미검증: 원 평가기의 성공 플래그는 실행 종료와 충돌 검사만 확인합니다.'
    if not path.is_file():
        assessment['limitations'].append('Missing posthoc directed-stair/full-flight assessment.')
        return
    evidence=json.loads(path.read_text())
    assessment.update(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        limitations=evidence.get('limitations',[]),
        directed_stair_task_success=evidence.get('posthoc_directed_stair_task_success'),
        full_flight_task_success=evidence.get('posthoc_full_flight_task_success'),
        final_geometry_derived_ground_rise_m=evidence.get('final_geometry_derived_ground_rise_m'),
        ground_height_source=evidence.get('ground_height_source'),
        center_provenance=evidence.get('center_provenance'))
    hashes=evidence.get('input_sha256',{})
    assessment['provenance_matches']=all(
        [digest for name,digest in hashes.items() if Path(name).name==filename]==[result['source_sha256'][kind]]
        for filename,kind in [('result.json','metadata'),('result.npz','arrays')])
    verdicts=[assessment['directed_stair_task_success'],assessment['full_flight_task_success']]
    if not assessment['provenance_matches'] or not all(type(v) is bool for v in verdicts):
        assessment['limitations'].append('Missing verdict or raw-input SHA256 mismatch; posthoc result cannot establish this run outcome.')
        return
    result['task_success']=bool(result['completed'] and result['data_valid'] and all(verdicts))
    result['task_verdict']='passed' if result['task_success'] else 'failed'
    result['task_interpretation']='사후 계단 방향 진행 및 전체 계단 완주 기준을 모두 적용한 판정입니다. 원 평가기 플래그는 별도로 보존합니다.'


def summarize(source):
    source=Path(source).resolve()
    if source.is_file(): source=source.parent
    paths={k:source/f'result.{suffix}' for k,suffix in [('metadata','json'),('status','status.json'),('arrays','npz')]}
    metadata=json.loads(paths['metadata'].read_text());statuses=json.loads(paths['status'].read_text())
    with np.load(paths['arrays'],allow_pickle=False) as archive: arrays={k:np.asarray(archive[k],dtype=float) for k in archive.files}
    required=('estimated_torque','joint_velocity','position','velocity','quaternion','gaps','trunk_gaps','contact_pair_force')
    if any(k not in arrays for k in required): raise ValueError('Missing required raw arrays')
    count=len(arrays['position'])
    if not count or len(statuses)!=count or any(len(v)!=count for v in arrays.values()): raise ValueError('Empty/mismatched raw samples and statuses')
    stamps=np.array([s['stamp_ns'] for s in statuses])
    if not np.array_equal(stamps,np.arange(count,dtype=np.int64)*5_000_000): raise ValueError('Expected contiguous 200 Hz lockstep stamps beginning at zero')
    order=metadata.get('joint_order')
    if order is None or len(order)!=12 or set(order)!=set(CANONICAL): raise ValueError('Missing/invalid declared joint order')
    indices=[order.index(n) for n in CANONICAL]
    for k in ('estimated_torque','isaac_estimated_torque','joint_position','joint_velocity','drive_kp','drive_kd','drive_limit','feedforward','position_target'):
        if k in arrays:
            if arrays[k].shape!=(count,12): raise ValueError(f'{k}: expected 12 joint columns')
            arrays[k]=arrays[k][:,indices]
    finite=np.logical_and.reduce([np.isfinite(v).reshape(count,-1).all(axis=1) for v in arrays.values()])
    finite &= np.linalg.norm(arrays['quaternion'],axis=1)>1e-12
    failures=np.flatnonzero(~finite | np.array([s['mode']=='FAULT' for s in statuses]))
    used=int(failures[0]) if len(failures) else count
    if not used: raise ValueError('No valid samples before first failure')
    arrays={k:v[:used] for k,v in arrays.items()};statuses=statuses[:used]
    shifts=[float(s['stand_forward_shift_m']) for s in statuses if 'stand_forward_shift_m' in s]
    if not np.isfinite(shifts).all(): raise ValueError('Nonfinite standing forward shift')
    shift_values=sorted(set(shifts))
    pair_order=metadata.get('same_leg_thigh_foot_pair_order',list(SAME_LEG_PAIRS))
    if 'same_leg_thigh_foot_gap_m' in arrays:
        if arrays['same_leg_thigh_foot_gap_m'].shape!=(used,4) or len(pair_order)!=4 or set(pair_order)!=set(SAME_LEG_PAIRS):
            raise ValueError('Expected four identified same-leg thigh-foot proxy columns')
    modes=np.array([s['mode'] for s in statuses]);tau=abs(arrays['estimated_torque'])
    quat=arrays['quaternion']/np.linalg.norm(arrays['quaternion'],axis=1)[:,None]
    tilt=np.rad2deg(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
    result=dict(report_schema=1,source=str(source),run=source.name,scenario=metadata.get('scenario'),seed=metadata.get('seed'),
        raw_task_success=metadata.get('task_success'),
        task_success=bool(metadata.get('task_success',False) and not len(failures)),completed=bool(metadata.get('completed',False)),failure=metadata.get('failure'),
        data_valid=not bool(len(failures)),first_excluded_sample=used if len(failures) else None,samples_recorded=count,samples_used=used,
        seconds_used=used*DT,joint_order=list(CANONICAL),torque_semantics=metadata.get('torque_semantics','unspecified estimate'),
        stand_forward_shift_m=shift_values[0] if len(shift_values)==1 else None,
        stand_forward_shift_values_m=shift_values,stand_forward_shift_recorded_fraction=len(shifts)/used,
        same_leg_thigh_foot_evidence=dict(geometry=metadata.get('same_leg_thigh_foot_geometry'),pair_order=pair_order,
            gate_applied=metadata.get('same_leg_thigh_foot_gate_applied'),physical_pairs_filtered=True,
            source_summary={key:metadata[key] for key in ('min_same_leg_thigh_foot_gap_m','same_leg_thigh_foot_proxy_overlap_fraction','same_leg_thigh_foot_per_pair_minimum_m') if key in metadata}),
        current_model=dict(formula='abs(torque)/(Kt*gear), efficiency=1',nm_per_a=NM_PER_A.tolist(),not_battery_current=True,thermal_safety_validated=False,
            specification_source='reports/motor_audit_20260930/specs/motors.json'),collision_scope=metadata.get('collision_scope','unspecified'),
        source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},modes={},timeline=[],
        actuation_checks=checks(arrays,metadata,statuses),lockstep_only=True)
    result['task_verdict']='passed' if result['task_success'] else 'failed'
    interpret_stair_task(result,source)
    def summarize_mask(mask):
        values=tau[mask];rms=np.sqrt(np.mean(values**2,axis=0));peak=values.max(axis=0);exceed=tau>RATED+1e-5
        joints=[]
        for j,name in enumerate(CANONICAL):
            joints.append(dict(joint=name,torque_rms_nm=float(rms[j]),torque_peak_nm=float(peak[j]),rated_torque_nm=float(RATED[j]),
                rated_exceed_fraction=float(exceed[mask,j].mean()),rated_exceed_s=float(exceed[mask,j].sum()*DT),
                rated_exceed_longest_s=longest(mask & exceed[:,j]),ideal_current_rms_a=float(rms[j]/NM_PER_A[j]),ideal_current_peak_a=float(peak[j]/NM_PER_A[j])))
        gaps=arrays['gaps'][mask];trunk=arrays['trunk_gaps'][mask];contact=arrays['contact_pair_force'][mask]
        selected=[s for s,on in zip(statuses,mask) if on];ticks=[s for s in selected if s['stamp_ns']%20_000_000==0]
        active=[s['wbc'] for s in ticks if s.get('wbc',{}).get('active',False)]
        compute=[s['compute_ms'] for s in ticks if 'compute_ms' in s]
        supports=[s['support_feet'] for s in selected if 'support_feet' in s]
        proxy=dict(available=False,minimum_gap_m=None,overlap_fraction=None,per_pair_minimum_m=None,
                   actual_contact_measured=False,physical_pairs_filtered=True)
        if 'same_leg_thigh_foot_gap_m' in arrays:
            same_leg=arrays['same_leg_thigh_foot_gap_m'][mask]
            proxy.update(available=True,minimum_gap_m=float(same_leg.min()),
                overlap_fraction=float((same_leg.min(axis=1)<0).mean()),
                per_pair_minimum_m=dict(zip(pair_order,same_leg.min(axis=0).tolist())))
        return dict(samples=int(mask.sum()),duration_s=float(mask.sum()*DT),joints=joints,
            same_leg_thigh_foot_proxy=proxy,
            height_mean_m=float(arrays['position'][mask,2].mean()),height_min_m=float(arrays['position'][mask,2].min()),
            speed_mean_m_s=float(np.linalg.norm(arrays['velocity'][mask],axis=1).mean()),
            speed_max_m_s=float(np.linalg.norm(arrays['velocity'][mask],axis=1).max()),maximum_tilt_deg=float(tilt[mask].max()),
            mean_absolute_mechanical_power_w=float(np.sum(tau[mask]*abs(arrays['joint_velocity'][mask]),axis=1).mean()),
            support_count_fraction={str(i):float(np.mean(np.array(supports)==i)) for i in range(5)} if supports else {},
            policy_compute_ms_p50_p99=np.quantile(compute,[.5,.99]).tolist() if compute else None,
            wbc=dict(active_policy_ticks=len(active),feasible_fraction=float(np.mean([w.get('feasible',False) for w in active])) if active else None,
                converged_fraction=float(np.mean([w.get('converged',False) for w in active])) if active else None),
            collision=dict(minimum_crossleg_capsule_gap_m=float(gaps.min()),crossleg_capsule_overlap_fraction=float((gaps.min(axis=1)<0).mean()),
                maximum_crossleg_contact_force_n=float(contact.max()),crossleg_contact_over_1n_fraction=float((contact.max(axis=1)>1.).mean()),
                minimum_trunk_proxy_gap_m=float(trunk.min()),trunk_proxy_overlap_fraction=float((trunk.min(axis=1)<0).mean()),
                trunk_actual_collider_present=False,scope='Trunk proxy overlap is geometric only; contact channels cover filtered cross-leg pairs, not all possible pairs'))
    for mode in ['ALL',*dict.fromkeys(modes)]:
        result['modes'][mode]=summarize_mask(np.ones(used,dtype=bool) if mode=='ALL' else modes==mode)
    changes=np.r_[0,np.flatnonzero(modes[1:]!=modes[:-1])+1,used]
    occurrences={}
    for index,(a,b) in enumerate(zip(changes[:-1],changes[1:]),1):
        mode=str(modes[a]);occurrences[mode]=occurrences.get(mode,0)+1
        mask=np.zeros(used,dtype=bool);mask[a:b]=True
        result['timeline'].append(dict(segment_index=index,mode=mode,mode_occurrence=occurrences[mode],
            start_s=float(a*DT),end_s=float(b*DT),**summarize_mask(mask)))
    return result


def plots(source, result):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError: return '<p>matplotlib 없음: 수치 표와 CSV를 참조하세요.</p>'
    with np.load(Path(source)/'result.npz') as a:
        count=result['samples_used'];t=np.arange(count)*DT
        fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
        ax[0].plot(t,a['position'][:count,2]);ax[0].set_ylabel('Root height (m)')
        ax[1].plot(t,np.linalg.norm(a['velocity'][:count],axis=1));ax[1].set_ylabel('Body speed (m/s)')
        ax[2].plot(t,np.max(abs(a['estimated_torque'][:count]),axis=1));ax[2].set_ylabel('Max |torque estimate| (Nm)');ax[2].set_xlabel('Simulation time (s)')
        for index,segment in enumerate(result['timeline']):
            for axes in ax: axes.axvspan(segment['start_s'],segment['end_s'],color='C'+str(index%10),alpha=.07);axes.grid(alpha=.2)
            ax[0].text((segment['start_s']+segment['end_s'])/2,1.02,segment['mode'],rotation=35,ha='left',fontsize=7,transform=ax[0].get_xaxis_transform())
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=140);plt.close(fig)
    return '<img alt="높이, 속도, 최대 추정 토크의 시간 변화" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'


def render_html(result, source):
    esc=html.escape;success=result['task_success'];verdict={'passed':'성공','failed':'실패','unverified':'미검증'}[result['task_verdict']]
    stair_text=''
    if 'stair_assessment' in result:
        assessment=result['stair_assessment']
        flag=lambda value:'통과' if value is True else '실패' if value is False else '미검증'
        limits=''.join(f'<li>{esc(str(v))}</li>' for v in assessment['limitations'])
        stair_text=f'<h2>사후 계단 판정</h2><p>{esc(result["task_interpretation"])} 원 task_success: {str(result["raw_task_success"]).lower()} · 방향 진행: {flag(assessment["directed_stair_task_success"])} · 전체 계단: {flag(assessment["full_flight_task_success"])}.</p><p>원자료 SHA256 일치: {assessment["provenance_matches"]} · 판정 자료: <code>{esc(assessment["source"])}</code> · SHA256: <code>{assessment["source_sha256"]}</code></p><ul>{limits}</ul>'
    shift_text=', '.join(f'{v:+.3f} m' for v in result['stand_forward_shift_values_m']) or '미기록'
    proxy_rows=[]
    rows=[]
    for mode,v in result['modes'].items():
        if mode=='ALL':continue
        c=v['collision'];j=v['joints']
        proxy=v['same_leg_thigh_foot_proxy']
        gap_text=f'{proxy["minimum_gap_m"]*1000:.1f}' if proxy['available'] else '미기록'
        overlap_text=f'{100*proxy["overlap_fraction"]:.1f}%' if proxy['available'] else '미기록'
        proxy_rows.append(f'<tr><td>{esc(mode)}</td><td>{gap_text}</td><td>{overlap_text}</td></tr>')
        rows.append(f'<tr><td>{esc(mode)}</td><td>{v["duration_s"]:.2f}</td><td>{v["height_mean_m"]:.3f}</td><td>{v["speed_mean_m_s"]:.3f}</td><td>{max(x["torque_rms_nm"] for x in j):.2f}</td><td>{max(x["ideal_current_rms_a"] for x in j):.2f}</td><td>{100*max(x["rated_exceed_fraction"] for x in j):.1f}%</td><td>{c["minimum_crossleg_capsule_gap_m"]*1000:.1f}</td><td>{c["maximum_crossleg_contact_force_n"]:.2f}</td><td>{100*c["trunk_proxy_overlap_fraction"]:.1f}%</td></tr>')
    checks_text=''.join(f'<li>{esc(k)}: '+('미기록' if not v.get('available') else '통과' if v.get('passed') else '불일치')+'</li>' for k,v in result['actuation_checks'].items() if isinstance(v,dict))
    details=[]
    for mode,v in result['modes'].items():
        jr=''.join(f'<tr><td>{j["joint"]}</td><td>{j["torque_rms_nm"]:.2f}</td><td>{j["torque_peak_nm"]:.2f}</td><td>{j["ideal_current_rms_a"]:.2f}</td><td>{j["ideal_current_peak_a"]:.2f}</td><td>{j["rated_exceed_s"]:.3f}</td><td>{j["rated_exceed_longest_s"]:.3f}</td></tr>' for j in v['joints'])
        details.append(f'<details><summary>{esc(mode)} 관절별 수치</summary><table><tr><th>관절</th><th>RMS Nm</th><th>최대 Nm</th><th>RMS A</th><th>최대 A</th><th>정격 초과 s</th><th>최장 연속 s</th></tr>{jr}</table></details>')
    segment_rows=[]
    for segment in result['timeline']:
        joints=segment['joints'];collision=segment['collision']
        segment_rows.append(f'<tr><td>{esc(segment["mode"])} #{segment["mode_occurrence"]}</td>'
            f'<td>{segment["start_s"]:.2f}–{segment["end_s"]:.2f}</td><td>{segment["duration_s"]:.2f}</td>'
            f'<td>{segment["height_min_m"]:.3f}</td><td>{segment["speed_mean_m_s"]:.3f} / {segment["speed_max_m_s"]:.3f}</td>'
            f'<td>{max(j["torque_rms_nm"] for j in joints):.2f}</td><td>{max(j["torque_peak_nm"] for j in joints):.2f}</td>'
            f'<td>{joints[8]["torque_rms_nm"]:.2f} / {joints[11]["torque_rms_nm"]:.2f}</td>'
            f'<td>{100*max(j["rated_exceed_fraction"] for j in joints):.1f}%</td>'
            f'<td>{collision["minimum_crossleg_capsule_gap_m"]*1000:.1f}</td><td>{collision["maximum_crossleg_contact_force_n"]:.2f}</td></tr>')
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V15.1 {esc(result['run'])} 통합 검증</title>
<style>body{{font:16px/1.6 sans-serif;max-width:1180px;margin:40px auto;padding:0 20px;color:#17212b}}h1{{font-size:28px}}.verdict{{color:{'#17683b' if success else '#b42318'};font-weight:bold}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#edf3f7}}img{{width:100%}}details{{margin:15px 0}}small{{color:#555}}code{{overflow-wrap:anywhere}}</style>
<h1>V15.1 실행 {esc(result['run'])} · <span class="verdict">전체 과제 {verdict}</span></h1>
<p>기록 {result['seconds_used']:.2f}초 · 실행 종료 {'정상' if result['completed'] else '오류'} · 원자료 {'유효' if result['data_valid'] else '첫 비정상 표본부터 제외'}.</p>
<p>기립 전방 이동 설정: <strong>{shift_text}</strong> · 상태 기록 비율 {100*result['stand_forward_shift_recorded_fraction']:.1f}%.</p>
<p>{esc(str(result.get('task_interpretation') or result['failure'] or ('요구 동작 순서 또는 최종 상태 기준 미달' if not success else '평가기가 요구한 순서와 최종 상태 기준 통과')))}</p>
{stair_text}
<p>토크는 직접 노력 명령 또는 implicit PD의 순간 추정값이며 PhysX 구동력 실측이 아닙니다. 이상 전류는 τ/(Kt×감속비), 효율 1 가정입니다. 배터리 전류·온도·연속 운전 안전성은 검증하지 않았습니다. 시뮬레이션 시간 lockstep 결과이며 실시간 제어 성능 증거가 아닙니다.</p>
{plots(source,result)}
<table><tr><th>모드</th><th>시간 s</th><th>높이 m</th><th>속도 m/s</th><th>최대 관절 RMS Nm</th><th>최대 RMS A</th><th>최대 정격 초과율</th><th>다리 간격 mm</th><th>접촉 최대 N</th><th>몸통 근사 겹침</th></tr>{''.join(rows)}</table>
<h2>연속 동작 구간별 결과</h2>
<p>같은 모드라도 중간에 다른 동작이 있으면 별도 구간입니다. STAND #1과 보행 뒤 STAND #2를 따로 비교하며, 모드 전체 평균으로 착지 후 관절 부하를 가리지 않습니다. 토크는 추정값입니다. 구간별 모든 관절·접촉·근사 형상 수치는 JSON과 segments / segment_joints CSV에 있습니다.</p>
<table><tr><th>구간</th><th>시작–끝 s</th><th>길이 s</th><th>최저 높이 m</th><th>속도 평균 / 최대 m/s</th><th>최대 관절 RMS Nm</th><th>최대 Nm</th><th>RL / RR 무릎 RMS Nm</th><th>최대 정격 초과율</th><th>다리 간격 mm</th><th>접촉 최대 N</th></tr>{''.join(segment_rows)}</table>
<p>다리 간격은 24개 교차 다리 캡슐 쌍, 접촉력은 기록된 필터 접촉 채널의 값입니다. 몸통 근사 형상 겹침은 실제 몸통 충돌이 아닙니다. 이 자산에는 몸통 충돌 형상이 없으며 제외된 접촉 쌍의 무충돌을 입증하지 않습니다.</p>
<h2>같은 다리 허벅지–발 근사</h2>
<p>허벅지 캡슐과 발 충돌 메시를 감싼 상자의 기하학적 간격이며 실제 접촉 측정이 아닙니다. 이 네 물리 접촉 쌍은 필터로 비활성화되어 있습니다. 음수는 근사 형상 겹침 가능성을 뜻하며 실제 메시 충돌 확정이나 접촉력 판정으로 사용하지 않습니다. 기록이 없는 실행은 미기록으로 표시합니다.</p>
<table><tr><th>모드</th><th>최소 근사 간격 mm</th><th>근사 겹침 비율</th></tr>{''.join(proxy_rows)}</table>
<h2>구동 설정 확인</h2><ul>{checks_text}</ul><p>getter는 기록된 시점의 실제 PhysX 값입니다. 배열 검사는 IsaacLab 버퍼의 전체 기록 검사로 구분했습니다. 누락된 기록은 통과로 바꾸지 않았습니다.</p>
{''.join(details)}<small>원자료: <code>{esc(result['source'])}</code><br>관절·모드 CSV와 JSON에 정격 초과 시간, 지지 발 수, WBC 수렴률 및 파일 SHA256을 기록했습니다.</small></html>'''


def write_report(source, output):
    source=Path(source).resolve();source=source.parent if source.is_file() else source
    output=Path(output);output.mkdir(parents=True,exist_ok=True);r=summarize(source);stem=f'integration_{r["run"]}'
    (output/f'{stem}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    joint_rows=[dict(mode=mode,duration_s=v['duration_s'],**j) for mode,v in r['modes'].items() for j in v['joints']]
    def summary_row(v):
        return dict(stand_forward_shift_m=r['stand_forward_shift_m'],
        **{k:v[k] for k in ('duration_s','height_mean_m','height_min_m','speed_mean_m_s','speed_max_m_s','maximum_tilt_deg','mean_absolute_mechanical_power_w')},
        **{k:value for k,value in v['collision'].items() if k!='scope'},
        **{f'same_leg_thigh_foot_proxy_{k}':v['same_leg_thigh_foot_proxy'][k] for k in ('available','minimum_gap_m','overlap_fraction')})
    mode_rows=[dict(mode=mode,**summary_row(v)) for mode,v in r['modes'].items()]
    def segment_identity(v):
        return {k:v[k] for k in ('segment_index','mode','mode_occurrence','start_s','end_s')}
    segment_rows=[dict(**segment_identity(v),**summary_row(v)) for v in r['timeline']]
    segment_joints=[dict(**segment_identity(v),duration_s=v['duration_s'],**j) for v in r['timeline'] for j in v['joints']]
    for suffix,rows in [('joints',joint_rows),('modes',mode_rows),('segments',segment_rows),('segment_joints',segment_joints)]:
        with (output/f'{stem}_{suffix}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/f'{stem}.html').write_text(render_html(r,source))
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sourcepath',nargs='+',required=True,type=Path)
    parser.add_argument('--outputdir',required=True,type=Path)
    args=parser.parse_args()
    for source in args.sourcepath:
        r=write_report(source,args.outputdir)
        print(json.dumps(dict(run=r['run'],task_success=r['task_success'],seconds=r['seconds_used'],output=str(args.outputdir/f'integration_{r["run"]}.html')),ensure_ascii=False))

if __name__=='__main__':main()
