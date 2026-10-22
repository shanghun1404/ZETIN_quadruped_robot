"""Frozen 2026-10-01 offline gates for V15.1 flat demo vs run 601.

No simulator calls. All rejected candidates remain in the report. Run:
python3 comparison.py --baseline ../integration/601 --candidate NAME=PATH --output .
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np

DT=.005
CANONICAL=tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))
RATED=np.tile([8.3,18.,18.],4)
LIMITS=np.tile([24.8,48.,48.],4)
NM_PER_A=np.tile([1.23,1.44,1.44],4)
SEQUENCE=['SIT','RISE','STABILIZE','STAND','BLEND_WALK','WALK','BRAKE','BLEND_STAND','STABILIZE','STAND','LOWER','SETTLE_SIT','SIT']
# Frozen before inspecting candidate artifacts. Numerical tolerance is not an improvement margin.
CONTRACT=dict(version='20261001-before-candidates-v1',scenario='flat',seed=42,duration_s=28.,sample_period_s=DT,
 final_sit_window_s=[26.5,28.],final_height_delta_m=.01,forward_progress_ratio=.9,final_crouch_rms_ratio=.9,
 stand_max_rated_rms_ratio=1.,walk_mean_joint_rms_ratio=1.,walk_command_mae_increase_m_s=.02,
 full_rated_rms_strict_reduction=True,peak_numeric_tolerance_nm=1e-4,comparison_numeric_tolerance=1e-12,
 crossleg_min_gap_m=-.002,crossleg_max_force_n=1.,sameleg_min_proxy_gap_m=0.,
 required_sequence=SEQUENCE,walk_command=[0.,.4,0.],
 demo_schedule='0.5s stand, 7s walk +body Y 0.4m/s, 17s stand, 23s sit; contact-dependent readiness and command fades may differ')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def all_finite(value):
    if isinstance(value,dict):return all(all_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(all_finite(v) for v in value)
    if isinstance(value,(float,int)):return bool(np.isfinite(value))
    return True


def clean(value):
    """Represent invalid numerical evidence as null, never silently accept it."""
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def analyze(arrays,metadata,statuses):
    required=('estimated_torque','joint_velocity','position','velocity','quaternion','gaps','trunk_gaps','same_leg_thigh_foot_gap_m','contact_pair_force')
    if any(k not in arrays for k in required):raise ValueError('Missing required raw array')
    a={k:np.asarray(v,dtype=float) for k,v in arrays.items()};n=len(a['position'])
    if not n or len(statuses)!=n or any(len(v)!=n for v in a.values()):raise ValueError('Empty or mismatched sample counts')
    stamps=np.array([s['stamp_ns'] for s in statuses])
    if not np.array_equal(stamps,np.arange(n,dtype=np.int64)*5_000_000):raise ValueError('Expected contiguous 200 Hz timestamps starting at zero')
    order=metadata.get('joint_order',[])
    if len(order)!=12 or set(order)!=set(CANONICAL):raise ValueError('Missing or invalid joint order')
    indices=[order.index(name) for name in CANONICAL]
    for key in ('estimated_torque','joint_velocity'):
        if a[key].shape!=(n,12):raise ValueError(f'Invalid {key} shape')
        a[key]=a[key][:,indices]
    expected={'position':(n,3),'velocity':(n,3),'quaternion':(n,4),'gaps':(n,24),'trunk_gaps':(n,8),'same_leg_thigh_foot_gap_m':(n,4),'contact_pair_force':(n,12)}
    if any(a[k].shape!=shape for k,shape in expected.items()):raise ValueError('Invalid raw motion or collision array shape')
    modes=np.array([s['mode'] for s in statuses]);t=np.arange(n)*DT
    command=np.asarray([s['command'] for s in statuses],float)
    bounded=np.asarray([s['bounded_torque'] for s in statuses],float)
    if command.shape!=(n,3) or bounded.shape!=(n,12):raise ValueError('Missing command or canonical bounded torque')
    finite=all(np.isfinite(v).all() for v in a.values()) and all_finite(statuses) and all_finite(metadata) and bool((np.linalg.norm(a['quaternion'],axis=1)>1e-12).all())
    boundaries=np.r_[0,np.flatnonzero(modes[1:]!=modes[:-1])+1,n]
    timeline=[dict(mode=str(modes[start]),start_s=float(start*DT),end_s=float(end*DT)) for start,end in zip(boundaries[:-1],boundaries[1:])]
    stand_segments=[(start,end) for start,end in zip(boundaries[:-1],boundaries[1:]) if modes[start]=='STAND']
    masks={'ALL':np.ones(n,bool)}
    masks.update({str(mode):modes==mode for mode in dict.fromkeys(modes)})
    masks['FINAL_CROUCH']=(t>=26.5)&(t<28.)
    for name,index in [('INITIAL_STAND',0),('POSTSTOP_STAND',1)]:
        mask=np.zeros(n,bool)
        if len(stand_segments)>index:
            start,end=stand_segments[index];mask[start:end]=True
        masks[name]=mask
    tau=a['estimated_torque'];windows={}
    for name,mask in masks.items():
        if not mask.any():windows[name]=None;continue
        selected=tau[mask];rms=np.sqrt(np.mean(selected**2,axis=0));peak=np.max(abs(selected),axis=0)
        joints=[dict(joint=joint,rms_nm=float(rms[j]),peak_nm=float(peak[j]),rms_over_rated=float(rms[j]/RATED[j]),rated_nm=float(RATED[j]),ideal_current_rms_a=float(rms[j]/NM_PER_A[j]),ideal_current_peak_a=float(peak[j]/NM_PER_A[j]),rated_exceed_seconds=float(np.count_nonzero(abs(selected[:,j])>RATED[j])*DT)) for j,joint in enumerate(CANONICAL)]
        windows[name]=dict(samples=int(mask.sum()),duration_s=float(mask.sum()*DT),joints=joints,
            max_joint_rms_over_rated=float(np.max(rms/RATED)),mean_joint_rms_nm=float(rms.mean()),
            rated_normalized_rms=float(np.sqrt(np.mean((selected/RATED)**2))),
            ideal_current_squared_integral_a2_s=float(np.sum((selected/NM_PER_A)**2)*DT),
            absolute_mechanical_work_j=float(np.sum(abs(selected*a['joint_velocity'][mask]))*DT),
            mean_absolute_mechanical_power_w=float(np.mean(np.sum(abs(selected*a['joint_velocity'][mask]),axis=1))),
            body_y_command_mae_m_s=float(np.mean(abs(a['velocity'][mask,1]-command[mask,1]))),
            height_mean_m=float(a['position'][mask,2].mean()))
    collision=dict(min_crossleg_gap_m=float(a['gaps'].min()),max_crossleg_force_n=float(a['contact_pair_force'].max()),min_sameleg_proxy_gap_m=float(a['same_leg_thigh_foot_gap_m'].min()),min_trunk_proxy_gap_m=float(a['trunk_gaps'].min()),trunk_overlap_fraction=float((a['trunk_gaps'].min(axis=1)<0).mean()))
    collision['passed']=bool(collision['min_crossleg_gap_m']>=-.002 and collision['max_crossleg_force_n']<=1. and collision['min_sameleg_proxy_gap_m']>=0.)
    walk=modes=='WALK'
    result=dict(samples=n,seconds=n*DT,stamp_first_ns=int(stamps[0]),stamp_last_ns=int(stamps[-1]),stamp_alignment_verified=True,
        metadata={k:metadata.get(k) for k in ('scenario','seed','seconds_recorded','completed','task_success','motion_success','collision_checks_pass','reset_after_initialization','failure','demo_sequence_required','collision_scope','collision_gate_semantics','same_leg_thigh_foot_gate_applied','torque_semantics')},
        finite_no_reset=bool(finite and metadata.get('reset_after_initialization') is False and not np.any(np.isin(modes,['FAULT','RESET']))),
        complete_motion_sequence=[s['mode'] for s in timeline]==SEQUENCE,
        final_sit_window=bool(n==5600 and np.all(modes[masks['FINAL_CROUCH']]=='SIT')),
        walk_command_correct=bool(walk.any() and np.allclose(command[walk],[0.,.4,0.],rtol=0.,atol=1e-12)),
        final_height_m=float(a['position'][-1,2]),forward_progress_world_y_m=float(a['position'][-1,1]-a['position'][0,1]),
        estimated_torque_peak_nm=np.max(abs(tau),axis=0).tolist(),bounded_command_peak_nm=np.max(abs(bounded),axis=0).tolist(),
        peak_within_limits=bool(np.all(abs(tau)<=LIMITS+1e-4) and np.all(abs(bounded)<=LIMITS+1e-4)),
        windows=windows,timeline=timeline,collision=collision)
    return clean(result)


def load_run(path):
    path=Path(path).resolve();path=path.parent if path.is_file() else path
    metadata=json.loads((path/'result.json').read_text());statuses=json.loads((path/'result.status.json').read_text())
    with np.load(path/'result.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    result=analyze(arrays,metadata,statuses)
    result.update(source=str(path),run=path.name,source_sha256={name:digest(path/name) for name in ('result.json','result.status.json','result.npz')})
    return result


def compare(baseline,candidate):
    b,c=baseline,candidate;bm,cm=b['metadata'],c['metadata'];gates={}
    def gate(name,passed,actual=None,limit=None):gates[name]=dict(passed=bool(passed),actual=actual,limit=limit)
    def metric(run,window,key):return (run['windows'].get(window) or {}).get(key)
    def relative(name,window,key,ratio=1.,strict=False):
        bv=metric(b,window,key);cv=metric(c,window,key);limit=None if bv is None else ratio*bv
        passed=cv is not None and limit is not None and (cv<limit if strict else cv<=limit+1e-12)
        gate(name,passed,cv,limit)
    gate('same_demo_scope',all(cm[k]==bm[k] for k in ('scenario','seed')) and cm['scenario']=='flat' and cm['seed']==42 and cm['demo_sequence_required'] is True and c['walk_command_correct'])
    gate('full_duration',c['samples']==5600 and cm['seconds_recorded']==28.,c['seconds'],28.)
    gate('task_motion_success',all(cm[k] is True for k in ('completed','task_success','motion_success')) and cm['failure'] is None)
    gate('complete_motion_sequence',c['complete_motion_sequence'])
    gate('finite_no_reset',c['finite_no_reset'])
    gate('collision',c['collision']['passed'] and cm['collision_checks_pass'] is True and cm['same_leg_thigh_foot_gate_applied'] is True and all(cm[k]==bm[k] and cm[k] is not None for k in ('collision_scope','collision_gate_semantics')),c['collision'])
    dh=None if c['final_height_m'] is None or b['final_height_m'] is None else abs(c['final_height_m']-b['final_height_m'])
    gate('final_height',dh is not None and dh<=.01+1e-12,dh,.01)
    progress=c['forward_progress_world_y_m'];minimum=.9*b['forward_progress_world_y_m']
    gate('forward_progress',progress is not None and progress>=minimum-1e-12,progress,minimum)
    gate('final_sit_window',c['final_sit_window'])
    relative('final_crouch_rms','FINAL_CROUCH','max_joint_rms_over_rated',.9)
    relative('initial_stand_rms','INITIAL_STAND','max_joint_rms_over_rated')
    relative('poststop_stand_rms','POSTSTOP_STAND','max_joint_rms_over_rated')
    relative('walk_rms','WALK','mean_joint_rms_nm')
    cv=metric(c,'WALK','body_y_command_mae_m_s');bv=metric(b,'WALK','body_y_command_mae_m_s');limit=None if bv is None else bv+.02
    gate('walk_command_mae',cv is not None and limit is not None and cv<=limit+1e-12,cv,limit)
    relative('full_rated_normalized_rms','ALL','rated_normalized_rms',strict=True)
    gate('peak_limits',c['peak_within_limits'],c['estimated_torque_peak_nm'],(LIMITS+1e-4).tolist())
    return dict(accepted=all(g['passed'] for g in gates.values()),failed_gates=[name for name,g in gates.items() if not g['passed']],gates=gates)


def render_html(report):
    esc=html.escape;b=report['baseline'];details=[]
    bw=b['windows']
    rows=[f'<tr><td>기준 601</td><td>비교 기준</td><td>{bw["FINAL_CROUCH"]["max_joint_rms_over_rated"]:.4f}</td><td>{bw["ALL"]["rated_normalized_rms"]:.4f}</td><td>{bw["WALK"]["body_y_command_mae_m_s"]:.4f}</td><td>—</td></tr>']
    for entry in report['candidates']:
        c=entry['run'];v=entry['comparison'];windows=c['windows']
        def value(window,key):
            x=(windows.get(window) or {}).get(key);return '미기록' if x is None else f'{x:.4f}'
        rows.append(f'<tr><td>{esc(entry["name"])}</td><td>{"채택 가능" if v["accepted"] else "기각"}</td><td>{value("FINAL_CROUCH","max_joint_rms_over_rated")}</td><td>{value("ALL","rated_normalized_rms")}</td><td>{value("WALK","body_y_command_mae_m_s")}</td><td>{esc(", ".join(v["failed_gates"]))}</td></tr>')
        gate_rows=''.join(f'<tr><td>{esc(name)}</td><td>{"통과" if g["passed"] else "미달"}</td><td>{esc(json.dumps(g["actual"],ensure_ascii=False))}</td><td>{esc(json.dumps(g["limit"],ensure_ascii=False))}</td></tr>' for name,g in v['gates'].items())
        details.append(f'<details><summary>{esc(entry["name"])} 판정 근거</summary><table><tr><th>기준</th><th>판정</th><th>측정값</th><th>문턱</th></tr>{gate_rows}</table></details>')
    for label,r in [('기준 601',b)]+[(e['name'],e['run']) for e in report['candidates']]:
        summaries=[];joint_sections=[]
        for mode,w in r['windows'].items():
            if w is None:continue
            summaries.append(f'<tr><td>{esc(mode)}</td><td>{w["duration_s"]:.3f}</td><td>{w["max_joint_rms_over_rated"]}</td><td>{w["ideal_current_squared_integral_a2_s"]}</td><td>{w["absolute_mechanical_work_j"]}</td></tr>')
            jr=''.join(f'<tr><td>{j["joint"]}</td><td>{j["rms_nm"]}</td><td>{j["peak_nm"]}</td><td>{j["ideal_current_rms_a"]}</td><td>{j["ideal_current_peak_a"]}</td><td>{j["rated_exceed_seconds"]}</td></tr>' for j in w['joints'])
            joint_sections.append(f'<details><summary>{esc(mode)} 12개 모터</summary><table><tr><th>모터</th><th>RMS Nm</th><th>최대 Nm</th><th>이상 RMS A</th><th>이상 최대 A</th><th>정격 초과 s</th></tr>{jr}</table></details>')
        provenance=esc(json.dumps({k:r[k] for k in ('source','source_sha256')},ensure_ascii=False,indent=2))
        details.append(f'<details><summary>{esc(label)} 상세 수치·원자료</summary><table><tr><th>구간</th><th>시간 s</th><th>최대 RMS/정격</th><th>Σ∫I²dt A²s</th><th>Σ∫|τq̇|dt J</th></tr>{"".join(summaries)}</table>{"".join(joint_sections)}<pre>{provenance}</pre></details>')
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V15.1 자세 토크 후보 비교</title><style>body{{font:15px/1.6 sans-serif;max-width:1250px;margin:32px auto;padding:16px;color:#18222c}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #ddd;padding:7px;text-align:right}}td:first-child,th:first-child{{text-align:left}}details{{margin:18px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>V15.1 자세 토크 후보 비교</h1><p>기준: 실행 601 · 평지 · seed 42 · 28초 · 200 Hz. 각 후보는 같은 데모의 재현 결과이며 다른 지형·시드로 일반화하지 않습니다.</p><p>후보 결과를 보기 전에 고정한 모든 문턱을 통과해야 채택 가능합니다. 기각된 후보도 아래에 보존합니다. 마지막 웅크림은 26.5–28.0초 전체가 SIT여야 하며 최대 관절 RMS/정격이 기준 대비 10% 이상 감소해야 합니다.</p><table><tr><th>후보</th><th>판정</th><th>마지막 웅크림 RMS/정격</th><th>전체 정격 정규화 RMS</th><th>WALK +Y 명령 MAE m/s</th><th>미달 기준</th></tr>{''.join(rows)}</table><p>토크는 명령 또는 implicit PD 순간 추정값이며 PhysX 실측 구동력이 아닙니다. 이상 전류는 HAA τ/1.23, HFE·KFE τ/1.44 A이며 효율 1 가정입니다. Σ∫I²dt는 동손의 대리지표로서 배터리 에너지나 온도가 아닙니다. 기계적 절대 일 Σ∫|τq̇|dt는 별도로 표시합니다.</p><p>충돌 검사는 기록된 교차 다리 접촉 ≤1 N, 캡슐 간격 ≥−2 mm, 같은 다리 허벅지–발 보수적 근사 간격 ≥0입니다. 몸통 근사 겹침은 별도 관측값이며 실제 몸통 충돌 판정이 아닙니다. 제외된 물리 접촉 쌍의 무충돌은 입증하지 않습니다.</p>{''.join(details)}<details><summary>사전 고정 판정 기준</summary><pre>{esc(json.dumps(report['contract'],ensure_ascii=False,indent=2))}</pre></details><p>전체 수치, 각 판정의 실제값·문턱, 입력 SHA256: summary.json.</p></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--candidate',action='append',default=[]);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    baseline=load_run(args.baseline);entries=[]
    for item in args.candidate:
        name,path=item.split('=',1);candidate=load_run(path);entries.append(dict(name=name,run=candidate,comparison=compare(baseline,candidate)))
    baseline_checks=compare(baseline,baseline)
    baseline_invalid=[k for k in baseline_checks['failed_gates'] if k not in ('final_crouch_rms','full_rated_normalized_rms')]
    if baseline_invalid:raise ValueError(f'Baseline itself invalid: {baseline_invalid}')
    report=dict(schema=1,contract=CONTRACT,comparison_source_sha256=digest(__file__),joint_order=CANONICAL,
        verification=dict(scoped_command='python3 -m unittest discover -s reports/v151/optimization_20261001 -p test_comparison.py',scoped_result='12 tests passed',repository_command='python3 -m pytest -q',repository_result='44 collection errors; tests did not execute',repository_limitations=['Local Python lacks torch for config and ROS14 imports','Archived and live test_motion.py, test_ros_transport.py, test_standing.py share module names and cause pytest import mismatches']),
        scope='Only flat demo seed42 full28s candidates compared to 601; simulation lockstep, not wall-clock control validation',
        metric_definitions=dict(full_rated_normalized_rms='sqrt(mean over samples and 12 joints of (estimated torque/rated torque)^2)',walk_rms='arithmetic mean of 12 individual joint RMS values',stand_rms='maximum of individual joint RMS/rated torque within each consecutive STAND segment',forward_progress='last world Y minus first world Y',walk_command_mae='mean abs(body-frame velocity Y - status command Y) during WALK',sample_interval='timestamp labels denote fixed 5ms slots; last stamp 27.995s covers through 28s'),
        current_model=dict(nm_per_a=NM_PER_A.tolist(),efficiency=1.,battery_current=False,temperature_validated=False),baseline=baseline,candidates=entries)
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'summary.json').write_text(json.dumps(clean(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n');(args.output/'report.html').write_text(render_html(report))
    print(json.dumps([dict(name=e['name'],**e['comparison']) for e in entries],ensure_ascii=False))

if __name__=='__main__':main()
