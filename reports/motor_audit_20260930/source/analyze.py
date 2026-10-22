"""Analyze the new ws1 rollouts only. Run after syncing completed raw data."""
from pathlib import Path
import csv, hashlib, html, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_math import joint_statistics

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
SPECS=json.loads((ROOT/'specs/motors.json').read_text())
VERSIONS=['V13','V14','V15','V16']
TERRAINS=['flat','stairs_up']
NAMES=[f'{leg}_{joint}' for leg in ['FL','FR','RL','RR'] for joint in ['HAA','HFE','KFE']]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':160})

def one(paths):
    paths=list(paths)
    if len(paths)!=1: raise RuntimeError(f'Expected one artifact, got {paths}')
    return paths[0]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load(version, terrain):
    if version=='V16':
        assert terrain=='flat', 'V16 stairs skipped at user request'
        p=one(DATA.glob(f'v16_*/V16_{terrain}_seed1001/motor_trace_seed1001.npz'))
        report=json.loads(p.with_name('eval_seed1001.json').read_text())
        traj=np.load(p.with_name('trace_seed1001.npz'))
        outcome={'survived':report['survived_envs'],'envs':report['num_envs'],
                 'progress_m':float(np.mean(report['progress_m'])), 'command_m_s':report['speed_command_m_s'],
                 'seconds':report['seconds'], 'seed':report['seed'], 'stair_pass':None,
                 'steps_mean':float(np.mean(report['completed_step_count'])),
                 'terrain_parity':'plane',
                 'torque_semantics':'explicit effort command after independent guard'}

    else:
        p=one(DATA.glob(f'rl_*/{version}_{terrain}_seed42.npz'))
        report=json.loads(p.with_suffix('.json').read_text())
        outcome={'survived':report['survived_envs'],'envs':report['initial_envs'],
                 'progress_m':report['mean_directed_progress_m'],
                 'command_m_s':report['metadata']['command_forward_m_s'],
                 'seconds':report['metadata']['measured_s'],'seed':42,
                 'stair_pass':report.get('path',{}).get('directed_stair_success_count'),
                 'path_metrics':report.get('path',{}),
                 'torque_semantics':'implicit PD estimate after guard and clip' if version=='V15' else 'explicit DCMotor command after guard and clipping',
                 'terrain_parity':'fixed_v2', 'actor_sha256':report['metadata']['actor_sha256']}
    d=np.load(p)
    names=d['joint_names'].tolist()
    assert names==NAMES,names
    tau=d['applied_torque_nm']; request=d['computed_torque_nm']; valid=d['alive'].astype(bool)
    times=d['time_s']; delta=np.diff(times)
    assert tau.shape==(len(times),outcome['envs'],12),tau.shape
    assert np.allclose(delta,.005,atol=1e-8)
    assert not (valid[1:]&~valid[:-1]).any(), 'Episode revived after failure'
    assert np.isfinite(tau[valid]).all()
    if version != 'V16':
        assert np.allclose(np.abs(tau[valid]).max(axis=0),report['actuation']['peak_applied_nm'],atol=1e-5)
        assert int(valid.sum())==report['actuation']['sampled_physics_steps']
        assert len(times)==round(outcome['seconds']/.005)
    else:
        assert len(times)==round(outcome['seconds']/.005)
        assert np.allclose(tau[::4],traj['torque_command_nm'],atol=1e-5)
        assert np.allclose(req_check := d['computed_torque_nm'][::4],traj['torque_requested_nm'],atol=1e-5,equal_nan=True)
    outcome.update(version=version,terrain=terrain,trace_sha256=sha(p),trace=str(p.relative_to(ROOT)),
                   physics_samples=len(times),sample_dt_s=.005)
    return tau,request,valid,times,outcome

rows=[]; outcomes=[]; traces={}; unavailable_segments=[]
for terrain in TERRAINS:
    for version in VERSIONS:
        if version=='V16' and terrain=='stairs_up':
            continue
        tau,req,valid,t,outcome=load(version,terrain)
        outcomes.append(outcome); traces[(version,terrain)]=(tau,valid,t)
        specs=[SPECS['HAA' if n.endswith('HAA') else 'HFE_KFE'] for n in NAMES]
        spec_args=[[s[k] for s in specs] for k in ['rated_torque_nm','sim_limit_nm','peak_torque_nm']]
        gains=[s['motor_kt_nm_per_a']*s['gear_ratio'] for s in specs]
        for segment,mask in [('all',np.ones_like(t,bool)),('after2s',t>=2)]:
            if not valid[mask].any():
                unavailable_segments.append(dict(version=version,terrain=terrain,segment=segment,reason='No surviving samples in segment'))
                continue
            values=joint_statistics(tau[mask],req[mask],valid[mask],.005,*spec_args,gains)
            for joint,stats in zip(NAMES,values):
                rows.append(dict(version=version,terrain=terrain,segment=segment,joint=joint,**stats))

with (ROOT/'per_motor.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'motor_specs':SPECS,'outcomes':outcomes,'per_motor':rows,'unavailable_segments':unavailable_segments}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
colors=['#2d70b3','#e68c23','#478f69','#975aa2']
for terrain in TERRAINS:
    fig,axes=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
    xx=np.arange(12)
    for vi,version in enumerate(VERSIONS):
        rr=[r for r in rows if r['version']==version and r['terrain']==terrain and r['segment']=='all']
        if not rr: continue
        for ax,key in zip(axes.flat,['torque_rms_nm','torque_peak_nm','current_rms_ideal_a','rated_exceed_fraction']):
            y=[r[key]*(100 if key=='rated_exceed_fraction' else 1) for r in rr]
            ax.bar(xx+(vi-1.5)*.19,y,.19,label=version,color=colors[vi])
    for ax in axes.flat:
        ax.set_xticks(xx,NAMES,rotation=55,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ax in axes[0]:
        ax.plot(xx,[8.3 if n.endswith('HAA') else 18 for n in NAMES],'k_',ms=12,label='Rated torque')
    axes[0,1].plot(xx,[24.8 if n.endswith('HAA') else 48 for n in NAMES],'r_',ms=12,label='Simulator peak limit')
    axes[0,1].plot(xx,[24.8 if n.endswith('HAA') else 53 for n in NAMES],'^',color='#385e46',ms=4,label='Hardware peak rating')
    handles, labels = axes[0,1].get_legend_handles_labels()
    limits_legend = [(h,l) for h,l in zip(handles,labels) if l not in VERSIONS]
    axes[0,1].legend([h for h,l in limits_legend],[l for h,l in limits_legend],loc='upper center',bbox_to_anchor=(.5,1.1),ncol=3,fontsize=8)
    axes[0,0].set_ylabel('RMS torque [Nm]');axes[0,1].set_ylabel('Peak absolute torque [Nm]')
    axes[1,0].set_ylabel('Ideal torque-producing RMS current [A]')
    axes[1,1].set_ylabel('Samples above rated torque [%]')
    axes[0,0].legend(ncol=3,fontsize=9)
    fig.suptitle(f'{terrain}: 12 motors | each original controller, different speeds\nV15 torque is a PD estimate; V16 stairs skipped by request; current assumes efficiency=1',fontsize=14)
    fig.savefig(figdir/f'{terrain}_comparison.png');fig.savefig(figdir/f'{terrain}_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(15,12),sharex=True,constrained_layout=True)
    for vi,version in enumerate(VERSIONS):
        if (version,terrain) not in traces: continue
        tau,valid,t=traces[(version,terrain)]
        # First environment is illustrative. Full tables pool all valid environments.
        for j,ax in enumerate(axes.flat):
            y=np.where(valid[:,0],tau[:,0,j],np.nan)
            ax.plot(t,y,lw=.55,color=colors[vi],label=version,alpha=.8)
    for j,ax in enumerate(axes.flat):
        rated=8.3 if NAMES[j].endswith('HAA') else 18
        ax.axhline(rated,color='k',ls='--',lw=.7);ax.axhline(-rated,color='k',ls='--',lw=.7)
        ax.set_title(NAMES[j]);ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel('Time [s]')
    for ax in axes[:,0]: ax.set_ylabel('Torque [Nm]')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{terrain} | environment 0 | dashed lines: rated torque\nV16 includes initial 1s stand; RL flat starts after 2s preroll',fontsize=14)
    fig.savefig(figdir/f'{terrain}_traces.png');fig.savefig(figdir/f'{terrain}_traces.pdf');plt.close(fig)

# Self-contained reviewable Korean HTML plus CSV/JSON and downloadable static plots.
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V13–V16 모터 부하 비교</title>
<style>body{font-family:Arial,'Noto Sans CJK KR',sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;line-height:1.6;color:#183044}h1,h2{color:#102f48}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{padding:8px 9px;border-bottom:1px solid #dce4ea;text-align:right}th{background:#edf3f7;position:sticky;top:0}td:first-child,th:first-child{text-align:left}.note{background:#edf4fa;padding:18px;border-left:4px solid #367aac}.bad{color:#af3929;font-weight:bold}img{width:100%}small{color:#506473}a{color:#226fad}</style>
<h1>V13–V16: 평지·계단 모터별 전류 및 토크</h1>
<p>새 Isaac Sim / PhysX 실행 · ws1 · Slurm admin QoS · 200 Hz 계측 · 조건당 8환경</p>
<div class="note"><b>전류는 환산 추정치입니다.</b> Iτ = |τ| / (Kt × 감속비), 기어 효율 η=1 기준.
HAA는 1.23 N·m/A, HFE/KFE는 1.44 N·m/A를 사용합니다. η=.85를 가정하면 표의 전류가 17.65% 증가합니다.
이 수치는 배터리 전류가 아닙니다. 제조사 ADC 전류와 Kt의 전류 정의가 확인되지 않아 <b>전류 정격 초과는 확정하지 않습니다.</b>
V15 토크는 IsaacLab implicit PD 추정치이고, V13/V14/V16은 시뮬레이션에 입력한 explicit 토크입니다.</div>
<p><b>판정:</b> 이번 유효 표본에서는 시뮬레이터 최대값과 실제 모터의 카탈로그 피크 토크 초과는 없었습니다. 다만 정격 토크를 넘는 순간은 V16 평지에도 있으며, RMS가 정격 이하라는 사실이 순간 정격 초과가 없음을 뜻하지 않습니다.</p><h2>모터 기준</h2><table><tr><th>관절 / 모터</th><th>정격 Nm</th><th>피크 Nm</th><th>시뮬레이터 한도 Nm</th><th>제조사 정격/피크 ADC</th></tr>
<tr><td>HAA · AK70-10 KV100</td><td>8.3</td><td>24.8</td><td>24.8</td><td>7.2 / 23.2</td></tr>
<tr><td>HFE/KFE · AK10-9 KV60 V3.0</td><td>18</td><td>53</td><td>48</td><td>10.7 / 31.9</td></tr></table>
<p>사용자가 모터 모델과 AK10-9 V3.0을 확인했습니다. 기존 시뮬레이터의 48 Nm 제한은 유지했습니다.
정격 초과 시간과 RMS는 부하 지표입니다. 짧은 피크가 정격보다 높다는 사실만으로 고장을 뜻하지 않으며, 온도와 연속 운전 가능 시간은 이 시험으로 판정하지 않습니다.</p>
<h2>실행 조건과 주행 결과</h2><table><tr><th>방식</th><th>지형</th><th>명령 m/s</th><th>기간 s</th><th>생존</th><th>평균 진행 m</th><th>계단 통과</th></tr>''']
for o in outcomes:
    p='—' if o['terrain']=='flat' else str(o['stair_pass']) if o['stair_pass'] is not None else '원자료 참조'
    parts.append(f"<tr><td>{o['version']}</td><td>{o['terrain']}</td><td>{o['command_m_s']:.2f}</td><td>{o['seconds']:g}</td><td>{o['survived']}/{o['envs']}</td><td>{o['progress_m']:.3f}</td><td>{p}</td></tr>")
parts.append('</table><h2>부하 요약</h2><table><tr><th>방식/지형</th><th>모터별 등가전류 RMS 범위 A</th><th>최대 토크 Nm</th><th>RMS가 정격보다 높은 모터</th><th>피크 한계 초과</th></tr>')
for o in outcomes:
    rr=[r for r in rows if r['version']==o['version'] and r['terrain']==o['terrain'] and r['segment']=='all']
    currents=[r['current_rms_ideal_a'] for r in rr]
    exceed=sum(r['rms_over_rated'] for r in rr)
    peak_exceed=any(r['motor_peak_exceed_fraction'] or r['sim_limit_exceed_fraction'] for r in rr)
    parts.append(f"<tr><td>{o['version']} / {o['terrain']}</td><td>{min(currents):.2f}–{max(currents):.2f}</td><td>{max(r['torque_peak_nm'] for r in rr):.2f}</td><td>{exceed}/12</td><td>{'있음' if peak_exceed else '없음'}</td></tr>")
parts.append('''</table><p>V13: 선택된 V11 actor4999 + V12 depth + V13 LPF/guard. V14: 선택된 actor5700 + 다리 충돌 모델. V15: 선택된 actor11699 + implicit PD. V16: 작업577의 frozen WBC/CEM 제어기.</p>
<p>V13~V15는 flat 0.4m/s, stairs 0.5m/s, 12초, seed42입니다. V16은 검증된 0.05m/s, 32초, seed1001을 유지합니다.
따라서 동일 작업량에 대한 전력 효율 순위로 해석할 수 없습니다. 평지 RL은 2초 preroll을 제외하며, V16의 초기 1초 기립은 포함합니다.
CSV에는 전체 구간(all)과 초기 2초 제외(after2s)를 모두 기록했습니다.</p>
<p>계단은 V13~V15의 기존 fixed_v2 지형이며 단높이 15.714cm입니다. V16은 사용자가 계단을 걷지 못한다고 알려주어 계단 평가를 제외했습니다. V16 계단 토크·전류 결과는 없습니다.</p>''')
for terrain in TERRAINS:
    parts.append(f'<h2>{"평지" if terrain=="flat" else "계단 상승"}: 모터별 결과</h2><img src="figures/{terrain}_comparison.png">')
    parts.append('<p>FL 앞왼쪽 · FR 앞오른쪽 · RL 뒤왼쪽 · RR 뒤오른쪽. HAA 좌우 벌림 · HFE 엉덩이 굽힘 · KFE 무릎.<br>전류와 토크는 첫 실패 이전의 유효 표본만 집계하며, 음의 토크도 절댓값/RMS에 포함합니다.</p>')
    for version in VERSIONS:
        rr=[r for r in rows if r['version']==version and r['terrain']==terrain and r['segment']=='all']
        if not rr:
            parts.append(f'<h3>{version}</h3><p>'+('계단: 사용자 요청으로 미실행.' if version=='V16' and terrain=='stairs_up' else '유효 표본 없음.')+'</p>')
            continue
        parts.append(f'<h3>{version}</h3><table><tr><th>모터</th><th>전류 평균 |A|</th><th>전류 RMS A</th><th>전류 피크 A</th><th>토크 RMS Nm</th><th>토크 피크 Nm</th><th>정격 초과 %</th><th>최장 연속 초과 s</th><th>RMS &gt; 정격</th><th>시뮬/모터 피크 초과</th></tr>')
        for r in rr:
            risk='예' if r['rms_over_rated'] else '아니오'
            limit=f"{'예' if r['sim_limit_exceed_fraction'] else '없음'} / {'예' if r['motor_peak_exceed_fraction'] else '없음'}"
            parts.append(f"<tr><td>{r['joint']}</td><td>{r['current_mean_abs_ideal_a']:.2f}</td><td>{r['current_rms_ideal_a']:.2f}</td><td>{r['current_peak_ideal_a']:.2f}</td><td>{r['torque_rms_nm']:.2f}</td><td>{r['torque_peak_nm']:.2f}</td><td>{r['rated_exceed_fraction']*100:.2f}</td><td>{r['rated_exceed_max_contiguous_s']:.3f}</td><td class='{('bad' if r['rms_over_rated'] else '')}'>{risk}</td><td>{limit}</td></tr>")
        parts.append('</table>')
    parts.append(f'<details><summary>12개 관절 토크 시계열 보기 · 환경0 예시</summary><img src="figures/{terrain}_traces.png"></details>')
parts.append('''<h2>해석 범위와 재현</h2><p>모든 조건은 원본 모델의 서로 다른 actuator와 마찰·제어 구조를 보존했습니다. 각 조건8환경, 단일 시드의 부하 측정이며 통계적 성능 보증은 아닙니다.
요구 토크(computed/requested)는 V13~V15에서 위치 목표 가드 이후, V16에서 토크 가드 이전이므로 동일한 정책 원요구량으로 비교하지 않습니다.
PhysX의 관절 반력 전체를 모터 토크라고 해석하지 않았습니다. V13~V15 생존 마스크의 종료 경계는 정책 주기20ms 해상도이며, V16 추가 계측은5ms입니다. 넘어짐을 일으킨 마지막 입력까지 포함하고 이후의 자동 리셋 데이터는 제외합니다.</p>
<p>전류 환산은 권선 저항, 드라이버 손실, 역기전력, 회생 제동, 무부하 전류 및 온도를 포함하지 않습니다.
제조사 정격/피크 토크÷전류와 Kt×감속비가 일치하지 않아 전류 수치의 정의·보정은 실기에서 추가 확인해야 합니다.</p>
<p><a href="per_motor.csv">모터별 CSV (전체 구간 + 유효 표본이 있는 초기2초 제외 구간)</a> · <a href="summary.json">결과 JSON</a> · <a href="specs/motors.json">환산 가정과 모터 사양</a> · <a href="source/analyze.py">분석 코드</a></p>
<p>제조사 출처: <a href="https://www.cubemars.com/product/ak70-10-kv100-robotic-actuator.html">AK70-10</a> · <a href="https://img.cubemars.com/products/cubemars-product-parameter/AK10-9-V3.0-KV60.pdf">AK10-9 V3.0 PDF</a></p></html>''')
(ROOT/'report.html').write_text('\n'.join(parts))
print(json.dumps(outcomes,indent=2,ensure_ascii=False))
print('Wrote',len(rows),'per-motor rows')
