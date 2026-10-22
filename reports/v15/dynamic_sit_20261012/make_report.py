"""Verify saved physical evidence and write the measured crouch report."""
from pathlib import Path
import hashlib,json,html,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
JOB=sys.argv[1]
p=OUT/'jobs'/JOB
r=json.loads((p/'results/dynamic_sit/result.json').read_text())
a=np.load(p/'results/dynamic_sit/telemetry.npz')
s=json.loads((p/'results/dynamic_sit/status.json').read_text())
checks=[]
for line in (p/'source.sha256').read_text().splitlines():
 sha,path=line.split('  ',1)
 f=p/'source'/path.split('/source/',1)[1]
 if hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise RuntimeError('Source hash mismatch '+str(f))
 checks.append(str(f.relative_to(p)))
n=len(a['state_stamp_ns']);t=a['state_stamp_ns']*1e-9
assert all(np.isfinite(a[k]).all() for k in a.files)
assert np.array_equal(a['command_stamp_ns'],np.arange(n,dtype=np.int64)*5_000_000)
assert np.array_equal(a['state_stamp_ns'],(np.arange(n,dtype=np.int64)+1)*5_000_000)
assert [x['stamp_ns'] for x in s]==a['command_stamp_ns'].tolist()
assert not r['reset_during_trial']
assert set(x['sit_style'] for x in s)=={'deep_crouch'}
q=a['joint_position'];knees=q[:,[2,5,8,11]]
height=a['position'][:,2]
seat=(t>=5.5)&(t<8);stand=(t>=1.5)&(t<2.)
summary=dict(job_id=int(JOB),result=r,source_sha256_count=len(checks),all_saved_arrays_finite=True,
 command_state_timestamps_valid=True,recorded_steps=n,usb=json.loads((p/'results/dynamic_sit/usb_transport.json').read_text()),
 requested_parameters=dict(sit_style='deep_crouch',height_drop_m=.18,posture_transition_s=2.),
 measured=dict(standing_height_m=float(np.median(height[stand])),crouched_height_m=float(np.median(height[seat])),
  height_drop_m=float(np.median(height[stand])-np.median(height[seat])),
  median_knee_degrees=np.degrees(np.median(knees[seat],axis=0)).tolist()),
 rated_rms_exceed_count=sum(r['rms_exceeds_rated']),collision_pairs=r.get('contact_pairs_over_1n',{}),
 scope='14s stand -> deep crouch -> stand via native ROS2 and Python/C USB codec emulator, authentic Isaac CAD physics; no hardware/Jetson timing claim',
 unresolved='Existing stop-to-stand robustness and 50Hz WBC realtime performance remain outside this isolated posture trial.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
axes[0].plot(t,height*100,label='Actual body origin')
refs=np.array([row['wbc_vectors']['base_position_reference_world'][2] for row in s])
axes[0].plot(a['command_stamp_ns']*1e-9,refs*100,'--',label='Commanded height')
axes[0].set_ylabel('Body height (cm)');axes[0].legend();axes[0].grid(alpha=.25)
for i,label in enumerate(['FL','FR','RL','RR']):axes[1].plot(t,np.degrees(knees[:,i]),label=label)
axes[1].set_xlabel('Simulation time (s)');axes[1].set_ylabel('Knee angle (deg)');axes[1].legend();axes[1].grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/'motion.png',dpi=160);plt.close(fig)
rows=''.join('<tr><td>'+name+'</td><td>'+f'{r["torque_rms_nm"][i]:.2f}'+'</td><td>'+f'{r["torque_peak_nm"][i]:.2f}'+'</td><td>'+f'{r["ideal_current_rms_a"][i]:.2f}'+'</td><td>'+f'{r["rated_torque_nm"][i]:.1f}'+'</td></tr>' for i,name in enumerate(r['metadata']['canonical_joint_order']))
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V15 깊은 웅크리기 검증</title><style>body{{font:16px system-ui;max-width:1080px;margin:32px auto;padding:0 20px;color:#222}}video,img{{width:100%;max-width:960px}}td,th{{padding:7px 18px;border-bottom:1px solid #ddd}}.ok{{color:#157344}}</style><h1>V15 더 낮게 웅크리기</h1><p>몸통 수평 유지 · 높이 감소 목표 10→18cm · 전환 목표 3→2초. Slurm {JOB}, compute / admin.</p><p>물리 시험 결과: <strong>{'PASS' if r['task_success'] else 'FAIL'}</strong> — {r['seconds_recorded']:.2f}s / 14s</p><video controls src="jobs/{JOB}/results/dynamic_sit/video/env_00.mp4"></video><h2>측정 결과</h2><p>서기 {summary['measured']['standing_height_m']*100:.1f}cm → 웅크리기 {summary['measured']['crouched_height_m']*100:.1f}cm, 실제 감소 {summary['measured']['height_drop_m']*100:.1f}cm.</p><p>정격 RMS 초과 {summary['rated_rms_exceed_count']}/12 · 피크 제한 초과 {r['simulator_peak_exceeded']} · 비인접 자가 접촉 최대 {r['max_nonadjacent_contact_force_n']:.3f}N.</p><img src="motion.png" alt="측정 몸통 높이와 네 다리 무릎 각도"><h2>모터별 추정</h2><table><tr><th>관절</th><th>RMS N·m</th><th>Peak N·m</th><th>RMS 상전류 A</th><th>정격 N·m</th></tr>{rows}</table><p>전류는 기존 AK70-10 / AK10-9 KV60 V3.0 토크 상수, 효율1로 환산한 모터 상전류 추정이다. 배터리 전류·실측 전류가 아니다. 토크는 적용한 feedforward + 제한된 pre-step PD 추정이다.</p><p>원본 CAD 13강체·67convex hull, 66비인접 쌍 검사. 관절로 직접 연결된 쌍의 기존 충돌 필터를 유지했다. 200Hz 명령과 50Hz 정책은 lockstep 시뮬레이션이며 실시간 실행 성능 증명이 아니다. 기존 stop→stand 복귀 안정화와 WBC 계산 속도는 별도 과제로 남는다.</p><h2>ROS2 / Jetson 선택 설정</h2><pre>initial_mode:=STAND sit_style:=deep_crouch height_drop:=0.18 posture_transition_s:=2.0</pre><p>기립 상태에서 mode_request에 sit / stand를 보내 전환한다. 기존 기본값과 이전 배포 묶음은 유지했다. 새 선택 기능을 포함한 <a href="v15_jetson_deep_crouch_20261012.tar.gz">배포 묶음</a>을 추가했다. 실제 Jetson/STM32/모터 활성화는 수행하지 않았다.</p><p><a href="summary.json">전체 측정·검증 기록</a> · <a href="jobs/{JOB}/results/dynamic_sit/result.json">원본 판정</a></p></html>'''
(OUT/'report.html').write_text(page)
print(json.dumps({k:v for k,v in summary.items() if k not in ['result','usb']},indent=2))
