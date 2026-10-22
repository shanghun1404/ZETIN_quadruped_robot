"""Summarize archived Isaac telemetry without changing acceptance criteria."""
import argparse
import html
import json
from pathlib import Path
import numpy as np


def build(directory,job):
    run=directory/'runs'/str(job)
    result=json.loads((run/'result.json').read_text())
    audit=json.loads((directory/'audit/usd_cpu_audit.json').read_text())
    summary=dict(job=job,result=result,old_mass_kg=audit['old']['total_mass_kg'],
                 usd_authored_mass_kg=audit['new']['total_mass_kg'])
    plot='';contact_text='접촉 데이터 없음'
    if (run/'result.npz').exists():
        raw=np.load(run/'result.npz')
        if 'position' in raw:
            pairs=[(p['body'],target) for p in result['joint_mapping']['pair_order'] for target in p['targets']]
            force=raw['nonadjacent_pair_force_n']
            assert force.shape[1]==len(pairs)
            categories={name:[] for name in ('trunk','cross_leg','same_leg_nonadjacent')}
            unique={}
            for i,(a,b) in enumerate(pairs):
                name='trunk' if 'base' in (a,b) else ('cross_leg' if a.split('_')[0]!=b.split('_')[0] else 'same_leg_nonadjacent')
                categories[name].append(i)
                key=' / '.join(sorted((a,b)))
                unique[key]=max(unique.get(key,0.),float(force[:,i].max()))
            summary['contact_max_n']={k:float(force[:,v].max()) for k,v in categories.items()}
            summary['unique_pair_max_n']=unique
            f,i=np.unravel_index(force.argmax(),force.shape)
            summary['peak_contact']=dict(pair=pairs[i],time_s=float(raw['state_stamp_ns'][f]*1e-9) if 'state_stamp_ns' in raw else f*.005,
                                         force_n=float(force[f,i]))
            summary['maximum_body_net_force_n']=float(np.linalg.norm(raw['body_net_force'],axis=-1).max())
            summary['joint_drive_limits_nm']=[24.8,48.,48.]*4
            summary['peak_drive_limit_exceed_fraction']=np.mean(abs(raw['estimated_torque'])>np.array(summary['joint_drive_limits_nm'])+1e-4,axis=0).tolist()
            summary['depth_policy_samples']=len(json.loads((run/'result.depth.json').read_text()))
            contact_text=' · '.join(f'{k}: {v:.3f} N' for k,v in summary['contact_max_n'].items())
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            t=raw['state_stamp_ns']*1e-9 if 'state_stamp_ns' in raw else np.arange(len(raw['position']))*.005
            fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
            axes[0].plot(t,raw['position'][:,1],label='Forward +Y');axes[0].plot(t,raw['position'][:,2],label='Trunk height')
            axes[0].set_ylabel('Position (m)');axes[0].legend()
            axes[1].plot(t,raw['velocity'][:,1],label='COM body velocity +Y');axes[1].axhline(.4,color='k',ls='--',label='Command')
            axes[1].set_ylabel('Speed (m/s)');axes[1].legend()
            for k,v in categories.items():axes[2].plot(t,force[:,v].max(axis=1),label=k)
            axes[2].axhline(1.,color='k',ls='--');axes[2].set_ylabel('Self contact (N)');axes[2].legend()
            for n in range(4):axes[3].plot(t,raw['estimated_torque'][:,n*3+2],label=['FL','FR','RL','RR'][n]+' KFE')
            axes[3].set_ylabel('PD estimate (N m)');axes[3].set_xlabel('State time (s)');axes[3].legend(ncol=4)
            for ax in axes:ax.grid(alpha=.25)
            fig.savefig(directory/'telemetry.png',dpi=160);plt.close(fig)
            plot='<img src="telemetry.png" alt="보행 속도, 자세, 자가 접촉, 무릎 토크" style="width:100%">'
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    success=bool(result.get('task_success'))
    state='통과' if success else '미통과'
    rows=[]
    for i,name in enumerate(result.get('joint_order',[])):
        rows.append('<tr><td>'+name+'</td>'+''.join(f'<td>{result[k][i]:.2f}</td>' for k in
                    ('torque_rms_nm','torque_peak_nm','ideal_current_rms_a'))+'</tr>')
    torque_table='<table><tr><th>관절</th><th>토크 RMS (Nm)</th><th>토크 피크 (Nm)</th><th>추정 전류 RMS (A)</th></tr>'+''.join(rows)+'</table>' if rows else ''
    video=f'<video controls preload="metadata" src="runs/{job}/video/env_00.mp4" style="width:100%"></video>' if (run/'video/env_00.mp4').exists() else ''
    mapping='<table><tr><th>다리</th><th>CAD 다리</th><th>HAA 부호</th><th>HFE 부호</th><th>KFE 부호</th></tr>'+''.join(
        f'<tr><td>{a}</td><td>{b}</td><td>{s[0]:+d}</td><td>{s[1]:+d}</td><td>{s[2]:+d}</td></tr>'
        for a,b,s in [('FL','FL',(-1,-1,1)),('FR','FR',(-1,1,-1)),('RL','BL',(1,-1,1)),('RR','BR',(1,1,-1))])+'</table>'
    number=lambda k: f'{result[k]:.4f}' if k in result else '—'
    document=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V15 새 CAD USD 검증</title>
    <style>body{{font:16px/1.6 sans-serif;max-width:1000px;margin:30px auto;padding:0 18px;color:#192633;background:#f9fafc}}
    h1,h2{{line-height:1.3}} table{{border-collapse:collapse;width:100%;background:white;margin:16px 0}}th,td{{padding:7px;border:1px solid #d5dce5;text-align:left}}
    pre{{overflow:auto;padding:16px;background:#eef1f5}} .state{{font-size:24px;color:{'#146b40' if success else '#af2626'}}}</style>
    <h1>최신 채택 V15 × 새 CAD USD</h1><p class="state">보행·자가 접촉 종합: {state}</p>
    <p>Slurm {job} · ws1 · compute/admin · ROS2 Jazzy · Isaac Sim · 명령 +Y 0.4 m/s · 물리 200 Hz / 정책 50 Hz</p>
    <p>USD: quadruped_20261011/quadruped_robot.usda. 새 질량 {summary['usd_authored_mass_kg']:.3f} kg, 기존 질량 {summary['old_mass_kg']:.3f} kg.
    정책은 기존 채택 V15 job339/model_11699, depth 학생은 V12 best.pt를 사용했다.</p>
    <table><tr><th>측정</th><th>결과</th><th>사전에 지정한 기준</th></tr>
    <tr><td>기록 물리 구간</td><td>{number('seconds_recorded')} s</td><td>요청 14 s 전부 완료</td></tr>
    <tr><td>전방 진행</td><td>{number('progress_y_m')} m</td><td>14 s에서 ≥ 3.6 m</td></tr>
    <tr><td>속도 MAE (첫 2 s 이후)</td><td>{number('command_mae_m_s')} m/s</td><td>≤ 0.15 m/s</td></tr>
    <tr><td>몸통 최저 높이</td><td>{number('torso_min_height_m')} m</td><td>≥ 0.20 m, 기울기 ≤ 60°</td></tr>
    <tr><td>비인접 쌍 최대 접촉</td><td>{number('max_nonadjacent_contact_force_n')} N</td><td>≤ 1 N</td></tr></table>
    <p>실행 오류: {html.escape(str(result.get('failure')))}</p>{video}{plot}
    <p>속도 오차 {number('command_mae_m_s')} m/s는 0.1500 m/s 기준에 가깝다. 측정한 첫 2 s 이후 평균 전방 속도는
    {float(raw['velocity'][400:,1].mean()):.4f} m/s로 0.4 m/s 명령보다 빠르다. 이 실행만으로 속도 추종의 충분한 여유를 확인한 것은 아니다.</p>
    <h2>자가 충돌 범위</h2><p>{html.escape(contact_text)}</p>
    <p>CAD 원본의 67개 convexHull을 사용하고 self-collision을 켰다. 13개 rigid body 중 비인접 66쌍을 양방향 접촉 센서로 확인했다.
    PhysX가 기본적으로 제외하는 직접 연결 관절 12쌍은 이 판정에 포함되지 않는다. 접촉력 1 N 이하 판정은 이 실행 구간의 관측 결과다.</p>
    <h2>관절 토크·전류</h2>{torque_table}
    <p>통계는 실제 기록 구간 전체의 값이다. 토크는 명령 적용 직전의 cap 제한 implicit PD 추정치이며 PhysX에서 측정한 구동 토크가 아니다.
    전류는 τ/(Kt×감속비), 효율 1 가정으로 HAA 1.23, HFE/KFE 1.44 Nm/A를 사용했다. 배터리 입력 전류와 다르다.</p>
    <h2>USD 연결</h2>{mapping}<p>관절 위치·속도·토크 모두 같은 부호로 변환한다. 새 base COM을 USD에서 읽어 원점 속도를 COM 속도로 보정한다.
    모델 질량·관성·충돌 형상을 원본대로 사용했고, 구동기 설정은 V15의 Kp 300/Kd 8, HAA 24.8/HFE·KFE 48 Nm, armature 0.02, friction 0.3을 적용했다.
    카메라는 V15의 84×84 depth 장착 위치와 방향으로 런타임에 추가했다.</p>
    <h2>재실행</h2><pre>ssh -o ClearAllForwardings=yes ws1
    sbatch --partition=compute --qos=admin /home/shanghun1404/v15_cad_20261011/config/experiments/v15_cad_20261011/eval.sbatch 14 42</pre>
    <p>ROS2 패키지 빌드 후: <code>ros2 launch v151_ros2_control v15_cad.launch.py actor_checkpoint:=... student_checkpoint:=...</code>
    Isaac 센서/브리지 namespace는 /v15cad, 걷기 명령은 /v15cad/cmd_vel의 linear.y이다.</p>
    <p><a href="summary.json">요약 JSON</a> · <a href="profile.json">프로파일·해시</a> · <a href="runs/{job}/result.json">원본 결과</a> ·
    <a href="runs/{job}/result.npz">원본 시계열</a> · <a href="audit/usd_cpu_audit.json">USD 비교</a> ·
    <a href="verification/walk_green.txt">실제 DDS 정책 동등성</a> · <a href="verification/command_types_green.txt">매핑·명령 자료형 테스트</a></p>
    <p>검증 범위는 선 자세에서 시작한 평지 V15 보행이다. 새 USD의 앉았다 일어서기, 계단, 실물 구동 결과는 이 보고서에 포함되지 않는다.</p></html>'''
    (directory/'report.html').write_text(document)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=int);a=parser.parse_args()
    build(Path(__file__).resolve().parent,a.job)
