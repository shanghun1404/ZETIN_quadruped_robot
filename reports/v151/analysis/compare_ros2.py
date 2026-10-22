"""Analyze frozen ROS2 comparison runs; fail closed on missing/failed rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

JOINT_ORDER = tuple(f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE'))
RATED = np.tile([8.3,18.,18.],4)
# reports/motor_audit_20260930/specs/motors.json: Kt*gear ratio, eta=1.
NM_PER_A = np.tile([.123*10,.16*9,.16*9],4)


def metrics_from_arrays(arrays, completed, seconds=14.):
    required=('position','quaternion','velocity','estimated_torque','joint_velocity','gaps','trunk_gaps','contact_pair_force')
    if any(name not in arrays for name in required):
        return dict(valid=False,survived=False,error='missing raw array')
    raw={k:np.asarray(arrays[k],dtype=float) for k in required}
    count=len(raw['position'])
    if not count or any(len(v)!=count for v in raw.values()):
        return dict(valid=False,survived=False,error='empty or mismatched raw arrays')
    finite=np.logical_and.reduce([np.isfinite(v).reshape(count,-1).all(axis=1) for v in raw.values()])
    bad=np.flatnonzero(~finite)
    n=int(bad[0]) if len(bad) else count
    if n==0:
        return dict(valid=False,survived=False,error='nonfinite first sample')
    a={k:v[:n] for k,v in raw.items()}
    tau=np.abs(a['estimated_torque'])
    if tau.shape!=(n,12) or a['joint_velocity'].shape!=(n,12):
        return dict(valid=False,survived=False,error='expected 12 canonical motor columns')
    rms=np.sqrt(np.mean(tau*tau,axis=0)); peak=tau.max(axis=0)
    quat=a['quaternion'];norm=np.linalg.norm(quat,axis=1)
    if (norm==0).any():
        return dict(valid=False,survived=False,error='zero quaternion')
    w,x,y,z=(quat/norm[:,None]).T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    exceed=tau>RATED
    longest=[]
    for joint in range(12):
        edges=np.diff(np.r_[False,exceed[:,joint],False].astype(int))
        spans=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
        longest.append(float(spans.max(initial=0)*.005))
    return dict(valid=not len(bad),survived=bool(completed and not len(bad) and n*.005>=seconds-.02),
        samples_used=n,recorded_seconds=n*.005,requested_seconds=seconds,
        first_nonfinite_sample=int(bad[0]) if len(bad) else None,
        progress_world_y_m=float(a['position'][-1,1]-a['position'][0,1]),
        command_speed_mae_m_s=float(np.mean(np.abs(a['velocity'][:,1]-.4))),
        mean_forward_speed_body_m_s=float(a['velocity'][:,1].mean()),
        maximum_lateral_deviation_m=float(np.max(np.abs(a['position'][:,0]-a['position'][0,0]))),
        maximum_heading_error_deg=float(np.rad2deg(np.max(np.abs(yaw-yaw[0])))),
        maximum_tilt_deg=float(np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y),-1,1))).max()),
        torque_rms_nm=rms.tolist(),mean_joint_torque_rms_nm=float(rms.mean()),
        maximum_joint_torque_rms_nm=float(rms.max()),torque_peak_nm=peak.tolist(),
        torque_peak_max_nm=float(peak.max()),torque_p99_max_nm=float(np.quantile(tau,.99,axis=0).max()),
        current_rms_ideal_a=(rms/NM_PER_A).tolist(),current_peak_ideal_a=(peak/NM_PER_A).tolist(),
        maximum_joint_current_rms_ideal_a=float((rms/NM_PER_A).max()),
        mean_absolute_power_W=float(np.mean(np.sum(tau*np.abs(a['joint_velocity']),axis=1))),
        rated_exceed_fraction=exceed.mean(axis=0).tolist(),rated_exceed_max_contiguous_s=longest,
        minimum_interleg_gap_m=float(a['gaps'].min()),interleg_negative_frame_fraction=float((a['gaps'].min(axis=1)<0).mean()),
        maximum_crossleg_contact_force_N=float(a['contact_pair_force'].max()),
        crossleg_contact_over_1N_frame_fraction=float((a['contact_pair_force'].max(axis=1)>1.).mean()),
        minimum_trunk_proxy_gap_m=float(a['trunk_gaps'].min()),
        trunk_proxy_overlap_frame_fraction=float((a['trunk_gaps'].min(axis=1)<0).mean()))


def compare_pair(baseline,candidate):
    if not baseline.get('valid') or not candidate.get('valid'):
        return dict(all_gates_pass=False,gates={'valid_both':False})
    gates={
        'survival':bool(baseline['survived'] and candidate['survived']),
        'progress':bool(baseline['progress_world_y_m']>0 and candidate['progress_world_y_m']>=.9*baseline['progress_world_y_m']-1e-12),
        'command_tracking':candidate['command_speed_mae_m_s']<=baseline['command_speed_mae_m_s']+.02+1e-12,
        'rms_not_increased':candidate['mean_joint_torque_rms_nm']<=baseline['mean_joint_torque_rms_nm']+1e-12,
        'power_not_increased':candidate['mean_absolute_power_W']<=baseline['mean_absolute_power_W']+1e-12,
        'some_torque_improvement':any(candidate[k]<baseline[k]-1e-9 for k in
             ('mean_joint_torque_rms_nm','mean_absolute_power_W','torque_peak_max_nm','torque_p99_max_nm')),
        'self_collision_contact':candidate.get('maximum_crossleg_contact_force_N',float('inf'))<=1.,
        'self_collision_geometry':candidate.get('minimum_interleg_gap_m',float('-inf'))>=-.002,
    }
    return dict(all_gates_pass=all(gates.values()),gates=gates,
                mean_joint_rms_ratio=candidate['mean_joint_torque_rms_nm']/max(baseline['mean_joint_torque_rms_nm'],1e-12),
                absolute_power_ratio=candidate['mean_absolute_power_W']/max(baseline['mean_absolute_power_W'],1e-12))


def load_run(path):
    try:
        result=json.loads((path/'result.json').read_text())
        if tuple(result['joint_order'])!=JOINT_ORDER:
            raise ValueError('unreviewed joint ordering')
        exit_code=int((path/'exit_code.txt').read_text())
        with np.load(path/'result.npz',allow_pickle=False) as arrays:
            metrics=metrics_from_arrays(arrays,bool(result.get('completed') and not result.get('failure') and exit_code==0))
        metrics.update(failure=result.get('failure'),exit_code=exit_code,
                       result_sha256=hashlib.sha256((path/'result.json').read_bytes()).hexdigest(),
                       raw_sha256=hashlib.sha256((path/'result.npz').read_bytes()).hexdigest())
        return metrics
    except Exception as error:
        return dict(valid=False,survived=False,error=repr(error))


def analyze(root,mode,selected=None):
    labels=('baseline339','candidate250','candidate500') if mode=='screen' else ('baseline339',selected)
    seeds=(42,) if mode=='screen' else (43,44)
    runs={f'{label}_{terrain}_seed{seed}':load_run(root/f'{label}_{terrain}_seed{seed}')
          for seed in seeds for label in labels for terrain in ('flat','rough')}
    comparisons={}
    for label in labels[1:]:
        comparisons[label]={f'{terrain}_seed{seed}':compare_pair(runs[f'baseline339_{terrain}_seed{seed}'],runs[f'{label}_{terrain}_seed{seed}'])
                            for seed in seeds for terrain in ('flat','rough')}
    passed=[label for label,values in comparisons.items() if all(v['all_gates_pass'] for v in values.values())]
    chosen=min(passed,key=lambda label:sum(v['mean_joint_rms_ratio']+v['absolute_power_ratio'] for v in comparisons[label].values())) if passed else None
    report=dict(mode=mode,seeds=seeds,selected=chosen,all_gates_pass=bool(chosen),runs=runs,comparisons=comparisons,
        gate_source='docs/v151_plan.md; require complete matched rollouts and at least one torque metric to improve',
        self_collision_gate=dict(frozen_date='2026-10-01',frozen_phase='before ROS2 comparative physics results',
            maximum_crossleg_contact_force_N=1.,minimum_interleg_gap_m=-.002,trunk_proxy_gates_adoption=False),
        torque_semantics='implicit PD estimates, not measured PhysX motor torque',
        current_model='abs(torque)/(Kt*gear); HAA1.23Nm/A,HFE/KFE1.44Nm/A,eta1; not battery current or thermal certification',
        collision_semantics='crossleg force includes feet; geometric capsule gaps/trunk proxy are independent of enabled collision pairs; negative trunk gap is not penetration depth',
        failed_sample_policy='stop at first nonfinite sample; evaluator forbids resets and stops at first runtime failure; failed rollouts cannot pass',
        heldout_selection='candidate fixed by seed42; seeds43/44 cannot choose a replacement')
    (root/'comparison_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/'selected.txt').write_text((chosen or '')+'\n')
    print(json.dumps({'mode':mode,'selected':chosen,'all_gates_pass':bool(chosen)}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--mode',choices=['screen','holdout'],default='screen')
    p.add_argument('--selected',choices=['candidate250','candidate500'])
    args=p.parse_args()
    if args.mode=='holdout' and not args.selected: p.error('holdout requires screening-selected candidate')
    analyze(args.root,args.mode,args.selected)
