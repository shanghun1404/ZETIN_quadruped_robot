"""Statistics for joint output torque and ideal torque-producing current.

Current = output torque / (motor Kt * gear ratio), efficiency assumed 1.
This is not a DC bus current model. ADC ratings are not used as hard thresholds.
"""
import numpy as np


def joint_statistics(torque, requested, valid, dt, rated, sim_limit, peak, nm_per_a):
    torque=np.asarray(torque, dtype=float)
    requested=np.asarray(requested, dtype=float)
    valid=np.asarray(valid, dtype=bool)
    if torque.ndim != 3 or requested.shape != torque.shape or valid.shape != torque.shape[:2]:
        raise ValueError('Expected torque [time,env,joint], valid [time,env]')
    if not valid.any() or not np.isfinite(torque[valid]).all() or not np.isfinite(requested[valid]).all():
        raise ValueError('No valid finite torque samples')
    if dt<=0 or not np.isfinite(dt):
        raise ValueError('Invalid sample interval')
    arrays=[np.asarray(a, float) for a in (rated, sim_limit, peak, nm_per_a)]
    if any(a.shape!=(torque.shape[2],) or not np.isfinite(a).all() or (a<=0).any() for a in arrays):
        raise ValueError('Invalid motor spec arrays')
    rated,sim_limit,peak,nm_per_a=arrays
    rows=[]
    for j in range(torque.shape[2]):
        a=np.abs(torque[:,:,j]); x=a[valid]; req=np.abs(requested[:,:,j])[valid]
        excess=(a>rated[j]+1e-5)&valid
        runs=np.zeros(valid.shape[1],int); longest=0
        for step in excess:
            runs=np.where(step, runs+1, 0)
            longest=max(longest,int(runs.max()))
        rms=float(np.sqrt(np.mean(x*x)))
        row=dict(valid_samples=int(valid.sum()), sample_env_seconds=float(valid.sum()*dt),
                 torque_mean_abs_nm=float(x.mean()),torque_rms_nm=rms,
                 torque_p95_nm=float(np.percentile(x,95)), torque_p99_nm=float(np.percentile(x,99)),
                 torque_peak_nm=float(x.max()),requested_peak_nm=float(req.max()),
                 rated_torque_nm=float(rated[j]),sim_limit_nm=float(sim_limit[j]),motor_peak_nm=float(peak[j]),
                 rated_exceed_fraction=float(np.mean(x>rated[j]+1e-5)),
                 sim_limit_exceed_fraction=float(np.mean(x>sim_limit[j]+1e-4)),
                 motor_peak_exceed_fraction=float(np.mean(x>peak[j]+1e-4)),
                 requested_sim_limit_exceed_fraction=float(np.mean(req>sim_limit[j]+1e-4)),
                 rated_exceed_max_contiguous_s=longest*dt,
                 rms_over_rated=bool(rms>rated[j]),
                 current_mean_abs_ideal_a=float(x.mean()/nm_per_a[j]),
                 current_rms_ideal_a=rms/nm_per_a[j],
                 current_peak_ideal_a=float(x.max()/nm_per_a[j]),
                 current_model_nm_per_a=float(nm_per_a[j]))
        rows.append(row)
    return rows
