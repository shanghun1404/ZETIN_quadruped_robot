"""Measured criteria for the 14-second stand -> deep crouch -> stand trial."""
import numpy as np


def dynamic_sit_verdict(name, records, history, completed, stand_height, sit_height):
    reasons=[]
    modes=list(dict.fromkeys(s['mode'] for s in history))
    if not completed: reasons.append('did_not_complete_requested_duration')
    expected=['STAND','LOWER','SETTLE_SIT','SIT','RISE','STABILIZE']
    missing=[m for m in expected if m not in modes]
    if missing: reasons.append('missing_modes:'+','.join(missing))
    if not history or history[-1]['mode']!='STAND': reasons.append('did_not_finish_in_STAND')
    t=records['state_stamp_ns']*1e-9
    quat=records['quaternion'];w,x,y,z=quat.T
    r21=2*(y*z+w*x);r22=1-2*(x*x+y*y)
    pitch=np.arctan2(r21,r22)
    fold=records['joint_position'][:,[8,11]].mean(axis=1)-records['joint_position'][:,[2,5]].mean(axis=1)
    supported=np.linalg.norm(records['shank_support_force'],axis=2)>5.
    metrics={}
    for label,mask,target in [('seated',(t>=5.5)&(t<8.),sit_height),('standing',t>=11.5,stand_height)]:
        if not mask.any():
            reasons.append('missing_'+label+'_hold_samples');continue
        height_error=float(abs(records['position'][mask,2]-target).max())
        speed=float(np.linalg.norm(records['velocity'][mask],axis=1).max())
        omega=float(np.linalg.norm(records['angular_velocity'][mask],axis=1).max())
        support_fraction=float(np.mean(supported[mask].all(axis=1)))
        metrics[label]=dict(max_height_error_m=height_error,max_speed_m_s=speed,
            max_angular_speed_rad_s=omega,four_foot_support_fraction=support_fraction,
            pitch_deg_range=np.degrees([pitch[mask].min(),pitch[mask].max()]).tolist(),
            median_rear_minus_front_knee_rad=float(np.median(fold[mask])))
        if height_error>.035: reasons.append(label+'_height_error_exceeds_0.035m')
        if speed>.12: reasons.append(label+'_speed_exceeds_0.12m_s')
        if omega>.35: reasons.append(label+'_angular_speed_exceeds_0.35rad_s')
        if support_fraction<.95: reasons.append(label+'_four_foot_support_below_95percent')
        if label=='seated':
            knees=records['joint_position'][mask][:,[2,5,8,11]]
            metrics[label]['median_knee_flexion_rad']=np.median(knees,axis=0).tolist()
            # Authored 18cm CAD IK: all knees 1.9631..1.9635 rad;
            # authored 10cm crouch: 1.6263..1.6266 rad. Allow 0.063rad tracking error.
            if np.median(knees,axis=0).min()<1.90: reasons.append('knees_not_deeply_folded')
            if abs(np.median(fold[mask]))>.15: reasons.append('crouch_not_symmetric_front_rear')
        if r22[mask].min()<.97: reasons.append(label+'_tilt_exceeds_stability_gate')
    return dict(motion_success=not reasons,reasons=reasons,modes_seen=modes,metrics=metrics)
