"""Verify generated study results and write the compact candidate summary."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).parent
x=json.loads((p/'search.json').read_text())
y=json.loads((p/'schedules.json').read_text())
rows=x['rows']+[r for s in y['schedules'] for r in s['rows']]+y['initial_crouch_shift_path']
assert len(x['rows'])==312 and len(rows)==454
assert all(r['ik_converged'] and r['lp_feasible'] and r['wbc_feasible'] for r in rows)
assert max(r['ik_max_error_m'] for r in rows)<1e-8
# LP solver default feasibility tolerance is 1e-7.
assert max(r['lp_base_equilibrium_error'] for r in rows)<1e-7
assert all(r['clearance_and_support_screen_pass'] for r in rows)
for r in rows:
    forces=np.array(r['lp_forces_world_N'])
    assert min(forces[:,2])>=5-1e-7
    assert np.max(np.abs(forces[:,:2])-.6/np.sqrt(2)*forces[:,2,None])<1e-7
assert x['contexts'][1]['final_measured_material_point_max_error_m']<.0017
comparison=[]
for context in ('nominal','601_final'):
    for height,old,new in ((.464,.03,.09),(.364,0.,.15)):
        a=next(r for r in x['rows'] if r['context']==context and abs(r['height_m']-height)<1e-5 and r['forward_shift_m']==old)
        b=next(r for r in x['rows'] if r['context']==context and abs(r['height_m']-height)<1e-5 and r['forward_shift_m']==new)
        comparison.append(dict(context=context,height_m=height,baseline_shift_m=old,candidate_shift_m=new,
            baseline_wbc_max_rated_fraction=a['wbc_max_rated_fraction'],
            candidate_wbc_max_rated_fraction=b['wbc_max_rated_fraction'],
            max_rated_fraction_reduction_percent=100*(1-b['wbc_max_rated_fraction']/a['wbc_max_rated_fraction']),
            candidate_wbc_torque_nm=b['wbc_torque_nm'],candidate_lp_max_rated_fraction=b['lp_max_rated_fraction'],
            candidate_min_normal_force_N=b['wbc_min_normal_force_N'],candidate_minimum_gaps_m=b['minimum_gaps_m']))
summary=dict(recommended_static_candidate=dict(stand_forward_shift_m=.09,crouch_forward_shift_m=.15,
    crouch_depth_m=.10,shift_formula='0.15-0.06*height_fraction',
    reason='minimum WBC max-rated fraction within requested stand/crouch ranges for both support geometries'),
    physical_success_claimed=False,comparison=comparison,
    schedules=[{k:v for k,v in s.items() if k!='rows'} for s in y['schedules']],
    numerical_verification=dict(pose_evaluations=len(rows),all_ik_lp_wbc_feasible=True,
        max_ik_error_m=max(r['ik_max_error_m'] for r in rows),
        max_lp_base_equilibrium_error=max(r['lp_base_equilibrium_error'] for r in rows),
        min_wbc_normal_force_N=min(r['wbc_min_normal_force_N'] for r in rows),
        min_wbc_friction_margin_N=min(r['wbc_friction_pyramid_margin_N'] for r in rows),
        strict_nonnegative_trunk_proxy_candidates=sum(r['minimum_gaps_m']['trunk_proxy']>=0 for r in rows)),
    risks=['Current WBC final601 crouch still reaches 21.60Nm FL_KFE (1.200 times 18Nm rated).',
           'Static LP lower bound 0.9015 does not establish production WBC allocation or physical feasibility.',
           'Every scanned pose has existing trunk-proxy overlap; strict nonnegative-all-proxy requirement has no candidate.',
           'Large initial 15cm forward shift requires a validated transition; static samples cannot establish dynamic safety.',
           'Stand 9cm is at the requested search boundary; policy handover may change and requires full 28s physical validation.',
           'Motor hardware joint limits and thermal endurance are not validated.',
           'Static601 baseline is a rest-state model counterfactual; it is not the recorded physical torque.'])
(p/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary['numerical_verification'],indent=2))
print('comparison',[(r['context'],r['height_m'],round(r['max_rated_fraction_reduction_percent'],2)) for r in comparison])
