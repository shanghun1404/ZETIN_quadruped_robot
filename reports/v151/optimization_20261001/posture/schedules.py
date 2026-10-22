"""Screen continuous height-dependent forward-reference paths geometrically."""
import json
import numpy as np
from search import OUT, SOURCE, MODEL, StandingController, contexts, evaluate


def main():
    controller=StandingController(MODEL)
    all_contexts=contexts(controller)
    boxes=json.loads((SOURCE/'result.json').read_text())['same_leg_thigh_foot_geometry']
    schedules=[]
    for stand,crouch in ((.06,.12),(.09,.15),(.06,.15)):
        rows=[]
        for context in all_contexts:
            for fraction in np.linspace(0.,1.,21):
                height=controller.nominal_height-.10+.10*fraction
                shift=crouch+(stand-crouch)*fraction
                row=evaluate(controller,context,height,shift,boxes)
                row['height_fraction']=float(fraction)
                rows.append(row)
        schedules.append(dict(stand_shift_m=stand,crouch_shift_m=crouch,rows=rows,
            all_ik_converged=all(r['ik_converged'] for r in rows),
            all_support_and_clearance_screens_pass=all(r.get('clearance_and_support_screen_pass') for r in rows),
            worst_wbc_max_rated_fraction=max(r['wbc_max_rated_fraction'] for r in rows),
            minimum_wbc_normal_force_N=min(r['wbc_min_normal_force_N'] for r in rows),
            maximum_ik_jacobian_condition=max(r['ik_jacobian_condition'] for r in rows)))
        print('screened',stand,crouch,flush=True)
    initial=[]
    for shift in np.linspace(0.,.15,16):
        initial.append(evaluate(controller,all_contexts[0],controller.nominal_height-.10,shift,boxes))
    result=dict(scope='21 rest-state samples per height/shift path, not a dynamic transition',
                shift_formula='crouch_shift+(stand_shift-crouch_shift)*height_fraction',
                fixed_height_drop_m=.10,schedules=schedules,initial_crouch_shift_path=initial,
                transition_acceleration_validated=False)
    (OUT/'schedules.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for s in schedules:
        print({k:v for k,v in s.items() if k!='rows'})


if __name__=='__main__':
    main()
