import json
from pathlib import Path

import numpy as np


def test_flat_regression_skips_rough_and_cannot_select_candidate(tmp_path):
    from targeted_compare import load_api, summarize
    api=load_api(Path(__file__).resolve().parents[3])
    for label,torque in [('baseline339',1.),('candidate25',2.)]:
        run=tmp_path/f'{label}_flat_seed42';run.mkdir()
        (run/'result.json').write_text(json.dumps(dict(joint_order=api.JOINT_ORDER,completed=True,failure=None)))
        (run/'exit_code.txt').write_text('0')
        n=2800
        position=np.zeros((n,3));position[:,1]=np.arange(n)*.002;position[:,2]=.45
        np.savez(run/'result.npz',position=position,quaternion=np.tile([1.,0,0,0],(n,1)),
            velocity=np.tile([0,.4,0],(n,1)),estimated_torque=np.full((n,12),torque),
            joint_velocity=np.ones((n,12)),gaps=np.full((n,24),.1),trunk_gaps=np.full((n,8),-.006),
            contact_pair_force=np.zeros((n,12)))
    result=summarize(tmp_path,api)
    assert result['selected'] is None
    assert result['candidates']['candidate25']['status']=='rejected_flat'
    assert result['candidates']['candidate25']['rough']=='skipped_after_flat_rejection'
