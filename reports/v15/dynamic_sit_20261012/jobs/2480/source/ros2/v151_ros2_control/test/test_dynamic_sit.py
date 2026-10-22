"""Deep-crouch references retain level body, foot anchors and joint limits."""
from pathlib import Path
import sys
import importlib.util
import json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'config'),str(ROOT/'ros2/v151_ros2_control')]

@pytest.mark.skipif(importlib.util.find_spec('pinocchio') is None, reason='CAD IK runs on ws1 CPU')
def test_deep_crouch_path_preserves_four_feet_and_returns_to_standing():
    from v151_ros2_control.cad_standing import CadStandingController
    ctl=CadStandingController(ROOT/'config/experiments/v15_cad_20261011/model/quadruped_cad.xml')
    q=ctl.nominal_joint_position.copy();base=np.array([0.,0.,ctl.nominal_height]);quat=[1.,0.,0.,0.]
    ctl.initialize(q,base,quat);anchors=ctl.contact_anchors.copy()
    for fraction in np.r_[np.linspace(1.,0.,21),np.linspace(0.,1.,21)]:
        height=ctl.nominal_height-.18*(1-fraction)
        q,goal=ctl.contact_consistent_reference(q,base,quat,height,np.ones(4,dtype=bool),center_body=True)
        np.testing.assert_allclose(ctl.contact_positions(q,goal,quat),anchors,atol=2e-7)
        assert np.all(np.abs(q.reshape(4,3))<=[np.pi/4,2*np.pi/3,7*np.pi/9])
        if fraction==0.:
            assert height==pytest.approx(ctl.nominal_height-.18)
            authored=json.loads((ROOT/'config/experiments/v15_cad_20261011/model/crouch.json').read_text())
            np.testing.assert_allclose(q,authored['deep_crouch']['q_canonical_rad'],atol=2e-6)
            assert q[[2,5,8,11]].min()>1.90
            assert np.ptp(q[[2,5,8,11]])<.01
    np.testing.assert_allclose(q,ctl.nominal_joint_position,atol=2e-6)
