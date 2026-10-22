"""Deep-crouch references retain level body, foot anchors and joint limits."""
from pathlib import Path
import sys
import importlib.util
import json
import os
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


@pytest.mark.skipif(importlib.util.find_spec('rclpy') is None, reason='Native ROS2 runs on ws1 CPU')
def test_deep_crouch_startup_preserves_standing_assumption_and_measured_gate():
    import rclpy
    from sensor_msgs.msg import JointState,Image
    from nav_msgs.msg import Odometry
    from v14_ros2_control import contract
    import v151_ros2_control.controller_node as controller
    from v151_ros2_control.cad_standing import CadStandingController
    actor=Path(os.environ.get('V15_ACTOR',ROOT/'train_jobs/2341/results/model_15699.pt'))
    student=Path(os.environ.get('V15_STUDENT','/home/projects/legacy/quadruped_training_smooth_v11/runs/v12_distillation_dagger/best.pt'))
    assert actor.is_file() and student.is_file()
    original=controller.StandingController;controller.StandingController=CadStandingController
    rclpy.init(args=['--ros-args','-p','actor_checkpoint:='+str(actor),'-p','student_checkpoint:='+str(student),
        '-p','model_path:='+str(ROOT/'config/experiments/v15_cad_20261011/model/quadruped_cad.xml'),
        '-p','initial_mode:=STAND','-p','sit_style:=deep_crouch','-p','height_drop:=0.18',
        '-p','posture_transition_s:=2.0','-p','stand_forward_shift_m:=0.0','-p','crouch_forward_shift_m:=0.0'])
    node=controller.ControllerNode()
    try:
        node.supervisor.request('sit')
        odom=Odometry();odom.pose.pose.position.z=node.standing.nominal_height+.006
        odom.pose.pose.orientation.w=1.
        image=Image();image.height=image.width=84;image.encoding='32FC1';image.step=336
        image.data=np.full((84,84),2.,dtype='<f4').tobytes()
        contact=JointState();contact.effort=[0.]*4
        state=JointState();state.name=list(contract.JOINT_NAMES)
        state.position=[contract.joint_default(name) for name in state.name];state.velocity=[0.]*12
        node.add(node.odoms,odom);node.add(node.images,image);node.add(node.contacts,contact);node.on_state(state)
        assert node.support_feet==0
        assert node.last_diag['support_count']==4
        assert node.output['mode']=='STAND'  # elapsed time alone cannot lower
        assert np.isfinite(node.stand_tau).all()
    finally:
        node.destroy_node();rclpy.shutdown();controller.StandingController=original
