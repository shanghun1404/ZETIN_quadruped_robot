"""Validate the extracted deployment runtime on native Jazzy, without simulator imports.

CPU/x86 verification only. Synthetic policy inputs check shape and finite
inference; they are not a physical or real sensor evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist

from v14_ros2_control import policy as policy_module
from v151_ros2_control import cad_standing
from v15_hardware_bridge.cad_controller import HardwareController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    root = parser.parse_args().bundle.resolve()
    actor = root / 'checkpoints/model_15699.pt'
    student = root / 'checkpoints/v12_student_best.pt'
    model = root / 'model/quadruped_cad.xml'
    for module in (policy_module, cad_standing):
        assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
    policy = policy_module.V14DepthPolicy(actor, student, device='cpu')
    action, target = policy.step(joint_pos=policy.default_pos, joint_vel=[0.] * 12,
        lin_vel_b=[0.] * 3, ang_vel_b=[0.] * 3, quat_wxyz=[1., 0., 0., 0.],
        command=[0.] * 3, depth_m=np.full((84, 84), 2., dtype=np.float32))
    assert action.shape == target.shape == (12,)
    assert np.isfinite(action.numpy()).all() and np.isfinite(target.numpy()).all()
    standing = cad_standing.CadStandingController(model)
    assert standing.model.body_count == 13
    assert abs(standing.model.total_mass - 23.09449815750122) < 1e-10
    rclpy.init(args=['--ros-args', '-p', f'actor_checkpoint:={actor}',
        '-p', f'student_checkpoint:={student}', '-p', f'model_path:={model}',
        '-p', 'stand_forward_shift_m:=0.0', '-p', 'crouch_forward_shift_m:=0.0'])
    node = HardwareController()
    try:
        assert node.initial == 'SIT' and node.demo is False
        assert isinstance(node.standing, cad_standing.CadStandingController)
        command = Twist(); command.linear.y = .1
        node.command(command)
        assert node.supervisor.requested == 'walk'
        node.check_operator()
        assert node.supervisor.requested == 'walk'
        time.sleep(.55)
        node.check_operator()
        assert node.supervisor.requested == 'stand' and node.request_command == [0.] * 3
    finally:
        node.destroy_node(); rclpy.shutdown()
    imported_simulator = sorted(name for name in sys.modules
        if name.split('.')[0] in {'isaaclab', 'isaaclab_rl', 'isaacsim', 'omni', 'pxr'})
    assert not imported_simulator, imported_simulator
    print(json.dumps(dict(passed=True, actor_input=235, action_output=12,
        student_depth_shape=[84, 84], physical_bodies=13,
        mass_kg=standing.model.total_mass, controller_initial='SIT',
        operator_timeout_checked_s=.5, simulator_imports=imported_simulator,
        scope='native x86 CPU/Jazzy initialization and synthetic finite inference; Jetson ARM timing not measured'), indent=2))


if __name__ == '__main__':
    main()
