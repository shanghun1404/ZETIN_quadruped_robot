"""Isaac Sim side of the V14 ROS2 loop: publish robot state, apply ROS2 position targets.

The IsaacLab action term no longer runs a policy, filter or actuator model. At
every 200 Hz physics step it publishes joint states (and, on each 50 Hz policy
tick, odometry and the V12 depth image), waits for the ROS2 joint command with
the same simulation stamp, and writes it as the PhysX joint-drive target. PD
control is done by the PhysX drive (``implicit_joint_drives``).

The actions passed to ``env.step`` come from the in-process V14 policy and are
never applied: they shadow the ROS2 nodes. ``parity_summary`` reports how far
the ROS2 target and command are from that policy followed by the V13 LPF/guard.

rclpy is Isaac Sim's bundled ROS2 Jazzy (isaacsim.ros2.bridge/jazzy); see
launch_v14_ros2_eval.sbatch for the environment.
"""

from __future__ import annotations

import array
import time
from collections.abc import Sequence

import torch

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from v13_torque_mdp import clip_position_target_to_motor_effort, low_pass_alpha, low_pass_step
from v14_ros2_control import contract


ACTION_SCALE = 0.2  # V13LowPassJointPositionActionCfg.scale with default offsets


def implicit_joint_drives(robot_cfg) -> None:
    """Replace IsaacLab's explicit DCMotor model with PhysX joint-drive PD (same gains/limits)."""
    for name, motor in robot_cfg.actuators.items():
        if not isinstance(motor, DCMotorCfg) or (
            motor.stiffness, motor.damping, motor.velocity_limit
        ) != (contract.PD_STIFFNESS, contract.PD_DAMPING, contract.MOTOR_VELOCITY_LIMIT):
            raise ValueError(f"actuator {name} is not the reviewed robot_v2 DCMotor")
        robot_cfg.actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=motor.joint_names_expr,
            stiffness=motor.stiffness,
            damping=motor.damping,
            effort_limit_sim=motor.effort_limit,
            friction=motor.friction,
            armature=motor.armature,
        )


class V14Ros2JointTargetAction(ActionTerm):
    """Lockstep ROS2 bridge; the ``actions`` passed to ``env.step`` only feed the parity shadow."""

    cfg: "V14Ros2JointTargetActionCfg"

    def __init__(self, cfg: "V14Ros2JointTargetActionCfg", env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("the ROS2 bridge drives exactly one robot")
        if round(env.physics_dt * 1e9) != contract.PHYSICS_DT_NS or env.cfg.decimation != contract.DECIMATION:
            raise ValueError("simulation timing differs from the ROS2 contract")
        self._joint_ids, names = self._asset.find_joints(list(contract.JOINT_NAMES), preserve_order=True)
        if tuple(self._asset.joint_names) != contract.JOINT_NAMES:
            raise ValueError(f"articulation joint order {self._asset.joint_names} differs from the policy contract")
        self._raw_actions = torch.zeros(1, len(names), device=self.device)
        self._default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default.clone()
        self._physics_step = 0
        self._commands, self._targets = {}, {}
        self._alpha = low_pass_alpha(cutoff_hz=cfg.shadow_cutoff_hz, control_dt_s=contract.PHYSICS_DT_S)
        self._limits = torch.tensor([contract.joint_effort_limit(n) for n in names], device=self.device)
        self._shadow_filtered = None
        self._target_diff, self._command_diff = [], []
        self._parity_open = True
        self._start_ros(cfg)

    def _start_ros(self, cfg) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, JointState

        if not rclpy.ok():
            rclpy.init()
        self._msg = dict(Clock=Clock, Image=Image, JointState=JointState, Odometry=Odometry)
        self._node = rclpy.create_node("v14_isaac_sim", namespace=cfg.namespace)
        qos = contract.lockstep_qos()
        self._state_pub = self._node.create_publisher(JointState, contract.JOINT_STATES, qos)
        self._odom_pub = self._node.create_publisher(Odometry, contract.ODOM, qos)
        self._image_pub = self._node.create_publisher(Image, contract.DEPTH_IMAGE, qos)
        self._clock_pub = self._node.create_publisher(Clock, "/clock", 10)
        command_sub = self._node.create_subscription(JointState, contract.JOINT_COMMAND, self._on_command, qos)
        self._node.create_subscription(JointState, contract.JOINT_TARGET, self._on_target, qos)
        # Both control nodes must be connected before the first physics step.
        deadline = time.monotonic() + cfg.startup_timeout_s
        while not (self._state_pub.get_subscription_count() >= 2 and self._odom_pub.get_subscription_count() >= 1
                   and self._image_pub.get_subscription_count() >= 1 and command_sub.get_publisher_count() >= 1):
            if time.monotonic() > deadline:
                raise TimeoutError("ROS2 policy/joint command nodes did not connect to the simulator")
            rclpy.spin_once(self._node, timeout_sec=0.1)
        self._node.get_logger().info(f"ROS2 control nodes connected; joint order {list(contract.JOINT_NAMES)}")

    def _on_command(self, message) -> None:
        self._commands[contract.stamp_ns(message.header.stamp)] = contract.ordered(message.name, message.position)

    def _on_target(self, message) -> None:
        self._targets[contract.stamp_ns(message.header.stamp)] = contract.ordered(message.name, message.position)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        import rclpy

        sec, nanosec = contract.step_stamp(self._physics_step)
        stamp = sec * 1_000_000_000 + nanosec
        tick = contract.is_policy_tick(stamp)
        if tick:
            self._publish_policy_inputs(sec, nanosec)
        self._publish_joint_state(sec, nanosec)
        deadline = time.monotonic() + self.cfg.step_timeout_s
        while stamp not in self._commands or (tick and stamp not in self._targets):
            if time.monotonic() > deadline:
                raise TimeoutError(f"no ROS2 joint command for simulation time {stamp * 1e-9:.3f}s")
            rclpy.spin_once(self._node, timeout_sec=0.01)
        command = torch.tensor(self._commands.pop(stamp), device=self.device)
        if command.shape != (self.action_dim,) or not bool(torch.isfinite(command).all()):
            raise ValueError("invalid ROS2 joint command")
        self._update_parity(command, torch.tensor(self._targets.pop(stamp), device=self.device) if tick else None)
        self._processed_actions[0] = command
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._physics_step += 1

    def _update_parity(self, command: torch.Tensor, target: torch.Tensor | None) -> None:
        """Run the in-process V13 LPF/guard on the shadow policy target, as IsaacLab would."""
        data = self._asset.data
        position, velocity = data.joint_pos[:, self._joint_ids], data.joint_vel[:, self._joint_ids]
        shadow_target = self._default + ACTION_SCALE * self._raw_actions
        if self._shadow_filtered is None:
            self._shadow_filtered = position.clone()
        self._shadow_filtered = low_pass_step(shadow_target, self._shadow_filtered, self._alpha)
        shadow_command = clip_position_target_to_motor_effort(
            self._shadow_filtered, position, velocity, self._limits,
            stiffness=contract.PD_STIFFNESS, damping=contract.PD_DAMPING,
            velocity_limit=contract.MOTOR_VELOCITY_LIMIT, fraction=self.cfg.shadow_guard_fraction,
        )
        if self._parity_open:
            self._command_diff.append(float((command - shadow_command[0]).abs().max()))
            if target is not None:
                self._target_diff.append(float((target - shadow_target[0]).abs().max()))

    def parity_summary(self) -> dict[str, object]:
        """ROS2-vs-in-process differences (rad) up to the first reset after the rollout starts."""
        def stats(values):
            return {"samples": len(values), "max_abs_rad": max(values, default=None),
                    "mean_abs_rad": sum(values) / len(values) if values else None}
        return {"policy_target": stats(self._target_diff), "joint_command": stats(self._command_diff),
                "shadow_cutoff_hz": self.cfg.shadow_cutoff_hz,
                "shadow_guard_fraction": self.cfg.shadow_guard_fraction}

    def _header(self, message, sec, nanosec, frame_id=""):
        message.header.stamp.sec, message.header.stamp.nanosec = sec, nanosec
        message.header.frame_id = frame_id
        return message

    def _publish_joint_state(self, sec, nanosec) -> None:
        data = self._asset.data
        message = self._header(self._msg["JointState"](), sec, nanosec)
        message.name = list(contract.JOINT_NAMES)
        message.position = data.joint_pos[0, self._joint_ids].tolist()
        message.velocity = data.joint_vel[0, self._joint_ids].tolist()
        message.effort = data.applied_torque[0, self._joint_ids].tolist()
        self._state_pub.publish(message)
        clock = self._msg["Clock"]()
        clock.clock.sec, clock.clock.nanosec = sec, nanosec
        self._clock_pub.publish(clock)

    def _publish_policy_inputs(self, sec, nanosec) -> None:
        data = self._asset.data
        odometry = self._header(self._msg["Odometry"](), sec, nanosec, "world")
        odometry.child_frame_id = "base"
        position, quat = data.root_pos_w[0].tolist(), data.root_quat_w[0].tolist()
        pose = odometry.pose.pose
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quat
        twist = odometry.twist.twist
        twist.linear.x, twist.linear.y, twist.linear.z = data.root_lin_vel_b[0].tolist()
        twist.angular.x, twist.angular.y, twist.angular.z = data.root_ang_vel_b[0].tolist()
        self._odom_pub.publish(odometry)

        depth = self._env.scene["depth_camera"].data.output["distance_to_image_plane"][0, :, :, 0]
        image = self._header(self._msg["Image"](), sec, nanosec, "front_depth")
        image.height, image.width = depth.shape
        image.encoding, image.is_bigendian, image.step = "32FC1", 0, depth.shape[1] * 4
        image.data = array.array("B", depth.float().contiguous().cpu().numpy().astype("<f4").tobytes())
        self._image_pub.publish(image)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # The ROS2 filter and last-action state are not reset; evaluations stop counting at termination.
        if self._physics_step:
            self._parity_open = False
            self._node.get_logger().warning(f"robot reset at physics step {self._physics_step}")


@configclass
class V14Ros2JointTargetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = V14Ros2JointTargetAction
    namespace: str = "/v14"
    startup_timeout_s: float = 120.0
    step_timeout_s: float = 30.0
    shadow_cutoff_hz: float = 25.0
    shadow_guard_fraction: float = 0.99
