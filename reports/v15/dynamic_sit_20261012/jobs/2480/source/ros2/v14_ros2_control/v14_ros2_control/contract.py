"""Topic, timing and joint contract shared by the Isaac Sim bridge and V14 ROS2 nodes.

Pure Python: the simulator, the nodes and the tests import it without ROS or Isaac.
"""

from __future__ import annotations

# GuideV13TorqueEnvCfg: 200 Hz physics, decimation 4 -> 50 Hz policy.
PHYSICS_DT_NS = 5_000_000
DECIMATION = 4
PHYSICS_DT_S = PHYSICS_DT_NS * 1e-9

# IsaacLab (PhysX) articulation order of robot_v2, which is the policy observation/action
# order; reported by the V14 USD in job 258. The bridge refuses to start on any other order.
JOINT_NAMES = (
    "FL_HAA", "FR_HAA", "FL_HFE", "FR_HFE",
    "RR_HAA", "RL_HAA", "RR_HFE", "RL_HFE",
    "FL_KFE", "FR_KFE", "RR_KFE", "RL_KFE",
)
# guide_robot_v2_cfg: stand keyframe and MJCF actuator limits.
DEFAULT_JOINT_POS = {"HAA": 0.0, "HFE": -0.55, "KFE": 1.10}
EFFORT_LIMIT_NM = {"HAA": 24.8, "HFE": 48.0, "KFE": 48.0}
PD_STIFFNESS = 300.0
PD_DAMPING = 8.0
MOTOR_VELOCITY_LIMIT = 50.0

# Relative names; the namespace (default /v14) comes from the ROS arguments.
JOINT_STATES = "joint_states"    # sim -> nodes, 200 Hz, position/velocity/effort
ODOM = "odom"                    # sim -> policy, 50 Hz, world pose + body-frame twist
DEPTH_IMAGE = "depth/image"      # sim -> policy, 50 Hz, 84x84 32FC1 metric depth
CMD_VEL = "cmd_vel"              # operator -> policy, base-frame command (+y is forward)
JOINT_TARGET = "joint_target"    # policy -> joint command node, 50 Hz raw position target
JOINT_COMMAND = "joint_command"  # joint command node -> sim, 200 Hz filtered position target


def joint_default(name: str) -> float:
    return DEFAULT_JOINT_POS[name.rsplit("_", 1)[1]]


def joint_effort_limit(name: str) -> float:
    return EFFORT_LIMIT_NM[name.rsplit("_", 1)[1]]


def stamp_ns(stamp) -> int:
    """builtin_interfaces/Time -> integer nanoseconds, the exact lockstep key."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def step_stamp(step: int) -> tuple[int, int]:
    """Physics step index -> (sec, nanosec) of simulation time."""
    if step < 0:
        raise ValueError("physics step must be non-negative")
    return divmod(step * PHYSICS_DT_NS, 1_000_000_000)


def is_policy_tick(nanoseconds: int) -> bool:
    """True on the first physics step of each 50 Hz policy period."""
    step, remainder = divmod(nanoseconds, PHYSICS_DT_NS)
    if remainder:
        raise ValueError(f"stamp {nanoseconds} ns is not on the 200 Hz physics grid")
    return step % DECIMATION == 0


def ordered(names, values, expected=JOINT_NAMES) -> list[float]:
    """Reorder a JointState field into ``expected`` order by name, rejecting gaps."""
    names = list(names)
    if len(values) != len(names):
        raise ValueError("joint names and values differ in length")
    lookup = dict(zip(names, values))
    if len(lookup) != len(names) or set(lookup) != set(expected):
        raise ValueError(f"joint names {names} do not match {list(expected)}")
    return [float(lookup[name]) for name in expected]


def lockstep_qos():
    """Reliable delivery; transient-local so a late-matched reader still gets the sample."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=64,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
