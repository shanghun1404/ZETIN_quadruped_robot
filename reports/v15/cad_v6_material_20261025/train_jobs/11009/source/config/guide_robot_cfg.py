import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# Set to match the HPC path once synced
_USD_PATH = "/home/kang/new_URDF/guide_quadruped.usd"

CALIBRATED_HALF_ANGLE_RAD = 0.85
CALIBRATED_HFE_RAD = -0.85
CALIBRATED_KFE_RAD = 1.70
CALIBRATED_SPAWN_HEIGHT_M = 0.36159140440134097

##
# Configuration
##

GUIDE_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, CALIBRATED_SPAWN_HEIGHT_M),
        joint_pos={
            ".*_HAA": 0.0,
            ".*_HFE": CALIBRATED_HFE_RAD,
            ".*_KFE": CALIBRATED_KFE_RAD,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
            effort_limit=53.0,
            saturation_effort=53.0,
            velocity_limit=50.0,
            stiffness=300.0,
            damping=8.0,
            friction=0.3,
        ),
    },
)
