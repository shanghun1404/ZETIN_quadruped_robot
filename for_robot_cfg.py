
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

# Get the directory of the current script
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_USD_PATH = os.path.join(_CURRENT_DIR, "for", "for.usd")

##
# Configuration
##

FOR_ROBOT_CFG = ArticulationCfg(
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
            solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "BODY_Revolute[-_].*": 0.0,
            ".*_MOTOR.*_Revolute[-_].*": 0.4,
            ".*_LEG.*_Revolute[-_].*": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[
                "BODY_Revolute[-_].*",
                ".*_MOTOR.*_Revolute[-_].*",
                ".*_LEG.*_Revolute[-_].*"
            ],
            effort_limit=100.0,
            saturation_effort=100.0,
            velocity_limit=21.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
