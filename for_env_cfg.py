
from isaaclab.utils import configclass
import isaaclab.managers as mg
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from for_robot_cfg import FOR_ROBOT_CFG
import torch

def lin_vel_x_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral (X) velocity because this robot's left/right axis is X, and forward is -Y."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])



@configclass
class ForRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # [필수] 신규 RSL-RL 버전을 위한 'commands' 관측 그룹 추가
        self.observations.commands = mg.ObservationGroupCfg(
            concatenate_terms=True,
            enable_corruption=False,
        )
        self.observations.commands.velocity_commands = mg.ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # [핵심] URDF 분석 결과, 로봇의 앞뒤 축은 Y (앞: -Y), 좌우 축은 X (왼쪽: +X) 입니다.
        # 따라서 전진을 명령하려면 lin_vel_x가 아니라 lin_vel_y에 마이너스 값을 주어야 합니다!
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)    # 좌우 이동 금지
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 0.0)   # 직진(-Y) 속도만 명령
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)   # 제자리 회전 유지


        # [필수] 신규 RSL-RL 버전을 위한 'critic' 관측 그룹 추가
        self.observations.critic = mg.ObservationGroupCfg(
            concatenate_terms=True,
            enable_corruption=False,
        )
        # policy 그룹의 모든 항목을 critic 그룹으로 복사하되, 없는 센서 항목은 제거
        import copy
        self.observations.critic = copy.deepcopy(self.observations.policy)
        
        # 평지 환경(Flat)일 경우 height_scan 항목이 있으면 제거 (에러 방지)
        if hasattr(self.observations.critic, "height_scan"):
            self.observations.critic.height_scan = None

        self.scene.robot = FOR_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/BODY"
        
        # increase action scale (0.05 was too small, preventing the robot from moving its legs to catch itself)
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "BODY"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "BODY"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.4, 0.4), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards - 자연스럽고 안정적인 보행 (Balanced Natural Gait)
        
        # 1. 몸체 높이 유지 (0.3m를 기준으로 너무 높거나 낮으면 감점)
        self.rewards.base_height_l2 = mg.RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={"target_height": 0.3, "asset_cfg": SceneEntityCfg(name="robot", body_names="BODY")},
        )
        
        # 2. 발을 높이 들게 유도 (질질 끌기 방지) -> 휴리스틱 보상 0으로 비활성화
        self.rewards.feet_air_time = mg.RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*LEG.*"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        
        # 3. 에너지 효율 및 부드러운 움직임 유도 (페널티 균형 조정)
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01  # 조사자료 권장에 따라 최소화
        
        # 추가적인 상속받은 마이너스 페널티 항목들을 모두 0으로 초기화 (CaT)
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        
        # 4. 전진 보상 (안정성과 속도 균형)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        # 5. 관절 가동 범위 끝단 사용 패널티 (다리가 몸통을 통과하는 현상 방지)
        self.rewards.dof_pos_limits.weight = -2.0
        
        # 6. 불필요한 바디 접촉 방지 (발을 제외한 모터, 다리 상단이 닿으면 감점 -> 썰매타듯 기어가는 부자연스러운 동작 방지)
        # CaT(Curriculum and Termination) 철학에 따라, 접촉 시 즉시 에피소드를 종료(Termination)하므로 별도의 마이너스 페널티는 불필요합니다. 0으로 비활성화합니다.
        self.rewards.undesired_contacts = mg.RewardTermCfg(
            func=mdp.undesired_contacts,
            weight=0.0,
            params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*MOTOR.*|.*LEG.*"), "threshold": 1.0},
        )
        
        # 배가 지면에 닿을 때도 강력한 벌점을 부여합니다. (CaT 철학에 따라 0으로 비활성화)
        self.rewards.base_contact_penalty = mg.RewardTermCfg(
            func=mdp.illegal_contact,
            weight=0.0,
            params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names="BODY"), "threshold": 1.0},
        )
        
        # 7. 횡방향(게걸음) 이동 방지 패널티
        # 로봇의 좌우 축이 X축이므로, X축 속도를 패널티로 줍니다.
        self.rewards.lin_vel_x_l2 = mg.RewardTermCfg(
            func=lin_vel_x_l2,
            weight=-2.0,
        )

        # CaT 이론 적용: 조기 종료(Termination)를 활성화하여 제자리 걸음 꼼수를 원천 차단
        # 단, 발 역할을 하는 LEG는 땅에 닿아야 하므로 예외 처리하고 BODY와 MOTOR가 닿으면 즉사
        self.terminations.base_contact = mg.TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*BODY.*|.*HIP.*|.*MOTOR.*"), "threshold": 1.0},
        )

@configclass
class ForFlatEnvCfg(ForRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # 조사자료 반영: 롤링(Roll) 억제 가중치가 너무 크면 굳어버리므로 적절히 완화 (-2.5 적용)
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.0
        
        # 전진 추종 보상 대폭 강화 (1.5 -> 3.0)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
