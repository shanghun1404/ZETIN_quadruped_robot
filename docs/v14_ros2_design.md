# V14 ROS2 제어 구조 — Isaac Sim은 시뮬레이터만

2026-09-27 사용자 요청: “아이작 심 내장 함수로 모터 제어하는 구조를 ROS2로 해서
Isaac Sim으로 시뮬레이션하는 구조로 바꿔서 V14로”. 사용자가 선택한 경계:
**ROS2는 위치 목표까지, PD는 Isaac Sim(PhysX) 관절 드라이브.** 구동 정책은
V14 작업110 `model_5700.pt` + 고정 V12 depth student. 보정 policy(작업186)는 포함하지 않는다.

## 이전 구조와 바뀐 점

| 단계 | 기존 (IsaacLab 내부) | V14 ROS2 |
|---|---|---|
| 정책 50 Hz | `OnPolicyRunner` + `V13DepthPolicyWrapper` | `policy_node` (ROS2) |
| 25 Hz LPF + 0.99 토크 가드 200 Hz | `V13LowPassJointPositionAction` | `joint_command_node` (ROS2) |
| PD → 토크 | IsaacLab `DCMotorCfg` (explicit, Python) | PhysX 관절 드라이브 (implicit) |
| 물리·센서 | IsaacLab env | 같은 env, `V14Ros2JointTargetAction` 브리지 |

ROS2 노드는 Isaac/IsaacLab을 import하지 않는다(rclpy + torch + `config/`의
V12/V13 모듈). 같은 노드를 실기 드라이버가 같은 토픽을 내면 그대로 쓸 수 있는 형태로 두었다.
실기 적용 자체는 검증하지 않았다.

## 토픽 (기본 namespace `/v14`)

| 토픽 | 타입 | 방향 | 주기 |
|---|---|---|---|
| `joint_states` | `sensor_msgs/JointState` (위치·속도·토크 추정) | sim → 두 노드 | 200 Hz |
| `odom` | `nav_msgs/Odometry` (world 자세, **base 프레임** 속도) | sim → policy | 50 Hz |
| `depth/image` | `sensor_msgs/Image` 84×84 `32FC1` 미터 | sim → policy | 50 Hz |
| `joint_target` | `sensor_msgs/JointState` 위치 = default + 0.2·action | policy → command | 50 Hz |
| `joint_command` | `sensor_msgs/JointState` 필터·가드 후 위치 목표 | command → sim | 200 Hz |
| `cmd_vel` | `geometry_msgs/Twist` (선택) | 운영자 → policy | 비주기 |
| `/clock` | `rosgraph_msgs/Clock` | sim | 200 Hz |

- **이 로봇의 전방은 base +y다.** `cmd_vel`은 학습 명령 그대로 (x, y, yaw rate)로
  해석한다. 전진 0.4 m/s는 `linear.y=0.4`.
- 관절 순서는 IsaacLab articulation 순서(`contract.JOINT_NAMES`)이며 메시지는 이름으로
  재정렬한다. 브리지는 시뮬레이터 순서가 계약과 다르면 시작하지 않는다.
- 모든 lockstep 토픽: reliable, transient-local, keep-last 64.

## Lockstep 순서

stamp = 물리 스텝 × 5 ms(정수 ns)로 메시지를 짝짓는다. 물리 스텝 k마다:

1. k%4==0이면 브리지가 `odom`·`depth/image`를 발행한다. 기존 관측과 같은 시점의 값이다.
2. 브리지가 `joint_states`(k)를 발행한다.
3. 정책 노드는 같은 stamp의 세 메시지가 모이면 235차원 관측을 만들고 `joint_target`(k)을 낸다.
   관측 순서는 V8/V11과 같다: 선속도, 각속도, 투영 중력, 명령, 관절 위치 오프셋,
   관절 속도, 직전 action(48), V12 예측 scan(187). 노이즈는 없다(평가와 동일).
4. command 노드는 틱이면 같은 stamp의 target을 기다린 뒤 LPF 한 스텝과 가드를 적용해
   `joint_command`(k)를 낸다. 가드는 `v13_torque_mdp` 함수를 그대로 쓴다.
5. 브리지는 `joint_command`(k)를 받을 때까지 대기(30 s 초과 시 오류)한 뒤 PhysX 드라이브
   목표로 쓰고 물리를 한 스텝 진행한다.

IsaacLab의 `env.step` 순서(관측 → action → 4회 [LPF·가드 → 물리])와 같다. 로봇이
넘어져 reset되면 ROS2 쪽 필터·직전 action은 초기화하지 않는다. 평가는 첫 종료까지만 센다.
실시간(비-lockstep) 모드는 구현하지 않았다.

## 액추에이터 차이

`implicit_joint_drives`가 `DCMotorCfg`를 같은 값의 `ImplicitActuatorCfg`로 바꾼다:
stiffness 300, damping 8, 토크 한도 HAA 24.8 / HFE·KFE 48 N·m, 마찰 0.3, 속도 한도는 USD 값.
PhysX 드라이브는 DCMotor의 **속도-토크 포화 곡선을 모델링하지 않는다.**
0.99 가드가 명시 PD 요청 토크를 그 곡선 안으로 제한하지만, implicit 적분 토크는 명시 PD와
다르다. 기록되는 토크는 IsaacLab의 implicit 추정값이다. 따라서 ROS2 결과와 기존 결과의
차이에는 통신 구조가 아니라 이 액추에이터 모델 차이가 포함된다.

## 검증 설계

- **Parity (그림자 정책):** `--control ros2`에서도 평가기는 기존 in-process 정책을 같은
  관측으로 계속 계산한다. 이 action은 적용하지 않는다. 브리지는 매 틱 ROS2 target과
  in-process target의 차이, 매 물리 스텝 ROS2 command와 in-process V13 LPF·가드 결과의
  차이를 기록한다(`metadata.ros2_parity`, rad). 첫 reset 이후는 제외한다.
- **A/B:** 같은 1로봇 fixed_v2 조건에서 `--control isaaclab`(기존 explicit DCMotor)과 비교.
- **단위·통합 테스트:** `ros2/v14_ros2_control/test` — 계약, LPF·가드의 독립 수식 대조,
  관측 배치, 실제 DDS 위 가짜 시뮬레이터 lockstep(Isaac 없음).

## 실행

HPC `ws1`에는 시스템 ROS2가 없으므로 Isaac Sim 번들 Jazzy
(`isaacsim.ros2.bridge/jazzy`)의 rclpy·라이브러리를 `PYTHONPATH`·`LD_LIBRARY_PATH`로 쓴다.
세 프로세스는 같은 Slurm 작업 안에서 localhost DDS로 통신한다.

```bash
cd /home/shanghun1404/v14_leg_collision/config
sbatch launch_v14_ros2_eval.sbatch ros2 <label> flat 42       # ROS2 노드 + Isaac Sim
sbatch launch_v14_ros2_eval.sbatch isaaclab <label> flat 42   # 같은 조건의 기존 제어
cd ../ros2 && srun --partition=compute --qos=admin --cpus-per-task=4 --mem=8G ./run_tests.sh
```

시스템 ROS2 Jazzy가 있는 PC에서는 `colcon build --packages-select v14_ros2_control` 후
`ros2 launch v14_ros2_control v14_ros2_control.launch.py config_dir:=<repo>/config
actor_checkpoint:=... student_checkpoint:=...`로 노드를 띄운다(노드는 torch가 필요하다).
이 경로는 이번에 실행 검증하지 않았다.

결과 기록: [reports/v14/ros2/README.md](../reports/v14/ros2/README.md).
