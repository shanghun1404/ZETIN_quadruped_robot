# V15.1 ROS2 제어

제어 노드는 Isaac API를 사용하지 않는다. 관절/자세/깊이/발 접촉만 받아서 관절 위치
목표를 발행한다. Isaac 어댑터와 실기 드라이버가 같은 계약을 사용할 수 있도록 구성했다.
현재 실행 검증 대상은 **시뮬레이션 시간 lockstep**이다.

## 구성

- `controller_node`: V15 정책 50 Hz, 모델 기반 기립 WBC 50 Hz, 관절 명령 200 Hz.
- `motion.py`: 낮은 자세 → 상승 → 안정 확인 → 보행 전환 → 보행 → 감속 → 기립 → 낮추기.
- `standing.py`: V16 WBC와 MJCF/Pinocchio 동역학. 질량행렬/중력/자코비안을 제어 노드가 계산.
- `config/v151_ros2_bridge.py`: Isaac 상태 발행과 PhysX 위치 목표 적용.
- `config/evaluate_v151_ros2.py`: 리셋 없이 첫 종료까지 평가, 200 Hz 토크/자가 충돌 기록.

## 토픽 (`/v151`)

| 토픽 | ROS 타입 | 의미 |
|---|---|---|
| `joint_states` | `sensor_msgs/JointState` | 이름이 있는 q(rad), dq(rad/s), 추정 토크(Nm), 200 Hz |
| `odom` | `nav_msgs/Odometry` | actor 원점 world pose, actor 원점 body-frame twist, 50 Hz |
| `depth/image` | `sensor_msgs/Image` | 84×84 little-endian `32FC1`, 미터, 50 Hz |
| `foot_contacts` | `sensor_msgs/JointState` | name=FL/FR/RL/RR_foot, effort=접촉력 크기(N), 50 Hz |
| `cmd_vel` | `geometry_msgs/Twist` | body-frame vx,vy,wz; **전방 +y** |
| `mode_request` | `std_msgs/String` | `sit`, `stand`, `walk` |
| `joint_command` | `sensor_msgs/JointState` | position=목표각, effort=제한 후 토크 진단값, 200 Hz |
| `status` | `std_msgs/String` | JSON: stamp_ns, 상태, 혼합비, 지지 발 수, WBC 진단, 계산 시간 |

`joint_command.effort`는 **진단값**이다. position 명령에 추가할 feedforward가 아니다.
고정 Kp=300 Nm/rad, Kd=8 Nm·s/rad에서 `q_target=q+(tau+Kd*dq)/Kp`로 변환한다.
기립·보행 토크를 혼합한 뒤 HAA24.8/HFE48/KFE48 Nm 및 속도 곡선 가드를 적용한다.
PhysX implicit 드라이브의 실제 적분 결과가 이 순간 토크 식과 동일하다고 가정하지 않는다.

정책 관측은 기존 V15와 동일하게 root COM 선속도를 사용한다. 어댑터가 보낸 actor 원점
속도에 `omega × root_com_offset`을 더해 변환한다. WBC에는 actor 원점 속도를 쓴다.
모든 lockstep 입력은 같은 정수 시각(ns)으로 매칭한다. 5 ms 간격, 정책은 매 4번째 틱이다.

## 실행 환경

ROS2 Jazzy, Python, NumPy, Torch, Pinocchio가 필요하다. 기존 `v14_ros2_control` 패키지와
검토된 저장소 `config/` 모듈(v12/v13 정책, v16 모델/WBC)을 함께 사용한다.
`config/`를 `PYTHONPATH`에 넣어야 한다. HPC 환경은 아래 실행 스크립트가 설정한다.

```bash
ssh -o ClearAllForwardings=yes ws1
sbatch /home/shanghun1404/v151_ros2/config/experiments/v151_20261001/ros2_eval.sbatch
```

추가 매개변수는 actor 경로, `demo|walk`, 지형, seed, 초 순서다. `demo`는 시험 시나리오용
명령을 노드 내부에서 발생시킨다. 실제 운영에서는 `demo=false`로 하고 외부 토픽을 사용한다.
패키지 launch는 `actor_checkpoint`, `student_checkpoint`, `model_path` 인자를 받는다.

## 실기 이식 시 남는 경계

하드웨어 드라이버의 시각/관절 이름/단위, IMU·오도메트리 추정, 발 접촉 추정, 깊이 카메라,
실제 모터 게인·지연·전류 응답이 필요하다. 현재 노드는 lockstep 센서 시각을 기다리므로
실시간 watchdog/통신 지연 처리와 모터 구동기 식별을 거쳐야 실기에 연결할 수 있다.
낮은 자세는 네 발로 지지하는 crouch이고, 뒤집힘/낙상 복구 기능은 포함하지 않는다.

## 충돌 해석

다른 다리의 thigh/calf/foot 쌍 접촉력을 기록한다. 같은 다리의 기구적으로 인접한 쌍은
기존 USD에서 제외된다. 추가로 8개 thigh/calf capsule 간 기하 간격을 측정한다.
몸통은 크기 0.4×0.52×0.15 m 상자와의 기하 간격을 별도로 측정한다. 이 상자는 근사이며
V14 보행 USD에는 몸통 접촉 형상이 없다. 따라서 상자와의 겹침은 실제 CAD 간섭/접촉과
동일한 판정이 아니다. 필터/형상 때문에 물리 접촉 0만으로 전체 자가 충돌 없음이라 하지 않는다.
