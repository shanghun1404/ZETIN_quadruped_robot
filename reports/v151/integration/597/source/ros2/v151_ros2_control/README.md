# V15.1 ROS2 제어

제어 노드는 Isaac API를 사용하지 않는다. 관절/자세/깊이/발 접촉만 받아서 관절 위치
목표·PD 게인·feedforward 토크를 발행한다. Isaac 어댑터와 실기 드라이버가 같은 계약을 사용할 수 있도록 구성했다.
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
| `joint_command` | `sensor_msgs/JointState` | position=목표각, effort=feedforward 토크(Nm), 200 Hz |
| `status` | `std_msgs/String` | JSON: stamp_ns, 상태, 혼합비, 지지 발 수, WBC 진단, 계산 시간 |

명령 스키마 2는 `joint_command`와 같은 stamp의 `status`를 원자적으로 맞춰 적용한다.
`status.stiffness_nm_rad`, `status.damping_nms_rad`가 그 스텝의 PD 게인이다.
기립에서는 Kp=Kd=0이고 `joint_command.effort`로 V16 WBC 토크를 직접 적용한다.
보행에서는 Kp=300, Kd=8, effort=0으로 기존 V15의 implicit PD 경로를 유지한다.
전환 중 보행 비율 w에 따라 Kp=300w, Kd=8w, feedforward=(1−w)·기립 토크를 사용하며,
총 순간 요청 토크가 24.8/48/48 Nm 및 속도 곡선 가드를 넘으면 feedforward와 목표각을 보정한다.
추가로 PhysX drive maxForce를 관절 한도−|feedforward|로 설정하여 두 구동 성분의 합을 제한한다.
이 제한은 모터 구동력에 해당하며 외력에 의한 관절 구조 반력의 한도는 아니다.
보행 비율 1에서는 이전의 위치 목표와 정확히 같은 값을 낸다.

실기 드라이버는 두 메시지의 stamp를 일치시킨 후 위치·게인·feedforward를 함께 적용해야 한다.
게인/토크 단위가 같은지 확인해야 하며, 메시지 하나만 독립 적용하면 안 된다.
시뮬레이터는 실제 PhysX 게인과 IsaacLab 추정기 게인을 함께 바꾸고, 구동 모드가 바뀔 때
PhysX getter로 실제 게인을 확인한다. 보행 중 토크는 implicit PD 순간 추정값이다.

정책 관측은 기존 V15와 동일하게 root COM 선속도를 사용한다. 어댑터가 보낸 actor 원점
속도에 `omega × root_com_offset`을 더해 변환한다. WBC에는 actor 원점 속도를 쓴다.
모든 lockstep 입력은 같은 정수 시각(ns)으로 매칭한다. 5 ms 간격, 정책은 매 4번째 틱이다.

정지 요청 후 0.3초 이상 지나고 두 발 이상이 접촉하며 몸통이 수평에 가까우면 기립 토크로
혼합 전환한다. 전환과 안정화 중에는 실제 접촉 발만 WBC 지지 제약에 사용한다.
보행 정책의 0속도 명령만으로 정지를 보장할 수 없으므로 WBC가 감속과 착지를 맡는다.
STAND/SIT 확정에는 여전히 네 발 접촉, 낮은 선속도·각속도, 목표 높이 확인이 필요하다.

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
