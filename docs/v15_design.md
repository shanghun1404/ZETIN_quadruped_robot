# V15 대규모 추가 훈련 — 준비 설계

2026-09-27 사용자 요청: V14 ROS2 결과의 1–3순위를 반영해 “대규모 추가 훈련하는 V15 준비”.
사용자가 고른 구조: **보행(V15-L)과 넘어짐 복구(V15-R)를 별도 정책으로 두고 ROS2에서 모드 전환.**
이 문서 작성 시점까지 한 것은 코드·자산·스모크 검증이다. 본 훈련은 아직 제출하지 않았다.

## 출발점 (측정값)

| 조건 (시드 42, 32환경) | 평지 생존 | 평지 진행 | 계단 오르막 15.7 cm (fixed_v2) |
|---|---:|---:|---|
| V14 5700, 기존 explicit DCMotor | 32/32 | 8.33 m (legacy 평지 평가) | heading 보정 시 26/32 생존, 통과 0/32 |
| V14 5700, PhysX implicit PD (작업278/279, fixed_v2) | 10/32 | 0.79 m | heading 제어 11/32 생존, 통과 0/32 |

PhysX PD는 V14 ROS2 구조가 쓰는 관절 드라이브다([V14 ROS2 결과](../reports/v14/ros2/README.md)).
위 표의 DCMotor 계단 값은 [V14 방향 실험](../reports/v14/direction/README.md) 기록이다.

## V15-L 보행 (1·2순위)

- 시작: V14 작업110 `model_5700.pt`(235→512→256→128→12 ELU), V12 depth student 고정(루프 포함).
- Plant: `implicit_joint_drives` — PhysX PD 300/8, 토크 한도 24.8/48 N·m. 환경마다 시작 시
  stiffness·damping을 U(0.8, 1.2)배로 고정(IsaacLab `randomize_actuator_gains`, implicit은 startup 권장).
- Action: `V15DelayedLowPassJointPositionAction` — 정책 목표를 0–3 물리 스텝(0–15 ms) 지연한 뒤
  V13 25 Hz LPF + 0.99 토크 가드를 적용한다. 가드는 공칭 gain을 쓴다(ROS2 command 노드와 같다).
  지연은 episode마다 새로 뽑는다([`v15_math.TargetDelayLine`](../config/v15_math.py)).
- 지형·명령: V14 계단 집중 지형 조합(역피라미드 계단 45%, 계단 25%, 상자·거친 지형 10%씩,
  경사 5%씩)과 terrain curriculum(L0 시작). lane heading 명령을 쓰고, yaw 보상은 원래 값을 쓴다.
  V14에서 강한 yaw 보상(std 0.25, 가중치 7.5)이 계단 균형을 악화시켰기 때문이다.
- 규모: 기본 512환경(스모크: 반복당 1.6–2.4 s, 12,288 전이). 지형 열 = 환경 수/4.
  환경끼리 깊이 영상에 서로 비치는 빈도를 V14 파일럿 수준(타일당 약 0.3–0.4대)으로 둔다.
- PPO: V14 설정(적응 lr 1e-3, desired_kl 0.01, 24 스텝/환경, 5 epoch) 유지, 100회마다 저장.
- 관측 노이즈는 V14처럼 끈다(V12 student 입력 조건 유지).

## V15-R 복구 (3순위)

- **자산 전제:** 원본은 발만 지면과 충돌한다. [`v15_usd_builder.py`](../config/v15_usd_builder.py)가
  V14 USD를 참조하는 overlay에 몸통 상자 충돌체를 추가한다. 상자 크기는 몸통 패널 형상 측정값
  (작업280)으로 정했다: 몸통 프레임 x ±0.20, y ±0.26, z 0–0.15 m. 다리 캡슐·발과는 충돌을 끈다.
  몸통 질량은 명시값 5.345 kg이라 충돌체로 바뀌지 않는다.
- 드롭 검증(작업300, 3 s, 정책 0): 서 있기 루트 0.438 m·몸통 접촉 0으로 V14와 같다. 뒤집힘 0.150 m,
  옆 0.238 m에서 몸통이 체중의 0.46–1.00배를 받친다. V14 USD는 뒤집힘 시 0.035 m로
  몸통이 약 11.5 cm 지면 아래에 묻힌다.
- 환경([`v15_recovery_env_cfg.py`](../config/v15_recovery_env_cfg.py)): 평지. V15-L과 같은
  액추에이터 경로(PhysX PD + gain·지연 랜덤화 + LPF·가드), 마찰·몸통 질량 랜덤화.
  episode 6 s, 조기 종료 없음. 리셋의 80%는 균일 무작위 자세로 0.8 m에서 낙하시키고,
  20%는 선 자세에서 시작한다.
- Actor 관측 42: 각속도, 투영 중력, 관절 위치 오프셋, 관절 속도, 직전 action(노이즈 포함).
  깊이·선속도·명령은 쓰지 않는다. Critic은 선속도와 루트 높이를 추가로 받는다.
  action scale은 0.5 rad다.
- 보상: 직립도², 직립 시 높이(0.445 m, std 0.08)·서기 자세, action 변화·관절 파워·관절 속도
  벌점, 가정 관절 범위 초과 벌점.
- **관절 한계 가정:** MJCF에 관절 `range`가 없어 시뮬레이션 관절에는 한계가 없다. 리셋 샘플링과
  범위 벌점에는 서기 자세 기준 HAA ±0.5, HFE ±1.2, KFE ±1.0 rad를 **가정값**으로 쓴다.
  실제 기구 한계로 교체해야 한다. 몸통-다리 충돌을 끈 상태라, 이 벌점이 없으면 다리가 몸통을
  통과하는 해를 학습할 수 있다.
- 규모: 4096환경, [V8 PPO](../config/v8_ppo_cfg.py) 기반(entropy 0.005), 처음부터 학습.
  스모크(작업304): 반복당 약 0.7 s, 98,304 전이(초당 약 14만).
- 평가([`evaluate_v15_recovery.py`](../config/evaluate_v15_recovery.py)): 256환경, 5 s. 성공은
  마지막 1 s 동안 기울기 < 20°이면서 루트 0.35–0.52 m를 유지하는 것이다.

## 통과 기준 (본 훈련 판정용)

- **L-1 (PhysX PD 적응):** PhysX PD 32환경, 평지 시드 42–44 각각 생존 ≥ 31/32, 진행 ≥ 7.5 m,
  토크 한도 초과 요청 0%. ROS2 1로봇 평지 3시드 생존. 험지 L4·내리막은 V14 DCMotor 값 대비 비열화.
- **L-2 (계단):** fixed_v2 15.7 cm 오르막 + heading, 시드 42–44에서 방향·경로 통과
  (`directed_stair_success`) > 0. 목표는 시드당 16/32 이상. 평지 L-1 유지가 전제다.
- **R (복구):** 넘어진 시작 성공률 ≥ 90%(시드 42–44), 선 시작 유지율 ≥ 99%, 토크 한도 초과 요청 0%.
- 모두 실패하면 사실대로 기각한다. 큰 훈련을 실행했다는 것 자체를 개선으로 해석하지 않는다.

## 실행 계획 (HPC `ws1`, `/home/shanghun1404/v15_large_scale`)

| 단계 | 명령 | 예상 |
|---|---|---|
| L 파일럿 | `sbatch launch_v15_locomotion.sbatch 512 1000` | 약 0.5–0.7 h |
| L 본 훈련 | `sbatch launch_v15_locomotion.sbatch 1024 6000` (또는 파일럿 이어서 `... <ckpt>`) | 1024환경 처리량 미측정; 512환경 기준 약 3–4 h |
| L 평가 | `sbatch launch_v15_eval.sbatch <ckpt> <label> flat 42` / `stairs_up 42 --heading-control` | 32환경 약 5분 |
| R USD·드롭 | `sbatch launch_v15_recovery_usd.sbatch` | 완료(작업300) |
| R 본 훈련 | `sbatch launch_v15_recovery.sbatch 4096 5000` | 약 1 h(반복당 0.7 s) |
| R 평가 | `sbatch launch_v15_recovery_eval.sbatch <ckpt> <label> fallen 42` | 약 3분 |

계정 QoS는 동시 GPU 2개·작업 4개다. 같은 계정의 다른 작업과 GPU를 나눠 쓰므로
L과 R을 한 GPU씩 병렬로 돌리는 것이 상한이다. 노드 메모리 58 GB에 맞춰 훈련 요청은 24 GB로 둔다.

## 이번 범위 밖 (후속)

- ROS2 정책 노드의 복구 모드 전환(기울기 기준 전환, 복구 checkpoint 로드)과 heading 명령의
  stamp 동기화(`TwistStamped`). V15 정책이 기준을 통과한 뒤 구현·검증한다.
- 실제 관절 한계, 실기 모터 모델, 비-lockstep 실시간 모드.

준비 단계 측정 기록: [reports/v15/prep](../reports/v15/prep/README.md).
