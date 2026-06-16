# FOR Quadruped Robot Training

이 디렉토리는 IsaacLab과 RSL-RL 라이브러리를 활용하여 "FOR"라는 이름의 4족 보행 로봇(Quadruped Robot)을 강화학습(RL)으로 학습시키고 평가하기 위한 환경 및 스크립트를 포함하고 있습니다.

## 주요 기술적 내용 정리

### 1. 학습 환경 구성 (`for_env_cfg.py`)
로봇이 학습할 두 가지 환경이 정의되어 있습니다:
- **ForRoughEnvCfg**: 거친 지형(Rough Terrain)에서의 보행 학습을 위한 환경
- **ForFlatEnvCfg**: 평지(Flat Terrain)에서의 보행 학습을 위한 환경

**핵심 설계 및 보상 체계:**
- **로봇 방향성**: URDF 분석을 통해 로봇의 전진 방향은 `-Y`축, 좌우 이동은 `X`축으로 설정되어 있습니다. (따라서 전진 명령시 `lin_vel_y`에 마이너스 값을 할당합니다).
- **보상(Rewards)**:
  - **전진 및 목표 추종**: 목표 X, Y 속도 추종(`track_lin_vel_xy_exp`) 보상.
  - **자세 및 안정성**: 몸체 높이(0.3m) 유지 보상, 롤링(Roll) 억제 패널티.
  - **페널티**: 관절 가동 범위 제한(limits) 패널티, 부자연스러운 게걸음(X축 이동) 방지 패널티(`lin_vel_x_l2`).
- **종료 조건(Terminations)**: CaT(Curriculum and Termination) 철학에 기반하여, 발(Leg)이 아닌 바디, 모터, 힙(Hip) 등이 지면과 접촉할 경우 즉시 에피소드를 조기 종료합니다.

### 2. PPO 하이퍼파라미터 및 알고리즘 설정 (`for_ppo_cfg.py`)
RSL-RL의 PPO(Proximal Policy Optimization) 알고리즘을 사용합니다:
- **네트워크 구조 (Actor-Critic)**:
  - 거친 지형(Rough): `[512, 256, 128]` 은닉층 및 GRU(128 차원) 적용.
  - 평지(Flat): `[128, 128, 128]` 은닉층 및 GRU(128 차원) 적용.
- **학습 파라미터**: `learning_rate=1.0e-3`, `entropy_coef=0.01`, `clip_param=0.2`, `gamma=0.99`. 

### 3. 로봇 및 물리 엔진 구성 (`for_robot_cfg.py`)
- 로봇의 시뮬레이션 에셋(`for/for.usd`)을 로드합니다.
- `DCMotorCfg`를 통해 다리 관절 모터를 설정합니다. (강성도 `stiffness=60.0`, 감쇠력 `damping=2.0`, 최대 노력 `effort_limit=100.0` 적용).

### 4. 주요 스크립트 기능
- **`train.py` / `run_train.sh`**: 모델을 학습하는 메인 스크립트입니다. Gym 환경(`Isaac-Velocity-Flat-For-v0`, `Isaac-Velocity-Rough-For-v0`)을 등록하고, 2048대의 로봇을 스폰하여 On-Policy 방식으로 훈련을 수행합니다.
- **`check_walking.py`**: 학습된 최신 모델(`.pt`)을 로드하여 Headless 모드로 로봇이 목표 위치로 직진 이동하는지 거리를 계산해 걷기를 검증합니다.
- **`play_video.py`**: 학습된 모델의 보행 시뮬레이션을 비디오(mp4)로 기록하기 위한 평가 스크립트입니다. 노트북 등 낮은 메모리 환경을 위한 최적화 옵션들이 주입되어 있습니다.
- **`test_stability.py`**: 단순 환경 로드 및 물리 안정성 검사를 위한 기본 테스트 스크립트입니다.

---

## 리비전(Revision) 정보
*현재 디렉토리 내에 `.git` 정보가 존재하지 않아 특정 커밋이나 리비전을 확인할 수 없습니다. 만약 기억하고 계신 버전 정보가 있다면 이곳에 업데이트 해주세요.*
