# V16 Implementation Plan

**Goal:** V15 토크를 측정하고 GUIDELINE 12단계를 따른 12관절 물리 제어기를 HPC에서 훈련·평가한다.
**Architecture:** 독립 토크 분석, 물리 모델/이산화, PhysX 제어/학습의 세 결과를 공통 manifest로 연결한다.
**Tech Stack:** Python, Pinocchio 2.7, SciPy, NumPy, PyTorch, IsaacLab/PhysX, Slurm, Matplotlib.
**Spec:** `docs/v16_design.md`.

## Global constraints

물리 계산·훈련·시뮬레이션은 HPC만. 원본 V15 baseline 불변. 자동 commit/push 없음.
몸통에는 외력 actuator를 주지 않는다. measured/estimated effort를 구분한다.
초기조건·latched input·cross coupling을 제거해 해석하지 않는다.

## Review focus

관절 정렬 순서; free-base mass matrix 크기 및 reference frame; 발 collision과 frame offset;
contact phase 불연속; 낮은 torque만 얻는 정지/넘어짐 후보의 채택 방지.

## Task 1 — V15 torque audit (병렬 agent)

- [x] 기존 결과의 source SHA/조건 확인, physics-step trace 없으면 재평가 (508).
- [x] flat/stairs_up/rough 동일 조건, 그래프·CSV·JSON·물리 한계 판정.
- [x] peak limit 접근·높은 포화를 측정하고 LPF/guard 후보 5개와 재훈련 정책을 baseline과 비교.
파일: `config/experiments/v16_20260929/v15_torque_*`, `reports/v16/v15_torque/`.

## Task 2 — 12DOF physical model and digital derivation

- [x] source MJCF rigid bodies/inertias/axes/frames를 Pinocchio로 구성. fixed/floating 두 모델.
- [x] 테스트 먼저: joint names/12+6 크기, mass SPD, RNEA/ABA roundtrip, point Jacobian finite difference.
- [x] HPC에서 모델 원본 hash와 nominal matrices 계산, PhysX와 비교 조건 명시 (520).
- [x] 초기조건 유지 exact ZOH를 analytical double integrator/ODE endpoint로 검증.
- [x] 모든 SISO channel의 상태공간 frequency response 일치,12단계 결과 생성.
파일: `config/v16_physical_model.py`, `config/derive_v16_discrete.py`, `config/tests/test_v16_model.py`.
Produces: model builder + named joint order + floating/fixed dynamics and exported local matrices.

## Task 3 — Digital controller and PhysX plant

- [x] 테스트 먼저: torque-speed envelope·slew·reset, reference continuity, joint order fail-fast.
- [x] IsaacLab runtime M/h/J 크기/API를 probe하고 실제 USD joint/inertia 관측 저장 (521/526/527).
- [x] standalone scene, direct torque, flat ground, fixed seed; 모델 baseline과 closed loop 비교.
- [x] contact support와 joint reference, digital state feedback 구현; trace 저장 경로 준비.
- [x] stand 및 짧은 보행 pilot, 폭주/포화/fall 진단; 최종 독립 보행 기준은 후속 평가로 분리.
파일: `config/v16_controller_math.py`, `config/run_v16_physics.py`, `config/tests/test_v16_controller.py`.
Consumes: joint order and physical parameters; produces rollout score/trace/diagnostics.

## Task 4 — Parameter training and holdout evaluation

- [x] CEM parameter bounds·checkpoint resume 테스트. Holdout seed1001–1003은 훈련 범위에서 제외.
- [x] rollout objective 생존/진행/최소4steps gate, 후보4회 최소값·qualified-only elite/checkpoint 저장.
- [x] pilot 통과 후512환경·128후보×4반복·추가200세대 훈련571 제출 및 실제 실행 확인.
- [x] afterok 평가582/573 및영상581 제출; 이전572/574는 저장 script의 metadata 누락을 확인해 실행 전 교체.
- [x] 훈련571/577 정상 완료 및582/573·578/579의12초/32초 보행 gate·source SHA 검증.
- [x] 영상583/584 정상 완료·source SHA·재생 검증. 이전581/580은 출력 폴더 충돌로 실패해 교체.
파일: `config/run_v16_physics.py --mode train`, `config/launch_v16*.sbatch`, `reports/v16/`.

## Task 5 — Evidence and modeling diagram

- [x] 모델의 rigid links/joint axes,18D generalized model,12 torque actuators, sampling/hold,
  controller/plant/contact feedback를 SVG/PDF/PNG 및 Mermaid로 작성.
- [x] 별도 reviewer가 단위/좌표/order/hold/contact/토크측정/실패판정 검토. 지적 수정 및 회귀 검증.
- [x] raw model/rollout/training source hash와 Slurm 상태 확인, 한국어 결과·자동 상태 보고서 작성.
- [x] 최종 모델의 관찰 카메라 영상 및 비교MP4 저장,24초·600프레임·25fps 확인.

## Execution ledger

Ruling: 사용자의 중간 결정 위임에 따라 재승인 없이 실행한다. 원본 코드는 보존하고 새 파일만 추가한다.
Ruling: 파라미터 학습(CEM)을 첫 V16 훈련으로 선택한다. PPO나 실기 보행 성공으로 표현하지 않는다.
Ruling: linked worktree가 이미 존재하여 추가 worktree 없이 새 V16 branch를 사용한다.
Ruling: 자동 commit/push는 하지 않는다. 설계·계획·실험 결과는 작업 폴더에 저장한다.

Review rulings: final alive gate and failure-time progress freeze; liftoff endpoint interpolation; speed/slew guard activity; boolean metric masking and failed-environment quarantine; slew only at digital ticks; contact point uses explicitly labeled flat-plane foot COM projection.

Evidence: CPU 523 전체 config suite 435 passed / 3 skipped; 524는 437 passed / 3 skipped.
그 뒤 armature 추가 및 command 방향 회귀를 포함해 로컬 controller 대수16 passed.
GPU 510 관절 순서 fail-fast → canonical permutation 수정.
521 기립 4/4 통과 뒤 SDK close hang → 521/미실행522 취소, 출력 저장 후 process 종료로 수정.
526 실제 USD/source rigid M/h/J 정합 확인; 527 독립 가속도 시험에서 solver의 armature 포함 확인 및 기립4/4 통과.
528 +X 64환경/3generation/6초 pilot은 모든 세대 0/64 생존으로 실패.
nominal Jacobian에서 source fore-aft +Y 확인; 529/530은 관리 DRAIN 동안 실행 전 취소됐다.
531–549 실패를 추적해 COM 준비·접촉 제약·실제 normal/friction 측정 및 QP 배분을 추가했다.
551 QP는 12초 생존8/8이나 첫 착지 후 하중 확인 gate에서 정지했다.
접촉 존재와 안정 하중을 분리하고 짧은 재평가 뒤 CEM을 늘린다.
536/542 V15 적응 훈련은 실제 행동 gate를 회복했으나 rough power +1.807%로 채택하지 않았다.
