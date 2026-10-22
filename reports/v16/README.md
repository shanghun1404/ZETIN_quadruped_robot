# V16 — 12DOF 물리 모델·디지털 제어 실험

2026-09-29 작업. 기존 V15 RL을 보존하고 독립적인 물리 제어기를 실험한다.
물리 계산·시뮬레이션·훈련은 `ssh ws1` HPC에서 진행한다.

## 결과 위치

| 항목 | 파일 |
|---|---|
| V15 토크 그래프·관절별 통계 | [v15_torque/README.md](v15_torque/README.md) |
| 원본 물리 모델 검증 | [physical_model/v16_model_verification.md](physical_model/v16_model_verification.md) |
| GUIDELINE 12단계 식·차분식 | [../../docs/v16_12_steps.md](../../docs/v16_12_steps.md) |
| 실제 V16 WBC → 자유 몸통 물리 블록 → 고정 몸통 등가 해석, 3쪽 | [figures/v16_modeling_diagrams.pdf](figures/v16_modeling_diagrams.pdf) |
| 물리적 게인·초기조건 설명 | [../../docs/v16_control_block_diagrams.md](../../docs/v16_control_block_diagrams.md) |
| 전달함수·비선형 수식 참고 도면, 3쪽 | [figures/v16_control_transfer_diagrams.pdf](figures/v16_control_transfer_diagrams.pdf) |
| 기존 물리 구조·구현 구성도 | [figures/v16_modeling_diagrams_architecture.pdf](figures/v16_modeling_diagrams_architecture.pdf) |
| 고정/자유 몸통의 관성 결합 비교 | [figures/v16_inertia_coupling.png](figures/v16_inertia_coupling.png) |
| 초기조건을 포함한 ZOH 응답 | [figures/v16_zoh_initial_conditions.png](figures/v16_zoh_initial_conditions.png) |
| 제어 방식·연구 근거 | [../../docs/v16_control_research.md](../../docs/v16_control_research.md) |
| 실행 계획 | [../../docs/v16_plan.md](../../docs/v16_plan.md) |
| 야간 훈련·평가·영상 자동 갱신 상태 | [overnight_status.md](overnight_status.md) |

## 현재 검증 범위

원본 MJCF는 54개 rigid body, 구동 관절 12개, 전체 질량 45.4513224696 kg이다.
몸통이 자유로운 모델은 추가 6개의 일반화 속도를 갖는다.
Slurm 520에서 원본 MuJoCo와 Pinocchio 모델 비교 및 13개 테스트를 통과했다.
질량행렬 최대 오차는 1.42e-14, 좌표 변환을 반영한 bias 오차는 5.68e-14다.
이 결과는 원본 모델 정합이다.
실제 USD/PhysX의 한 기립 상태 비교는 [526 검증 보고서](physical_model/physx_comparison_526_verification.md)에 있다.
54개 body와 13개 관성 cluster를 비교했으며 rigid 질량행렬 최대 차이는 3.72e-6,
foot COM Jacobian 최대 차이는 6.36e-7이다. 접촉·전체 보행 안정성 검증과는 구분한다.

PhysX가 내보낸 rigid 질량행렬에는 joint armature .02 kg·m²가 빠져 있었다.
527의 무접촉·초기속도 0·한 physics step 실험에서 armature를 추가한 가속도 예측의
최대 상대 오차는 8.67e-7이었다([원자료](physics_jobs/527/forward_dynamics_validation.json)).
현재 runtime은 이를 추가한다. 실제 USD의 damping은 0, source MJCF는 1.5이므로
아래 source 선형 모델의 pole을 실제 USD 폐루프 pole로 표현하지 않는다.

5/20 ms exact ZOH와 초기조건을 포함한 결합 모델, 144개 SISO 채널 및 차분식이 저장돼 있다.
이 선형 모델은 명시한 고정 몸통 nominal 상태 주변의 모델이다.
보행에는 자유 몸통의 비선형 M/h/J, 모터 관성과 접촉력 배분을 사용한다.
몸통에 외력 actuator를 주지 않으며 12개 관절에만 effort를 입력한다.
첫 훈련은 CEM으로 제어 파라미터를 학습한다. PPO 학습과 구분한다.

## 작업 기록

| Slurm | 작업 | 확인된 상태 |
|---|---|---|
| 508 | V15 baseline 토크 재평가 | COMPLETED, 기존 339 결과 재현 |
| 516 | V15 guard/LPF 후보 3개 × 평지/거친 지형 | COMPLETED; 두 지형의 모든 사전 gate를 통과한 후보 없음 |
| 525 | V15 guard/LPF 추가 후보 2개 | COMPLETED; 모든 gate를 통과한 후보 없음 |
| 536/542 | V15 모터 설정 적응 훈련·비교 평가 | 훈련·평가 완료; 행동 회복, rough 추정 power +1.807%로 전체 gate 실패 |
| 517 | 기존 config 회귀 테스트 | 424 passed, 11 skipped |
| 523 | source 물리 테스트를 포함한 config 회귀 | COMPLETED, 435 passed, 3 skipped |
| 524 | objective/resume 회귀 추가 | COMPLETED, 437 passed, 3 skipped |
| 520 | 원본 모델·12단계 최종 검증 | COMPLETED, 13 passed 및 MuJoCo 비교 통과 |
| 510 | V16 첫 기립 probe | FAILED: native 관절 순서 차이를 사전 검증에서 발견 |
| 521 | 관절 이름 permutation 수정 후 기립 probe | 4/4 기립 통과·자료 저장 후 SDK 종료 hang; 작업 CANCELLED |
| 522 | 이전 CEM pilot | 실행 전 CANCELLED; 521 종료 hang 수정 후 새 작업으로 대체 |
| 526 | 실제 USD/PhysX ↔ source 정합 | COMPLETED, 좌표 변환·armature/damping 차이 검증 |
| 527 | armature 가속도 검증 및 기립 재시험 | COMPLETED, 4/4 기립 통과; 정상 종료 확인 |
| 528 | 64환경 × 3세대, +X 방향 pilot | COMPLETED, 모든 세대 생존 0/64; 보행 실패 |
| 529/530 | +Y 평가·pilot | GPU 관리 중 실행 전 취소; 결과 없음 |
| 531/533/535/537 | +Y·지지 제약·COM 준비를 순차 비교 | 완료; 첫 보행 중 실패, 원자료 보존 |
| 539/543 | rigid ground point / measured patch 비교 | 완료; ground는 첫 발, measured는 준비 중 실패 |
| 544/545/548 | static joint friction 0 비교 | 기립 4/4, 보행 0/8; dynamic/viscous 값은 이 작업에서 미기록 |
| 547 | QP/Torch를 포함한 CPU config 회귀 | COMPLETED; 479 passed /3 skipped, 최신 runtime 통합 검증과 구분 |
| 549 | static/dynamic/viscous 모두 0, LS 힘 배분 | 완료; 보행 0/8, 세 마찰 값 모두 실제 0 확인 |
| 550/551 | 동일 마찰 0 조건, QP 힘 배분 | 기립 4/4, 12초 생존 8/8; 첫 발 착지 후 하중 gate에서 정지, 보행 완료 아님 |
| 552/553 | 접촉 존재와 하중 확인 분리 | 16초 생존 각각 2/8·4/8; 일부 두 번째 다리 준비 중 정지 또는 넘어짐 |
| 554 | 접촉 mask/QP Torch CPU 회귀 | COMPLETED; 482 passed /3 skipped |
| 555 | 기존 64환경×12세대 CEM | 완료; 단일 최고 후보 8걸음·0.518m, 이후 분포는 정지 쪽으로 수축; 독립 성공 아님 |
| 557 | 몸통 가속도 추종 비중1000 | 3–5걸음 진행 후 0/8 생존; 목표 위치 전환 불연속 확인 |
| 558 | 최소 jerk COM reference | 16초 8/8 생존, 5–9걸음; 다음 궤적의 시작 조건 보완 전 자료 |
| 559/560 | COM 궤적 CPU 검증 | 559의482 tests에는 COM tests 미포함; 경로 수정 뒤560에서 COM 25 tests 통과 |
| 561 | 궤적 완료 조건·handoff 보완 | 16초 8/8 생존, 5–9걸음, 0.319–0.530m; 목표 속도보다 느림 |
| 563 | 부드러운 COM, 128환경×24세대 | 완료; 훈련 내 최고0.720m·9걸음, 독립 평가 아님 |
| 564 | 가속도·접촉력 공동 QP CPU 검증 | 16 tests 통과, Torch CPU parity 포함; GPU 속도·보행은 별도 검증 |
| 565/566 | WBC 기립 비교 | 565 실행 전 취소, 리뷰에서 발견한 수치 실패 gate 보완 후566 제출 |
| 566 | WBC 보완 후 기립 | 완료; 4/4 생존5초, 모든tick QP numerical tolerance 내 |
| 567 | 최신 전체 CPU config 회귀 | 완료; 539 passed /3 skipped, 64 subtests passed |
| 568 | 반복 검증128후보×4환경×12세대 | 완료; 최종512/512 생존, 선택 후보4회 최소진행0.751m·9걸음 |
| 569/570 | QP/WBC 동일 파라미터 비교 | 569 실행 중 출력경로 오류를 발견해취소; 수정된570 완료, QP8/8/WBC0/8 생존 |
| 571 | 같은 분포·RNG에서 추가200세대 | 정상 완료2시간3분27초, 최종generation211; 최종512/512 생존 |
| 572/582 | 훈련과 분리한12초 독립 평가 | 572 저장 script의 duration metadata 누락을 발견해 실행 전 취소; 최신582를 afterok571로 제출,1001–1003 각32환경 및PD 비교 |
| 573 | 고정된 같은 모델의32초 평가 | 완료;1001–1003 각32/32 생존, 진행1.543–1.544m, 보행 기준 통과 |
| 574/581/583 | QP 모델24초 RGB 영상 | 574 실행 전 교체,581은 출력 폴더 충돌로 실패; 준비 입력과 출력 폴더를 분리한583 완료;24초 생존·15걸음·1.209m |
| 575 | WBC 가속도 가중치의 변수 스케일 보정 | 16초8/8생존·각9걸음·0.719–0.722m, QP tolerance 위반/guard0; 최적화 수렴률은약76% |
| 576/577 | WBC128후보×4환경,8+120세대 | 정상 완료;577은2시간19분26초, 최종generation127·512/512 생존 |
| 578/579 | WBC 독립12초·32초 평가 | 모두 완료·보행 기준 통과;32초 각32/32 생존, 진행1.549–1.550m |
| 580/584 | WBC 모델24초 RGB 영상 | 580은 출력 폴더 충돌로 실패; 동일 frozen controller로584 완료;24초 생존·15걸음·1.122m |

510 실패 로그를 보존했으며 상태·M/h/J·토크를 동일한 canonical 순서로 변환하도록 수정했다.
528 이후 nominal Jacobian으로 HAA 운동은 주로 X, HFE/KFE 운동은 주로 Y임을 확인했다.
독립 V16 runtime의 기본 진행 축을 +Y로 수정했다. V15 policy 설정을 바꾼 것은 아니다.
quarter-offset duty 범위를 .80–.90으로 바꾸고 COM 준비·단일 swing·착지 확인 FSM을 추가했다.
LS 배분기의 회전모멘트 잔차와 실제 접촉의 차이를 기록하고 QP를 비교했다.
551은 넘어지지 않았지만 새 발에 하중을 배분하는 조건과 20N 하중 확인 조건이 같아 정지했다.
접촉 존재 1N과 안정 하중 20N을 분리했다. 이후 몸통 목표를 따라가도록 투영 가중치를 높이고,
다리 전환 시 약12cm였던 COM 목표 점프를 최소 jerk 궤적으로 연결했다.
561의 궤적 속도·가속도 한계는 reference에 적용되며 실제 로봇 운동을 제한하는 보장은 아니다.
이전8/8 pilot과 구분하여, 최종 모델은12초·32초 독립 seed 평가에서 두 제어기 모두 보행 기준을 통과했다.
32초1001–1003 각32환경에서 각각96/96생존했다.
현재 force-only QP는 일부 tick에서 수치 제약을 위반하며 별도 토크 guard가 명령을 제한한다.
새 whole-body QP는 가속도와 접촉력을 함께 조정하는 비교 옵션으로 준비했다.
실행·실패 근거는 [controller_debug.md](controller_debug.md)에 있다.

HPC 작업 경로: `/home/shanghun1404/v16_12dof_model/`.
각 GPU 작업은 `job_sources/<job_id>/`에 소스를 복사하고 manifest에 SHA-256을 기록한다.
결과는 `physics_jobs/<job_id>/`; CEM은 매 generation `checkpoint.json`을 원자적으로 저장한다.
각 GPU 작업은 GPU 1개를 요청한다. HPC는 현재 GPU 2개/58GB로 복귀했으며 자원에 맞춰 병렬 실행한다.
Git 자동 commit/push는 하지 않았다.

로컬 관찰 스크립트는 Slurm 상태·완료 결과를2분 간격으로 복사하고
`overnight_status.md/json`을 갱신한다. 훈련·시뮬레이션은 이 스크립트에서도 실행하지 않는다.
전체 후속 작업은 Slurm 의존성으로 연결되어 있다. 실패/미완료 상태를 성공으로 바꾸지 않는다.

575의 [12관절 토크](physics_jobs/575/figures/trace_seed71_torque.png),
[COM와제약](physics_jobs/575/figures/trace_seed71_control.png),
[primal/dual 및가속도 추종](physics_jobs/575/figures/trace_seed71_convergence.png)을 확인할 수 있다.
현재 명령속도는0.05m/s, 평지에서의 실험이다. V15의0.4m/s 정책과 소비전력을 직접 우열 비교하지 않는다.

## 최종 모델 보행 영상

[나란히 비교MP4](../../videos/v16/20260929/v16_comparison.mp4) · [접촉력 QP](../../videos/v16/20260929/v16_contact_qp.mp4) · [전신 QP](../../videos/v16/20260929/v16_wbc.mp4).

HPC583/584 정상 완료. 각각24초·600프레임·25fps·960×640이며 둘 다15걸음 동안 생존했다.
같은seed1001·명령0.05m/s에서 QP1.209m, WBC1.122m 진행했다. 비교 영상은 왼쪽QP·오른쪽WBC이며 원래 시간을 유지했다.
source/checkpoint SHA 검증,MP4 전체 decode,12초 화면 구도 확인을 마쳤다. [영상 원자료와SHA](../../videos/v16/20260929/manifest.json)에 기록했다.
