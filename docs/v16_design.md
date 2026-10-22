# V16: 12개 구동 관절의 물리 모델과 디지털 제어 실험

요청일: 2026-09-29. 사용자는 약 9시간 부재 중 설계 결정을 위임했다.
목적은 RL을 대체한다고 주장하는 것이 아니라 새로운 물리 모델 기반 제어를 실험하는 것이다.
기준 문서는 작업 공간 루트의 `GUIDELINE.md`이며 12단계 모두 식·코드·검증 결과를 연결한다.

## 범위와 완료 기준

1. 선택된 V15 `339/model_11699.pt`의 physics-step 토크 그래프와 관절별 RMS/피크/포화/전력 기록.
   토크 요청과 implicit PD 추정, solver 측정치를 구분한다. 높은 포화가 확인되면 V15를 덮어쓰지
   않고 LPF/guard 후보를 같은 평가 조건에서 비교한다. 실기 연속 정격이 없으므로 열 안전을 추정하지 않는다.
2. 12개 관절을 실제 질량·질량중심·관성·축·관절 위치로 모델링한다. floating base는 추가 6개
   일반화 속도를 갖는다. 몸통에 가상 actuator를 넣지 않는다. 원본 MJCF와 USD/PhysX 간 모델 차이는 검증한다.
3. 물리식에서 local coupled state-space, 초기조건을 포함한 Laplace 해, ZOH, sampled-time,
   B-operator, 각 SISO 채널, 최종 차분식까지 12단계 유도 및 수치 검증을 남긴다.
   접촉 변경·큰 운동에서 하나의 선형 전달함수가 전체 비선형 로봇을 표현한다고 주장하지 않는다.
4. 독립 제어기는 물리 기반 접촉/중력 보상과 상태 피드백을 사용한다. 가용한 PhysX M/h/J API를
   실제 런타임에서 확인하고, 지원되지 않으면 임의의 mass matrix를 만들지 않는다.
5. HPC에서만 모델 계산, 물리 시뮬레이션, 제어 파라미터 훈련을 실행한다. 로컬은 편집·분석·그림만.
6. 짧은 검증을 통과한 뒤 CEM(Cross-Entropy Method)로 제어 이득·주기·궤적 크기를 최적화한다.
   이는 제어 파라미터 학습이며 PPO 학습이나 보행 성공을 의미하지 않는다. seed를 나눈 평가와
   실패도 보존하며 checkpoint/JSON/NPZ/Slurm 로그/모델링 다이어그램을 남긴다.

## 선택한 구조와 대안

- 선택: 실제 rigid-body dynamics + 디지털 상태 피드백 + 접촉을 고려한 직접 관절 effort 제어.
  작은 실험에서 해석 가능하며 토크를 직접 제한하고 physics model과 실제 plant를 비교할 수 있다.
- 대안: frozen V15 RL reference에 모델 기반 저수준 제어를 붙이는 실험. 비교용으로 유용하지만
  V16의 독립 보행 실험을 대신하지 않는다.
- 대안: full constrained MPC. 접촉 QP와 모델/solver 검증을 포함한 구현 비용이 커서 첫 밤의 필수 범위로 잡지 않는다.

모델 기반 제어의 범위는 nominal stand 및 평지 주기 보행부터다. 초기 gate 실패 시 큰 훈련보다
원인을 수정한다. 계단 성능이나 실기 안정성은 별도 검증 없이는 주장하지 않는다.

## 물리 모델 및 이산화

일반화 좌표 Q=(base pose,q), 속도 v=(base twist,dq), q∈R^12.
M(Q)dv+h(Q,v)=Sᵀτ+J_c(Q)ᵀλ; stance contact는 J_c dv+dot(J_c)v=0.
S=[0_(12×6),I_12], τ는 관절 토크이고 몸통 wrench는 actuator가 아니다.
댐핑·마찰·접촉이 model API에 포함되는지 별도로 기록한다.

고정 base 관절 검증에서는 M_j(q)ddq+C_j(q,dq)dq+g_j(q)+D dq=τ.
평형 q0,τ0=g(q0) 주변 x=[δq,δdq]에서 dx=A x+B δτ+d,
A=[[0,I],[-M0^-1 K_g,-M0^-1 D]], B=[[0],[M0^-1]].
입력은 T 동안 고정하고 augmented block exponential로 Ad,Bd,dd를 계산한다.
x[k+1]=Ad x[k]+Bd u[k]+dd; 초기조건 응답 Ad^k x0를 명시한다.
G(B)=C(I-Ad B)^-1 Bd B는 지연 연산자 B의 MIMO 유리함수다.
SISO 채널 G_ij를 분해해도 다른 입력과 접촉력 항은 사라지지 않는다.

## 제어와 학습

- 제어 상태: 관절 위치/속도, 몸통 자세·속도, 발 위치/접촉, reference phase.
- 제어 입력: 각 관절의 직접 torque, 정격 peak 24.8 N·m(HAA),48 N·m(HFE/KFE),
  속도 envelope와 slew/effort guard를 적용한다.
- 학습 변수: 양의 PD/상태 피드백 이득, gait 주기/진폭, 접촉력 보상 비율.
- 목적: 생존·방향 진행·명령 추종을 우선하고 torque RMS/포화/파워를 벌점으로 둔다.
  정지 정책이 torque만 낮아서 우승하지 못하도록 진행과 생존 gate를 적용한다.
- rollout은 부정확/비유한 상태를 종료하고 실패 이유를 기록한다.
- 후보 선택과 최종 평가 seed를 분리한다. 훈련 checkpoint와 모델/코드 SHA-256을 결합한다.

## 실행·결과 기록

Git branch `v16-12dof-model`, 기존 linked worktree 사용; 자동 commit/push 없음.
HPC `/home/shanghun1404/v16_12dof_model/`, Slurm compute/QoS admin.
물리 GPU는 현재 한 개이므로 GPU 작업은 Slurm에서 직렬 실행한다.
CPU 모델 작업은 별도 Slurm으로 실행하고 다른 사용자의 작업을 취소하지 않는다.
9시간은 작업 계획 기간이며 성공이나 종료 시각을 보장하는 값이 아니다.

결과는 `reports/v16/`, 설계/12단계/제어 diagram은 `docs/v16*`에 기록한다.
V15 토크 분석은 `reports/v16/v15_torque/`; 원본 V15 보고서는 수정하지 않는다.

## 자체 검토

관절12DOF와 free-base6DOF 구분, 초기조건·ZOH 입력 명시, 접촉·비선형 한계 명시,
직접 torque와 추정값 구분, baseline 보존, 성공/실패 분리, HPC 전용 실행이 설계에 포함됐다.
사용자의 결정 위임에 따라 spec/plan 재승인과 자동 commit 절차는 생략한다.
