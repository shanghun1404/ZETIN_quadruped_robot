# V16 물리 기반 제어 다이어그램

[실제 WBC부터 시작하는 3쪽 PDF](../reports/v16/figures/v16_modeling_diagrams.pdf) ·
[전달함수·수식 참고 PDF](../reports/v16/figures/v16_control_transfer_diagrams.pdf) ·
[수식 유도와 게인 설명](v16_control_block_diagrams.md)

첫 그림은 실제 훈련된 WBC가 목표 가속도와 접촉력을 계산하고 토크를 PhysX에
적용하는 **20 ms/5 ms 폐루프**다. 두 번째는 그 물리 plant를 작은 힘·관성·적분
블록으로 분해한 것이다. 세 번째의 \(D_c,K_c\)는 로렌츠 교수 방식에 따른
고정 몸통 **이상적 등가 해석**으로, 실제 WBC 코드의 토크 경로와 구분한다.

![실제 V16 WBC와 물리 plant의 작은 블록 연결](../reports/v16/figures/v16_actual_wbc_blocks.png)

![자유 몸통의 일반화 힘 합산, 관성 및 상태 적분 블록](../reports/v16/figures/v16_physical_force_balance.png)

![고정 몸통 12관절 모델의 이상적 등가 감쇠·강성](../reports/v16/figures/v16_physical_state_feedback.png)

아래의 연속/이산 전달함수 그림은 [참고 PDF](../reports/v16/figures/v16_control_transfer_diagrams.pdf)에 있다.

![연속시간 전달함수 블록선도](../reports/v16/figures/v16_control_s_domain.png)

![이산시간 피드백 블록선도](../reports/v16/figures/v16_control_z_domain.png)

아래는 구현 구조를 설명하는 참고 그림이다.
모델 원본·수치 검증은 `v16_12_steps.md`, 실제 실행 결과는 `reports/v16/`에서 확인한다.

## 물리 모델

```mermaid
flowchart TB
    Base["자유 몸통: 위치·자세 / 6개 속도 자유도\n가상 몸통 actuator 없음"]
    Base --> FL1[FL HAA]
    Base --> FR1[FR HAA]
    Base --> RL1[RL HAA]
    Base --> RR1[RR HAA]
    FL1 --> FL2[FL HFE] --> FL3[FL KFE] --> FLF[FL 발 접촉]
    FR1 --> FR2[FR HFE] --> FR3[FR KFE] --> FRF[FR 발 접촉]
    RL1 --> RL2[RL HFE] --> RL3[RL KFE] --> RLF[RL 발 접촉]
    RR1 --> RR2[RR HFE] --> RR3[RR KFE] --> RRF[RR 발 접촉]
    Model["54개 source rigid member의 질량·COM·관성을 집계\nM(Q) vdot + h(Q,v) = Sᵀ tau + Jcᵀ lambda"]
    Base -.-> Model
```

12DOF는 4다리×3개 구동 관절이다. 몸통 자유도 포함 시 v는18성분이고,
quaternion configuration은19성분이다. 고정 base 시험만 q12,v12,state24가 된다.

## 실제 제어 루프

```mermaid
flowchart LR
    R["COM 준비 crawl FSM\nPREPARE·SWING·SETTLE"] --> Smooth["완료를 확인한 최소 jerk COM 궤적\n연속 위치·속도·가속도"]
    R --> IK["발 COM Jacobian IK\n몸통·관절 reference"]
    Smooth --> FB
    IK --> FB["이산 상태 피드백\nexact-ZOH pole placement"]
    State["관절 q,dq\n몸통 COM·자세·속도"] --> FB
    State --> Dyn["PhysX 강체 모델 oracle\n18×18 M + armature, h, J"]
    FB --> Constraint["선택한 stance point 가속도 투영\nbody/joint task 근사"]
    Constraint --> ID["역동역학 + 접촉력 QP\nbase wrench·마찰·토크 bounds"]
    FB --> WBC["비교 옵션: whole-body QP\n가속도18·접촉력12 공동 조정"]
    Dyn --> ID
    Dyn --> WBC
    Contact --> WBC
    WBC --> Guard
    ID --> Guard["관절 토크·속도·변화율 guard"]
    Guard --> Hold["디지털 tau command\n20ms hold / 5ms 물리 적분"]
    Hold --> Plant["PhysX 자유 몸통·12관절\n지면 접촉·마찰·충돌"]
    Plant --> State
    Plant --> Contact["지면 normal·friction·mean patch\n접촉 존재1N / 안정하중20N"]
    Contact --> R
    Contact --> Constraint
    Plant --> Measure["토크 command / q,dq / 진행\n생존·접촉·전력 trace"]
    Measure --> CEM["CEM 제어 파라미터 훈련\n후보 반복 시험·별도 독립 평가"]
    CEM -.-> R
    CEM -.-> FB
```

QP 모드는 base-wrench equality, 비음수 수직력, 보수적인 마찰 피라미드, 토크·속도·변화율 bounds를
80회 ADMM으로 계산한다. 실제 numerical constraint violation과 infeasible tick 비율을 기록한다.
유한 반복 결과와 별도 모터 guard를 사용하므로 feasibility·최적성·보행 안정성을 보장하지 않는다.
기존 LS 배분기를 비교 옵션으로 보존했다. force-only QP는 주어진 가속도를 고정한다.
새 WBC 옵션은 가속도18·접촉력12를 변수로 두고 같은 contact Jacobian으로
몸통 동역학·접촉 가속도·마찰·토크 bounds를 계산한다. 토크는 동역학 식에서 제거해 복원한다.
320회 ADMM이 기본이며 수치 finite/feasible 여부를 기록한다. 이는 preview MPC가 아니다.
WBC는 model controller에서만 허용하고 solver의 비유한 출력은 해당 환경을 absorbing failure로 처리한다.
CPU의 손계산·Torch 검증과 실제 GPU 기립/보행 검증은 구분한다.
준비 보행은 발에 고정된 local ground point의 가속도를 근사한다. 측정 patch는 압력 재분배로 움직일 수 있고
같은 material point가 보장되지 않는다. 발 COM과 collision contact point 차이를 실행 자료에 명시한다.
20ms 동안 command는 유지하지만 5ms emergency effort projection이 활성화되면 실제 applied command는
변할 수 있다. 선형 exact-ZOH 해석은 포화/접촉 phase 변화가 없는 국소 모델에 해당한다.

## 12단계 유도와 온라인 코드의 관계

```mermaid
flowchart LR
    P["1 물리 → 2 ODE"] --> L["3 초기조건 Laplace\n4 상태 해"]
    L --> H["5 latched step\n6 연속 응답"]
    H --> D["7 T에서 표본\n8 결합 차분식"]
    D --> B["9 B 연산자\n10 MIMO 전달함수"]
    B --> S["11 모든 SISO 경로\n12 각 경로 차분식"]
    D --> Online["안정된 행렬 상태 갱신/디지털 피드백\n고차 scalar recurrence 직접제어와 구분"]
```

SISO 경로를 구해도 off-diagonal 관절 결합과 초기조건 응답은 유지된다.
