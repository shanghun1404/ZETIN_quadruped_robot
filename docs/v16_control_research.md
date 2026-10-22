# V16 물리 모델 기반 제어 연구

작성: 2026-09-29. 범위: 사용자 요청의 별도 제어 실험 설계와 1차 문헌 조사. 이 문서는 보행 실험이나 훈련 결과를 주장하지 않는다. 기존 RL을 대체한다는 결론도 내리지 않는다.

## 권고

**12개 관절을 모두 포함하는 강체 동역학을 바탕으로, 접촉 모드별 MIMO 모델을 만들고 exact ZOH 이산화와 이산 상태 피드백을 검증하는 V16 실험을 우선한다.** 물리 파라미터 보정과 제어기 gain 탐색은 시스템 식별/제어기 최적화로 기록한다. PPO 정책 훈련과 혼동하지 않는다.

우선순위는 물리 항 검증 → 고정기저 12관절 결합 시험 → 자유기저 서기 → 별도 보행 reference다. 보행 reference는 analytic gait/foot trajectory/IK로 만들 수 있고, 시간이 허용하면 frozen RL reference를 **별도 비교군**으로 추가한다. RL reference를 사용한 결과는 RL 없는 독립 보행의 결과로 집계하지 않는다.

아래 식은 이 문서의 도출이다. 외부 문헌은 약어의 용례, API 기능 및 기존 제어 구조의 근거다. 도출의 `exact`는 **샘플 구간 안에서 동결한 선형 모델과 실제 latched 입력**에 대한 정확성이다. 접촉이 바뀌는 비선형 PhysX 로봇 전체를 exact linear plant로 표현했다는 뜻이 아니다.

## 1. 약어 확인 결과

| 사용자 표기 | 1차 문헌에서 확인한 표기 | V16 판단 |
|---|---|---|
| MID | Manipulated Input Decoupling | 입력 간 결합 보상 후보. 역행렬/의사역행렬의 rank·conditioning·포화·지연을 검증한 뒤 사용한다. |
| DID | Disturbance Input Decoupling | 측정/추정 가능한 외란의 feedforward 보상 후보. 접촉 힘을 정확히 알거나 임의로 독립 입력처럼 지정할 수 있다고 가정하지 않는다. |
| VZR | Virtual Zero Reference; VZRD는 Virtual Zero Reference Decoupling | 문헌의 모터/전력변환기 용례를 확인했다. 로봇에서 어떤 물리 항과 reference를 뜻하는지 수식으로 정의한 뒤 적용한다. |
| NLDSFb | Catania 학위논문에 `No Linear Decoupling State Feedback`라는 정확한 표기가 있고, 원 논문 계보는 Johnson–Lorenz의 friction compensation 연구로 이어진다. | 일반적인 nonlinear decoupling state feedback와 유사한 용례지만 사용자 의도의 정확한 확장은 아직 확정하지 않는다. V16 구현에는 명확한 명칭인 feedback linearization/computed torque를 사용한다. |
| SDFb | 해당 철자의 확장은 확인하지 못했다. 다른 1차 문헌은 SFBD, DCCSFb, DSFb를 사용한다. | 사용자 표기를 다른 약어와 같다고 단정하지 않는다. 이번 실험에서는 discrete state feedback라고 식과 함께 명시한다. |

MID/DID의 정확한 표기는 Schütt–Eckel의 [EPE 2018 저자 포스터](https://www.netz-stabil.uni-rostock.de/storages/uni-rostock/Alle_IEF/NetzStabil/Vortraege/EPE2018_Poster_Schuett.pdf)에서 확인했다. SFBD/VZRD/MID/DID는 [UW 학위논문, Systematic Control Approaches for Three-Phase AC-DC](https://asset.library.wisc.edu/1711.dl/ENF57DEOU465X8Y/R/file-462ba.pdf)에 함께 나타난다. VZR는 [UW 학위논문, Deadbeat Flux and Torque Control for Six-Step Operation](https://asset.library.wisc.edu/1711.dl/ETS2SH6PRNECX8R/R/file-62095.pdf)의 약어 목록 및 LR circuit 논의에서 확인했다. 이 논문은 저항 보상 오차 때문에 reference 추종 오차가 남을 수 있다는 점도 다룬다.

NLDSFb 표기 근거는 [Mario Pulvirenti 학위논문](https://www.iris.unict.it/retrieve/8e41ac77-971c-48fa-8248-adc700b11e3e/TESI_MarioPulvirentiA.pdf)의 검색 색인에서 확인했다. 전체 PDF 직접 열기는 403으로 실패했다. 원 논문 [Johnson–Lorenz, Experimental Identification of Friction and its Compensation in Precise, Position Controlled Mechanisms, 1992](https://minds.wisconsin.edu/handle/1793/10262)의 공식 저장소 메타데이터는 확인했지만 원문 PDF 접근이 실패했으므로, 원 논문이 그 약어 철자를 쓰는지까지 확정하지 않았다. **미확정 약어가 V16 실험 실행의 전제가 되어서는 안 된다.**

## 2. 모델 차원과 물리 범위

12DOF는 이 로봇의 **actuated joint 수**다. 자유기저 강체 모델의 일반화 속도는 18차원이다.

\[
q=(p_b,R_b,q_j)\in SE(3)\times\mathbb R^{12},\qquad
v=(v_b,\omega_b,\dot q_j)\in\mathbb R^{18}.
\]

quaternion 저장 시 raw configuration은 19성분이고, 자세의 국소 오차는 3성분을 사용한다. 따라서 local error state는 \(x=(\delta q,\delta v)\in\mathbb R^{36}\)다. quaternion 4성분에 독립적인 선형 상태 4개를 부여하지 않는다. 고정기저 시험은 \(q_j,v_j\in\mathbb R^{12}\), \(x\in\mathbb R^{24}\)다.

물리 모델의 각 link에는 질량, COM, 관성, joint transform/axis, gear/armature, damping/friction, 한계가 필요하다. 54개 USD body가 있어도 actuated joint는 12개이며, rigid하게 연결된 cosmetic body의 질량/관성을 빠뜨리면 안 된다. 모델은 실제 시뮬레이션에 올라간 USD의 물리 값과 일치해야 한다. 원 MJCF와 변환/수정 USD의 값이 다를 수 있다.

로컬 [guide_robot_v2_cfg.py](../config/guide_robot_v2_cfg.py)에는 총 질량 45.451 kg, HAA 24.8 N·m, HFE/KFE 48.0 N·m, actuator stiffness 300, damping 8, friction 0.3가 기록되어 있다. 이것은 **설정/기존 기록**이며 V16 런타임 측정값을 대신하지 않는다. 특히 friction 숫자를 Coulomb torque나 viscous coefficient로 임의 해석하지 말고 설치된 actuator/PhysX 구현의 단위와 의미를 확인한다. 같은 설정 파일의 body frame offset 관련 기록에 따라 foot Jacobian/contact point는 실제 collision geometry 위치와 대조해야 한다.

## 3. 자유기저와 접촉 방정식

입력 \(u=\tau_j\in\mathbb R^{12}\), 선택행렬 \(S=[0_{12\times6}\ I_{12}]\)를 사용한다.

\[
M(q)\dot v+h(q,v)+d(q,v)=S^T\tau_j+J_c(q)^T\lambda,
\qquad \dot q=\mathcal N(q)v.
\]

\(h=Cv+g\), \(d\)는 별도 damping/friction/actuator 항이다. \(\lambda\)는 지면 접촉 힘이다. base 6개 축에 commanded wrench를 넣지 않는다. [MIT contact 강의](https://underactuated.mit.edu/contact.html)는 자유기저 모델에 contact Jacobian과 constraint force를 추가하는 구조 및 mode/guard/reset 모델을 설명한다.

발이 미끄러지지 않는 지속 접촉 phase에서는 \(J_c v=0\), \(J_c\dot v+\dot J_cv=0\)를 사용한다.

\[
\begin{bmatrix}M&-J_c^T\\J_c&0\end{bmatrix}
\begin{bmatrix}\dot v\\\lambda\end{bmatrix}
=
\begin{bmatrix}S^T\tau_j-h-d\\-\dot J_cv\end{bmatrix}.
\]

접촉은 일방향이고 마찰 한계가 있다. 최소 \(\lambda_n\ge0\), \(\|\lambda_t\|\le\mu\lambda_n\) 및 발의 separation/penetration 조건이 필요하다. 중복 contact constraint는 rank 검사를 거쳐야 하며, 단순 KKT inverse가 항상 존재한다고 가정하지 않는다. 접촉 전환과 충격은 별도의 hybrid reset이고, 같은 LTI 전달함수를 touchdown 전후에 무조건 계속 쓰면 안 된다.

자유비행에서는 base row를 제거해 관절에 대한 유효 inertia를 얻을 수 있다. \(M\)을 b/j block으로 나누면

\[
M_{\rm eff}=M_{jj}-M_{jb}M_{bb}^{-1}M_{bj},\qquad
h_{\rm eff}=h_j-M_{jb}M_{bb}^{-1}h_b,
\]

\[\tau_j=M_{\rm eff}\ddot q_j+h_{\rm eff}\]

이다(이 식에는 \(d\)를 생략했으므로 실제 구현에서는 같은 소거로 포함한다). **free-base에서 \(M_{jj}\)만 써서 computed torque를 만드는 것은 일반적으로 다른 모델이다.** 지속 접촉에서는 위 KKT/contact projection을 사용해야 한다.

## 4. GUIDELINE 12단계의 V16 도출

표기 충돌을 피하려고 입력행렬을 \(F\), backshift 연산자를 \(B\)로 쓴다. 접촉 phase \(\sigma\)와 nominal operating point/trajectory 주변의 국소 오차 모델을

\[\dot x=A_cx+F_cu+E_cw+c_c,\qquad y=C_cx+D_cu\]

로 정한다. \(u\)는 nominal torque에서의 편차다. 고정점에서는 \(c_c=0\)이 되도록 평형 입력을 찾아야 한다. moving reference에서는 feasible nominal trajectory의 미분을 빼고 남은 defect를 \(c_c\)로 기록한다. 단순 \(x-x_{ref}\)만 사용하고 reference velocity/acceleration을 빼지 않으면 오차 동역학이 달라진다.

| 단계 | V16에서 남길 식/산출물 |
|---|---|
| 1 물리 | 각 link의 Newton–Euler/강체 관성, actuator 및 접촉 모델. asset hash·joint order·frames·units. |
| 2 ODE | 위 nonlinear \(\dot q,\dot v\), phase별 local \(A_c,F_c,E_c,c_c\). fixed base의 absolute-input 식은 \(\dot q_j=v_j\), \(\dot v_j=M^{-1}(\tau_j-h-d)\). |
| 3 Laplace+초기조건 | 구간 시간을 \(\theta=t-kT\)로 놓고 \((sI-A_c)X(s)=x_k+F_cU(s)+E_cW(s)+c_c/s\). 모든 상태의 \(x_k\)를 유지한다. |
| 4 상태별 해 | \(X(s)=(sI-A_c)^{-1}[x_k+F_cU(s)+E_cW(s)+c_c/s]\). 행별 전개에도 cross-coupled 초기조건을 삭제하지 않는다. |
| 5 latched step | \(U(s)=u_k/s\), hold 외란 가정이면 \(W(s)=w_k/s\). 실제 hold 대상·period·latency를 로그에 남긴다. |
| 6 구간 응답 | \(x(\theta)=e^{A_c\theta}x_k+\int_0^\theta e^{A_c(\theta-r)}(F_cu_k+E_cw_k+c_c)dr\). |
| 7 다음 sample | \(\theta=T\) 대입, \(x_{k+1}=A_dx_k+F_du_k+E_dw_k+c_d\). |
| 8 결합 차분식 | 각 행 \(x_i[k+1]=\sum_l A_{d,il}x_l[k]+\sum_j F_{d,ij}u_j[k]+\cdots\). off-diagonal 항을 보존한다. |
| 9 B-operator | \(Bx[k]=x[k-1]\). \((I-A_dB)x[k]=F_dB u[k]+E_dB w[k]+c_d\). 상수항의 backshift는 같은 상수다. 시작 시각의 homogeneous response는 별도 초기 memory로 보존한다. |
| 10 결합 전달행렬 | **zero IC에 대해서만** \(G(B)=C_c(I-A_dB)^{-1}F_dB+D_c\). nonzero IC의 응답은 단계 6–8의 homogeneous term을 추가한다. |
| 11 SISO parsing | 모든 \(G_{ij}(B)=N_{ij}(B)/D_{ij}(B)\)를 도출하고 \(y_i=\sum_{j=1}^{12}G_{ij}u_j+\)외란/초기조건 응답으로 재구성한다. |
| 12 scalar 차분식 | \(D_{ij}(B)=1+\sum_{l=1}^r a_lB^l\), \(N_{ij}(B)=\sum_{l=0}^m b_lB^l\)이면 \(y_{ij}[k]=-\sum_{l=1}^r a_ly_{ij}[k-l]+\sum_{l=0}^m b_lu_j[k-l]\), \(y_i=\sum_jy_{ij}\). nonzero IC memory를 별도 초기화한다. |

SISO parsing은 각 input/output pair의 scalar 표현이다. **물리 관절들이 독립이라는 결론, \(G\)의 off-diagonal가 0이라는 결론, 12개 diagonal PID로 전 상태를 안정화했다는 결론을 주지 않는다.** 12 joint-position outputs는 12×12의 144 scalar 경로를 갖는다. 전 상태 출력이면 경로 수는 더 많다. 전달함수 식을 만들었다는 이유로 불안정한 pole-zero cancellation이나 고차 recurrence를 그대로 실시간 구현하지 않는다. 동일 모델의 matrix state update 또는 검증된 최소실현/안정된 section 구현을 함께 남긴다.

고정 phase/LTI에서 초기조건 응답은 \(y_h[k]=C_cA_d^kx_0\)로 **한 번만** 더할 수 있다. 이 경우 144개의 zero-state scalar 경로는 모두 zero memory로 시작하고 \(y_i[k]=y_{h,i}[k]+\sum_j y_{ij}[k]\)로 재구성한다. \(x_0\)의 효과를 각 input pair에 중복 주입하지 않는다. phase 전환 시에는 살아 있는 상태와 memory를 보존/변환하며 새 equilibrium에서 모두 0으로 reset하지 않는다.

행렬지수로 계산하면 \(A_c\)가 singular해도 처리할 수 있다.

\[
\exp\left(T\begin{bmatrix}A_c&F_c&E_c&c_c\\0&0&0&0\\0&0&0&0\\0&0&0&0\end{bmatrix}\right)
=\begin{bmatrix}A_d&F_d&E_d&c_d\\0&I&0&0\\0&0&I&0\\0&0&0&1\end{bmatrix}.
\]

역행렬이 존재할 때만 \(F_d=A_c^{-1}(A_d-I)F_c\)와 비교할 수 있다. singular double integrator에서 이 inverse 공식을 쓰지 않는다. [SciPy cont2discrete 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html)는 ZOH 변환과 state-space 입력을 지원한다.

### Hold/초기조건 검증 요점

- 기존 [v8_env_cfg.py](../config/v8_env_cfg.py)는 physics dt=0.005 s, decimation=4다. V16 실제 실행값을 다시 기록한다. 5 ms low-level torque와 20 ms reference hold는 서로 다른 샘플링 계층이다.
- 현재 구현 담당자의 V16 선택은 **20 ms torque-command ZOH, 5 ms PhysX integration**이다. 따라서 제어용 \(A_d,F_d\)는 \(T=0.020\)으로 만들고 4개 physics substep에서 command를 유지한다. 제한기가 substep마다 torque를 변경하면 command hold와 actual torque hold를 구분해 로그/모델에 남긴다. 아래 합성 대수 검증의 5 ms는 실제 V16 sample period의 측정값이 아니다.
- tau command를 hold하면 위 torque-input 식을 사용한다. **position target을 hold하는 implicit PD는 torque를 hold하지 않는다.** joint position/velocity가 바뀌면 torque가 바뀐다. position-input plant라면 PD stiffness/damping·필터·제한·지연을 모델 안에 넣는다.
- LPF가 5 ms마다 갱신되면 20 ms raw policy target은 hold여도 applied target/torque는 hold가 아니다. 필터 state를 포함하거나 실제 actuator period로 재도출한다.
- 한 sample 입력 지연이면 \(x_{k+1}=A_dx_k+F_du_{k-1}\). \(z_k=(x_k,u_{k-1})\)로 확장해 \(z_{k+1}=\begin{bmatrix}A_d&F_d\\0&0\end{bmatrix}z_k+\begin{bmatrix}0\\I\end{bmatrix}u_k\)를 사용한다. 지연을 Euler 근사에 숨기지 않는다.
- nonzero position, velocity, actuator/filter memory, delayed command를 각각 주는 회귀 검증이 필요하다. `zero IC TF`의 impulse 응답만 맞아도 초기조건 도출 검증을 통과한 것은 아니다.
- linearization의 Jacobian은 manifold integrate/difference로 perturb한다. contact mode 경계를 가로지르는 finite difference는 smooth local derivative로 취급하지 않는다.

## 5. 제어기 비교

관절 추종 오차는 \(e=q_r-q_j\), \(\dot e=\dot q_r-\dot q_j\)로 정의한다.

| 방식 | 적용 식/장점 | 제한 및 실험 위치 |
|---|---|---|
| 12 joint PD | \(\tau=K_pe+K_d\dot e+\tau_{ff}\). 비교군이 간단하다. | coupling은 그대로 남는다. gain 튜닝으로 물리 모델이 완성되는 것은 아니다. |
| computed torque | fully actuated fixed base에서 \(\tau=M(q)(\ddot q_r+K_d\dot e+K_pe)+h+d\). 모델이 정확하면 \(\ddot e+K_d\dot e+K_pe=0\). | 우선 12joint bench 검증. 자유기저/접촉에 그대로 적용할 수 없다. inertia·friction 오차, saturation, discrete hold가 이상적 error equation을 바꾼다. |
| feedback linearization/MID | 선택 output의 decoupling matrix를 이용해 input/output coupling을 보상한다. | 18 DOF에 입력12라서 임의의 full-state dynamics를 모든 축에서 지정할 수 없다. 선택 output의 relative degree, rank, zero dynamics, contact constraints가 필요하다. 역행렬이 불량하면 regularized solve를 쓴다. |
| discrete MIMO LQR | \(u=-Kx+u_{ff}\), \(K=(R+F_d^TPF_d)^{-1}F_d^TPA_d\). 결합 항을 보존하고 상태/입력 비용을 명시한다. | local stabilizable model에 대한 설계. 입력 saturation·contact switching까지 자동 보장하지 않는다. contact constraint에 맞는 reduced coordinates 또는 projected model이 필요하다. |
| constrained whole-body QP | \(\dot v,\lambda,\tau_j\)를 동시에 찾고 base dynamics·contact·friction·torque limits를 강제한다. | standalone gait의 현실적 기반. nominal foot/body task가 먼저 필요하다. QP feasible rate·solve time·residual을 측정한다. |

[MIT LQR 강의](https://underactuated.mit.edu/lqr.html)는 nonlinear system의 local/trajectory linearization과 discrete Riccati를 설명한다. [SciPy solve_discrete_are](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html)의 DARE solver를 사용한다면 residual, P의 대칭성/유효성, closed-loop eigenvalue 및 stabilizability를 함께 확인한다. global translation/yaw 같은 unregulated neutral mode를 포함한 모델에 무조건 positive Q를 주고 실패를 gain 문제로 해석하지 않는다.

기존 4족의 model-based 보행 구조는 [Di Carlo et al., Cheetah 3 convex MPC, 2018](https://dspace.mit.edu/entities/publication/bc8c7e1e-5830-443f-a879-787947111fcf)와 [Kim et al., Whole-Body Impulse Control and MPC, 2019](https://arxiv.org/abs/1909.06586)가 근거다. 전자는 보행 계획에서 simplified body dynamics를 사용한다. 따라서 V16의 **full 12joint low-level dynamics**와 high-level reduced planner를 구분해야 한다. 논문의 다른 로봇 성능/속도/연산시간을 이 asset의 성능 예상치로 옮기지 않는다.

## 6. 질량행렬 및 bias 항 확보 경로

### A. PhysX/Omni Physics를 먼저 확인

설치된 IsaacLab의 `robot.root_physx_view`가 노출하는 기능과 backend 버전을 확인하고 다음 getter 존재/shape를 저장한다.

| API | floating base 예상 shape | 주의 |
|---|---|---|
| `get_generalized_mass_matrices()` | (N,18,18) | 실제 joint order와 root velocity frame을 기록한다. |
| `get_coriolis_and_centrifugal_compensation_forces()` | (N,18) | base 항을 포함하는 새 API인지 확인한다. |
| `get_gravity_compensation_forces()` | (N,18) | 반환값의 부호/단위를 static gravity test로 확인한다. |
| `get_jacobians()` | body/link별 6×18 경로 | body COM frame, foot contact offset, world/local frame을 검증한다. |

[NVIDIA Omni Physics tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)는 generalized mass matrix의 floating-base shape와 world-frame base velocity를 명시한다. 같은 문서에서 legacy `get_mass_matrices()`는 root acceleration→root force 변환을 제공하지 않고, legacy gravity/coriolis getter는 base 항을 생략한다고 설명한다. 버전이 다른 wrapper의 함수 이름만 믿고 12×12 matrix를 전체 floating-base M이라고 표시하지 않는다.

[NVIDIA inverse dynamics 공식 예제](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/extensions/runtime/source/omni.physics.tensors/docs/inverse_dynamics.html)는 generalized M, Coriolis, gravity로 inverse dynamics를 구성한다. 이 계산은 contact/damping/joint friction을 포함하지 않는다. 공식 floating gripper 예제는 base에 별도 force/torque를 적용하지만 **12 actuator quadruped 실험에서는 그 부분을 사용하면 안 된다.**

검증 순서: shape/joint permutation → M symmetry/positive definiteness → gravity/static support → no-contact forward acceleration residual → contact force 포함 residual → local finite-difference/Jacobian residual. `computed_torque`, `applied_torque`, drive/contact/constraint force의 의미를 구별한다. 필요한 API가 없으면 matrix 이름을 바꾸는 것으로 대체하지 않는다.

### Jacobian 원점: COM과 actor origin을 구별

[PhysX 5.3 articulation Jacobian 문서](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/Articulations.html)는 dense Jacobian이 **link COM의** world-frame spatial velocity \([v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]\)로 mapping한다고 명시한다. [PhysX 5.5 문서](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html)는 link velocity/acceleration이 COM 기준이고 pose는 actor-frame 기준이라고 구별한다.

wrapper에 대한 직접 근거도 있다. [IsaacLab 공식 changelog](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/refs/changelog.html)는 이전 `Articulation.get_jacobians()` passthrough가 COM 기준이며 link-origin pose와 함께 사용하면 offset bias가 생긴다고 기록한다. [공식 PhysX backend source](https://isaac-sim.github.io/IsaacLab/develop/_modules/isaaclab_physx/assets/articulation/articulation_data.html)는 `body_com_jacobian_w`에서 `_root_view.get_jacobians()`를 읽고, `body_link_jacobian_w`에서 COM→origin 변환을 한다. 이 최신 문서의 새 property 존재를 기존 설치에 가정하지 말고 raw API의 기준점 근거로 사용한다.

접촉점 \(p\)가 foot COM과 다르면 \(r=p-p_{COM}\)를 world frame으로 계산한다.

\[
v_p=v_{COM}+\omega\times r,\qquad
J_{v,p}=J_{v,COM}-[r]_\times J_\omega.
\]

foot collision center와 foot COM이 같은 경우에만 \(r=0\)으로 둔다. link actor pose의 baked offset을 COM offset으로 재사용하지 않는다. 실제 contact가 sphere의 아래 표면이면 collision center와 contactpoint도 구별한다. acceleration/contact constraint에는 \(\alpha\times r+\omega\times(\omega\times r)\) 또는 동등한 \(\dot Jv\)를 넣는다. `J @ generalized_velocity`를 `body_com_vel_w`와 대조하면 원점/frame/axis ordering을 검증할 수 있다.

base의 generalized linear velocity 역시 root COM 기준인데 state position을 actor origin으로 사용한다면 \(\dot p_{actor}=v_{COM}-\omega\times(p_{COM}-p_{actor})\)를 반영해야 한다. root COM position과 COM velocity로 통일하는 방식도 가능하다. `root_pos_w`와 `root_lin_vel_w`라는 이름만으로 같은 원점이라고 가정하지 않는다.

### B. Pinocchio로 독립적인 강체 모델

물리 inertia가 보존된 URDF 또는 실제 loaded USD에서 검증된 export로 모델을 만든다. **MJCF를 직접 읽어 Pinocchio model을 구성하는 경로도 가능하며 현재 모델 담당자는 이 경로를 선택했다.** joint/body/inertial/geom transform과 defaults를 해석하고, 명시 inertial과 geom에서 추론한 inertia가 동일 asset을 나타내는지 PhysX와 대조해야 한다. `JointModelFreeFlyer`를 넣은 모델의 `nq=19`, `nv=18`과 joint-name mapping을 확인한다. `crba`로 M, `rnea(q,v,0)`/`nonLinearEffects`로 h, Jacobian/derivatives로 local model을 구성할 수 있다. Pinocchio 버전에 따라 CRBA의 upper triangle만 채워지므로 문서/반환 matrix를 확인하고 lower triangle을 적절히 채운다. [공식 CRBA/RNEA 예제](https://docs.ros.org/en/noetic/api/pinocchio/html/mimic__dynamics_8py_source.html), [공식 energy 검증 소스](https://docs.ros.org/en/noetic/api/pinocchio/html/unittest_2energy_8cpp_source.html), [RNEA derivatives API](https://docs.ros.org/en/rolling/p/pinocchio/generated/function_namespacepinocchio_1a792a17cdc02c7ba97fe065a3d8310f20.html).

MJCF Euler 해석은 [MuJoCo compiler 공식 문서](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq)에 따른다. `angle=radian`, `eulerseq` 생략이면 default `xyz`다. MuJoCo 소문자는 intrinsic rotating axes, 대문자는 parent-fixed extrinsic axes다. default xyz의 child-to-parent rotation은 \(R_x(a)R_y(b)R_z(c)\)다. SciPy의 case 규칙은 반대이므로 이것은 `Rotation.from_euler('XYZ', [a,b,c], degrees=False)`에 해당한다. [SciPy from_euler 공식 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html). URDF rpy는 MuJoCo `XYZ`이며 \(R_z(c)R_y(b)R_x(a)\), SciPy `xyz`에 해당한다. 세 축 모두 nonzero인 각도로 compiled source rotation을 대조해야 단일축 테스트로 숨는 ordering 오류를 잡을 수 있다.

PhysX world-frame base twist와 Pinocchio의 frame convention이 다를 수 있다. 두 모델을 비교하려면 twist transformation과 joint permutation을 통일한다. 부호/프레임 문제가 해결되기 전 inertia fit으로 그 차이를 흡수하지 않는다. 9시간 안에 conversion 경로가 검증되지 않으면 PhysX 기반 모델을 먼저 검증하고 Pinocchio 미완료 사실을 남긴다.

## 7. 9시간 실행 구조 제안

아래 시간은 **예산 배분 제안**이며 실제 처리시간/통과 보장이 아니다. 기존 [v15_design.md](v15_design.md)에 계정 상한 GPU 2개·작업4개가 기록되어 있으므로 실제 queue/account 상태를 확인한 뒤 그 범위에서 병렬화한다.

| 순서/예산 | 실행 | 판정 및 산출물 |
|---|---|---|
| 0–1 h | asset/actuator/API audit, no-contact/fixed-base 물리 항 검증, 12단계 수식/수치 검증 | model manifest, units/frames/joint order, residual. 실패하면 원인 해결이 먼저다. |
| 1–2 h | bounded multi-joint excitation, physical residual fit, phase별 identification | command/applied torque·q/v/base/contact 로그, train/validation split. 동역학의 모든 inertia를 근거 없이 자유변수로 두지 않는다. |
| 2–4 h | PD/CT/MIMO LQR gain 또는 Q/R 탐색, fixed-base/standing 짧은 반복 | RMS/p95 error, coupling response, applied effort/slew/power, fall/contact/QP feasibility. reject 원인도 보존. |
| 4–7 h | analytic gait + contact-aware body/swing controller. 별도 frozen-reference 비교 가능 | reference 출처, contact phase, gait/candidate ID, randomization/calibration settings. |
| 7–9 h | 고정 candidate의 hold-out seed/초기조건/외란 평가, 보고서 | accepted/rejected 판정, 재현 명령, model/controller artifact, limitation. |

CPU에서는 small matrix exponential/DARE/QP gain/precomputed phase model을 구성하고 GPU에서는 많은 PhysX 환경의 bounded rollout을 병렬 평가한다. 18×18 solve와 state-feedback은 tensor batch로 실행 가능하지만, 매 environment×매5ms에서 Python loop·CPU transfer·symbolic TF 변환을 하지 않는다. runtime solver throughput은 작은 smoke로 측정한 뒤 environment 수를 정한다.

보정은 물리 예측 residual \(\dot v_{measured}-f(q,v,\tau,\sigma)\)와 추종 residual을 분리한다. torque scale/delay/viscous/Coulomb 항처럼 식별 가능한 작은 parameter set부터 시작한다. stance contact force, joint acceleration 노이즈, inertial parameter의 비식별성을 고려한다. gain 탐색은 controller calibration이고 policy training이 아니다. 최적화 seed를 평가 seed로 재사용해 최종 성능을 부풀리지 않는다.

최소 비교는 동일 plant/initialization/reference에서 PD vs fixed-base CT vs MIMO feedback이다. 자유기저 보행에서 physical-contact/QP 경로가 실패하면 12DOF 모델 검증 결과와 보행 실패를 각각 보고한다. 모델이 맞고 reference가 실현 불가능할 수도 있으므로 controller error와 gait planning error를 분리한다.

## 8. 검증 기록과 남은 한계

초기 문헌 연구 단계에서는 실제 로봇/PhysX rollout, HPC 제출, RL 훈련을 실행하지 않았다. 다음은 그 단계의 로컬 Python 3, NumPy 1.26.4, SciPy 1.11.4 **합성 선형 모델 대수 검증**이다. 이후 HPC CPU 수학 회귀는 아래 9절에 별도로 기록한다.

\(M=\begin{bmatrix}2&0.4\\0.4&1.5\end{bmatrix}\), \(K=\begin{bmatrix}4&-0.6\\-0.6&3\end{bmatrix}\), \(D=\begin{bmatrix}0.7&0.1\\0.1&0.5\end{bmatrix}\), \(T=0.005\), rng seed16으로 만들었다. \(A_c=[0,I;-M^{-1}K,-M^{-1}D]\), \(F_c=[0;M^{-1}]\)다.

- block `expm`과 `cont2discrete(method='zoh')`: 최대 절대 차 0.0.
- nonzero \(x_0\)와 held \(u\)에 대한 `solve_ivp` vs \(A_dx_0+F_du\): 최대 절대 차 \(1.11\times10^{-16}\).
- 두 input×두 position-output의 SISO recurrence vs matrix response: 최대 절대 차 \(1.12\times10^{-14}\).
- off-diagonal position-input coefficient \(F_{d,0,1}=-1.75924\times10^{-6}\): parsing 후에도 coupling이 남는 예시다.
- random SPD 18×18 M, full-row-rank 9×18 J의 contact KKT: EOM residual \(1.06\times10^{-14}\), constraint residual \(5.55\times10^{-15}\). command actuation의 base 6성분은 0으로 두었다.

이 결과는 초기조건·hold·backshift convention과 KKT 부호의 합성 검증이다. loaded robot asset의 M/h 정확성, 접촉 모델 오차, sampled nonlinear stability, gait 성공을 입증하지 않는다. phase마다 \(A_d-F_dK\)가 stable하다는 것만으로 switching/hybrid gait의 안정성이 보장되지도 않는다. 최종 acceptance는 실행 artifact와 hold-out rollout 증거로 정해야 한다.

## 9. 실제 SDK friction 단위 및 force QP 회귀 감사

### 9.1 Legacy 계수와 현재 effort 모델

HPC의 설치 파일을 읽어 `/opt/isaaclab/VERSION = 2.3.2`, Python package
`isaacsim-5.1.0.0.dist-info/METADATA`의 `Version: 5.1.0.0`을 확인했다.
설치된 `articulation.py`의 925–945행은 IsaacSim major<5에서 legacy coefficient
setter를 사용하고, major>=5에서는 `get/set_dof_friction_properties`를 사용한다.
후자의 세 열은 static effort, dynamic effort, viscous coefficient다.
[OmniPhysics 107.3 공식 API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.ArticulationView.get_dof_friction_properties).

| 경로 | friction의 의미 | 이 실험에 대한 해석 |
|---|---|---|
| PhysX legacy `setFrictionCoefficient` | 무차원, 전달되는 관절 spatial load에 비례하는 저항 용량 | PhysX 5.5 설명을 그대로 최신 writer에 적용할 수 없다. |
| IsaacSim 5.1 / IsaacLab 2.3.2 `friction` | static friction effort | SI revolute joint에서는 Nm 단위의 정지 마찰 상한이다. 실제 항상 가해지는 일정 토크는 아니다. |
| 동일 경로 `dynamic_friction` | dynamic friction effort | SI revolute joint에서는 움직일 때의 Coulomb 저항 토크 크기다. |
| 동일 경로 `viscous_friction` | 속도 비례 저항 계수 | SI revolute joint에서는 Nm/(rad/s), 저항은 `-coefficient * speed`다. |

구분 근거는 [PhysX 5.5 legacy API](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/_api_build/classPxArticulationJointReducedCoordinate.html),
[IsaacLab 2.3.2 ActuatorBaseCfg의 버전 주석](https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/actuators/actuator_base_cfg.html),
[PhysX 5.6.1 PxJointFrictionParams](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxJointFrictionParams.html)다.
따라서 현재 실제 writer에 넣는 `.3`은 static effort cap `.3 Nm`로 해석할
근거가 있으며, legacy의 무차원 `.3`이나 모든 상태에 `.3*sign(dq)`를 더하는
보상 토크로 해석해서는 안 된다. 원인 분석에는 설정 이름 외에 getter로
읽은 세 열을 canonical joint 순서로 기록해야 한다.

### 9.2 544 / 545 / 548은 static만 0으로 만든 실험

로컬 보존 manifest에서 세 작업 모두 `joint_friction=0`, 측정 static 값
12개가 0임을 확인했다. dynamic/viscous 측정 항목은 보존되어 있지 않다.
설치된 actuator constructor는 cfg 값이 `None`이면 USD 값을 유지한다.
현재 버전 writer는 static만 제공했을 때 dynamic/viscous를 바꾸지 않는다.
[공식 writer 구현](https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/assets/articulation/articulation.html#Articulation.write_joint_friction_coefficient_to_sim).

따라서 이 작업들은 **static effort 0, dynamic/viscous는 USD 값을 유지한
실험**이다. 세 마찰 항이 모두 0이라는 근거로 사용할 수 없다. 실제 남은
dynamic/viscous 값과 그 효과는 해당 작업의 getter 로그 없이 확정할 수 없다.
당시 manifest의 `dimensionless ... not a constant Nm torque` 문구는 현재
writer branch의 단위를 잘못 설명한다. 원본 artifact는 보존하고 이 감사
기록으로 해석을 정정한다. 부모는 후속 실험에서 세 값 모두 명시적으로 0으로
설정하고 세 getter 배열을 보존하기로 했다. 후속 실험 결과는 여기서 주장하지 않는다.

### 9.3 M/h와 실제 solver 사이의 남은 힘

PhysX의 Coriolis 및 force-from-acceleration utility는 joint friction,
link damping, joint drives를 포함하지 않는다.
[PhysX inverse dynamics 설명](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html#coriolis-and-centrifugal-compensation-force).
그러므로 `M_eff*a+h`를 실제 전체 힘이라고 부르면 안 된다. 진단에는
actuation, 측정 ground normal/friction, 실제 contact point, 다른 body의 접촉,
관절 limit/velocity 제약과 solver discretization을 함께 고려해야 한다.
현재 inherited `GUIDE_ROBOT_V2_CFG`의 link linear/angular damping은 둘 다 0이며,
V16 explicit actuator의 stiffness/damping도 0으로 구성되어 있다. 이 항들이
누락될 수 있다는 일반적인 사실만으로 현재 실패의 원인이라고 단정하지 않는다.

548의 `generalized_force_unmodeled_5ms`는 여러 누락 항과 시간/point 근사 오차를
함께 포함한다. 아직 측정되지 않은 dynamic/viscous friction의 실제 크기와
동일시하지 않는다. 또한 frozen control-tick M/h/J와 post-step contact force를
사용한 예측이므로 이산 시간 오차가 남는다.

`stance_point=measured`는 현재 patch centroid를 매 tick 다른 body-local point로
선택할 수 있다. 이때 이전 tick J와의 차분에는 point-selection drift도
포함된다. 동일한 material point의 강체 가속도와 patch centroid의 이동은
별개의 양이다. 따라서 543의 prep 실패는 current-patch position anchor/Jdot
제약까지 바뀐 실험으로 해석하며, 측정 contact point 자체의 정확도만 비교한
결과로 해석하지 않는다. body-local point를 latch한 ground mode에는 이
선택 drift가 없다. 이것은 코드의 좌표 관계에 근거한 추론이며 실패 원인
확정이나 실제 no-slip 보장은 아니다.

### 9.4 HPC NumPy/Torch CPU 회귀 547

부모의 승인을 받아 최신 QP/crawl/controller math와 전체 tests를
`/home/shanghun1404/v16_12dof_model/test_suite`에 복사하고 기존 CPU regression
launcher로 실행했다. `run_v16_physics.py`는 변경하거나 덮어쓰지 않았다.
547은 `CUDA_VISIBLE_DEVICES=''`, CPU 요청 1, memory 2G, walltime 10분,
GPU 요청 없음으로 제출했다. Slurm 기록은 `COMPLETED`, `ExitCode=0:0`,
elapsed 5초, `AllocCPUS=2`다. 실제 로그 결과는 **479 passed, 3 skipped,
1 warning, 11 subtests passed**다. 경고는 기존
`test_v12_training.py::test_compute_distillation_batch_supports_real_gradient_optimization`의
requires-grad tensor를 float로 바꾸는 항목이다. 최신 QP의 NumPy 수학 테스트와
Torch CPU parity/warm-state 테스트가 이 suite에 포함되어 통과했다.

- 원본 로그: HPC `/home/shanghun1404/v16_12dof_model/tests-547.out`.
- 원본 snapshot: HPC `/home/shanghun1404/v16_12dof_model/test_sources/547`.
- QP SHA256: `d3d2e5f8d0391544afff7da9510eb43a1c0d216de0797c90a39785097c155f2e`.
- QP tests SHA256: `e4316d515a68f97b20ed8cef739211a751c2c9ea849b6814f26905a732c7a6dd`.
- crawler SHA256: `af8779c993e1e377c9344dd59f6b7f1091e46b2c6cb386bb8cc07658860483e2`.
- controller math SHA256: `d61e41071599fb748c0a2dd422749b5aa4fa7a40188532e4458cfa5710d09e37`.

546은 최초 제출이며, CUDA 노출을 확실히 차단하려고 취소를 요청하기 전에
이미 5초 만에 `COMPLETED 0:0`으로 끝났다. 취소된 작업이라고 기록하지 않는다.
CPU 범위 검증의 근거는 CUDA를 비운 547을 사용한다. GPU parity 성능/통과는
이 기록에서 주장하지 않는다.

547의 runtime snapshot은 기존 `a7ff295d...`(535 source)이며 최신 QP/friction
runtime 통합 버전은 아니다. 최신 math와 tests의 회귀 근거이지 최신 runtime,
실제 QP allocator의 PhysX 효과나 보행 성공 근거가 아니다. 이 agent는 실제
PhysX rollout이나 RL training을 실행하지 않았다.

### 9.5 접촉 presence/load 분리와 최신 CPU 회귀 554

`config/v16_crawl_math.py::contact_support_masks`는 normal force [N]로 두
Boolean mask를 반환한다. 기본값은 `presence = finite & (Fz > 1)` 및
`loaded = finite & (Fz > 20)`이다. 3 N으로 착지한 발은 allocator에 포함할
수 있지만, 안정 하중을 요구하는 FSM readiness는 아직 통과하지 않는다.
각 임계값은 유한 양수이고 presence 임계값이 load 임계값보다 작아야 한다.
NaN/±inf/0/음수 force는 두 판정 모두 false다. 기존 `Fz > 20`의 엄격한
경계 조건을 유지했으며, 입력을 바꾸지 않고 NumPy/Torch backend와 device를
보존한다. 접촉 mask 자체가 접촉 물리나 보행 안정성을 보장하지는 않는다.

누락 helper 때문에 새 NumPy 테스트 2개가 실패하는 RED를 먼저 확인했고,
구현 후 로컬 crawl suite는 **23 passed, 3 skipped**, controller/crawl/QP
suite 합계는 **55 passed, 4 skipped**였다. 로컬 skip은 Torch 미설치에
따른 항목이다. 존재하지 않는 heldout 테스트 파일명을 사용한 최초 추가
명령과 Torch 필수 v15 테스트를 로컬에서 모은 명령은 collection 단계에서
실패했으며, 이 둘을 회귀 통과 근거로 포함하지 않는다.

부모가 업로드한 최신 runtime/math/launcher를 HPC `test_suite`에 복사하고
최신 전체 tests를 업로드한 뒤 CPU regression **554**를 제출했다.
`CUDA_VISIBLE_DEVICES=''`, CPU 요청 1, memory 2G, walltime 10분, GPU 요청
없음이다. Slurm은 **COMPLETED 0:0**, elapsed 5초, `AllocCPUS=2`를 기록했다.
전체 결과는 **482 passed, 3 skipped, 1 warning, 18 subtests passed**다.
새 helper의 Torch CPU parity, dtype/device 및 입력 비변형 검증도 포함되어
통과했다. 경고는 547과 같은 기존 v12 requires-grad scalar 변환이다.

- 로그: HPC `/home/shanghun1404/v16_12dof_model/tests-554.out`.
- snapshot: HPC `/home/shanghun1404/v16_12dof_model/test_sources/554`.
- runtime SHA256: `1b5fcd7c0eb632d5fbc754a8536c069ab8c17beec920c581dbedb02e065c4d34`.
- controller math SHA256: `dce4085f0de3a6c72a5eb346c0c88fdc9b036a5f7ab5e2fc0ae329fd034aabaf`.
- crawler SHA256: `0f8cfd5574ecd129bafdaf7df8bb150878bf24434c38c04f75c0477e5725fa09`.
- crawler tests SHA256: `f0e1274f0426bda86dc0133d434a666f38fa160277f834a6f729d0df4273c4da`.
- QP SHA256는 547과 동일하다.

완료 후 snapshot의 전체 test 파일 및 runtime/math/launcher, 총 62개 파일의
SHA256이 로컬 현재 파일과 일치함을 확인했다. 이 검증은 최신 수학과 소스
계약에 대한 CPU 회귀이며, GPU parity, solver의 실제 접촉 하중 확보 또는
반복 보행 성공의 근거로 확장하지 않는다.


## 10. COM 목표 연속화와 whole-body QP 비교

557에서는 COM 목표를 따라가는 비중을1000으로 높여3–5걸음을 진행했으나 모두 넘어졌다.
다음 지지삼각형으로 전환할 때 목표가 약12cm 바뀌었고, 몸통 요청 가속도가 불연속이었다.
최소 jerk reference를 도입하고 endpoint까지 완료된 뒤 다음segment를 시작하도록했다.
561은16초8/8생존·5–9steps이나 실제 진행속도는명령.05m/s보다느렸다.
reference의.30m/s·1.2m/s²한계는 실제root/wholeCOM의운동을제한하는 보장이아니다.

Force-only QP는 투영된 가속도를고정하므로 그가속도가마찰·토크 제약에충돌하면
힘만조정해서해결할수없다. 새WBC는 a18과lambda12를공동변수로두고 다음을계산한다.

- base: M_b a + h_b = J_b^T lambda
- active contact: J_c a = anchor_feedback - Jdot*v
- unilateral/friction pyramid/velocity/slew torque bounds
- tau = M_j a + h_j - J_j^T lambda (독립변수에서제거)

몸통/관절 가속도 추종과 normalized torque·force regularization은softcost다.
body/contact와force/torque를함께조정하는근거는
[Kim et al. WBIC 원논문](https://arxiv.org/html/1909.06586#S4.SS2)의relaxation구조에서참고했다.
현재구현은논문의MPC/WBIC 전체재현이아니며 preview MPC나실시간500Hz실행을주장하지않는다.
ADMM의primal/dual잔차와실제반환값제약을별도로기록한다
([OSQP solver 정의](https://osqp.org/docs/solver/index.html)).

564의CPU16tests/Torch검증과566의4/4기립을통과했다.
570 동일파라미터보행에서force-only QP8/8, WBC0/8이었다.
WBC의첫SWING끝발위치가예정착지보다높고멀었으며, 초기제약위반보다앞서나타났다.
이는가중치/목표추종수렴/정지마찰을분리해조사할근거다. 유일원인으로확정하지않는다.

WBC변수스케일은linear3·angular10·joint50이다.
normalized metric에base1000/joint1을직접넣으면unscaled linear/joint비중은277777.8이된다.
비교옵션unscaled는weights=W*s²/2500을사용해가속도오차의원래base1000/joint1비율을유지한다.
force/torque regularizer는같으므로전체objective가동등한실험은아니다.
575는이옵션과full18desired/returned가속도 및primal/dual/converged trace를비교한다.
주훈련571은기존에검증한force-only QP의immutable소스로계속실행한다.

## 11. 반복 CEM과 독립 평가

555에서는단일최고점수를오래보존하고, qualified walking후보가부족해도stop elite로분포를좁혔다.
확대훈련은128개고유후보를4환경씩반복해총512환경에서시험한다.
모든4회가최종alive·명령거리25%이상·완료steps4이상일때만qualified다.
현재세대의observed minimum으로incumbent와challenger를비교한다.
elite는qualified후보로만구성하고std floor5%range를유지한다.
qualified가2개미만이면mean을유지하고std를넓힌다.
4회의관측최솟값은다른교란/지형에서의통계적성공보장이아니다.
CEM의quantile/noise선택위험은
[Goschin et al. ICML2013 원논문](https://proceedings.mlr.press/v28/goschin13.pdf)를참고했다.

568의12세대뒤571에서같은분포·RNG·소스로200세대를추가한다.
훈련seed는101–312,holdout1001–1003과분리한다.
최종모델을고정한뒤12초32환경×3seed와같은seed의32초평가를실행한다.
두길이는paired horizon평가이며서로독립적인192회로합산하지않는다.
source SHA·trainingmanifest SHA·모터/마찰/COM/QP조건·실제평가길이를대조한다.
모델보행기준통과와finite-iteration QP제약/guard활동·실기열정격을구분해기록한다.


575의unscaled 비교는16초8/8생존·각9steps·.719–.722m/평균속도약.0495m/s였다.
800control ticks 전체에서numerical feasible,maxconstraint≤2.28e−5,guard0을기록했다.
그러나converged비율은75.5–76.375%,maxdualresidual.01596으로모든tick의최적해를주장하지않는다.
이는570과동일파라미터/seed의가중치비교이며독립최종평가가아니다.
576에서512환경·128후보×4반복으로8세대를시험하고,577에서같은소스/분포/RNG로120세대를추가한다.
WBC훈련seed401–528도1001–1003평가seed와분리한다.578/579는12/32초,580은RGB영상이다.
두제어기를같은seed에서비교하므로서로독립적인샘플수로합산하지않는다.
