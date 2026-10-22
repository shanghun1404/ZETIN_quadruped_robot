# V16 원본 물리모델에서 디지털 차분식까지: 12단계

## 검증 범위와 산출물

기준은 workspace `GUIDELINE.md`의 12단계다. 원본은 HPC
`/home/projects/legacy/robot_v2/quadruped/quadruped.xml`이며 SHA-256은
`8ef5712d8604110d1efc7ed92845bd83107e0ef42e1fa0f379448387c4793c6f`이다.
최종 모델 계산과 독립 컴파일러 검증은 CPU Slurm **520**에서 수행했다.
원본 MJCF를 Pinocchio로 직접 구성한 결과는 MuJoCo 3.14.0과 비교했다.
이는 변환된 USD/PhysX plant와의 일치 검증을 대신하지 않는다.

| 산출물 | 내용 |
|---|---|
| `config/v16_physical_model.py` | source-specific MJCF parser, fixed/free dynamics, 명시적 관절 매핑 |
| `config/derive_v16_discrete.py` | nominal 선형화, exact ZOH, 144 SISO 채널, 차분식·초기조건 검증 |
| `config/tests/test_v16_model.py` | 독립 fixture와 실제 source의 13개 테스트 |
| `reports/v16/physical_model/v16_model_manifest.json` | 원본·코드 hash, 54개 body의 질량/COM/관성, 검증 수치 |
| `reports/v16/physical_model/v16_model_tensors.npz` | 실제 계산된 M/g/Kg/D/A/B/C/Ad/Bd/초기조건 trajectory |
| `reports/v16/physical_model/v16_model_siso_channels.json` | 분자/분모, 모달 pole/residue, 채널별 상태 인덱스 |
| `reports/v16/physical_model/v16_model_difference_equations.txt` | 모든 채널의 계수와 직접 차분식 |
| `reports/v16/physical_model/v16_model_mujoco_oracle.json` | 독립 source compiler 비교, raw/corrected bias 차이 |

관절 배열 순서는 항상
`FL_HAA, FL_HFE, FL_KFE, FR_HAA, FR_HFE, FR_KFE, RL_HAA, RL_HFE, RL_KFE, RR_HAA, RR_HFE, RR_KFE`다.
원본 traversal은 FL, FR, **RR, RL**이므로 단순 배열 복사는 금지한다.
free-base 일반화 속도는 `[base linear(3), base angular(3), canonical joints(12)]`이며
Pinocchio base 속도/가속도는 body-local 좌표다.
`velocity(dq12,base_twist=...)`는 공개 dynamics API에 바로 넣을 수 있는 canonical 순서를 반환한다.
`configuration`은 내부 Pinocchio q 순서를 반환하므로 두 종류의 배열을 혼동하지 않는다.

## 1. 물리 요소별 모델

각 body의 실제 질량 m, body 좌표 COM c, COM 기준 대칭 관성 I를 읽는다.
`fullinertia`의 순서는 `(Ixx,Iyy,Izz,Ixy,Ixz,Iyz)`이다. 강체 질량행렬은
각 body의 kinetic energy를 합산해 정의한다.

\[
\mathcal T=\frac12\sum_b\left(m_b\|v_{c_b}\|^2+
\omega_b^T R_b I_b R_b^T\omega_b\right)=\frac12v^TM_{rigid}(Q)v.
\]

54개 body를 모두 한 번씩 추가한다. 관절이 없는 cosmetic/fixed body도 질량이 있으므로
가장 가까운 movable joint에 SE(3)로 옮겨 관성을 집계한다. body frame은 별도로 보존한다.
임의 hinge 축 a와 body-local anchor p에는
`parent_joint_T_body * Translate(p)`로 joint를 배치하고
`joint_T_body = Translate(-p)`를 사용한다. 이 처리로 body offset과 관절 기준점이 모두 유지된다.

원본 `compiler angle="radian"`과 기본 `eulerseq="xyz"`를 사용한다.
MJCF 소문자는 intrinsic rotation이므로
`R_body = Rx(a) Ry(b) Rz(c)`이다. URDF의 rpy와 구분한다.
[MuJoCo Euler 정의](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq),
[MJCF inertial/joint 정의](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial).

원본 default joint damping은 **1.5 N·m·s/rad**, armature는 **0.02 kg·m²**다.
`M = M_rigid + diag(armature)`로 rotor inertia를 명시적으로 더한다.
Pinocchio 내부 armature는 0으로 두어 API별 자동 포함 차이를 피한다.
HAA peak metadata는 24.8 N·m, HFE/KFE는 48 N·m이며 source motor ctrlrange에서 읽는다.
이를 실기 연속 정격으로 해석하지 않는다.

총 질량은 **45.451322469636764 kg**이다. fixed-base `nq=nv=12`;
free-base `nq=19, nv=18`이다. base pose의 quaternion 때문에 nq와 nv가 다르다.
몸통에 가상 actuator는 없다.

## 2. ODE와 입력/출력

floating-base rigid dynamics의 구조는

\[
M(Q)\dot v+h_{rigid}(Q,v)+D_vv=S^T\tau+J_c(Q)^T\lambda,
\qquad S=[0_{12\times6}\ I_{12}].
\]

stance contact가 유지되면 추가로
`J_c dot(v) + dot(J_c) v = 0`이 필요하다. 현재 parser는 contact solver를 만들지 않는다.
`point_jacobian`은 body-local point가 주어졌을 때의 world-aligned translational Jacobian을 제공한다.
발 body origin이나 테스트 point를 실제 mesh contact patch로 가정하지 않는다.

12단계 수치 유도에는 **고정 base, 무접촉 nominal stand**를 사용한다.
`q0 = [0,-0.55,1.10]`를 각 다리에 적용하고 `dq0=0`, `tau0=g(q0)`로 둔다.

\[
M_j(q)\ddot q+C_j(q,\dot q)\dot q+g_j(q)+D\dot q=\tau.
\]

`x=[delta q; delta dq]`, `u=tau-tau0`, 출력 `y=delta q=Cx`로 정의한다.

\[
\dot x=Ax+Bu+d,\quad
A=\begin{bmatrix}0&I\\-M_0^{-1}K_g&-M_0^{-1}D\end{bmatrix},\quad
B=\begin{bmatrix}0\\M_0^{-1}\end{bmatrix},\quad C=[I\ 0].
\]

`M0`는 source armature 포함 질량행렬, `Kg=partial g/partial q|q0`이며
`D=1.5 I`는 source 값이다. nominal equilibrium에서는 `d=0`이다.
중력 미분은 중앙 차분과 Pinocchio RNEA analytic derivative를 비교했고 최대 차이는
**7.96e-10 N·m/rad**다. source gravity equilibrium effort의 앞 세 값은
`[2.9226552033897706,-2.9401706257469535,0.14175731892978527] N·m`이다.
나머지 관절도 JSON/NPZ에 실제 계산 값을 보존한다.

## 3. 초기조건을 포함한 Laplace 변환

한 샘플 구간의 시작 상태를 `x_k=x(kT)`라 하고, 해당 구간의 시간 원점을 0으로 옮긴다.

\[
sX(s)-x_k=AX(s)+BU(s)+d/s,
\quad (sI-A)X(s)=x_k+BU(s)+d/s.
\]

`-x_k`를 생략하면 nonzero initial state 응답을 잃는다. 본 유도와 실행 코드에서는 유지한다.

## 4. 결합 상태별 해

resolvent `R(s)=(sI-A)^-1`를 사용하면

\[
X_i(s)=\sum_{\ell=1}^{24}R_{i\ell}(s)x_{k,\ell}
+\sum_{j=1}^{12}[R(s)B]_{ij}U_j(s)+[R(s)d]_i/s.
\]

각 상태의 해에는 다른 상태들의 초기조건과 모든 토크 입력이 남는다.
`M0` 또는 `Kg`의 off-diagonal 항을 버려서 개별 관절 double integrator로 바꾸지 않는다.

## 5. 래치된 입력 대입

컴퓨터가 샘플 사이에 토크를 유지하므로
`u(kT+t)=u[k]` for `0 <= t < T`이고 `U(s)=u[k]/s`다.

\[
X(s)=R(s)x_k+R(s)B\,u[k]/s+R(s)d/s.
\]

샘플마다 갱신하는 delta torque를 사용하며 실제 torque는 `tau0+u[k]`이다.

## 6. 연속 구간의 step response

\[
x(kT+t)=e^{At}x_k+\left(\int_0^t e^{A\eta}B\,d\eta\right)u[k]
+\int_0^t e^{A\eta}d\,d\eta.
\]

`A^-1`을 사용하지 않아 singular A인 double integrator에도 적용할 수 있다.
`exact_zoh`는 augmented exponential을 계산한다.

\[
\exp\left(T\begin{bmatrix}A&B&d\\0&0&0\\0&0&0\end{bmatrix}\right)
=\begin{bmatrix}A_d&B_d&d_d\\0&I&0\\0&0&1\end{bmatrix}.
\]

## 7. 다음 샘플 시점의 결합 응답

위 식에 `t=T`를 대입한다.

\[
x_{k+1}=A_dx_k+B_du[k]+d_d.
\]

`T=0.005 s`와 `T=0.020 s` 두 주기를 실제 계산했다.
nonzero initial state와 held input의 DOP853 ODE endpoint는 exact ZOH 결과와 각각
**2.12e-14**, **2.05e-14** 최대 차이를 보였다.

## 8. sampled-time 결합 차분식

24개 상태의 모든 행에 대해

\[
x_i[k+1]=\sum_{\ell=1}^{24}(A_d)_{i\ell}x_\ell[k]
+\sum_{j=1}^{12}(B_d)_{ij}u_j[k]+(d_d)_i.
\]

임의 초기조건에서의 전체 응답은

\[
x[k]=A_d^kx[0]+\sum_{r=0}^{k-1}A_d^{k-1-r}(B_du[r]+d_d).
\]

`sampled_response`와 NPZ `T*_initial_condition_trajectory`가 초기조건 항을 보존한다.
초기조건 응답과 zero-initial input 응답의 중첩 오차는 두 주기 모두 **1.2e-16 이하**였다.

## 9. B-operator 결합 모델

backshift는 `B x[k]=x[k-1]`이다. `k>=1`에서는

\[
(I-A_d B)x[k]=B_d B u[k]+d_d.
\]

unilateral sequence로 `x[k<0]=0`을 정의하면 초기조건을 명시한 정확한 식은

\[
(I-A_d B)x[k]=B_d B u[k]+d_d H[k-1]+x[0]\delta[k].
\]

따라서 초기조건은 전달함수 바깥에서 사라지는 값이 아니라 별도 source sequence다.

## 10. 결합 전달함수

zero-initial input transfer matrix는

\[
G(B)=C(I-A_dB)^{-1}B_dB.
\]

전체 출력에는

\[
Y(B)=G(B)U(B)+C(I-A_dB)^{-1}x[0]
+C(I-A_dB)^{-1}d_d\frac{B}{1-B}
\]

가 남는다. `G`만으로 nonzero-initial response나 contact wrench를 나타낼 수 없다.

## 11. 모든 SISO 채널과 안정적인 계산

각 토크 입력 j와 위치 출력 i에 대해

\[
G_{ij}(B)=\frac{b_{ij,0}+b_{ij,1}B+\cdots+b_{ij,n}B^n}
{1+a_{ij,1}B+\cdots+a_{ij,n}B^n},\qquad b_{ij,0}=0.
\]

이는 144개 input/output channel의 분해다. 실제 관절들이 독립적으로 동작한다는 증거가 아니다.
출력은 `y_i=sum_j G_ij u_j + initial response + affine response`로 중첩한다.
fixed base에서는 서로 다른 다리 사이의 구조적 채널이 정확히 0이다.
각 다리 안의 3관절 결합은 유지하며 free base/contact에서는 다리 사이 coupling도 생긴다.

24차 다항식을 일괄 구성하면 불필요한 상쇄를 키운다. 코드에서는 input에서 도달 가능하고
output에 영향을 주는 구조적 상태만 남겨, 이 모델에서 최대 **6차** 분모를 사용한다.
분자 계수는 Markov sequence와 characteristic polynomial의 convolution으로 계산한다.
pole/residue도 함께 내보내

\[
G_{ij}(B)=\sum_r\frac{\rho_{ij,r}B}{1-p_rB}
\]

로 주파수 응답을 평가한다. eigenvector condition number가 1e8 이상이면
축소 state-space evaluation으로 돌아간다. 실제 제어 실행에는 `Ad,Bd`를 권장한다.

| 검증 | 5 ms | 20 ms |
|---|---:|---:|
| modal SISO vs state-space 최대 절대 주파수 오차 | 3.20e-14 | 1.18e-14 |
| modal SISO vs state-space 최대 상대 오차 | 1.75e-12 | 6.93e-14 |
| 직접 다항식 최대 상대 오차 | 9.75e-7 | 7.69e-10 |
| scalar 차분식 vs state-space, 100 samples | 8.91e-10 | 1.70e-11 |

주파수 검증은 digital frequency `[0.02,pi-0.02] rad/sample`의 90점이다.
`B≈1`의 다항식 상쇄를 전 주파수에서 해결했다고 주장하지 않는다.

## 12. 최종 독립 차분식과 코드 배열

각 채널의 zero-initial output을 `y_ij`로 두면

\[
y_{ij}[k]=-\sum_{r=1}^{n}a_{ij,r}y_{ij}[k-r]
+\sum_{r=1}^{n}b_{ij,r}u_j[k-r].
\]

한 샘플 input delay가 있다. 전체 출력은 채널들을 합산하고 초기조건/affine response를 더한다.
`difference_response`가 이 recurrence를 직접 실행한다. 초기조건을 임의의 zero history로 바꾸지 않는다.
nonzero initial state는 위 Step 8의 state propagation으로 계산하며, 전달함수 recurrence만 사용할
경우에는 `C Ad^r x[0]`에 대응하는 각 채널의 초기 history를 별도 유도해야 한다.

실제 계산된 5 ms `FL_HFE <- FL_HFE` 채널의 B 계수 예시는 다음과 같다.

```python
a = [1.0, -5.692817817835767, 13.484237390576224,
     -17.00809071621117, 12.047037967967434,
     -4.542664155081638, 0.7122973306617153]
b = [0.0, 5.979854672367232e-05, -0.00016445036890971797,
     9.181235522915982e-05, 0.00011443229874770087,
     -0.00014547393579900656, 4.388111916138628e-05]
# y_hfe_hfe[k] = -sum(a[r]*y_hfe_hfe[k-r], r=1..6)
#                  +sum(b[r]*u_hfe[k-r], r=1..6)
```

같은 출력의 `FL_KFE` 입력 채널도 0이 아니다.
첫 numerator 계수 `b1=-2.54917409056729e-05`이므로 diagonal 채널 하나만 계산하면
HFE/KFE의 동역학 결합을 잃는다. 모든 144개 채널과 두 주기의 실제 계수는 JSON/TXT에 있다.

## 검증 수치와 좌표계 주의

Slurm 520에서 13개 테스트가 통과했다. fixed/free mass 최소 eigenvalue는 각각
**0.0283143**, **0.0281797**이며 양수다. raw RNEA/ABA roundtrip 최대 오차는
**2.72e-14**, source armature를 더한 mass solve roundtrip은 **9.83e-15**다.
발 body-local point Jacobian과 manifold finite difference의 최대 차이는 **1.01e-10**이다.

별도 MuJoCo compiler oracle은 nominal과 네 개 perturbation으로 비교했다.
54개 body pose 최대 차이는 **7.78e-16**, 18x18 mass는 **1.43e-14**,
joint gravity는 **1.16e-14**, point Jacobian은 **7.22e-16**이었다.

MuJoCo의 free-joint linear velocity는 world frame, angular velocity는 body-local이다.
Pinocchio는 둘 다 body-local이다.
[MuJoCo free-joint 좌표](https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst),
[Pinocchio free-flyer 좌표](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/structpinocchio_1_1JointModelFreeFlyerTpl.html).
base orientation이 identity여도 가속도 좌표의 moving-frame 항은 남는다.

\[
a_{MJ,linear}=a_{Pin,linear}+\omega\times v_{linear},\quad
h_{MJ}=h_{Pin}-M[\omega\times v_{linear};0].
\]

이 항을 적용한 bias 최대 차이는 **5.69e-14**다. 보정 전의 **1.8016223709071821** 차이도
oracle JSON에 보존했다. 두 API의 bias를 이름만 보고 직접 비교하면 잘못된 판정을 낼 수 있다.
본 oracle의 perturbation은 base orientation identity로 제한된다.

## 선형화 사용 범위와 한계

이 결과는 source damping/armature를 포함한 무접촉 fixed-base local model이다.
평형 주변 30개 random sample, torque perturbation `|delta tau|<=0.05 N·m`에서
비선형과 선형 acceleration derivative의 최대 차이는 다음과 같다.

| 최대 위치 변화(rad)·속도 변화(rad/s) | 최대 derivative 차이 (rad/s²) |
|---|---:|
| 0.001 | 0.000263 |
| 0.010 | 0.00323 |
| 0.050 | 0.0638 |

이는 유한 표본의 오차 기록이며 보장된 attraction region이나 보행 안정성 증명이 아니다.
큰 움직임, floating base, 접촉 전환, 토크 포화, 마찰, solver timestep/implicit drive의 영향은
이 하나의 전달함수에 포함되지 않는다. 원본 MJCF와 변환 USD/PhysX의 collision 및 inertia 설정은
별도 runtime probe와 동일 q/order/frame/armature 조건으로 비교해야 한다.

재현 명령은 `config/experiments/v16_20260929/v16_model_verify.sbatch`에 있다.
최초 Red 507, API/fixture 실패 509·511·512, 초기 Green 513,
raw bias 좌표 비교 실패 514, compiler Green 515, velocity 순서 regression Red 518,
API Green 519, 차분식 출력까지 포함한 최종 Green 520 로그를 모두 보존한다.
모델·시뮬레이션 계산은 HPC에서만 수행했고 자동 commit/push는 하지 않았다.
