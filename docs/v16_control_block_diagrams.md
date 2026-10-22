# V16 실제 전신 제어 루프와 물리 블록선도

[실제 제어부터 시작하는 3쪽 PDF](../reports/v16/figures/v16_control_block_diagrams.pdf) ·
[실제 WBC 루프](../reports/v16/figures/v16_actual_wbc_blocks.png) ·
[자유 몸통 물리 요소 블록](../reports/v16/figures/v16_physical_force_balance.png) ·
[고정 몸통 등가 피드백](../reports/v16/figures/v16_physical_state_feedback.png) ·
[전달함수·수식 참고 3쪽](../reports/v16/figures/v16_control_transfer_diagrams.pdf) ·
[연속시간 그림](../reports/v16/figures/v16_control_s_domain.png) ·
[이산시간 그림](../reports/v16/figures/v16_control_z_domain.png) ·
[실제 비선형 제어식](../reports/v16/figures/v16_control_nonlinear.png)

주 PDF의 **1쪽이 최종 WBC Job 577의 실제 코드 경로**, 2쪽은 그 물리 plant의
힘·관성 블록, 3쪽은 별도의 **고정 몸통 선형 등가 해석**이다.
연속/이산 전달함수와 자세한 비선형 수식은 참고 PDF에 있다.
3쪽의 가상 감쇠·강성 \(D_c,K_c\)는 런타임 코드에 직접 존재하는 제어 블록이 아니다.

## 0. 로렌츠 교수의 물리 기반 표기를 V16에 적용한 범위

[UW–Madison의 Robert D. Lorenz 소개](https://engineering.wisc.edu/news/robert-lorenz-pioneer-controls-engineering-passes-away/)는
그의 연구를 physics-based controls로 설명한다. 로렌츠의
[1986년 상태변수 제어기 논문](https://www.scribd.com/document/324738025/Synthesis-of-State-Variable-Controllers-for-Industrial-Servo-Drives-Lorenz-86-08)은
개별 물리 상태방정식에서 시작해 관성·감쇠·강성에 해당하는 피드백을 나란히 그리고,
제어 게인의 물리 단위를 확인하는 방법을 제시한다. 2–3쪽은 이 **설명 원칙을
V16 사족 로봇에 맞춰 새로 그린 것**이다. 로렌츠의 원도면이나 정식 사족 로봇 표기법이라는 뜻은 아니다.

### 1쪽: 실제 V16 전신 QP 제어

[런타임 코드](../config/run_v16_physics.py)에서는 매 20 ms에
관절 위치·속도·자세 오차로 목표 가속도 \(a_d\in\mathbb R^{18}\)를 만든다.
\(q_r\)는 보행 상태와 측정 자세를 사용하는 IK에서 만들어지고,
측정 상태 버스는 세 오차와 몸통 목표·동역학 모델에 다시 들어간다.
발 접촉 상태와 동일한 \(M,h,J_c\)를 사용해
[WBC QP](../config/v16_wbc_qp.py)가 \(a^*,\hat\lambda^*\)를 함께 조정한다.
그 결과를 작은 연산 블록 \(Ma^*\), \(h\), \(J_c^T\hat\lambda^*\)로 나눠 합산하고,

\[
\tau_k^*=S\big(Ma^*+h-J_c^T\hat\lambda^*\big),\qquad
S=[0_{12\times6}\ I_{12}]
\]

로 **12개 관절 토크만** 꺼낸다. QP 내부에는 몸통 동역학, 접촉점 가속도,
마찰 피라미드, 모터 토크 경계가 있다. QP는 320회 유한 ADMM이므로
모든 tick의 수렴·제약 충족을 보장하는 표기가 아니다.
그림에서 \(h(Q,v)\)로 들어가는 점선은 QP 출력이 아니라 동일한 동역학 모델의
입력이다. \(M\)과 \(J_c\)도 QP와 토크 재구성에서 공유한다.
이후 20 ms slew 제한 \(\Gamma_T\), 토크 ZOH, 각 5 ms 물리 step의 속도·effort
보호 \(\Gamma_{5\rm ms}\)를 순서대로 거쳐 PhysX에 들어간다.
실제 지면 접촉력 \(\lambda_{\rm phys}\)은 QP가 예측한
\(\hat\lambda^*\)와 별도다. 측정 \(Q,v\)와 접촉이 다음 tick의 기준값·모델·QP로 돌아간다.

### 2쪽: 자유 몸통 동역학을 작은 물리 블록으로 분해

각 화살표는 작은 연산 하나를 통과한다. 12개 모터 토크는 \(S^T\),
발의 실제 접촉력은 \(J_{\rm phys}^T\)를 거쳐 일반화 힘이 된다.
이 쪽은 물리 plant 내부를 펼친 그림이라 \(Q,v\)의 외부 포트는 현재 물리 상태를
뜻하며, 전체 상태 귀환은 1쪽에 표시했다.
중력 \(g(Q)\), 코리올리·원심 관성항 \(c(Q,v)\)은 음의 입력,
모델에 없는 마찰 등의 힘 \(d\)는 별도 입력이다. 이들을 합한 뒤
\(M_{\rm eff}^{-1}\)가 일반화 가속도를 내고, 적분기 두 개와
\(N(Q)\)가 속도·자세를 만든다. \(Q\)의 19개 좌표에는 몸통 quaternion이 포함된다.

몸 전체로 보면 지면 접촉력과 중력만 외력이다. 관절 모터 토크는 몸과 다리 사이의
**내부 작용·반작용**이라 질량중심의 외력 합에 직접 더하지 않는다.

\[
m\ddot{\mathbf c}=m\mathbf g_w+\sum_i\lambda_i,\qquad
\dot{\mathbf H}_c=\sum_i(\mathbf p_i-\mathbf c)\times\lambda_i.
\]

이 모멘트 식은 각 발 접촉을 힘 하나와 작용점으로 줄인 식이다.
실제 PhysX의 분포 접촉·잔여 토크는 이 근사에 남을 수 있다.
반면 일반화 좌표에는 관절 모터 토크가 \(S^T\tau\)로 들어간다.

\[
M_{\rm eff}(Q)\dot v+h(Q,v)=S^T\tau_{\rm applied}
+J_{\rm phys}(Q)^T\lambda_{\rm phys}+d,
\quad S=[0_{12\times6}\ I_{12}].
\]

\(M_{\rm eff}=M_{\rm rigid}+S^T(0.02I_{12})S\)로 source armature를 포함한다.
\(h\)에는 관성 bias와 중력이 들어가고, \(d\)는 모델에 없는 마찰 등이다.
위 식의 앞 6행은 **구동하지 않는 몸통의 힘·모멘트 평형**, 뒤 12행은
**모터 토크가 들어가는 관절 평형**이다. \(J^T\lambda\)는 발의 뉴턴 단위 힘을
몸통 wrench와 각 관절의 N·m 토크로 바꾼다. 질량·COM·관성의 출처와
원본 검증은 [12단계 유도](v16_12_steps.md)에 있다.

### 3쪽: 고정 몸통 등가 해석의 물리 계수

원본 **고정 몸통·무접촉·정지 평형**의 국소 모델은

\[
M_0\ddot y+D_0\dot y+K_gy=u,\qquad
K_g=\left.\frac{\partial g}{\partial q}\right|_{q_0},\quad
y=q-q_0,\quad u=\tau-g(q_0).
\]

이 참고 그림에서 \(D_0\)와 \(K_g\)는 **원본 plant의 물리적 피드백 블록**이고,
\(D_c\)와 \(K_c\)는 **제어기가 더하는 별도 블록**이다. 네 결과를 합쳐
기준 토크에서 빼므로 물리 감쇠와 제어 감쇠의 기여가 분리되어 보인다.
\(M_0^{-1}\) 뒤 두 적분기가 각속도·관절각 상태를 만든다.
\(K_g\)는 중력 토크의 미분으로, 재료 스프링이라는 뜻이 아니며 양수일 필요도 없다.
\(M_0\)와 \(K_g\)의 비대각항은 관절 사이 결합을 남긴다.

이 모델에서 이상적인 역동역학 보상이 정확하고 제한기가 없다면,
\(r=q_r-q_0\), \(v_r\)를 기준 관절속도로 두어

\[
u_{\rm cmd}=D_0\dot y+K_gy+M_0
\big[K_pr+K_dv_r-K_d\dot y-(1+\alpha)K_py\big]
\]

라고 쓸 수 있다. 그림에서는 동등하게 \(u_{\rm ref}=M_0(K_pr+K_dv_r)\)로 두고

\[
u_{\rm cmd}=u_{\rm ref}-D_c\dot y-K_cy,\qquad
D_c=M_0K_d-D_0,\quad K_c=M_0(1+\alpha)K_p-K_g
\]

로 표현했다. plant의 물리 계수와 제어 계수를 합하면

\[
M_0\ddot y+\underbrace{M_0K_d}_{D_{\rm virtual}}\dot y+
\underbrace{M_0(1+\alpha)K_p}_{K_{\rm virtual}}y
=M_0(K_pr+K_dv_r).
\]

따라서 \(K_p\,[\mathrm{s}^{-2}]\)와 \(K_d\,[\mathrm{s}^{-1}]\)는
관절 가속도 게인이지만, \(M_0\)를 거친 폐루프의 등가 감쇠와 강성은 각각
\(D_{\rm virtual}\,[\mathrm{N\,m\,s/rad}]\),
\(K_{\rm virtual}\,[\mathrm{N\,m/rad}]\)로 읽을 수 있다.
이는 **이상적인 고정 몸통 모델의 해석**이다. 실제 V16의 자유 몸통·접촉·QP·토크 제한까지
스프링/댐퍼 하나로 단순화하지 않는다. 원본 \(D_0=1.5I\)이고 실제 USD damping은 0이다.

예를 들어 저장된 \(M_0\)의 FL_HFE 대각 성분은 \(0.21420959\,\mathrm{kg\,m^2}\)다.
WBC 훈련 Job 577의 게인을 대입하면 이 축의 등가 감쇠는
\(5.82074\,\mathrm{N\,m\,s/rad}\), 등가 강성은
\(102.10841\,\mathrm{N\,m/rad}\)다. FL_HFE→FL_KFE 비대각 관성
\(M_{0,1,2}=0.01349329\,\mathrm{kg\,m^2}\)도 남아, 해당 등가 강성
성분은 \(6.43192\,\mathrm{N\,m/rad}\)다.

## 1. 관절 제어식과 연속시간 블록선도

최종 훈련의 frozen runtime은 관절 가속도 명령을 다음과 같이 계산한다.

\[
a_j[k]=K_p(q_r[k]-q[k])+K_d(v_r[k]-\dot q[k])
       +\alpha K_p(q_0-q[k]).
\]

`q0`는 고정 nominal 자세, `alpha`는 학습한 posture gain이다.
관절 `Kp`의 단위는 \(\mathrm{s}^{-2}\), `Kd`는 \(\mathrm{s}^{-1}\)이며 **가속도 피드백 게인**이다.
토크 PD의 \(\mathrm{N\,m/rad}\) 게인으로 바꾸어 읽지 않는다.

\(y=q-q_0\), \(r=q_r-q_0\)로 놓고, 이상적인 역동역학 상쇄 뒤
\(\ddot y=a_j\), 비포화, 외부 reference를 가정하면 연속시간 등가는

\[
A(s)=\underbrace{(K_p+K_ds)}_{F_r(s)}R(s)
     -\underbrace{((1+\alpha)K_p+K_ds)}_{H_y(s)}Y(s).
\]

\[
G_a(s)=\frac{1}{s^2},\qquad
\frac{Y(s)}{R(s)}=
\frac{G_a(s)F_r(s)}{1+G_a(s)H_y(s)}
=\frac{K_ds+K_p}{s^2+K_ds+(1+\alpha)K_p}.
\]

따라서 분모에는 posture 항이 남고, 이 등가 모델의 DC gain은
\(1/(1+\alpha)\)이다. 실제 IK reference는 현재 상태에도 의존하므로
이 값이 보행 전체의 reference 추종 성능을 나타내지는 않는다.
전달함수 참고 PDF의 1쪽 입력과 출력은 절대 관절각이 아닌 **nominal 자세에 대한 편차**다.

## 2. 20 ms 이산 피드백과 이상적인 가속도 ZOH

실제 코드의 reference 속도는

\[
v_r[k]=\operatorname{clip}
\left(\frac{q_r[k]-q_r[k-1]}{T},-3,3\right),\qquad T=0.020\ \mathrm{s}.
\]

관절 가속도는 \(\pm150\ \mathrm{rad/s^2}\)로 제한한다.
두 제한과 QP의 가속도 변경을 제외한 등가 그림에서는

\[
D_b(z)=\frac{1-z^{-1}}{T},\qquad F_r(z)=K_p+K_dD_b(z).
\]

**측정 속도** \(\dot q[k]\)는 별도 feedback이다.
이를 \(D_b(z)q[k]\)로 대체하면 실제 V16과 다른 제어기가 된다.

상쇄된 단일 축의 가속도가 샘플 사이 일정하다는 이상화를 추가하면

\[
\begin{bmatrix}y[k+1]\\v[k+1]\end{bmatrix}
=\begin{bmatrix}1&T\\0&1\end{bmatrix}
\begin{bmatrix}y[k]\\v[k]\end{bmatrix}
+\begin{bmatrix}T^2/2\\T\end{bmatrix}a[k].
\]

\[
G_q(z)=\frac{T^2(z+1)}{2(z-1)^2},\qquad
G_v(z)=\frac{T}{z-1}.
\]

\[
\frac{Y(z)}{R(z)}=
\frac{G_q(z)F_r(z)}{1+(1+\alpha)K_pG_q(z)+K_dG_v(z)}.
\]

내부 피드백의 특성 다항식은 다음과 같다.

\[
D_{cl}(z)=(z-1)^2+K_dT(z-1)
          +\frac12(1+\alpha)K_pT^2(z+1).
\]

전체 reference-to-position 전달함수를 다항식 비율로 정리하면
\(F_r\)의 reference memory 때문에 일반적으로 분모는 \(zD_{cl}(z)\)다.
이는 reference 입력의 지연을 포함한 표현이고, homogeneous feedback의
두 pole은 \(D_{cl}=0\)에서 구한다.

실제 V16은 **가속도가 아닌 토크를 20 ms 유지**한다.
샘플 사이 \(M,h\)가 변하므로 비포화만으로 가속도 ZOH가 성립하지 않는다.
전달함수 참고 PDF의 2쪽은 실제 reference/feedback 구조와 **이상적인 상쇄 후 가속도-ZOH plant**의 조합이다.
실제 비선형 보행 전체의 pulse transfer function으로 표현하지 않는다.
PhysX 적분 주기는 이와 다른 \(0.005\ \mathrm{s}\)다.

### 최종 체크포인트의 실제 게인

| 제어기 / 훈련 Job | \(K_p\) [s⁻²] | \(K_d\) [s⁻¹] | \(\alpha\) | \((1+\alpha)K_p\) [s⁻²] |
|---|---:|---:|---:|---:|
| 접촉력 QP / 571 | 319.305390 | 27.468040 | 0.529243 | 488.295564 |
| 전신 QP / 577 | 289.332263 | 27.173097 | 0.647501 | 476.675238 |

`digital_acceleration_gains`의 학습된 frequency/damping에 대한 지정 pole 일치는
\(\alpha=0\)인 이상적인 이산 모델에서 성립한다. 자세 복원 항을 추가하면 pole이 달라진다.
게인은 각 훈련의 source SHA가 일치하는 frozen helper로 계산했다.

## 3. 12DOF 물리 모델의 전달함수와 초기조건

GUIDELINE 12단계에 사용한 **고정 base, 무접촉, 정지평형** source 모델에서
\(\tau_0=g(q_0)\), \(u=\tau-\tau_0\), \(y=q-q_0\)다.

\[
M_0\ddot y+D_0\dot y+K_gy=u,\qquad
K_g=\left.\frac{\partial g}{\partial q}\right|_{q_0}.
\]

\[
G_{\tau q}(s)=\left(M_0s^2+D_0s+K_g\right)^{-1}
\in\mathbb C^{12\times12}.
\]

이는 행렬 **역행렬**이며, nonzero off-diagonal 관절 결합을 유지한다.
저장된 source 모델은 \(D_0=1.5I\)이고, 실제 USD의 joint damping은 0이다.

\[
Y(s)=G_{\tau q}(s)U(s)+Y_{IC}(s),\quad
Y_{IC}=G_{\tau q}(s)\left[M_0(sy(0)+\dot y(0))+D_0y(0)\right].
\]

토크 ZOH에 대응하는 이 모델의 정확한 이산 전달행렬은

\[
G_{\tau q,d}(z)=C_q(zI-A_d)^{-1}B_d,
\quad A_d=e^{AT},\quad B_d=\int_0^T e^{A\eta}B\,d\eta.
\]

\(A,B,C_q\)와 실제 계산한 두 주기의 \(A_d,B_d\)는
`reports/v16/physical_model/v16_model_tensors.npz`에 있다.
초기조건 응답은 전달함수의 zero-initial input response와 별도로 보존한다.

## 4. 실제 전신 QP와 비선형 피드백

전달함수 참고 PDF의 3쪽 전신 QP는 가속도 \(a\in\mathbb R^{18}\)와 접촉력
\(\hat\lambda\in\mathbb R^{12}\)를 함께 조정한다.

\[
(a^*,\hat\lambda^*)\simeq
\underset{a,\lambda\in\mathcal C_k}{\arg\min}
\left\{\frac12\|a-a_d\|_W^2+
\frac{\epsilon_\tau}{2}\|\tau\oslash L\|^2+
\frac{\epsilon_\lambda}{2}\|\lambda/(mg)\|^2\right\},
\]

\[
\tau=S(Ma+h-J_c^T\lambda),\quad
S=[0_{12\times6}\ I_{12}],\quad
\epsilon_\tau=\epsilon_\lambda=0.01.
\]

\(W\)는 solver의 scaling을 실제 가속도 좌표로 옮긴 tracking weight이고,
\(\oslash\)는 성분별 나눗셈이다. 근사 기호는 **320회 유한 ADMM**을 표시한다.
이 그림이 매 tick 정확한 minimizer나 물리적 접촉의 보장을 뜻하지 않는다.
\(\mathcal C_k\)에는 몸통 동역학, active-contact 가속도, 마찰 피라미드,
normal force 및 motor torque/speed/slew bounds가 포함된다.

관절 reference는 발 Jacobian IK, 몸통 xy reference는 최소 jerk COM 궤적으로 생성한다.
xy 가속도는 \(\ddot c_r+K_c(c_r-c)+D_c(\dot c_r-\dot c)\)를 사용한다.
몸통 z는 root height feedback, 회전은 quaternion error/각속도 feedback이며 그림의
\(a_d\) 묶음에 포함한다. 이 COM task 가속도를 root 좌표의 목표 가속도에 적용하는 근사가 있다.

최종 토크의 실제 순서는 다음과 같다.

\[
\tau_k^*\ \longrightarrow\ \Gamma_T\ \longrightarrow\ H_T
\ \longrightarrow\ \Gamma_{5\mathrm{ms}}\ \longrightarrow\ \mathcal P.
\]

\(\Gamma_T\)는 digital tick마다 먼저
\(\operatorname{clip}(\tau_k^*,\tau_{prev}-2500T,\tau_{prev}+2500T)\)를 적용한다.
\(\tau_{prev}\)는 저장된 직전 실제 적용 토크다.
그 뒤 토크 요청을 ZOH로 유지하고, \(\Gamma_{5\mathrm{ms}}\)는 실제 속도에 따른
effort envelope를 매 physics step 적용한다. slew 제한을 5 ms마다 다시 적용하지 않는다.
유한 ADMM의 torque bounds만으로 digital slew clipping을 대체하지 않는다.
물리 plant의 \(\lambda_{phys}\)는 QP의 예측 접촉력 \(\hat\lambda^*\)와 별개다.
\(d\)는 모델에 포함하지 않은 마찰 등의 잔여 일반화 힘이다.
접촉 전환과 active bounds 때문에 전체 보행을 하나의 고정된 유리 전달함수로 쓰지 않는다.

## 재생성 및 검증

```bash
python3 config/experiments/v16_20260929/render_v16_control_math.py
```

새 물리 모델이나 시뮬레이션을 실행하지 않고 저장된 결과만 읽는다.
s/z 전달함수는 독립 state-space resolvent와 비교하고,
\(D_{cl}\)은 feedback state matrix의 특성식과 비교한다.
12DOF 물리 전달함수와 초기조건 항도 저장된24상태 resolvent와 대조한다.
SHA와 오차는 `reports/v16/figures/v16_control_math_verification.json`에 기록한다.
기존 구현 구성도 PDF는 `v16_modeling_diagrams_architecture.pdf`로 보존했다.
전달함수·비선형 수식 참고 PDF는 `v16_control_transfer_diagrams.pdf`다.
