# PhysX ↔ 원본 MJCF 비교 준비

실제 계산은 완성된 probe `physx_nominal.npz`, `manifest.json`, `jacobian_validation.json`을 받은 뒤
HPC CPU Slurm에서 수행한다. 준비 단계에서 모델 계산이나 simulator를 실행하지 않았다.
코드는 `config/experiments/v16_20260929/v16_compare_physx.py`이며 local은 독립 좌표 대수 6개만 검증했다.

## Snapshot 계약

- `q,dq,joint_names`: explicit canonical joint order. generalized M/h/g/coriolis와 J columns는 root6+joints12.
- root actor pose는 `root_actor_position`, `root_quaternion_wxyz`; root COM pose/twist도 별도 보존.
- `root_actor_twist_w`와 `root_com_linear_velocity`, `root_angular_velocity`의 관계를 independently 확인한다.
- `body_positions,body_quaternions`는 actor world/wxyz; `coms`는 actor-local COM position+xyzw rotation.
- `inertias`는 link COM 기준, rigid-body actor/prim axes 표현, **column-major9**.
  source fullinertia의 body axes 표현과 비교하기 위해 각 actor rotation으로 world 표현을 만든다.
  COM principal quaternion을 다시 관성에 적용하지 않는다.
- `joint_armatures,joint_damping,joint_friction`, `gravity_world`, `env_origin`도 저장한다.
- `jacobian_validation.json`은 foot velocity의 origin이 actor인지 COM인지 명시한다.

위 tensor 의미는 설치 SDK 107.3.26에 맞춘
[Omni Physics 107.3 Articulation API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html) 계약이다.
다른 버전 RigidBodyView의 inertia 설명을 섞어 principal axes로 다시 회전하지 않는다.

## 좌표 변환

root actor orientation `R`, actor→COM world offset `r_w`, Pin base actor-local velocity `[v_A,omega]`에 대해

\[
v_{PX}=T v_{Pin},\quad T_6=\begin{bmatrix}R&-[r_w]_\times R\\0&R\end{bmatrix}.
\]

`T`의 joint block은 identity다. 질량과 generalized force는 dual transform을 사용한다.

\[
M_{PX\to Pin}=T^TM_{PX}T,\qquad
g_{PX\to Pin}=T^Tg_{PX},\qquad
J_{PX\to Pin}=J_{PX}T.
\]

bias에는 moving frame acceleration 항도 필요하다.

\[
\dot T v=\left[R(\omega\times v_A+\omega\times(\omega\times r_{A,COM}));0;0_{12}\right],
\quad h_{PX\to Pin}=T^T(h_{PX}+M_{PX}\dot T v).
\]

대수 fixture는 90도 회전·nonzero COM offset·angular velocity·centripetal term으로 부호를 확인했다.
원본의 fixed members는 source Pin joint별 집계로 비교하고, PhysX에 같은 이름 body가 남으면
해당 source cluster에 대응되는 runtime body들을 world에서 합산한다.
따라서 individual body mass 차이를 fixed-member 병합과 혼동하지 않는다.

USD가 actor frame을 이동/회전했을 가능성이 있으므로 실제 COM/actor point의 J와 속도는
`r_source = R_source.T @ (p_PhysX_world - p_source_world)`로 같은 **world point**를 source frame에
표현해 비교한다. 같은 local coordinates를 그대로 사용한 J 차이는 별도 raw diagnostic으로 남긴다.

## 포함/제거 조건

- source `M_rigid`, `M_source_with_armature(.02)` 두 모드를 모두 내보낸다.
- PhysX M의 armature 포함 여부는 처음에는 `unknown`이다.
  실제 armature values와 `included` 선언이 있을 때만 PhysX diagonal armature를 제거한다.
- 비교 source bias에는 damping을 넣지 않는다. source damping 1.5가 포함된 bias도 별도 배열로 보존한다.
  PhysX gravity/coriolis API에는 damping·friction·contact wrench가 포함되지 않는 조건을 기록한다.
- root base Jacobian의 6 columns는 world COM/actor, link velocity point는 COM/actor의 네 가지
  이론적 형태와 각각 비교한다. 가장 작은 값을 자동으로 골라 좌표 계약을 바꾸지 않는다.
- 변환 USD가 source와 동일하다는 가정으로 결과를 통과시키지 않는다. 실제 차이를 JSON/NPZ/CSV로 남긴다.
- source base actor의 pose에는 측정된 PhysX root actor pose를 대입하는 명시적 correspondence를 사용한다.
  USD root actor frame 자체가 재정의됐다면 별도의 독립 root gauge 측정 후 physical parameter 차이로 해석해야 한다.

## 준비 코드 검증·리뷰

로컬 독립 대수 6개가 통과했고 Python syntax compile과 `git diff --check`도 통과했다.
읽기 전용 리뷰에서 지적된 같은 world point 정렬 문제는 회귀 fixture와 함께 수정했다.
수정 후 root actor correspondence 가정 범위에서 remaining Critical/Important 오류가 없음을 확인받았다.
실제 source model 비교 실행의 증거로 이 대수 결과를 사용하지 않는다.

## 실행

GPU probe가 완료된 뒤 다음 CPU job을 제출한다.

```bash
sbatch /home/shanghun1404/v16_12dof_model/model/reports/physx_comparison.sbatch \
  /path/to/completed/probe unknown
```

로컬 대수 검증 결과는 `physx_comparison_algebra.log`다. 실제 수치 비교 결과는 준비 문서에 만들지 않는다.
