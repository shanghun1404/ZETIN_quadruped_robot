# V15.1 자기충돌 계측 범위

현재 비교에서는 기존 V14 USD 충돌 필터를 변경하지 않는다. 따라서 접촉력이 0이라는
결과를 로봇 전체의 무충돌 증명으로 해석하지 않는다.

| 대상 | 계측 | 한계 |
|---|---|---|
| 서로 다른 다리의 thigh/calf 캡슐 | 24개 캡슐 쌍의 기하 간격 | 반경 20 mm, 중심선 길이 230 mm의 기존 충돌 형상 |
| 서로 다른 다리의 thigh/calf/foot | 54개 고유 body 쌍을 포함하는 12개 접촉 채널 | 각 채널은 상대 body 9개의 최대 힘이므로 개별 상대 쌍 식별은 불가 |
| 같은 다리의 thigh와 foot | FL/FR/RL/RR 순서의 4개 캡슐–foot 경계 상자 간격 | 실제 접촉 쌍은 USD에서 필터링됨; 새 계측은 독립적인 보수적 기하 검사 |
| 8개 thigh/calf 캡슐과 몸통 | 기존 몸통 상자 프록시와의 간격 | 실제 몸통 collider가 없으며 foot–몸통 및 기타 부품은 포함하지 않음 |

같은 다리의 thigh–calf, calf–foot뿐 아니라 **thigh–foot도 필터링되어 있다**.
thigh–foot에는 중간 calf 구간이 있으므로 이 제외를 모두 “서로 인접한 링크”로 부르면
정확하지 않다. leg 캡슐–지면 접촉도 제외되어 있다. 관절 하우징, 커버 등 모델링하지 않은
부품의 간섭, 실제 CAD 전체 간섭, 접촉 오프셋 및 solver 오차는 이 검사로 판정하지 않는다.

## 실제 foot collider 근거

2026-10-01에 ws1의 구성된 V14 USD를 SimulationApp 없이 읽었다.
네 발 모두 `.../{leg}_foot/collisions/foot/foot`에 20,094개 정점을 가진 `Mesh`가 있으며,
물리 근사는 `convexHull`이다. 구나 상자가 아니므로 foot actor 좌표계에서 모든 정점을
포함하는 AABB를 사용한다. 이 상자는 convex hull 전체를 포함한다.

- mesh→foot 변환: 균일 스케일 `0.0010000000474974513`, 추가 회전·평행이동 없음.
- **정점 자체가 원점에서 떨어져 있음**: foot 좌표계의 상자 중심은
  `(0.1798692178, 0.3099000086, -0.5425212965)` m.
- 상자 반크기는 `(0.0199852381, 0.0199880991, 0.0225212718)` m.
- 원시 bounds, 변환, USD 해시 및 실제 필터 여부:
  [foot_collider_evidence.json](../reports/v151/geometry/foot_collider_evidence.json).

평가기는 상수를 맹목적으로 재사용하지 않고 실행 중 구성된 USD의 정점과 상대 변환을
읽어 상자를 다시 만든다. 콜라이더 종류나 `convexHull` 근사가 달라지면 평가를 중단한다.
따라서 foot actor 원점이나 COM을 발 중심으로 가정하지 않는다.

## 결과 해석

NPZ의 `same_leg_thigh_foot_gap_m`은 `[physics step, 4]`이며 순서는 FL, FR, RL, RR이다.
JSON은 전체 최소 간격, 쌍별 최소 간격, 프록시 겹침 표본 비율 및 생성한 상자 근거를
별도로 기록한다. 기존 cross-leg 후보 채택 기준에는 이 값을 섞지 않는다.

간격은 `distance(thigh capsule centerline, solid foot proxy box) - capsule radius`이다.
양수이면 이 두 모델 형상은 분리되어 있다. 음수이면 **상자 프록시의 간섭 가능성**이며
실제 convex hull의 충돌이나 침투 깊이를 확정하지 않는다. 중심선이 상자 안에 있으면
간격이 `-capsule radius`에서 포화된다. 물리 접촉 필터는 그대로 보존한다.

이 채널이 없는 기존 실행을 새 검사 통과로 소급 분류하지 않는다. 새 물리 실행의 기록이나
충분한 원래 body pose 자료로 별도 계산한 결과만 이 검사의 근거로 사용한다.
