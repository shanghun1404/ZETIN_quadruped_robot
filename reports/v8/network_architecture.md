# V8 신경망 구조

체크포인트에서 직접 추출한 실제 구조 (`results/network_structure.json`).

## 구조

```
관측 48차원
  base_lin_vel 3 | base_ang_vel 3 | projected_gravity 3
  velocity_commands 3 | joint_pos 12 | joint_vel 12 | last_action 12
        |
  Actor  : 48 -> 512 -> 256 -> 128 -> 12   (ELU)   190,860 params
  Critic : 48 -> 512 -> 256 -> 128 -> 1    (ELU)   189,441 params
  std (학습되는 행동 표준편차)                        12 params
        |
  12개 관절 목표각도  (x scale 0.2 + 기본자세 오프셋)
```

- **총 380,313 파라미터**
- `rsl_rl` 기본 `ActorCritic` (순수 MLP, 순환 없음)
- 관측 정규화 없음 / 초기 탐색 std = 1.0 / entropy_coef 0.0025
- PPO: num_steps_per_env 24, lr 1e-3 (adaptive), gamma 0.99, lam 0.95, desired_kl 0.01

## v5 계열과의 차이

| | v5 (SmoothActorCritic) | v8 |
|---|---|---|
| 구조 | GRU + CNN 커스텀 | 순수 MLP |
| 관측 | 238차원 (proprio 46 + 지형스캔 187 + 명령 5) | 48차원 |
| Actor | [256, 256, 128] | [512, 256, 128] |
| 초기 std | 0.35 | 1.0 |
| 행동 경로 | 저역 스무딩 필터 | 필터 없음 |

GRU를 뺀 이유는 성능이 아니라 **교란 변수 제거**입니다. 보행 실패 원인이 리워드인지
신경망인지 구분되지 않아, 검증된 레시피(MLP)로 먼저 리워드 가설을 확정한 것입니다.

## 아직 복원해야 할 항목

- **지형 스캐너 (187차원)** — 현재 `height_scanner = None`. 평지 전용이며 험지/계단·Stage 2 비전에 필수.
- **`base_lin_vel` 제거** — 실물에서 직접 측정 불가(상태추정 필요). 진짜 proprioception-only가 되려면 빼야 함.
