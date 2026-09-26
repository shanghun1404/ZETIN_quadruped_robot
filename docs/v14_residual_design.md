# V14 보정 정책 파일럿

사용자가 2026-09-26에 승인한 구조: 기존 보행 policy와 V12 시각 모델을
고정하고, 작은 관절 보정 policy만 HPC에서 훈련한다. 자동 commit하지 않는다.

## 구조

- 기준 actor는 V14 작업110의 model_5700.pt. 기존235차원 depth 관측을 그대로
  받으며, 실제로 적용한 합성 action이 다음 관측의 previous action에 들어간다.
- 새 MLP actor/critic은128×128 ELU, 입력307차원:
  기존235 + 기준 action12 + 다리 캡슐 간격24 + 접근속도24 + 직전 보정12.
- 보정은 `delta_rad=.05*tanh(raw_action)`, 합성 action은
  `base_action+delta_rad/.2`. 출력 평균을0으로 초기화하고 초기 raw noise std=.05.
  확률적 탐색까지 tanh 한도에 포함한다. 기존 action scale=.2를 검증한다.
- 합성 action 이후 기존25 Hz LPF와0.99 토크 가드를 통과한다.
- 간격은 기존20 mm 외피의 서로 다른 다리24쌍으로 계산한다. 강체 자세의
  기하학적 거리만 사용한다. 실제 로봇 적용 시 관절 기반 FK 계산 구현이 필요하다.
  접근속도는 직전 간격 차분이며 episode reset에서0, 같은 step을 다시 읽어도
  이력이 변하지 않는다. 간격/속도/보정 입력은 각각.1m/1m/s/.05rad로 정규화하고
  간격·속도 입력만 ±5로 제한한다.
- 기존 보행 보상 유지. 추가 cost는 max(relu((.02-gap)/.02), pair)^2를9까지
  제한하고 weight=-2.0. 보정 크기 mean((delta/.05)^2)는-.25,
  보정 변화 mean(((delta-prev_delta)/.05)^2)는-.05. 모두 env reward manager에서
  reset 전에 계산하고 기존 control dt 계수를 적용한다.
- 조향 복구는 비교군과 동일하게 사용한다. yaw reward는 원래 std=2.0,
  weight=5.0을 유지한다. 강한 방향 보상 적응은 적용하지 않는다.

## 훈련과 비교

32개 고유 타일, 기존 V14 stair terrain mix/curriculum, 초기 L0, train seed46.
새 policy만 PPO 고정LR1e-4, entropy0,300 updates,50회 간격 저장.
기준 actor/student는 eval mode, requires_grad=False. 시작/종료 tensor hash 일치 검증.
최초 보정0 checkpoint를 별도 저장하고 실제 fixed_v2 계단 seed42에서
기존5700+조향의 JSON/trajectory와 비교한다. 이 검증이 통과한 뒤 훈련한다.
checkpoint 옆 manifest에 base/student/USD 해시와 보정 상한·관측/정책 구조를
기록하고 평가 시 일치를 강제한다.

통과 기준: 0 보정에서 동일한 행동·실제 첫 episode 궤적 재현, 훈련 중 보정
상한 준수와 frozen 가중치 불변, 고정15.7cm 계단에서 충돌·생존·진행 동시 개선,
평지 생존 유지 및 토크 초과0. 파일럿에서 유효한 후보만 추가 시드에서 확인한다.
작은 간격만 좋아지고 조기에 넘어지는 모델은 제외한다. 개선이 없으면 사실대로
기각하며 전체 V14가 해결됐다고 주장하지 않는다.
