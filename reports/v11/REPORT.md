# robot_prama Stage 1 V11 훈련 보고서

작성일: 2026-08-12 KST

## 결론

선택 모델은 `final/checkpoint/model_4999.pt`이다.

- checkpoint iteration: `4999`
- SHA256: `53a6a5002f4cafb87889d777fcc567157245c786f82376db08962c9227acc2b7`
- 관측: 기존 48차원 proprioception + 전방 높이 187점 = 235차원
- 전진축: robot_v2 base `+y`
- rough 고정 난이도 4: seeds 42–46 전부 통과
- 5–20 cm 상승 계단: seeds 42–46 전부 통과
- 실제 15 cm 기준: 모든 시드에서 최소 15.714 cm 행을 50% 이상 통과
- 선택 정책의 15.714 cm 계단 통과 영상: `final/videos/v11_stairs/`
- 선택 정책의 15.714 cm 계단 하강 영상: `final/videos/v11_stairs_down/`
- V12 vision 학습 완료: 84×84 depth를 기존 187점 scan으로 변환하는 Tiny ViT를
  실제 IsaacLab 카메라 데이터로 distillation하고 1회 DAgger 보강함
- 선택 V12 checkpoint: `final/vit_stage2_trained/checkpoints/dagger/best.pt`
- 선택 V12 SHA256: `74c0c15644aec802c2637e5054286ea261e0fd6060637506eee61776af3b6f05`

이 결과는 IsaacLab 시뮬레이션 검증 결과이며 실제 로봇 하드웨어 검증은 아직 수행하지 않았다.

## V12 Depth Tiny ViT 훈련 및 비교

V11 `model_4999.pt`를 수정하지 않은 교사로 고정하고, 전방 depth 카메라 입력만으로
187점 terrain scan을 복원하는 Tiny ViT를 훈련했다. 학생 정책은 매 step의
`depth [1,84,84]`와 기존 48차원 proprioception만 사용한다. privileged 187점 scan과
교사 action은 학습 label 및 평가 오차 계산에만 사용했다.

- teacher rollout train: 65,536 samples, seed 42, 32 shards
- teacher rollout validation: 8,192 samples, seed 142, 4 shards
- student rollout DAgger: 32,768 samples, seed 242, 16 shards
- 전체 52개 shard의 SHA-256과 sample count 재검증 통과
- 초기 학습: 20 epochs, best epoch 20
- DAgger 재학습: 초기 best에서 10 epochs, best epoch 7
- 최종 validation: total loss `0.00184242`, scan MAE `0.0347112`,
  action MAE `0.0277216`
- 초기 best 대비: scan MAE `0.0365792 → 0.0347112`, action MAE
  `0.0298209 → 0.0277216`

같은 rough L4, stairs-up/down L5와 seeds 42–44에서 각 모델을 9회씩 폐루프
평가했다.

| 모델 | 전체 통과 | 평균 생존 | 평균 속도 | 평균 scan MAE | 평균 action MAE |
|---|---:|---:|---:|---:|---:|
| 초기 ViT | 6/9 | 0.8924 | 0.4611 m/s | 0.05446 | 0.05342 |
| DAgger ViT | 7/9 | 0.9167 | 0.4384 m/s | 0.05041 | 0.04796 |

| 시나리오 | 초기 통과 | DAgger 통과 | 초기→DAgger 핵심 변화 |
|---|---:|---:|---|
| rough L4 | 2/3 | 3/3 | 평균 생존 `0.8125 → 0.8854` |
| stairs-up L5 | 1/3 | 1/3 | 평균 traversal `0.3542 → 0.4479` |
| stairs-down L5 | 3/3 | 3/3 | 평균 traversal `1.0000 → 0.8958` |

DAgger 모델은 전체 통과 수, 평균 생존율, scan/action 오차가 더 좋아 최종 모델로
선택했다. 다만 평균 속도는 `0.4611 → 0.4384 m/s`로 낮아졌고, 계단 오르기는
여전히 2개 시드가 엄격한 통과 기준에 미달하며, 계단 내리기 traversal도 소폭
낮아졌다. 따라서 이 결과를 모든 계단 조건의 해결로 해석하지 않는다.

## 최종 5시드 결과

| 모델/시나리오 | 통과 시드 | 최저 생존 | 최저 속도 | 최저 계단 성공 | 모든 시드가 통과한 최고 계단 |
|---|---:|---:|---:|---:|---:|
| blind V10.4 @2000, rough L4 | 3/5 | 0.8750 | 0.4023 m/s | — | — |
| blind V10.4 @2000, stairs | 0/5 | 0.8750 | 0.1788 m/s | 0.0156 | 없음 |
| scanner V11 @4999, rough L4 | 5/5 | 0.9062 | 0.5543 m/s | — | — |
| scanner V11 @4999, stairs | 5/5 | 0.9375 | 0.4149 m/s | 0.5938 | level 5 = 0.157 m |

V11 rough 평가의 최저 direct-contact phase valid fraction은 1.0이었다. Seed 42의
RL–RR 직접 접촉 상관 중앙값은 `-0.8956`으로 뒷다리 교대도 유지됐다.

최종 기준:

- rough: survival >= 0.90, speed >= 0.30 m/s, direct phase valid >= 0.90
- stairs: survival >= 0.80, 전체 climb success >= 0.50,
  실제 `hardest_step_height_cleared_m >= 0.15`이며 해당 행의 success >= 0.50

## 개발 의사결정

### V10.5를 생략한 이유

V10.3/V10.4의 checkpoints 1000/2000/2999를 seeds 42–46에서 총 30회 다시
측정했다. 기존 HFE pooled correlation이 보였던 RL–RR 문제는 직접 접촉 위상에서는
재현되지 않았다. 여섯 체크포인트 모두 rear weakness 확인 시드가 0/5였다. 따라서
근거 없는 위상 보상을 추가하는 V10.5 훈련은 생략하고 V10.4 @2000을 V11의 terrain
bootstrap으로 선택했다.

### V11 전이 방식

actor/critic 첫 층을 `[512,48]`에서 `[512,235]`로 확장했다. 기존 48열은 그대로
복사하고 스캔 187열은 0으로 시작했다. 대응하는 Adam 모멘트도 같은 방식으로
확장했다. 원본/확장 모델은 0 스캔 입력에서 actor와 critic 모두 최대 절대 오차 0을
기록했다. 변환 전에는 검증된 V10 파라미터 순서와 Adam 필수 상태(`step`,
`exp_avg`, `exp_avg_sq`, AMSGrad일 때 `max_exp_avg_sq`)를 강제한다.

## 발견하고 수정한 실패

1. **optimizer reset 붕괴**
   - 첫 구현은 Adam 상태와 iteration을 초기화했다.
   - 확장 직후 모델은 생존 100%였지만 첫 PPO update의 `model_0`은 생존 0%였다.
   - Adam 모멘트, iteration, adaptive learning rate를 함께 복원하자 5-update smoke
     이후에도 생존 100%를 유지했다.

2. **resume 시 terrain curriculum reset**
   - 모델 checkpoint에는 환경별 terrain row가 저장되지 않는다.
   - 단순 resume은 이전 종료 level 5.17 대신 0.77에서 다시 시작했다.
   - `--initial-terrain-level`을 추가하고 terrain origin을 재배치한 뒤 reset하도록
     수정했다. level 5 smoke는 4.985, level 6 smoke는 5.985에서 시작했다.

3. **계단 평가의 최종 높이 오판**
   - 빠른 정책이 쉬운 계단을 넘은 뒤 반대편으로 내려가면 12초의 최종 높이가 낮아
     성공을 실패로 기록했다.
   - 접근 거리 2.1 m 이후의 최고 높이를 사용하도록 수정했다. 수평 접근 거리와 최종
     생존 조건은 그대로 유지했다.

4. **집계 입력 약화 가능성 차단**
   - 집계기가 호출자 지정 시드나 결과 JSON의 자체 보고 level에 의존할 수 있었다.
   - 집계 시드를 정확히 42–46으로 고정하고, 계단 게이트는 level 번호가 아니라 평가기가
     계산한 실제 계단 높이(m)를 직접 사용하도록 변경했다.

## 훈련 경과

- V10.4 @2000 exact bootstrap
- V11 @2999: terrain level 약 5.17; rough 3/5, 계단 목표 미달
- level-5 continuation V11 @3999: terrain level 약 5.93; rough 5/5,
  일부 시드가 15.7 cm 통과
- level-6 continuation V11 @4999: terrain level 약 6.04; rough 5/5,
  stairs 5/5

최종 checkpoint의 스캐너 actor input-weight L2는 0에서 실제 비영 값으로 증가해
정책이 단순 blind 복제 상태에 머물지 않았음을 확인했다.

## 검증

- V11 완료 시 원격 전체 정적 테스트: `522 passed`
- V12 완료 후 새 원격 전체 테스트: `554 passed in 2.04s`
- V12 train/validation/DAgger manifest: 52 shards, 106,496 samples, 모든 SHA 일치
- V12 선택 checkpoint: strict-load, 모든 상태·forward 유한, 출력 `[2,187]`
- V12 폐루프 원시 JSON: 초기 9개 + DAgger 9개, seeds 42–44와 actor equivalence 0 확인
- live scanner/physics probe: `ok=true`
- zero-scan bootstrap equivalence: `ok=true`
- 최종 raw evaluation JSON: rough 5개 + stairs 5개, 정확한 seeds 42–46 확인
- 모든 최종 JSON에서 비유한(non-finite) 수치 없음
- 남아 있는 V11 training process 없음
- 상승 계단 영상: H.264, 1280×720, 50 fps, 599 frames, 11.98 s;
  seed 0 텔레메트리 `climbed=true`, 종료 생존, 진행 5.584 m
- 하강 계단 영상: H.264, 1280×720, 50 fps, 599 frames, 11.98 s;
  seed 8 텔레메트리 `descended=true`, 종료 생존, 진행 4.671 m,
  접근 후 최대 하강 1.425 m. 전체 MP4를 오류 없이 디코딩했다.

## V12 Tiny ViT 구조와 훈련 상태

현재 V11은 순환망이 없는 Actor-Critic MLP다. actor와 critic은 각각
`235 → 512 → 256 → 128` ELU 은닉층을 쓰며, actor는 12개 관절 action mean을
출력한다. 입력 235개는 48개 proprioception 뒤에 187개 height scan을 붙인 순서다.

훈련한 V12 front-end는 다음 경계를 유지한다.

`depth [1,84,84] → Conv2d(1,128,k=7,s=7) → 144 patch + CLS →`
`Transformer 4 blocks / 4 heads / FFN 256 → LayerNorm → Linear(128,187) → tanh`

- depth 정규화: 0.1–5.0 m를 `[0,1]`로 clip, NaN/Inf는 far 값으로 치환
- 학습 손실: scan Smooth-L1 + `0.25 ×` frozen-teacher action MSE
- 파라미터: `579,387`; 최종 checkpoint strict-load 및 유한값 forward 검증 통과
- `final/vit_stage2/tiny_depth_vit_initial.pt`는 비교용 초기화 가중치다.
- 실제 훈련 checkpoint, 데이터 manifest, 평가 결과와 로그는
  `final/vit_stage2_trained/`에 보관했다.

다음 실행 단계는 실패가 남은 계단 오르기 분포를 별도 분석하고, 추가 학습 전에
현재 카메라 정책을 실제 robot_v2 하드웨어의 저속·안전 구간에서 검증하는 것이다.

## 아카이브 구조

- `final/checkpoint/model_4999.pt`: 선택 모델
- `final/checkpoint/params/`: 선택 run의 env/agent YAML
- `final/results/v11_final_4999_multiseed/`: final rough 원시 결과와 집계
- `final/results/v11_final_4999_physical_stairs/`: 실제 높이 필드를 포함한 final stairs
  원시 결과와 집계
- `final/results/v10_4_source_final_multiseed/`: blind 기준선 rough 평가
- `final/results/v10_4_source_physical_stairs/`: 동일 실제 높이 판정의 blind stairs 기준선
- `final/videos/v11_stairs/`: V11 @4999의 15.714 cm 계단 측면 영상과 텔레메트리
- `final/videos/v11_stairs_down/`: V11 @4999의 15.714 cm 하강 계단 측면 영상,
  텔레메트리와 실행 로그
- `final/vit_stage2/`: Tiny ViT 초기 상태와 합성 forward/backward smoke 보고서
- `final/vit_stage2_trained/`: 초기/DAgger 훈련 checkpoint, 데이터 manifest,
  폐루프 비교 JSON, 검증 보고서, 소스와 실행 로그
- `final/baseline_audit_raw/`: V10.3/V10.4 30개 기준선 원시 JSON
- `final/logs/`: 실패 run, exact-resume run, level-5/6 continuation 로그
- `final/run_control/`: probe, equivalence, evaluator suite 로그
- `config/`: 최종 소스와 테스트 사본

## 남은 한계

- 실제 robot_v2 하드웨어의 계단/거친 지형 검증은 별도 안전 절차가 필요하다.
- rough 평가는 생성 지형 row 4에 고정해 재현성을 확보했다. 더 높은 row 및 외부 terrain
  세트에 대한 out-of-distribution 평가는 후속 단계다.
- V12의 시뮬레이션 depth 데이터 수집·최적화·폐루프 검증은 완료했지만, 실제 카메라의
  노이즈·지연·시야 차이와 robot_v2 하드웨어 성능은 아직 검증하지 않았다.
- V12 계단 오르기는 seeds 42–44 중 1개만 엄격한 기준을 통과했다. 후속 개선 시에는
  현재 결과를 기준선으로 유지하고 새로운 분포 데이터로 별도 비교해야 한다.
