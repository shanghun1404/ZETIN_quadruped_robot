# V6 CAD 외란·CAN·모터 지연 강건화 추가 훈련 (2026-10-25)

## 결론

Jetson 번들의 `model_15699.pt`(SHA `9a440065…`)를 V6 CAD(`assets/quadruped_v6_20261025`, 25.64 kg)에서 11,200 iteration 이어서 학습했다.
학습 중 밀기, 명령 지연·흔들림·누락, CAN 피드백 지연, 모터 토크 지연, 센서 노이즈를 점진적으로 키웠다(`--robust-profile motor`, retention 가중치 5).

최종 `model_26899.pt`(SHA `b8bc2e68…`)는 다음과 같다.

- **좋아진 점** (랜덤화 평가, 64대 × 40초)
  - 1.0 m/s 밀기 넘어짐: 12 → 5회
  - 어려운 조합: 32 → 19회. 모터 어려운 조합: 47 → 34회
  - 노이즈 2배: 넘어짐 1 → 0회, 요 각속도 오차 0.200 → 0.121 rad/s
- **나빠진 점**
  - 기존 ROS2/Isaac 9개 시험: **9/9 → 8/9**.
    - `yaw` 시험에서 0.25 rad/s 명령에 약 0.31 rad/s로 돌았다.
    - 그래서 14초 동안 목표 3.50 rad 대신 4.28 rad를 회전했다(오차 0.78 rad, 기준 0.6 rad).
  - 정상 조건 추종 오차
    - 속도: 0.030 → 0.037 m/s
    - 요 각속도: 0.025 → 0.038 rad/s

### 채택 기준 판정 (사용자 결정 2026-10-25)

| 기준 | 결과 | 판정 |
|---|---|---|
| ROS2/Isaac 9개 시험 9/9 | 8/9 (`yaw` 미통과) | **불합격** |
| 정격 RMS 토크 초과 0 | 9개 시험 모두 0. 최대는 무릎 0.88배(`stop_restart` FL_KFE) | 합격 |
| 1.0 m/s 밀기 넘어짐 ≤ model_15699(12회) | 5회 | 합격 |

**채택하지 않는다.** Jetson 모델은 `model_15699`를 그대로 유지한다.

중간 체크포인트 `model_24700`(SHA `3e11b4e2…`, 랜덤화 최대치에서 4,500 iteration 학습한 시점)도 마찬가지로 채택하지 않는다.
이 모델도 `yaw` 시험에서 오차 0.80 rad로 미통과해 8/9이다.

모든 결과는 Isaac 시뮬레이션이며, 실제 로봇 검증 결과가 아니다.

## 넣은 랜덤화

구현은 [`config/v15_cad_robust.py`](../../../config/v15_cad_robust.py)에 있다(SHA `1b0a0ddf…`, 학습 job 소스와 동일).
값은 [`train_jobs/11022/results/training_contract.json`](train_jobs/11022/results/training_contract.json)의 `robust` 항목에서 옮겼다.
물리 스텝은 5 ms, 정책 주기는 20 ms다.

| 항목 | 시작 (iteration 15700) | 끝 (ramp 4,500 iteration 이후) |
|---|---|---|
| 밀기 (8–12초마다 무작위 방향 수평 속도 변화) | U(0, 0.3) m/s | U(0, 1.0) m/s |
| 명령 지연 (에피소드마다) | 0–3 스텝 (0–15 ms) | 0–4 스텝 (0–20 ms) |
| 지연 흔들림 (스텝마다 추가, 더 오래된 목표로는 되돌아가지 않음) | 0 | 최대 2 스텝 (10 ms) |
| 명령 누락 (직전 목표 유지) | 0% | 5% |
| CAN 피드백 지연 (관절 위치·속도) | 0 | 0–2 스텝 (0–10 ms) |
| 모터 토크 1차 지연 시간상수 | 0 | 0–10 ms |
| 센서 노이즈 (균등 ±, 배율) | 0.25배 | 1배 = 몸통 선속도 0.15 m/s, 각속도 0.2 rad/s, 중력 방향 0.05, 관절각 0.02 rad, 관절속도 1.0 rad/s |

- 지연 상한 근거는 Jetson 측정값과 CAN 250 kbit/s 응답(약 4.7 ms)이다([10-23 보고서](../robust_20261023/README.md) 참조).
- 모터 토크 지연 0–10 ms는 **추정값**이다. 측정하지 않았고, AK 매뉴얼에도 응답 시간이 없다.
- 노이즈 크기도 **추정값**이다.

## 훈련

| 항목 | 값 |
|---|---|
| 시작·retention teacher | `model_15699` (job2341), retention 가중치 5.0 (10-23 실행은 1.0) |
| 프로필 | `rms_retain`, lr 3e-5, 평지·요철 혼합. 정책 std는 첫 세그먼트에서만 하한 0.10 적용 |
| 환경 수 | 64 (ws1 빈 자원에 맞춰 2 CPU, 15.9 GB) |
| 로봇 | V6 USD SHA `ab30c2dc…`, 25.6410 kg, 충돌체 75개 |

1시간 25분 이하 세그먼트로 나누어 각 세그먼트가 앞 세그먼트의 체크포인트에서 이어 학습했다.

| 세그먼트 (job) | 저장된 iteration | 랜덤화 레벨 | seed | 소요 시간 | 마지막 체크포인트 SHA |
|---|---|---|---|---|---|
| 10854 | 15700–17500 (1,801) | 0 → 0.40 | 73 | 1:25:13 (시간 제한) | `1c6f90a5…` |
| 10855 | 17501–20200 (2,700) | 0.40 → 1 (20196에서 1) | 1874 | 1:25:29 (시간 제한) | `a603d0e6…` |
| 10856 | 20201–22800 (2,600) | 1 | 4574 | 1:25:28 (시간 제한) | `9c39af75…` |
| 10857 | 22801–24700 (1,900) | 1 | 7174 | 1:25:29 (시간 제한) | `3e11b4e2…` |
| 11021 | 24701–26100 (1,400) | 1 | 9074 | 0:53:12 | `0d41a1b5…` |
| 11022 | 26101–26899 (799) | 1 | 10474 | 0:31:18 | `b8bc2e68…` |

- 처음 4개 세그먼트는 2,700–2,900회를 계획했지만 1:25 제한에 걸렸다.
  - 각 세그먼트는 마지막으로 저장된 체크포인트(100회 간격)에서 이어졌다.
  - 저장 뒤에 돈 몇십 회는 버려졌다.
- 남은 iteration은 [`train_target.sbatch`](../../../config/experiments/v6_robust_20261025/train_target.sbatch)로 채웠다.
  - 이 런처는 목표 26899까지 최대 1,400회씩 돈다.
  - 목표에 닿은 세그먼트가 `v6_robust_train_jobs/chain_10854_final.pt` 링크를 만든다.
  - 예비로 걸어 둔 11023–11025는 링크를 보고 학습 없이 끝났다.
- 각 세그먼트의 `checkpoints.sha256` 첫 줄(시작 actor)은 앞 세그먼트 마지막 체크포인트의 SHA와 일치한다.
- 이번 체인에서는 Isaac Sim 시작 실패가 한 번도 없었다.

학습 분포에서 넘어짐(base_tilt) 종료 비율의 1,000 iteration 평균은 다음과 같다([training_curve.png](training_curve.png)).

| 구간 | 평균 |
|---|---|
| 15700–16699 | 0.51% |
| 20200–21199 (레벨 1 직후) | 3.11% |
| 25900–26899 | 1.96% |

그래프에서 세그먼트가 시작될 때마다 보상이 크게 떨어지는 것은 통계 버퍼가 다시 시작되면서 생기는 로그상의 현상이다.
곡선은 [`merge_training_logs.py`](merge_training_logs.py)로 세그먼트 로그를 iteration당 한 번씩만 남겨 합친 뒤 그렸다.

## 평가 1: 랜덤화 시나리오 비교

[`evaluate_v15_cad_robust.py`](../../../config/evaluate_v15_cad_robust.py)로 평가했다(SHA `71fb4812…`).

- job 10853 = base, 11026 = final
- 학습과 같은 지형·명령 분포, 64대 × 2,000 정책 스텝(40초), seed 2026, 시나리오 값 고정
- 넘어짐 = base_tilt, low_height, physics_diverged 종료

표의 각 칸은 `넘어짐 횟수 / 평면 속도 오차(m/s) / 요 각속도 오차(rad/s)`다.

| 시나리오 | 조건 | base 15699 | final 26899 |
|---|---|---|---|
| nominal | 지연 0–15 ms | 0 / 0.030 / 0.025 | 0 / 0.037 / 0.038 |
| push_0.5 | 밀기 0.5 m/s | 1 / 0.035 / 0.025 | 0 / 0.044 / 0.040 |
| push_1.0 | 밀기 1.0 m/s | **12** / 0.047 / 0.034 | **5** / 0.052 / 0.044 |
| push_1.5 | 밀기 1.5 m/s (학습 범위 밖) | 54 / 0.065 / 0.040 | 56 / 0.071 / 0.050 |
| delay_30ms | 지연 30 ms 고정 | 0 / 0.039 / 0.041 | 0 / 0.039 / 0.037 |
| delay_40ms | 지연 40 ms 고정 | 0 / 0.050 / 0.086 | 0 / 0.045 / 0.048 |
| missed_10pct | 누락 10% | 0 / 0.032 / 0.028 | 0 / 0.036 / 0.036 |
| noise_1x | 노이즈 1배 | 1 / 0.059 / 0.114 | 0 / 0.051 / 0.066 |
| noise_2x | 노이즈 2배 | 1 / 0.100 / 0.200 | 0 / 0.078 / 0.121 |
| combined_train_end | v1 학습 끝 조건 | 4 / 0.072 / 0.135 | 4 / 0.060 / 0.074 |
| combined_hard | 밀기 1.0 + 지연 30 ms + 누락 10% + 노이즈 1.5배 | 32 / 0.116 / 0.205 | 19 / 0.093 / 0.119 |
| delay_jitter | 지연 10 ms + 흔들림 ≤10 ms | 0 / 0.032 / 0.029 | 0 / 0.035 / 0.034 |
| feedback_10ms | 피드백 지연 10 ms | 0 / 0.033 / 0.043 | 0 / 0.034 / 0.033 |
| motor_lag_10ms | 모터 지연 10 ms | 0 / 0.034 / 0.032 | 0 / 0.041 / 0.046 |
| motor_lag_20ms | 모터 지연 20 ms (학습 범위 밖) | 0 / 0.043 / 0.051 | 0 / 0.050 / 0.063 |
| combined_motor_end | 이번 학습 끝 조건 | 6 / 0.075 / 0.142 | 4 / 0.064 / 0.082 |
| combined_motor_hard | 밀기 1.0 + 지연 20 ms + 흔들림 + 누락 10% + 노이즈 1.5배 + 피드백 10 ms + 모터 10 ms | 47 / 0.133 / 0.237 | 34 / 0.107 / 0.151 |

- final의 넘어짐과 요 오차가 base보다 같거나 적은 조건:
  - 센서 노이즈, 고정 지연 30·40 ms, 피드백 지연
  - 여러 외란을 섞은 조합 4개
- final의 요 각속도 오차가 base보다 조금 큰 조건(0.005–0.015 rad/s):
  - nominal, 밀기만 있는 조건, 누락 10%, 지연 흔들림, 모터 지연
  - 1.0 m/s 밀기는 넘어짐이 줄었다.
  - 1.5 m/s 밀기는 54 → 56회로 비슷하다.
- seed 1개, 64대 표본이다. 넘어짐 횟수가 몇 회 차이 나는 것은 우연일 수 있다.

중간 체크포인트 `model_24700`의 랜덤화 평가는 job 11036으로 제출했다. ws1이 다른 사용자 작업으로 차 있어 이 보고서를 쓸 때까지 시작하지 못했다(Slurm 예상 시작 23:10).

## 평가 2: 기존 9개 ROS2/Isaac 시험

- 시험 코드는 round2 job 2343 스냅샷이다([`config/experiments/v6_cad_20261025/suite.sbatch`](../../../config/experiments/v6_cad_20261025/suite.sbatch), seed 42).
- base 기록은 [V6 보고서](../cad_v6_20261025/report.html)의 job 10813·10808이다.
- 중간·최종 결과는 [suite_jobs/](suite_jobs/)에 있다. 각 시험의 `video/env_00.mp4`가 영상이다.

각 칸은 결과와 `모터 RMS/정격 최대 (관절)`이다.

| 시험 | base 15699 (10813·10808) | mid 24700 (11037·11038) | final 26899 (11027·11028) |
|---|---|---|---|
| flat_14s | 통과 · 0.86 (RR_KFE) | 통과 · 0.83 (FL_KFE) | 통과 · 0.85 (FL_KFE) |
| stop_restart | 통과 · 0.85 (RR_KFE) | 통과 · 0.80 (RR_KFE) | 통과 · 0.88 (FL_KFE) |
| reverse | 통과 · 0.92 (RL_KFE) | 통과 · 0.76 (FL_HFE) | 통과 · 0.85 (RR_HAA) |
| lateral | 통과 · 0.93 (RR_KFE) | 통과 · 0.87 (RL_KFE) | 통과 · 0.82 (RL_KFE) |
| yaw | 통과, 회전 오차 0.11 rad · 0.81 (RR_KFE) | **미통과**, 회전 오차 0.80 rad · 0.77 (RL_KFE) | **미통과**, 회전 오차 0.78 rad · 0.73 (FL_KFE) |
| rough | 통과, 옆 이동 0.53 m · 0.85 (RR_KFE) | 통과, 옆 이동 0.21 m · 0.82 (RR_KFE) | 통과, 옆 이동 0.54 m · 0.83 (FL_KFE) |
| standing_hold | 통과 · 0.64 (RL_KFE) | 통과 · 0.64 (RL_KFE) | 통과 · 0.64 (RL_KFE) |
| sit_rise_walk_stop_sit | 통과 · 0.79 (RR_KFE) | 통과 · 0.77 (RR_KFE) | 통과 · 0.78 (RR_KFE) |
| deep_crouch_rise_walk_stop_sit | 통과 · 0.83 (RR_KFE) | 통과 · 0.82 (RR_KFE) | 통과 · 0.82 (RR_KFE) |

- 정격은 HAA 8.3 N·m, HFE·KFE 18 N·m다. RMS는 각 시험 전체 기록 구간의 값이다.
- 세 모델 모두 정격 초과와 사양 최대 토크 초과가 0이다.
- 11027·11037의 Slurm 상태가 FAILED인 것은 미통과 시험이 있어 0이 아닌 값으로 끝났기 때문이다. 실행 자체가 실패한 것은 아니다.

`yaw` 시험에서 명령 0.25 rad/s 대비 측정 요 각속도는 다음과 같다(`telemetry.npz`의 `angular_velocity` z 성분 평균).

| 구간 | base | mid | final |
|---|---|---|---|
| 0–2 s | 0.262 | 0.309 | 0.272 |
| 4–8 s | 0.245 | 0.308 | 0.309 |
| 12–14 s | 0.262 | 0.307 | 0.311 |

강건화한 두 모델은 2초 이후 명령보다 약 1.2배 빠르게 꾸준히 돈다. 시간이 갈수록 오차가 커지는 drift가 아니라 일정한 이득 오차다.

## 해석 (추정, 검증 안 함)

- 10-23 강건화(다른 CAD, v1 프로필, retention 1)도 `yaw`에서 4.14 rad를 회전해 미통과했다.
  - 이번에는 retention을 5로 올렸는데도 같은 방향의 회전 초과가 남았다.
  - 따라서 retention 가중치보다 랜덤화 자체가 원인일 가능성이 있다.
- 각속도 노이즈 ±0.2 rad/s가 `yaw` 시험 명령(0.25 rad/s)과 비슷한 크기다. 그래서 정책이 요 각속도 관측을 덜 믿게 되었을 수 있다.
- 어느 요인인지는 하나씩 떼어 보는 실험(노이즈만 / 지연·모터만 / 밀기만)을 해야 알 수 있다.

## 진행 중 문제와 처리

- **처리량**: 64환경에서 1:25 세그먼트 하나에 약 1,800–2,700 iteration이었다.
  - 세그먼트마다 학습 전 준비(Isaac 시작·preflight)에 약 10분이 들었다. job 소요 시간에서 로그의 학습 경과 시간을 뺀 값이다.
  - iteration당 평균 1.63–2.43초로, 세그먼트마다 달랐다.
  - 계획한 4세그먼트로는 11,200회를 채우지 못했다.
  - 그래서 처음 걸어 둔 afterok 평가 10858–10860을 취소했다.
  - 대신 이어 학습 11021–11025와 평가 11026–11028로 다시 걸었다.
- **대기열**: 다른 사용자 작업이 GPU 8 shard를 거의 다 쓰고 있었다.
  - 9개 시험 job(11027·11028)의 시간 제한을 대기 중 `scontrol update`로 25분으로 줄여 빈 시간에 들어가게 했다.
  - 11037·11038은 처음부터 `--time=00:25:00`으로 제출했다.
  - 복사된 `launcher.sbatch`에는 원래 값 45분이 남아 있다.

## 재현

ws1 작업 폴더는 `/home/shanghun1404/v15_cad_20261011`이다(`v6_robust_train_jobs/`, `v6_robust_eval_jobs/`, `v6_suite_jobs/`).
런처는 [`config/experiments/v6_robust_20261025/`](../../../config/experiments/v6_robust_20261025/)에 있다.

```bash
# 세그먼트 1–4 (train.sbatch iterations ramp num_envs)
sbatch train.sbatch 2900 4500 64                                                     # 10854
RESUME_JOB=10854 sbatch --dependency=afterany:10854 train.sbatch 2900 4500 64        # 10855
RESUME_JOB=10855 sbatch --dependency=afterany:10855 train.sbatch 2700 4500 64        # 10856
RESUME_JOB=10856 sbatch --dependency=afterany:10856 train.sbatch 2700 4500 64        # 10857
# 목표 26899까지 (train_target.sbatch target ramp num_envs cap link)
RESUME_JOB=10857 sbatch --dependency=afterany:10857 train_target.sbatch 26899 4500 64 1400 chain_10854_final.pt   # 11021
RESUME_JOB=11021 sbatch --dependency=afterany:11021 train_target.sbatch 26899 4500 64 1400 chain_10854_final.pt   # 11022
# 랜덤화 평가와 9개 시험
sbatch robust_eval.sbatch 2000 64 all base=train_jobs/2341/results/model_15699.pt          # 10853
sbatch robust_eval.sbatch 2000 64 all final=v6_robust_train_jobs/chain_10854_final.pt      # 11026
sbatch ../v6_cad_20261025/suite.sbatch flat_14s,stop_restart,reverse,lateral,yaw 42 <checkpoint>
sbatch ../v6_cad_20261025/suite.sbatch rough,standing_hold,sit_rise_walk_stop_sit,deep_crouch_rise_walk_stop_sit 42 <checkpoint>
```

표·그래프는 로컬에서 다음 명령으로 만들었다.

```bash
python3 -I merge_training_logs.py <합친 로그>
python3 -I ../../../config/experiments/v15_robust_20261023/summarize.py . <합친 로그> \
    eval_jobs/10853/results/robust_eval.json eval_jobs/11026/results/robust_eval.json \
    --suite "base 15699=../cad_v6_20261025/suite_jobs/10813/results/suite.json,../cad_v6_20261025/suite_jobs/10808/results/suite.json" \
    --suite "mid 24700=suite_jobs/11037/results/suite.json,suite_jobs/11038/results/suite.json" \
    --suite "final 26899=suite_jobs/11027/results/suite.json,suite_jobs/11028/results/suite.json"
```

## 파일

- `train_jobs/<job>/`
  - 런처, 소스·체크포인트 SHA, 학습 로그, Slurm 출력
  - 계약·preflight·완료 기록, TensorBoard 이벤트
  - 세그먼트별 마지막 체크포인트 (최종은 `train_jobs/11022/results/model_26899.pt`)
- `eval_jobs/10853`, `11026`: 랜덤화 평가 (소스 폴더 제외, `source.sha256` 포함)
- `suite_jobs/11027`, `11028` (final), `11037`, `11038` (mid 24700): 9개 시험 결과, 시계열, 영상, 소스
- `summary.json`, `tables.md`, `training_curve.png`, `robust_falls.png`: summarize.py 출력
