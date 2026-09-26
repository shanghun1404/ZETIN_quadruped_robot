# V14 보정 policy — 확대 훈련

2026-09-26 사용자 요청: “계속 훈련해봐 훈련 사이즈 늘려서”.

**HPC 본 훈련 작업186 진행 중. 성능 개선 여부는 아직 판정하지 않았다.**

2026-09-26T19:20:28.439801+09:00 확인: 추가 232/3,000회,
712,704개 전이를 처리했고 model200/300을 저장했다.
[확인 시점의 진행 정보](progress_snapshot.json), [Slurm 상태](hpc_jobs.txt).
최종3,000회 훈련과 예약된 평가 결과는 아직 완료되지 않았다.

## 변경한 규모

| 항목 | 기존 파일럿174 | 확대 훈련186 |
|---|---:|---:|
| 병렬 환경 | 32 | 128 |
| 업데이트당 전이 | 768 | 3,072 |
| 업데이트 | 300 | 3,000회 추가 |
| 이번 작업의 총 전이 | 230,400 | 9,216,000 |
| 시작 정책 | zero residual | 작업174 model_100 |
| 새 iteration 번호 | 0–299 | 101–3100 |

환경은4배, 이번 실행의 학습 데이터 양은 기존300회 파일럿의40배다.
actor/critic128×128, 307차원 관측, 보상, seed46, 고정LR1e-4는 그대로 유지했다.
V14 model5700 보행 actor와 V12 시각 student를 동결하고 보정 policy만 훈련한다.
기존 조향 제어, .05rad 보정 상한,25Hz LPF,0.99 토크 가드를 유지한다.
terrain curriculum은 새 환경에서 L0부터 다시 시작하며128개 고유 타일을 사용한다.
optimizer도 복원하되 물리 상태·난수 상태를 그대로 재개하는 bitwise restart는 아니다.

## 실행과 검증

- **184:** 128환경20회 시험 훈련 `COMPLETED 0:0`,2분44초.
  iteration101–120,61,440개 전이, host MaxRSS13,617,796KiB.
  기존 actor/student hash 동일, residual finite, 최대 관측 보정.014667287rad.
  [검증](smoke184_verification.json), [manifest](smoke184_manifest.json), [로그](smoke184.log).
- **186:** 원래 model_100부터 다시 시작해128환경3,000회 훈련.
  admin QoS,1GPU,8CPU,32GiB,2시간 상한. 시험 학습 속도는 약1초/update였다.
- **190:** 100회 추가 업데이트 후 중간model200의 계단 seed42 평가를 제출했다.
  checkpoint hash가 훈련 중 manifest에 등록되고 검증되는 것도 확인했다.
- **187/188/189:** 작업186이 정상 완료된 뒤 실행하도록 Slurm `afterok:186` 의존성 예약.
  최종model3100의 fixed_v2 계단 seed42/43 및 평지 seed42 평가다.
  각32환경12초, 계단 실제 높이15.714cm. 1GPU/16GiB로 평가한다.
- 본 훈련의 실제 시작 복사본도 원래 model100과 policy/critic/optimizer가 정확히
  일치했다. [동일성 검사](train186_resume_identity.json).
- 재개 코드 및 중간 저장 검증: 실제 설치된 RSL runner를 사용하는 HPC CPU 검사18개,
  로컬 순수 기하/평가/동일성 관련 검사27개 통과. 컴파일, bash -n, diff whitespace 검사 통과.
- 독립 리뷰의 Important1건(중간 checkpoint 미봉인)을 수정했다. 각 checkpoint를
  임시 파일에서 원자적으로 교체한 뒤 SHA-256을 manifest에 등록하여 훈련 중에도
  평가·재개할 수 있다. Minor1건(zero-only 전이 메타데이터)도 수정했다.
- 학습된 policy/critic/optimizer를 복원하고 완료 iteration100 다음인101부터 시작한다.
  보정 출력을0으로 다시 초기화하지 않는 것을 실제 runner 검사로 확인했다.

## 산출물과 재현

HPC 훈련 폴더: `/home/shanghun1404/v14_leg_collision/residual_jobs/186/`.
100회마다 저장하며 최종은 `model_3100.pt`다. manifest도 같은 폴더에 있다.
훈련 완료 때 `verification.json`에 frozen hash와 보정 상한 검사 결과를 기록한다.
HPC 평가 결과: `/home/shanghun1404/v14_leg_collision/results/direction/scale186_final_*`.

```bash
cd /home/shanghun1404/v14_leg_collision/config
sbatch launch_v14_residual_scale.sbatch \
  /home/shanghun1404/v14_leg_collision/residual_jobs/174/model_100.pt 128 3000
```

[구현 계획](../../../docs/v14_residual_scale_plan.md), [코드 해시](code_sha256.json),
[이전 파일럿 결과](../residual/README.md).
기존 후보와 기준 모델을 보존했다. 큰 훈련 실행 자체를 성능 개선으로 해석하지 않는다.
