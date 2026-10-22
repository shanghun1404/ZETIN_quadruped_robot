# 두 번째 V15 토크 보상 적응

원본 339에서 guard0.99·LPF25Hz를 유지하고 기존 joint_torques L2 노름 벌점만 1.25배로 강화했다. [실측 실패와 계획](../finetune536/second_adaptation_plan.md). 별도 파일의 CPU 계약 8개, 소스 구성/AST, bash-n 및 부모·독립 검토 후 Slurm556을 제출했다. **556은 COMPLETED0:0(12분, MaxRSS13.05GiB), 평가562는 COMPLETED0:0(9분31초, MaxRSS10.06GiB)다. 새 정책은 두 지형 모두 토크 기준에 탈락하여 채택하지 않는다. 원본 339를 유지하고 holdout 및 추가 V15 GPU 작업은 제출하지 않는다.**

| 항목 | 설정 |
|---|---|
| 입력 | frozen339 model11699, 학생·USD도 고정SHA 확인 |
| 실제 토크 벌점 | 기존−0.0005 → −0.000625, 함수는12관절 추정 applied_torque의L2 노름 |
| Plant | 원래guard0.99 / LPF25Hz / alpha0.4399008465 |
| PPO | algorithm·모든optimizer LR3.375e-5, fixed, moments복원 |
| 규모 |512env /250iter /seed47 /최종11949 |
| 기존 유지 항목 |rollout24, physics.005/control.02, episode20s, delay0–3, gainU(.8,1.2), finite/fall방어 |
| 파워 관련 reward |joint_power/torque_pressure는기존처럼None |
| 자원 |GPU1/CPU4/16G/40분/compute·admin |

`v15_torque_reward_finetune.py/.sbatch`는536의검증된wrapper를별도파일로복사하여기존plant를복원하고위reward만변경한다. 기존함수/weight/sourceSHA를확인하고, 실제reward manager의모든active weight가변경후config와일치하는지도기록·검증한다. 학습전후모든원본·입력·overlay의SHA를확인한다. 카메라는동결student의실제depth관측에필요해유지한다.

`v15_torque_reward_eval*`는완료된training job ID를인수로받으며총2조건을직렬평가한다. marker/model11949SHA·finite/std/nonfinite·실제LR/guard/LPF/alpha·토크reward·다른reward불변·검토된wrapperSHA를확인한뒤새정책만flat/rough32env seed42 fixed_v2/12s로평가한다. 원본339@.99/25의실제508자료를재사용한다. 평가source와train source/marker/모델·원자료SHA를구분한다.

아래는 이미 완료한 작업의 재현 명령이다. 현재 추가 제출 계획은 없다:

```bash
ssh -o ClearAllForwardings=yes ws1 \
 'sbatch /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/v15_torque_reward_finetune.sbatch 512 250'
# 실제 완료556 marker/model/contract 검증을 필수로 수행한다.
ssh -o ClearAllForwardings=yes ws1 \
 'sbatch /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/v15_torque_reward_eval.sbatch 556'
```

기존508 기준을 유지했다. 두 지형 모두 생존·진행≥90%·명령오차·험지 경로/heading·RMS/절대파워 비증가·한계부하 개선 gate를 통과해야 채택 후보가 된다. seed43/44 holdout은 screening 통과를 전제로 준비했지만 **gate 실패로 실행하지 않았다**. 순위를 torque 단독으로 정하거나 기준을 낮추지 않았다.

벌점은L2노름이며토크제곱/파워/열손실의직접벌점이아니다. 모든토크·파워는implicit PD추정이며실제solver motor torque와모터continuous정격/열안전은미확인이다. [해시와상태](status.json).

## 508 기준선 재사용 확인

부모 지시에 따라 동일 원본339 조건을 다시 실행하지 않는다. baseline508의flat/rough 원자료8개와모델·학생·USD·evaluator·actuator/articulation·action/mdp/bridge·recorder 소스11개의총19개 SHA를HPC에서읽기전용으로대조했다. [HPC 대조](baseline508_HPC_readonly_reuse_verification.json).

기존trace의실제PhysX gain300/8과guard.99를확인했고,LPF25Hz alpha0.4399008465의갱신식최대오차는flat5.96e-8/rough1.19e-7rad였다. [trace 물리계약](baseline508_reuse_verification.json). delay0/reset/fixed_v2는동결evaluator 소스로확인한다. 새평가의실제delay0과metadata(지형origin32개·terrainL4·seed42·heading·warmup·dt·입력·collision)를원본과비교한뒤gate를계산한다. 비교에는새actor만바뀐다. GPU의서로다른물리장치에서실행한자료이므로비트단위동일시뮬레이션을주장하지않는다. 각새평가종료후508의19개 SHA도다시확인한다.

## 556 완료·562 평가

실제reward/LR/guard/alpha/source 계약을확인한후학습이완료됐다. 최종model11949 SHA는`3bd366c775e6bc347bec28fb6716ad54708ddc80633643090734261cb2f93a1e`다.
최종policy와optimizer모든tensor가유한하고savedLR3.375e-5,std0.111672–0.146007/iteration11949를CPU로확인했다. [CPU모델확인](checkpoint_cpu_verification.json). completion/actualcontract/manifest와4개model의7개데이터SHA를HPC및복사본에서대조했다. 학습250회reward·loss는모두유한하고nonfinite/physics_diverged는0이다. [학습로그숫자](training_log_verification.json).

실제 완료556과508 재사용19SHA를 검증한 읽기 전용 eval binding이 통과했다. [실제입력계약](eval_input_readonly_verification.json). Slurm562의 preflight에서 `reward_training_verified:true`를 확인했으며, 두 평가가 정상 완료됐다. 최신 상태와 선정 여부는 [status.json](status.json)에 기록한다.

## 562 최종 평가: 채택하지 않음

변화율은 원본339@.99/25Hz의 완료508 대비다. 동작 지표는 본 평가 0–12초, 토크 지표는 본 평가 2–12초의 최초 종료 전 alive sample을 사용한다.

| 지형 | 생존 | 진행 m | 속도 m/s | RMS 변화 | 절대 관절 파워 변화 | 80% 이상 체류 변화 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| flat |32/32|5.746468|0.483491|+1.743%|+13.123%|+10.474%|동작 통과, 토크 3개 기준 실패|
| rough |31/32|3.868222|0.326450|+0.325%|+4.127%|+3.115%|동작 통과, 토크 3개 기준 실패|

flat의 추정 절대 관절 파워는9.64355W/관절, rough는12.87792W/관절이다. normalized peak는 flat0.990526/rough0.993381이며, 한도99% 이상 체류는 각각0.6333%/3.9542%다. normalized peak·관절별 p99 평균·80% 체류 중 어느 것도 두 조건에서1% 개선되지 않았다. flat FR_KFE RMS는58.460%, rough FL_HFE는19.685% 증가해 관절 간 부하 이동도 관측됐다. flat 경로 유지13/32는 참고 지표이며, heading off 평지에 경로 기준을 새로 추가하지 않았다. rough 경로는32/32로 원래 기준을 통과했다.

토크 벌점을 강화했어도 이 250회 pilot의 고정 평가에서는 요구 부하가 감소하지 않았다. 학습 reward 증가나 유한한 checkpoint를 동작·토크 개선의 증거로 사용하지 않는다. 한 seed 결과이며, 다른 학습 기간·계수의 효과를 입증하거나 배제하는 결과도 아니다.

새 원자료8개의 SHA가 HPC와 로컬에서 일치한다. 모든 frozen source/model과 재사용508의19개 SHA도 평가 종료 후 다시 대조했다. [데이터 검증](eval_data_hash_verification.json), [최종 HPC 검증](final_HPC_verification.json), [screening JSON](evaluation/screening_summary.json), [CSV](evaluation/screening_metrics.csv), [12관절 CSV](evaluation/screening_joint_metrics.csv), [PNG](evaluation/screening_torque_comparison.png), [PDF](evaluation/screening_torque_comparison.pdf).

원자료로 분석만 재작성하는 명령(물리 실행 없음):

```bash
python3 reports/v16/v15_torque/analyze_sweep.py \
  --manifest reports/v16/v15_torque/optimization/reward_finetune/eval_data/v15_torque_reward_eval_manifest.json \
  --sweep-dir reports/v16/v15_torque/optimization/reward_finetune/eval_data \
  --output-dir reports/v16/v15_torque/optimization/reward_finetune/evaluation
```

## 단위와 해석 한계

토크는 N·m, slew는 N·m/s다. 파워는 추정 `mean(abs(τ·ω))`의 관절 평균 W이며 전기 소비·권선 발열이 아니다. configured effort24.8N·m(HAA)/48N·m(HFE·KFE)는 시뮬레이션 최대 한도이고 continuous 모터 정격이 아니다. computed/applied는 각각 clipping 전/후 implicit PD 추정치다. 실제 solver motor torque, 모터 정격·열 상수·냉각이 없으므로12초 결과나 peak 한도 이내 여부로 열 안전을 입증하지 않는다.
