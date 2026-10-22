# V15 motor adaptation536

**최종 상태:** 학습536과 평가542 모두 정상 완료했다. 적응536은 험지 추정 절대 관절 파워+1.807%로 탈락하여 채택하지 않는다. 후속 토크 벌점 pilot556도 평가562에서 탈락했고, 원본339를 유지한다. [두 모델 최종 비교](../trained_model_comparison.md). 아래 진행·부분 결과는 당시 기록이며 holdout과 추가 V15 GPU 작업은 미제출이다.

부모독립검토뒤512환경/250회로536을제출했고실제actual contract와COMPLETED0:0(11분50초,MaxRSS13.19GiB)을확인했다.
현재확인한LR는PPO멤버/optimizer모두3.375e-5/fixed, restoredstate17개,
guard0.97/LPF23Hz/alpha0.4194708244, 시작iteration11700, rollout24/episode20s/delay0–3다.
Kp실제범위240.018–359.983, Kd6.400–9.599이며기존Finite wrapper/physics_diverged종료를유지했다.
학습중Diagnostics/nonfinite_steps0을확인했으며이는고정지형보행/토크gate의통과를뜻하지않는다.

## 동등 평가542 진행

원본339@guard0.97/23Hz와536final11949@같은설정을각각flat/rough32환경seed42/fixed_v2/12초로평가한다.
평지warmup2초/명령0.4/headingoff, 험지L4/warmup0/명령0.5/headingon을유지한다.
동작·토크acceptance gate의공통reference는기존339@guard0.99/25Hz의진단508이다.
두23Hz모델간차이는정책적응의효과로별도비교하되,새모델을채택하려면공통fixedgate도통과해야한다.

부모독립검토에서Critical/Important없음을확인한뒤GPU1/CPU4/16G/40분실행기를전송하고542를제출했다. 현재RUNNING이며4조건을직렬실행한다.
완료536의marker/std/nonfinite/실제LR/guard와최종checkpoint SHA를검증한뒤,
모델별SHA가다른runtime manifest를생성한다. baseline/eval overlay SHA와훈련source/marker/contract,
새평가JSON/NPZ hash를구분해보존한다. 원본evaluator와capture는고정SHA를사용한다.

준비파일: config/experiments/v16_20260929/v15_torque_finetune_eval*.
제출전CPU3개입력계약검증,두label의23Hz소스구성/AST검증,bash-n검증을통과했다. 현재물리평가는HPC의Slurm542에서진행중이다.

```bash
python3 reports/v16/v15_torque/analyze_sweep.py \
  --manifest reports/v16/v15_torque/optimization/finetune536/eval_data/v15_torque_finetune_eval_manifest.json \
  --sweep-dir reports/v16/v15_torque/optimization/finetune536/eval_data \
  --output-dir reports/v16/v15_torque/optimization/finetune536/evaluation
```

분석기는각candidate의기대한actor SHA를검증하며,기존5후보는원래339SHA검증을유지한다.
학습·평가토크는implicit PD추정치이며실제solver motor torque/continuous정격/열 안전은미확인이다.

## 최종학습검증

최종model11949 SHA256은`06ca0ec3fadd743c1033a75ccf133193e3ad0432f3112b79698467ac0efd4e7a`다.
CPUcheckpoint검사에서모든policy/optimizer tensor가유한하고optimizer LR3.375e-5,
std범위0.113475–0.141864와iteration11949를확인했다. 최종nonfinite_steps는0이다.
HPC completion/contract/manifest와4개checkpoint의data SHA를모두대조했다.
[CPU확인](checkpoint_cpu_verification.json),[실제536입력계약확인](eval_input_readonly_verification.json),[상태](status.json).

평가입력검증기는실제536을HPC에서읽기전용으로검사해통과했다. 이검사에서는파일을쓰거나물리평가를실행하지않았다.
542의실제preflight가완료536marker/checkpoint/contract를검증하고두모델SHA를각각바인딩했음을확인했다.
536은sacct에서COMPLETED이지만scheduler의완료기록이이미제거되어afterok의dependency오류가발생했고,부모가dependency없이542를제출했다. wrapper의실제완료536입력검증은유지했다.
원본339와536의보행/토크고정gate는542의4조건이모두완료된뒤분석한다. Runtime manifest의template prepared 문구는준비시점기록이며,현재실행상태는status.json에기록한다.

## 542 부분 결과: 원본339 설정변경

첫2조건의8개원자료SHA검증과실제guard0.97/23Hz/alpha0.4194708244/gain3008확인을마쳤다.
평지는생존32/32,진행5.86783m,속도0.488747m/s이며기존행동/토크gate를통과했다.
험지는생존30/32<31,경로유지29/32<32,after2s절대joint power12.9561W(+4.759%)로전체gate를실패했다.
험지속도0.334843m/s,진행3.97203m,RMS−2.692%,한계80%이상시간−3.247%,peakfraction0.974468만으로채택하지않는다.
새536정책의2조건은진행중이며최종선정은아직없다. [부분분석](evaluation/screening_summary.json).

## 542 부분 결과: 적응536 평지

새536의평지는생존32/32,진행5.322419m,속도0.436953m/s로고정gate를통과했다. 진행은90%기준5.312737m대비0.009682m의여유가있다。
508대비after2s RMS−9.999%,절대joint power−5.311%,80%이상시간−24.823%,peakfraction0.973558.
slew+2.776%,RL_HAA RMS+17.858%,거리당추정절대에너지+5.710%는증가했으므로모든부하가줄었다고표현하지않는다.
같은설정339대비RMS−9.556%/power−4.588%,slew+5.370%. 마지막rough조건완료뒤전체gate를판정한다.

학습pilot.log도HPC와SHA를대조하고250회모든reward/value/surrogate/entropy/std가유한함을확인했다.
최종평균reward289.73,physics_diverged/nonfinite는250회모두0이다. 평가지표를대신하지않는다. [로그숫자확인](training_log_verification.json).

## 542 최종 판정

542는 **COMPLETED 0:0 / 18분35초 / MaxRSS10.41GiB**로 완료했다. 위 준비·진행·부분결과 문단은 해당 시점 기록이다. 네 조건의16개원자료 SHA와 HPC의모든frozen source/model checksum을검증했다.

| 정책·설정 | 지형 | 생존 | 경로 유지 | 진행 m | 속도 m/s | RMS 변화 | 절대joint power 변화 | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 339 / .97·23Hz | flat |32|32|5.867826|.488747|−.490%|−.758%|통과|
| 339 / .97·23Hz | rough |30|29|3.972032|.334843|−2.692%|+4.759%|생존·경로·파워 실패|
| 536 / .97·23Hz | flat |32|32|5.322419|.436953|−9.999%|−5.311%|통과|
| 536 / .97·23Hz | rough |32|32|3.931280|.328642|−2.601%|+1.807%|파워 실패|

변화율은 기존339@.99/25Hz 진단508의 살아있는 after2s 샘플 대비다. 새536의험지에서는 행동gate가모두통과했지만추정절대joint power12.59098W가원본12.36750W보다높아기존토크gate를실패했다. 따라서 두후보모두채택하지않고선정null/holdout미제출로유지한다. 같은 .97/23설정의학습전후관측에서는험지생존30→32/경로29→32와파워감소를보였지만원래수준은회복하지못했다. 생존이달라평균에포함되는유효샘플집합도달라지며한seed로일반화효과를입증하지않는다.

[최종screening JSON](evaluation/screening_summary.json), [CSV](evaluation/screening_metrics.csv), [12관절CSV](evaluation/screening_joint_metrics.csv), [PNG](evaluation/screening_torque_comparison.png), [PDF](evaluation/screening_torque_comparison.pdf), [동일설정학습전후](evaluation/policy_adaptation_same_settings.json), [16개데이터해시검증](eval_data_hash_verification.json).

부모가다음적응계획을승인했다. 원본339의guard.99/LPF25Hz를유지하고기존토크L2노름벌점만1.25배로강화하는별도실험이다. [다음계획](second_adaptation_plan.md). 536/542파일및원본V15는보존한다. 어떠한후보도실제solver motor torque/열안전을입증하지않았다.
