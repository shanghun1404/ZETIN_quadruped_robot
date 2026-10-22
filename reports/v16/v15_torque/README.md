# V15 선택 보행 모델 토크 진단

대상은 훈련 339의 `model_11699.pt`다. 훈련 403의 `model_16699.pt`는 채택하지 않았다.
모든 물리 평가·학습은 HPC `ws1`에서 수행했고, 각 Slurm GPU 작업 안에서 평가 조건을 순차 실행했다. 로컬에서는 저장된 수치 분석과 그림 작성만 한다.

**최종 선택은 원본339 그대로다.** 가드/LPF 후보5개, 추가 설정339@.97/23, 재훈련536과556 모두 변경하지 않은 제어·토크 기준에서 탈락했다. 두 재훈련 정책은 채택하지 않았으며 holdout과 추가 V15 GPU 작업은 제출하지 않는다. [전체 최종 판정](optimization/final_v15_torque_decision.json), [두 재훈련 비교](optimization/trained_model_comparison.md).

## 측정값의 의미

- `computed_torque_nm`: 위치 목표가 가드를 통과한 뒤, IsaacLab `ImplicitActuator`가 계산한 `Kp(qtarget-q)+Kd(vtarget-v)+ff`. actuator clipping 전 **추정값**이다. 정책의 원래 요구 토크를 뜻하지 않는다.
- `applied_torque_nm`: 위 추정값을 actuator 한도로 clip한 값. 이 평가에서는 PhysX implicit drive를 사용하므로 **실제 solver의 모터 출력 토크를 측정한 값이 아니다**. 그래프 제목과 범례에도 estimate를 표시한다.
- PhysX는 위치/속도 목표와 gain을 받아 내부 implicit PD 적분을 한다. IsaacLab 설치 원문은 implicit drive의 토크를 명시적으로 얻지 못해 근사 토크를 계산한다고 설명한다. 근거: [설치 소스 발췌](source/implicit_actuator_evidence.txt), [해시](data/source_sha256.txt).
- `get_dof_actuation_forces`는 외부에서 지정한 DOF actuation/feedforward force와 연관된 별도 API다. 이를 implicit drive의 실제 모터 토크로 바꾸어 부르지 않았다. projected joint force 역시 접촉·구속 반력을 포함할 수 있어 모터 출력으로 사용하지 않았다.
- 정격 continuous torque, 토크 상수, 권선 저항, 냉각, gear 효율/정격, 열 모델은 제공되지 않았다. **24.8/48 N·m는 이 시뮬레이터의 설정 최대 한도**이며 실제 모터의 연속 정격이라고 가정하지 않는다. peak가 이 한도 이하여도 열 안전은 입증되지 않는다.

## 프로토콜과 유효 구간

seed42, 32환경, `fixed_v2`, physics 200Hz(0.005s), policy 50Hz(0.02s), guard0.99,
25Hz 목표 LPF, 평가 action delay0, V14 leg collision USD, self collision/leg terrain filter 동일 조건이다.
선택 모델 SHA256은 `e2bb747acb4737f9e73dc34446cfae692ca2564ed36560fb1f137814eea46d33`이다.

| 조건 | 전진 명령 | heading | 지형 | 평가 전 준비 구간 | 본 평가 |
|---|---:|---|---|---:|---:|
| 평지 | 0.4m/s | 꺼짐 | plane | 2s | 12s |
| 계단 상승 | 0.5m/s | 켜짐 | L5, step15.7cm | 0s | 12s |
| 험지 | 0.5m/s | 켜짐 | L4 | 0s | 12s |

원본 `launch_v15_eval.sbatch`와 기존 339 결과의 조건을 유지했다. 준비 구간이 없는 지형에 새로운 2초 준비 구간을 넣어 시작 위치를 바꾸지 않았다.
분석은 `preroll`(평지의 평가 전 2초), `measured_first2s`(본 평가0–2초), `measured_after2s`(2–12초),
`measured_all`(0–12초)을 나눈다. 구간별 값은 [CSV](per_joint_segments.csv), [JSON](summary.json)에 있다.

`alive`는 최초 termination 이후 자동 reset된 환경을 제외한다. termination을 포함한 마지막 policy step의 physics frame은 포함되므로 경계에 최대20ms의 해상도 한계가 있다.
평지 준비 구간은 최초 episode가 끝난 뒤의 reset 데이터도 별도 `first_episode_alive` mask로 제거한다.
토크 slew는 같은 구간의 양끝 frame이 모두 alive인 연속 physics sample만 쓴다.

## 결과

**진단: V15의 반복 토크 요구를 줄이는 실험이 필요했다.** 가드 뒤 추정 토크는 설정 최대 한도를 넘지 않지만, 평지와 험지에서 첫 2초 이후에도 반복적으로 한도 근처를 사용한다. 후속 후보 중 동작과 부하 기준을 함께 충족한 설정은 없었다. 실제 모터의 연속 정격이 없으므로 열 안전이나 정격 초과 여부는 판정하지 않는다.

Slurm508은 `COMPLETED 0:0`, 경과15분45초, MaxRSS11.86GiB. 세 지형의 모든 기존339 seed42 actuation 항목과 생존/진행/속도 값이 정확히 재현됐고, HPC JSON/NPZ12파일의 SHA256이 로컬과 일치했다.

### 본 평가 2–12초, alive sample

| 지형 | 생존 | 본 평가 진행 | pooled p99 추정 | 80%이상 체류 | 99%이상 체류 | 가드 목표 변경 | 가드 전 한도 초과 | 관절별 평균 abs(τ·ω) | p99 slew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 평지 | 32/32 | 5.903m | 46.351N·m | 3.48% | 1.01% | 2.06% | 1.91% | 8.525W | 2776N·m/s |
| 계단↑ | 32/32 | 0.817m | 30.887N·m | 3.78% | 0.60% | 1.46% | 1.37% | 4.589W | 2440N·m/s |
| 험지 | 31/32 | 3.718m | 47.524N·m | 13.81% | 3.75% | 9.39% | 9.08% | 12.368W | 3665N·m/s |

`99%이상 체류`는 원본 지표 `applied_saturation_fraction`의 정의다. 실제 solver drive가 포화됐다는 관측은 아니다. 추정 actuator hard clipping과 가드 뒤 한도 초과는 세 조건 모두0%였다. 정책/필터의 가드 전 요구는 험지9.08% sample에서 설정 한도를 넘었고, 가드가9.39% 목표를 바꿨다.

평지에서도 RR_KFE의 2–12초 p99는47.533N·m, 99%한도 이상 체류5.52%다. 험지의 RR_KFE는 RMS26.967N·m(최대 한도56.18%), 80%이상 체류28.96%, 99%이상 체류10.08%이며 RL_KFE도99%이상9.29%다. 초기 준비/충돌 순간만으로 설명되지 않는 반복 부담이다.

### 험지 관절별 추정 토크(2–12초)

| 관절 | peak N·m | RMS N·m | p99 N·m | 80%이상 체류 | 99%이상 체류 |
|---|---:|---:|---:|---:|---:|
| FL_HAA | 24.575 | 13.663 | 24.553 | 21.05% | 4.94% |
| FL_HFE | 47.555 | 18.830 | 47.522 | 8.20% | 3.00% |
| FL_KFE | 47.586 | 27.016 | 47.525 | 23.11% | 3.39% |
| FR_HAA | 24.579 | 11.615 | 24.553 | 14.18% | 4.82% |
| FR_HFE | 47.557 | 20.907 | 47.524 | 11.49% | 3.38% |
| FR_KFE | 47.591 | 22.047 | 47.522 | 11.32% | 1.67% |
| RL_HAA | 24.558 | 10.717 | 24.552 | 11.25% | 1.31% |
| RL_HFE | 47.557 | 13.709 | 41.122 | 1.52% | 0.13% |
| RL_KFE | 47.578 | 24.023 | 47.526 | 19.33% | 9.29% |
| RR_HAA | 24.563 | 10.234 | 24.553 | 6.45% | 1.87% |
| RR_HFE | 47.547 | 19.454 | 47.521 | 8.89% | 1.17% |
| RR_KFE | 47.581 | 26.967 | 47.529 | 28.96% | 10.08% |

전체 관절·지형·준비/초기/이후 구간의 peak/RMS/p99와 power/slew는 CSV/JSON에 보관했다. 대표 환경은 평지20, 계단8, 험지11이다.

평가에서 실제 PhysX/IsaacLab gain은 모두300/8이고 가드 nominal gain과 정확히 같았다. 이번 평가의 gain 불일치는0이다. 훈련 gain 범위를 고정 관측 상태에 대입한 네 모서리 최댓값의 한도 초과 비율은 평지4.06%, 계단3.55%, 험지14.06%다. 이는 고정 gain의 재평가/발생 확률/실제 출력 관측이 아니라 sample별 대수적 worst-corner 민감도다.

가드/LPF 후보의 생존·명령 추종·진행·험지 방향을 함께 비교한 결과는 [최적화 기록](optimization/README.md)에 보관했다.

## 가드와 gain

[계측 overlay](../../../config/experiments/v16_20260929/v15_torque_capture.py)는 원본 evaluator의 SHA256을 확인하고
기록만 추가한다. 원본 파일, 모델, 환경 설정, 가드, 보상, action은 수정하지 않는다.
physics step의 `scene.write_data_to_sim` 이후/실제 stepping 전에 q, qdot, filtered 목표, 가드 후 목표, 두 추정 토크를 기록한다.

평가의 실제 PhysX stiffness/damping과 IsaacLab 데이터의 stiffness/damping을 함께 저장한다.
가드는 nominal Kp300/Kd8을 사용한다. V15 훈련에서는 startup에 각 gain을 U(0.8,1.2)로 바꾸므로,
훈련의 실제 gain과 nominal 가드 gain은 다를 수 있다. 이번 평가에서 이 randomization이 켜졌는지는 저장된 실제 gain으로 판단한다.

`preguard_over_limit_fraction`은 가드 입력(filtered 위치 목표)을 nominal PD로 환산한 값이 설정 최대 한도를 넘는 비율이다.
`guard_target_clip_fraction`은 목표 위치가 가드에 의해 실제로 바뀐 비율이다. `computed`만 보면 이 부담은 숨겨진다.
가드는 V13 DCMotor의 nominal torque-speed envelope(50rad/s)를 목표 위치에 적용하지만, PhysX implicit drive는 이 DCMotor 모델을 실행하지 않는다.
따라서 저장된 최대 effort와 목표 제한을 실제 모터의 torque-speed/thermal 성능 검증으로 해석하지 않는다.

`gain_corner_counterfactual_*`는 기록된 q/qdot/목표를 고정하고 훈련 gain 범위의 네 모서리(0.8/1.2)를 대입한 **대수적 민감도 분석**이다.
무작위 gain의 재평가를 실행하거나 해당 초과 토크를 실제로 관측한 결과가 아니다. 동작·접촉·정책이 달라지는 효과도 반영하지 않는다.

## 그림과 데이터

- [평지12관절](figures/flat_12joint_timeseries.png), [계단12관절](figures/stairs_up_12joint_timeseries.png), [험지12관절](figures/rough_12joint_timeseries.png): 살아남은 환경 중 2–12초 12관절 RMS가 중앙값에 가까운 환경을 대표로 고른다. 32환경의 alive sample p05–p95 band를 함께 표시한다.
- [평지32환경 분포](figures/flat_32env_distribution.png), [계단32환경 분포](figures/stairs_up_32env_distribution.png), [험지32환경 분포](figures/rough_32env_distribution.png): 관절별 환경 RMS/p99/한도80%이상 체류율. box는 환경 quartile, whisker는5–95 percentile이다.
- [관절 파워·slew 비교](figures/terrain_power_slew_comparison.png). 모든 그림의 같은 이름 PDF도 저장한다.
- [관절별 구간 집계](per_joint_segments.csv): peak/RMS/p99, 한도 대비 비율, 80/90/99% 체류율, clipping, 가드, mechanical power, slew.
- [환경별 관절 집계](per_environment_joint_segments.csv), [대표환경 시계열과32환경 band](representative_timeseries.csv), [생존·진행량](environment_outcomes.csv).
- raw physics trace/계측 trace/trajectory/평가JSON은 [data](data/)에 보관한다. [데이터 SHA256](data/data_sha256.txt), [소스 SHA256](data/source_sha256.txt)으로 HPC 원본을 확인한다.
- mechanical power는 관절별 `τ·ω`의 양/음/절대값이다. `mean(abs(τ·ω))`는 소비 전력이나 권선 발열을 직접 뜻하지 않으며 regenerative braking도 검증하지 않는다.

## 복구 모델416

기존 평가 집계를 먼저 사용했다: [요약JSON](recovery_existing_summary.json).
넘어진 시작256환경의 성공률은 시드42/43/44에서86.72/84.77/85.55%, 중앙 복구 시간1.40/1.42/1.42s.
선 시작은256/256이며 시드42 넘어진 평가는 비물리 상태 reset2회다. **기존 복구 평가에는 토크 metrics/trace가 없다.**
이 보고서로 복구 토크나 열 안전을 판정하지 않는다. 보행의 병목 진단에 필요한 GPU작업만 먼저 실행했다.

## 재현

HPC 준비 및 제출:

```bash
ssh -o ClearAllForwardings=yes ws1 'mkdir -p /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929'
scp -o ClearAllForwardings=yes config/experiments/v16_20260929/v15_torque_capture.py config/experiments/v16_20260929/v15_torque_eval.sbatch ws1:/home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/
ssh -o ClearAllForwardings=yes ws1 'sbatch /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/v15_torque_eval.sbatch'
```

본 평가 작업은 Slurm508, HPC 원본 디렉터리 `/home/shanghun1404/v15_large_scale/results/v16_20260929/v15_torque_508`.
GPU1/CPU4/16GB/40분, QoSadmin/compute. 한 allocation에서 세 평가를 순차 실행한다.
앞선506은 계측 중 CPU PhysX gain tensor에 GPU indices를 사용한 오류로4분52초 후 `FAILED 1:0`.
그 시점에는 평가 trace/수치가 생성되지 않았으며, [실패로그](data/failed506_flat.log)를 보관했다.
CPU getter의 계측 인덱스만 수정해508을 제출했다. 실패 원인은 물리 모델이나 actor, OOM이 아니다(MaxRSS10.1GiB).

저장된 데이터의 분석/그림 재작성(물리 실행 없음):

```bash
python3 reports/v16/v15_torque/analyze.py
```

분석기는 trace shape/시간 간격/finite/실제 effort 한도와 JSON 원본 집계(peak/p99/saturation/power/slew/alive sample 수)를 독립적으로 검증하고
불일치하면 산출 전에 중단한다. 자동 commit/push는 수행하지 않는다.

## 후속 최적화 결과

[후보5개 비교](optimization/README.md)는 생존/진행/추종/방향 및 토크 gate를 함께 적용했다.
모든 후보가 적어도 하나의 gate에서 탈락해 선정 후보가 없으며, 기준을 낮추거나 토크 값만으로 채택하지 않았다.
원본 V15 설정과339 actor를 보존했다. 부모 요청으로 [짧은 motor adaptation pilot 준비](optimization/finetune_plan.md)를 남겼고,
고정LR/optimizer 보존/원본hash/소스구성에 대한CPU4개검증은 통과했다. 준비 파일은 부모의 독립검토 뒤536(512환경/250회)으로 제출되어 COMPLETED0:0(11분50초)로 완료했고542에서 고정 동작·토크 기준을 평가했다. [536학습·평가](optimization/finetune536/README.md)에 기록했다.

## 두 적응 pilot의 최종 결과

536은학습을완료했으나542고정평가에서험지절대기계적joint power가508대비1.807%높아탈락했다. 설정만변경한339@.97/23도생존·경로·파워기준에실패했다. 두후보를채택하지않고holdout을제출하지않았다. [536/542최종표·CSV·PNG/PDF](optimization/finetune536/README.md).

원본339의guard.99/LPF25Hz를 유지하고 기존 토크L2노름 벌점만1.25배로 강화한 두번째 pilot556도 완료했다. 평가562에서 flat/rough 동작 gate는 통과했지만 토크 RMS는 각각1.743%/0.325%, 추정 절대 관절 파워는13.123%/4.127% 증가했고 한계부하 개선 기준도 실패했다. **536과556을 모두 채택하지 않고 원본339를 유지한다.** [556/562 결과와 재사용508 계약](optimization/reward_finetune/README.md).

[두 모델 비교표·CSV·JSON·PNG/PDF](optimization/trained_model_comparison.md)와 [최종92개 원자료 해시 검증](optimization/final_raw_data_verification.json)을 남겼다. seed43/44 holdout은 실행하지 않았고 추가 V15 GPU 작업을 제출하지 않는다. 모든 토크·파워는 PD 추정치이며 실제 solver motor torque와 continuous 정격·열 안전은 입증되지 않았다.
