# V15 보행 토크 최적화 비교

**최종 결과:** 아래 가드/LPF 후보5개와 추가339@.97/23, 학습536·556이 모두 기존 gate에서 탈락했다. 새 정책은 채택하지 않으며 원본339를 유지한다. [전체 판정 JSON](final_v15_torque_decision.json), [재훈련 비교](trained_model_comparison.md). holdout과 추가 V15 GPU 작업은 미제출이다. 아래 설계와 holdout 절차는 실험 당시 계획이며, 통과 후보가 없어 holdout은 실행하지 않았다.

baseline의 평지 본 평가2–12초에서도 RR_KFE p99가47.533N·m이고 한도99%이상 체류5.523%였다.
첫 순간만의 부담이 아니므로, 모델339를 고정하고 목표 LPF/가드 조합을 비교한다. 원본 V15 설정/평가 코드/actor는 수정하지 않는다.

## 설계와 실행 순서

기존 가드/LPF 흐름에 대한 한정된 실험이다. 목표는 **생존·전진·명령 추종·험지 방향을 유지하면서 반복 토크 요구를 낮추는 것**이다.
이 기준을 통과하지 못한 후보는 토크 값이 낮아도 채택하지 않는다. 사용자 결정 위임에 따라 승인 재요청 없이 수행한다.

| 후보 | guard fraction | LPF cutoff |
|---|---:|---:|
| baseline | 0.99 | 25Hz |
| g095_lpf15 | 0.95 | 15Hz |
| g090_lpf25 | 0.90 | 25Hz |
| g090_lpf10 | 0.90 | 10Hz |

1. 기존 조건으로 baseline을 재현한다(진단508).
2. 한 GPU allocation에서 세 후보×평지/험지 seed42를 순차 실행한다(516).
3. 원본 JSON과 physics trace에서 gate와 실제 적용 cutoff/guard/alpha를 확인한다.
4. 모든 후보가 완료된 뒤 gate 통과 후보를 비교한다. 선정은 holdout 전 잠정이다.
5. 통과 후보가 있으면 seed43/44에서 제어 성능을 다시 검증한다. 험지는 같은 시드 baseline이 없어 baseline도 실행한다.

## 후보 채택 gate

실험 전에 기록한 [manifest](../../../../config/experiments/v16_20260929/v15_torque_candidates.json) 기준이다. **모터 정격이나 안전 기준을 뜻하지 않는다.**

두 지형 모두:

- 생존 환경 수≥baseline(평지32/32, 험지31/32).
- 평균 directed progress≥baseline90%(seed42 평지5.313m, 험지3.346m).
- 명령 대비 평균 전진 속도 오차≤baseline 오차+0.02m/s.
- 험지 corridor 유지 환경 수≥baseline, 환경별 최대 heading error 평균≤baseline+2°.
- 2–12초 alive sample의 관절 RMS 평균과 `mean|τ·ω|`가 baseline보다 증가하지 않아야 한다.
- normalized peak, 관절별 p99 평균, 한도80%이상 체류율 중 적어도 하나가1%이상 감소해야 한다.

gate를 통과한 후보만 RMS40%/절대 mechanical power25%/80%체류25%/p99slew10%의 baseline 대비 비율을 평균해 비교한다.
가중치는 후보 간 비교를 위한 실험 판단이며 열 모델을 대체하지 않는다. 원본 baseline은 과속(평지0.493m/s, 명령0.4)하고
험지에서는 명령0.5m/s 대비0.317m/s이므로, 이 비교가 절대적 명령 추종 성능 통과를 뜻하지도 않는다.
평균 감소가 모든 관절의 감소를 뜻하지 않아 관절별 RMS 재분배도 함께 기록한다. 첫 후보의 험지 RL_HAA RMS는 약14% 증가했으며,
해당 후보는 속도 gate에서도 탈락했다. 실제 모터별 연속 정격이 없으므로 관절 간 부하 이동을 열 안전으로 해석하지 않는다.

## 사전 선언한 후속 중간 후보

초기 세 후보가 전부 탈락하는 경우에만 guard0.95/LPF20Hz를 평지와 험지에서 평가한다.
15Hz 후보의 험지 속도 저하와25Hz 후보의 험지 파워 증가 사이에서 확인할 중간 설정이라는 **가설**이다.
20Hz의 실제 성능을 예측한 결과로 취급하지 않는다. 첫 중간 후보가 원래 모든 gate를 통과하면 추가 실험을 생략하고,
실패하면 guard0.97/LPF20Hz를 동일 조건으로 평가한다. 두 경우 모두 기준은 위와 같다.

[후속 manifest](../../../../config/experiments/v16_20260929/v15_torque_refinement_candidates.json)와
[조건부 실행기](../../../../config/experiments/v16_20260929/v15_torque_refinement.sbatch)에 이 순서를 고정했다.
GPU1/16GB/최대40분의 한 작업 안에서 최대4개 평가를 순차 실행하며, 부모의 V16 pilot528 이후 의존성을 연결한다.
기존516 실행 파일은 바꾸지 않고 별도20Hz overlay를 사용한다. 모든 데이터/소스와 실행 중 판정 JSON에 해시를 기록한다.
CPU 판정기의 ratio 계산은 기존 집계기의 실제15Hz 데이터와 독립 대조해1e-12 상대 오차 이내로 일치했다.
`analyze_sweep.py`는 사전 선언한 조건부 생략을 기록하며, 실행하지 않은 후보의 결과를 만들어 넣지 않는다.

## 원본과 실제 설정 검증

[후보 overlay](../../../../config/experiments/v16_20260929/v15_torque_candidate_capture.py)는 hash가 고정된 baseline 계측기/원본 evaluator를 읽고,
새 evaluator의 메모리 안에서 `env_cfg.actions.joint_pos.cutoff_hz`만 설정한다. guard는 검토된 CLI 인자를 사용한다.
매 결과에 `actual_cutoff_hz`, `actual_guard_fraction`, `actual_lpf_alpha`, baseline/candidate 소스 SHA256을 기록한다.
기본 PD gain300/8, PhysX implicit plant, guard speed50rad/s, delay0, actor/student/USD, 지형/heading/seed/기간을 유지한다.
guard fraction은 속도에 따른 양/음 torque envelope의 반폭에 적용된다. envelope의 midpoint가 속도에 따라 이동하므로
`guard0.95`가 모든 sample에서 정적 peak limit의95%를 넘지 않는다는 뜻은 아니다. 실제 관측한 peak/한도 비율을 별도로 비교한다.

## 결과와 한계

초기 작업516은31분59초에 정상 종료했다. 세 후보 모두 전체 채택 gate에서 탈락했으며 선정 후보는 없다.

| 후보 | 지형 | 생존 | 진행 m | 속도 m/s | RMS평균 변화 | 추정 파워 변화 | p99slew 변화 | 탈락 기준 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| g095_lpf15 | flat | 32/32 | 5.753 | 0.475 | -2.20% | -7.14% | -18.62% | 통과 |
| g095_lpf15 | rough | 32/32 | 3.414 | 0.285 | -3.29% | -9.08% | -20.19% | command_tracking |
| g090_lpf25 | flat | 32/32 | 5.840 | 0.486 | -0.50% | -0.21% | -1.08% | 통과 |
| g090_lpf25 | rough | 32/32 | 3.858 | 0.322 | -3.36% | +0.80% | -0.20% | power_not_increased |
| g090_lpf10 | flat | 32/32 | 5.616 | 0.460 | -3.96% | -13.59% | -36.87% | 통과 |
| g090_lpf10 | rough | 32/32 | 3.187 | 0.266 | -2.89% | -9.27% | -30.70% | progress, command_tracking |

토크 관련 변화는2–12초 alive sample 기준이다. g095_lpf15는 험지 추종에 실패했고,
g090_lpf25는 험지 평균 추정 파워가0.80% 증가했다. g090_lpf10은 험지 진행과 추종에 실패했다.
또한 g090_lpf25의 한도80%이상 체류율은 평지15.18%/험지8.53% 증가했다. 낮은 peak만으로 부하 개선을 단정하지 않는다.
모든 후보의 한도99%이상 체류는0%지만, 동작과 부하 기준을 함께 통과한 후보는 없다.

작업525는18분37초에 정상 종료했다(최대RSS10.97GiB). 처음 `afterany:522`로 제출했고, 부모가 V16 probe/pilot을 교체한 뒤 `afterany:528`로 변경했다.
초기3개와 중간2개 후보 모두 고정 gate를 통과하지 못해 선정 후보가 없고 holdout은 제출하지 않았다.
[최종5후보 집계](final_candidate_summary.json)에 실제 성공/실패를 보존했다.

### 중간 후보 실측

`g095_lpf20`은 평지32/32·5.838m·0.486m/s로 모든 gate를 통과했지만,
험지32/32·3.463m·0.289m/s로 명령 추종 기준(최소0.297m/s)을 통과하지 못했다.
험지 corridor 유지도31개로 baseline32개에 못 미쳤다. 험지2–12초 RMS평균1.22%/추정파워4.72%/p99slew11.38% 감소에도 채택하지 않는다.
험지 한도80%이상 체류는5.04% 증가했고 RR_HAA RMS는9.97% 증가했다.
사전 조건에 따라 동일525 allocation에서 `g097_lpf20`을 이어서 평가했다.
이 후보도 평지32/32·5.850m·0.487m/s는 통과했지만, 험지31/32·3.442m·0.293m/s와corridor31개로 추종/corridor 기준을 통과하지 못했다.
험지 RMS평균1.55%/추정파워3.03%/p99slew9.40% 감소에도 채택하지 않는다. FL_HFE RMS는10.11% 증가했다.
[후속 집계](refinement/screening_summary.json)와 [PNG](refinement/screening_torque_comparison.png)/[PDF](refinement/screening_torque_comparison.pdf)에 실측을 기록했다.

모든 토크·파워·slew는 IsaacLab의 implicit PD **추정치**다. 실제 solver motor torque/전기 소비/continuous 정격/열 안전은 검증되지 않았다.
LPF와 guard의 조정은 제한된 후보 실험이며, 기구·모터 모델 변경이나 actor 재훈련은 포함하지 않는다.
후보 비교/holdout은 평지와 험지에 한정된다. 계단의 최적화 후 성능은 별도 검증 없이 일반화하지 않는다.

## 산출물과 재현

- `screening_summary.json`: gate별 판정, baseline 대비 비율, 실제 설정 검증, 모든 후보 완료 여부, 잠정 선정.
- `screening_metrics.csv`, `screening_joint_metrics.csv`: 제어 결과와 관절별 토크 수치.
- `screening_torque_comparison.png/.pdf`: baseline+후보의 지형별12관절 RMS/체류율/파워.
- `data/`: HPC 결과 원본과 manifest/해시.

HPC 작업516(완료) 및525(후속): GPU1/CPU4/16GB/40분, compute/QoSadmin, `afterany:510`으로 stand probe 뒤 실행.
HPC 디렉터리 `/home/shanghun1404/v15_large_scale/results/v16_20260929/v15_torque_sweep_516`.

```bash
ssh -o ClearAllForwardings=yes ws1 'sbatch --dependency=afterany:510 /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/v15_torque_sweep.sbatch'
python3 reports/v16/v15_torque/analyze_sweep.py
```

Slurm 의존 작업 번호는 재실행 시 실제 stand probe 작업에 맞춘다. baseline와 sweep 원본의 SHA256을 맞춘 후 분석한다.
자동 commit/push는 수행하지 않는다.

## seed43/44 holdout 기준

선정 설정을 바꾸지 않고 평지/험지 두 시드에서 위 제어와2–12초 토크 gate를 모두 유지한다.
기존 평지 baseline JSON에는 RMS가 없으므로 flat/rough 모두 동일 시드 baseline trace를 새로 생성한다.
시드별4조건(baseline flat/rough, 후보 flat/rough)을 한 allocation에서 순차 실행하고, 두 GPU 작업은 직렬 연결한다.
GPU1/16GB/40분 한도에서8개 평가를 한꺼번에 실행하지 않는다. 후보 선정 뒤 부모의 V16 학습과 순서를 조율한 후 제출한다.
추가로 모든 조건에서 전체0–12초 추정 파워 증가 없음, p99 또는 normalized peak1%이상 감소,
p99slew 증가5%이하를 요구한다. [holdout protocol](../../../../config/experiments/v16_20260929/v15_torque_holdout_protocol.json)에 고정했다.
이 결과도 모터 연속 정격이나 solver 실측을 대신하지 않는다.

후속 결과 원본: `/home/shanghun1404/v15_large_scale/results/v16_20260929/v15_torque_refinement_525`.
초기516의 JSON/trace/보충trace/trajectory24개 SHA256은 모두 로컬 복사본과 일치했다.

## 후속 실험 설정 적용과 재현 명령

20Hz 후보는 [별도 overlay](../../../../config/experiments/v16_20260929/v15_torque_refinement_capture.py)에서
`env_cfg.actions.joint_pos.cutoff_hz=20.0`을 환경 생성 전에 적용하고,
`--effort-guard-fraction 0.95` 또는 `0.97`을 원본 evaluator에 전달한다.
`V15_TORQUE_CANDIDATE`는 결과 label, `V15_TORQUE_CUTOFF_HZ=20`은 overlay의 cutoff 입력이다.
실행기의 case 문에 후보별 실제 CLI 값이 고정되어 있고, 결과 JSON의 actual 필드로 다시 대조한다.
현재 선택되지 않은 설정을 기본 배포값으로 바꾸는 명령이 아니다.

```bash
scp -o ClearAllForwardings=yes config/experiments/v16_20260929/v15_torque_refinement* \
  config/experiments/v16_20260929/v15_torque_capture.py \
  ws1:/home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/
ssh -o ClearAllForwardings=yes ws1 'sbatch --dependency=afterany:528 /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/v15_torque_refinement.sbatch'
python3 reports/v16/v15_torque/analyze_sweep.py \
  --manifest config/experiments/v16_20260929/v15_torque_refinement_candidates.json \
  --sweep-dir reports/v16/v15_torque/optimization/refinement/data \
  --output-dir reports/v16/v15_torque/optimization/refinement
```

재현 시 Slurm 의존 번호는 실제 선행 작업으로 바꾼다. 물리 평가에는 HPC/Slurm을 사용한다.
로컬 명령은 저장된 trace 분석만 수행한다. 선정 근거는 gate별 결과와 관절별 CSV이며, peak 감소만으로 선정하지 않는다.

## 다음 조치와 준비 상태

고정 gate를 낮추지 않는다. 가까운 guard0.97/LPF23Hz 두 지형 screening은20Hz와baseline25Hz 사이를 확인할
한정된 다음 가설이며 통과를 예측하는 실측 결과가 아니다. 20Hz 마지막 후보의 속도 부족은0.00366m/s였지만
corridor도1개 부족하므로 속도만 맞추면 해결된다고 볼 수 없다.

현재 부모는 V16 물리 모델 검증을 우선한다. V15 추가 GPU 실험은 제출하지 않았다.
부모 요청으로 [짧은 motor adaptation pilot 계획](finetune_plan.md)과
[별도 trainer overlay](../../../../config/experiments/v16_20260929/v15_torque_finetune.py)/
[Slurm 준비 파일](../../../../config/experiments/v16_20260929/v15_torque_finetune.sbatch)을 작성했다.
이 파일은339를 guard0.97/LPF23Hz에 적응시키며 고정LR3.375e-5를 algorithm과optimizer에 함께 적용한다.
원본 환경/보상/episode와 기존finite/fall 방어를 유지하는1차 적응 실험이다. 준비/CPU검증과부모독립검토뒤536(512환경/250회)으로제출되어COMPLETED0:0(11분50초)로완료했다. 학습checkpoint의유한성/고정LR/해시는검증했으며보행/토크gate는아직미평가다.
기존 torque_pressure/joint_power 보상이 비활성이므로 이 pilot이 토크 부담을 직접 최소화하거나 줄일 것으로 보장하지 않는다.
채택은 재평가/holdout의 동작과 토크 gate를 통과한 실제 결과에만 근거한다.

## 536/542 적응 결과와 다음 실험

536은학습을완료했으나542고정평가에서험지절대기계적joint power가508대비1.807%높아탈락했다. 설정만변경한339@.97/23도생존·경로·파워기준에실패했다. 두후보를채택하지않고holdout을제출하지않았다. [536/542최종표·CSV·PNG/PDF](finetune536/README.md).

부모가원본339의guard.99/LPF25Hz를유지하고기존토크L2벌점만1.25배로강화하는두번째pilot계획을승인했다. 별도wrapper/실행기와평가변형을준비하고CPU검증을마쳤으며,현재부모검토/GPU순서조율중이다. [두번째적응](reward_finetune/README.md).
