# V15 재훈련 두 모델의 최종 비교

**원본339 model11699를 유지한다. 536과556은 모두 채택하지 않는다.** 최초 설정 후보5개와339@.97/23도 동일 기준에서 실패했다. 선정 후보는 null이며 holdout과 추가 V15 GPU 작업을 제출하지 않는다.

공통 기준선은 완료508의 원본339@guard0.99/LPF25Hz다. 두 학습은 각각 원본339에서 시작했으며512환경·250회·seed47·고정LR3.375e-5·optimizer moments 복원을 유지했다. 536은guard.97/LPF23Hz에서 기존 보상으로 적응했다. 556은원래guard.99/LPF25Hz에서 기존 토크 L2노름 벌점만−.0005→−.000625로 강화했다. 파워 reward를 추가하지 않았다.

동작 지표는32환경·seed42·fixed_v2·본 평가12초의 원본 집계다. 토크 지표는 본 평가2–12초 alive sample이다. flat 준비2초와 본 평가 최초2초는 구분했다. 재사용508의 원자료와 입력·소스·gain·delay·terrain·heading·warmup 계약을 확인했다.

| 학습 / 평가 | 설정 | 지형 | 생존 | 진행 m | 속도 m/s | RMS 변화 | 절대 관절 파워 변화 | 80% 이상 체류 변화 | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|536 /542|.97·23Hz|flat|32/32|5.322419|0.436953|−9.999%|−5.311%|−24.823%|해당 지형 통과|
|536 /542|.97·23Hz|rough|32/32|3.931280|0.328642|−2.601%|+1.807%|−2.610%|파워 실패|
|556 /562|.99·25Hz, torque×1.25|flat|32/32|5.746468|0.483491|+1.743%|+13.123%|+10.474%|토크 3개 기준 실패|
|556 /562|.99·25Hz, torque×1.25|rough|31/32|3.868222|0.326450|+0.325%|+4.127%|+3.115%|토크 3개 기준 실패|

모든 행의 원래 동작 gate는 통과했지만 양 지형의 동작과 토크 gate를 함께 통과한 모델은 없다. flat536의 진행 기준 여유는0.009682m였고 일부 관절 RMS와 거리당 추정 절대에너지는 증가했다. flat556의 FR_KFE RMS는58.460%, rough556의 FL_HFE는19.685% 증가했다. 전체 관절 평균만으로 개별 부하 개선을 주장하지 않는다. 12초·한 seed screening이며 미실행 holdout 성능을 추정하지 않는다.

## 그림·원자료·모델 식별

- [통합 비교 PNG](trained_model_comparison.png), [PDF](trained_model_comparison.pdf): RMS와 추정 절대 파워 변화. 녹색 영역은 두 비증가 기준만 충족하는 영역이며 전체 gate 통과를 뜻하지 않는다.
- [536 관절별 PNG](finetune536/evaluation/screening_torque_comparison.png), [PDF](finetune536/evaluation/screening_torque_comparison.pdf), [상세 보고서](finetune536/README.md).
- [556 관절별 PNG](reward_finetune/evaluation/screening_torque_comparison.png), [PDF](reward_finetune/evaluation/screening_torque_comparison.pdf), [상세 보고서](reward_finetune/README.md).
- [비교 CSV](trained_model_comparison.csv), [JSON](trained_model_comparison.json), [전체 최종 판정](final_v15_torque_decision.json), [92개 원자료 SHA 검증](final_raw_data_verification.json).

| 모델 | SHA256 |
|---|---|
| 원본339 / model11699 |`e2bb747acb4737f9e73dc34446cfae692ca2564ed36560fb1f137814eea46d33`|
| 학습536 / model11949 |`06ca0ec3fadd743c1033a75ccf133193e3ad0432f3112b79698467ac0efd4e7a`|
| 학습556 / model11949 |`3bd366c775e6bc347bec28fb6716ad54708ddc80633643090734261cb2f93a1e`|

학습536/556、평가542/562의 기존 source·marker·checkpoint·원자료를 각각 보존했다. 562에서는 원본339 동일 조건을 재실행하지 않고508을 재사용했으며 최종 데이터8개와 frozen source/model·재사용508의19SHA를 다시 대조했다. 서로 다른 물리 GPU 간 비트 단위 동일 시뮬레이션은 주장하지 않는다.

## 단위와 열 해석

토크 N·m, slew N·m/s, 파워는 추정 `mean(abs(τ·ω))`의 관절 평균 W다. 파워는 전기 소비나 권선 발열이 아니다. configured effort24.8N·m(HAA)/48N·m(HFE·KFE)는 시뮬레이션 최대 한도이며 continuous 모터 정격이 아니다. computed/applied torque는 각각 clipping 전/후 implicit PD 추정치로 실제 solver 모터 출력이 아니다. 정격·열 상수·냉각이 없으므로 peak가 한도 아래여도 열 안전은 입증되지 않는다.

저장된 자료로 이 통합 비교를 다시 만드는 명령(물리 실행 없음):

```bash
python3 reports/v16/v15_torque/finalize_comparison.py
```
