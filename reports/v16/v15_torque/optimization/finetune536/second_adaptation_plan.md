# 두 번째 적응 실험 계획

## 근거와 결정

542에서 536 정책은 평지·험지 행동 gate를 모두 통과했지만, 험지 after2s |추정 토크×속도| 평균이 12.59098 W/joint로 508보다 1.807% 높아 탈락했다. 설정만 변경한 339는 험지 생존30/32, 경로29/32, 파워+4.759%로 탈락했다. 고정 gate를 유지하고 두 후보를 채택하지 않는다.

다음은 frozen339를 다시 출발점으로 삼아 원래 guard0.99·LPF25Hz에서 기존 joint_torques 벌점만 1.25배로 강화한다. 실제 기존 weight는−0.0005이며 제안값은−0.000625다. 설치 함수는 12관절 applied_torque의 L2 노름을 반환한다. 토크제곱이나 기계적 파워의 직접 벌점은 아니므로 파워 감소는 검증할 가설이며 예측된 결과가 아니다. 이번에 joint_power reward는 추가하지 않는다. [HPC 읽기 전용 근거](../../source/second_adaptation_reward_readonly.json).

## 구현·검증 순서

- 별도 v15_torque_reward_finetune.py/.sbatch를 작성한다. 536/542 소스와 원본 V15 파일은 보존한다.
- frozen339/학생/USD/원본 trainer/core source/벌점 함수 소스 SHA를 확인한다.
- LR3.375e-5/fixed를 algorithm과 optimizer 모든 group에 명시하고 복원 moments를 보존한다.
- 512환경/250회/seed47, rollout24, physics0.005s/control0.02s, episode20s, delay0–3, gain U(.8,1.2), 기존 finite/fall 방어를 유지한다.
- 실제 reward manager의 weight/함수와 다른 reward weights, 실제 guard/LPF/alpha/LR를 contract에 기록하고 검증한다.
- CPU 계약 검증과 부모 독립검토 후 GPU1/CPU4/16G/40분으로 제출한다. GPU 순서는 부모와 조율한다.
- 최종 model11949가 finite이고 completion/모델 SHA가 일치하면 새정책 flat/rough32env seed42 fixed_v2 12s의2조건을평가한다. 부모지시에따라source/원자료/물리계약을확인한기존508원본339자료를기준선으로재사용한다.
- 기존508의 생존·90%전진·명령오차+0.02·험지방향·RMS/파워 비증가·한계부하개선 gate를 그대로 적용한다. 일부 실패하면 채택하지 않는다.
- 전부 통과한 경우에만 seed43/44에서 원래339와 새정책의 fresh trace를 수집한다. 기존 holdout 추가 기준도 유지한다.

## 현재 상태

부모·독립 검토와 GPU 조율 후 제출한556은 COMPLETED0:0으로 완료했고, 실제 reward/LR/guard/alpha/source 계약을 확인했다. 기존14개 다른 reward는 불변이고 torque L2계수만−0.000625다. 완료 checkpoint/marker를 검증한 뒤2조건 평가562를 제출했으며, 이 작업도 COMPLETED0:0으로 완료했다.

562에서 flat/rough 동작 gate는 통과했으나 토크 RMS는 각각+1.743%/+0.325%, 추정 절대 관절 파워는+13.123%/+4.127%였고 한계부하 개선 기준도 실패했다. **536과556 모두 채택하지 않고 원본339를 유지한다.** holdout과 추가 V15 GPU 작업은 제출하지 않는다. [최종 두 모델 비교](../trained_model_comparison.md), [556/562 상세](../reward_finetune/README.md). 어떠한 후보도 solver motor torque·모터 연속 정격·열 안전을 입증하지 않았다.
