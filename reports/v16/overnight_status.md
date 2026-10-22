# V16 야간 훈련·평가 상태

마지막 확인(KST): 2026-09-29T10:23:47.445114+09:00

HPC의 실제 Slurm 상태와 저장 결과를 로컬로 복사한 기록이다. 진행 중인 훈련 점수는 독립 평가 결과가 아니다.

| Job | 작업 | 상태 | 실행시간 | 종료코드 |
|---|---|---|---|---|
| 563 | physics_jobs | COMPLETED | 00:14:12 | 0:0 |
| 568 | physics_jobs | COMPLETED | 00:07:32 | 0:0 |
| 570 | comparison_jobs | COMPLETED | 00:02:05 | 0:0 |
| 571 | physics_jobs | COMPLETED | 02:03:27 | 0:0 |
| 582 | evaluation_jobs | COMPLETED | 00:02:23 | 0:0 |
| 573 | evaluation_jobs | COMPLETED | 00:05:16 | 0:0 |
| 583 | video_jobs | COMPLETED | 00:05:58 | 0:0 |
| 575 | physics_jobs | COMPLETED | 00:01:18 | 0:0 |
| 576 | physics_jobs | COMPLETED | 00:09:46 | 0:0 |
| 577 | physics_jobs | COMPLETED | 02:19:26 | 0:0 |
| 578 | evaluation_jobs | COMPLETED | 00:03:28 | 0:0 |
| 579 | evaluation_jobs | COMPLETED | 00:08:33 | 0:0 |
| 584 | video_jobs | COMPLETED | 00:06:46 | 0:0 |

## 훈련 진행

- 563: generation 23, 생존 128, 선택 후보의 진행 0.719m, 완료 steps 9.
- 568: generation 11, 생존 512, 선택 후보의 진행 0.751m, 완료 steps 9.
- 571: generation 211, 생존 512, 선택 후보의 진행 0.751m, 완료 steps 9.
- 576: generation 7, 생존 510, 선택 후보의 진행 0.752m, 완료 steps 9.
- 577: generation 127, 생존 512, 선택 후보의 진행 0.753m, 완료 steps 9.

## 독립 평가

접촉력 QP의12초·32초 보행 기준: **통과**.

가속도·접촉력 WBC의12초·32초 보행 기준: **통과**.

기준은 생존95% 이상, 평균 속도 오차0.05m/s 이하, 평균 진행이 명령 거리의75–125%, 유한 통계다.
1001–1003 seed는 QP 훈련101–312와 WBC 훈련401–528에서 분리한다. 같은 고정 모델을12초와32초에서 평가하며 평가 사이에 재튜닝하지 않는다.
두 길이와 두 제어기는 같은 평가 seed를 사용하는 비교이므로 서로 독립적인 시행으로 합산하지 않는다.
QP의 유한 반복 제약 위반과 별도 토크 guard는 각 평가 JSON에 기록한다. 보행 기준 통과가 모든 QP 제약의 만족을 뜻하지 않는다.

- 582: [True](evaluation_jobs/582/README.md)

- 573: [True](evaluation_jobs/573/README.md)

- 578: [True](evaluation_jobs/578/README.md)

- 579: [True](evaluation_jobs/579/README.md)

## 영상

- [env_00.mp4](video_jobs/583/video_seed1001/env_00.mp4)
- [env_00.mp4](video_jobs/584/video_seed1001/env_00.mp4)

V15의 추가 토크 최적화 후보는 채택하지 않았고339 원본을 유지한다. V16은 독립적인 물리 제어 실험이다.
