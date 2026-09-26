# V14 residual 훈련 확대 계획

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** 검증된 model_100의 보정 policy를 128개 환경에서 3,000회 추가 훈련한다.
**Architecture:** frozen base/student, 307차원 입력, 128×128 MLP, 보상과 PPO 설정은 유지한다. 체크포인트의 actor/critic/optimizer를 복원하고 완료 iteration 다음부터 번호를 부여한다. 별도 출력 폴더에 결과를 보존한다.
**Tech Stack:** Isaac Lab, installed RSL-RL, PyTorch, Slurm admin / HPC ws1.
**Spec:** `docs/v14_residual_design.md` + 사용자 요청 “계속 훈련해봐 훈련 사이즈 늘려서”.

## Global Constraints

- 훈련과 시뮬레이션은 HPC에서만 실행. 기존 worktree에서 작업하며 자동 commit 금지.
- 기본 비교 후보 residual_jobs/174/model_100.pt 및 frozen 자산 해시 유지.
- 128환경 × 24step = 업데이트당 3,072개 전이; 3,000회 = 9,216,000개 추가 전이.
- 먼저 128환경/20회 smoke. 메모리 부족 시 64환경으로 조정하고 근거를 기록.
- 별도 smoke 이후 본 훈련도 원래 model_100부터 시작. seed46, LR1e-4, 상한 .05rad 유지.

## Review Focus

- 재개 가중치가 zero 초기화에 의해 지워지지 않는가: 실제 RSL runner 재개 테스트.
- optimizer state가 누락되지 않는가: 1회 update 후 저장/복원된 state 비교.
- iteration이 중복되거나 마지막 파일 검증이 틀리지 않는가: 저장된 100 → 다음101, 종료번호 명시.
- 잘못된 자산/변조된 checkpoint가 실행되지 않는가: 기존 manifest 부정 입력 테스트 유지.
- 128환경의 GPU/host 메모리와 고유 terrain origin: HPC smoke 및 런타임 검사.

## Task 1: 실제 재개와 확대 launcher

Files: `config/v14_residual_policy.py`, `config/train_v14_residual.py`, `config/launch_v14_residual_scale.sbatch`, `config/tests/test_v14_residual.py`.
Interface: `resume_residual(runner, checkpoint, output_checkpoint)` restores policy/optimizer, saves a pre-update copy, returns provenance with next_iteration and policy hash. Trainer validates the existing manifest first.

- [x] 실제 설치된 OnPolicyRunner로 nonzero policy/optimizer/iteration 복원 테스트 작성 후 RED 확인.
- [x] helper 및 CLI `--resume-residual`, `--save-interval` 구현. 원래 fresh/zero 경로 보존.
- [x] 확대 launcher는 환경/iteration 인자를 받아 1GPU, 8CPU, 32G, 2시간으로 요청.
- [x] HPC CPU 단위 테스트, compile, bash -n, 기존 관련 회귀 검사 통과.

## Task 2: HPC 훈련과 비교

Files: `reports/v14/residual_scale/` (새 산출물).
Consumes: Task1 trainer와 checkpoint sidecar.

- [x] 128환경/20회 smoke에서 재개 hash, frozen hash, finite, correction bound, 자원 사용 확인.
- [ ] 128환경/3,000회 본 훈련 실행, 중간 checkpoint를 기존 fixed_v2 계단 seed42와 비교.
- [ ] 최종 또는 유효 중간 후보를 seed43/평지로 평가하고 기존 model100과 비교.
- [ ] 최종 리뷰 및 실제 수행/진행 상태를 보고서에 기록. 성능 개선 여부와 훈련 완료 여부를 구분.
