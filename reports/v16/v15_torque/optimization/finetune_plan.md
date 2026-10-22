# V15 제한적 재학습 준비 계획

> **For agentic workers:** Use executing-plans to implement the preparation inline. Parent `/root` reviews the prepared result and decides submission. No automatic commit, GPU submission, or training execution.

**Goal:** 고정339 checkpoint를 guard0.97/LPF23Hz에 적응시키는 짧은 PPO pilot을 원본 변경 없이 준비한다.

**Architecture:** 기존 V15 trainer를 SHA256으로 고정한 뒤 메모리에서 설정을 적용한다. 저장 optimizer를 복원하고 algorithm과 모든 optimizer group에 실제 저장 LR3.375e-5를 명시적으로 맞춘다. 기존 환경/보상/rollout/종료 방어를 유지하고 별도 디렉터리에 실제 설정과 checkpoint 해시를 기록한다.

**Tech Stack:** Python, RSL-RL PPO, IsaacLab/PhysX, Slurm. 로컬에서는 Python 문자열 구성과 CPU 단위 검증만 수행한다.

**Spec:** 부모의2026-09-29 재학습 준비 지시와 [최적화 기준](README.md).

## 제약과 검토 항목

- 원본 V15 코드/모델 보존, 새 파일은 허용된 `v15_torque_*`/보고서 범위에만 작성.
- 학습 actor339/model11699 SHA256 고정; 모델403을 선택하지 않음.
- GPU1/CPU4/16GB/40분. 현재 노드DRAIN이므로 파일 준비만 수행.
- 기본 pilot1024환경/250회, 최대500회. 339의1024환경 MaxRSS15.50GiB였으므로16GB 요청 여유가 작다. 필요하면 부모가512환경으로 낮춰 제출한다.
- 물리dt0.005s, 정책dt0.02s, rollout24step, episode20s, gain U(0.8,1.2), delay0–3step, 기존 지형/관측/finite 방어 유지.
- 저장 optimizer LR3.375e-5와 설정0.001 adaptive가 다르다. 설치 PPO는 adaptive에서 algorithm 멤버로 optimizer LR를 덮어쓰므로 양쪽을 고정하고 실제값을 기록한다.
- 성공 marker는 checkpoint/동기 writer flush 확인 뒤 기록하며, optional Kit shutdown에 의존해 GPU를 붙잡지 않도록 종료한다.
- 이 준비는 물리/학습 통과 증거가 아니다. 5개 기존 후보의 실패를 보존하고 새 모델은 동일 gate로 평가한다.

## 작업

- [x] CPU 검증: LR reset이 모든 group과 algorithm에 적용되고 optimizer state가 보존되는지 확인; 비정상 group과 원본 hash 변경은 거부.
- [x] `config/experiments/v16_20260929/v15_torque_finetune.py`: hash 고정 overlay, 실제 적용값/고정 LR 기록, bounded pilot, 종료 marker.
- [x] `v15_torque_finetune.sbatch`: 원본339 입력, 별도 출력/source hash, GPU1/16G/40분, 미제출.
- [x] syntax/구성 검증과 한국어 권고 문서. 부모 검토가 끝나기 전 학습 실행 없음.

## 평가 계획

pilot250회까지만 먼저 수행하고 PPO/std/손실/비유한 진단을 확인한다. 500회는 추가 판단 뒤의 별도 pilot 상한이다.
마지막 checkpoint를 flat/rough32환경/seed42/fixed_v2/12초/nominal300/8/delay0에서 재평가한다.
baseline339와 새guard0.97/LPF23에서 재학습 전339를 모두 비교해 정책 적응과 설정 변화 효과를 구분한다.
생존·진행90%·속도오차 allowance0.02m/s·험지corridor/heading·2–12초 RMS/추정power·한도요구 gate를 그대로 유지한다.
통과 후에만 seed43/44에서 baseline/candidate trace를 새로 생성하는 직렬 holdout을 진행한다.
계단 결과를 일반화하지 않으며, 기립/평지/험지/계단 각각 별도 결과로 기록한다. 모터 연속정격/실제solver토크/열 안전은 확인되지 않는다.

## 읽기 전용 조사 근거와 권고

[checkpoint와 설치 라이브러리 원문](../source/training_resume_readonly.json)을 CPU로 읽어 저장했다.
339 마지막 optimizer LR은3.375e-5이고 std범위0.11752–0.14228이다. agent.yaml은0.001/adaptive이며
OnPolicyRunner.load는optimizer state만 복원하고 PPO의learning_rate 멤버는복원하지 않는다.
PPO adaptive update는그멤버를optimizer group에 다시쓴다. 단순`--resume`가낮은LR를지킬것이라고가정하지않는다.
403의성능저하원인으로이기제를단정하지않으며, 새pilot은양쪽LR를명시적으로고정해기록한다.

기존launch_v15_locomotion.sbatch는24GB/6시간이고현재40분/16GB규격에맞지않으므로별도실행기를준비했다.
339의마지막iteration시간2.47초로만환산하면250회약10.3분/500회약20.6분이며초기화/하드웨어변화는별도다.
시간예측을성공보장으로쓰지않고Slurm40분상한을유지한다.

가까운23Hz/guard0.97 screening을먼저실행하면학습변화없이2조건만으로기존339를확인할수있다.
이후에도gate실패면bounded motor adaptation pilot이합리적인다음단계다. 부모가병렬GPU여력을활용해pilot을먼저검토할수도있으며,
이경우같은23Hz/guard0.97의재학습전339를reference로반드시평가해야한다.
현재파일은V15의현행finite/physics_diverged방어를그대로계승한다. 이방어는339당시이후추가된기존코드라는이력도유지한다.
새torque/power보상은도입하지않았고부하감소는외부gate에서실측한다.

부모검토후에만실행할명령:

```bash
scp -o ClearAllForwardings=yes config/experiments/v16_20260929/v15_torque_finetune.py \
  config/experiments/v16_20260929/v15_torque_finetune.sbatch \
  ws1:/home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/
ssh -o ClearAllForwardings=yes ws1 'sbatch /home/shanghun1404/v15_large_scale/config/experiments/v16_20260929/v15_torque_finetune.sbatch 1024 250'
```

이명령은이번작업에서실행하지않았다. 메모리여유가필요하면첫인자를512로바꾼다.
`finetune_actual_contract.json`에서실제cutoff23/guard0.97/alpha, fixedschedule, algorithm/group LR, 시작iteration11700,
rollout24/episode20s, PDgain범위, delay0–3, finitewrapper를확인한다.
완료는`run_completed.json`의checkpoint SHA256과policy std/finite상태및nonfinite_steps를확인한다.
writerflush/close와CUDA동기화뒤완료marker를기록하고os._exit0으로종료한다. 물리/행동평가gate는별도이며marker는채택을뜻하지않는다.

검증 결과: `python3 reports/v16/v15_torque/test_finetune_contract.py`4개통과,
`bash -n config/experiments/v16_20260929/v15_torque_finetune.sbatch`통과. 로컬물리/학습실행없음.

## 카메라와 checkpoint 이름 재확인

[설치소스 읽기전용 근거](../source/camera_checkpoint_readonly.json)에 camera/RSL 원문과해시를남겼다.
`v12_camera.py`의84×84 tiled distance_to_image_plane 영상은매policy관측에서고정student를통과하므로
`--enable_cameras`가필수다. 설치TiledCamera는해당flag가없으면초기화에서RuntimeError를발생시킨다.
렌더링을빼면같은훈련관측/환경을유지할수없어준비한sbatch의flag를유지한다.

설치RSLlearn은start_iter부터start_iter+N전까지loop하고매번current_learning_iteration=it를쓴후
마지막model_{current_learning_iteration}.pt를저장한다. 기존resume+1을그대로유지하므로
11700시작/250회11949종료/500회12199종료가맞으며새overlay의완료파일경로와일치한다.
512환경/250회로부모가첫pilot을검토중이다. 파일기본값1024는CLI첫인자512로덮어쓰며actual contract에실제512가남는다.
두GPU환경의Slurm할당과Kit렌더링GPU맵은부모가첫실제dual-job로그/UUID로검증할예정이며이번CPU검증에서는확인하지않았다.

## 실제 실행 업데이트

부모독립검토에서Critical/Important없음을확인한뒤536을512환경/250회로제출했다.
HPC `/home/shanghun1404/v15_large_scale/results/v16_20260929/v15_torque_finetune_536`,
현재Slurm RUNNING을확인했다. 기존준비단계의미제출기록은그시점의기록이며최신상태는실제RUNNING이다.
실제contract/LR/std/비유한진단과최종checkpoint는파일생성뒤확인한다. 보행/토크gate는완료모델의별도평가에서판정한다.

최신완료:536은11분50초에COMPLETED0:0,MaxRSS13.19GiB. model11949와고정LR/finite/std/데이터SHA를검증했다.
동등평가파일v15_torque_finetune_eval*를준비했고실제536입력바인딩의HPC읽기전용검사를통과했다.
평가4파일독립검토뒤부모가542를제출했고RUNNING/완료536실제입력preflight통과를확인했다. 4조건결과로고정gate를판정할예정이다.
