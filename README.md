# robot_dog: Stage 1 V8–V13

IsaacLab 기반 사족보행 프로젝트의 V8부터 V12까지를 정리한 소스 아카이브와
V13 토크 최적화 파일럿이다.
학습·평가 코드, 정적 테스트, 설계 문서와 작은 평가 결과를 포함한다. 체크포인트,
데이터셋 shard, 영상, 실행 로그와 캐시는 저장소 크기 및 배포 안전성을 위해 제외했다.

이 결과는 시뮬레이션에서 검증되었으며 실제 로봇 하드웨어 검증 결과가 아니다.

## 버전 흐름

- **V8 / V8.1**: Spot 보행 보상 recipe와 duty-floor 실험
- **V9**: rough terrain 및 계단용 terrain curriculum
- **V10 / V10.1–V10.4**: robot_v2, duty-floor, blind terrain 및 WASD 명령 정책
- **V11**: 187점 height scan을 사용하는 terrain/stairs 정책
- **V12**: `84×84` depth 입력을 187점 scan으로 distillation한 Tiny ViT 전단
- **V13 torque pilot**: V12 전단을 고정하고 보행 액터를 재학습한다. 200 Hz 물리 스텝의
  25 Hz 1차 위치 목표 LPF, 액션 변화량 제곱 페널티(`-2.5`), 관절별 토크 한계 감점
  (`-0.2`)과 파워 감점(`-0.2`)만 추가한다.

V11의 선택 정책은 rough L4와 약 15.7cm 계단 평가를 5개 시드에서 통과했다.
V12 DAgger 모델은 rough/stairs-up/stairs-down 9회 평가 중 7회를 통과했지만,
stairs-up은 3개 시드 중 1개만 엄격한 기준을 통과했다. 자세한 수치와 한계는
[`reports/v11/REPORT.md`](reports/v11/REPORT.md)와
[`reports/v12/REPORT.md`](reports/v12/REPORT.md)에 기록되어 있다.

V13은 V11 액터의 235차원 입출력과 V12 Tiny ViT를 그대로 사용한다. V12 자체는
보상 함수가 없는 depth→scan 전단이므로, 새 토크 보상은 액터 PPO 재학습에 적용한다.
`config/train_v13_torque.py`가 V12 학생 체크포인트를 고정해 카메라 관측을 액터에
연결한다. `config/evaluate_v13_torque.py`는 평지·험지·계단을 같은 시드에서 비교한다.
V13 파일럿 액터는 평지 토크를 낮췄지만 험지·오르막 보행이 약해졌다. 따라서
기존 V11/V12 보행 액터에 V13 필터와 0.99 모터 토크 가드를 적용한 조합을
우선 후보로 선택했다. 가드는 평가 시 `--effort-guard-fraction 0.99`로 켠다.
평지·지형 결과와 실기 적용 전 한계는
[`reports/v13/GUARD_VALIDATION.md`](reports/v13/GUARD_VALIDATION.md)에 있다.

## 저장소 구조

```text
config/       학습·평가·probe 코드와 정적 테스트
docs/         V10.5–V12 설계 및 실행 계획
reports/      버전별 실험 보고서와 설정 snapshot
results/      작은 JSON/YAML 평가 및 집계 결과
```

`config/`는 기존 스크립트의 flat import와 source-contract 테스트를 보존한다.
V8–V12가 상속하는 V5–V7 계열 모듈 일부는 호환 의존성으로 포함했다.
또한 V10 WASD 환경이 공용 보상 구현을 `v13_mdp.py`라는 이름으로 import한다.
V13 학습·평가 파일과 소규모 평가 결과는 `config/`와 `reports/v13/`에 있다.

## 정적 테스트

IsaacLab 런타임이 필요 없는 테스트는 다음과 같이 실행한다.

```bash
cd config
python -m pytest -q tests
```

일부 테스트에는 PyTorch가 필요하다. 실제 학습·시뮬레이션 스크립트는 프로젝트에서
사용한 Isaac Sim 5.1 / IsaacLab 환경과 robot_v2 USD asset이 별도로 필요하다.
원본 코드에는 개발 머신의 절대 경로가 남아 있으므로 다른 머신에서 실행할 때는
USD, checkpoint, output 경로를 먼저 수정해야 한다.

## 제외한 바이너리 아티팩트

- `*.pt`, `*.pth`, `*.ckpt` 학습 checkpoint
- V12 rollout/dataset shard
- `*.mp4`, 이미지 및 실행 로그
- TensorBoard/W&B 출력과 캐시

보고서에는 선택 checkpoint의 SHA-256과 평가 결과가 남아 있어 원본 아카이브와
대조할 수 있다. 바이너리 모델을 나중에 공개하려면 Git LFS 또는 별도 release
artifact를 사용하는 것을 권장한다.
