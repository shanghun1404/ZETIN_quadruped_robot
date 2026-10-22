# V16 physical model 검증 기록

## 최종 검증

- HPC CPU Slurm 520: `COMPLETED`, exit `0:0`.
- `/opt/isaacsim-env/bin/python -m unittest discover -s config/tests -p test_v16_model.py -v`: **13 passed**, 0 skipped. 물리 fixture 5개, 실제 robot 4개, 독립 대수 4개.
- `derive_v16_discrete.py`: source hash, actual tensors, 144 SISO channels × two sample periods, nonzero initial state trajectory, frequency/time response comparisons.
- independent `v16_model_mujoco_oracle.py`: MuJoCo 3.14.0 source compiler와 Pinocchio 2.7.0의 mass/g/body pose/point Jacobian/frame-corrected bias 비교 통과.
- 로컬은 모델을 실행하지 않고 대수만 검증: `PYTHONPATH=config python3 -m pytest config/tests/test_v16_model.py -q` **4 passed, 9 skipped**. `v16_model_local_algebra.log` 보존.
- `python3 -m py_compile`은 production, derivation, tests, oracle 파일에 성공했고 `git diff --check`도 성공했다.

## Red/Green 이력

| Slurm job | 결과 | 내용 |
|---|---|---|
| 507 | FAILED | 구현이 없는 상태에서 10개 테스트가 expected assertion으로 실패 |
| 509 | FAILED | Pinocchio 2.7 `Frame` 생성자 parent_frame 인수 누락 확인 |
| 511 | FAILED | real robot 검증 통과, 1DOF EigenPy Jacobian squeeze 사례 확인 |
| 512 | FAILED | 1DOF shape 실패와 scalar difference 기능 미구현 확인 |
| 513 | COMPLETED | 최초 11개 테스트와 실제 matrices 생성 |
| 514 | FAILED | 12개 테스트·모달 SISO 통과, free-base bias 좌표 비교 오류 검출 |
| 515 | COMPLETED | MuJoCo source comparison에서 moving-frame acceleration 보정 후 통과 |
| 518 | FAILED | velocity constructor가 canonical 대신 Pin traversal 순서를 반환하는 회귀 확인 |
| 519 | COMPLETED | velocity API canonical 통일 후 13개 테스트 통과 |
| 520 | COMPLETED | 최종 코드와 scalar zero-channel 출력, 13개 테스트 및 모든 source oracle 통과 |

실패 로그도 삭제하지 않았다. 515/519/520의 수치 결과는 같으며 manifest의 code hash는 최종 520 실행 코드다.
near-unit pole의 polynomial cancellation regression은 로컬 독립 대수 fixture에서
최대 절대 오차 0.029로 먼저 실패한 뒤 모달 평가를 추가하여 통과했다.

## 전체 프로젝트 suite 제한

로컬 Python 3.12에는 torch가 설치되지 않아 전체
`PYTHONPATH=config python3 -m pytest config/tests -q`는 **23개 collection error**, exit 2로 중단됐다.
이것을 전체 suite 통과로 보고하지 않는다. 실제 로그는 `v16_model_local_suite.log`다.
실패 파일은 다음과 같다.

`test_aggregate_v11_multiseed.py`, `test_evaluation_cfg.py`, `test_expand_v11_checkpoint.py`,
`test_flat_reference_gate.py`, `test_gait_phase_metrics.py`, `test_grounded_gate.py`, `test_guide_frame.py`,
`test_v10_wasd_commands.py`, `test_v11_eval_metrics.py`, `test_v11_layout.py`,
`test_v12_camera_contract.py`, `test_v12_collection_contract.py`, `test_v12_dataset.py`,
`test_v12_eval_metrics.py`, `test_v12_teacher_actor.py`, `test_v12_training.py`,
`test_v12_vision_student.py`, `test_v13_depth_policy.py`, `test_v13_torque_math.py`,
`test_v13_torque_metrics.py`, `test_v14_heading.py`, `test_v14_residual.py`, `test_v15_math.py`.

부모 agent는 별도 HPC CPU 517에서 전체 config/tests를 424 passed, 11 skipped,
기존 warning 1개로 완료했다고 전달했다. 그 전체 실행의 원본 증거는 부모의 검증 기록을 따른다.
본 담당의 직접 확인은 520의 모든 source physics 검증과 위 로컬 대수 결과다.

## 독립 코드 리뷰

연구 agent의 읽기 전용 검토에서 현재 source MJCF·명시한 좌표계·fixed-base nominal 범위의
Critical/Important 오류는 발견되지 않았다. canonical mapping, fixed member 관성의 1회 집계,
joint anchor/축, source armature/damping 분리, 초기조건, ZOH 입력 지연, 144 SISO 중첩을 확인했다.
두 production 파일의 hash와 job 520 manifest가 같음도 별도로 확인했다.

Minor: 현재 derive의 `d=0` superposition 검증을 향후 `d!=0`인 nominal model로 확장한다면,
`initial_only`와 `zero_initial` 양쪽에 drift를 넣어 중복 계산하지 않도록 한쪽 drift를 0으로 두거나
별도 drift-only response를 빼야 한다. 현재 명시한 equilibrium `d=0` 결과에는 영향이 없다.

## 비교 범위

MuJoCo 검증은 원본 MJCF의 compiler semantics 검증이다. 변환 USD/PhysX의 mass/armature,
관절 drive damping/contact collision/frame convention은 별도 runtime 비교 대상이다.
발 point Jacobian은 지정된 body-local test point를 사용했으며 실제 contact patch라고 주장하지 않는다.
아직 이 문서의 결과로 독립 보행·실기 안정성·계단 성능을 주장하지 않는다.
