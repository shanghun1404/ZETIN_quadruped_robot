# V14 초기 후보 실험 기록

이 폴더는 2026-09-24 초기 USD 후보와 보행 진단에 사용한 Slurm 스크립트다.
일부는 기각한 15 mm 캡슐·충돌 그룹 자산을 참조한다. 순차적으로 여러 Isaac
프로세스를 띄우는 스크립트에서는 두 번째 렌더러 초기화가 실패한 적이 있다.
`flat_stand_seed42.json`을 쓰는 smoke는 명령 속도 0 평가이며 중립 자세
서기 검사를 대신하지 않는다.

현재 평가에는 `../../launch_v14_single_eval.sbatch` 또는
`../../launch_v14_validation.sbatch`를 사용한다.
정식 중립 자세/교차 물리 검사는 `../../probe_v14_leg_physics.py`가 담당한다.
선택 자산과 실험 결과는 `../../../reports/v14/README.md`에 기록한다.

`launch_v14_radius015_adapted.sbatch`는 9월 26일 적응 완료된 5700 액터로
15 mm 외피를 다시 비교한 후속 진단이다. 오르막 11/32로 개선하지 못했다.
