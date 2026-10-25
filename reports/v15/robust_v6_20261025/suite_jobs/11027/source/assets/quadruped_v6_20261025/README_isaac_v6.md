# Isaac USD v6 (몸통 라운드, 2026-10-25)

`isaac_usd_v5/` 와 같은 생성기·관절. 바뀐 것은 몸통 셸 2개뿐:
- `meshes/body_shell_F.stl`, `body_shell_B.stl` — 모서리 R35 라운드 형상(Fusion "quadruped robot_v6_round_body_2026-10-24")
- `model.json` 의 두 셸 질량·무게중심·관성: 1.046 → 1.128 kg, 1.036 → 1.118 kg
- 전체 질량 25.48 → 25.64 kg (base 링크만 +0.16 kg, 다리 링크 변화 없음)

열 파일: `quadruped_scene.usda`. 관절·게인·충돌 설정 등 나머지 설명은 `README_isaac_v5_ref.md`(v5 문서 그대로) 참고.
MuJoCo 교차 확인(`mujoco_check.py ../isaac_usd_v5 .`, 결과 `mujoco_check_result.json`, 키 v5/v6): 무릎 최대 토크 서 있기 9.7→9.8, 5 cm 낙하 26.5→26.7, 스쿼트 15.3→15.4 N·m, 낙하 발 반력 218 N 동일, 토크 포화 없음.
