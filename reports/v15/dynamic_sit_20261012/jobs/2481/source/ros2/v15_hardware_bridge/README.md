# V15 Jetson ↔ STM32G474 USB

Ubuntu24.04/ROS2 Jazzy, Jetson Orin Nano Super용 준비 패키지다.
현재 시험 완료한 model15699와 CAD 기립 제어기를 사용한다. 실행 시 모터는 기본 비활성이다.

```mermaid
flowchart LR
  S[깊이 / 자세·속도 / 발 접촉 측정] --> Y[ROS2 센서 시각 정렬]
  M[STM32 실제 관절 q/dq/상태] -->|USB| G[USB gateway]
  G --> Y
  Y --> P[V15 정책 50Hz + CAD 기립 WBC]
  P -->|각도·게인·토크 200Hz| G
  G -->|USB 바이너리| M
  M --> C[모터 CAN 어댑터]
```

## 실행과 배포

압축 묶음·설치·환경 검사는 [Jetson 배포 절차](../../deploy/jetson_v15/README.md)를 따른다.
`hardware.launch.py`의 필수 인자는 `actor_checkpoint`, `student_checkpoint`, `model_path`, `hardware_config`다.
기본 namespace `/v15real`, `enable_motors=false`, `demo=false`, 초기 자세 `SIT`다.
초기 자세는 네 발을 바닥에 둔 앉은 자세여야 한다. 넘어짐에서 일어나는 복구 정책은 포함하지 않는다.

토픽 이름은 기존 제어 계약과 같다. `cmd_vel`은 전방 `linear.y`, 우측 `linear.x`, 회전 `angular.z`다.
보행 명령은 0.5초 이내에 갱신해야 한다. 끊기면 기립 정지로 전환한다.
`mode_request`에 `stand`/`sit`을 보낼 수 있다. `walk`에도 갱신되는 `cmd_vel`이 필요하다.

## USB 포맷

STM32와 공유하는 규약은 [설계](../../docs/superpowers/specs/2026-10-12-v15-jetson-usb-design.md)에 있다.
canonical 순서 FL/FR/RL/RR × HAA/HFE/KFE, 출력축 SI 단위다.
목표각·목표속도·Kp·Kd·feedforward 토크를 한 패킷으로 전송한다.
기존 `joint_command`와 `status`는 같은 정수 ns stamp에서만 결합한다.
Wire각은 canonical 관절 좌표다. 실제 모터 ID/회전 부호/영점 보정은 MCU 어댑터에서 한 번 적용한다.
각도만 송신하면 순수 토크 제어인 기립 동작이 보존되지 않는다.

ST-LINK 가상 COM 기본 경로는 NUCLEO-G474RE/MB1367의 LPUART1 PA2/PA3이며 baud921600,8N1을 준비했다.
`serial_port`는 `/dev/serial/by-id/…` 고정 경로로 설정한다. 실제 baud/케이블/Jetson 지연은 실기에서 측정해야 한다.
host watchdog50ms, MCU100ms. 만료·센서 시각 누락·오류는 latch되고 STOP을 보낸다.
재연결/프로세스 재시작은 자동으로 모터를 활성화하지 않는다.

## 실제 센서 입력

| 토픽 | 형식 / 좌표 |
|---|---|
| `raw/joint_states` | USB gateway가 발행, canonical q(rad)/dq(rad/s)/effort(Nm), MCU5ms논리시각 |
| `raw/odom` | `nav_msgs/Odometry`, root actor 원점 world pose, child_frame_id=base의 body twist |
| `raw/depth/image` | 84×84 little-endian `32FC1`, step336, 미터 |
| `raw/foot_contacts` | `JointState`, FL/FR/RL/RR_foot, effort=측정 접촉력 크기(N) |

오도메트리·접촉은75ms, 깊이는200ms 이내의 실제 ROS clock 측정 stamp가 필요하다.
10ms 넘게 미래인 시각도 거부한다. MCU 논리 시각과 실제 센서 wall clock을 구분한다.
첫 완전한 센서 세트를 논리0에 정렬한 뒤 5ms MCU 틱 누락 시 중단한다.
깊이/오도메트리/접촉 값은 생성하거나 임의의 고정값으로 대체하지 않는다.

## 모터 설정과 검증 범위

외부 JSON에 `calibration_verified=true` 및 12개 `motors`를 설정해야 host 활성화가 가능하다.
항목은 `{joint, bus, id, direction, zero_rad}`다. bus/id는 실제 주소, direction은 ±1,
zero_rad는 실제 출력축 영점이다. 이 JSON은 host 검증용이며 MCU CAN 테이블도 동일하게 구현해야 한다.
예시 ID나 CAD의 sign을 실제 배선 보정값으로 자동 채택하지 않는다.
MCU `ready/read/apply/disarm` hook은 실제 모터 통신·단위·측정을 구현하기 전까지 비활성이다.

Python/C framing·원자적 ROS2 명령·센서 freshness·watchdog는 CPU/native Jazzy 시험으로 확인한다.
Isaac 검증 어댑터는 실제 PTY와 동일 C codec을 통과시킨 관절 상태·명령을 기존 물리에 적용한다.
이 검증은 실제 USB 전기 계층, G474 MCU 실행, CAN 대역폭, Jetson 주기/열 성능을 측정하지 않는다.
초기 검증 결과/코드 해시는 `reports/v15/jetson_usb_20261012`에 기록한다.
