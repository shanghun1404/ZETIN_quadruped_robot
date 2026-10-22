# V15 Jetson Orin Nano Super / STM32G474 USB 연결

## 요청과 완료 범위

검증된 V15 토크 최적화 후보를 사용자가 원할 때 Jetson에 복사할 수 있는 ROS2 배포 묶음으로 준비한다.
대상은 Jetson Orin Nano Super, Ubuntu 24.04, ROS2 Jazzy, STM32G474 Nucleo다.
보드는 NUCLEO-G474RE/MB1367로 가정한다. 모터는 HAA AK70-10, HFE/KFE AK10-9 KV60 V3.0.
이번에는 소프트웨어·USB 규약·STM32 공통 C 코드·G474 연결 예제·검증·배포 절차를 준비한다.
실제 Jetson 설치, MCU 플래싱, 모터 구동은 수행하지 않는다. 사용자 요청이 배포 준비 구현을 승인한다.
현재 작업 디렉터리의 기존 결과와 미커밋 파일을 보존하고 새 패키지에 구현한다. 자동 커밋하지 않는다.

최종 후보 model15699 SHA256
`9a440065d0f93835abbf02278f79fff91aea95a18c3bf4020b83bb092f9e6db6`.
현재 원본 CAD에서 외부 ROS2/Isaac Sim 비계단 9/9, 정격 RMS 초과 0/108, 기록된 자가 접촉 0 N.
USB 연결 경로는 별도로 검증하며, 이 수치를 실제 USB·Jetson·모터에서 재현했다고 말하지 않는다.

## 구조와 선택

기존 ROS2 제어기 앞뒤에 센서 시각 정렬과 USB 게이트웨이를 추가한다.
Jetson은 정책 50 Hz와 기립 WBC를 계산하고 관절 명령을 200 Hz로 보낸다.
STM32는 12개 목표각·목표속도·Kp·Kd·feedforward 토크를 받아 모터 드라이버에 전달하고,
실제 q/dq/토크·상태를 200 Hz로 반환한다. 각도만 보내면 Kp=Kd=0인 기립 WBC가 재현되지 않는다.
로컬 시간 정렬은 MCU 피드백 5 ms 간격을 사용하며, 외부 오도메트리·깊이·발 접촉 입력을 같은 틱에 모은다.
입력 공백이나 틱 누락을 숨기지 않고 명령을 중단한다. 기존 제어기·235→12 관측/출력·게인·한도는 보존한다.

대안으로 micro-ROS는 MCU ROS 메시지를 제공하지만 middleware/메모리/펌웨어 작업이 늘어난다.
ros2_control 하드웨어 플러그인은 통합 관리에 좋지만 현재 위치·가변 게인·토크 동시 계약을 다시 설계해야 한다.
이번 준비에는 작은 ROS2 Python 패키지와 고정 바이너리 USB 규약을 선택한다.

## ROS2 경계

새 패키지 `v15_hardware_bridge`:

- `usb_gateway`: `joint_command`와 `status`의 동일 stamp를 결합하고 바이너리로 전송한다.
- `sensor_sync`: `raw/joint_states`, `raw/odom`, `raw/depth/image`, `raw/foot_contacts`를 기존 입력으로 정렬한다.
- `cad_controller`: 기존 ControllerNode와 CadStandingController를 연결한다. demo=false, 초기 SIT, 정지 명령.
- 이름은 기존 policy 순서로 수신하고 USB에서는 FL/FR/RL/RR × HAA/HFE/KFE 순서로 정렬한다.
- SI: 관절 출력축 rad, rad/s, N·m, Kp N·m/rad, Kd N·m·s/rad. 전방 +Y, 우측 +X, 상방 +Z.
- `enable_motors=false` 기본. 켜려면 실제 모터 ID/부호/영점과 검증 표시가 설정되어야 한다.
- 관절 보정은 MCU 모터 어댑터에서 한 번만 적용한다. CAD 부호를 실기 배선 부호라고 가정하지 않는다.

새로운 센서 드라이버나 가짜 오도메트리/깊이/발 접촉은 만들지 않는다. 현재 후보는 깊이 입력을 필요로 한다.
센서 입력은 실제 측정 시각이 있는 메시지다. 오도메트리/접촉 최대 age 75 ms, 깊이 200 ms,
현재 시각보다 10 ms 이상 미래인 메시지 거부. MCU 피드백은 50 ms 이내여야 하며 주기 누락은 오류다.
호스트의 실제 시각과 제어기의 5 ms 논리 시각을 구분한다. use_sim_time=false가 실기 기본이다.

## USB 규약 v1 (Python과 C가 동일하게 구현)

little-endian IEEE754 float32. 헤더 `'<2sBBIIQH'`, 22 bytes:
magic ASCII `QD`, version uint8=1, type uint8, session uint32(nonzero), sequence uint32,
stamp_ns uint64, payload_length uint16. 마지막 CRC16-CCITT-FALSE 2 bytes,
poly0x1021/init0xffff/xorout0; 헤더+payload 전체에 적용. 최대 payload256 bytes.
partial/coalesced/noisy input은 스트림 파서로 처리하고 CRC 실패·잘못된 길이 뒤 다시 동기화한다.

| type | payload |
|---|---|
| 1 HELLO | empty; 다른 session이면 MCU disarm·새 세션 시작. 현재 session 재전송은 ACK만 반환 |
| 2 HELLO_ACK | uint32 capabilities=1 (full q/dq/kp/kd/tau command 지원) |
| 3 COMMAND | uint32 enabled(0/1), position[12], velocity[12], kp[12], kd[12], effort[12]; 244 bytes |
| 4 FEEDBACK | uint32 acknowledged_command_sequence, fault_flags, armed; position[12], velocity[12], effort[12]; 156 bytes |
| 5 STOP | empty; 즉시 disarm |

양쪽은 프레임 version/type/길이/CRC/session을 검증한다. sequence는 uint32 half-range 비교로
새 명령만 수용하며 duplicate/replay/다른 session은 watchdog을 연장하지 않는다.
COMMAND: 유한 값, |q|<=12.5 rad, |dq|<=50 rad/s, 0<=Kp<=300, 0<=Kd<=8,
|effort|<=24.8/48/48 N·m. 이 포맷 한계와 실제 관절 기계 한계는 별도다.
STM32가 실제 q/dq로 PD+feedforward 총 토크와 기존 .99 속도 곡선 가드를 다시 검사한다.
실제 모터 CAN/MIT 인코딩·단위·ID·전원/enable는 보드별 callback으로 연결한다.
실기 보정과 CAN 인터페이스가 완성되기 전에는 활성 명령을 수용하지 않는다.

호스트는 새 난수 session으로 HELLO/ACK 후 전송. 명령/피드백 stale 50 ms이면 STOP 후 연결 fault.
MCU watchdog은 마지막 유효한 새 COMMAND부터100 ms, STOP/잘못된 수치도 disarm.
fault는 현재 값과 다른 session의 HELLO로 명시적으로 세션을 시작하기 전까지 latch.
현재 session의 HELLO 재전송은 fault·sequence·watchdog 상태를 초기화하지 않는다. USB 재연결은 자동 arm하지 않는다.
직렬 baud921600,8N1. COMMAND268 bytes×200 Hz=53,600 B/s=536,000 bit/s;
FEEDBACK180 bytes×200 Hz=36,000 B/s=360,000 bit/s (각 방향). 실제 baud 지원과 부하를 보드에서 확인한다.

## G474 연결

MB1367 STLINK-V3E USB VCP → LPUART1 PA2(TX)/PA3(RX), 기본 solder bridge 유지.
G474 예제는 CubeMX LPUART1 DMA 수신/송신과 5 ms timer 콜백 위치를 제시한다.
HAL 코드·FDCAN 핀·모터 CAN ID·전류/상태 단위는 확정하지 않았으므로 board hooks를 명시한다.
공통 프로토콜 C 코드는 호스트 gcc에서 컴파일하고 Python 생성 프레임과 교차 검증한다.

## 배포 묶음

colcon src에 기존 ROS2 두 패키지와 새 패키지, 필요한 순수 Python config 모듈과 CAD 모델,
최종 actor·동결 student 체크포인트·SHA manifest·환경 검사·빌드/실행 스크립트·STM32 소스·문서를 포함한다.
절대 ws1 경로나 Isaac Sim 설치가 필요 없어야 한다. ARM64/Ubuntu24.04/Jazzy를 검사한다.
Torch는 CPU 경로로 시작하여 Jetson의 지연을 측정하고 GPU wheel 선택을 BSP와 분리한다.
서비스 자동 시작·OS 재설치·모터 enable 기본 켜기는 포함하지 않는다.

## 검증과 완료 기준

TDD: 손상/분할/합쳐진 프레임, CRC 알려진 벡터, session/replay/wrap, 값 검증, 원자적 stamp 매칭,
호스트/MCU watchdog, 부호·영점 경계, 실제 PTY로 송수신 및 Python↔C 교차 검증.
ROS2 Jazzy 원격 native 환경에서 실제 메시지로 gateway를 시험한다.
Isaac Sim에서는 최종 후보의 평지와 통합 기립 시험에 USB 바이너리 왕복을 삽입하고,
같은 동작/토크/접촉/시계열 기준을 적용한다. 물리 작업은 ws1 Slurm compute/QoS admin만 사용한다.
실제 Jetson/MCU/모터 시간 성능과 플래싱은 이번에 미검증으로 표시한다.

## 확인한 공식 자료

- [ROS2 Jazzy Ubuntu24.04/arm64 설치](https://repo.test.ros2.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [NVIDIA JetPack7.2.1/Ubuntu24.04/Orin 지원](https://developer.nvidia.com/embedded/jetpack/downloads)
- [ST UM2505, MB1367 LPUART1 VCP](https://www.st.com/resource/en/user_manual/dm00556337-stm32g4-nucleo64-boards-mb1367-stmicroelectronics.pdf)

현재 Jetson BSP 버전은 원격 접근이 없어 실측하지 않았다. 설치 전에 환경 검사 스크립트로 기록한다.
