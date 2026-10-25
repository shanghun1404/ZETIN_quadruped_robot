# STM32G474 USB v1 준비 코드

NUCLEO-G474RE/MB1367을 가정한 공통 C11 프로토콜 코어와 G474 HAL 연결 예제다.
Jetson Orin Nano Super / Ubuntu 24.04 / ROS2 Jazzy의 `v15_hardware_bridge`와 같은 규약을 사용한다.
기본 포팅 예제의 확인 범위는 호스트 GCC, 실제 native 프로세스, Python/C 바이너리 상호 호환,
HAL stand-in으로 실제 포팅 예제 branch를 컴파일·실행한 회귀 시험이다.
CubeMX 프로젝트, 실제 CAN/MIT 모터 드라이버, MCU 플래시 파일은 포함하지 않는다.
실제 Jetson, STLINK VCP baud/부하, G474 IRQ/DMA 타이밍, 모터 구동은 검증하지 않았다.

2026-10-20 별도 진단 펌웨어에서는 실제 Jetson↔NUCLEO-G474RE의 STLINK VCP 921600baud와
FDCAN1/2/3 external loopback, PB9/PB6/PB4 TX 패드의 명령·가상 응답 프레임을 확인했다.
`examples/qd_ak_virtual.*`는 HAA=AK70-10, HFE/KFE=AK10-9 KV60 V3의 MIT/Servo 각도 프레임 에뮬레이터다.
비활성·영 게인 명령만 허용하며 실제 모터 동역학이나 측정값을 만들지 않는다.
CAN 수신 명령을 가상 각도 응답으로 바꿔 다시 CAN을 거친 뒤 복원한다. 실제 모터 제어용 어댑터는 계속 별도다.
200Hz에서 드러난 파서의 바이트별 반복 검색은 수신 묶음별 검색으로 수정했다.
모드별 1시간 시험의 실행 상태와 결과는 [G474 AK 진단 보고서](../../reports/v15/jetson_g474_ak_20261020/report.html)에 기록한다.
트랜시버·모터가 없어 CAN_H/L, 외부 ACK와 실제 모터 동작은 확인하지 않았다.

## 빌드와 검증

저장소 루트에서:

```sh
make -C firmware/stm32g474_usb test all
python3 firmware/stm32g474_usb/tests/test_emulator.py
python3 firmware/stm32g474_usb/tests/test_cross_language.py
```

`build/test_protocol`은 CRC 알려진 벡터, 헤더·payload, 분할·노이즈·손상 복구,
세션·sequence wrap·재전송, 100 ms watchdog, STOP, sticky fault, 전체 PD+FF 토크와 상태/보드 실패를 검증한다.
`build/test_g474_port`는 테스트용 `main.h`를 사용해 실제 HAL 예제 branch를 빌드하고
foreground 지연과 5 ms 시각, 누락된 tick, HELLO 처리 중 새 ISR fault, 반복 HELLO의 IRQ gate 유지 동작을 확인한다.
실제 STM32 HAL 라이브러리·CubeMX 생성 프로젝트·G474 주변장치는 이 시험에 포함하지 않는다.
emulator 시험은 실제 C 프로세스에 바이너리를 주고 활성화 거부·STOP·NaN·실측 상태 주입·실시간 watchdog을 확인한다.
교차 시험은 다섯 frame type의 바이트 일치, 100개 난수 명령/피드백, 분할 스트림을 검사한다.
`tests/test_emulator.py`는 표준 라이브러리로 기대 프레임을 만들며, 교차 시험만 ROS2 패키지의 순수 Python 모듈을 import한다.
이 코드는 ROS2나 STM32 HAL 없이 빌드된다. `-ffast-math`는 유한 값 검사를 깨뜨릴 수 있으므로 사용하지 않는다.

검증 기록은 저장소의 `reports/v15/jetson_usb_20261012/verification/firmware_red.log`와 `firmware_green.log`다.

## USB 규약

헤더는 little-endian `<2sBBIIQH`, 22 bytes다.

| 필드 | 형식 |
|---|---|
| magic, version, type | ASCII `QD`, uint8=1, uint8 |
| session, sequence | uint32, uint32; session은 0 금지 |
| stamp_ns, payload_length | uint64, uint16 |
| CRC | 헤더+payload의 CRC16-CCITT-FALSE; 마지막 uint16 |

CRC poly=0x1021, init=0xffff, xorout=0이다. 최대 payload는 256 bytes다.
`"123456789"`의 CRC는 `0x29b1`이다. 타입마다 길이를 엄격히 확인한다.

| type | payload |
|---|---|
| 1 HELLO | empty |
| 2 HELLO_ACK | uint32 capabilities=1 |
| 3 COMMAND | uint32 enabled(0/1), q[12], dq[12], kp[12], kd[12], ff[12]; 244 bytes |
| 4 FEEDBACK | uint32 ack_sequence, fault_flags, armed(0/1), actual_q[12], actual_dq[12], actual_tau[12]; 156 bytes |
| 5 STOP | empty |

각 배열 값은 IEEE754 float32다. 순서는 `FL_HAA, FL_HFE, FL_KFE, FR_HAA, ... RR_KFE`다.
관절 출력축 SI 단위는 rad, rad/s, N·m, N·m/rad, N·m·s/rad다.
COMMAND의 형식 한도는 |q|≤12.5, |dq|≤50, 0≤kp≤300, 0≤kd≤8,
|ff|≤관절별 [24.8,48,48]이며 NaN/Inf는 거부한다. 실제 관절 기계 한도는 별도로 보드 어댑터가 검사해야 한다.
FEEDBACK은 실제 상태를 보존하며 형식 한도 대신 유한 값과 armed=0/1을 검사한다.
capabilities=1은 전체 명령 형식 지원을 뜻하며 실제 모터 사용 가능 여부를 보증하지 않는다.

## 공개 C API와 상태 규칙

`include/qd_protocol.h`가 공개 API다. 동적 할당을 하지 않는다.

| API | 용도 |
|---|---|
| `qd_encode_frame`, `qd_decode_frame` | 타입/길이/session/CRC 검증과 framing |
| `qd_parser_init`, `qd_parser_feed` | 분할·병합·노이즈 입력에서 callback으로 정상 frame 전달 |
| `qd_pack/unpack_command`, `qd_pack/unpack_feedback` | 수치와 배열 계약 검증 |
| `qd_controller_init` | 기본 disarm, 보드 callbacks 등록 |
| `qd_controller_on_frame` | session/sequence·명령 처리와 HELLO_ACK 생성 |
| `qd_controller_tick` | 5 ms마다 실제 상태 재검사, 100 ms watchdog, 명령 재전달 |
| `qd_controller_feedback` | 현재 실제 상태를 다시 읽고 FEEDBACK 생성 |
| `qd_controller_fault` | IO/외부 오류를 latch하고 즉시 disarm |
| `qd_guard_command` | 실측 q/dq로 총 PD+FF 토크 검사 |

코어 호출은 하나의 실행 문맥에서 직렬화한다. ISR에서 같은 controller/parser를 동시에 변경하지 않는다.
parser callback의 frame 포인터는 callback 동안만 유효하다. 필요하면 복사한다.
struct 메모리 전체를 wire로 쓰지 않는다. padding과 CPU endian을 피하도록 encode/pack API를 사용한다.
프레임 파서는 수치가 잘못된 COMMAND도 전달한다. controller가 그 명령을 받으면 fault와 disarm을 수행한다.
CRC/header가 손상된 입력은 parser가 버리며 watchdog을 갱신하지 않는다.

현재 session과 다른 nonzero session의 HELLO만 새 세션을 시작하고 disarm·명령 sequence 초기화·fault 해제를 수행한다.
같은 session의 HELLO는 ACK만 다시 만들며 armed, sequence, fault와 watchdog 기준 시각을 초기화하지 않는다.
ACK는 요청 HELLO의 session/sequence/stamp를 echo한다. FEEDBACK header sequence는 MCU에서 따로 증가한다.
COMMAND의 sequence가 이전 수용 명령보다 uint32 half-range로 앞설 때만 수용한다.
duplicate/replay/다른 session은 watchdog을 연장하지 않는다.
활성 상태에서 마지막 유효한 새 COMMAND로부터 100 ms가 되면 disarm하고 watchdog fault가 유지된다.
STOP과 enabled=0 명령은 즉시 disarm한다. STOP은 기존 fault를 해제하지 않는다.
잘못된 수치, 전체 토크 초과, 불완전한 보드, 실측 실패도 disarm하고 다른 session의 HELLO 전까지 fault를 유지한다.
유효한 disabled 명령은 보드가 준비되지 않아도 받을 수 있다.

| fault bit | 의미 |
|---|---|
| 1 | watchdog |
| 2 | COMMAND 수치/형식 |
| 4 | PD+FF 전체 토크 가드 |
| 8 | 보드 준비 또는 전달 실패 |
| 16 | 실제 상태 읽기/유한 값 실패 |
| 32 | 포트 RX/TX/타이머 IO 오류 |

토크는 `kp*(q_target-q_actual) + kd*(dq_target-dq_actual) + ff`다.
각 관절 cap을 사용하여 현재 V15 `motion.py`와 같은 endpoint 속도 곡선을 적용한다:

```text
upper = .99 * cap * clip( 1 - dq_actual/50,  0, 1)
lower = .99 * cap * clip(-1 - dq_actual/50, -1, 0)
```

초과 명령을 클리핑해 계속 구동하지 않고 fault로 중단한다.
float32로 목표각을 복원할 때의 비교 허용분은 각 값의 반 ULP와 게인으로 계산한다.
`δx = .5 * (nextafterf(|x|,+Inf)-|x|)`로 정의하면, 게인이 0보다 클 때 해당 항을 포함하여
`4*FLT_EPSILON*cap + δff + kp*(δq_target+δq_actual) + δkp*|q_target-q_actual|
+ kd*(δdq_target+δdq_actual) + δkd*|dq_target-dq_actual|`를 사용한다.
각 경계의 허용분은 이 값과 원래 속도 곡선까지 남은 1% 여유 중 작은 값으로 제한한다.
속도 한계에서 0인 가속 토크 경계에는 허용분을 적용하지 않는다. credible float32 오차를 넘는 요청은 계속 거부한다.
`qd_controller_tick`은 저장된 명령을 현재 q/dq로 다시 검사한다. CAN 어댑터도 실제 드라이버 인코딩·갱신 주기에 맞춰 한도를 유지해야 한다.

## 실제 보드 어댑터

`qd_board`의 ready/read/apply/disarm 네 callback이 모두 있어야 enabled=1을 수용한다.
ready는 12개 실제 모터의 CAN 포트, ID, 모델별 인코딩, 출력축 단위, 부호, 영점,
측정 freshness, 모터 enable/disable 경로와 기계 한도 검증이 끝났음을 확인해야 한다.
read는 12개 실제 q/dq/tau를 완전히 채우고 오래되거나 일부 누락된 상태는 false로 반환한다.
apply는 일부 모터라도 전송/수용이 실패하면 false를 반환한다. disarm은 그 부분 전달까지 포함해 enable을 즉시 철회해야 한다.
상태 읽기가 없으면 `qd_controller_feedback`은 false를 반환한다. 실제 측정 대신 영 상태를 만들어 전송하지 않는다.

현재 모터 정보는 HAA AK70-10, HFE/KFE AK10-9 KV60 V3.0까지다.
CAN ID, CAN 핀, 실제 전류/토크 단위, 전원/enable GPIO, 관절 부호와 영점은 미확정이다.
`examples/board_hooks_disabled.c`는 ready/read/apply=false이며 실제 구동을 할 수 없다.
이 파일의 disarm 경로까지 실제 하드웨어로 구현한 후 ready를 변경한다.
보정은 MCU 어댑터에서 한 번만 적용한다. CAD의 부호를 실제 배선 부호로 가정하지 않는다.
드라이버 값이 이미 출력축 단위라는 것을 검증한 경우에만
`q_canonical=sign*(q_motor-zero)`, `dq_canonical=sign*dq_motor`, `tau_canonical=sign*tau_motor`를 적용할 수 있다.
회전자 값이면 검증된 감속비/단위 변환을 먼저 적용해야 한다.

### AK MIT 실제 모터 어댑터 (2026-10-20, 모터 미연결)

`examples/qd_ak_motor.{h,c}`는 위 네 callback을 실제 AK MIT CAN 프레임으로 구현한 이식용 C 모듈이다.
`qd_ak_motor_board()`가 `qd_board`를 만들고, 펌웨어는 CAN 송신 함수만 넘긴다. 호스트 GCC 시험은 14개다.

`board/`는 이 어댑터를 쓰는 NUCLEO-G474RE 펌웨어다. 클럭·LPUART1·FDCAN 핀 배치는 2026-10-20 트랜시버 시험 fixture와 같다.
FDCAN은 정상 모드, 프레임당 한 번만 전송(자동 재전송 끔)한다. 수신은 IRQ에서 소프트웨어 링으로 옮긴다.
비트레이트는 `CAN_KBPS`(기본 1000, 5000의 약수)로 빌드할 때 정한다. 송신 대기 한도는 비트 시간에 비례한다(1 Mbit/s 0.5 ms, 250 kbit/s 2 ms).
속도마다 `BUILD`를 따로 쓴다. 250 kbit/s 측정은 [G474 CAN 250 kbit/s 시험](../../reports/v15/jetson_g474_can250_20261022/README.md)에 있다.
`board/g474_ak_config.c`는 아직 전부 0인 미검증 설정이라 모터에 아무것도 보내지 않는다.
`tools/g474_ak_swd.py`는 플래시·진단 읽기와, 꺼짐 상태에서 지정 ID 하나에 휴지 프레임 1개를 보내는 probe를 제공한다.
probe 응답과 설정에 없는 모든 수신 프레임은 진단 영역 `unknown[16]`에 남아 응답 ID 확인에 쓴다.
빌드는 `make -C board VENDOR=<ST HAL/CMSIS 경로>`다.
Jetson 벤치 결과는 [G474 AK 펌웨어 시험](../../reports/v15/jetson_g474_ak_motor_20261020/README.md)에 기록한다.

| 동작 | 구현 |
|---|---|
| 설정 | 관절별 프로토콜·버스·ID·응답 ID·부호·영점·명령 범위·응답 환산값. `verified=QD_AK_CONFIG_VERIFIED`와 검증 통과 전에는 송신 0, ready=false |
| 설정 검증 | 부호 ±1, 같은 버스 ID 중복, 응답 ID가 다른 모터의 명령 ID와 겹침, 범위 비유한/0, 영점이 범위 밖이면 거부 |
| apply | 12개를 모두 먼저 인코딩한다. 하나라도 범위를 넘으면 자르지 않고 전송 0개로 거부한다. 레거시는 처음 한 번 모터 모드 진입(FC)을 보낸다 |
| disarm·poll | 레거시는 모터 모드 종료(FD), V3는 게인·속도·토크 0 MIT 프레임. 한 모터 송신이 실패해도 나머지 모터에 계속 보낸다 |
| read·ready | 12개 모두 10 ms 이내 응답이 있고 오류 코드 0일 때만 true. 한 번 오래된 응답은 새 응답 전까지 쓰지 않는다 |

비트 배치와 float↔uint 변환식의 출처는 CubeMars 공식 매뉴얼이다(2026-10-20 내려받음).
레거시 MIT는 [AK Series Module Driver User Manual V1.0.15.X](https://www.cubemars.com/images/file/20240611/1718085712815162.pdf) 5.3,
SHA256 `1471508a08542221a7f5a81eb56a7c1403d3af7de074f5b7e83bed1fe93c4fee`.
V3 MIT는 [AKA Series Module Product Manual V3.0.0](https://www.cubemars.com/images/file/20250207/1738921575887499.pdf) 4.2·4.3.1,
SHA256 `0de26957898535c6bc0253ee9a8d18b9e4c235a22b93da99287b2994d281819f`.
매뉴얼 예제의 `float_to_uint`는 x=max에서 상위 비트로 넘쳐 0이 되므로 최대 코드로 묶었다. 범위 밖 값은 예제처럼 자르지 않고 거부한다.

다음 값은 코드에 넣지 않았다. 실제 모터에서 확인해 설정에 채워야 한다.

| 항목 | 근거 상태 |
|---|---|
| AK70-10의 펌웨어 종류(레거시/V3) | 2026-10-22 사용자 확인: AK70-10은 V2, AK10-9는 V3. V2는 AK70-10 범위표가 있는 레거시 매뉴얼(V1.0.15.X)로 본다 |
| CAN 비트레이트 | 두 매뉴얼 사양표는 1Mbps. 레거시는 "Not recommended to change"이고 설정 화면에 비트레이트 칸이 없다. V3 Basic Settings에는 CAN Bitrate 칸이 있으나 선택값이 없다 |
| AK10-9 V3 속도·토크 범위 | 같은 V3 매뉴얼 안에서 범위표 ±28 rad/s·±54 N·m, 예제 코드 ±33 rad/s·±65 N·m로 다르다 |
| AK70-10 레거시 범위 | V1.0.15.X 범위표 위치 ±12.5 rad·속도 ±50 rad/s·토크 ±25 N·m·Kp 0–500·Kd 0–5. 예제 코드는 ±30·±18로 다르다 |
| 응답 프레임 ID | 레거시는 "0X00+Driver ID"로만 적혀 있다. V3 업로드 ID는 매뉴얼에 없다(가상 시험의 `0x2900`+ID는 출처 없음) |
| V3 응답 단위 | 위치 0.1°, 전기 RPM, 전류 A다. 출력축 rad·rad/s·N·m 환산에 극쌍 수·감속비·토크 상수 확인이 필요하다 |
| 레거시 응답 토크 범위 | 예제가 정의되지 않은 `I_MAX`를 쓴다 |
| 꺼짐 상태 응답 | 레거시가 FD에도 상태를 응답하는지, V3 업로드가 주기형인지 확인한다. 응답이 없으면 ready=false로 남아 구동하지 않는다 |
| 하드웨어 차단 | V3 매뉴얼에 모터 모드 종료 명령이 없다. CAN이 끊기면 MCU가 모터를 멈출 수 없으므로 전원 차단·비상정지 회로가 별도로 필요하다 |

## NUCLEO-G474RE / MB1367 HAL 연결 예제

MB1367의 기본 STLINK-V3E VCP 연결은 LPUART1 PA2(TX)/PA3(RX)이며 보드 revision에 맞는 기본 solder bridge 상태를 확인한다.
호스트 USB 케이블은 STLINK의 CN1에 연결한다. 이 경로는 MCU USB CDC 주변장치를 사용하는 방식과 다르다.
보드 회로/bridge 경로는 [ST UM2505](https://www.st.com/resource/en/user_manual/dm00556337-stm32g4-nucleo64-boards-mb1367-stmicroelectronics.pdf)를 확인한다.

CubeMX에서 실제 NUCLEO-G474RE 프로젝트를 만든 뒤 아래 포팅 지점을 연결한다:

1. LPUART1 921600 baud, 8 data bits, no parity, 1 stop bit, TX/RX, no flow control.
   PA2/PA3의 올바른 alternate function과 kernel clock을 CubeMX에서 생성한다.
2. RX DMA circular, TX DMA normal, byte width, memory increment. UART global IRQ와 RX/TX DMA IRQ를 활성화한다.
   RX half-transfer/transfer-complete/IDLE event 모두 유지한다. UART와 RX DMA callback이 서로 중첩되지 않도록 IRQ 우선순위를 설정한다.
3. TIM6는 예제 이름이다. 실제 timer clock으로 정확한 5 ms update IRQ를 생성한다.
   SysTick/HAL tick도 동작해야 하며 watchdog과 timer IRQ를 다른 긴 ISR이 가리지 않게 한다.
4. 두 공통 `src/*.c`, include 경로, `examples/g474_hal_port.c`, `examples/board_hooks_disabled.c`를 추가하고
   `QD_STM32_HAL_EXAMPLE`을 정의한다. 예제가 기대하는 `hlpuart1`, `htim6` handle 이름을 실제 생성 코드와 맞춘다.
5. 생성된 초기화 뒤 `qd_g474_start()`, main loop에서 계속 `qd_g474_poll()`을 호출한다.
   HAL callbacks는 기존 callbacks와 병합한다. 중복 정의하지 않는다.

예제는 RX ISR에서 bounded queue로 복사하고 foreground에서 parser/controller를 처리한다.
TX DMA buffer는 완료 callback 전까지 보존된다. RX overflow, UART 오류, TX queue 포화, 타이머 처리 누락은 IO fault를 만들고 disarm한다.
IRQ watchdog은 foreground가 지연되어도 `qd_g474_board_disarm_from_isr()`를 호출한다.
이 hook은 CAN 응답을 기다리지 않는 실제 enable 철회 경로로 구현해야 한다.
UART 오류 후 DMA 복구는 선택한 HAL 버전에 맞춰 foreground에서 완성해야 한다.
FEEDBACK stamp는 timer IRQ마다 정확히 5 ms씩 증가하는 64 bit 논리 시각이다.
foreground는 IRQ를 잠시 막고 pending tick 수와 시각을 함께 캡처한다.
처리할 tick이 둘 이상이면 IO fault와 시각 공백이 유지된다. 실제 watchdog 경과 시간은 HAL_GetTick으로 측정한다.
HELLO는 처리 전 IRQ fault epoch를 캡처한다. 다른 session의 HELLO이고 그 뒤 새 ISR fault가 없을 때만 이전 IRQ fault를 해제한다.
동일한 fault bit가 다시 발생해도 epoch가 바뀌므로 새 오류를 지우지 않는다.
DMA/callback API의 근거는 [ST 공식 HAL UART 확장 코드](https://github.com/STMicroelectronics/stm32g4xx-hal-driver/blob/master/Src/stm32g4xx_hal_uart_ex.c)다.

COMMAND 268 bytes × 200 Hz는 방향당 536000 bit/s(8N1)이며,
FEEDBACK 180 bytes × 200 Hz는 반대 방향으로 360000 bit/s다.
실제 STLINK baud 지원, DMA/IRQ 부하, CAN 전달 시간과 오실로스코프 측정은 보드에서 확인해야 한다.

## Native emulator

`build/qd_emulator`는 raw frame stdin을 파싱하고 ACK/FEEDBACK raw frame을 stdout에 쓴다.
stderr는 가상 실행임을 표시한다. 실제 UART/CAN을 열지 않는다.
기본은 enable 거부와 명시된 가상 영 상태다. `--virtual-enable`만 가상 활성 명령을 허용한다.
기본 FEEDBACK은 5 ms 예약 틱의 논리 시각으로 주기당 하나만 전송한다. COMMAND/STOP마다 추가 FEEDBACK을 만들지 않는다.
Linux 스케줄링으로 틱을 놓치면 시각 공백이 드러나며 오래된 측정값에 과거 틱 시각을 붙여 채우지 않는다.
`--frame-driven`은 HELLO에 ACK, COMMAND/STOP에 FEEDBACK을 반환하는 단건 프로토콜 시험 모드다.
이 모드의 응답 시각은 실제 monotonic 시각이며 5 ms 격자를 사용하는 gateway 입력으로 사용하지 않는다.
두 모드 모두 real monotonic 시간으로 watchdog을 검사한다.

```sh
firmware/stm32g474_usb/build/qd_emulator --frame-driven --virtual-enable --measurement /tmp/measured.bin
```

`--measurement` 파일은 헤더/CRC 없는 156-byte `<III36f` FEEDBACK payload다.
마지막 36 floats의 q/dq/tau를 각 read마다 다시 읽는다. 파일의 처음 세 uint32는 상태 주입에 사용하지 않는다.
시뮬레이터는 임시 파일을 완전히 쓴 뒤 rename하여 새 측정값을 원자적으로 교체할 수 있다.
파일이 짧거나 없거나 수치가 유효하지 않으면 상태 읽기가 실패한다.
이 경로는 실제 상태를 C 가드에 넣는 소프트웨어 시험용이며 MCU/CAN 지연이나 실제 구동 검증을 대신하지 않는다.

`build/libqd_protocol.so`로 같은 공개 API를 ctypes 등에서 호출할 수도 있다.
`tests/test_cross_language.py`에 frame/command/measurement의 ctypes 선언과 호출 예제가 있다.
