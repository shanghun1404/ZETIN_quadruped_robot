# G474 실제 AK 펌웨어 벤치 시험 (2026-10-20)

Jetson Orin Nano Super에서 `firmware/stm32g474_usb/board/`를 빌드해 NUCLEO-G474RE에 플래시했다.
모터와 외부 CAN 노드는 없다. 트랜시버는 직전 시험과 같은 TJA1051T/3 구성이다.
설정은 미검증(전부 0)이므로 이 시험은 "설정 없이는 아무것도 구동하지 않는다"는 게이트와 통신 경로만 확인한다.

| 단계 | 결과 |
|---|---|
| 빌드 | arm-none-eabi-gcc 13.2.1, `-Wall -Wextra -Werror`, FLASH 25,220 B, RAM 15,272 B |
| 플래시 | `** Verified OK **`. 직전 플래시 512 KiB를 먼저 덤프했고 트랜시버 fixture v3 빌드와 일치했다 |
| 부팅 | boot_stage 3, 160 MHz, config_valid 0, CAN 송신 0 |
| USB 게이트 | HELLO_ACK capability 1. 비활성 명령 1,000개 수락, 0 게인·0 토크 활성 명령 1개는 거부(fault 8 = 보드 미준비). FEEDBACK 0개, CAN 송신 0 |
| probe | 버스 1 레거시 ID1, 버스 2·3 V3 ID1에 휴지 프레임 각 1개. 각 버스 송신 1, LEC 3(ACK 오류), TEC 8, 수신 0 |
| 30초 | 새 세션에서 fault 0으로 해제, 비활성 명령 6,000개 수락, UART 오류·오버플로·missed tick 0, 추가 CAN 송신 0 |

FEEDBACK이 0개인 것은 의도한 동작이다. 실제 모터 상태가 없으면 가짜 상태를 만들지 않는다.
LEC 3과 TEC 8은 트랜시버 전원이 정상이고 응답할 노드가 없을 때의 결과이며, 한 번만 전송했음을 뜻한다.
실모터 응답, 응답 ID, 단위, 부호·영점과 구동은 시험하지 않았다.

| 파일 | SHA256 |
|---|---|
| `build/g474_ak_firmware.elf` | `dfbf7be73e8742135db9be74e14061d0d7858d160e1ca6b8113df0fe6a5fba65` |
| `build/g474_ak_firmware.bin` | `2fc1b965f4bfe9627d2301604e9821a38b92b2dc9e4f435c68de6af4a961a2ed` |
| 이전 플래시 덤프(Jetson `~/robot_dog/g474_ak_motor_20261020/backup/`) | `b3338c23c9c04bad29b879761704d2b878915d7b64cde491c7ce282efd7fea04` |

원본 기록: [results/](results/) (단계별 SWD 진단 JSON, USB 결과 JSON, 플래시 로그).
