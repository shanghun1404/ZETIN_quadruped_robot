# G474 CAN 250 kbit/s 시험 (2026-10-22, 500 kbit/s 비교 추가)

질문: CAN을 250 kbit/s로 해도 목표 제어가 되는가.
목표는 [V15 Jetson USB 설계](../../../docs/superpowers/specs/2026-10-12-v15-jetson-usb-design.md)의 값이다. 12관절 목표각·속도·Kp·Kd·토크를 200 Hz로 보내고, 실제 q/dq/토크를 200 Hz로 받는다.
배선은 버스마다 4개 모터다. CAN1에 HAA AK70-10(사용자 확인 V2, 레거시 MIT, 표준 ID), CAN2·CAN3에 HFE·KFE AK10-9 V3(확장 ID)가 붙는다.
그래서 한 버스는 5 ms마다 명령 4개와 응답 4개, 모두 8바이트인 프레임 8개를 실어야 한다.

모터와 외부 CAN 노드는 없다. 트랜시버는 2026-10-20 시험과 같은 TJA1051T/3 구성이다.

## 방법

- `bench/g474_can_bench.c`는 시험 전용 펌웨어다. 클럭·핀은 AK 펌웨어와 같다.
  - 먼저 정상 모드에서 버스마다 모터가 아닌 ID로 프레임을 1개 보낸다. 3개 버스 모두 ACK 오류가 나야 다음 단계로 간다. 그 속도로 듣는 노드가 없다는 뜻이다.
  - 그다음 외부 루프백으로 바꾼다. 각 버스가 실제 TX 핀과 트랜시버로 송신하고 자기 프레임을 받는다.
  - 버스 1은 표준 8바이트, 버스 2·3은 확장 8바이트 프레임이다. 데이터는 전부 0과 xorshift 난수 두 가지다.
  - burst는 버스마다 1,000개를 연속으로 보낸다. 3개 버스를 동시에 돌린다.
  - schedule은 5 ms마다 버스당 8개를 보내고, 8번째 프레임을 받을 때까지 걸린 시간을 잰다. 패턴마다 2,000주기(10초)다.
- 결과는 측정이 끝난 뒤 SWD로 읽는다(`bench/run_bench.py`). 측정 중에는 디버거가 개입하지 않는다.
- 이어서 실제 AK 펌웨어를 `CAN_KBPS=250`으로 빌드해 플래시했다. USB 게이트, probe, 30초 시험은 2026-10-20 절차와 같다.
- 사용자가 분기선(stub) 40 cm 이하를 보장할 수 없다고 해서, 같은 벤치를 500 kbit/s로도 돌렸다. 그 뒤 보드에는 250 kbit/s AK 펌웨어를 다시 올렸다.

## 결과

| 항목 | 250 kbit/s | 500 kbit/s | 1 Mbit/s |
|---|---|---|---|
| 비트 타이밍(NBTP) | 분주 20, 13+2 tq, 샘플점 87.5 %, 3개 버스 모두 | 분주 10, 같은 tq | 분주 5, 같은 tq |
| 실측 비트레이트 | 250.000–250.007 kbit/s | 500.001–500.016 kbit/s | 1000.003–1000.035 kbit/s |
| 표준 프레임 1개 | 460.0 µs(115 bit, 난수) / 504.0 µs(126 bit, 0) | 230.0 / 252.0 µs | 115.0 / 126.0 µs |
| 확장 프레임 1개 | 547.9–548.1 µs(137.0 bit) / 590.5 µs(147.6 bit) | 274.0–274.1 / 295.3 µs | 137.0 / 147.6 µs |
| 버스 1(표준), 8프레임 완료 | 평균 3.68–4.03 ms, 최대 4.03 ms (5 ms의 74–81 %) | 평균 1.84–2.02 ms, 최대 2.02 ms (37–40 %) | 평균 0.92–1.01 ms, 최대 1.01 ms |
| 버스 2·3(확장), 8프레임 완료 | 평균 4.39–4.73 ms, 최대 4.73 ms (5 ms의 88–95 %) | 평균 2.20–2.37 ms, 최대 2.38 ms (44–48 %) | 평균 1.11–1.20 ms, 최대 1.20 ms |
| 주기 넘침(다음 주기 전까지 미완료) | 0 / 4,000주기 × 3버스 | 0 | 0 |
| 주기 시작 지연 100 µs 초과 | 0 | 0 | 0 |

이론 최악값도 계산했다. 8바이트 프레임의 비트 스터핑 최대치를 넣으면 표준은 135 bit, 확장은 160 bit다(프레임 간 3 bit 포함).
250 kbit/s에서 8프레임은 표준 4.32 ms, 확장 5.12 ms가 되어 확장 버스는 5 ms를 넘는다. 500 kbit/s에서는 2.16 ms, 2.56 ms, 1 Mbit/s에서는 1.08 ms, 1.28 ms다.

실제 AK 펌웨어(`CAN_KBPS=250`) 결과:

| 단계 | 결과 |
|---|---|
| 빌드 | FLASH 25,220 B, RAM 15,272 B. 같은 소스의 1 Mbit/s 빌드 BIN은 2026-10-20 플래시본과 바이트 단위로 같다 |
| 플래시·부팅 | `** Verified OK **`, boot_stage 3, 160 MHz, config_valid 0, NBTP 0x02130c01(250 kbit/s) |
| USB 게이트 | capability 1, 비활성 명령 1,000개 수락, 0 게인·0 토크 활성 명령 1개 거부(fault 8). FEEDBACK 0, CAN 송신 0 |
| probe | 버스 1 레거시 ID1, 버스 2·3 V3 ID1에 휴지 프레임 각 1개. 송신 1, LEC 3(ACK 오류), TEC 8, 수신 0 |
| 30초 | 새 세션에서 fault 0, 비활성 명령 6,000개 수락, UART 오류·오버플로·missed tick·CAN 송신 실패 0, 추가 CAN 송신 0 |

## 판단

- 250 kbit/s에서도 200 Hz·버스당 4개 모터는 시간상 들어간다. 그러나 V3 버스는 5 ms 중 4.39–4.73 ms를 쓴다. 남는 시간은 0.27–0.61 ms이고, 최악 스터핑이면 넘친다.
- 이 측정에는 모터가 명령을 받고 응답하기까지의 처리 간격이 없다. 매뉴얼에 이 값이 없다. 오류 프레임도 없다. 이 둘이 더해지면 여유는 더 줄어든다.
- 마지막 모터의 응답은 주기 시작 뒤 최대 약 4.7 ms에 온다. 1 Mbit/s(약 1.2 ms)보다 약 3.5 ms 늦다. Jetson이 받는 q/dq가 그만큼 오래된 값이다.
- 모터 쪽 지원은 확인되지 않았다.
  - 레거시 매뉴얼 V1.0.15.X 1.1절은 CAN 비트레이트를 "1Mbps (Not recommended to change)"로 적는다. 4.1.1.4절 화면에는 비트레이트 칸이 보이지 않는다.
  - V3 매뉴얼 V3.0.0 3.1.1.1절 Basic Settings에는 "CAN Bitrate" 칸(화면 값 1Mbps)이 있다. 고를 수 있는 값은 적혀 있지 않다.
- AK 펌웨어의 CAN 송신 대기 한도 0.5 ms는 250 kbit/s에서 프레임 1개(측정 460–590 µs)보다 짧았다. TX FIFO가 3칸이라 버스의 4번째 명령은 첫 프레임이 끝날 때까지 기다리므로, 그대로였다면 매 주기 실패했을 것이다. 한도를 비트 시간에 비례하게 바꿨다(250 kbit/s에서 2 ms). 설정이 미검증이라 이 경로는 실제 4프레임 송신으로는 시험하지 않았다.

## 분기선(stub) 길이

사용자 조건: 버스에서 모터로 가는 분기선이 40 cm 이하로 유지된다는 보장이 없다. 비트레이트별 분기선 한도는 출처마다 다르다(2026-10-22 확인).

| 출처 | 1 Mbit/s | 500 kbit/s | 250 kbit/s |
|---|---|---|---|
| [TI ISO1044 데이터시트](https://www.ti.com.cn/document-viewer/cn/lit/html/ZHCSL04A/GUID-A6F5DC69-886D-461C-80D0-61375D5EBEF3) | "ISO 11898-2 ... maximum bus length of 40 m and maximum stub length of 0.3 m" (비트레이트 구분 없음) | — | — |
| [CiA CANopen lower layers](https://can-cia.org/can-knowledge/canopen-lower-layers) | stub 1.5 m(누적 7.5 m), 버스 25 m | 5.5 m(27.5 m), 100 m | 11 m(55 m), 250 m |
| [HMS 지원 문서](https://www.hms-networks.com/support/tech-support/kb-articles/6845465781010-What-is-the-maximum-length-of-a-stub-line-in-my-CAN-network-) (표에 단위 없음, m로 읽음) | stub 2 m(누적 10 m) | 5 m(25 m) | 10 m(50 m) |

- 가장 엄격한 0.3 m를 1 Mbit/s에 적용하면 40 cm 분기선은 넘는다. 500 kbit/s와 250 kbit/s는 위 두 표 모두 미터 단위를 허용한다.
- 레거시 매뉴얼 1.2.3절의 AK70-10 CAN 포트는 4핀에 CAN_H 2개, CAN_L 2개다. 핀 배치로 보면 모터를 거쳐 다음 모터로 이어 배선할 수 있어 보이지만, 매뉴얼에 그렇게 적혀 있지는 않다.
- V3 매뉴얼 1.2.2절의 AK10-9 V3 커넥터는 전원·시리얼과 함께 CAN_H/CAN_L 한 쌍뿐이다. 그래서 하네스에서 분기해야 한다.

## 파일

| 파일 | SHA256 |
|---|---|
| `build/fw250/g474_ak_firmware.elf` | `3997ac7117fb089f36122921d3c239fd7e6445bc12995233a582877dd89dbe56` |
| `build/fw250/g474_ak_firmware.bin` | `5c8d2592d51b0a489316ea1f71a16f29c570993eb3365e8e2d0b80d97288bd40` |
| `build/fw1000/g474_ak_firmware.bin` | `2fc1b965f4bfe9627d2301604e9821a38b92b2dc9e4f435c68de6af4a961a2ed` (2026-10-20과 같음) |
| `build/bench250/g474_can_bench.elf` | `4f3daed2643740eb73c834800e7748c3fdc7cb00f10dedc7dc7435b53ba2fdc5` |
| `build/bench500/g474_can_bench.elf` | `5c39523b0fb0f0bd12d84d75708b76ba565916e826ebc5cbc286016ae37424ec` |
| `build/bench1000/g474_can_bench.elf` | `1c72fc18cacb70158e9f8df945ae4c1d4b93c74ad2c87fbf967a69840f4fef0f` |

시험 뒤 보드에는 250 kbit/s AK 펌웨어가 남아 있다(500 kbit/s 벤치 뒤 다시 플래시, `results/fw250/reflash_after_bench500/`).
빌드는 `make -C bench VENDOR=<ST HAL> BOARD=<firmware/stm32g474_usb/board> CAN_KBPS=250`이다.
원본 기록은 [results/](results/)에 있다. 벤치 요약은 `bench*/summary.json`, AK 펌웨어 단계별 진단은 `fw250/`이다.
