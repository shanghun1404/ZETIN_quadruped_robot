"""Two 15-minute runs on G474 v3; physical RX checks before and after each."""
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import g474_control as control

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'stress_30min_20261020_1254'
PORT = '/dev/serial/by-id/usb-STMicroelectronics_STLINK-V3_004F002D3235511137333439-if02'
SECONDS = 900
WARMUP = 5
EXPECTED_COMMANDS = (SECONDS + WARMUP) * 200


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def run_command(args, log, timeout):
    with log.open('w') as output:
        return subprocess.run(args, cwd=ROOT, stdout=output,
                              stderr=subprocess.STDOUT, timeout=timeout).returncode


def physical_check(mode, phase):
    folder = RUN / 'captures' / f'{mode}-{phase}'
    folder.mkdir(parents=True)
    control.configure(mode, capture=True, mode=1)
    rc = run_command([sys.executable, 'probe_link.py', '--port', PORT, '--capture',
                      '--output', str(folder/'uart.json')], folder/'uart.log', 20)
    if rc:
        raise RuntimeError(f'{mode} {phase}: UART probe failed')
    counters = control.snapshot(folder/f'{mode}-pad', capture=True)
    failures = control.verify_counts(counters, 1)
    rc = run_command([sys.executable, 'verify_capture.py', mode, '--run-dir', str(folder)],
                     folder/'verify.log', 20)
    verification = json.loads((folder/f'{mode}-pad/verification.json').read_text())
    if rc or not verification['passed'] or failures:
        raise RuntimeError(f'{mode} {phase}: physical RX verification failed: {failures}')
    return dict(passed=True, tx_frames=verification['tx_verified'],
                rx_frames=verification['rx_verified'], verified_utc=utc())


def render(state):
    rows = []
    for mode, item in state['modes'].items():
        host, counters = item['host'], item['mcu']
        measured = host['measured']
        rows.append(f'<tr><td>{mode.upper()}</td><td>{host["measured_elapsed_seconds"]:.6f}s</td>'
                    f'<td>{measured["commands"]:,}</td><td>{host["final_ack"]:,}</td>'
                    f'<td>{host["errors"]}/{counters["can_errors"]}</td>'
                    f'<td>{measured["deadline_misses_over_5ms"]}</td>'
                    f'<td>{"PASS" if item["passed"] else "FAIL"}</td></tr>')
    error = html.escape(state.get('error', ''))
    (RUN/'report.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>G474 트랜시버 30분 시험</title><style>body{font:18px system-ui;max-width:1100px;margin:40px auto;line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:12px}</style><h1>G474 트랜시버 30분 시험</h1><p>MIT 900초 + 서보 900초 측정. 각 5초 예열은 별도다. 12관절, 200Hz, CAN1=PA12/PA11, CAN2=PB13/PB12, CAN3=PA15/PA8. 사용자가 연결한 TJA1051T/3, E5V=5V, VIO=3.3V 구성을 사용했다.</p><p>각 모드 전후에 TX 및 실제 RX 패드를 기록해 명령12개·가상 응답12개의 CAN ID·데이터·CRC를 확인한다. 연속 운용은 FDCAN external loopback과 가상 AK 응답으로 검사한다. 실제 RX 전체 비트열을 30분 내내 검증하는 시험은 아니다. 모터·외부 CAN 노드는 없으며 외부 ACK, 실모터 각도 추종, AI·WBC 동시 부하는 포함하지 않는다.</p><table><tr><th>모드</th><th>측정 시간</th><th>측정 명령</th><th>최종 ACK(예열포함)</th><th>호스트/CAN 루프백 오류</th><th>송신 일정 지연&gt;5ms</th><th>프로토콜 판정</th></tr>''' + ''.join(rows) + f'''</table><p>{error}</p><p>지연 수치는 별도 지표이며 프로토콜 PASS가 하드 실시간 주기 충족을 뜻하지 않는다. 모든 QD 명령은 비활성, 속도·게인·요청 토크는 0이다. 모터 활성화 명령은 보내지 않았다.</p><p><a href="result.json">전체 결과</a> · <a href="mit/summary.json">MIT</a> · <a href="servo/summary.json">서보</a> · <a href="provenance.json">소스 해시</a></p></html>''')


def main():
    lock = (ROOT/'.stress.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RUN.mkdir(exist_ok=False)
    state = dict(pid=os.getpid(), started_utc=utc(), stage='preparing',
                 measured_seconds_total=1800, seconds_per_mode=SECONDS, modes={}, captures={})

    def status(stage):
        state['stage'] = stage
        state['updated_utc'] = utc()
        save(RUN/'status.json', state)

    status('preparing')
    try:
        names = ['build/g474_can_fixture.elf', 'build/g474_can_fixture.bin',
                 'firmware/g474_can_fixture.c', 'firmware/g474_diag.h', 'core/src/qd_protocol.c',
                 'core/examples/qd_ak_virtual.c', 'ak_stress.py', 'g474_control.py',
                 'verify_capture.py', 'run_stress_30min.py', 'pin_map.json']
        save(RUN/'provenance.json', dict(host=os.uname().nodename,
             sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}))
        before = control.snapshot(RUN/'initial_state')
        if any(before[k] for k in ('uart_errors', 'rx_overflows', 'tx_errors')):
            raise RuntimeError('Pre-existing MCU UART error; initial evidence preserved')
        for mode in ('mit', 'servo'):
            status(f'physical_before_{mode}')
            state['captures'][f'{mode}_before'] = physical_check(mode, 'before')
            status(f'running_{mode}')
            directory = RUN/mode
            rc = run_command([sys.executable, 'ak_stress.py', '--port', PORT, '--mode', mode,
                              '--seconds', str(SECONDS), '--warmup-seconds', str(WARMUP),
                              '--output-dir', str(directory)], RUN/f'{mode}.log', SECONDS + WARMUP + 60)
            status(f'verifying_{mode}')
            host = json.loads((directory/'summary.json').read_text())
            mcu = control.snapshot(directory/'swd')
            failures = control.verify_counts(mcu, EXPECTED_COMMANDS)
            if mcu['missed_ticks']:
                failures.append(f'missed_ticks={mcu["missed_ticks"]}')
            item = dict(host=host, mcu=mcu, exit_code=rc, counter_failures=failures,
                        passed=rc==0 and host['passed'] and not failures
                        and host['measured_elapsed_seconds'] >= SECONDS)
            state['modes'][mode] = item
            save(RUN/f'{mode}_verified.json', item)
            if not item['passed']:
                raise RuntimeError(f'{mode} stress failed; evidence preserved')
            status(f'physical_after_{mode}')
            state['captures'][f'{mode}_after'] = physical_check(mode, 'after')
        state['passed'] = True
        state['finished_utc'] = utc()
        status('complete')
    except BaseException as error:
        state['passed'] = False
        state['error'] = f'{type(error).__name__}: {error}'
        state['finished_utc'] = utc()
        status('failed')
        raise
    finally:
        try:
            control.configure('mit', mode=1)
        except Exception as error:
            state['cleanup_error'] = repr(error)
            state['passed'] = False
            status('failed')
        save(RUN/'result.json', state)
        render(state)


if __name__ == '__main__':
    main()
