"""Run each confirmed AK profile for a measured hour; no motor enable path."""
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import g474_control as control

ROOT=Path(__file__).resolve().parent
PORT='/dev/serial/by-id/usb-STMicroelectronics_STLINK-V3_004F002D3235511137333439-if02'
RUN=ROOT/'stress_mit_servo_1h'

def utc():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def command(args,log):
    with Path(log).open('w') as out:
        return subprocess.run(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT).returncode

def report(result):
    rows=[]
    for mode,item in result['modes'].items():
        host=item['host'];m=host['measured'];swd=item['swd']
        rows.append(f'<tr><td>{mode.upper()}</td><td>{host["measured_elapsed_seconds"]:.6f}s</td><td>{m["commands"]:,}</td><td>{host["final_ack"]:,}</td><td>{host["errors"]}</td><td>{swd["can_errors"]}</td><td>{m["max_target_error_rad"]:.8f}rad</td><td>{"PASS" if item["passed"] else "FAIL"}</td></tr>')
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>G474 AK virtual stress test</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;color:#17212e}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:12px}code{background:#eee}p{line-height:1.7}</style><h1>G474 AK 가상 프레임 시험</h1><p>HAA=AK70-10, HFE·KFE=AK10-9 KV60 V3. CAN1=HAA, CAN2=HFE, CAN3=KFE. 각 버스에 ID1~4, 총12관절. 200Hz, 모드별 5초 예열 + 3,600초 측정.</p><p>실제 Jetson USB 직렬 통신과 G474의 FDCAN 3개를 사용했습니다. External loopback에서 CAN_TX 물리 패드를 GPIOB IDR로 표본화해 명령12개·가상 응답12개의 ID, DLC, 데이터, CRC를 각각 확인했습니다. 트랜시버·모터는 연결되지 않았으므로 CAN_H/L, 외부 ACK, 실제 모터 각도·전류·기계 동작은 검증 범위에 포함되지 않습니다.</p><p>가상 응답은 CAN에서 수신한 명령 바이트의 목표 각도를 복원하는 프로토콜 에뮬레이터입니다. MIT 동역학이나 실제 추종 성능을 뜻하지 않습니다. 모든 QD 명령은 비활성이며 velocity·kp·kd·effort=0입니다. 모터 활성화/원점 설정/모드 변경 CAN 메시지는 보내지 않습니다.</p><table><tr><th>모드</th><th>실제 측정 시간</th><th>측정 명령</th><th>최종ACK (예열포함)</th><th>호스트 오류</th><th>CAN 오류</th><th>최대 목표각 양자화차</th><th>결과</th></tr>'''+''.join(rows)+'''</table><p>PASS는 명령 수·ACK·소스 피드백 연속성·가상 각도·비활성 상태·MCU 프레임별 송수신 횟수와 오류 카운터를 함께 검사한 결과입니다. Linux 호스트의 5ms 초과 일정 지연과 ACK 지연은 summary.json에 별도로 남겼으며 하드 실시간 보장을 뜻하지 않습니다.</p><p>원본 펌웨어 512KiB 백업과 빌드/플래시 검증 기록을 보존했습니다. 최초 200Hz 실패는 패킷 파서의 바이트별 반복 검색 때문에 생겼고, 묶음 단위 검색으로 수정한 뒤 장시간 시험을 시작했습니다.</p><p>상세 결과: <a href="result.json">result.json</a>, <a href="mit/summary.json">MIT summary</a>, <a href="servo/summary.json">Servo summary</a>. 원시 UART 바이트는 각 모드 uart_rx.bin/uart_tx.bin에 저장했습니다.</p></html>'''
    (RUN/'report.html').write_text(body)


def main():
    RUN.mkdir(exist_ok=True)
    lock=(ROOT/'.stress.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),stage='preparing',started_utc=utc(),scope='real G474 FDCAN external loopback + virtual AK angle; no transceiver/motor',modes={})
    if (RUN/'result.json').exists():raise RuntimeError('Refusing to overwrite a completed run')
    def status(stage):state['stage']=stage;state['updated_utc']=utc();save(RUN/'status.json',state)
    try:
        files=['build/g474_can_fixture.elf','build/g474_can_fixture.bin','ak_stress.py','g474_control.py','run_modes.py','firmware/g474_can_fixture.c','firmware/g474_diag.h','core/src/qd_protocol.c','core/examples/qd_ak_virtual.c','original_flash_512k.bin']
        save(RUN/'provenance.json',dict(start_utc=utc(),host=os.uname().nodename,files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}))
        for mode in ('mit','servo'):
            # No UART owner is active while SWD halts/reconfigures the target.
            status(f'capture_{mode}')
            control.configure(mode,capture=True)
            rc=command([sys.executable,str(ROOT/'probe_link.py'),'--port',PORT,'--capture','--output',str(RUN/f'capture-{mode}-probe.json')],RUN/f'capture-{mode}-probe.log')
            if rc:raise RuntimeError(f'{mode} capture probe failed')
            captured=control.snapshot(ROOT/f'capture-{mode}',capture=True)
            if control.verify_counts(captured,1):raise RuntimeError(f'{mode} capture counts failed')
            rc=command([sys.executable,str(ROOT/'verify_capture.py'),mode],RUN/f'capture-{mode}-decode.log')
            if rc:raise RuntimeError(f'{mode} physical TX-pad decode failed')
            # Capture requests are now clear. HELLO in the runner resets run counters.
            status(f'running_{mode}')
            directory=RUN/mode
            rc=command([sys.executable,str(ROOT/'ak_stress.py'),'--port',PORT,'--mode',mode,'--seconds','3600','--warmup-seconds','5','--output-dir',str(directory)],RUN/f'{mode}.log')
            status(f'verifying_{mode}')
            host=json.loads((directory/'summary.json').read_text())
            swd=control.snapshot(directory/'swd')
            failures=control.verify_counts(swd,721000)
            if swd['missed_ticks']:failures.append(f'missed_ticks={swd["missed_ticks"]}')
            item=dict(host=host,swd=swd,exit_code=rc,counter_failures=failures,
                      passed=rc==0 and host['passed'] and host['measured_elapsed_seconds']>=3600 and not failures,
                      verified_utc=utc())
            state['modes'][mode]=item;save(RUN/f'{mode}_verified.json',item)
            if not item['passed']:raise RuntimeError(f'{mode} stress validation failed; evidence preserved')
        state['passed']=True;state['finished_utc']=utc();status('complete');save(RUN/'result.json',state);report(state)
    except BaseException as error:
        state['passed']=False;state['error']=f'{type(error).__name__}: {error}';state['finished_utc']=utc();status('failed');save(RUN/'result.json',state);report(state)
        raise

if __name__=='__main__':main()
