#!/usr/bin/env python3
"""SWD flash/snapshot/probe for board/g474_ak_firmware.

probe sends ONE rest frame (LEGACY exit-motor-mode or V3 zero-gain MIT)
to a single CAN ID while the firmware is disarmed. It never enables a motor.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

SCALARS = '''magic version boot_stage clock_hz rx_bytes rx_overflows uart_errors uart_tx_errors
frames_received commands_received commands_accepted commands_rejected feedback_sent missed_ticks
armed fault_flags config_valid adapter_rejected adapter_send_failures polls can_ring_overflows
probe_request probe_done probe_bus probe_protocol probe_id probe_sent unknown_count'''.split()
BUS = 'tx_queued tx_full tx_errors rx_frames rx_lost bus_off last_psr last_ecr'.split()
LOG = 'bus extended id dlc data_lo data_hi tick_ms'.split()
UNKNOWN = 16
DIAG_BYTES = (len(SCALARS) + 3 * len(BUS) + UNKNOWN * len(LOG)) * 4
PROTOCOLS = {'legacy': 1, 'v3': 2}


def symbol(elf, name):
    for line in subprocess.check_output(['arm-none-eabi-nm', str(elf)], text=True).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise SystemExit(f'{name} not found in {elf}')


def openocd(args, commands, log):
    cmd = [args.openocd, '-s', args.scripts, '-f', args.cfg]
    for item in commands:
        cmd += ['-c', item]
    with Path(log).open('w') as output:
        subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=60)


def snapshot(args, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = symbol(args.elf, 'g_diag')
    openocd(args, ['init', 'halt', f'dump_image {output}/diag.bin 0x{base:08x} {DIAG_BYTES}',
                   'resume', 'shutdown'], output / 'swd.log')
    words = list(struct.unpack(f'<{DIAG_BYTES // 4}I', (output / 'diag.bin').read_bytes()))

    def take(keys):
        row = dict(zip(keys, words[:len(keys)]))
        del words[:len(keys)]
        return row
    state = take(SCALARS)
    state['bus'] = [take(BUS) for _ in range(3)]
    log = [take(LOG) for _ in range(UNKNOWN)]
    assert not words and state['magic'] == 0x47414b4d and state['version'] == 1, state
    count = state['unknown_count']
    order = range(max(0, count - UNKNOWN), count)
    state['unknown'] = [log[i % UNKNOWN] for i in order]
    for row in state['bus']:
        row['last_lec'] = row['last_psr'] & 7
        row['tec'] = row['last_ecr'] & 0xff
    (output / 'diag.json').write_text(json.dumps(state, indent=2) + '\n')
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--elf', type=Path, required=True)
    p.add_argument('--openocd', required=True)
    p.add_argument('--scripts', required=True)
    p.add_argument('--cfg', required=True)
    sub = p.add_subparsers(dest='action', required=True)
    f = sub.add_parser('flash')
    f.add_argument('output', type=Path)
    s = sub.add_parser('snapshot')
    s.add_argument('output', type=Path)
    r = sub.add_parser('probe')
    r.add_argument('output', type=Path)
    r.add_argument('--bus', type=int, choices=(1, 2, 3), required=True)
    r.add_argument('--protocol', choices=sorted(PROTOCOLS), required=True)
    r.add_argument('--id', type=int, required=True)
    a = p.parse_args()
    if a.action == 'flash':
        a.output.mkdir(parents=True, exist_ok=True)
        openocd(a, [f'program {a.elf.resolve()} verify reset exit'], a.output / 'flash.log')
        log = (a.output / 'flash.log').read_text()
        assert '** Verified OK **' in log, 'flash verification missing'
        print(json.dumps({'flashed': str(a.elf), 'verified': True}))
        return
    if a.action == 'snapshot':
        print(json.dumps(snapshot(a, a.output)))
        return
    assert 1 <= a.id <= 255
    before = snapshot(a, a.output / 'before')
    assert not before['armed'], 'refusing to probe while armed'
    base = symbol(a.elf, 'g_diag')
    addr = {k: base + SCALARS.index(k) * 4 for k in ('probe_bus', 'probe_protocol', 'probe_id', 'probe_request')}
    openocd(a, ['init', 'halt', f'mww 0x{addr["probe_bus"]:08x} {a.bus}',
                f'mww 0x{addr["probe_protocol"]:08x} {PROTOCOLS[a.protocol]}',
                f'mww 0x{addr["probe_id"]:08x} {a.id}',
                f'mww 0x{addr["probe_request"]:08x} {before["probe_request"] + 1}',
                'resume', 'shutdown'], a.output / 'request.log')
    time.sleep(0.2)
    after = snapshot(a, a.output / 'after')
    b = a.bus - 1
    result = dict(bus=a.bus, protocol=a.protocol, id=a.id,
                  handled=after['probe_done'] == before['probe_request'] + 1,
                  sent=after['probe_sent'] - before['probe_sent'],
                  tx_queued=after['bus'][b]['tx_queued'] - before['bus'][b]['tx_queued'],
                  rx_frames=after['bus'][b]['rx_frames'] - before['bus'][b]['rx_frames'],
                  last_lec=after['bus'][b]['last_lec'], tec=after['bus'][b]['tec'],
                  new_unknown=after['unknown'][len(after['unknown']) - min(
                      UNKNOWN, after['unknown_count'] - before['unknown_count']):]
                  if after['unknown_count'] > before['unknown_count'] else [])
    (a.output / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
