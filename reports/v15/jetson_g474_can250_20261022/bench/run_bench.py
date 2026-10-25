#!/usr/bin/env python3
"""Flash g474_can_bench.elf, wait for it to finish, read g_bench over SWD.

The bench needs about 22 s. The script waits without touching SWD, so the
timing loops are not disturbed, then halts the idle core to read (reads of a
core sleeping in WFI without a halt return invalid data on this setup).
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

FIELDS = [('magic', 1), ('version', 1), ('kbps', 1), ('stage', 1), ('faults', 1), ('tx_errors', 1),
          ('rx_errors', 1), ('nbtp', 3), ('interlock_psr', 3), ('interlock_rx', 3), ('interlock_ok', 1),
          ('burst_frames', 6), ('burst_cycles', 6), ('burst_ts_bits', 6),
          ('sched_cycles', 2), ('sched_late_start', 2), ('sched_max_cycles_late', 2),
          ('sched_samples', 6), ('sched_max_cycles', 6), ('sched_sum_us', 6), ('sched_overrun', 6)]
WORDS = sum(n for _, n in FIELDS)
CPU_HZ = 160e6
FDCAN_HZ = 80e6
PATTERNS = ('zero', 'xorshift')
FRAMES = ('standard', 'extended', 'extended')


def symbol(elf, name):
    for line in subprocess.check_output(['arm-none-eabi-nm', str(elf)], text=True).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise SystemExit(f'{name} not found in {elf}')


def openocd(a, commands, log):
    cmd = [a.openocd, '-s', a.scripts, '-f', a.cfg]
    for item in commands:
        cmd += ['-c', item]
    with Path(log).open('w') as output:
        subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=60)


def read(a, base):
    raw = a.output / 'bench.bin'
    openocd(a, ['init', 'halt', f'dump_image {raw} 0x{base:08x} {WORDS * 4}', 'resume', 'shutdown'],
            a.output / 'read.log')
    words = list(struct.unpack(f'<{WORDS}I', raw.read_bytes()))
    state = {}
    for name, n in FIELDS:
        state[name] = words[0] if n == 1 else words[:n]
        del words[:n]
    return state


def summary(s):
    out = dict(kbps=s['kbps'], stage=s['stage'], faults=s['faults'], tx_errors=s['tx_errors'],
               rx_errors=s['rx_errors'], interlock_ok=bool(s['interlock_ok']))
    out['bit_timing'] = []
    for nbtp in s['nbtp']:
        brp, seg1, seg2 = (nbtp >> 16 & 0x1ff) + 1, (nbtp >> 8 & 0xff) + 1, (nbtp & 0x7f) + 1
        out['bit_timing'].append(dict(prescaler=brp, tseg1=seg1, tseg2=seg2, sjw=(nbtp >> 25) + 1,
                                      configured_bps=FDCAN_HZ / (brp * (1 + seg1 + seg2)),
                                      sample_point=(1 + seg1) / (1 + seg1 + seg2)))
    out['interlock'] = [dict(lec=psr & 7, rx=rx) for psr, rx in zip(s['interlock_psr'], s['interlock_rx'])]
    out['burst'] = []
    out['schedule'] = []
    for p, pattern in enumerate(PATTERNS):
        for b in range(3):
            i = p * 3 + b
            frames = s['burst_frames'][i]
            us = s['burst_cycles'][i] / CPU_HZ * 1e6 / (frames - 1) if frames > 1 else None
            bits = s['burst_ts_bits'][i] / (frames - 1) if frames > 1 else None
            out['burst'].append(dict(pattern=pattern, bus=b + 1, frame=FRAMES[b], frames=frames,
                                     us_per_frame=us, bits_per_frame=bits,
                                     measured_bps=bits / us * 1e6 if us and bits else None))
            samples = s['sched_samples'][i]
            mean = s['sched_sum_us'][i] / samples / 1000 if samples else None
            out['schedule'].append(dict(pattern=pattern, bus=b + 1, frame=FRAMES[b],
                                        cycles=s['sched_cycles'][p], samples=samples,
                                        overrun=s['sched_overrun'][i], mean_ms=mean,
                                        max_ms=s['sched_max_cycles'][i] / CPU_HZ * 1e3,
                                        mean_load=mean / 5 if mean else None))
        out[f'schedule_{pattern}_late_start_over_100us'] = s['sched_late_start'][p]
        out[f'schedule_{pattern}_max_start_late_us'] = s['sched_max_cycles_late'][p] / CPU_HZ * 1e6
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--elf', type=Path, required=True)
    p.add_argument('--openocd', required=True)
    p.add_argument('--scripts', required=True)
    p.add_argument('--cfg', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait', type=float, default=30)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    openocd(a, [f'program {a.elf.resolve()} verify reset exit'], a.output / 'flash.log')
    assert '** Verified OK **' in (a.output / 'flash.log').read_text(), 'flash verification missing'
    base = symbol(a.elf, 'g_bench')
    time.sleep(a.wait)
    state = read(a, base)
    assert state['magic'] == 0x43414e42 and state['version'] == 1, state
    result = summary(state)
    (a.output / 'raw.json').write_text(json.dumps(state, indent=2) + '\n')
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
