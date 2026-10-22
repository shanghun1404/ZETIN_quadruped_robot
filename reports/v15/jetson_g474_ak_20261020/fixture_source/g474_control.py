"""SWD control only between UART tests; never halt the target during a run."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parent
OPENOCD = ROOT.parent / 'h533_link_20261017/tools/openocd-st/src/openocd'
SCRIPTS = ROOT.parent / 'h533_link_20261017/tools/openocd-st/tcl'
SCALARS = '''magic version clock_hz boot_stage requested_mode current_mode mode_changes
rx_bytes rx_overflows uart_errors tx_errors frames_received commands_received commands_accepted
commands_routed route_rejected feedback_sent missed_ticks apply_calls ready_checks session
capture_requested capture_done capture_samples_per_round capture_rounds can_errors gpio_moder
gpio_afrl gpio_afrh pwr_cr3 capture_sequence requested_profile current_profile virtual_replies
virtual_decoded virtual_errors parser_rejected max_route_cycles max_parser_cycles max_rx_queued max_loop_cycles'''.split()
BUS = '''queued completed received rx_mismatch timeouts tx_cancelled last_psr last_ecr txbrp txbto txbcf cccr nbtp test_reg'''.split()
JOINT = '''queued completed received last_id expected_lo expected_hi received_lo received_hi'''.split()
DIAG_BYTES = (len(SCALARS) + 3*len(BUS) + 12*len(JOINT))*4

def addresses():
    raw = subprocess.check_output(['arm-none-eabi-nm', '-n', str(ROOT/'build/g474_can_fixture.elf')], text=True)
    found = {parts[2]: int(parts[0], 16) for line in raw.splitlines() if len(parts := line.split()) == 3}
    assert found['g_diag'] == 0x20000000, 'Unexpected diagnostic layout'
    return found['g_diag'], found['g_capture']

def oc(commands, log):
    args = [str(OPENOCD), '-s', str(SCRIPTS), '-f', str(ROOT/'g474_openocd.cfg')]
    for item in ['init', 'halt', *commands, 'resume', 'shutdown']:
        args += ['-c', item]
    with Path(log).open('w') as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True, cwd=ROOT, timeout=40)

def configure(profile, capture=False, mode=1):
    base, _ = addresses()
    assert profile in ('mit', 'servo') and mode in (0, 1, 2)
    values = {'requested_mode': mode, 'requested_profile': int(profile == 'servo'),
              'capture_done': 0, 'capture_requested': int(capture)}
    oc([f'mww 0x{base+SCALARS.index(k)*4:08x} {v}' for k,v in values.items()], ROOT/f'configure-{profile}.log')

def snapshot(output, capture=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base, trace = addresses()
    commands = [f'dump_image {output}/diag.bin 0x{base:08x} {DIAG_BYTES}']
    if capture:
        commands += [f'dump_image {output}/capture.bin 0x{trace:08x} {8*800*8}']
    oc(commands, output/'swd.log')
    blob = (output/'diag.bin').read_bytes()
    assert len(blob) == DIAG_BYTES == 716
    data = list(struct.unpack('<'+'I'*(len(blob)//4), blob))
    def take(keys):
        row = dict(zip(keys, data[:len(keys)]));del data[:len(keys)];return row
    state = take(SCALARS)
    state['bus'] = [take(BUS) for _ in range(3)]
    state['joint'] = [take(JOINT) for _ in range(12)]
    assert not data and state['magic'] == 0x4743414e and state['version'] == 2
    state['max_route_us'] = state['max_route_cycles']/state['clock_hz']*1e6
    (output/'diag.json').write_text(json.dumps(state, indent=2)+'\n')
    return state

def verify_counts(state, commands):
    failures = []
    for key in ('rx_overflows','uart_errors','tx_errors','route_rejected','can_errors',
                'virtual_errors','parser_rejected','apply_calls'):
        if state[key] != 0: failures.append(f'{key}={state[key]}')
    for key in ('commands_received','commands_accepted','commands_routed'):
        if state[key] != commands: failures.append(f'{key}={state[key]} expected {commands}')
    for key in ('virtual_replies','virtual_decoded'):
        if state[key] != commands*12: failures.append(f'{key}={state[key]} expected {commands*12}')
    for b,row in enumerate(state['bus']):
        for key in ('queued','completed','received'):
            if row[key] != commands*8: failures.append(f'bus{b+1}.{key}={row[key]} expected {commands*8}')
        for key in ('rx_mismatch','timeouts','tx_cancelled'):
            if row[key]: failures.append(f'bus{b+1}.{key}={row[key]}')
    for j,row in enumerate(state['joint']):
        for key in ('queued','completed','received'):
            if row[key] != commands*2: failures.append(f'joint{j}.{key}={row[key]} expected {commands*2}')
    return failures

if __name__ == '__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('configure');c.add_argument('profile',choices=['mit','servo']);c.add_argument('--capture',action='store_true');c.add_argument('--can-mode',type=int,default=1)
    s=sub.add_parser('snapshot');s.add_argument('output');s.add_argument('--capture',action='store_true');s.add_argument('--commands',type=int)
    a=p.parse_args()
    if a.action=='configure': configure(a.profile,a.capture,a.can_mode)
    else:
        state=snapshot(a.output,a.capture)
        errors=verify_counts(state,a.commands) if a.commands is not None else []
        print(json.dumps(dict(commands=state['commands_routed'],virtual_decoded=state['virtual_decoded'],
                              can_errors=state['can_errors'],profile=state['current_profile'],max_route_us=state['max_route_us'],failures=errors)))
        if errors: raise SystemExit(1)
