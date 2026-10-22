import argparse
import json
import re
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('returncode',type=int)
args=p.parse_args();out=args.output
def read(name):
    path=out/name
    return json.loads(path.read_text()) if path.exists() else None
controller=read('ros_controller_benchmark.json');wbc=read('wbc_result.json')
fields=['magic','version','clock_hz','boot_stage','rx_bytes','rx_overflows','uart_errors','tx_errors',
    'frames_received','commands_received','commands_accepted','feedback_sent','missed_ticks','max_rx_depth',
    'session','apply_calls','ready_checks','last_uart_error','parse_calls','parse_cycles_mod32','max_parse_cycles',
    'tx_cycles_mod32','max_tx_cycles','consumed_bytes','max_tick_lateness_ms','icache_enabled']
log=(out/'mcu-final.log').read_text() if (out/'mcu-final.log').exists() else ''
match=re.search(r'H533_DIAG\n([^\n]+)',log)
values=[int(x,0) for x in match[1].split()] if match else []
diag=dict(zip(fields,values)) if len(values)==len(fields) else None
samples=[json.loads(line) for line in (out/'os_samples.jsonl').read_text().splitlines()]
checks=dict(process_exited_successfully=args.returncode==0,
    controller_integration_passed=bool(controller and controller['integration_checks_passed']),
    measured_hour_commands_complete=bool(controller and controller['measured_complete']==720000),
    continuous_wbc_completed=bool(wbc and wbc['passed'] and wbc['measured_samples']==180000),
    mcu_counters_available=diag is not None)
if diag is not None and controller is not None:
    expected=controller['physical_gateway']['observed_command_publications']
    checks.update(mcu_received_every_observed_command=diag['commands_received']==expected,
                  mcu_accepted_every_observed_command=diag['commands_accepted']==expected,
                  no_mcu_io_errors=all(diag[k]==0 for k in ('rx_overflows','uart_errors','tx_errors')),
                  motor_apply_never_called=diag['apply_calls']==0,
                  no_mcu_missed_ticks=diag['missed_ticks']==0)
plans=read('planner_samples.json') or []
source_stamps=sorted(row['stamp_ns'] for row in plans)
arrival_stamps=sorted(row['received_ns'] for row in plans)
ai_coverage=dict(samples=len(plans),walk_samples=sum(row['mode']=='WALK' for row in plans),expected_samples=180000,
    max_source_gap_ms=max(((b-a)/1e6 for a,b in zip(source_stamps,source_stamps[1:])),default=None),
    max_arrival_gap_ms=max(((b-a)/1e6 for a,b in zip(arrival_stamps,arrival_stamps[1:])),default=None),
    criterion='At least 99% of nominal AI completions, covers both window edges within 40ms, no completion gap above100ms; stricter every-cycle checks remain in integration result.')
ai_coverage['passed']=bool(len(plans)>=178200 and ai_coverage['walk_samples']>=178200 and ai_coverage['max_source_gap_ms'] is not None
    and ai_coverage['max_source_gap_ms']<=100 and ai_coverage['max_arrival_gap_ms']<=100)
duration_complete=bool(controller and controller['requested_seconds']-controller['warmup_seconds']==3600
    and controller['elapsed_stream_s']>=3605 and controller['planner_first_measured_source_s'] is not None
    and controller['planner_first_measured_source_s']<=5.04 and controller['planner_last_measured_source_s']>=3604.96
    and ai_coverage['passed']
    and wbc and wbc['passed'] and wbc['measured_samples']==180000)
report=dict(passed=all(checks.values()),hour_compute_duration_completed=duration_complete,
    ai_coverage=ai_coverage,checks=checks,returncode=args.returncode,
    gc_configuration='Warmup then collect+freeze of prepared heaps; automatic collection of new objects remains enabled.',
    post_fault_scope='After a latched command fault, the production command/gateway paths stop sending commands. Only synthetic sensors, the CUDA planner and independent continuous C++ WBC keep running. This is not a successful hour of STM command transmission.',
    controller=controller,wbc=wbc,mcu_diagnostics=diag,failure=read('failure.json'),
    os=dict(samples=len(samples),
        min_memory_available_mb=min((s['memory_available_mb'] for s in samples),default=None),
        peak_process_rss_mb=max((s['process_rss_mb'] for s in samples),default=None),
        max_temperatures_c={name:max(s['temperatures_c'].get(name,0) for s in samples)
                           for name in {n for s in samples for n in s['temperatures_c']}}),
    scope='Synthetic sensors; production CUDA AI and command pipeline, separate continuous production C++ CAD WBC, real H533 UART, permanently disabled motor outputs.')
fault_trace=re.search(r'FAULT_TRACE (\{[^\n]+\})',(out/'controller.log').read_text())
report['first_command_fault_trace']=json.loads(fault_trace[1]) if fault_trace else None
(out/'stress_result.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'live_status.txt').write_text('1시간 시험 종료\nAI/WBC 계산 지속: '+('완료' if duration_complete else '기준 미충족')+'\n통합 제어 판정: '+('통과' if report['passed'] else '실패')+
    '\n상세 결과: '+str(out/'stress_result.json')+'\n')
print(json.dumps(dict(passed=report['passed'],checks=checks,os=report['os']),indent=2))
raise SystemExit(0 if report['passed'] else 1)
