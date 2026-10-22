"""Render measured outcomes without altering original verdicts."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
shutil.copytree(a.run, a.output / 'data', dirs_exist_ok=True)
analysis = a.output / 'analysis_scripts'
analysis.mkdir(exist_ok=True)
for name in ('audit_fixed_hour.py', 'plot_fixed_hour.py', 'report_fixed_hour.py'):
    shutil.copy2(Path(__file__).resolve().parent / name, analysis / name)
load = lambda name: json.loads((a.run / name).read_text())
audit = load('audited_result.json')
request = load('stress_start.json')
controller = load('ros_controller_benchmark.json')
raw = load('stress_result.json')
start = datetime.fromisoformat(request['start_utc']).astimezone(timezone(timedelta(hours=9)))
window = f"{start + timedelta(seconds=request['warmup_seconds']):%Y-%m-%d %H:%M:%S} – {start + timedelta(seconds=request['seconds']):%Y-%m-%d %H:%M:%S} KST"
duration = '완료' if audit['execution_duration_completed'] else '기준 미충족'
verdict = '통과' if audit['integration_passed'] else '미통과'
for name in ('audited_result.json', 'stress_start.json', 'stress_result.json', 'ros_controller_benchmark.json',
             'wbc_result.json', 'diagnostic_cleanup_result.json', 'mcu-final.log', 'os_samples.jsonl',
             'raw_sha256.json', 'shutdown_verification.json'):
    if (a.run / name).exists():
        shutil.copy2(a.run / name, a.output / name)
def msrow(label, values):
    if not values:
        return f'<tr><th>{html.escape(label)}</th><td colspan="5">표본 없음</td></tr>'
    return f"<tr><th>{html.escape(label)}</th><td>{values['samples']:,}</td><td>{values['mean_ms']:.3f}</td><td>{values['p99_ms']:.3f}</td><td>{values['max_ms']:.3f}</td><td>{values['above_20ms']}</td></tr>"
rows = ''.join(msrow(label, values) for label, values in
    [('CUDA AI ROS 계획', audit['ai']), ('C++ WBC', audit['wbc']['independently_recomputed']),
     ('명령 계산', audit['commands']['command_compute_ms'])])
failed = ', '.join(audit['failed_original_integration_gates']) or '없음'
faults = len(audit['faults_during_stream'])
commands = audit['commands']
mcu = audit['mcu_diagnostics'] or {}
checks = ''.join(f'<tr><th>{html.escape(key)}</th><td>{"통과" if value else "실패"}</td></tr>'
                 for key, value in audit['checks'].items())
gc_count = sum(r['during_stream_events_ge_1ms'] for r in audit['gc'].values())
gc_max = max((r['max_during_stream_ms'] for r in audit['gc'].values()), default=0)
os_result = audit['os']
temps = ', '.join(f'{html.escape(key)} {value:.2f}°C' for key, value in os_result['max_temperatures_c'].items())
pg = controller['physical_gateway']
fault_text = html.escape(json.dumps(audit['faults_during_stream'], ensure_ascii=False, indent=2))
document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>수정 후 AI·WBC·H533 1시간 스트레스 시험</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1080px;margin:36px auto;padding:0 22px;color:#182634}}h1{{font-size:27px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #ccd4dc;text-align:left}}td{{font-variant-numeric:tabular-nums}}.result{{background:#f1f4f8;padding:16px}}pre{{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}}img{{width:100%}}a{{color:#175d96}}</style>
<h1>수정 후 AI·WBC·H533 1시간 스트레스 시험</h1>
<p>{window}. 5초 예열 후 {audit['measured_duration_s']:g}초 측정. Orin Nano Super, 25W, 일반 스케줄러·기본 CPU 배치.</p>
<p class="result"><strong>시간 운용: {duration} · 엄격한 통합 판정: {verdict}</strong><br>
측정 명령 {commands['measured_complete']:,}/{commands['measured_expected']:,}개, 계획 {audit['ai']['samples']:,}/{audit['ai']['expected_samples']:,}개,
WBC {audit['wbc']['independently_counted_measured_rows']:,}개. 운용 중 fault 이벤트 {faults}건.
실패한 기존 판정 항목: {html.escape(failed)}.</p>
<p>입력 센서·관절·접촉 값과 MCU 피드백은 시험용 가상 값이다. 실제 USB 게이트웨이와 NUCLEO-H533을 사용했고 모터 출력은 계속 비활성이다. WBC는 별도 프로세스에서 연속 계산한다. 실로봇 폐루프 제어·G474·CAN·실모터 검증 결과가 아니다.</p>
<h2>측정 결과</h2><table><tr><th>계산</th><th>표본</th><th>평균 ms</th><th>p99 ms</th><th>최대 ms</th><th>20ms 초과</th></tr>{rows}</table>
<p>명령 간격 최대 {commands['max_command_interval_ms']}ms, 사용 계획 최대 source age {commands['max_plan_age_ms']}ms.
계획 완료 간 최대 공백 {audit['ai']['max_completion_gap_ms']:.3f}ms. 기존 50ms 신선도 보호 조건과 피드백 표본 수 허용치 ±2개를 유지했다.</p>
<img src="timing_and_resources.png" alt="1시간의 초별 최대 계획·WBC 계산 시간, 온도와 메모리 여유">
<h2>STM 수신과 자원</h2><p>예열 포함 명령 {mcu.get('commands_received')}개 수신·{mcu.get('commands_accepted')}개 수락.
UART 오류 {mcu.get('uart_errors')}, 수신 오버플로 {mcu.get('rx_overflows')}, 송신 오류 {mcu.get('tx_errors')}, missed tick {mcu.get('missed_ticks')}, motor apply {mcu.get('apply_calls')}.
측정 구간 피드백 {pg['raw_feedback_samples']:,}/{pg['expected_raw_feedback_samples']:,}개, {pg['raw_feedback_hz']:.6f}Hz, source tick 누락 {pg['missing_feedback_ticks']}개.</p>
<p>준비·측정·저장 구간을 포함한 최소 메모리 여유 {os_result['min_memory_available_mb']:.1f}MiB, 관측된 작업 프로세스 RSS 최대 {os_result['peak_process_rss_mb']:.1f}MiB.
기록된 최고 온도: {temps}. 측정 도구가 기록한 운용 중 1ms 이상 GC pause {gc_count}건, 최대 {gc_max:.3f}ms.</p>
<h2>판정과 원본 보존</h2><p>원본 WBC의 passed는 시간 구간 계산 완료 판정이며, 모든 계산이 20ms 이내라는 뜻은 아니다. 주기 초과 횟수는 위 표와 별도로 그대로 표시했다.
보호 정지가 발생하면 명령은 정지 상태를 유지하고 AI·WBC 부하만 계속하도록 실행했다. 이후 계산을 성공적인 STM 명령 전송으로 집계하지 않았다. 모든 원본 판정과 로그를 보존했다.</p>
<pre>{fault_text}</pre><table>{checks}</table>
<h2>근거</h2><ul>
<li><a href="audited_result.json">독립 감사</a> · <a href="stress_result.json">원본 전체 판정</a> · <a href="ros_controller_benchmark.json">원본 제어 판정</a> · <a href="wbc_result.json">WBC 원본</a></li>
<li><a href="diagnostic_cleanup_result.json">저장·종료 검사</a> · <a href="shutdown_verification.json">프로세스 종료 확인</a> · <a href="mcu-final.log">STM 카운터</a></li>
<li><a href="raw_sha256.json">원시 파일 SHA256</a> · <a href="SHA256SUMS">보고서 SHA256</a> · <a href="../jetson_latency_fix_20261018/report.html">수정 직후 5분 결과</a> · <a href="../jetson_h533_stress_20261017/report.html">이전 1시간 실패 결과</a></li>
<li><a href="data/ros_samples.csv">전체 명령 CSV</a> · <a href="data/planner_samples.json">전체 계획 표본</a> · <a href="data/wbc_samples.csv">전체 WBC CSV</a> · <a href="data/gateway_writes.csv">USB 쓰기 CSV</a> · <a href="data/code_used/benchmark_with_stm.py">실행 당시 코드</a></li>
</ul></html>'''
(a.output / 'report.html').write_text(document)
hashes = []
for path in sorted(a.output.rglob('*')):
    if path.is_file() and path.name != 'SHA256SUMS':
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(a.output).as_posix())
(a.output / 'SHA256SUMS').write_text('\n'.join(hashes) + '\n')
print(a.output / 'report.html')
