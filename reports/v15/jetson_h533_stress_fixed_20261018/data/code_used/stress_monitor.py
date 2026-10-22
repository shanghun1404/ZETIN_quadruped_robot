"""Record OS/process memory and temperatures without accessing the serial port."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--pid',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
with (args.output/'os_samples.jsonl').open('w',buffering=1) as log:
    while Path(f'/proc/{args.pid}').exists():
        processes={}
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                fields={line.split(':',1)[0]:line.split(':',1)[1].strip()
                    for line in (proc/'status').read_text().splitlines() if ':' in line}
                processes[int(proc.name)]=dict(ppid=int(fields['PPid']),name=fields['Name'],
                    rss_kb=int(fields.get('VmRSS','0 kB').split()[0]))
            except (OSError,KeyError,ValueError):continue
        descendants={args.pid}
        while True:
            new=descendants|{pid for pid,info in processes.items() if info['ppid'] in descendants}
            if new==descendants:break
            descendants=new
        memory={line.split(':')[0]:int(line.split(':')[1].strip().split()[0])
            for line in Path('/proc/meminfo').read_text().splitlines()}
        temperatures={}
        for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
            try:temperatures[(zone/'type').read_text().strip()]=int((zone/'temp').read_text())/1000
            except (OSError,ValueError):pass
        sample=dict(utc=datetime.now(timezone.utc).isoformat(),
            memory_available_mb=memory['MemAvailable']/1024,
            swap_used_mb=(memory['SwapTotal']-memory['SwapFree'])/1024,
            process_rss_mb=sum(processes[pid]['rss_kb'] for pid in descendants if pid in processes)/1024,
            processes={str(pid):processes[pid] for pid in descendants if pid in processes},
            temperatures_c=temperatures,load_average=os.getloadavg())
        log.write(json.dumps(sample)+'\n')
        temp=args.output/'os_latest.tmp';temp.write_text(json.dumps(sample,indent=2)+'\n')
        temp.replace(args.output/'os_latest.json')
        try:
            progress=json.loads((args.output/'progress.json').read_text())
            wbc=json.loads((args.output/'wbc_progress.json').read_text())
            hottest=max(temperatures.values(),default=0)
            text=(f"AI + C++ WBC 1시간 스트레스 시험\n\n"
                  f"경과: {progress['elapsed_s']/60:.1f} / {progress['total_s']/60:.1f} 분\n"
                  f"목표: AI 50 Hz · WBC 50 Hz · STM 명령 200 Hz\n"
                  f"제어 출력: {progress['command_publications']:,} 개\n"
                  f"MCU 최근 ACK: {progress['latest_mcu_ack']:,}\n"
                  f"WBC 계산: {wbc['completed']:,} 회\n"
                  f"제어/통신 오류: {progress['faults_during_stream']} 건\n"
                  f"최고 센서 온도: {hottest:.1f} °C\n"
                  f"메모리 여유: {sample['memory_available_mb']:.0f} MB\n\n"
                  f"모터 비활성 · 센서/관절 피드백은 시험용 가상 값\n"
                  f"갱신: {sample['utc']}\n")
            (args.output/'live_status.txt').write_text(text)
        except (OSError,KeyError,ValueError):pass
        time.sleep(5)
