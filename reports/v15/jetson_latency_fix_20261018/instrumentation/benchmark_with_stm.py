#!/usr/bin/env python3
"""Synthetic-sensor CUDA/C++/DDS control with a real, disabled H533 USB gateway.

H533 feedback is explicitly synthetic zero state. This tests the output path
through the production gateway, not physical sensors, actuators or closed-loop
robot behavior. Requires the diagnostic firmware already verified separately.
"""
import argparse
import array
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"samples": 0}
    return dict(samples=len(a), mean_ms=float(a.mean()),
                p50_ms=float(np.percentile(a, 50)),
                p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()),
                above_5ms=int((a > 5).sum()), above_20ms=int((a > 20).sum()))


def inputs(root):
    from v14_ros2_control import contract
    from v151_ros2_control.motion import CANONICAL
    metadata = json.loads((root / "model/cad_model.json").read_text())
    q = np.asarray(metadata["canonical_nominal_q"])
    return dict(joint_pos=q[[CANONICAL.index(n) for n in contract.JOINT_NAMES]],
                joint_vel=np.zeros(12), lin_vel_b=np.zeros(3), ang_vel_b=np.zeros(3),
                quat_wxyz=[1., 0., 0., 0.], command=[0., .4, 0.],
                depth_m=np.linspace(.2, 4., 84*84, dtype=np.float32).reshape(84, 84))


def gpu_info():
    import torch
    assert torch.cuda.is_available(), "CUDA is not available in this Python environment"
    return dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                compiled_architectures=torch.cuda.get_arch_list(),
                cudnn=torch.backends.cudnn.version(),
                cpu_threads=torch.get_num_threads(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def micro(args):
    import torch
    from v14_ros2_control.policy import V14DepthPolicy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu_info(),
              "method": "Full policy.step: CPU inputs, depth normalization, student, actor, CPU outputs; explicit CUDA synchronization"}
    policies = {d: V14DepthPolicy(args.root / "checkpoints/model_15699.pt",
                                 args.root / "checkpoints/v12_student_best.pt", device=d)
                for d in ("cpu", "cuda")}
    cases = []
    for i in range(16):
        sample = inputs(args.root)
        sample["depth_m"] = sample["depth_m"] + .05 * np.sin(i)
        sample["joint_pos"] = sample["joint_pos"] + .015 * np.sin(np.arange(12) + i)
        sample["joint_vel"] = .02 * np.cos(np.arange(12) + i)
        cases.append(sample)
    errors = []
    for sample in cases:
        outputs = [policies[d].step(**sample)[0] for d in ("cpu", "cuda")]
        errors.append(float((outputs[0] - outputs[1]).abs().max()))
        assert torch.allclose(*outputs, atol=2e-4, rtol=2e-4), errors[-1]
    result["cpu_cuda_action_max_absolute_error"] = max(errors)
    result["numerical_comparison_samples"] = len(cases)
    for device, policy in policies.items():
        # Restore the constructor's ordinary CPU tensor after diagnostic inference.
        # reset() alone cannot mutate an inference tensor outside inference_mode.
        policy.last_action = torch.zeros_like(policy.last_action)
        policy.reset()
        for i in range(100):
            policy.step(**cases[i % len(cases)])
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = []
        for i in range(1000):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            policy.step(**cases[i % len(cases)])
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        result[device] = stats(elapsed)
        print(device, json.dumps(result[device]), flush=True)
    (args.output / "policy_microbenchmark.json").write_text(json.dumps(result, indent=2) + "\n")


def child(args):
    import torch
    import rclpy
    import v151_ros2_control.controller_node as controller
    from v15_hardware_bridge.cad_controller import HardwareController
    if args.mode=='child' and args.startup_delay:time.sleep(args.startup_delay)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from gc_timing import prepare_heap
    if args.mode=='child':
        os.environ['ROBOT_STRESS_OUTPUT']=str(args.output.resolve())
        os.environ['ROBOT_STRESS_SECONDS']=str(args.seconds)
        import v151_ros2_control.policy_process as policy_module
        from stress_policy_worker import serve
        policy_module._serve=serve
    rclpy.init(args=["--ros-args", "-r", "__ns:=" + args.namespace,
                    "-p", "planner_only:=true", "-p", "device:=" + args.device,
                    "-p", "wbc_backend:=cpp",
                    "-p", "actor_checkpoint:=" + str(args.root / "checkpoints/model_15699.pt"),
                    "-p", "student_checkpoint:=" + str(args.root / "checkpoints/v12_student_best.pt"),
                    "-p", "model_path:=" + str(args.root / "model/quadruped_cad.xml"),
                    "-p", "initial_mode:=" + ("WALK" if args.scenario == "walk" else "STAND"),
                    "--log-level", "warn"]+(['-r','joint_states:=policy_joint_states'] if args.mode=='child' else []))
    if args.mode == 'command_child':
        from v15_hardware_bridge.command_node import JointCommandNode
        node=JointCommandNode()
        original_fail=node.fail
        def traced_fail(reason):
            last=node.stage.last_plan
            print('FAULT_TRACE',json.dumps(dict(reason=str(reason),stamp=node.stage.latest_joint_stamp,
                last_plan=last[0]['source_stamp_ns'] if last else None,
                receipt_age_ms=(time.monotonic_ns()-last[1])/1e6 if last else None,
                pending=sorted(node.stage.plans))),flush=True)
            original_fail(reason)
        node.fail=traced_fail
        try:
            executor=rclpy.get_global_executor()
            executor.add_node(node);executor.spin_once(timeout_sec=0)
            prepare_heap(args.output/'gc_prepare_command.json')
            (args.output/'command_ready.json').write_text(json.dumps({'pid':os.getpid()}))
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
        return
    node = HardwareController()
    # Only this diagnostic worker dumps buffered samples on close. Give that
    # post-run dump time to finish; production policy shutdown remains 5s.
    production_join=node.policy.process.join
    diagnostic_join_used=False
    def join_with_diagnostic_dump(timeout=None):
        nonlocal diagnostic_join_used
        if timeout==5. and not diagnostic_join_used:
            diagnostic_join_used=True
            timeout=60.
        return production_join(timeout=timeout)
    node.policy.process.join=join_with_diagnostic_dump
    if args.input_timing:
        from input_timing_probe import instrument
        instrument(node,args.output,args.seconds)
    # Finish lazy CUDA/PTX/kernel initialization before starting timed sensors.
    # Use the actual policy worker, then restore its ordinary initial action.
    warmup_started=time.perf_counter_ns()
    for _ in range(100):node.policy.step(**inputs(args.root))
    node.policy.reset()
    executor=rclpy.get_global_executor()
    executor.add_node(node);executor.spin_once(timeout_sec=0)
    prepare_heap(args.output/'gc_prepare_planner.json')
    trace_data=np.empty((round(args.seconds*50)+1000,3),dtype=np.int64)
    trace_data.fill(0)
    trace_count=0
    from gc_timing import check_gc
    original_drain=node.drain
    def traced_drain():
        nonlocal trace_count
        start=time.perf_counter_ns();previous=node.last
        original_drain()
        check_gc()
        if node.last!=previous:
            if trace_count>=len(trace_data):raise OverflowError('planner trace capacity exceeded')
            trace_data[trace_count]=(node.last,start,time.perf_counter_ns());trace_count+=1
    node.drain=traced_drain
    try:
        info = gpu_info() if args.device == "cuda" else {"torch": torch.__version__}
        info.update(node.policy.info)
        info['pre_stream_policy_warmup_steps']=100
        info['pre_stream_policy_warmup_ms']=(time.perf_counter_ns()-warmup_started)/1e6
        (args.output / "child_ready.json").write_text(json.dumps(info, indent=2))
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        primary_error=sys.exc_info()[1]
        cleanup_errors=[]
        try:
            with (args.output/'planner_trace.jsonl').open('w') as stream:
                for stamp,begin,end in trace_data[:trace_count]:
                    stream.write(json.dumps(dict(stamp_ns=int(stamp),start_ns=int(begin),end_ns=int(end)))+'\n')
        except BaseException as error:
            cleanup_errors.append(f'planner trace: {type(error).__name__}: {error}')
        try:
            node.destroy_node()
        except BaseException as error:
            cleanup_errors.append(f'planner destroy: {type(error).__name__}: {error}')
        finally:
            if rclpy.ok():rclpy.shutdown()
        worker=node.policy.process
        worker_state=dict(worker_pid=worker.pid,worker_exitcode=worker.exitcode,
            worker_alive=worker.is_alive(),planner_trace_samples=trace_count,errors=cleanup_errors)
        (args.output/'planner_cleanup_result.json').write_text(json.dumps(worker_state,indent=2)+'\n')
        if primary_error is None and (cleanup_errors or worker_state['worker_alive'] or worker.exitcode!=0):
            raise RuntimeError(f'planner diagnostic cleanup failed: {worker_state}')


def ros(args):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState, Image
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from v14_ros2_control import contract
    metadata = json.loads((args.root / "model/cad_model.json").read_text())
    ns = "/cuda_timing_" + str(os.getpid())
    ready = args.output / "child_ready.json"
    command_ready=args.output/'command_ready.json'
    wbc_ready=args.output/'wbc_ready.json'
    gateway_ready=args.output/'gc_prepare_gateway.json'
    stress_start=args.output/'stress_start.json'
    gateway_ready.unlink(missing_ok=True)
    wbc_process=None
    wbc_log=None
    if args.continuous_wbc:
        wbc_ready.unlink(missing_ok=True);stress_start.unlink(missing_ok=True)
        wbc_log=(args.output/'wbc_stress.log').open('w')
        wbc_process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('continuous_wbc.py')),
            '--root',str(args.root),'--output',str(args.output),'--seconds',str(args.seconds)],stdout=wbc_log,stderr=subprocess.STDOUT,start_new_session=True)
    command_ready.unlink(missing_ok=True)
    ready.unlink(missing_ok=True)
    log = (args.output / "controller.log").open("w")
    gateway_log = (args.output / 'gateway.log').open('w')
    gateway_process = subprocess.Popen([
        sys.executable, str(Path(__file__).with_name('gateway_timing_probe.py')),
        str(args.output/'gateway_writes.csv'),'--seconds',str(args.seconds),
        '--ros-args', '-r', '__ns:=' + ns,
        '-p', 'hardware_config:=' + str(args.physical_gateway_config),
        '-p', 'enable_motors:=false'], stdout=gateway_log, stderr=subprocess.STDOUT,start_new_session=True)
    process = subprocess.Popen([sys.executable, __file__, "child", "--device", args.device,
                                "--root", str(args.root), "--output", str(args.output),
                                "--namespace", ns, "--scenario", args.scenario,"--seconds",str(args.seconds),
                                "--startup-delay",str(5. if args.live_startup else 0.)]+(["--input-timing"] if args.input_timing else []), stdout=log, stderr=subprocess.STDOUT,start_new_session=True)
    command_process=subprocess.Popen([sys.executable,__file__,'command_child','--root',str(args.root),
        '--output',str(args.output),'--namespace',ns],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    node = executor = None
    rows = {}
    planner_rows={}
    if args.numeric_records:
        from timing_records import NumericRecords,NumericLog
        rows=NumericRecords(round(args.seconds*200),5_000_000,dict(step=int,stamp_ns=int,
            scheduled_ns=int,sent_ns=int,policy_tick=bool,command_received_ns=int,compute_ms=float,
            mode=str,walk_weight=float,planner_compute_ms=float,plan_age_ms=float,status_received_ns=int,
            policy_publish_ns=int))
        planner_rows=NumericRecords(round(args.seconds*50),20_000_000,
            dict(stamp_ns=int,received_ns=int,compute_ms=float,mode=str))
    faults=[];streaming=False;faults_during_stream=0
    hardware_statuses=[];physical_feedback=[];received_count=status_count=0
    latest_mcu_ack=0;hardware_acknowledged=False;policy_plan_count=0
    from logging_buffers import LatestSnapshotWriter
    from gc_timing import check_gc
    progress_writer=LatestSnapshotWriter(args.output/'progress.json',announce=lambda value:
        print(f"{value['elapsed_s']:.0f}s: sent={value['sensor_samples']} received={value['command_publications']}",flush=True))
    if args.numeric_records:
        faults=NumericLog(round((args.seconds+180)*20),dict(reason=str,during_stream=bool,source=str))
        physical_feedback=NumericLog(round((args.seconds+180)*200),dict(stamp_ns=int,received_ns=int))
        hardware_statuses=NumericLog(round((args.seconds+180)*10),dict(session=int,sequence=int,
            mcu_stamp_ns=int,received_monotonic_ns=int,ack_sequence=int,armed=bool,fault_flags=int,
            enabled=bool,observed_ns=int,fault=bool,reason=str))
    try:
        deadline = time.monotonic() + 120
        while (not gateway_ready.exists() or not command_ready.exists() or (not args.live_startup and not ready.exists()) or
               (args.continuous_wbc and not wbc_ready.exists())):
            if gateway_process.poll() is not None:
                raise RuntimeError('Physical gateway startup failed; see gateway.log')
            if wbc_process is not None and wbc_process.poll() is not None:
                raise RuntimeError('Continuous WBC startup failed')
            if process.poll() is not None or command_process.poll() is not None:
                raise RuntimeError("Controller startup failed; see controller.log")
            if time.monotonic() > deadline:
                raise TimeoutError("Controller startup exceeded 120 seconds")
            time.sleep(.1)
        rclpy.init()
        node = rclpy.create_node("synthetic_sensor_timing", namespace=ns)
        qos = contract.lockstep_qos()
        pubs = [node.create_publisher(typ, topic, qos) for typ, topic in (
            (JointState, "joint_states"), (Odometry, "odom"), (Image, "depth/image"),
            (JointState, "foot_contacts"))]
        command_pub = node.create_publisher(Twist, "cmd_vel", 10)
        policy_state_pub=node.create_publisher(JointState,'policy_joint_states',qos)

        def receive_command(message):
            nonlocal received_count
            stamp = contract.stamp_ns(message.header.stamp)
            if stamp in rows:
                if 'command_received_ns' not in rows[stamp]:received_count+=1
                rows[stamp]["command_received_ns"] = time.perf_counter_ns()

        def receive_status(message):
            nonlocal status_count
            value = json.loads(message.data)
            stamp = value["stamp_ns"]
            if stamp in rows:
                if 'status_received_ns' not in rows[stamp]:status_count+=1
                rows[stamp].update(compute_ms=value["compute_ms"], mode=value["mode"],
                                   walk_weight=value["walk_weight"],
                                   planner_compute_ms=value.get('planner_compute_ms'),
                                   plan_age_ms=value.get('plan_age_ms'),
                                   status_received_ns=time.perf_counter_ns())

        def receive_plan(message):
            nonlocal policy_plan_count
            value=json.loads(message.data)
            if value['source_stamp_ns'] not in planner_rows and value['status']['walk_weight']>0:
                policy_plan_count+=1
            planner_rows[value['source_stamp_ns']]=dict(stamp_ns=value['source_stamp_ns'],received_ns=time.perf_counter_ns(),
                compute_ms=value['status']['planner_compute_ms'],mode=value['status']['mode'])

        node.create_subscription(JointState, "joint_command", receive_command, qos)
        node.create_subscription(String, "status", receive_status, qos)
        node.create_subscription(String,'control_plan',receive_plan,qos)
        def record_fault(value):
            nonlocal faults_during_stream
            faults.append(value)
            faults_during_stream+=int(value['during_stream'])
        node.create_subscription(String,'hardware_fault',lambda m:record_fault(dict(reason=m.data,during_stream=streaming)),10)
        def hardware_status(message):
            nonlocal latest_mcu_ack,hardware_acknowledged
            value=json.loads(message.data)
            hardware_statuses.append(dict(value,observed_ns=time.perf_counter_ns()))
            if 'ack_sequence' in value:
                hardware_acknowledged=True
                latest_mcu_ack=max(latest_mcu_ack,value['ack_sequence'])
            if value.get('fault'):
                record_fault(dict(reason=value.get('reason'),during_stream=streaming,source='usb_gateway'))
        node.create_subscription(String,'hardware_status',hardware_status,10)
        node.create_subscription(JointState,'raw/joint_states',lambda m:physical_feedback.append(
            dict(stamp_ns=contract.stamp_ns(m.header.stamp),received_ns=time.perf_counter_ns())),10)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        deadline = time.monotonic() + 15
        while not ((pubs[0].get_subscription_count() if args.live_startup else
                   all(p.get_subscription_count() for p in pubs) and command_pub.get_subscription_count()) and
                   node.count_publishers(ns + "/joint_command") and
                   node.count_publishers(ns + "/status")):
            executor.spin_once(timeout_sec=.01)
            if time.monotonic() > deadline:
                raise TimeoutError("DDS endpoints did not match")
        deadline=time.monotonic()+5
        while not hardware_acknowledged:
            executor.spin_once(timeout_sec=.005)
            if gateway_process.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Physical gateway handshake failed; see gateway.log')
        if args.cpu_layout:
            required=set(range(6))
            if not required.issubset(os.sched_getaffinity(0)):raise RuntimeError('CPU layout requires cores0..5')
            child_info=json.loads(ready.read_text())
            assignments=[(os.getpid(),0),(gateway_process.pid,1),(command_process.pid,2),(process.pid,3),(child_info['worker_pid'],4),(wbc_process.pid,5)]
            for pid,cpu in assignments:
                subprocess.run(['taskset','--all-tasks','--pid','--cpu-list',str(cpu),str(pid)],check=True,stdout=subprocess.DEVNULL)
            (args.output/'cpu_layout.json').write_text(json.dumps(assignments))
            if args.critical_fifo:
                for pid,cpu in assignments:
                    if cpu not in (2,3,4,5):continue
                    subprocess.run(['sudo','-n','chrt','--fifo','--pid','10',str(pid)],check=True)
        gateway_scheduler='SCHED_OTHER'
        if args.gateway_fifo:
            subprocess.run(['sudo','-n','chrt','--all-tasks','--fifo','--pid','10',str(gateway_process.pid)],check=True)
            if os.sched_getscheduler(gateway_process.pid)!=os.SCHED_FIFO:
                raise RuntimeError('Gateway realtime scheduling was not applied')
            gateway_scheduler='SCHED_FIFO priority10 (test process only)'
        sample = inputs(args.root)
        state = JointState(name=list(contract.JOINT_NAMES), position=sample["joint_pos"].tolist(),
                           velocity=[0.] * 12)
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.
        odom.pose.pose.position.z = metadata["nominal_root_height_m"]
        image = Image(height=84, width=84, encoding="32FC1", is_bigendian=0, step=336,
                      data=array.array("B", sample["depth_m"].astype("<f4").tobytes()))
        contacts = JointState(name=[n + "_foot" for n in ("FL", "FR", "RL", "RR")],
                              effort=[50.] * 4)
        period_ns = contract.PHYSICS_DT_NS
        count = int(args.seconds * 1e9 / period_ns)
        from gc_timing import prepare_heap
        prepare_heap(args.output/'gc_prepare_publisher.json')
        start = time.perf_counter_ns() + 50_000_000
        stress_temp=args.output/'stress_start.tmp'
        stress_temp.write_text(json.dumps(dict(start_perf_ns=start,
            start_utc=datetime.now(timezone.utc).isoformat(),seconds=args.seconds,warmup_seconds=5)))
        stress_temp.replace(stress_start)
        streaming=True
        for step in range(count):
            scheduled = start + step * period_ns
            while True:
                wait = (scheduled - time.perf_counter_ns()) / 1e9
                if wait <= 0:
                    break
                executor.spin_once(timeout_sec=min(wait, .0005))
            if (process.poll() is not None or command_process.poll() is not None or
                gateway_process.poll() is not None or
                (wbc_process is not None and wbc_process.poll() is not None)):
                raise RuntimeError("Controller exited during measurement; see controller.log")
            check_gc();progress_writer.check()
            if faults_during_stream and not args.continue_after_fault:
                raise RuntimeError('Control or gateway fault during streaming; stopping test')
            if step % 20 == 0:
                request = Twist()
                t=step*.005
                walking=(3<=t<8 or 13<=t<17) if args.scenario=='quick' else (10<=t<32 or 38<=t<44)
                request.linear.y = .4 if args.scenario == "walk" or walking else 0.
                command_pub.publish(request)
            stamp = step * period_ns
            state.header.stamp.sec, state.header.stamp.nanosec = contract.step_stamp(step)
            rows[stamp] = dict(step=step, stamp_ns=stamp, scheduled_ns=scheduled,
                               sent_ns=time.perf_counter_ns(), policy_tick=step % 4 == 0)
            if step % 4 == 0:
                for message in (odom, image, contacts):
                    message.header.stamp = state.header.stamp
                for pub, message in zip(pubs[1:], (odom, image, contacts)):
                    pub.publish(message)
                rows[stamp]["policy_publish_ns"]=time.perf_counter_ns()
                policy_state_pub.publish(state)
            pubs[0].publish(state)
            # Keep draining even if the publisher is briefly late.
            for _ in range(2):
                executor.spin_once(timeout_sec=0)
            if step and step % 2000 == 0:
                progress=dict(elapsed_s=step*.005,total_s=args.seconds,
                    sensor_samples=step+1,command_publications=received_count,
                    latest_mcu_ack=latest_mcu_ack,
                    faults_during_stream=faults_during_stream,
                    planner_alive=process.poll() is None,command_alive=command_process.poll() is None,
                    gateway_alive=gateway_process.poll() is None,
                    wbc_alive=wbc_process is not None and wbc_process.poll() is None,
                    mode=rows.get(stamp,{}).get('mode'),updated_utc=datetime.now(timezone.utc).isoformat())
                progress_writer.submit(progress)
        streaming=False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if received_count==len(rows) and status_count==len(rows):
                break
            executor.spin_once(timeout_sec=.001)
        if args.numeric_records:
            faults=[dict(row) for row in faults]
            rows=rows.materialize();planner_rows=planner_rows.materialize()
            physical_feedback=[dict(row) for row in physical_feedback]
            hardware_statuses=[dict(row) for row in hardware_statuses]
        warmup_seconds=10 if args.live_startup else 5
        measured = [r for r in rows.values() if r["step"] >= warmup_seconds*200]
        complete = [r for r in measured if "command_received_ns" in r and "compute_ms" in r]
        for row in complete:
            row["command_latency_ms"] = (row["command_received_ns"] - row["sent_ns"]) / 1e6
            row["scheduled_to_command_ms"] = (row["command_received_ns"] - row["scheduled_ns"]) / 1e6
            row["publisher_lateness_ms"] = (row["sent_ns"] - row["scheduled_ns"]) / 1e6
        def report(subset):
            return {key: stats([r[key] for r in subset]) for key in
                    ("compute_ms", "command_latency_ms", "scheduled_to_command_ms", "publisher_lateness_ms")}
        arrivals = [r["command_received_ns"] for r in complete]
        result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), device=args.device,
                      scenario=args.scenario,
                      child=json.loads(ready.read_text()),
                      method="Actual C++/CUDA planner and independent command process; production device/backend parameters; no constructor override",
                      sensors="Synthetic depth, nominal joints, odometry and contacts over DDS; actual production USB gateway and H533, motors disabled; MCU feedback synthetic zeros",
                      power_mode=subprocess.check_output(["sudo", "-n", "nvpmodel", "-q"], text=True).strip(),
                      requested_seconds=args.seconds, warmup_seconds=warmup_seconds,
                      continue_after_fault=args.continue_after_fault,
                      live_startup=args.live_startup,faults_during_stream=[f for f in faults if f['during_stream']],
                      first_command_source_ms=min((s/1e6 for s,r in rows.items() if 'command_received_ns' in r),default=None),
                      published_total=count, measured_expected=len(measured),
                      measured_complete=len(complete), missing=len(measured)-len(complete),
                      measured_command_hz=(len(arrivals)-1)*1e9/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else 0,
                      all=report(complete),
                      policy_ticks=report([r for r in complete if r["policy_tick"]]),
                      non_policy_ticks=report([r for r in complete if not r["policy_tick"]]),
                      by_mode={m: report([r for r in complete if r["mode"] == m])
                               for m in sorted({r["mode"] for r in complete})},
                      command_interval_ms=stats(np.diff(arrivals)/1e6))
        measured_plans=[r for s,r in planner_rows.items() if s>=warmup_seconds*1_000_000_000]
        result['planner_compute_ms']=stats([r['compute_ms'] for r in measured_plans])
        expected_plans=round((args.seconds-warmup_seconds)*50)
        result['planner_expected_samples']=expected_plans
        result['planner_samples_complete']=len(measured_plans)==expected_plans
        result['planner_measured_samples']=len(measured_plans)
        result['planner_first_measured_source_s']=min((r['stamp_ns']/1e9 for r in measured_plans),default=None)
        result['planner_last_measured_source_s']=max((r['stamp_ns']/1e9 for r in measured_plans),default=None)
        result['planner_by_mode']={m:stats([r['compute_ms'] for r in measured_plans if r['mode']==m]) for m in sorted({r['mode'] for r in measured_plans})}
        result['plan_age_ms']=stats([r['plan_age_ms'] for r in complete])
        measured_feedback=[row for row in physical_feedback
            if start+warmup_seconds*1_000_000_000 <= row['received_ns'] < start+int(args.seconds*1e9)]
        feedback_arrivals=[row['received_ns'] for row in measured_feedback]
        result['physical_gateway']=dict(
            scheduler=gateway_scheduler,
            configuration=str(args.physical_gateway_config),motors_enabled=False,
            controller_consumes_synthetic_nominal_joints=True,
            mcu_feedback_is_synthetic_zero=True,
            raw_feedback_samples=len(measured_feedback),
            expected_raw_feedback_samples=round((args.seconds-warmup_seconds)*200),
            last_feedback_before_end_ms=(start+int(args.seconds*1e9)-feedback_arrivals[-1])/1e6 if feedback_arrivals else None,
            raw_feedback_hz=(len(feedback_arrivals)-1)*1e9/(feedback_arrivals[-1]-feedback_arrivals[0])
                if len(feedback_arrivals)>1 else 0,
            missing_feedback_ticks=sum(max(0,(b['stamp_ns']-a['stamp_ns'])//5_000_000-1)
                for a,b in zip(measured_feedback,measured_feedback[1:])),
            status_messages=len(hardware_statuses),
            any_armed=any(value.get('armed',False) for value in hardware_statuses),
            any_mcu_fault=any(value.get('fault_flags',0) for value in hardware_statuses),
            gateway_faults_during_stream=[f for f in faults if f['during_stream'] and f.get('source')=='usb_gateway'],
            observed_command_publications=sum('command_received_ns' in row for row in rows.values()),
            last_observed_cumulative_ack=max((value.get('ack_sequence',0) for value in hardware_statuses),default=0))
        result['continuous_wbc_enabled']=args.continuous_wbc
        result['elapsed_stream_s']=(time.perf_counter_ns()-start)/1e9
        result['gateway_ack_fraction']=(result['physical_gateway']['last_observed_cumulative_ack']/received_count
                                       if received_count else 0)
        result['integration_checks_passed']=(result['missing']==0 and
            result['planner_samples_complete'] and
            not result['faults_during_stream'] and
            not result['physical_gateway']['any_armed'] and
            not result['physical_gateway']['any_mcu_fault'] and
            result['physical_gateway']['missing_feedback_ticks']==0 and
            abs(result['physical_gateway']['raw_feedback_hz']-200)<1 and
            abs(len(measured_feedback)-result['physical_gateway']['expected_raw_feedback_samples'])<=2 and
            result['physical_gateway']['last_feedback_before_end_ms']<20 and
            result['gateway_ack_fraction']>=.99)
        (args.output/'hardware_statuses.json').write_text(json.dumps(hardware_statuses,indent=2)+'\n')
        (args.output/'physical_feedback.json').write_text(json.dumps(physical_feedback)+'\n')
        (args.output/'planner_samples.json').write_text(json.dumps(measured_plans,indent=2)+'\n')
        result["by_mode_policy_ticks"] = {m: report([r for r in complete if r["mode"] == m and r["policy_tick"]])
                                           for m in result["by_mode"]}
        fields = sorted({k for row in rows.values() for k in row})
        with (args.output / "ros_samples.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows.values())
    except BaseException as error:
        snapshot=dict(error=f'{type(error).__name__}: {error}',
            observed_commands=received_count,
            elapsed_s=(time.perf_counter_ns()-start)/1e9 if 'start' in locals() else None,
            faults=[dict(row) for row in faults],last_hardware_statuses=hardware_statuses[-20:])
        (args.output/'failure_recent_samples.json').write_text(json.dumps([dict(row) for row in rows.values()][-400:],indent=2)+'\n')
        (args.output/'failure.json').write_text(json.dumps(snapshot,indent=2)+'\n')
        raise
    finally:
        from logging_buffers import finalize_diagnostic_cleanup
        from gc_timing import close_all_gc
        primary_error=sys.exc_info()[1]
        worker_expected=100+policy_plan_count
        artifact_specs={
            'ros_samples.csv':dict(kind='csv',count=len(rows),fields=['step','stamp_ns','scheduled_ns','sent_ns']),
            'planner_trace.jsonl':dict(kind='jsonl',count=len(planner_rows),fields=['stamp_ns','start_ns','end_ns']),
            'policy_worker_timing.csv':dict(kind='csv',count=worker_expected,fields=['step','start_ns','end_ns','thread_cpu_ms','runqueue_wait_ms']),
            'policy_worker_result.json':dict(kind='json',values=dict(passed=True,requested_steps=worker_expected,completed_results=worker_expected)),
            'planner_cleanup_result.json':dict(kind='json',values=dict(worker_exitcode=0,worker_alive=False,planner_trace_samples=len(planner_rows),errors=[])),
            'gateway_trace_metrics.json':dict(kind='json',values=dict(buffer_overflows=0,logging_overflows=0)),
            'gateway_writes.csv':dict(kind='csv',count_from=('gateway_trace_metrics.json','write_calls'),
                fields=['started_ns','ended_ns','kind','sequence','error'],
                count_where=dict(field='kind',equals='3',count_from=('gateway_trace_metrics.json','command_write_calls'))),
            'serial_syscalls_slow.jsonl':dict(kind='jsonl',count_from=('gateway_trace_metrics.json','slow_or_failed_writes'),fields=['sequence','start_ns','end_ns','events']),
            'gc_gateway.jsonl':dict(kind='jsonl'),
            'gc_policy.jsonl':dict(kind='jsonl'),
            f'gc_child_{process.pid}.jsonl':dict(kind='jsonl'),
            f'gc_command_child_{command_process.pid}.jsonl':dict(kind='jsonl'),
            f'gc_ros_{os.getpid()}.jsonl':dict(kind='jsonl')}
        if args.continuous_wbc:
            requested_wbc=round(args.seconds*50)
            artifact_specs.update({
                'wbc_samples.csv':dict(kind='csv',count=requested_wbc,fields=['step','scheduled_ns','started_ns','compute_ms']),
                'wbc_result.json':dict(kind='json',values=dict(passed=True,requested_samples=requested_wbc,completed_samples=requested_wbc)),
                'gc_wbc.jsonl':dict(kind='jsonl')})
        if args.input_timing:
            artifact_specs['planner_input_timing.csv']=dict(kind='csv',fields=['phase','stamp_ns','start_ns','end_ns','thread_cpu_ms'])
        resources=[('progress writer',progress_writer.close),('gateway log',gateway_log.close),('controller log',log.close)]
        if wbc_log is not None:resources.append(('WBC log',wbc_log.close))
        if node is not None:resources.append(('ROS node',node.destroy_node))
        if executor is not None:resources.append(('ROS executor',executor.shutdown))
        resources.append(('ROS shutdown',lambda:rclpy.shutdown() if rclpy.ok() else None))
        resources.append(('GC dump',close_all_gc))
        finalize_diagnostic_cleanup(args.output,
            dict(wbc=wbc_process,gateway=gateway_process,planner=process,command=command_process),
            artifact_specs,primary_error=primary_error,resources=resources,benchmark_result=locals().get('result'))
    if 'result' in locals():
        print((args.output/'ros_controller_benchmark.json').read_text(),flush=True)



if __name__ == "__main__":
    # Noninteractive shell background jobs can inherit SIGINT ignored.
    signal.signal(signal.SIGINT,signal.default_int_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("micro", "ros", "child", "command_child"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--seconds", type=float, default=50.)
    parser.add_argument("--namespace", default="/cuda_timing")
    parser.add_argument("--scenario", choices=("all", "walk", "quick"), default="all")
    parser.add_argument('--live-startup',action='store_true')
    parser.add_argument('--startup-delay',type=float,default=0.)
    parser.add_argument('--physical-gateway-config',type=Path)
    parser.add_argument('--continuous-wbc',action='store_true')
    parser.add_argument('--gateway-fifo',action='store_true')
    parser.add_argument('--cpu-layout',action='store_true')
    parser.add_argument('--numeric-records',action='store_true',default=True,
        help='Use bounded numeric sample storage (always enabled; flag kept for compatibility).')
    parser.add_argument('--input-timing',action='store_true',
        help='Enable optional planner input boundary instrumentation into a bounded RAM buffer.')
    parser.add_argument('--critical-fifo',action='store_true')
    parser.add_argument('--continue-after-fault',action='store_true',
        help='Keep synthetic AI/WBC load running after latched motor-off faults; integration still fails.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from gc_timing import watch_gc
    close_gc=watch_gc(args.output/f'gc_{args.mode}_{os.getpid()}.jsonl')
    try:
        {"micro": micro, "ros": ros, "child": child, "command_child":child}[args.mode](args)
    finally:
        close_gc()
