import csv
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

SCRATCH=Path('/home/scienceowl/Downloads/jetson_setup/h533_link_20261017/latency_fix')
sys.path.insert(0,str(SCRATCH))

def module(name,**values):
    result=types.ModuleType(name);result.__dict__.update(values);return result

class IntegrationTests(unittest.TestCase):
    def test_gateway_retains_partial_write_and_slow_syscall_without_hot_file_io(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);destination=folder/'gateway_writes.csv';observed=[]
            class Serial:
                out_waiting=0
                def fileno(self):return 99999
            class Gateway:
                def __init__(self):self.serial=Serial();self.sequence=0
                def _write(self,kind,stamp,payload):
                    os.write(self.serial.fileno(),payload);time.sleep(.003);self.sequence+=1
                def destroy_node(self):pass
            def spin(node):
                node._write(3,0,b'abcd')
                observed.append(destination.exists())
                observed.append((folder/'serial_syscalls_slow.jsonl').exists())
            rclpy=module('rclpy',init=lambda **kw:None,spin=spin,ok=lambda:False,
                get_global_executor=lambda:types.SimpleNamespace(add_node=lambda node:None,spin_once=lambda **kw:None))
            serialposix=module('serial.serialposix',select=types.SimpleNamespace(select=lambda *args:([],[],[])))
            stubs={'v15_hardware_bridge':module('v15_hardware_bridge'),
                'v15_hardware_bridge.usb_gateway':module('v15_hardware_bridge.usb_gateway',UsbGateway=Gateway),
                'serial':module('serial',serialposix=serialposix),'serial.serialposix':serialposix,'rclpy':rclpy}
            with patch.dict(sys.modules,stubs),patch.object(sys,'argv',['gateway',str(destination),'--seconds','.1']),patch.object(os,'write',return_value=2):
                runpy.run_path(str(SCRATCH/'gateway_timing_probe.py'),run_name='__main__')
            self.assertEqual(observed,[False,False])
            with destination.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['payload_bytes'],'4')
            details=json.loads((folder/'serial_syscalls_slow.jsonl').read_text())
            self.assertEqual(details['events'][0]['requested'],4)
            self.assertEqual(details['events'][0]['returned'],2)
            metrics=json.loads((folder/'gateway_trace_metrics.json').read_text())
            self.assertEqual(metrics['partial_writes'],1);self.assertEqual(metrics['buffer_overflows'],0)
            gc.unfreeze()

    def test_wbc_saves_complete_exact_timestamps_only_after_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);observed=[]
            class Controller:
                nominal_joint_position=np.zeros(12);nominal_height=.4
                def __init__(self,*args,**kwargs):pass
                def initialize(self,*args):pass
                def contact_consistent_reference(self,*args,**kwargs):return np.zeros(12),np.zeros(3)
                def compute(self,*args,**kwargs):
                    observed.append((folder/'wbc_samples.csv').exists())
                    return np.zeros(12),{'finite':True,'feasible':True}
            (folder/'stress_start.json').write_text(json.dumps(dict(start_perf_ns=time.perf_counter_ns()+100_000_000,seconds=.2,warmup_seconds=0)))
            stubs={'torch':module('torch',set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None),
                'v151_ros2_control':module('v151_ros2_control'),
                'v151_ros2_control.cad_standing':module('v151_ros2_control.cad_standing',CadStandingController=Controller)}
            with patch.dict(sys.modules,stubs),patch.object(sys,'argv',['wbc','--root',str(folder),'--output',str(folder),'--seconds','.2']):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(SCRATCH/'continuous_wbc.py'),run_name='__main__')
            self.assertEqual(result.exception.code,0)
            self.assertTrue(observed);self.assertFalse(any(observed))
            with (folder/'wbc_samples.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),10)
            self.assertEqual(int(rows[1]['scheduled_ns'])-int(rows[0]['scheduled_ns']),20_000_000)
            result=json.loads((folder/'wbc_result.json').read_text());self.assertTrue(result['passed'])
            gc.unfreeze()

    def test_input_probe_defers_csv_and_keeps_phases(self):
        stubs={'v14_ros2_control':module('v14_ros2_control'),
            'v14_ros2_control.contract':module('v14_ros2_control.contract',stamp_ns=lambda stamp:stamp)}
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,stubs):
            probe=runpy.run_path(str(SCRATCH/'input_timing_probe.py'))
            node=types.SimpleNamespace(states=object(),odoms=object(),images=object(),contacts=object(),last=20_000_000,
                add=lambda *args:None,plan_pub=types.SimpleNamespace(publish=lambda *args:None),destroy_node=lambda:None)
            probe['instrument'](node,Path(folder),1)
            message=types.SimpleNamespace(header=types.SimpleNamespace(stamp=20_000_000))
            node.add(node.states,message);node.plan_pub.publish(message)
            destination=Path(folder)/'planner_input_timing.csv';self.assertFalse(destination.exists())
            node.destroy_node()
            with destination.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual([row['phase'] for row in rows],['states','publish'])
            self.assertEqual([int(row['stamp_ns']) for row in rows],[20_000_000,20_000_000])

if __name__=='__main__':unittest.main()
