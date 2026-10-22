import csv
import importlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SCRATCH=Path('/home/scienceowl/Downloads/jetson_setup/h533_link_20261017/latency_fix')
import sys
sys.path.insert(0,str(SCRATCH))

class LoggingTests(unittest.TestCase):
    def test_worker_defers_csv_until_close_and_keeps_integer_times(self):
        from stress_policy_worker import ConnectionProbe
        class Pipe:
            def recv(self):return ('step',{})
            def send(self,value):pass
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,ROBOT_STRESS_OUTPUT=folder,ROBOT_STRESS_SECONDS='1'):
            probe=ConnectionProbe(Pipe())
            probe.recv();probe.send(('result',(None,None)))
            destination=Path(folder)/'policy_worker_timing.csv'
            self.assertFalse(destination.exists(),'timed worker wrote CSV before close')
            probe.close()
            with destination.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertGreater(int(rows[0]['end_ns']),int(rows[0]['start_ns']))
            self.assertIn('runqueue_wait_ms',rows[0])

    def test_buffer_keeps_exact_ns_and_fails_closed_on_overflow(self):
        from logging_buffers import NumericBuffer
        with tempfile.TemporaryDirectory() as folder:
            b=NumericBuffer(1,[('stamp_ns','i8'),('elapsed_ms','f8')])
            b.append((9007199254740993,1.25))
            self.assertFalse(b.data.dtype.hasobject)
            with self.assertRaises(OverflowError):b.append((2,2.))
            destination=Path(folder)/'samples.csv';b.write_csv(destination)
            with destination.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows,[{'stamp_ns':'9007199254740993','elapsed_ms':'1.25'}])

    def test_buffer_rejects_truncated_diagnostic_strings(self):
        from logging_buffers import NumericBuffer
        b=NumericBuffer(1,[('error','U4')])
        with self.assertRaises(OverflowError):b.append(('12345',))
        self.assertEqual(b.count,0)

    def test_progress_writer_does_file_io_on_its_own_thread(self):
        from logging_buffers import LatestSnapshotWriter
        with tempfile.TemporaryDirectory() as folder:
            destination=Path(folder)/'progress.json'
            main_thread=threading.get_ident();writers=[]
            original=Path.write_text
            def observed(path,*args,**kwargs):
                writers.append(threading.get_ident())
                return original(path,*args,**kwargs)
            with patch.object(Path,'write_text',observed):
                writer=LatestSnapshotWriter(destination)
                writer.submit({'count':1});writer.submit({'count':2});writer.close()
            import json
            self.assertEqual(json.loads(destination.read_text()),{'count':2})
            self.assertTrue(writers)
            self.assertNotIn(main_thread,writers)

    def test_progress_writer_surfaces_file_errors(self):
        from logging_buffers import LatestSnapshotWriter
        with tempfile.TemporaryDirectory() as folder:
            writer=LatestSnapshotWriter(Path(folder)/'missing'/'progress.json')
            writer.submit({'count':1})
            with self.assertRaises(RuntimeError):writer.close()

if __name__=='__main__':unittest.main()
