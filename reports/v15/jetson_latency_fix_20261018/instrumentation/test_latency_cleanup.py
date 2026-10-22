import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import logging_buffers

class Process:
    def __init__(self,code=0,timeout=False):
        self.returncode=code;self.pid=None;self.timeout=timeout;self.killed=False;self.signaled=False
    def poll(self):return None if self.timeout or self.signaled is False else self.returncode
    def send_signal(self,sig):self.signaled=True
    def wait(self,timeout=None):
        if self.timeout and not self.killed:raise subprocess.TimeoutExpired('dump',timeout)
        return self.returncode
    def kill(self):self.killed=True;self.returncode=-9

class CleanupTests(unittest.TestCase):
    def finish(self,*args,**kwargs):
        self.assertTrue(hasattr(logging_buffers,'finalize_diagnostic_cleanup'),
            'deferred child dump failures have no parent cleanup audit')
        return logging_buffers.finalize_diagnostic_cleanup(*args,**kwargs)

    def test_deferred_dump_failure_revokes_success_and_fails_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            report=folder/'ros_controller_benchmark.json'
            report.write_text(json.dumps({'integration_checks_passed':True}))
            with self.assertRaisesRegex(RuntimeError,'diagnostic cleanup failed'):
                self.finish(folder,{'planner':Process(code=1)}, {}, primary_error=None)
            self.assertFalse(json.loads(report.read_text())['integration_checks_passed'])
            audit=json.loads((folder/'diagnostic_cleanup_result.json').read_text())
            self.assertFalse(audit['passed']);self.assertEqual(audit['children']['planner']['exitcode'],1)

    def test_primary_error_is_preserved_while_cleanup_still_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            original=ValueError('first failure')
            audit=self.finish(Path(folder),{'planner':Process(code=1)}, {},primary_error=original)
            self.assertFalse(audit['passed']);self.assertIn('first failure',audit['primary_error'])

    def test_missing_or_truncated_raw_log_revokes_success(self):
        for content in (None,'step,start_ns\n1,123'):
            with self.subTest(content=content),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder)
                if content is not None:(folder/'samples.csv').write_text(content)
                with self.assertRaisesRegex(RuntimeError,'diagnostic cleanup failed'):
                    self.finish(folder,{}, {'samples.csv':dict(kind='csv',count=1,fields=['step','start_ns'])})
                self.assertFalse(json.loads((folder/'diagnostic_cleanup_result.json').read_text())['passed'])

    def test_timeout_is_failure_and_all_children_are_reaped(self):
        with tempfile.TemporaryDirectory() as folder:
            children={name:subprocess.Popen([sys.executable,'-c',
                'import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);print("ready",flush=True);time.sleep(30)'],
                start_new_session=True,stdout=subprocess.PIPE,text=True) for name in ('planner','gateway')}
            try:
                for process in children.values():self.assertEqual(process.stdout.readline().strip(),'ready')
                with self.assertRaisesRegex(RuntimeError,'diagnostic cleanup failed'):
                    self.finish(Path(folder),children,{},timeout=.05)
                self.assertTrue(all(p.poll() is not None for p in children.values()))
                for p in children.values():
                    with self.assertRaises(ProcessLookupError):os.kill(p.pid,0)
            finally:
                for p in children.values():
                    if p.poll() is None:p.kill();p.wait()
                    p.stdout.close()

    def test_owned_group_is_cleaned_after_leader_exits_before_shutdown(self):
        self.check_descendant_cleanup(leader_exits_before_cleanup=True)

    def test_owned_group_is_cleaned_when_leader_disappears_at_timeout(self):
        self.check_descendant_cleanup(leader_exits_before_cleanup=False)

    def check_descendant_cleanup(self,leader_exits_before_cleanup):
        import time
        child_script='import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);time.sleep(30)'
        leader_script=(f'import subprocess,sys,time,signal;'
            f'child=subprocess.Popen([sys.executable,"-c",{child_script!r}]);'
            f'print(child.pid,flush=True);'
            + ('sys.exit(1)' if leader_exits_before_cleanup else
               'signal.signal(signal.SIGINT,signal.SIG_IGN);time.sleep(30)'))
        leader=subprocess.Popen([sys.executable,'-c',leader_script],
            start_new_session=True,stdout=subprocess.PIPE,text=True)
        child_pid=int(leader.stdout.readline().strip())
        process=leader
        if leader_exits_before_cleanup:
            leader.wait(timeout=5)
        else:
            # Reproduce the race: the leader is reaped just as wait reports its
            # deadline, while its independently running worker still owns PGID.
            class LeaderDisappearsAtDeadline:
                pid=leader.pid
                raced=False
                def poll(self):return leader.poll()
                def send_signal(self,sig):pass
                def wait(self,timeout=None):
                    if not self.raced:
                        self.raced=True;leader.kill();leader.wait(timeout=5)
                        raise subprocess.TimeoutExpired('leader dump',timeout)
                    return leader.wait(timeout=timeout)
                def kill(self):leader.kill()
            process=LeaderDisappearsAtDeadline()
        try:
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(RuntimeError,'diagnostic cleanup failed'):
                    self.finish(Path(folder),{'planner':process},{},timeout=.05)
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                try:state=Path(f'/proc/{child_pid}/stat').read_text().split(') ',1)[1].split()[0]
                except FileNotFoundError:break
                if state=='Z':break
                time.sleep(.01)
            else:self.fail('worker remained alive after its leader disappeared')
            self.assertIsNotNone(leader.poll())
        finally:
            try:os.killpg(leader.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            leader.wait(timeout=5);leader.stdout.close()

    def test_normal_spawn_teardown_waits_for_tracker_and_ignores_zombies(self):
        script="""
import multiprocessing
import signal
import time
worker=multiprocessing.get_context('spawn').Process(target=time.sleep,args=(30,),daemon=True)
worker.start()
print('ready',flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    worker.terminate()
    worker.join()
"""
        for attempt in range(10):
            with self.subTest(attempt=attempt),tempfile.TemporaryDirectory() as folder:
                leader=subprocess.Popen([sys.executable,'-c',script],start_new_session=True,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    self.assertEqual(leader.stdout.readline().strip(),'ready')
                    audit=self.finish(Path(folder),{'planner':leader},{},timeout=2)
                    self.assertTrue(audit['passed']);self.assertEqual(leader.returncode,0)
                finally:
                    try:os.killpg(leader.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    leader.wait(timeout=5);leader.stdout.close();leader.stderr.close()

    def test_success_report_is_written_only_after_cleanup_and_artifact_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);report=folder/'ros_controller_benchmark.json';observed=[]
            (folder/'samples.csv').write_text('step,start_ns\n1,9007199254740993\n')
            audit=self.finish(folder,{}, {'samples.csv':dict(kind='csv',count=1,fields=['step','start_ns'])},
                resources=[('cleanup',lambda:observed.append(report.exists()))],
                benchmark_result={'integration_checks_passed':True})
            self.assertEqual(observed,[False]);self.assertTrue(audit['passed'])
            self.assertEqual(json.loads(report.read_text()),
                {'integration_checks_passed':True,'diagnostic_cleanup_passed':True})

    def test_complete_but_short_raw_csv_is_rejected_against_request_count(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);(folder/'samples.csv').write_text('step,start_ns\n1,123\n')
            with self.assertRaisesRegex(RuntimeError,'1 records; expected 2'):
                self.finish(folder,{}, {'samples.csv':dict(kind='csv',count=2)})

    def test_cleanup_write_failure_cannot_replace_primary_exception(self):
        from unittest.mock import patch
        original=ValueError('primary inference failure')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as raised:
                try:raise original
                finally:
                    with patch.object(Path,'write_text',side_effect=OSError('disk full')):
                        self.finish(Path(folder),{}, {},primary_error=original)
            self.assertIs(raised.exception,original)

    def test_gateway_keyboard_interrupt_code_is_accepted_after_signal(self):
        observed=subprocess.run([sys.executable,'-c','raise KeyboardInterrupt'],capture_output=True).returncode
        self.assertEqual(observed,-signal.SIGINT)
        with tempfile.TemporaryDirectory() as folder:
            audit=self.finish(Path(folder),{'gateway':Process(code=observed)}, {})
            self.assertTrue(audit['passed'])

if __name__=='__main__':unittest.main()
