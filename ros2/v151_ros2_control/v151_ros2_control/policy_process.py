"""One ordered policy process; a direct pipe submits work before WBC begins."""
import multiprocessing
import os
import torch
from v14_ros2_control.policy import V14DepthPolicy


def _serve(connection,actor,student,device):
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        policy=V14DepthPolicy(actor,student,device=device)
        connection.send(('ready',dict(worker_pid=os.getpid(),
            actor_parameter_device=str(next(policy.actor.parameters()).device),
            student_parameter_device=str(next(policy.student.parameters()).device),
            cuda_graph_enabled=policy.cuda_graph_enabled)))
        while True:
            operation,inputs=connection.recv()
            if operation=='close':break
            try:
                if operation=='reset':
                    policy.reset();connection.send(('reset',None))
                elif operation=='step':
                    action,target=policy.step(**inputs)
                    connection.send(('result',(action.numpy().copy(),target.numpy().copy())))
                else:raise ValueError('unknown policy operation')
            except Exception as error:
                connection.send(('error',(type(error).__name__,str(error))))
    except (EOFError,BrokenPipeError):pass
    except Exception as error:
        connection.send(('error',(type(error).__name__,str(error))))
    finally:connection.close()


class _Result:
    def __init__(self,owner):self.owner=owner;self.joined=False

    def result(self):
        if self.joined:raise RuntimeError('policy result already joined')
        self.joined=True
        try:action,target=self.owner._receive('result')
        finally:self.owner.pending=False
        self.owner.last_action=torch.from_numpy(action)
        return self.owner.last_action,torch.from_numpy(target)


class ProcessPolicy:
    def __init__(self,actor,student,*,device='cpu'):
        context=multiprocessing.get_context('spawn')
        self.connection,child=context.Pipe()
        self.process=context.Process(target=_serve,args=(child,actor,student,device),daemon=True)
        self.pending=False;self.closed=False
        self.process.start();child.close()
        try:self.info=self._receive('ready')
        except BaseException:
            self.close();raise
        self.worker_pid=self.info['worker_pid'];self.last_action=torch.zeros(12)

    def _receive(self,expected):
        try:kind,data=self.connection.recv()
        except (EOFError,OSError) as error:raise RuntimeError('policy worker disconnected') from error
        if kind=='error':
            name,reason=data
            exception=ValueError if name=='ValueError' else RuntimeError
            raise exception(f'policy worker {name}: {reason}')
        if kind!=expected:raise RuntimeError(f'unexpected policy response: {kind}')
        return data

    def submit(self,**inputs):
        if self.pending:raise RuntimeError('previous policy result must be joined')
        self.connection.send(('step',inputs));self.pending=True
        return _Result(self)

    def step(self,**inputs):return self.submit(**inputs).result()

    def reset(self):
        if self.pending:raise RuntimeError('policy reset requires a joined result')
        self.connection.send(('reset',None));self._receive('reset')
        self.last_action=torch.zeros(12)

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.process.is_alive():self.connection.send(('close',None))
        except (OSError,EOFError):pass
        self.process.join(timeout=5.)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5.)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=5.)
        self.connection.close()
