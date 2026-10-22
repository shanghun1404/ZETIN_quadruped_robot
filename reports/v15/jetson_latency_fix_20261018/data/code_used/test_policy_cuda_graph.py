"""Graph replay must preserve varying inputs, recurrent state and safety checks."""
import numpy as np
import pytest
import torch

from v14_ros2_control.policy import V14DepthPolicy
from test_policy import rsl_actor_checkpoint, student_checkpoint


@pytest.fixture
def checkpoints(tmp_path):
    actor=tmp_path/'actor.pt';student=tmp_path/'student.pt'
    rsl_actor_checkpoint(actor);student_checkpoint(student)
    return actor,student


def sample(rng):
    depth=rng.uniform(-.1,6.,size=(84,84)).astype(np.float32)
    depth[0,:3]=[np.nan,np.inf,-np.inf]
    return dict(joint_pos=rng.normal(size=12),joint_vel=rng.normal(size=12),
        lin_vel_b=rng.normal(size=3),ang_vel_b=rng.normal(size=3),
        quat_wxyz=[1.,0.,0.,0.],command=[0.,.4,0.],depth_m=depth)


def test_cpu_path_never_constructs_a_cuda_graph(checkpoints,monkeypatch):
    def forbidden():raise AssertionError('CPU inference touched CUDA capture')
    monkeypatch.setattr(torch.cuda,'CUDAGraph',forbidden)
    policy=V14DepthPolicy(*checkpoints,device='cpu',cuda_graph=True)
    assert not policy.cuda_graph_enabled
    assert torch.isfinite(policy.step(**sample(np.random.default_rng(1)))[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires a real CUDA device')
def test_graph_preserves_varying_input_sequence_reset_and_previous_outputs(checkpoints):
    eager=V14DepthPolicy(*checkpoints,device='cuda',cuda_graph=False)
    graph=V14DepthPolicy(*checkpoints,device='cuda',cuda_graph=True)
    assert graph.cuda_graph_enabled and not eager.cuda_graph_enabled
    rng=np.random.default_rng(23);saved=[]
    for step in range(24):
        if step==12:eager.reset();graph.reset()
        inputs=sample(rng);expected=eager.step(**inputs);actual=graph.step(**inputs)
        for left,right in zip(expected,actual):torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
        saved.append((actual,tuple(value.clone() for value in actual)))
    for outputs,copies in saved:
        for value,copy in zip(outputs,copies):torch.testing.assert_close(value,copy,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires a real CUDA device')
def test_graph_replay_retains_nonfinite_action_rejection(checkpoints):
    graph=V14DepthPolicy(*checkpoints,device='cuda',cuda_graph=True)
    with torch.no_grad():next(graph.actor.parameters()).fill_(float('nan'))
    with pytest.raises(RuntimeError,match='non-finite policy action'):
        graph.step(**sample(np.random.default_rng(3)))
