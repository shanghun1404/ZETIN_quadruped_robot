import gc,json,time
from pathlib import Path
import numpy as np
import torch
from v14_ros2_control.policy import V14DepthPolicy
from benchmark_with_stm import inputs

torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path('/home/scienceowl/robot_dog/cpp_20261017/v15_jetson_20261012')
policies={name:V14DepthPolicy(root/'checkpoints/model_15699.pt',root/'checkpoints/v12_student_best.pt',device='cuda',cuda_graph=enabled) for name,enabled in [('eager',False),('graph',True)]}
cases=[]
for i in range(32):
 sample=inputs(root);sample['joint_pos']+=.015*np.sin(np.arange(12)+i);sample['joint_vel']=.02*np.cos(np.arange(12)+i);sample['depth_m']+=.05*np.sin(i);cases.append(sample)
errors=[]
for sample in cases:
 a,b=[p.step(**sample)[0] for p in policies.values()];errors.append(float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
for p in policies.values():
 p.reset()
 for i in range(100):p.step(**cases[i%32])
gc.collect();gc.freeze();result=dict(max_action_error=max(errors),samples_compared=len(errors),device='cuda:0',gc_enabled=gc.isenabled())
for name,p in policies.items():
 times=[];cpus=[]
 for i in range(1000):
  before=time.perf_counter_ns();cpu=time.thread_time_ns();p.step(**cases[i%32]);cpus.append((time.thread_time_ns()-cpu)/1e6);times.append((time.perf_counter_ns()-before)/1e6)
 result[name]=dict(mean_ms=float(np.mean(times)),p99_ms=float(np.percentile(times,99)),max_ms=max(times),mean_thread_cpu_ms=float(np.mean(cpus)),cuda_graph_enabled=p.cuda_graph_enabled)
Path('cuda_graph_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
