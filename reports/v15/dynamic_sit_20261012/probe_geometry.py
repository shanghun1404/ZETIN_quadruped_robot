import json,sys
from pathlib import Path
import numpy as np
from v151_ros2_control.cad_standing import CadStandingController
p=Path(sys.argv[1]); ctl=CadStandingController(p)
q0=ctl.nominal_joint_position; h0=ctl.nominal_height
feet=ctl.contact_positions(q0,[0,0,h0],[1,0,0,0])
rows=[]
for drop,pitch,shift in [(.13,.35,-.045),(.14,.4,-.04),(.14,.45,-.04),(.15,.45,-.03)]:
 q=q0.copy(); h=h0-drop; quat=[np.cos(pitch/2),np.sin(pitch/2),0,0]; pos=[0,shift,h]
 base=ctl._base(pos,quat)
 for it in range(100):
  err=(feet-ctl.contact_positions(q,pos,quat)).ravel()
  if np.max(abs(err))<1e-8:break
  jac=ctl._jacobians(q,base)[:,:,6:].reshape(12,12)
  q+=np.clip(np.linalg.solve(jac.T@jac+1e-8*np.eye(12),jac.T@err),-.1,.1)
 rows.append(dict(drop=drop,pitch=pitch,shift=shift,q=q.tolist(),error=float(np.max(abs(err))),hip_height_front_rear=(h+np.array([.315,-.315])*np.sin(pitch)+.075*np.cos(pitch)).tolist()))
print(json.dumps(rows,indent=2))
