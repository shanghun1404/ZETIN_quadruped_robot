"""Bounded body-translation feasibility scan, original stationary feet."""
import json
import numpy as np
from scipy.optimize import least_squares
from scan_geometry import OUT,g,nominal,h0,lo,hi,feet0,geometry
rows=[]
for shift in np.arange(-.1,.101,.02):
 q=nominal.copy();last=None
 for drop in np.arange(.18,.341,.005):
  h=h0-drop
  def error(q):return (g.measure(q,[0,shift,h],[1,0,0,0])[0]-feet0).ravel()
  fit=least_squares(error,q,bounds=(lo+1e-6,hi-1e-6),xtol=1e-9,ftol=1e-9,gtol=1e-9,max_nfev=70)
  q=fit.x;res=float(abs(error(q)).max());geo=geometry(q,h)
  row=dict(shift_m=float(shift),drop_m=float(drop),height_m=float(h),q=q.tolist(),error_m=res,**geo)
  if res>1e-5 or geo['collision_candidates'] or geo['lowest_nonfoot_z_m']<.003:
   row['first_invalid']=True;rows.extend([last,row] if last else [row]);print(json.dumps({k:v for k,v in last.items() if k not in ['q','collision_candidates']}) if last else 'no reachable depth',flush=True);break
  last=row
 else:rows.append(last);print('reached scan bottom',shift,drop,flush=True)
(OUT/'shift_scan.json').write_text(json.dumps(rows,indent=2)+'\n')
