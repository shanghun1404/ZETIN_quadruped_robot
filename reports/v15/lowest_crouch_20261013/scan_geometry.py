"""Offline authored-hull/IK screen; real PhysX validation remains required."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linprog
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'config'))
p=ROOT/'reports/v15/jetson_usb_20261012/verification/analyze_handover_2355.py'
spec=importlib.util.spec_from_file_location('recorded_geometry',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
g=m.RecordedGeometry();nominal=np.asarray(g.metadata['canonical_nominal_q']);h0=g.metadata['nominal_root_height_m']
limits={j.get('name'):np.fromstring(j.get('range'),sep=' ') for j in g.root.iter('joint')}
lo=np.array([limits[n][0] for n in g.names]);hi=np.array([limits[n][1] for n in g.names])
feet0,_,_=g.measure(nominal,[0,0,h0],[1,0,0,0])
foot_paths={f['collider'] for f in g.feet}
adjacent={frozenset(('base',f'{leg}_hip')) for leg in ['FL','FR','BL','BR']}
adjacent|={frozenset((f'{leg}_{a}',f'{leg}_{b}')) for leg in ['FL','FR','BL','BR'] for a,b in [('hip','thigh'),('thigh','shank')]}

def geometry(q,h):
 poses=g.poses(q,[0,0,h],[1,0,0,0]);shapes=[]
 for i,record in enumerate(g.hull_records):
  rot,origin=poses[record['owner']];v=g.hulls[f'vertices_{i}']@rot.T+origin
  planes=g.hulls[f'planes_{i}'];normal=planes[:,:3]@rot.T;offset=planes[:,3]-normal@origin
  shapes.append((record,v.min(0),v.max(0),normal,offset))
 low=min((s[1][2],s[0]['path']) for s in shapes if s[0]['path'] not in foot_paths)
 hits=[]
 for a in range(len(shapes)):
  for b in range(a+1,len(shapes)):
   ra,mina,maxa,na,da=shapes[a];rb,minb,maxb,nb,db=shapes[b]
   if ra['owner']==rb['owner'] or frozenset((ra['owner'],rb['owner'])) in adjacent:continue
   if np.any(np.minimum(maxa,maxb)<=np.maximum(mina,minb)):continue
   normal=np.vstack([na,nb]);fit=linprog([0,0,0,-1],A_ub=np.c_[normal,np.ones(len(normal))],b_ub=-np.r_[da,db],bounds=[(None,None)]*4,method='highs')
   if not fit.success:raise RuntimeError(fit.message)
   if fit.x[3]>1e-6:hits.append(dict(first=ra['path'],second=rb['path'],shared_interior_radius_m=float(fit.x[3])))
 return dict(lowest_nonfoot_z_m=float(low[0]),lowest_nonfoot_collider=low[1],collision_candidates=hits)

if __name__=="__main__":
 q=nominal.copy();rows=[]
 for drop in np.arange(0,.361,.005):
  h=h0-drop
  def error(q):return (g.measure(q,[0,0,h],[1,0,0,0])[0]-feet0).ravel()
  fit=least_squares(error,q,bounds=(lo+1e-6,hi-1e-6),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=100)
  q=fit.x;res=float(abs(error(q)).max());geo=geometry(q,h)
  row=dict(drop_m=float(drop),root_height_m=float(h),q=q.tolist(),ik_max_error_m=res,joint_margin_rad=float(np.minimum(q-lo,hi-q).min()),**geo)
  rows.append(row)
  if round(drop*1000)%20==0 or geo['lowest_nonfoot_z_m']<.01:
   print(json.dumps({k:v for k,v in row.items() if k not in ['q','collision_candidates']})+' hits='+str(len(geo['collision_candidates'])),flush=True)
  if res>1e-5 or geo['collision_candidates'] or geo['lowest_nonfoot_z_m']<-.01:break
 (OUT/'geometry_scan.json').write_text(json.dumps(dict(scope='CPU authored convex hulls, fixed four material foot points, no PhysX cooking/contact offsets',nominal_height_m=h0,rows=rows),indent=2)+'\n')
