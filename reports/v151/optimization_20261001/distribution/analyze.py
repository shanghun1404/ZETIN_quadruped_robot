"""CPU-only objective study; production modules are read, never modified."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from v151_ros2_control.standing import StandingController
from v16_controller_math import wbc_acceleration_weights
import v16_wbc_qp

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
ctl=StandingController(ROOT/'reports/v16/physical_model/v16_model_source_quadruped.xml')
data=np.load(ROOT/'reports/v151/integration/601/result.npz')
status=json.load(open(ROOT/'reports/v151/integration/601/result.status.json'))
hard=np.tile([24.8,48.,48.],4); rated=np.tile([8.3,18.,18.],4)
scale=np.r_[[3.]*3,[10.]*3,[50.]*12]
track=wbc_acceleration_weights(1000.,'unscaled')
weight=ctl.model.total_mass*9.81
source=Path(v16_wbc_qp.__file__).read_text().replace(
 'H = _eye(30,M)*tracking_diagonal[...,None,:] + torque_regularization*(_transpose(Ttau)@Ttau)',
 'Tobj = Ttau * (limit / _array([8.3,18.,18.]*4,M))[..., :,None]\n    H = _eye(30,M)*tracking_diagonal[...,None,:] + torque_regularization*(_transpose(Tobj)@Tobj)')
source=source.replace('q = q+torque_regularization*_matvec(_transpose(Ttau),tau_bias)',
 'q = q+torque_regularization*_matvec(_transpose(Tobj),h[...,6:]/_array([8.3,18.,18.]*4,M))')
ns={}; exec(compile(source,'<analysis-only-rated-objective>','exec'),ns)
candidate=ns['solve_wbc_qp']

def case(q,p,quat,vel,ades,acont,lower,upper,tag,t,logged_tau=None):
 base=ctl._base(p,quat)
 M=ctl.model.mass_matrix(q,**base); h=ctl.model.bias(q,vel,include_damping=False,**base)
 J=ctl._jacobians(q,base)
 target=J@acont if acont is not None else np.zeros((4,3))
 return dict(M=M,h=h,Jcontact=J,desired_acceleration=ades,stance=np.ones(4),
  contact_target=target,limits=hard,tau_lower=lower,tau_upper=upper,weight=weight,
  acceleration_weights=track),dict(tag=tag,time=t,logged_tau=logged_tau)

cases=[]; meta=[]
for drop in [0.,.10]:
 q,height=ctl.crouch_pose(drop,forward_shift_m=.03)
 args,m=case(q,[0,.03,height],[1,0,0,0],np.zeros(18),np.zeros(18),None,-.9*hard,.9*hard,'static_'+str(drop),None)
 cases.append(args); meta.append(m)
for tag,start,end in [('first_stand',3.92,7.),('poststop_stand',18.8,23.),('final_crouch',26.02,28.)]:
 for i in np.linspace(round(start/.02),round(end/.02)-1,20).astype(int)*4:
  s=status[i]; vec=s['wbc_vectors']; q=data['joint_position'][i].astype(float); dq=data['joint_velocity'][i].astype(float)
  quat=data['quaternion'][i].astype(float); p=data['position'][i].astype(float)
  ang=data['angular_velocity'][i].astype(float)
  R=ctl.model.pin.Quaternion(quat[[1,2,3,0]]).matrix()
  lin=R.T@data['world_velocity'][i].astype(float)
  entry=0 if i<1600 else 3460
  eq=data['quaternion'][entry].astype(float); ER=ctl.model.pin.Quaternion(eq[[1,2,3,0]]).matrix()
  yaw=np.arctan2(ER[1,0],ER[0,0]); c,sn=np.cos(yaw),np.sin(yaw)
  RT=np.array([[c,-sn,0],[sn,c,0],[0,0,1]])
  aw=np.clip(33.0577*(np.array(vec['base_position_reference_world'])-p)-19.5438*(R@lin),-15,15)
  ar=np.clip(32.1831*ctl.model.pin.log3(R.T@RT)-12.6411*ang,-15,15)
  ref=np.array(vec['joint_reference']); prev=np.array(status[i-4]['wbc_vectors']['joint_reference'])
  aj=np.clip(ctl.kp*1.6475*(ref-q)+ctl.kd*(np.clip((ref-prev)/.02,-3,3)-dq),-150,150)
  ades=np.r_[R.T@aw-np.cross(ang,lin),ar,aj]
  speed=np.clip(dq/50,-1,1)
  lo=.9*hard*np.maximum(-1-speed,-1); hi=.9*hard*np.minimum(1-speed,1)
  args,m=case(q,p,quat,np.r_[lin,ang,dq],ades,np.array(vec['acceleration']),lo,hi,tag,i*.005,np.array(s['requested_torque']))
  cases.append(args); meta.append(m)

def exact(args,norm,reg,fixed=None):
 M,h,J=args['M'],args['h'],args['Jcontact'].reshape(12,18)
 T=np.c_[M[6:]*scale,-weight*J[:,6:].T]
 C=np.r_[np.c_[M[:6]*scale,-weight*J[:,:6].T],np.c_[J*scale,np.zeros((12,12))]]
 b=np.r_[-h[:6],args['contact_target'].ravel()]
 if fixed is not None:
  C=np.r_[C[:6],np.c_[np.eye(18),np.zeros((18,12))]]; b=np.r_[b[:6],fixed/scale]
 H=np.diag(np.r_[track,np.full(12,.01)])+reg*(T/norm[:,None]).T@(T/norm[:,None])
 g=np.r_[-track*args['desired_acceleration']/scale,np.zeros(12)]+reg*(T/norm[:,None]).T@(h[6:]/norm)
 # Row scaling improves the KKT condition without changing constraints.
 row=np.linalg.norm(C,axis=1); C=C/row[:,None]; b=b/row
 K=np.block([[H,C.T],[C,np.zeros((len(C),len(C)))]])
 x=np.linalg.solve(K,np.r_[-g,b])[:30]
 F=np.zeros((20,30)); lo=np.full(20,-np.inf); hi=np.zeros(20)
 for k in range(4):
  F[5*k,20+3*k]=weight; lo[5*k]=0; hi[5*k]=500
  for a,(axis,sign) in enumerate([(0,1),(0,-1),(1,1),(1,-1)]):
   F[5*k+1+a,18+3*k+axis]=sign*weight; F[5*k+1+a,20+3*k]=-.6/np.sqrt(2)*weight
 D=np.r_[F,T]; dl=np.r_[lo,args['tau_lower']-h[6:]]; du=np.r_[hi,args['tau_upper']-h[6:]]
 viol=max(np.max(dl-D@x),np.max(D@x-du),0)
 success=True
 if viol>1e-7:
  r=minimize(lambda z:.5*z@H@z+g@z,x,jac=lambda z:H@z+g,
   constraints=[LinearConstraint(C,b,b),LinearConstraint(D,dl,du)],method='SLSQP',options={'ftol':1e-11,'maxiter':500})
  x=r.x; success=bool(r.success)
 return x[:18]*scale,(weight*x[18:]).reshape(4,3),T@x+h[6:],dict(success=success,ineq=max(np.max(dl-D@x),np.max(D@x-du),0),eq=np.max(np.abs(C@x-b)))

rows=[]
variants=[('baseline',hard,.01),('rated_001',rated,.01),('rated_01',rated,.1),('rated_1',rated,1.),('rated_10',rated,10.)]
for n,(args,m) in enumerate(zip(cases,meta)):
 results={}
 for name,norm,reg in variants:
  a,f,t,d=exact(args,norm,reg)
  fn=v16_wbc_qp.solve_wbc_qp if name=='baseline' else candidate
  aa,ff,tt,dd,_=fn(**args,torque_regularization=reg,iterations=320)
  fixed_a=np.array(status[round(m['time']/.005)]['wbc_vectors']['acceleration']) if m['time'] is not None else np.zeros(18)
  fa,ffix,tfix,fd=exact(args,norm,reg,fixed=fixed_a)
  results[name]=dict(torque=t.tolist(),forces=f.tolist(),base_acceleration=a[:6].tolist(),base_tracking_error=(a[:6]-args['desired_acceleration'][:6]).tolist(),
   tracking_cost=float(np.sum(track*((a-args['desired_acceleration'])/scale)**2)),rated_squared=float(np.sum((t/rated)**2)),exact_diagnostics=d,
   admm_torque=tt.tolist(),admm_feasible=bool(dd['feasible']),admm_converged=bool(dd['converged']),admm_constraint_violation=float(dd['max_constraint_violation']),
   admm_base_acceleration=aa[:6].tolist(),admm_exact_torque_gap=float(np.max(np.abs(tt-t))),fixed_acceleration_torque=tfix.tolist(),fixed_diagnostics=fd)
 row=dict(tag=m['tag'],time=m['time'],desired_acceleration=args['desired_acceleration'].tolist(),results=results)
 if m['logged_tau'] is not None: row['logged_torque']=m['logged_tau'].tolist()
 rows.append(row)
 if n%10==0: print('processed',n+1,'/',len(cases),flush=True)
summary={}
for tag in sorted(set(r['tag'] for r in rows)):
 subset=[r for r in rows if r['tag']==tag]; summary[tag]={}
 for name,_,_ in variants:
  rr=[r['results'][name] for r in subset]; tor=np.array([r['torque'] for r in rr]); at=np.array([r['admm_torque'] for r in rr]); fix=np.array([r['fixed_acceleration_torque'] for r in rr])
  summary[tag][name]=dict(samples=len(rr),exact_joint_rms=np.sqrt(np.mean(tor**2,axis=0)).tolist(),admm_joint_rms=np.sqrt(np.mean(at**2,axis=0)).tolist(),
   fixed_acceleration_joint_rms=np.sqrt(np.mean(fix**2,axis=0)).tolist(),base_tracking_rms=np.sqrt(np.mean(np.array([r['base_tracking_error'] for r in rr])**2,axis=0)).tolist(),
   tracking_cost_mean=float(np.mean([r['tracking_cost'] for r in rr])),rated_squared_mean=float(np.mean([r['rated_squared'] for r in rr])),
   admm_feasible_count=sum(r['admm_feasible'] for r in rr),admm_converged_count=sum(r['admm_converged'] for r in rr),admm_max_constraint_violation=max(r['admm_constraint_violation'] for r in rr),
   admm_max_exact_torque_gap=max(r['admm_exact_torque_gap'] for r in rr),exact_max_inequality_violation=max(r['exact_diagnostics']['ineq'] for r in rr))
notes=['Offline counterfactuals on baseline states; not closed-loop evidence.',
 'Production sources unchanged. Rated objective obtained by in-memory substitution only; hard bounds and tracking weights unchanged.',
 'Contact target reconstructed as J times logged acceleration, hence drops logged tiny equality residual.',
 'Desired acceleration reconstructed from logged state and references; target yaw latched at 0 or 17.3 s (BLEND_STAND entry).',
 '20 samples per steady phase; all four contacts assumed after confirming logged stance.',
 'Exact optimum uses equality KKT and SLSQP if inequality-active. ADMM is production fixed 320 iterations, cold initialized.',
 'Fixed-acceleration result permits contact force redistribution only and cannot sacrifice acceleration tracking.']
(OUT/'results.json').write_text(json.dumps(dict(notes=notes,summary=summary,rows=rows),indent=2))
for tag,s in summary.items():
 print(tag)
 for name,v in s.items(): print(name,'knees',np.round(np.array(v['exact_joint_rms'])[2::3],3),'fixed',np.round(np.array(v['fixed_acceleration_joint_rms'])[2::3],3),'baseerr',np.round(v['base_tracking_rms'],3),'ADMM',v['admm_feasible_count'],'gap',round(v['admm_max_exact_torque_gap'],3))
