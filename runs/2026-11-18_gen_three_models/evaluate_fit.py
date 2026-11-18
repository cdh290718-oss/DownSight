import json,sys,numpy as np,torch
from pathlib import Path
from common import RUN,OUT,ROOT,read,setup_model,sha,write
from geometry import BodyKinematics

def align(x,y):
 mx,my=x.mean(0),y.mean(0);a=x-mx;b=y-my;u,s,v=np.linalg.svd(a.T@b);sgn=1 if np.linalg.det(v.T@u.T)>0 else -1;diag=np.array([1.,1.,sgn]);R=(v.T*diag)@u.T;scale=np.sum(s*diag)/np.sum(a*a);return scale*a@R.T+my
cfg=read(RUN/'protocol.json');done=read(OUT/'fit_complete.json');assert done['exit_status']==0 and done['protocol_sha256']==sha(RUN/'protocol.json');gtfile=ROOT.parent/'gen_egosmplx_run/dfm_reference_virtual.npz';gt=np.load(gtfile);frames=list(map(int,gt['frames']));model,_=setup_model();layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False);fk=BodyKinematics(layer);z=np.load(OUT/'budget_20000.npz');ids=done['ids'];d=read(OUT/'diagnostic_20000.json');nameindex=[16,17,18,19,20,21,1,2,4,5,7,8];records=[];torch.set_num_threads(2)
for k,(pair,condition,method) in enumerate(ids):
 if int(z['selected_step'][k])<0:records.append(dict(pair=pair,model=condition,status='failed'));continue
 t=lambda x:torch.tensor(z[x][k:k+1],dtype=torch.float32)
 with torch.no_grad():native,_=fk(t('body_pose'),t('betas'),t('global0'),t('delta'),t('translation'))
 pred=native[0,nameindex].numpy();frame=int(pair.split('_')[1]);ref=gt['cam3_body_xyz'][frames.index(frame),nameindex];assert np.isfinite(ref).all();err=np.linalg.norm(pred-ref,axis=1)*1000;pa=np.linalg.norm(align(pred,ref)-ref,axis=1)*1000
 records.append(dict(pair=pair,model=condition,status='ok',absolute_per_joint_mm=err.tolist(),pa_per_joint_mm=pa.tolist(),selected_step=int(z['selected_step'][k]),convergence_pass=bool(d['convergence_pass'][k])))
write(RUN/'evidence/3d_results.json',dict(command='evaluate_fit.py',exit_status=0,reference='DFM model body reference, unvalidated',joint_indices=nameindex,records=records));print('evaluated',len(records),flush=True)
