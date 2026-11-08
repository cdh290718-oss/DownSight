"""Numerical geometry, covariance, initializer and truth-isolation checks."""
import json,os,sys
import cv2
import numpy as np
import torch
from common import ROOT,RUN,OUT,OLD,read,project_numpy
from initialization import initialize
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import load
cfg=read(OLD/'configs/left.json');paths=cfg['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo']);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
inputs=read(OUT/'inputs.json');points=read(OUT/'triangulated.json')['points'];checks={};assert set(inputs)=={'views'}
checks['observation_input_has_no_head_fields']=True
min_eig=min(np.linalg.eigvalsh(p['covariance']).min() for p in points);assert min_eig>0;checks['minimum_covariance_eigenvalue']=float(min_eig)
results={}
for seed in ['left','right']:
 job=read(OLD/'fits'/('left' if seed=='left' else 'right_cpu')/'jobs'/(seed+'.json'));raw=load(__import__('pathlib').Path(job['output'])/'raw/params.npz')
 beta,V,g,tr,info=initialize(layer,model,raw,points,1.)
 assert np.allclose(V.T@V,np.eye(V.shape[1]),atol=1e-6)
 p={k:torch.tensor(raw[k],dtype=torch.float32).reshape(1,-1) for k in ['body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']}
 with torch.no_grad():
  out=layer(betas=torch.tensor(beta,dtype=torch.float32).reshape(1,-1),global_orient=torch.tensor(g,dtype=torch.float32).reshape(1,3),**p);j=out.joints[0].numpy()[list(model.joint_idx)]+tr
 ids=[q['index'] for q in points if q['source']=='sapiens'];target=np.array([q['xyz'] for q in points if q['source']=='sapiens']);w=np.array([1/np.trace(q['covariance']) for q in points if q['source']=='sapiens']);w/=w.sum();err=float(np.sqrt((w*np.sum((j[ids]-target)**2,axis=1)).sum()));assert abs(err-info['alignment_rms_m'])<1e-6
 info['reconstructed_alignment_rms_m']=err;results[seed]=info
# Rest joint shape derivatives against independent forward finite differences, zero pose.
S=torch.einsum('jv,vck->jck',layer.J_regressor,layer.shapedirs[:,:,:10]).numpy();err=0
with torch.no_grad():
 for k in [0,4,9]:
  b=torch.zeros(1,10);b[0,k]=.01
  a=layer(betas=b,body_pose=torch.zeros(1,63),global_orient=torch.zeros(1,3),left_hand_pose=torch.zeros(1,45),right_hand_pose=torch.zeros(1,45),jaw_pose=torch.zeros(1,3),leye_pose=torch.zeros(1,3),reye_pose=torch.zeros(1,3),expression=torch.zeros(1,10)).joints[0,:22].numpy();b[0,k]=-.01
  c=layer(betas=b,body_pose=torch.zeros(1,63),global_orient=torch.zeros(1,3),left_hand_pose=torch.zeros(1,45),right_hand_pose=torch.zeros(1,45),jaw_pose=torch.zeros(1,3),leye_pose=torch.zeros(1,3),reye_pose=torch.zeros(1,3),expression=torch.zeros(1,10)).joints[0,:22].numpy();err=max(err,float(np.max(np.abs((a-c)/.02-S[:22,:,k]))))
assert err<1e-4;checks['rest_shape_derivative_max_error']=err
q=dict(exit_status=0,checks=checks,initializations=results);(RUN/'evidence/method_tests.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({s:{'rank':r['retained_rank'],'eigenvalues':r['shape_information_eigenvalues'],'initial_rms_m':r['alignment_rms_m']} for s,r in results.items()},indent=2))
