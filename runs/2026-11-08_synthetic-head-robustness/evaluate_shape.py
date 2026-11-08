"""Held-out shape evaluation in common neutral pose, separate from fitting."""
import hashlib,json,os
import numpy as np
import torch
from common import ROOT,RUN,OUT,OLD,read
from egosmplx_pipeline.bootstrap import setup
cfg=read(OLD/'configs/left.json');paths=cfg['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo']);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
source=OLD/'dataset/samples/000001/smplx_reconstruction.npz'
with np.load(source) as z:gt=z['betas'].copy().reshape(1,-1)
with torch.no_grad():
 def shape(beta):
    z=torch.zeros(1,3);out=layer(betas=torch.tensor(beta,dtype=torch.float32).reshape(1,-1),global_orient=z,body_pose=torch.zeros(1,63),left_hand_pose=torch.zeros(1,45),right_hand_pose=torch.zeros(1,45),jaw_pose=z,leye_pose=z,reye_pose=z,expression=torch.zeros(1,10));v=out.vertices[0].numpy();j=out.joints[0].numpy();return v-j[0],j-j[0]
 gv,gj=shape(gt)
 def measures(v,j):
    pairs={'shoulder_joint_width':(16,17),'hip_joint_width':(1,2),'head_to_pelvis':(15,0),'neck_to_pelvis':(12,0),'left_upper_arm':(16,18),'right_upper_arm':(17,19),'left_forearm':(18,20),'right_forearm':(19,21),'left_thigh':(1,4),'right_thigh':(2,5),'left_shin':(4,7),'right_shin':(5,8)}
    result={k:float(np.linalg.norm(j[a]-j[b])*1000) for k,(a,b) in pairs.items()};result['neutral_vertex_y_extent']=float((v[:,1].max()-v[:,1].min())*1000);return result
 cases=[('B_previous',ROOT/'outputs/synthetic_stereo_abcd_20261108/B_left/final.npz'),('D_oracle_shape',ROOT/'outputs/synthetic_stereo_abcd_20261108/D_left/final.npz'),('E_previous',ROOT/'outputs/synthetic_shape_prior_20261108/E_left/final.npz')]+[(c['case']+'_'+c['seed'],OUT/(c['case']+'_'+c['seed'])/'final.npz') for c in read(RUN/'cases.json')]
 rows=[]
 for label,path in cases:
    with np.load(path) as z:b=z['betas'].copy()
    v,j=shape(b);metrics=measures(v,j);rows.append(dict(case=label,betas_squared_mean=float(np.mean(b*b)),betas_error_rmse=float(np.sqrt(np.mean((b-gt)**2))),neutral_pelvis_centered_vertex_mean_mm=float(np.linalg.norm(v-gv,axis=1).mean()*1000),neutral_native22_joint_mean_mm=float(np.linalg.norm(j[:22]-gj[:22],axis=1).mean()*1000),measures_mm=metrics,measure_error_mm={k:metrics[k]-measures(gv,gj)[k] for k in metrics},source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
result=dict(exit_status=0,scope='Held-out GT betas evaluation only. Zero body/hand/face pose, global rotation and expression for every shape; subtract own native pelvis, no scale/rotation alignment. Y extent is a neutral-mesh bounding extent, not measured standing stature. Coefficient norm alone is not proof of anatomical validity.',truth_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),truth_betas_squared_mean=float(np.mean(gt*gt)),truth_measures_mm=measures(gv,gj),cases=rows)
(RUN/'evidence/shape_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('evaluated shapes',len(rows))
