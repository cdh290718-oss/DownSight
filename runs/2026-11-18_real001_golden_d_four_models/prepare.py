"""Use only observed image points and the same estimated historical Head anchor."""
import copy,sys,os,json
from pathlib import Path
import numpy as np,torch,cv2
from common import ROOT,RUN,OUT,read,write,sha,setup_model
from geometry import BodyKinematics,project
records=read(OUT/'records.json');pred=read(OUT/'detections.json');anchor=read(ROOT/'outputs/real_d_stereo_20261115/anchors/oracle.json');inputs=read(ROOT/'outputs/real_d_stereo_20261115/inputs.json');model,Camera=setup_model();torch.set_num_threads(2)
from egosmplx_pipeline import fusion
fusion.initialize(model);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
# Confirm the archived anchor still matches its historical SMPL-X source.
p=dict(np.load(records[0]['original']['body']));t=lambda v:torch.tensor(v,dtype=torch.float32).reshape(1,-1)
with torch.no_grad():native=layer(**{k:t(p[k]) for k in fusion.PARAMS}).joints[0].numpy()+p['transl'].reshape(3)
R=cv2.Rodrigues(p['global_orient'].reshape(3).astype(float))[0]
for j in [3,6,9,12,15]:R=R@cv2.Rodrigues(p['body_pose'].reshape(21,3)[j-1].astype(float))[0]
assert np.allclose(native[15],anchor['position'],atol=1e-6);assert np.allclose(R,anchor['rotation'],atol=1e-6)
commonmask=[[all(pred[m][v]['points'][j]['in_image'] for m in pred) for v in range(2)] for j in range(12)];assert all(any(x) for x in commonmask)
items=[]
for name,views in pred.items():
 points=[]
 for j,q in enumerate(views[0]['points']):
  part=q['name'].split('_')[1];assert int(model.joint_idx[q['index']])==q['native_index']
  points.append(dict(name=q['name'],part=part,index=q['index'],native_index=q['native_index'],group='shoulder' if part=='shoulder' else 'wrist' if part=='wrist' else 'body',xy=[views[v]['points'][j]['xy'] for v in range(2)],valid=commonmask[j],scores=[views[v]['points'][j]['score'] for v in range(2)],accepted=False))
 items.append(dict(pair='real001_'+name,condition='baseline',detector=name,inputs=copy.deepcopy(inputs),anchor=copy.deepcopy(anchor),points=points))
write(OUT/'prepared.json',{'items':items});write(OUT/'estimated_head.json',anchor)
# Validate the projector on finite image rays before committing to 20k steps.
projection_checks=[]
for rec in records:
 cam=Camera(rec['row']['calibration']);uv=np.array([[300.,250.],[640.,360.],[950.,550.]]);xyz=cam.camera2world_ray(uv)*1.2;np_uv=cam.world2camera(xyz);pt_uv=project(torch.tensor(xyz,dtype=torch.float32),cam).detach().numpy();delta=float(np.max(np.abs(np_uv-pt_uv)));assert delta<0.01,delta;projection_checks.append({'camera':rec['camera'],'calibration_sha256':sha(rec['row']['calibration']),'torch_numpy_projection_max_px':delta})
write(RUN/'evidence/preparation.json',dict(command='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <PYTHON_ENV>/bin/python prepare.py',exit_status=0,head_source='historical original cam3 SMPL-X body, native Head position and kinematic rotation; estimate, not ground truth',head_source_sha256=sha(records[0]['original']['body']),head_anchor=anchor,camera_transforms=inputs,projection_checks=projection_checks,common_joint_view_mask=commonmask,retained_joint_views=sum(map(sum,commonmask)),detectors=list(pred),shape_rank=0,hand_observation_loss=False,body_3d_observation_loss=False,source_fit_sha256=sha(ROOT/'runs/2026-11-17_newdetector_d10/fit.py'),fit_copy_sha256=sha(RUN/'fit.py'),geometry_copy_sha256=sha(RUN/'geometry.py')))
print('Prepared four independent fits, retained joint-views',sum(map(sum,commonmask)),flush=True)
