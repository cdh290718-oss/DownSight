"""Separate anchor preparation; only cameras and automatic detections go to fitting inputs."""
import json,hashlib
import cv2
import numpy as np
from common import ROOT,RUN,OUT,read,project_numpy
cfg=read(RUN/'settings.json');src=ROOT/'outputs/synthetic_shape_prior_20261108/inputs.json';old=read(src)
inputs={'views':old['views']};(OUT/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
anchor={'position':old['head_position_m'],'rotation':old['head_rotation_matrix']}
(OUT/'anchors').mkdir(exist_ok=False)
(OUT/'anchors/oracle.json').write_text(json.dumps(anchor,indent=2)+'\n')
perturbations=[]
# Deterministic bounded SE3 stress tests, not Monte Carlo confidence intervals.
directions=[np.array([1.,1.,1.]),np.array([-1.,1.,2.])];axes=[np.array([1.,-2.,1.]),np.array([2.,1.,-1.])]
for mm in [10,20]:
 for d,(direction,axis) in enumerate(zip(directions,axes)):
  for sign in [-1,1]:
   shift=sign*mm/1000*direction/np.linalg.norm(direction);aa=sign*np.deg2rad(3)*axis/np.linalg.norm(axis);name='perturb_%d_d%d_%s'%(mm,d,'minus' if sign<0 else 'plus')
   x={'position':(np.array(anchor['position'])+shift).tolist(),'rotation':(cv2.Rodrigues(aa)[0]@np.array(anchor['rotation'])).tolist()}
   (OUT/'anchors'/(name+'.json')).write_text(json.dumps(x,indent=2)+'\n');perturbations.append(dict(case=name,translation_delta_m=shift.tolist(),rotation_axis_angle_rad=aa.tolist()))
(RUN/'evidence/perturbations.json').write_text(json.dumps(perturbations,indent=2)+'\n')
left,right=[inputs['views'][s] for s in ['left','right']];cameras=[v['camera'] for v in [left,right]];T=np.array(cameras[1]['left_to_view'])
def triang(uv):
 rays=[cv2.undistortPointsIter(np.array(q,float).reshape(1,1,2),np.array(c['K']),np.array(c['distortion']),None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2) for q,c in zip(np.array(uv).reshape(2,2),cameras)]
 h=cv2.triangulatePoints(np.eye(4)[:3],T[:3],rays[0].reshape(2,1),rays[1].reshape(2,1))[:,0];return h[:3]/h[3]
rows=[]
def add(name,index,group,uv,scores,definition,source,matched=True):
 uv=np.array(uv,float);xyz=triang(uv);J=np.zeros((3,4))
 for k in range(4):
  d=np.zeros(4);d[k]=.1;J[:,k]=(triang(uv.reshape(4)+d)-triang(uv.reshape(4)-d))/.2
 pixel_sigma=cfg['triangulation_pixel_sigma']/np.maximum(np.array(scores),.3)
 covariance_pixel=J@np.diag(np.repeat(pixel_sigma**2,2))@J.T;cov=covariance_pixel+np.eye(3)*cfg['definition_sigma_m'][definition]**2
 other=T[:3,:3]@xyz+T[:3,3];center=-T[:3,:3].T@T[:3,3];ray1=xyz/np.linalg.norm(xyz);ray2=(xyz-center)/np.linalg.norm(xyz-center);angle=float(np.rad2deg(np.arccos(np.clip(ray1@ray2,-1,1))));errors=np.linalg.norm(np.stack([project_numpy(xyz[None],c)[0] for c in cameras])-uv,axis=1)
 accepted=bool(matched and np.isfinite(xyz).all() and min(xyz[2],other[2])>1e-4 and angle>=cfg['triangulation_min_angle_deg'] and np.isfinite(errors).all() and errors.max()<=cfg['triangulation_max_reprojection_px'])
 rows.append(dict(name=name,index=index,group=group,source=source,xyz=xyz.tolist(),covariance=cov.tolist(),pixel_covariance=covariance_pixel.tolist(),definition_sigma_m=cfg['definition_sigma_m'][definition],pixel_sigma=pixel_sigma.tolist(),angle_deg=angle,reprojection_px=errors.tolist(),accepted=accepted,matched=bool(matched)))
rb={p['name']:p for p in right['body_points']}
for p in left['body_points']:
 if p['name'] in rb:
  r=rb[p['name']];add(p['name'],p['index'],p['group'],[p['xy'],r['xy']],[np.sqrt(p['weight']),np.sqrt(r['weight'])],p['name'].split('_')[-1],'sapiens')
for i,h in enumerate(['left','right']):
 if not all(h in v['hands'] for v in [left,right]):continue
 a,b=[v['hands'][h] for v in [left,right]];idx=[12+i]+list(range(25+20*i,45+20*i))
 for k in cfg['hand_fit_indices']:
  add(h+'_hand_'+str(k),idx[k],'hand_'+h,[a['xy'][k],b['xy'][k]],[a['confidence'],b['confidence']],'hand','wilor',k in a['stereo_passed_indices'] and k in b['stereo_passed_indices'])
q=dict(points=rows,assumptions='Independent pixel noise 3/confidence px and isotropic metric definition noise; local linear covariance, excludes calibration/bias correlation. Assumptions fixed before GT evaluation.',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
(OUT/'triangulated.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');(RUN/'evidence/triangulated.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
cases=[dict(case='legacy_oracle',seed='left',method='legacy',anchor='oracle'),dict(case='legacy_none',seed='left',method='legacy',anchor=None)]
for seed in ['left','right']:
 for anchorname in ['oracle','none']:cases.append(dict(case=anchorname,seed=seed,method='improved',anchor=None if anchorname=='none' else anchorname))
for p in perturbations:cases.append(dict(case=p['case'],seed='left',method='improved',anchor=p['case']))
(RUN/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print('accepted',sum(p['accepted'] for p in rows),'/',len(rows),'cases',len(cases))
