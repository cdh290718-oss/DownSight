"""Diagnostic triangulation of automatic points; never a fitting input."""
import json
import cv2
import numpy as np
from common import RUN,OUT,OLD,read,project_numpy
inputs=read(OUT/'inputs.json');cl,cr=[inputs['views'][s]['camera'] for s in ['left','right']];T=np.array(cr['left_to_view']);R,t=T[:3,:3],T[:3,3]
with np.load(OLD/'dataset/samples/000001/left_egosmplx65.npz') as z:gt=z['joints_3d_camera_m'];visible=z['in_frame_no_occlusion_test']

def triang(q):
    rays=[]
    for uv,c in zip(np.asarray(q).reshape(2,2),[cl,cr]):
        u=cv2.undistortPointsIter(uv.reshape(1,1,2),np.array(c['K']),np.array(c['distortion']),None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(2);rays.append(u)
    h=cv2.triangulatePoints(np.eye(4)[:3],T[:3],rays[0].reshape(2,1),rays[1].reshape(2,1))[:,0]
    return h[:3]/h[3]
pairs=[]
left,right=[inputs['views'][s] for s in ['left','right']];rb={q['name']:q for q in right['body_points']}
for p in left['body_points']:
    if p['name'] in rb:pairs.append(('sapiens',p['name'],p['index'],np.array([p['xy'],rb[p['name']]['xy']]),True))
for h in ['left','right']:
    if not all(h in v['hands'] for v in [left,right]):continue
    a,b=[v['hands'][h] for v in [left,right]];i=0 if h=='left' else 1;ids=[12+i]+list(range(25+i*20,45+i*20))
    for k,idx in enumerate(ids):pairs.append(('wilor',h+'_'+str(k),idx,np.array([a['xy'][k],b['xy'][k]]),k in a['stereo_passed_indices']))
rows=[]
for source,name,idx,uv,passed in pairs:
    xyz=triang(uv);jac=np.zeros((3,4));epsilon=.1
    for k in range(4):
        plus=uv.copy().reshape(4);minus=uv.copy().reshape(4);plus[k]+=epsilon;minus[k]-=epsilon;jac[:,k]=(triang(plus)-triang(minus))/(2*epsilon)
    cov=4*jac@jac.T;xyz_r=R@xyz+t;pred=np.stack([project_numpy(xyz[None],c)[0] for c in [cl,cr]]);error=(xyz-gt[idx])*1000
    rows.append(dict(source=source,name=name,model_index=idx,xyz_left_m=xyz.tolist(),error_xyz_mm=error.tolist(),absolute_mm=float(np.linalg.norm(error)),truth_left_inframe=bool(visible[idx]),positive_depth_both=bool(xyz[2]>0 and xyz_r[2]>0),reprojection_mean_px=float(np.linalg.norm(pred-uv,axis=1).mean()),stereo_epipolar_gate_pass=bool(passed),covariance_left_m2=cov.tolist(),depth_std_mm=float(np.sqrt(cov[2,2])*1000)))
result=dict(scope='Diagnostics only; actual 2D detections, generating camera calibration. No GT points used in reconstruction. Fit uses 2D residuals only.',covariance_assumption='Independent isotropic 2 px noise per image coordinate; finite-difference linear propagation. Excludes detector bias, anatomical definition and calibration uncertainty; not calibrated confidence.',points=rows)
(RUN/'evidence/triangulation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('triangulated',len(rows),'automatic point pairs')
