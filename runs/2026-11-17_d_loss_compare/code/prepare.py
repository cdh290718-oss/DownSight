import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
root=Path.cwd();run=root/'runs/2026-11-17_d_loss_compare';out=root/'outputs/d_loss_compare_20261117';sys.path.insert(0,str(root/'runs/2026-11-04_session-hand6-v2/code'));from prepare import Camera
arch=root.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';cams=[Camera(arch/'body_calibration_fisheye'/('camera_'+x+'.json')) for x in ('left','right')];rig=json.loads((root/'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json').read_text());R=np.array(rig['rig_rotation']);t=np.array(rig['rig_translation_m']);data=json.loads((root/'outputs/newdetector_d10_20261117/prepared.json').read_text());stats=[]
for row in data['items']:
 for p in row['points']:
  info={'accepted':False,'reason':'missing view'}
  if all(p['valid']):
   uv=np.array(p['xy']); rays=[cam.camera2world_ray(np.array([u],float))[0] for cam,u in zip(cams,uv)];origin=-R.T@t;b=R.T@rays[1];d=np.linalg.lstsq(np.stack([rays[0],-b],axis=1),origin,rcond=None)[0];a=rays[0]*d[0];bb=origin+b*d[1];gap=float(np.linalg.norm(a-bb));angle=float(np.degrees(np.arccos(np.clip(rays[0]@b,-1,1))));info={'accepted':False,'reason':'geometry','gap_m':gap,'angle_deg':angle,'distances_m':d.tolist()}
   if min(d)>=.12 and max(d)<=2.5 and gap<=.04 and angle>=1:
    def residual(x):return np.concatenate([cams[0].world2camera(x[None])[0]-uv[0],cams[1].world2camera((R@x+t)[None])[0]-uv[1]])
    fit=least_squares(residual,(a+bb)/2,diff_step=1e-4,max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10);x=fit.x;err=np.linalg.norm(residual(x).reshape(2,2),axis=1);C=np.linalg.pinv(fit.jac.T@fit.jac)*8**2+np.eye(3)*.008**2;direction=x/np.linalg.norm(x);depthstd=float(np.sqrt(max(0,direction@C@direction)));ok=bool(fit.success and np.isfinite(C).all() and x[2]>.02 and (R@x+t)[2]>.02 and err.max()<=20 and depthstd<=.5);info.update(accepted=ok,reason='accepted' if ok else 'depth_or_reprojection',depth_std_m=depthstd,reprojection_max_px=float(err.max()))
    if ok:
     sig={'shoulder':.03,'hip':.03,'elbow':.02,'knee':.02,'ankle':.015,'wrist':.008}.get(p['part'],.03);p.update(xyz=x.tolist(),covariance=(C+np.eye(3)*sig**2).tolist())
  p['accepted']=info['accepted'];stats.append(dict(pair=row['pair'],joint=p['name'],**info))
data['protocol_sha256']=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest();(out/'prepared.json').open('x').write(json.dumps(data,indent=2));(run/'evidence/gating.json').write_text(json.dumps(dict(command='prepare.py',exit_status=0,accepted=sum(x['accepted'] for x in stats),total=len(stats),records=stats),indent=2)+'\n');print('accepted',sum(x['accepted'] for x in stats),'/',len(stats))
