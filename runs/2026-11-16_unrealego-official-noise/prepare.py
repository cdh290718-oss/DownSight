"""Prepare independent observations. Raw coordinates remain under ignored outputs."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]; RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/2026-11-04_session-hand6-v2/code'))
from prepare import Camera
read=lambda p: json.loads(Path(p).read_text())
ARCH=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
OUT=ROOT/'outputs/unrealego_official_noise_v2_20261116'
cfg=read(RUN/'protocol.json'); rig=read(ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json'); R=np.array(rig['rig_rotation']); t=np.array(rig['rig_translation_m']); cams=[Camera(ARCH/'body_calibration_fisheye'/('camera_'+x+'.json')) for x in ['left','right']]
names=['Neck','R_Shoulder','R_Elbow','R_Wrist','L_Shoulder','L_Elbow','L_Wrist','R_Hip','R_Knee','R_Ankle','R_Toe','L_Hip','L_Knee','L_Ankle','L_Toe']; ids=[7,9,11,13,8,10,12,2,4,6,19,1,3,5,16]
autonames={1:'right_shoulder',2:'right_elbow',3:'right_wrist',4:'left_shoulder',5:'left_elbow',6:'left_wrist',7:'right_hip',8:'right_knee',9:'right_ankle',11:'left_hip',12:'left_knee',13:'left_ankle'}
def tri(uv,sigma):
 rays=[c.camera2world_ray(np.array([u],float))[0] for c,u in zip(cams,uv)]; origin=-R.T@t; b=R.T@rays[1]; A=np.stack([rays[0],-b],axis=1); d=np.linalg.lstsq(A,origin,rcond=None)[0]; a=rays[0]*d[0]; bb=origin+b*d[1]; gap=float(np.linalg.norm(a-bb)); angle=float(np.degrees(np.arccos(np.clip(rays[0]@b,-1,1)))); info={'gap_m':gap,'angle_deg':angle,'distances_m':d.tolist(),'geometry_ok':False,'depth_ok':False}
 if min(d)<cfg['min_ray_distance_m'] or max(d)>cfg['max_ray_distance_m'] or gap>cfg['max_ray_gap_m'] or angle<cfg['min_ray_angle_deg']:return info
 def residual(x):return np.concatenate([cams[0].world2camera(x[None])[0]-uv[0],cams[1].world2camera((R@x+t)[None])[0]-uv[1]])
 fit=least_squares(residual,(a+bb)/2,diff_step=1e-4,max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10); x=fit.x; err=np.linalg.norm(residual(x).reshape(2,2),axis=1); C=np.linalg.pinv(fit.jac.T@fit.jac)*sigma**2+np.eye(3)*cfg['metric_floor_m']**2; direction=x/np.linalg.norm(x); depthstd=float(np.sqrt(max(0,direction@C@direction))); ok=bool(fit.success and np.isfinite(C).all() and x[2]>.02 and (R@x+t)[2]>.02 and err.max()<=cfg['max_reprojection_px']);info.update(geometry_ok=ok,depth_ok=bool(ok and depthstd<=cfg['depth_sigma_threshold_m']),depth_std_m=depthstd,reprojection_max_px=float(err.max()),xyz=x.tolist(),covariance=C.tolist());return info
OUT.mkdir(parents=True,exist_ok=False)
summary=[]
for pair in range(1,5):
 name=f'body{pair:02d}'; old=ROOT/'outputs/unrealego_gt2d_ideal_20261115'/name; obs=read(old/'observations_local.json'); gt=np.asarray(obs['views'],float); base=gt[:,:,:2]; valid=(gt[:,:,2]>0)&(base[:,:,0]>=280)&(base[:,:,0]<1000)&(base[:,:,1]>=0)&(base[:,:,1]<720)
 det=[]
 for side in ['left','right']:
  path=ARCH/'body_work_full_image'/(name+'_'+side)/'observations'/(name+'_'+side)/'sapiens/keypoints.json'; det.append({q['name']:q for q in read(path)['keypoints']})
 conditions=[(f'gt_s{sigma}_seed{seed}',seed,sigma) for sigma in cfg['noise_sigma_px'] for seed in cfg['seeds']]
 for condition,seed,sigma in conditions:
  xy=np.zeros((2,15,2)); good=np.zeros((2,15),bool)
  if seed is not None:
   rng=np.random.default_rng(np.random.SeedSequence([seed,pair,int(sigma)]));xy=base+rng.normal(0,sigma,size=base.shape);good=valid&(xy[:,:,0]>=280)&(xy[:,:,0]<1000)&(xy[:,:,1]>=0)&(xy[:,:,1]<720)
  else:
   for j,n in autonames.items():
    for v in range(2):
     q=det[v].get(n)
     if q:
      xy[v,j]=[q['x'],q['y']];good[v,j]=q['score']>=cfg['auto_score_threshold'] and 280<=q['x']<1000 and 0<=q['y']<720
  pts=[]
  for j,(n,idx) in enumerate(zip(names,ids)):
   q=tri(xy[:,j],sigma) if good[:,j].all() else {'geometry_ok':False,'depth_ok':False};kind=n.split('_')[-1].lower();sig={'shoulder':.03,'hip':.03,'elbow':.02,'knee':.02,'ankle':.015,'wrist':.008,'neck':.03,'toe':.03}.get(kind,.03); z={'name':n,'index':idx,'source':'sapiens','group':'shoulder' if kind=='shoulder' else 'wrist' if kind=='wrist' else 'body','accepted':q['depth_ok'],'valid':good[:,j].tolist(),'xy':xy[:,j].tolist(),'sigma_px':sigma,'tri':q}
   if q.get('depth_ok'):z.update(xyz=q['xyz'],covariance=(np.array(q['covariance'])+np.eye(3)*sig**2).tolist())
   pts.append(z)
  dest=OUT/name/condition;dest.mkdir(parents=True); payload={'points':pts,'sigma_px':sigma,'condition':condition,'pair':name}; (dest/'observations.json').write_text(json.dumps(payload,indent=2)+'\n'); summary.append({'pair':name,'condition':condition,'view_points':int(good.sum()),'accepted_3d':sum(p['accepted'] for p in pts),'2d_fallback':sum(not p['accepted'] and any(p['valid']) for p in pts)})
(RUN/'evidence/preparation.json').write_text(json.dumps({'exit_status':0,'conditions':summary},indent=2)+'\n');print('prepared',len(summary))
