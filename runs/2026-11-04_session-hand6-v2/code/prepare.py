"""Freeze exact demo inputs and derive stereo targets without network inference."""
import argparse, json, hashlib, os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from egosmplx_pipeline.camera import FishEyeCameraCalibrated as Camera
from egosmplx_pipeline.v2_policy import interior

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def triangulate(cams,R,t,uv):
 rays=[c.camera2world_ray(np.array([u],float))[0] for c,u in zip(cams,uv)]
 origin=-R.T@t; b=R.T@rays[1];matrix=np.stack([rays[0],-b],axis=1)
 distances=np.linalg.lstsq(matrix,origin,rcond=None)[0];a=rays[0]*distances[0];bb=origin+b*distances[1]
 angle=float(np.rad2deg(np.arccos(np.clip(rays[0]@b,-1,1))))
 record={'ray_gap_m':float(np.linalg.norm(a-bb)),'ray_distances_m':distances.tolist(),'parallax_deg':angle,'accepted':False}
 if np.min(distances)<.12 or np.max(distances)>2.5 or record['ray_gap_m']>.04 or angle<1.:
  record['reason']='ray_distance_gap_or_parallax';return record
 def residual(x):return np.concatenate([cams[0].world2camera(x[None])[0]-uv[0],cams[1].world2camera((R@x+t)[None])[0]-uv[1]])
 fit=least_squares(residual,(a+bb)/2,diff_step=1e-4,max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 x=fit.x;cov=np.linalg.pinv(fit.jac.T@fit.jac)*8.**2+np.eye(3)*.008**2
 eig,vec=np.linalg.eigh(cov);std=np.sqrt(np.maximum(eig,0));error=residual(x).reshape(2,2)
 record.update(point_cam3_m=x.tolist(),covariance_cam3_m2=cov.tolist(),std_principal_m=std.tolist(),reprojection_residual_px=error.tolist())
 if not fit.success or not np.isfinite(cov).all() or std.max()>.5 or np.linalg.norm(error,axis=1).max()>20 or x[2]<.02 or (R@x+t)[2]<.02:
  record['reason']='reprojection_depth_or_uncertainty';return record
 record.update(accepted=True,reason='accepted',whiten_cam3=((vec*(1/std))@vec.T).tolist());return record

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--evidence',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 manifest=read(a.archive/'manifest.json');cfg=read('configs/pipeline.head01.developer.json');A=np.array(read(manifest['calibration_source'])['poseCamera2']['A']);R=A[:3,:3];t=A[:3,3]/1000
 records=[];inputs=[];tri_records=[]
 for pair in manifest['pairs']:
  frames=pair['frames'];cams=[Camera(f['calibration']) for f in frames];observations=[];weights=[];matches=[]
  for frame in frames:
   inp={'pair':pair['id'],'id':frame['id'],'camera':frame['camera'],'hashes':{}}
   for key,path in frame['frozen_files'].items():
    actual=sha(path);assert actual==frame['sha256'][key],(frame['id'],key);inp['hashes'][key]=actual
   inputs.append(inp);match=read(frame['frozen_files']['recipe'])['matches'];matches.append(match)
   ids=[v['candidate'] for v in match.values()];assert len(set(ids))==len(ids)
   w=np.load(frame['frozen_files']['wilor']);xy=np.zeros((2,21,2));weight=np.zeros((2,21))
   for si,side in enumerate(('left','right')):
    if side not in match:continue
    q=match[side];xy[si]=w['joints_2d'][q['candidate']];confidence=float(q['confidence'])
    weight[si]=interior(xy[si],cams[0].img_size)*confidence**2
    # Retain the frozen assignment for fusion; conflicted hands cannot provide stereo 3D.
   observations.append(xy);weights.append(weight)
  tri=np.zeros((2,21,3));cov=np.zeros((2,21,3,3));tw=np.zeros((2,21))
  for si,side in enumerate(('left','right')):
   agrees=all(side in m and m[side].get('detector_handedness_agrees',True) for m in matches)
   for j in range(21):
    rec={'pair':pair['id'],'side':side,'joint':j,'accepted':False}
    if not agrees:rec['reason']='missing_or_conflicted_identity'
    elif min(weights[0][si,j],weights[1][si,j])<=0:rec['reason']='edge_or_missing_2d'
    else:
     rec.update(triangulate(cams,R,t,[xy[si,j] for xy in observations]))
     if rec['accepted']:
      tri[si,j]=rec['point_cam3_m'];cov[si,j]=rec['covariance_cam3_m2'];tw[si,j]=min(weights[0][si,j],weights[1][si,j])
    tri_records.append(rec)
  for c,frame in enumerate(frames):
   context=a.out/'contexts'/(frame['id']+'.json');points=tri.copy();whiten=np.zeros_like(cov)
   if c:points=points@R.T+t
   for si,j in zip(*np.where(tw>0)):
    C=R@cov[si,j]@R.T if c else cov[si,j];e,v=np.linalg.eigh(C);whiten[si,j]=(v*(1/np.sqrt(e)))@v.T
   write(context,dict(pair=pair['id'],frame=frame['id'],matches=matches[c],hand_xy=observations[c].tolist(),hand_weights=weights[c].tolist(),tri=points.tolist(),whiten=whiten.tolist(),tri_weights=tw.tolist(),peer_calibration=frames[1-c]['calibration'],peer_R=(R.T if c else R).tolist(),peer_t=(-R.T@t if c else t).tolist(),timestamp_difference_seconds=pair['timestamp_difference_seconds']))
   sap=read(frame['predictions']['sapiens']);sap['keypoints'].append(dict(name='_v2_context',path=str(context.resolve())))
   target=a.out/'sapiens'/(frame['id']+'.json');write(target,sap)
   row=dict(id=frame['id'],camera=frame['camera'],image=frame['image'],calibration=frame['calibration'],predictions=dict(frame['predictions']))
   row['predictions']['stereo_context']=str(context.resolve())
   row['predictions']['wilor']=frame['frozen_files']['wilor'];row['predictions']['sapiens']=str(target.resolve())
   man=a.out/'manifests'/(frame['id']+'.json');write(man,dict(frames=[row]))
   settings=dict(cfg,pipeline_profile='stereo_guided_original_seam_v2_20261104',stereo_v2=True,gpu='',cached_wilor_focal_px=25000,manifest=str(man.resolve()),output=str((a.out/'full_cpu_v1'/frame['id']).resolve()))
   conf=a.out/'configs'/(frame['id']+'.json');write(conf,settings)
   records.append(dict(pair=pair['id'],id=frame['id'],camera=frame['camera'],config=str(conf.resolve()),row=row,original=frame['frozen_files'],context=str(context.resolve())))
 write(a.out/'inputs.json',dict(records=records,archive=str(a.archive.resolve()),calibration=manifest['calibration_source']))
 write(a.evidence/'triangulation.json',dict(pixel_sigma_assumption=8,metric_floor_m=.008,calibration_uncertainty_included=False,synchronization_uncertainty_included=False,points=tri_records))
 write(a.evidence/'input_hashes.json',dict(manifest_sha256=sha(a.archive/'manifest.json'),stereo_calibration_sha256=sha(manifest['calibration_source']),inputs=inputs))
 write(a.evidence/'settings.json',dict(profile='guided original seam plus stereo_v2',item7_implemented=False,raw_network_rerun=False,head_reference_weight=.1,observation_multiplier=.01,body_2d=1,sapiens_wrist_2d=1,wilor_wrist_2d_each=.5,wrist_3d_each=.5,hand_2d_each=.5,hand_3d_each=.25,head_visibility_weight=.03,pixel_scale=20,hand_min_weight_denominator=4,head_depth_smooth_m=.01,head_border_smooth_px=10,steps=cfg['fitting'],pair_timestamps=[dict(pair=p['id'],difference_seconds=p['timestamp_difference_seconds']) for p in manifest['pairs']]))
 print(json.dumps({'frames':len(records),'accepted_3d_by_pair':{p['id']:sum(q['accepted'] for q in tri_records if q['pair']==p['id']) for p in manifest['pairs']}}))
if __name__=='__main__':main()
