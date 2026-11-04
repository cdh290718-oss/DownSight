"""Held-out diagnostics after fits. DFM used only here as reference."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ref=np.load(a.reference,allow_pickle=False);obs=np.load(a.inputs/'baseline/observations.npz',allow_pickle=False);frames=obs['frames'][10:];index={int(f):i for i,f in enumerate(ref['frames'])};ix=[index[int(f)] for f in frames];cal=json.loads((a.inputs/'baseline/camera_calibration.json').read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);gt=ref['cam2_body_xyz'][ix];valid=ref['cam2_body_valid'][ix]
def project(x,ds):
 fx,fy,cx,cy,xi,alpha=ds;x,y,z=x.T;d=np.sqrt(x*x+y*y+z*z);z1=xi*d+z;den=alpha*np.sqrt(x*x+y*y+z1*z1)+(1-alpha)*z1;return np.stack((fx*x/den+cx,fy*y/den+cy),axis=-1)
def stats(v):
 x=np.asarray(v,float);return {'count':len(x),'median_cm':float(np.median(x)*100) if len(x) else None,'p90_cm':float(np.percentile(x,90)*100) if len(x) else None,'per_frame_cm':(x*100).tolist()}
variants=['baseline','published_gate','recomputed_gate','wrist_only','published_gate_wrist_only','finger_2d_only','finger_3d_only','published_gate_cross_hand'];out={'note':'DFM is a post-fit model reference, not independent truth. Camera calibration image-estimated; head location is a DFM-derived simulated fixed calibration.','bad_frame_rule':'Either wrist in baseline, published gating, wrist-only, or cross-hand gating at 500 or 1500 steps has Euclidean difference >= 5 cm in a DFM-valid test frame.','frames':frames.tolist(),'variants':{},'per_frame':[]}
fits={};inps={}
for mode in variants:
 inps[mode]=np.load(a.inputs/mode/'observations.npz',allow_pickle=False)
 for steps in (500,1500):
  name=f'{mode}_{steps}';fit=np.load(a.results/name/'test.npz',allow_pickle=False);assert np.array_equal(fit['frame_ids'],frames);fits[name]=fit;row={}
  for side,si,gi in [('left',12,20),('right',13,21)]:
   err=fit['joints_cam2'][:,si]-gt[:,gi];u=gt[:,gi]/np.maximum(np.linalg.norm(gt[:,gi],axis=1,keepdims=True),1e-9);mask=valid[:,gi]&np.isfinite(err).all(1);row[side]={'euclidean':stats(np.linalg.norm(err,axis=1)[mask]),'along_view_absolute':stats(np.abs((err*u).sum(1))[mask]),'valid_frames':frames[mask].tolist()}
  out['variants'][name]=row
previous=np.load(a.previous,allow_pickle=False);out['baseline_previous_comparison']={'frame_ids_equal':bool(np.array_equal(previous['frame_ids'],frames)),'max_joint_difference_m':float(np.max(np.linalg.norm(previous['joints_cam2'][:,[12,13]]-fits['baseline_500']['joints_cam2'][:,[12,13]],axis=-1)))}
names=obs['names'].tolist();fingeridx={'left':[28,32,36,40,44],'right':[48,52,56,60,64]}
for i,f in enumerate(frames):
 fr={'frame':int(f),'wrists':{}}
 for side,si,gi in [('left',12,20),('right',13,21)]:
  oi=names.index(side+'_wrist');true=gt[i,gi];point=obs['tri'][i+10,oi];tf=bool(obs['tri_valid'][i+10,oi]);refgood=bool(valid[i,gi]);raw=obs['xy'][i+10,:,oi];gtcam3=(true-t)@R;truthuv=[project(true[None,:],cal['cam2_ds'])[0],project(gtcam3[None,:],cal['cam3_ds'])[0]]
  v={'reference_valid':refgood,'observation_stereo_valid':tf,'observation_vs_dfm_cm':float(np.linalg.norm(point-true)*100) if tf and refgood else None,'observation_pixel_vs_dfm_px':[float(np.linalg.norm(raw[c]-truthuv[c])) if refgood else None for c in range(2)],'raw_uv':raw.tolist(),'reference_projected_uv':[x.tolist() for x in truthuv] if refgood else None,'scores':obs['score'][i+10,:,oi].tolist(),'fits':{}}
  for mode in variants:
   for steps in (500,1500):
    name=f'{mode}_{steps}';j=fits[name]['joints_cam2'][i,si];j3=(j-t)@R;v['fits'][name]={'vs_dfm_cm':float(np.linalg.norm(j-true)*100) if refgood else None,'vs_observation_cm':float(np.linalg.norm(j-point)*100) if tf else None,'projected_uv':[project(j[None,:],cal['cam2_ds'])[0].tolist(),project(j3[None,:],cal['cam3_ds'])[0].tolist()],'stereo_used':bool(inps[mode]['tri_valid'][i+10,oi]),'cam2_used':bool(inps[mode]['valid'][i+10,0,oi])}
  fingers=[names.index(side+'_'+x) for x in ('thumb4','forefinger4','middle_finger4','ring_finger4','pinky_finger4')];finger_tri=obs['tri_valid'][i+10,fingers];fitted=fits['baseline_500']['joints_cam2'][i,fingeridx[side]];dist=np.linalg.norm(fitted-obs['tri'][i+10,fingers],axis=1)*100;v['baseline_fingertip_stereo_count']=int(finger_tri.sum());v['baseline_fingertip_fit_to_observation_cm']=dist[finger_tri].tolist();v['bad_frame']=bool(refgood and max(v['fits'][f'{m}_{st}']['vs_dfm_cm'] for m in ('baseline','published_gate','wrist_only','published_gate_cross_hand') for st in (500,1500))>=5)
  fr['wrists'][side]=v
 out['per_frame'].append(fr)
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
