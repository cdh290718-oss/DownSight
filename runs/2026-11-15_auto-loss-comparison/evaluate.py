"""Read GT body only after all fits have completed; publish derived metrics only."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_auto_loss_comparison_20261115';ARCH=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';sys.path.insert(0,str(ROOT/'runs/2026-11-04_session-hand6-v2/code'));sys.path.insert(0,str(ARCH))
from prepare import Camera
from evaluation_common import align
read=lambda p:json.loads(Path(p).read_text())
records=[json.loads(s) for s in (OUT/'execution.jsonl').read_text().splitlines()]; assert len(records)==12,(len(records),12)
gt=np.load(ARCH/'body_gt_official.npz'); rig=read(ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json');R=np.array(rig['rig_rotation']);t=np.array(rig['rig_translation_m']);cams=[Camera(ARCH/'body_calibration_fisheye'/('camera_'+x+'.json')) for x in ['left','right']]; rows=[]
for rec in records:
 row=dict(rec); dest=OUT/rec['pair']/'auto'; method=rec['method']; obs=read(dest/'observations.json');row.update(valid_views=sum(sum(p['valid']) for p in obs['points']),three_d_points=sum(p['accepted'] for p in obs['points']))
 if rec['exit_status']!=0:rows.append(row);continue
 params=np.load(dest/method/'final.npz'); joints=params['smplx_joint_cam']; idx=[p['index'] for p in obs['points']]; pred=joints[idx].astype(float); pairidx=(int(rec['pair'][-2:])-1)*2; target=gt['joints_m'][pairidx].astype(float); pa=np.linalg.norm(align(pred,target)-target,axis=1)*1000; absolute=np.linalg.norm(pred-target,axis=1)*1000;opt=read(dest/method/'optimization.json'); views=[]
 clean=read(ROOT/'outputs/unrealego_gt2d_ideal_20261115'/rec['pair']/'observations_local.json')
 for v in range(2):
  projected=cams[v].world2camera(pred if v==0 else pred@R.T+t); truth=np.array(clean['views'][v])[:,:2]; d=np.linalg.norm(projected-truth,axis=1); inp=[np.linalg.norm(projected[i]-p['xy'][v]) for i,p in enumerate(obs['points']) if p['valid'][v]];views.append({'clean_gt15_mean_px':float(d.mean()),'input_mean_px':float(np.mean(inp)) if inp else None,'input_count':len(inp),'clean_per_joint_px':d.tolist()})
 row.update(pa15_mm=float(pa.mean()),absolute15_mm=float(absolute.mean()),pa_per_joint_mm=pa.tolist(),absolute_per_joint_mm=absolute.tolist(),views=views,shape_rank=opt['initialization']['retained_rank'],convergence=opt['convergence'],selected_step=opt['best_step']);rows.append(row)
rows.sort(key=lambda r:(r['pair'],r['method']));(RUN/'evidence/results.json').write_text(json.dumps({'exit_status':0,'records':rows},indent=2,allow_nan=False)+'\n');print('evaluated',sum(r['exit_status']==0 for r in rows),'/',len(rows))
