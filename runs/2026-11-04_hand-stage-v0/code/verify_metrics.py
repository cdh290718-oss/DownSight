"""Independently preserve per-frame hand error evidence and check summary statistics."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=np.load(a.reference,allow_pickle=False);metrics=json.loads(a.metrics.read_text());idx={int(f):i for i,f in enumerate(r['frames'])};old=set(range(0,5131,270));out={}
def palm(x):
 a=x[5]-x[0];b=x[17]-x[0];a=a/np.linalg.norm(a);z=np.cross(a,b);z=z/np.linalg.norm(z);return np.stack([a,np.cross(z,a),z],axis=-1)
for mode in ('v0','wilor','two_stage'):
 rows=[]
 for path in sorted((a.fits/mode).glob('batch_*.npz')):
  with np.load(path,allow_pickle=False) as data:
   for frame,joints in zip(data['frame_ids'],data['joints_cam2']):
    i=idx[int(frame)];item={'frame':int(frame),'hands':{}}
    for side,si,wi,start in [('left',0,12,25),('right',1,13,45)]:
     pred=joints[[wi]+list(range(start,start+20))].astype(np.float64);truth=r['cam2_hand_xyz'][i,si];distance=np.linalg.norm((pred-pred[:1])-(truth-truth[:1]),axis=1)*100;P=palm(pred);Q=palm(truth);angle=float(np.degrees(np.arccos(np.clip((np.sum(P*Q)-1)/2,-1,1))));item['hands'][side]={'relative_point_distances_cm':distance.tolist(),'mean_relative_21_cm':float(distance.mean()),'palm_angle_deg':angle}
    rows.append(item)
 rows.sort(key=lambda x:x['frame']);assert len(rows)==179;out[mode]=rows
 for side in ('left','right'):
  for subset in ('all','new160'):
   values=[x['hands'][side] for x in rows if subset=='all' or x['frame'] not in old];ref=metrics['variants'][mode]['hand'][side][subset]
   for key,metric in [('mean_relative_21_cm','mean_relative_21_point_error_cm'),('palm_angle_deg','palm_rotation_error_deg')]:
    arr=[x[key] for x in values];assert abs(np.median(arr)-ref[metric+'_median']['value'])<1e-9;assert abs(np.percentile(arr,90)-ref[metric+'_p90']['value'])<1e-9
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
