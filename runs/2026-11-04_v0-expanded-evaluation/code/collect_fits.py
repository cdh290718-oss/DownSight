"""Validate completed partitions and link local fit outputs for reference-only evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result={}
for mode in ('v0','wilor'):
 target=a.fits/mode;target.mkdir(exist_ok=False);paths=[]
 for part in ('first60','last120','tail60'):
  paths+=sorted((a.fits/f'{mode}_{part}').glob('batch_*.npz'))
 by_batch={}
 for path in paths:
  batch=int(path.stem.split('_')[1]);assert batch not in by_batch,(mode,batch);by_batch[batch]=path
 assert sorted(by_batch)==list(range(0,180,10)),(mode,sorted(by_batch))
 seen=[];maxerr=0.;files=[]
 for batch,path in sorted(by_batch.items()):
  with np.load(path,allow_pickle=False) as d:
   frame=d['frame_ids'];j=d['joints_cam2'];assert np.isfinite(j).all();assert all(batch<=int(f)//30<batch+10 for f in frame);seen.extend(frame.tolist());maxerr=max(maxerr,float(np.linalg.norm(d['head_joint_cam2']-d['shared_head_center_cam2_m'],axis=1).max()))
  (target/path.name).symlink_to(path.resolve());files.append({'batch':batch,'frames':frame.tolist(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 assert len(seen)==len(set(seen))==179 and set(seen)==set(range(0,5400,30))-{120}
 result[mode]={'batches':len(paths),'frames_predicted':len(seen),'missing_frames':[120],'max_head_constraint_error_m':maxerr,'batches_detail':files}
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
