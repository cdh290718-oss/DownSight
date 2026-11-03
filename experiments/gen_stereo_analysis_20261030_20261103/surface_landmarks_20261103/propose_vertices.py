"""Create unverified elbow surface vertex candidates from a neutral SMPL-X mesh."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();m=np.load(a.model,allow_pickle=True);v=m['v_template'];j=m['J_regressor']@v;result={}
for side,i in [('left',18),('right',19)]:
 c=j[i];mask=(abs(v[:,0]-c[0])<.012)&(abs(v[:,1]-c[1])<.025)&(np.linalg.norm(v-c,axis=1)<.09);ids=np.flatnonzero(mask);assert len(ids)>0
 for label,vid in [('olecranon',ids[np.argmin(v[ids,2])]),('cubital_fossa',ids[np.argmax(v[ids,2])])]:
  result[side+'_'+label]=dict(vertex_id=int(vid),neutral_xyz_m=v[vid].tolist(),joint_xyz_m=c.tolist(),neutral_distance_to_joint_m=float(np.linalg.norm(v[vid]-c)),candidate_count=int(len(ids)),status='unverified geometric proposal; not anatomical correspondence')
out=dict(model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),model_vertex_count=len(v),method='neutral template elbow joint 18/19; |dx| < .012m, |dy| < .025m, radius < .09m; minimum z as posterior and maximum z as anterior. This is a coordinate heuristic, not manually validated.',proposals=result)
a.out.write_text(json.dumps(out,indent=2)+'\n')
