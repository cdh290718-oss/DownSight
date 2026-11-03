"""Compare triangulations to DFM body only as an unvalidated reference."""
import argparse,json,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();dfm=np.load(a.reference);r=json.loads((a.results/'decisions.json').read_text());refidx={int(f):i for i,f in enumerate(dfm['frames'])};lookup={(x['frame'],x['name']):x for x in r};rows=[]
for q in r:
 side=q['name'].split('_')[0];part=q['name'].split('_')[-1]
 if side not in ('left','right') or part not in ('elbow','hip') or not q['stereo']:continue
 j={('left','elbow'):18,('right','elbow'):19,('left','hip'):1,('right','hip'):2}[side,part];k=refidx[q['frame']]
 if not dfm['cam2_body_valid'][k,j]:continue
 ref=dfm['cam2_body_xyz'][k,j];x=np.array(q['xyz_cam2_m']);rows.append(dict(frame=q['frame'],name=q['name'],reference_minus_observed_range_cm=float(100*(np.linalg.norm(ref)-np.linalg.norm(x))),euclidean_cm=float(100*np.linalg.norm(x-ref))))
with (a.out/'body_reference_differences.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['frame','name','reference_minus_observed_range_cm','euclidean_cm'],lineterminator='\n');w.writeheader();w.writerows(rows)
# Point proximity assesses dependence risk, not actual error correlations.
near=[]
for frame in sorted({q['frame'] for q in r}):
 for side in ('left','right'):
  e=lookup[frame,side+'_elbow']
  for part in ('olecranon','cubital_fossa'):
   m=lookup[frame,side+'_'+part]
   for c in range(2):
    if e['eligible'][c] and m['eligible'][c]:near.append(dict(frame=frame,side=side,marker=part,camera=c+2,distance_px=float(np.linalg.norm(np.array(e['xy'][c])-np.array(m['xy'][c])))))
with (a.out/'marker_elbow_proximity.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=['frame','side','marker','camera','distance_px'],lineterminator='\n');w.writeheader();w.writerows(near)
print('DFM_body_pairs',len(rows),'proximity_pairs',len(near))
