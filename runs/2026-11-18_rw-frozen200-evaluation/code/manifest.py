import hashlib,json
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];root=r.parents[1];out=root/'outputs/rw_test_completion_20261117';d=json.loads((out/'extraction.json').read_text());assert d['error'] is None and d['found']==d['requested']==200 and d['complete_pairs']==100
p=root.parent/'EgoSMPLX/data/UnrealEgo_RW_Test/prepared/samples.npz';z=np.load(p,allow_pickle=False);xy=z['keypoints2d'];seq=z['source_sequence'];frame=z['frame_index'];side=z['camera_side'];mapping=[m for m in json.loads((root/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json').read_text())['joint_mapping'] if m['rw_index'] is not None];rows=[]
for x in d['records']:
 i=x['source_index'];assert str(seq[i])==x['sequence'] and int(frame[i])==x['frame'] and str(side[i])==x['view'];w,h=x['size'];points=np.zeros((308,2),np.float32);mask=np.zeros(308,np.int32)
 for m in mapping:
  pt=xy[i,m['rw_index'],:2];assert np.isfinite(pt).all();j=m['sapiens308_id'];points[j]=pt;mask[j]=int(0<=pt[0]<w and 0<=pt[1]<h)
 x['id']=Path(x['image']).stem;x['split']='test';x['keypoints308_xy']=points.tolist();x['keypoints308_supervision_mask']=mask.tolist();rows.append(x)
rows.sort(key=lambda a:a['id']);(out/'test_manifest.json').open('x').write(json.dumps({'records':rows},indent=2)+'\n');print('manifest',len(rows),flush=True)
