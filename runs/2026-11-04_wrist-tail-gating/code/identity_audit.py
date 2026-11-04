"""Audit cross-hand ambiguity using detections and triangulations, without DFM."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,required=True);p.add_argument('--published',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.observations,allow_pickle=False);names=d['names'].tolist();pub={(int(x['frame']),x['joint']):x for x in json.loads(a.published.read_text())};out=[]
for i,f in enumerate(d['frames']):
 for finger in ('thumb4','forefinger4','middle_finger4','ring_finger4','pinky_finger4'):
  il=names.index('left_'+finger);ir=names.index('right_'+finger);pair=np.linalg.norm(d['xy'][i,:,il]-d['xy'][i,:,ir],axis=1);tri=bool(d['tri_valid'][i,il] and d['tri_valid'][i,ir]);dis=float(np.linalg.norm(d['tri'][i,il]-d['tri'][i,ir])*100) if tri else None;sides={}
  for side,k in [('left',il),('right',ir)]:
   same=names.index(side+'_wrist');other=names.index(('right' if side=='left' else 'left')+'_wrist');ok=bool(d['tri_valid'][i,k] and d['tri_valid'][i,same] and d['tri_valid'][i,other]);ds=float(np.linalg.norm(d['tri'][i,k]-d['tri'][i,same])*100) if ok else None;do=float(np.linalg.norm(d['tri'][i,k]-d['tri'][i,other])*100) if ok else None;r=pub[(int(f),side+'_'+finger)];sides[side]={'stereo_available':bool(d['tri_valid'][i,k]),'same_wrist_distance_cm':ds,'opposite_wrist_distance_cm':do,'published_gate_status':r['status'],'published_gate_reason':r['reason'],'cam2_valid':bool(d['valid'][i,0,k]),'cam3_valid':bool(d['valid'][i,1,k])}
  out.append({'frame':int(f),'finger':finger,'left_right_uv_separation_px':pair.tolist(),'left_right_3d_separation_cm':dis,'sides':sides})
with a.output.open('x') as f:json.dump({'note':'Detector identity diagnostics; proximity is evidence of ambiguity, not confirmed ground truth or an automatic swap.','points':out},f,indent=2,allow_nan=False);f.write('\n')
