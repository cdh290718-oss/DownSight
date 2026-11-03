"""Describe cloud-to-body distances independently of any DFM annotations."""
import os,sys,argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
repo=Path(os.environ['EGOSMPLX_BACKEND']);source=Path(os.environ['DOUBLE_SPHERE_SOURCE']);sys.path.insert(0,str(source));os.environ['EGOSMPLX_REPO_ROOT']=str(repo);os.environ['EGOSMPLX_CONFIG']=str(repo/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py')
from egosmplx_pipeline.bootstrap import setup
smpl,_=setup(repo)
p=argparse.ArgumentParser();p.add_argument('--cloud',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cloud=np.load(a.cloud,allow_pickle=False);selected=json.loads((a.results/'long_selection.json').read_text());lbs=smpl.layer['neutral'].lbs_weights.detach().cpu().numpy();ids=np.flatnonzero(np.isin(lbs.argmax(1),[3,6,9,12,13,14,16,17,18,19]));
if len(ids)>3000:ids=ids[np.linspace(0,len(ids)-1,3000,dtype=int)]
result={}
for cond in ('baseline','dense','baseline_long','dense_long'):
 seeds=selected[cond[:-5]]['seeds'] if cond.endswith('_long') else range(7)
 for seed in seeds:
  entry={}
  for split in ('train','test'):
   pred=np.load(a.results/f'{cond}_{seed}'/(split+'.npz'),allow_pickle=False);offset=0 if split=='train' else 10;fr=[]
   for i,f in enumerate(pred['frame_ids']):
    j=offset+i;pts=cloud['points_cam2'][j,:cloud['counts'][j]];dist=cKDTree(pred['vertices_cam2'][i,ids]).query(pts)[0]
    fr.append({'frame':int(f),'sample_count':len(dist),'median_cm':float(np.median(dist)*100),'p90_cm':float(np.percentile(dist,90)*100)})
   entry[split]=fr
  result[f'{cond}_{seed}']=entry
with a.output.open('x') as f:json.dump({'note':'Nearest model vertex distance to independently predicted clothing/upper-arm stereo points; neither an accuracy metric nor independent 3D truth.', 'vertex_count':len(ids),'results':result},f,indent=2);f.write('\n')
