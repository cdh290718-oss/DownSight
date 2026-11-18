import json,sys,copy,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('runs/2026-11-18_gen_head_camera_alignment/code').resolve()))
from export_assets import project
R=Path('runs/2026-11-18_gen_raw_calibration_four_models'); O=Path('outputs/gen_raw_calibration_four_models_20261118');a=json.load(open('outputs/gen_head_camera_alignment_20261118/head_camera_20pairs.json'));z=np.load('outputs/gen_head_camera_alignment_20261118/parsed_body_vio.npz');prev=json.load(open('outputs/gen_three_models_20261118/prepared.json'))['items'];orig=json.load(open('outputs/gen_original_comparison_20261118/prepared.json'))['items']; rows=[];truth={};errs={}
from scipy.spatial.transform import Rotation
vr=Rotation.from_quat(z['vio_quaternions_xyzw']).as_matrix();vp=z['vio_positions'];t=np.array(a['T_right_virtual_from_left_virtual']);cameras=a['cameras']; ds={v:cameras[v]['virtual_params_fx_fy_cx_cy_xi_alpha'] for v in cameras}
for x in a['pairs']:
 f=x['frame'];name=f'gen_{f:05d}';truth[name]={}
 for v in ('cam3','cam4'):
  T=np.array(cameras[v]['T_vio_from_virtual']);r=vr[f]@T[:3,:3];p=vp[f]+vr[f]@T[:3,3];xyz=(z['positions'][f]-p)@r;uv,valid=project(xyz,ds[v]);inside=valid&(uv[:,0]>=0)&(uv[:,0]<1280)&(uv[:,1]>=0)&(uv[:,1]<720);truth[name][v]={'xyz':xyz.tolist(),'uv':uv.tolist(),'valid':inside.tolist()}
for item in prev+orig:
 row=copy.deepcopy(item);pair=row['pair'];frame=int(pair.split('_')[1]);head=next(x for x in a['pairs'] if x['frame']==frame);row['anchor']={'position':head['head_virtual']['cam3']['position_m'],'rotation':head['head_virtual']['cam3']['R_schema_interpretation']};row['inputs']['views']['right']['camera']['left_to_view']=t.tolist()
 for p in row['points']:
  idx=p['native_index'];p['valid']=[truth[pair][v]['valid'][idx] for v in ('cam3','cam4')]
  for k,v in enumerate(('cam3','cam4')):
   if p['valid'][k]:errs.setdefault(row['condition'],[]).append({'pair':pair,'view':v,'joint':p['name'],'error_px':float(np.linalg.norm(np.array(p['xy'][k])-truth[pair][v]['uv'][idx]))})
 rows.append(row)
assert len(rows)==80
cfg=json.load(open(R/'protocol.json'));cfg['conditions']=['original','rw500','rw5000','rw20000'];cfg['scope']='MCAP calibration, DFM head position and unverified MCAP orientation candidate; same samples and DFM in-frame masks';(R/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');(O/'prepared.json').write_text(json.dumps({'items':rows,'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()}));(O/'reference.json').write_text(json.dumps(truth));(R/'evidence/2d.json').write_text(json.dumps({'records':errs,'head_orientation':'MCAP candidate, unverified SMPL-X axes','validity':'reference in frame, not visibility'},indent=2)+'\n');print('prepared',len(rows))
