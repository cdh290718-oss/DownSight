"""Validate fixed input identity, v2 semantics and completed output invariants."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from egosmplx_pipeline.io import read,write,load
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();data=read(a.inputs);checks=[]
for r in data['records']:
 cfg=read(r['config']);out=Path(cfg['output']);state=read(out/'run_state.json') if (out/'run_state.json').exists() else {};root=out/'frames'/r['id'];check=dict(frame=r['id'],complete=state.get('state')=='complete')
 if not check['complete']:checks.append(check);continue
 v=read(root/'validation.json');raw=load(root/'raw/params.npz');final=load(root/'fused/body_params.npz');recipe=read(root/'fused/recipe.json');original=read(r['original']['recipe'])['matches'];ctx=read(r['context'])
 check.update(profile=v['pipeline_profile'],reload=v['passed'],matches_unchanged=recipe['matches']==original,head_reference_weight=cfg['head_neck']['rigid_head_weight'],cpu=cfg['gpu']=='',fixed_shape_and_hand_parameters=all(np.array_equal(raw[k],final[k]) for k in ['betas','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']),fused_obj_error_m=v['obj_reload_max_abs_m'],local_surface_accepted=v.get('local_surface_accepted'))
 if recipe['matches']:
  check['masked_depth_m']=read(root/'fused/geometry_validation.json')['body_constraints']['body_minimum_z_m']
  check['wrist_floor_m']=read(root/'fused/geometry_validation.json')['body_constraints']['wrist_floor_m']
 assert check['profile']=='stereo_guided_original_seam_v2_20261104' and check['reload'] and check['matches_unchanged'] and check['fixed_shape_and_hand_parameters']
 assert all(x==0 for x in check.get('wrist_floor_m',[]))
 assert check.get('masked_depth_m',.02)>=.02-1e-6
 checks.append(check)
write(a.out,dict(frames=checks,all_complete=all(x['complete'] for x in checks),scope='fixed shape/fingers, original frozen matching, CPU route, no radial wrist floor, masked depth, reconstruction'))
print(json.dumps({'complete':sum(q['complete'] for q in checks),'total':len(checks)}))
