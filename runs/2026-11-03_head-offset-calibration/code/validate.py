"""Independent structural checks on Gen and synthetic outputs; no DFM access."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--gen',type=Path,required=True);p.add_argument('--synth',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();obs=np.load(a.observations,allow_pickle=False);cases=['zero']+[f'{axis}_{sign}_{mag}' for mag in (1,2,5,10) for axis in 'xyz' for sign in 'pm']+['long_zero']+[f'long_{axis}_{sign}_5' for axis in 'xyz' for sign in 'pm'];out=[]
for case in cases:
 d=a.gen/case;s=json.loads((d/'settings.json').read_text());test=np.load(d/'test.npz',allow_pickle=False);assert np.array_equal(test['frame_ids'],obs['frames'][10:]);assert s['anchor']=='hard' and not s['translation_prior_used'];expected=1500 if case.startswith('long_') else 500;assert s['steps']==expected;assert np.isfinite(test['joints_cam2']).all();err=float(np.linalg.norm(test['head_joint_cam2']-test['shared_head_center_cam2_m'],axis=-1).max());assert err<1e-6;out.append({'case':case,'hard_anchor_max_error_m':err,'test_frame_ids':test['frame_ids'].tolist(),'status':'passed'})
expected_s=[f'axes{ax}_deg{deg}_trial{trial}' for ax in (1,2) for deg in (0,10,20,40,60) for trial in (0,1)]+[f'long_axes{ax}_deg{deg}_trial0' for ax,deg in ((2,0),(1,40),(1,60),(2,40),(2,60))];synth=[]
for name in expected_s:
 log=a.synth/(name+'.txt');f=a.synth/(name+'.json');assert log.exists();text=log.read_text();assert 'CASE_EXIT ' in text
 if f.exists():
  d=json.loads(f.read_text());assert 'CASE_EXIT 0' in text and len(d['centers_m'])==7 and len(d['error_cm'])==7;assert d['head_camera_rotation_max_error_deg']<1e-3 and np.isfinite(d['centers_m']).all();synth.append({'case':name,'status':'completed','head_camera_rotation_max_error_deg':d['head_camera_rotation_max_error_deg']})
 else:synth.append({'case':name,'status':'failed','exit_line':[x for x in text.splitlines() if x.startswith('CASE_EXIT')][-1]})
with a.output.open('x') as f:json.dump({'gen_count':len(out),'gen':out,'synthetic':synth,'note':'A passed structural check does not establish optimization convergence or realism.'},f,indent=2,allow_nan=False);f.write('\n')
