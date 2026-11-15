import hashlib,json
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;out=r/'outputs/unrealego_auto_loss_comparison_20261115'; execution=[json.loads(x) for x in (out/'execution.jsonl').read_text().splitlines()];assert len(execution)==12
checks=[]
for pair in [f'body{i:02d}' for i in range(1,5)]:
 methods=['mixed','2d']+(['3d'] if pair!='body04' else []); data=[json.loads((out/pair/'auto'/method/'optimization.json').read_text()) for method in methods]; bases=[x['initialization']['basis'] for x in data];assert all(x==bases[0] for x in bases);roots=[x['initial_global'] for x in data];assert all(x==roots[0] for x in roots);checks.append({'pair':pair,'same_shape_basis':True,'same_root_initialization':True,'shape_rank':data[0]['initialization']['retained_rank']})
a=np.load(out/'body04/auto/mixed/final.npz');b=np.load(out/'body04/auto/2d/final.npz');diff=float(np.max(np.abs(a['smplx_joint_cam']-b['smplx_joint_cam'])));assert diff<1e-6
records=[]
for row in execution:
 d=dict(row)
 if row['status']=='ok':
  text=(out/row['pair']/'auto'/(row['method']+'.log')).read_text(errors='replace'); progress=[x for x in text.splitlines() if x.startswith('D ') or x.startswith('DONE D ')]; assert progress[-1].startswith('DONE D ');d['log_excerpt']=progress[-3:]
 records.append(d)
(p/'evidence/execution.json').write_text(json.dumps({'command':'<BODY_PYTHON> runs/2026-11-15_auto-loss-comparison/run.py','records':records},indent=2)+'\n')
(p/'evidence/verification.json').write_text(json.dumps({'exit_status':0,'conditions':checks,'body04_mixed_vs_2d_max_joint_m':diff,'notes':'Same initialization and body shape basis verified; body04 has no 3D branch. Raw checkpoints not published.'},indent=2)+'\n');print('same initialization and basis verified, body04 max diff',diff)
