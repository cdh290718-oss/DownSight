import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
run=Path(__file__).resolve().parent;root=run.parents[1];out=root/'outputs/gen_head_position_only_20261118';prior=root/'outputs/gen_raw_calibration_four_models_20261118';a=json.load(open(root/'outputs/gen_head_camera_alignment_20261118/head_camera_20pairs.json'));z=np.load(root/'outputs/gen_head_camera_alignment_20261118/parsed_body_vio.npz');old=json.load(open(root/'outputs/gen_three_models_20261118/prepared.json'))['items'];old={x['pair']:x for x in old if x['condition']=='rw500'};rows=json.load(open(prior/'prepared.json'))['items'];frames=list(z['frames']);T=np.array(a['cameras']['cam3']['T_vio_from_virtual']);rot=R.from_quat(z['vio_quaternions_xyzw']).as_matrix()
for x in rows:
 f=int(x['pair'].split('_')[1]);k=frames.index(f);new_world=rot[f]@T[:3,:3];old_world=z['cam3_R_world_from_virtual'][k];x['anchor']['rotation']=(new_world.T@old_world@np.array(old[x['pair']]['anchor']['rotation'])).tolist()
cfg=json.load(open(run/'protocol.json'));cfg['head_rotation_weight']=0.;cfg['scope']='Head position only diagnostic: MCAP bone display rotations rejected; network orientation used only for initialization transformed into new camera frame';cfg['baseline_commit']='a71e21a';
for key in ['intervention','secondary','fixed_mixed_route','statistics']:cfg.pop(key,None)
(run/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n');(out/'prepared.json').write_text(json.dumps({'items':rows,'protocol_sha256':hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()}));(out/'reference.json').write_bytes((prior/'reference.json').read_bytes());(run/'evidence/2d.json').write_bytes((root/'runs/2026-11-18_gen_raw_calibration_four_models/evidence/2d.json').read_bytes())
print('Prepared',len(rows),'fits; same observations/reference; head rotation loss disabled')
