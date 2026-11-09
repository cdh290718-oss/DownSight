"""Extract only permitted initialization factors; all fits use identical observations."""
import hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,OLD,read
OUT.mkdir(exist_ok=False)
for d in ['anchors','seeds','runtime/cache','runtime/matplotlib']:(OUT/d).mkdir(parents=True,exist_ok=True)
previous=ROOT/'outputs/synthetic_body_target_oracle_20261109'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assets=[]
for rel in ['inputs.json','triangulated.json','anchors/oracle.json']:
 src=previous/rel;dest=OUT/rel;dest.write_bytes(src.read_bytes());assets.append(dict(source=str(src.relative_to(ROOT)),destination=str(dest.relative_to(ROOT)),sha256=sha(src),identical=src.read_bytes()==dest.read_bytes()))
source=OLD/'fits/left/frames/left/raw/params.npz'
with np.load(source) as z:
 shape=z['betas'].reshape(10).tolist();body=z['body_pose'].reshape(21,3).copy();body[[9,10]]=0
for name,data in [('shape',shape),('pose',body.tolist())]:
 (OUT/'seeds'/(name+'.json')).write_text(json.dumps(dict(values=data,source_sha256=sha(source),usage='Initialization factor only; foot local rotations set to zero for every case.'),indent=2)+'\n')
paths=read(OLD/'configs/left.json')['paths'];(OUT/'model_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
q=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-09_initialization-ablation/prepare.py',copied_assets=assets,network_source=str(source.relative_to(ROOT)),network_source_sha256=sha(source),seed_sha256={n:sha(OUT/'seeds'/(n+'.json')) for n in ['shape','pose']},scope='D reads neither seed; B pose only; C shape only; A both. No GT body coordinates/shape/pose extracted. Body and hand targets automatic. Network global rotation/translation/hands/face never extracted.',baseline='6d5994401b33450ea3343d20f44f2cc54e9ac5d6')
(RUN/'evidence/preparation.json').write_text(json.dumps(q,indent=2)+'\n');print('prepared')
