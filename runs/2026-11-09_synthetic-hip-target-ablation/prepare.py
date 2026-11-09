"""Extract only the two GT hips; preserve all baseline observation and anchor bytes."""
import hashlib,json
from pathlib import Path
import numpy as np
from common import ROOT,RUN,OUT,OLD,read
from hip_targets import replace_targets
PREV=ROOT/'outputs/synthetic_head_robustness_20261108';PRUN=ROOT/'runs/2026-11-08_synthetic-head-robustness'
OUT.mkdir(exist_ok=False);(OUT/'anchors').mkdir();(OUT/'runtime/cache').mkdir(parents=True);(OUT/'runtime/matplotlib').mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
copied=[]
for relative in ['inputs.json','triangulated.json','anchors/oracle.json']:
 source=PREV/relative;dest=OUT/relative;dest.write_bytes(source.read_bytes());copied.append(dict(source=str(source.relative_to(ROOT)),destination=str(dest.relative_to(ROOT)),source_sha256=sha(source),destination_sha256=sha(dest),identical=source.read_bytes()==dest.read_bytes()))
source=OLD/'dataset/samples/000001/left_egosmplx65.npz';names=read(OLD/'dataset/joint_names_egosmplx65.json')['names'];assert names[1:3]==['L_Hip','R_Hip']
with np.load(source) as data:hips=data['joints_3d_camera_m'][[1,2]].copy()
override=dict(coordinate_frame='cam3 CV metres',targets=[dict(name=name,index=i,xyz=xyz.tolist()) for name,i,xyz in zip(['left_hip','right_hip'],[1,2],hips)],source_resource=str(source.relative_to(ROOT)),source_sha256=sha(source),scope='Only these two GT positions exported for fitting. All remaining GT joints, shape and pose excluded.')
(OUT/'hip_targets.json').write_text(json.dumps(override,indent=2,allow_nan=False)+'\n');(RUN/'evidence/hip_targets.json').write_text(json.dumps(override,indent=2,allow_nan=False)+'\n')
automatic=read(OUT/'triangulated.json')['points'];changed,intervention=replace_targets(automatic,override);same,control=replace_targets(automatic,None);assert same==automatic
assert (RUN/'settings.json').read_bytes()==(PRUN/'settings.json').read_bytes()
assert (RUN/'initialization.py').read_bytes()==(PRUN/'initialization.py').read_bytes()
q=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-09_synthetic-hip-target-ablation/prepare.py',copied_assets=copied,settings_sha256=sha(RUN/'settings.json'),initializer_sha256=sha(RUN/'initialization.py'),hip_override_sha256=sha(OUT/'hip_targets.json'),intervention=intervention,control_exact=True,settings_and_initializer_identical_to_previous=True)
(RUN/'evidence/preparation.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
