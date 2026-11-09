"""Export existing body target GT coordinates only, with no change to initialization observations."""
import hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,OLD,read
from body_targets import replace_targets
PREV=ROOT/'outputs/synthetic_head_robustness_20261108';PRUN=ROOT/'runs/2026-11-08_synthetic-head-robustness'
OUT.mkdir(exist_ok=False);(OUT/'anchors').mkdir();(OUT/'target_overrides').mkdir();(OUT/'runtime/cache').mkdir(parents=True);(OUT/'runtime/matplotlib').mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
copied=[]
for relative in ['inputs.json','triangulated.json','anchors/oracle.json']:
 source=PREV/relative;dest=OUT/relative;dest.write_bytes(source.read_bytes());copied.append(dict(source=str(source.relative_to(ROOT)),destination=str(dest.relative_to(ROOT)),source_sha256=sha(source),destination_sha256=sha(dest),identical=source.read_bytes()==dest.read_bytes()))
automatic=read(OUT/'triangulated.json')['points'];source=OLD/'dataset/samples/000001/left_egosmplx65.npz';names=read(OLD/'dataset/joint_names_egosmplx65.json')['names']
with np.load(source) as data:truth=data['joints_3d_camera_m'].copy()
records=[]
for scope in ['hips','body']:
 selected=[p for p in automatic if p['source']=='sapiens' and p['accepted'] and (scope=='body' or p['name'] in ['left_hip','right_hip'])]
 assert len(selected)==(12 if scope=='body' else 2)
 override=dict(coordinate_frame='cam3 CV metres',scope=scope,targets=[dict(name=p['name'],index=p['index'],model_name=names[p['index']],xyz=truth[p['index']].tolist()) for p in selected],source_resource=str(source.relative_to(ROOT)),source_sha256=sha(source),usage='Only listed GT coordinates exposed to fitting loss; no shape/pose GT, initialization or information-subspace changes.')
 path=OUT/'target_overrides'/(scope+'.json');path.write_text(json.dumps(override,indent=2,allow_nan=False)+'\n');(RUN/'evidence'/(scope+'_targets.json')).write_bytes(path.read_bytes())
 points,intervention=replace_targets(automatic,override);records.append(dict(scope=scope,target_override_sha256=sha(path),intervention=intervention))
assert replace_targets(automatic,None)[0]==automatic
for name in ['settings.json','initialization.py']:assert (RUN/name).read_bytes()==(PRUN/name).read_bytes()
q=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-09_synthetic-body-target-oracle/prepare.py',copied_assets=copied,settings_sha256=sha(RUN/'settings.json'),initializer_sha256=sha(RUN/'initialization.py'),overrides=records,control_exact=True,settings_and_initializer_identical_to_previous=True,all_body_definition='12 existing Sapiens body targets. WiLoR 10 points remain automatic; no pelvis, toe/heel or new targets.')
(RUN/'evidence/preparation.json').write_text(json.dumps(q,indent=2)+'\n');print('Prepared unchanged inputs, hips 2 targets, body 12 targets')
