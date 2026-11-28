from pathlib import Path
import sys,numpy as np,torch,json
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(RUN.parent/'2026-11-27_video34_elbow_audit'))
import audit as a
from render_result import load_obj
torch.set_num_threads(1)
ref=a.original.w.ref
ref.OLD=ref.OUT=ref.compare.OUT=a.OUT
ref.PRIOR=a.PRIOR
ref.cameras=ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=a.cameras
model,_=ref.prev.ref.common.setup_model()
layer=model.layer['neutral'].eval()
cams=list(map(a.read,a.cameras()))
rows=a.read(a.OUT/'fit/0/prepared.json')['items'][:10]
pred=a.read(a.OUT/'predictions.json')
folders={'v5':ROOT/'outputs/video003_strict_v5_20261127','shift':ROOT/'outputs/video003_v5_nose_20261128','head':ROOT/'outputs/video003_v5_headonly_20261128'}
evidence={'shift':RUN.parent/'2026-11-28_video003_v5_nose/evidence','head':RUN.parent/'2026-11-28_video003_v5_headonly/evidence'}
out={};pooled={name:{g:[] for g in ('body','shoulder','elbow','wrist','hip','knee','ankle','hand_wrist')} for name in folders}
for row in rows:
 key=row['pair'];tfs=[np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in ('left','right')]
 target=a.elbow_observations(row,['L','R'],pred,key)
 valid=((target>=0)&(target<[1280,720])).all(2)
 frame={}
 for name,folder in folders.items():
  p=folder/'poses'/(key+('_v5.npz' if name=='v5' else '.npz'))
  pack=dict(np.load(p));v,j=a.pack_pose(layer,pack,pack['locked'])
  if name=='v5':
   rec=a.read(RUN.parent/'2026-11-27_video003_strict_v5/evidence'/(key+'_render.json'))
   obj=folder/'export'/rec['obj']
  else:
   rec=a.read(evidence[name]/(key+'.json'))
   obj=folder/'export'/(key+'.obj')
  assert a.sha(obj)==rec['obj_sha256']
  vo,fo=load_obj(obj)
  assert np.array_equal(fo,np.asarray(model.face)) and np.max(np.abs(vo-v))<1e-8
  uv=np.stack([a.projection(j@t[:3,:3].T+t[:3,3],c) for t,c in zip(tfs,cams)])
  groups={g:[] for g in pooled[name]}
  for pt in row['points']:
   label=pt['name'].split('_')[-1].lower()
   for e in range(2):
    if pt['valid'][e]:
     dist=float(np.linalg.norm(uv[e,pt['native_index']]-pt['xy'][e]))
     groups['body'].append(dist);groups[label].append(dist)
  for e in range(2):
   for i in range(2):
    if valid[e,i+2]:
     groups['hand_wrist'].append(float(np.linalg.norm(uv[e,20+i]-target[e,i+2])))
  for g,vals in groups.items():pooled[name][g].extend(vals)
  frame[name]={g:dict(rmse=float(np.sqrt(np.mean(np.square(vals)))) if vals else None,n=len(vals),errors=vals) for g,vals in groups.items()}
 out[key]=frame
summary={n:{g:dict(rmse=float(np.sqrt(np.mean(np.square(vals)))) if vals else None,n=len(vals)) for g,vals in groups.items()} for n,groups in pooled.items()}
assert summary['v5']['body']['n']==240 and abs(summary['v5']['body']['rmse']-10.150797915380956)<1e-5
for key,fr in out.items():
 for group in ('body','shoulder','elbow','wrist','hip','knee','ankle','hand_wrist'):
  assert np.allclose(fr['v5'][group]['errors'],fr['head'][group]['errors'],atol=1e-4)
a.write(RUN/'evidence/results.json',dict(summary=summary,frames=out,exit_status=0))
print(json.dumps(summary))
for key,fr in out.items(): print(key,*[round(fr[n]['body']['rmse'],2) for n in folders])
