"""Use the archived head-position-only Golden D objective and unchanged candidate implementations."""
import sys
import shutil
import numpy as np
import torch
import cv2
from paths import *
sys.path.insert(0,str(ROOT/'runs/2026-11-23_fullbody_joint_candidates'))
import search as joint
import candidates as wrist
prev=joint.prev
base,ref=prev.base,prev.ref
SHARDS=8
def shard(n):return FIT/str(n)

def prepare():
    z=np.load(SOURCE);seq,frames,side,xy,heads=[z[k] for k in ('source_sequence','frame_index','camera_side','keypoints2d','keypoints3d_fit')]
    records=read(OUT/'manifest.json');pred=read(OUT/'predictions.json');images=read(OUT/'images.json')
    assert sha(OUT/'manifest.json')==read(RUN/'evidence/sample.json')['manifest_sha256']
    source=read(base.GOLD/'evidence/preparation.json')
    assert sha(SOURCE)==source['dataset_sha256']
    rigs={r['session']:r for r in source['rigs']}
    assert {(r['calibration_sequence'],r['calibration_frame']) for r in rigs.values()}.isdisjoint((s['sequence'],s['frame']) for s in records)
    model,camera=ref.common.setup_model()
    layer=model.layer['neutral'];rest=(layer.J_regressor@layer.v_template).detach().numpy()
    paths=ref.camera_files();cams=[camera(p) for p in paths];adapters=[read(p)['input_adapter'] for p in paths]
    cfg=base.OUT/'protocol.json'
    info=read(base.RUN/'evidence/inputs.json')
    assert all(sha(p)==info['cameras'][p.name] for p in paths)
    assert sha(cfg)==info['protocol_sha256']
    assert sha(base.OUT/'prepared.json')==info['solver_input_sha256']
    template=read(base.OUT/'prepared.json')['items'][0]['points'];mapping=prev.mapping()
    inputs=[[] for _ in range(SHARDS)];fallback=[];total=0
    for n,record in enumerate(records):
        pair=record['pair'];li,ri=record['indices'];s=record['sequence']
        assert all(seq[i]==s and frames[i]==record['frame'] and side[i]==v for i,v in zip((li,ri),('left','right')))
        T=rigs[s.rsplit('/',1)[0]]['left_to_right']
        views={'left':{'camera':{'left_to_view':np.eye(4).tolist()}},'right':{'camera':{'left_to_view':T}}}
        points=[{k:p[k] for k in ('index','name','native_index','accepted','group')} for p in template]
        for p in points:p.update(xy=[],valid=[])
        for i,v,ad in zip((li,ri),('left','right'),adapters):
            name=pair+'_'+v+'.png'
            assert pred[name]['image_sha256']==images[name]
            for p,j,raw_idx in zip(points,mapping,ref.common.OBS):
                coord=np.asarray(pred[name]['xy'][j])
                gtxy=xy[i,raw_idx,:2]
                assert np.isfinite(gtxy).all()
                valid=bool(((gtxy>=0)&(gtxy<872)).all())
                p['xy'].append((coord*ad['scale']+ad['offset']).tolist());p['valid'].append(valid)
                total+=valid
        row=dict(pair=pair,condition='baseline',head_position=heads[li,0].tolist(),points=points,inputs={'views':views})
        seed,reason=base.trial.body_seed(row,cams)
        if reason:
            seed=base.shoulder_hip_recovery(row,cams,rest);fallback.append(dict(pair=pair,reason=reason))
        row['initial_body_rotation']=seed.tolist();inputs[n%SHARDS].append(row)
    for n,batch in enumerate(inputs):
        dest=shard(n);dest.mkdir(exist_ok=True)
        write(dest/'prepared.json',dict(items=batch))
        shutil.copyfile(cfg,dest/'protocol.json')
    write(RUN/'evidence/inputs.json',dict(pairs=len(records),valid_2d=total,fallback=fallback,cameras=info['cameras'],
          protocol_sha256=sha(cfg),rig_source_sha256=sha(base.GOLD/'evidence/preparation.json'),
          shards={str(n):sha(shard(n)/'prepared.json') for n in range(SHARDS)},input_image_hashes_sha256=sha(OUT/'images.json'),exit_status=0))

def fit(n):
    base.fit.RUN=base.fit.OUT=shard(n)
    base.fit.Problem=base.trial.TorsoOnly
    base.fit.camera_files=ref.camera_files
    sys.argv=[__file__,'--threads','1']
    base.fit.main()

def candidates(n):
    folder=shard(n); cfg=read(folder/'protocol.json')
    base.fit.camera_files=ref.camera_files
    model,camera=ref.common.setup_model()
    adapters=[read(p)['input_adapter'] for p in ref.camera_files()]
    pack=dict(np.load(folder/'budget_20000.npz'))
    done=read(folder/'fit_complete.json')
    assert done['steps']==20000 and done['exit_status']==0
    rows=read(folder/'prepared.json')['items']
    assert sha(folder/'prepared.json')==read(RUN/'evidence/inputs.json')['shards'][str(n)]
    records=[];saved=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            key=row['pair'];assert done['ids'][i]==[key,'baseline','2d']
            if pack['selected_step'][i]<0:
                records.append(dict(pair=key,failed='no feasible baseline checkpoint'));continue
            imgs=[cv2.imread(str(OUT/'images'/(key+'_'+v+'.png'))) for v in ('left','right')]
            arms=wrist.frame_candidates(row,pack,i,model,camera,imgs,adapters,cfg)
            b={f:pack[f][i] for f in joint.FIELDS}
            w={k:v.copy() for k,v in b.items()}
            for (s,e,_),choices in zip(wrist.ARMS,arms):
                w['body_pose'][[s-1,e-1]]=min(choices,key=lambda c:c['loss'])['pose']
            options,probes=joint.frame(row,w,b,model,camera,cfg)
            losses=[o['loss'] for o in options]
            assert abs(losses[0]-pack['loss'][i])<2e-6
            select=[0,int(np.argmin(losses[:3])),int(np.argmin(losses))]
            records.append(dict(pair=key,offset=len(saved),selected=[options[j]['name'] for j in select],losses=[losses[j] for j in select],probes=probes))
            for j in select:
                opt=options[j]
                saved.append(dict(b,**{f:opt[f] for f in joint.FIELDS[:3]}))
    np.savez_compressed(folder/'selected.npz',**{f:np.stack([r[f] for r in saved]) for f in joint.FIELDS})
    write(folder/'selected.json',records)

if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1)
    task=sys.argv[1]
    if task=='prepare':prepare()
    else:
        sys.addaudithook(base.fit.forbid_gt)
        (fit if task=='fit' else candidates)(int(sys.argv[2]))
