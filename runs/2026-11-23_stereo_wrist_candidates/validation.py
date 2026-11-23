"""Prepare held-out validation nuisance data, then fit the locked v2 baseline separately."""
import sys
from collections import Counter
from shared import *

def prepare():
    predfile=ROOT/'outputs/rw_a20000_evaluation_20261117/rw20000_validation_predictions.json'
    pred={r['id']:r for r in read(predfile)}
    pairs=pair_images(VM)
    subjects={a['subject'] for a,b in pairs}
    train=set(read(ROOT/'runs/2026-11-17_rw-a20000-training/evidence/dataset.json')['summary']['train']['subjects'])
    test={a['subject'] for a,b in pair_images(TM)}
    assert not (subjects & (train|test)) and not (train & test)
    source=ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Train/prepared/samples.npz'
    z=np.load(source)
    seq, frame, side=z['source_sequence'],z['frame_index'],z['camera_side']
    xyz,heads=z['keypoints3d'],z['keypoints3d_fit'][:,0]
    used={(r['sequence'],r['frame']) for pair in pairs for r in pair}
    used_seq={s for s,f in used}
    lookup={(str(s),int(f),str(v)):i for i,(s,f,v) in enumerate(zip(seq,frame,side))}
    rigs={}; calibrations=[]; rows=[]; truths={}
    camspec=[read(p) for p in ref.camera_files()]
    model,camera=ref.common.setup_model()
    cams=[camera(p) for p in ref.camera_files()]
    layer=model.layer['neutral']
    rest=(layer.J_regressor@layer.v_template).detach().numpy()
    ids=mapping()
    audit=read(base.RUN/'evidence/inputs.json')
    assert sha(base.OUT/'prepared.json')==audit['solver_input_sha256']
    template=read(base.OUT/'prepared.json')['items'][0]['points']
    fallbacks=0
    for n,pair in enumerate(pairs):
        l,r=pair
        session=l['sequence'].rsplit('/',1)[0]
        if session not in rigs:
            candidates=[k for k in lookup if k[2]=='left' and k[0].rsplit('/',1)[0]==session and k[:2] not in used and (*k[:2],'right') in lookup]
            key=min(candidates,key=lambda k:(k[0] in used_seq,k))
            a=xyz[lookup[key]].astype(float); b=xyz[lookup[(*key[:2],'right')]].astype(float)
            u,_,v=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
            rot=v.T@np.diag([1.,1.,np.linalg.det(v.T@u.T)])@u.T
            t=np.eye(4);t[:3,:3]=rot;t[:3,3]=b.mean(0)-rot@a.mean(0)
            rigs[session]=t
            calibrations.append(dict(subject=l['subject'],excluded_frame=True,excluded_sequence=key[0] not in used_seq))
        views={'left':{'camera':{'left_to_view':np.eye(4).tolist()}},'right':{'camera':{'left_to_view':rigs[session].tolist()}}}
        pts=[]
        for p,sid in zip(template,ids):
            q={k:p[k] for k in ('name','index','native_index','group','accepted')}
            q['xy']=[];q['valid']=[]
            for eye,(record,cal) in enumerate(zip(pair,camspec)):
                assert pred[record['id']]['image_sha256']==record['image_sha256']
                adapt=cal['input_adapter']
                q['xy'].append((np.array(pred[record['id']]['xy'][sid])*adapt['scale']+adapt['offset']).tolist())
                q['valid'].append(bool(record['keypoints308_supervision_mask'][sid]))
            pts.append(q)
        li,ri=(record['source_index'] for record in pair)
        assert all(lookup[(v['sequence'],v['frame'],v['view'])]==v['source_index'] for v in pair)
        assert np.max(np.linalg.norm(xyz[li]@rigs[session][:3,:3].T+rigs[session][:3,3]-xyz[ri],axis=1))<1e-4
        tag='val%03d'%(n+1)
        row=dict(pair=tag,condition='baseline',points=pts,inputs={'views':views},head_position=heads[li].tolist())
        seed,reason=base.trial.body_seed(row,cams)
        if reason:
            seed=base.shoulder_hip_recovery(row,cams,rest);fallbacks+=1
        row['initial_body_rotation']=seed.tolist()
        rows.append(row);truths[tag]=dict(gt=xyz[li].tolist(),subject=l['subject'])
    assert len(rows)==200
    write(VAL/'prepared.json',dict(items=rows))
    (VAL/'protocol.json').write_bytes((base.OUT/'protocol.json').read_bytes())
    write(OUT/'validation_truth.json',truths)
    write(RUN/'evidence/splits.json',dict(base='4461ff0c066c242f8de21b0b1130f3c25916a554',
          train=sorted(train),validation=dict(Counter(l['subject'] for l,r in pairs)),test=sorted(test),
          calibrations=calibrations,fallback_initializations=fallbacks,
          hashes=dict(validation_manifest=sha(VM),predictions=sha(predfile),validation_source=sha(source),
                      validation_prepared=sha(VAL/'prepared.json'),test_manifest=sha(TM),protocol=sha(VAL/'protocol.json'))))

def fit():
    base.fit.RUN=base.fit.OUT=VAL
    base.fit.camera_files=ref.camera_files
    base.fit.Problem=base.trial.TorsoOnly
    sys.argv=[__file__,'--threads','1']
    base.fit.main()

if __name__=='__main__':
    (prepare if sys.argv[1]=='prepare' else fit)()
