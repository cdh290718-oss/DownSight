"""Bind the frozen wrist solver and original validation scorer to identical v4 body seeds."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/validation200_v5_20261127'
BASE=ROOT/'outputs/validation200_fingers_20261125'
ARCH=RUN.parent/'2026-11-25_validation200_fingers'
read=lambda p:json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,allow_nan=False))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(key):
    z=np.load(BASE/'poses'/(key+'.npz'))
    assert sha(BASE/'poses'/(key+'.npz'))==receipts[key]['sha256']
    x=z['fixed_full']; pack=dict(locked=x,global0=z['global0'],betas=np.zeros(10,dtype=x.dtype))
    row=items[key]
    world=[w.old.mat(x[21])@w.old.mat(pack['global0'])]
    for i in range(1,22):world.append(world[int(layer.parents[i])]@w.old.mat(x[i-1]))
    trans=[np.array(row['inputs']['views'][e]['camera']['left_to_view']) for e in w.SIDES]
    def project(points):return np.stack([w.old.projection(points@t[:3,:3].T+t[:3,3],c) for t,c in zip(trans,cams)])
    result=x.copy(); hands={}
    w.prior.KEYS=(key,)
    for i,side in enumerate(w.SIDES):
        target=np.stack([pred[pairs[key][e]['id']][order+(21 if i==0 else 0)] for e in w.SIDES])
        dist=np.linalg.norm(target-target[:,:1],axis=2)
        good=((target>=0)&(target<872)).all(-1)&(dist<3.2*np.median(dist[:,w.ROOTS],axis=1)[:,None])
        for v,c in enumerate(cams):
            a=c['input_adapter'];target[v]=target[v]*a['scale']+a['offset']
        pose,hands[side]=w.prior.choose_hand(key,i,layer,pack,means,rest,world,project,target,good[:,w.FINGER_JOINTS])
        if pose is not None:
            sl=slice(23+15*i,38+15*i); result[sl]=pose[sl]; result[19+i]=pose[19+i]
    path=OUT/'rows'/(key+'.npz')
    np.savez_compressed(path,v5=result)
    return key,dict(hands=hands,sha256=sha(path),seed_sha256=receipts[key]['sha256'])

def score(key):
    z=np.load(BASE/'poses'/(key+'.npz')); start=z['v3']
    path=OUT/'rows'/(key+'.npz');assert sha(path)==selected[key]['sha256']
    x=torch.tensor(np.load(path)['v5'])
    orig=dict(body_pose=start[:21],delta=start[21],translation=start[22],global0=z['global0'])
    frame=g.EdgeFrame(key,items[key],orig,model,cam,cfg)
    with torch.no_grad():
        v,n,_,ok=frame.state(x);assert ok
        target=np.asarray(truth[key]['gt'])[obs];pred=n[0,natives].numpy()
        hits,_=frame.gate.check(v.numpy()); counts=g.summary(hits,frame.labels)
        m=dict(pa_mm=float(1000*np.linalg.norm(g.prev.align(pred,target)-target,axis=1).mean()),nonpa_mm=float(1000*np.linalg.norm(pred-target,axis=1).mean()),full_outline_px=frame.observed(v)[0].item()*12,left_self=counts.get('L_hand/L_hand',0),right_self=counts.get('R_hand/R_hand',0),target_counts={'/'.join(p):counts.get('/'.join(sorted(p)),0) for p in g.PAIRS},finger_max_delta_rad=float(x[23:].abs().max()))
    old=historical['rows'][key]
    assert abs(m['nonpa_mm']-old['methods']['fixed_full']['nonpa_mm'])<.001
    return key,dict(subject=old['subject'],methods=dict(v3=old['methods']['v3'],v4=old['methods']['fixed_full'],v5=m))

if __name__=='__main__':
    torch.set_num_threads(1)
    keys=['val%03d'%n for n in range(1,201)]
    folder=ROOT/'outputs/stereo_wrist_candidates_20261123'
    items={r['pair']:r for r in read(folder/'validation_fit/prepared.json')['items']}
    if sys.argv[1]=='fit':
        sys.path.insert(0,str(RUN.parent/'2026-11-27_wrist_only_default'))
        import wrist_only as w
        w.prior.RUN=RUN;w.prior.OUT=OUT
        model,_=w.ref.prev.ref.common.setup_model();layer=model.layer['neutral'].eval()
        means=layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
        zero=np.zeros((53,3),dtype='f');zero[23:]=-means
        _,rest=w.old.mesh(layer,zero,dict(global0=np.zeros(3),betas=np.zeros(10,dtype='f')))
        cams=[read(ROOT/'outputs/rw_frozen200_golden_d_20261118'/('camera_'+e+'.json')) for e in w.SIDES]
        records=read(ROOT/'outputs/rw_a20000_dataset_20261117/validation.json')['records'];by={}
        for r in records: by.setdefault(r['pair_id'],{})[r['view']]=r
        pairs=dict(zip(keys,[by[k] for k in sorted(by)]))
        predictions=read(ROOT/'outputs/rw_a20000_evaluation_20261117/rw20000_validation_predictions.json')
        pred={r['id']:np.array(r['xy']) for r in predictions}
        assert {r['image_sha256'] for r in records}=={r['image_sha256'] for r in predictions}
        order=np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
        receipts=read(ARCH/'evidence/selections.json')
        selected=dict(map(fit,keys))
        write(RUN/'evidence/selection.json',selected)
    else:
        sys.path.insert(0,str(ARCH))
        import score as g
        model,cam=g.prev.ref.common.setup_model()
        cfg=read(folder/'validation_fit/protocol.json')
        historical=read(ARCH/'evidence/results.json')
        selected=read(RUN/'evidence/selection.json');truth=read(folder/'validation_truth.json')
        assert sha(folder/'validation_truth.json')==historical['reference_sha256']
        obs=g.prev.ref.common.OBS;natives=np.array(g.prev.ref.common.NATIVE_IDS)[obs]
        rows=dict(map(score,keys))
        g.MODES=('v3','v4','v5')
        write(RUN/'evidence/results.json',dict(summary=g.aggregate(rows.values()),paired=g.changes(rows.values(),'v5','v4'),subjects={s:g.aggregate([r for r in rows.values() if r['subject']==s]) for s in sorted({r['subject'] for r in rows.values()})},rows=rows))
