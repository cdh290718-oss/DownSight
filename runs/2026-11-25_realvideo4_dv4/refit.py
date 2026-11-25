"""Replay the original four v3 candidate banks, then apply the locked Golden D v4 branch."""
from pathlib import Path
import sys
import numpy as np
import torch
import cv2

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo4_dv4_20261125'
OLD=ROOT/'outputs/realvideo4_dv3_20261125'
PRIOR=RUN.parent/'2026-11-25_realvideo4_dv3'
sys.path.insert(0,str(RUN.parent/'2026-11-25_floor24_foot_lock'))
from lock import FixedFeet, FEET, WEIGHT, search, prev, compare, from_branch, guard
import candidates as wrist
read,write,sha=prev.read,prev.write,prev.sha

def cameras():
    return [OLD/(side+'_cal.json') for side in ('left','right')]

def run(i):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    sys.addaudithook(guard)
    manifest=read(ROOT/'configs/golden_d.current.json')
    assert manifest['method_id']=='golden-d-2d-v4' and all(sha(ROOT/x['path'])==x['sha256'] for x in manifest['locked_files'])
    compare.OUT=OUT
    prev.ref.camera_files=prev.base.fit.camera_files=cameras
    folder=OLD/'fit/0'
    cfg=read(folder/'protocol.json')
    inputs=read(PRIOR/'evidence/inputs.json')
    assert sha(folder/'protocol.json')==inputs['protocol'] and sha(folder/'prepared.json')==inputs['shards']['0']
    assert [sha(p) for p in cameras()]==read(PRIOR/'evidence/calibration.json')['converted']
    assert sha(folder/'selected.npz')==read(PRIOR/'evidence/results.json')['selected_hash']
    row=read(folder/'prepared.json')['items'][i]
    key=row['pair']
    sel=read(folder/'selected.json')[i]
    assert sel['pair']==key
    raw=dict(np.load(folder/'budget_20000.npz'))
    pack=dict(np.load(folder/'selected.npz'))
    orig={k:pack[k][sel['offset']+2] for k in search.FIELDS}
    base={k:raw[k][i].copy() for k in search.FIELDS}
    seed={k:v.copy() for k,v in base.items()}
    seg=read(RUN/'evidence/segmentation.json')['eyes']
    eyes=[key+'_'+s for s in ('left','right')]
    assert all(sha(OUT/'seg'/(e+'.npy'))==seg[e]['sampled_sha256'] and sha(OLD/'images'/(e+'.png'))==seg[e]['image_sha256'] for e in eyes)
    ims=[cv2.imread(str(OLD/'images'/(e+'.png'))) for e in eyes]
    model,cam=prev.ref.common.setup_model()
    with torch.no_grad():
        arms=wrist.frame_candidates(row,raw,i,model,cam,ims,[read(p)['input_adapter'] for p in cameras()],cfg)
        for (s,e,_),opts in zip(wrist.ARMS,arms):
            seed['body_pose'][[s-1,e-1]]=min(opts,key=lambda b:b['loss'])['pose']
        branches,probes=search.frame(row,seed,base,model,cam,cfg)
        best=min(branches,key=lambda b:b['loss'])
        assert sel['probes']==probes and sel['selected'][-1]==best['name']
        assert all(np.array_equal(best[k],orig[k]) for k in search.FIELDS[:3])
    frame=FixedFeet(key,row,orig,model,cam,cfg)
    assert not frame.v3[FEET].any() and not orig['betas'].any()
    evolved,_=frame.descend(frame.v3,False,True,'v3')
    assert frame.logs[0]['steps']==72
    fitted=dict(orig,body_pose=evolved[:21].numpy(),delta=evolved[21].numpy(),translation=evolved[22].numpy())
    with torch.no_grad():
        new,newprobes=search.frame(row,fitted,fitted,model,cam,cfg)
        pool={'v3':frame.v3,'adam':evolved}
        pool.update({'v3_'+b['name']:from_branch(frame.v3,b) for b in branches})
        pool.update({'new_'+b['name']:from_branch(evolved,b) for b in new})
        scores={}
        for name,x in pool.items():
            assert not x[FEET].any() and not x[23:].any()
            v,_,raw_loss,ok=frame.state(x)
            if ok:
                edge,details=frame.observed(v)
                scores[name]=dict(body=raw_loss.item(),contour=edge.item(),total=raw_loss.item()+WEIGHT*edge.item())
        winner=min(scores,key=lambda k:scores[k]['total'])
    path=OUT/'poses'/(key+'.npz')
    np.savez_compressed(path,locked=pool[winner].numpy(),global0=orig['global0'],betas=orig['betas'])
    write(RUN/'evidence'/(key+'.json'),dict(pair=key,method=manifest['method_id'],selected=winner,pose_hash=sha(path),
          steps=frame.logs,choice_scores=scores,supported=[list(e.targets) for e in frame.eyes],weight=WEIGHT,
          v3_unchanged_start=True,foot_and_hand_locks_checked=True,probes=[probes,newprobes],exit_status=0))
    print(key,winner,flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
