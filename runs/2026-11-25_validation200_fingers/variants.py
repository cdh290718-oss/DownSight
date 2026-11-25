"""Run both locked-hand treatments on the identical archived golden D v3 pose."""
import sys
import cv2
import numpy as np
import torch
from inputs import OUT, EdgeFrame, from_branch, prev, search, guard, WEIGHT, MaskedEye, HANDS


def main(first,step):
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model,cam=prev.ref.common.setup_model()
    folder=prev.VAL
    audit=prev.read(search.RUN/'evidence/validation_search.json')
    for p,k in [(folder/'prepared.json','prepared_sha256'),(folder/'protocol.json','protocol_sha256'),
                (search.OUT/'validation_bank.npz','parameter_sha256'),(search.OUT/'validation_choices.json','choices_sha256')]:
        assert prev.sha(p)==audit[k]
    cfg=prev.read(folder/'protocol.json')
    items=prev.read(folder/'prepared.json')['items']
    bank=dict(np.load(search.OUT/'validation_bank.npz'))
    choices=prev.read(search.OUT/'validation_choices.json')
    keys=sorted(choices)
    assert keys==[f'val{i:03d}' for i in range(1,201)]
    dest=OUT/'poses'
    for pair in keys[first::step]:
        path=dest/(pair+'.npz')
        manifest=dest/(pair+'.json')
        if manifest.exists() and path.exists() and prev.sha(path)==prev.read(manifest)['sha256']:continue
        record=choices[pair]
        i=record['offset']+record['best']
        orig={k:bank[k][i] for k in search.FIELDS}
        assert not orig['betas'].any()
        item=items[record['index']]
        assert pair==item['pair']
        frame=EdgeFrame(pair,item,orig,model,cam,cfg)
        assert frame.lock_fingers and not frame.v3[23:].any()
        digests={}
        weights=frame.layer.lbs_weights.detach().numpy()
        assert weights.shape[1]==55
        # Wrist weights anchor the palm; columns 25..54 belong to the 30 SMPL-X finger joints.
        finger_weights=weights[:,25:].sum(1)[frame.faces].sum(1)
        wrist_weights=weights[:,[20,21]].sum(1)[frame.faces].sum(1)
        finger_faces=np.isin(frame.labels,HANDS) & (finger_weights>wrist_weights)
        eyes=frame.eyes
        labels=[]
        for side in ('left','right'):
            name=pair+'_'+side
            f=OUT/'seg'/(name+'.npy')
            digests[name]=prev.sha(f)
            labels.append(np.load(f).argmax(0))
        with torch.no_grad():start=frame.state(frame.v3)[0]
        states={};details={}
        for mode in ('fixed_full','palm'):
            frame.eyes=eyes if mode=='fixed_full' else [MaskedEye(eye,mode,lab,finger_faces,start) for eye,lab in zip(eyes,labels)]
            support=[{k:[len(t),len(t)] for k,t in e.targets.items()} if mode=='fixed_full' else e.counts for e in frame.eyes]
            evolved,_=frame.descend(frame.v3,False,True,'v3')
            fitted=dict(orig,body_pose=evolved[:21].numpy(),delta=evolved[21].numpy(),translation=evolved[22].numpy())
            with torch.no_grad():
                new,_=search.frame(item,fitted,fitted,model,cam,cfg)
                candidates={'v3':frame.v3,'adam':evolved}
                for slot,name in enumerate(record['names']):
                    x={k:bank[k][record['offset']+slot] for k in search.FIELDS}
                    candidates['v3_'+name]=from_branch(frame.v3,x)
                candidates.update({'new_'+v['name']:from_branch(evolved,v) for v in new})
                seen=set();scores={}
                for tag,x in candidates.items():
                    if frame.lock_fingers:assert torch.equal(x[23:],frame.v3[23:])
                    ident=x.numpy().tobytes()
                    if ident in seen:continue
                    seen.add(ident)
                    v,_,raw,ok=frame.state(x)
                    if ok:scores[tag]=raw.item()+WEIGHT*frame.observed(v)[0].item()
                name=min(scores,key=scores.get);states[mode]=candidates[name].numpy()
                details[mode]=dict(chosen=name,loss=scores[name],support=support)
        np.savez_compressed(path,global0=orig['global0'],v3=frame.v3.numpy(),**states)
        prev.write(manifest,dict(pair=pair,methods=details,seg_sha256=digests,sha256=prev.sha(path),finger_faces=int(finger_faces.sum()),exit_status=0))
        print(pair,flush=True)

if __name__=='__main__': main(*map(int,sys.argv[1:3]))
