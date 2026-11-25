"""Paired counterfactuals on the archived standard-shape v3 starts."""
import sys
import cv2
import numpy as np
import torch
from geometry_inputs import RUN,OUT,EV,PREV,PRIOR,GridEye,from_branch,fit_probe,Frame,prev,search,guard,WEIGHT
from hand_edges import MaskedEye,HANDS
MODES=['fixed_full','palm','no_hand']

class EdgeFrame(Frame):
    lock_fingers=True

    def cap(self,x):
        value=super().cap(x)
        if self.lock_fingers:value[23:]=self.v3[23:]
        return value

    def observed(self,v,grad=False):
        valid=[eye.compare(v,grad) for eye in self.eyes if eye.targets]
        return (sum(t[0] for t in valid)/len(valid),[t[1] for t in valid]) if valid else (v.sum()*0,[])


def main(first,step,group):
    sys.addaudithook(guard)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    fit_probe.Eye=GridEye
    prev.base.fit.camera_files=prev.ref.camera_files
    model,cam=prev.ref.common.setup_model()
    folder=prev.base.OUT
    audit=prev.read(search.RUN/'evidence/test_search.json')
    for p,k in [(folder/'prepared.json','prepared_sha256'),(folder/'protocol.json','protocol_sha256'),
                (search.OUT/'test_bank.npz','parameter_sha256'),(search.OUT/'test_choices.json','choices_sha256')]:
        assert prev.sha(p)==audit[k]
    cfg=prev.read(folder/'protocol.json');items=prev.read(folder/'prepared.json')['items']
    bank=dict(np.load(search.OUT/'test_bank.npz')); choices=prev.read(search.OUT/'test_choices.json')
    archive=prev.read(PREV/'evidence/selections.json')['rows']
    seg=prev.read(PREV/'evidence/segmentation.json')['eyes']
    dest=OUT/('poses' if group=='fixed' else 'free')
    dest.mkdir(exist_ok=True)
    for n in range(first,100,step):
        pair='pair%03d'%(n+1)
        path=dest/(pair+'.npz');manifest=dest/(pair+'.json')
        if manifest.exists() and path.exists() and prev.sha(path)==prev.read(manifest)['sha256']:continue
        record=choices[pair];i=record['offset']+record['best']
        orig={k:bank[k][i] for k in search.FIELDS};assert not orig['betas'].any()
        priorfile=PRIOR/'poses'/(pair+'.npz');assert prev.sha(priorfile)==archive[pair]['pose_sha256']
        item=items[record['index']]
        frame=EdgeFrame(pair,item,orig,model,cam,cfg)
        frame.lock_fingers=group=='fixed'
        assert np.array_equal(frame.v3.numpy(),np.load(priorfile)['v3']) and not frame.v3[23:].any()
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
            f=PRIOR/'seg'/(name+'.npy');assert prev.sha(f)==seg[name]['sampled_sha256']
            labels.append(np.load(f).argmax(0))
        with torch.no_grad():start=frame.state(frame.v3)[0]
        states={};details={}
        for mode in (MODES if group=='fixed' else ['free_palm']):
            frame.eyes=eyes if mode=='fixed_full' else [MaskedEye(eye,('palm' if mode=='free_palm' else mode),lab,finger_faces,start) for eye,lab in zip(eyes,labels)]
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
        np.savez_compressed(path,global0=orig['global0'],**states)
        prev.write(manifest,dict(pair=pair,methods=details,sha256=prev.sha(path),finger_faces=int(finger_faces.sum()),exit_status=0))
        print(pair,flush=True)

if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else 'fixed')
