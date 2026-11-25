"""Regenerate v3 and compare matched refinements without and with the frozen stereo contour term."""
import os,sys,time
from pathlib import Path
import numpy as np
import torch
import cv2
from shared_run import OUT,EV,ROOT,prev,search,wrist,Frame,Eye,NAMES,STRIDE,boundary,WEIGHT,guard
import fit_probe
import open3d as o3d

class GridEye(Eye):
    """Same targets, using only the original pixels consumed by Eye.compare."""
    def __init__(self,item,side,cam,adapter,labels,faces):
        self.cam,self.adapter,self.labels,self.faces=cam,adapter,labels,faces
        self.transform=np.array(item['inputs']['views'][side]['camera']['left_to_view'])
        path=OUT/'seg'/(item['pair']+'_'+side+'.npy')
        deadline=time.monotonic()+900
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(path.name)
            time.sleep(1)
        prob=np.load(path).astype(float)
        lab=prob.argmax(0)
        self.shape=lab.shape
        yy,xx=np.indices(self.shape)
        self.pix=np.stack((xx,yy),axis=-1)*STRIDE+STRIDE//2
        ray=cam.camera2world_ray(self.pix.reshape(-1,2)*adapter['scale']+adapter['offset'])
        rays=np.zeros((len(ray),6),np.float32)
        rays[:,3:]=ray
        self.rays=o3d.core.Tensor(rays)
        self.targets={}
        self.support={}
        for name,ident in NAMES.items():
            mask=lab!=0 if ident is None else lab==ident
            sure=(1-prob[0] if ident is None else prob.max(0))>=.55
            edge=boundary(mask)&sure
            self.support[name]=dict(area=int((mask&sure).sum()),edge=int(edge.sum()))
            if (mask&sure).sum()>=12 and edge.sum()>=6:
                self.targets[name]=self.pix[edge].astype(float)
        self.transform_t=torch.tensor(self.transform,dtype=torch.float32)
        assert self.targets, (item['pair'],side,'no observed contour')


def from_branch(seed,branch):
    x=seed.clone()
    for idx,key in [(slice(21),'body_pose'),(21,'delta'),(22,'translation')]:
        x[idx]=torch.as_tensor(branch[key]).reshape(x[idx].shape)
    return x


def worker(first,step,size,source='archive'):
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    fit_probe.Eye=GridEye
    prev.base.fit.camera_files=prev.ref.camera_files
    model,cam=prev.ref.common.setup_model()
    folder=prev.base.OUT
    audit=prev.read(search.RUN/'evidence/test_search.json')
    packpath=(folder if source=='archive' else OUT/'cold')/'budget_20000.npz'
    if source=='archive':assert prev.sha(packpath)==audit['baseline_sha256']
    assert prev.sha(folder/'prepared.json')==audit['prepared_sha256'] and prev.sha(folder/'protocol.json')==audit['protocol_sha256']
    pack=dict(np.load(packpath))
    cfg=prev.read(folder/'protocol.json')
    items=prev.read(folder/'prepared.json')['items']
    ids={row['pair']:i for i,row in enumerate(items) if row['condition']=='rw20000'}
    gallery=prev.read(prev.base.GOLD/'evidence/preparation.json')['pairs']
    assert len(ids)==len(gallery)==100
    old=prev.read(search.OUT/'test_choices.json')
    archived=dict(np.load(search.OUT/'test_bank.npz'))
    adapters=[prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    for n in range(first,size,step):
        pair='pair%03d'%(n+1)
        where=OUT/'poses'/(pair+'.json')
        if where.exists() and prev.read(where)['baseline_sha256']==prev.sha(packpath):continue
        rec=next(r for r in gallery if r['pair']==pair)
        imgs=[]
        for id_,digest in zip(rec['image_ids'],rec['image_sha256']):
            path=prev.TM.parent/'images'/(id_+'.png')
            assert prev.sha(path)==digest
            imgs.append(cv2.imread(str(path)))
        i=ids[pair];item=items[i]
        with torch.no_grad():
            arms=wrist.frame_candidates(item,pack,i,model,cam,imgs,adapters,cfg)
            base={k:pack[k][i].copy() for k in search.FIELDS}
            seed={k:v.copy() for k,v in base.items()}
            for (s,e,_),opts in zip(wrist.ARMS,arms):
                seed['body_pose'][[s-1,e-1]]=min(opts,key=lambda b:b['loss'])['pose']
            options,probes=search.frame(item,seed,base,model,cam,cfg)
            best=min(options,key=lambda v:v['loss'])
            orig=dict(seed,**{k:best[k] for k in search.FIELDS[:3]})
            j=old[pair]['offset']+old[pair]['best']
            diff=max(float(np.max(abs(orig[k]-archived[k][j]))) for k in search.FIELDS)
            frame=Frame(pair,item,orig,model,cam,cfg)
        states={'v3':frame.v3}
        methods={}
        for method,outline in [('matched',False),('outline',True)]:
            evolved,_=frame.descend(frame.v3,False,outline,'new_v3')
            fitted=dict(orig,body_pose=evolved[:21].numpy(),delta=evolved[21].numpy(),translation=evolved[22].numpy())
            with torch.no_grad():
                newopts,_=search.frame(item,fitted,fitted,model,cam,cfg)
                candidates={'v3':frame.v3,'adam':evolved}
                candidates.update({'v3_'+p['name']:from_branch(frame.v3,p) for p in options})
                candidates.update({'new_'+p['name']:from_branch(evolved,p) for p in newopts})
                seen=set()
                scores={}
                for name,x in candidates.items():
                    key=x.numpy().tobytes()
                    if key in seen:continue
                    seen.add(key)
                    v,_,raw,ok=frame.state(x)
                    if not ok:continue
                    loss=raw.item()
                    if outline:loss+=WEIGHT*frame.observed(v)[0].item()
                    scores[name]=loss
                bestname=min(scores,key=scores.get)
                states[method]=candidates[bestname]
                methods[method]=dict(choice=bestname,candidates=len(scores),loss=scores[bestname])
        path=OUT/'poses'/(pair+'.npz')
        np.savez_compressed(path,global0=orig['global0'],**{k:v.numpy() for k,v in states.items()})
        prev.write(where,dict(pair=pair,subject=rec['subject'],archive_max_parameter_difference=diff,
             v3_choice=best['name'],prior_v3_choice=old[pair]['names'][old[pair]['best']],baseline_sha256=prev.sha(packpath),
             regenerated_chain_probes=probes,methods=methods,pose_sha256=prev.sha(path),
             eye_support=[list(e.targets) for e in frame.eyes],exit_status=0))
        print(pair,methods['outline']['choice'],flush=True)


if __name__=='__main__':worker(*map(int,sys.argv[1:4]),*sys.argv[4:])
