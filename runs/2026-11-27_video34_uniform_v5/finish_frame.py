"""Use the purple placement and shoulder/elbow adjustment of the 003/004 reference."""
from base import *
import torch,numpy as np
sys.path.insert(0,str(RUN.parent/'2026-11-27_wrist_only_default'))
import wrist_only as w
from calibrate import rays

def prepare_arm(key,index,row,model,cam,pred):
    import palm_batch as pb
    ref=w.ref
    ref.OLD=ref.OUT=OUT
    ref.PRIOR=RUN
    ref.cameras=cameras
    ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=cameras
    ref.compare.OUT=OUT
    v4=OUT/'poses'/(key+'.npz')
    assert sha(v4)==read(RUN/'evidence'/(key+'.json'))['pose_hash']
    xy=np.array([pred[key+'_'+e+'.png']['xy'] for e in SIDES])
    channels=[62,49,53,57,61,41,28,32,36,40]
    usable=np.isfinite(xy[:,channels]).all() and ((xy[:,channels]>=0)&(xy[:,channels]<[1280,720])).all()
    reason=''
    if usable:
        # The optional palm seed needs positive-depth stereo rays for both hands.
        for ids in (channels[:5],channels[5:]):
            transform=np.array(row['inputs']['views']['right']['camera']['left_to_view'])
            rr=[rays(pts,read(p)) for pts,p in zip(xy[:,ids],cameras())]
            R,t=transform[:3,:3],transform[:3,3]
            for a,b in zip(rr[0],rr[1]@R):
                if min(np.linalg.lstsq(np.stack((a,-b),axis=1),-R.T@t,rcond=None)[0])<=0:
                    usable=False;reason='nonpositive palm triangulation'
    else:reason='palm seed points outside image'
    if usable:
        pb.main(index)
        palm=OUT/'palm/poses'/(key+'.npz')
    else:palm=v4
    pack=dict(np.load(palm)); start=pack['locked'].copy()
    settings=read(RUN.parent/'2026-11-26_realvideo4_mount_pitch/evidence/temporal.json')['zero']
    assert settings['deg']==0
    start[22]+=np.array(settings['translation_m'])
    initial=dict(body_pose=start[:21],delta=start[21],translation=start[22],global0=pack['global0'],betas=pack['betas'])
    frame=ref.FixedFeet(key,row,initial,model,cam,read(OUT/'fit/0/protocol.json'))
    seed=torch.tensor(start); targets=[]; valid=[]
    for n in range(2):
        elbows=[next(p for p in row['points'] if p['name']==s+'_Elbow') for s in ['L','R']]
        coords=np.array([p['xy'][n] for p in elbows]+[xy[n,62],xy[n,41]])
        targets.append(coords);valid.append(((coords>=0)&(coords<[1280,720])).all(1))
    target=seed.new_tensor(np.array(targets));mask=torch.tensor(np.array(valid))
    move=torch.tensor([15,16,17,18])
    delta=torch.zeros_like(seed[move],requires_grad=True)
    def pose():return seed.index_copy(0,move,seed[move]+delta)
    def projection(x):
        j=w.paired.body(frame,x)[1]
        limb=j[[18,19,20,21]]
        return j,torch.stack([w.paired.project(limb@e.transform_t[:3,:3].T+e.transform_t[:3,3],e.cam) for e in frame.eyes])
    solver=torch.optim.LBFGS([delta],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        solver.zero_grad();_,uv=projection(pose())
        err=(uv-target).square().sum(-1)
        def group(sl):
            m=mask[:,sl]
            return (err[:,sl]*m).sum()/m.sum().clamp_min(1)
        loss=group(slice(2))+4*group(slice(2,4))+25*delta.square().sum(1).mean()
        loss.backward()
        return loss
    solver.step(closure)
    x=pose().detach();j0,_=projection(seed);j1,uv=projection(x)
    assert torch.isfinite(uv).all() and (j0[15]-j1[15]).norm()<1e-6
    return dict(pack,locked=x.numpy()),dict(v4_sha256=sha(v4),palm_sha256=sha(palm),palm_fallback=reason,valid_elbow_wrist_by_eye=valid[0].tolist()+valid[1].tolist(),purple_translation_m=settings['translation_m'])

def main(index):
    index=int(index);torch.set_num_threads(1)
    rows=read(OUT/'fit/0/prepared.json')['items']
    row=rows[index];key=row['pair']
    manifest=read(ROOT/'configs/golden_d.current.json')
    assert manifest['method_id']=='golden-d-2d-v5' and all(sha(ROOT/f['path'])==f['sha256'] for f in manifest['locked_files'])
    model,cam=w.ref.prev.ref.common.setup_model()
    pred=read(OUT/'predictions.json'); assert sha(OUT/'predictions.json')==read(RUN/'evidence/detection.json')['predictions_sha256']
    pack,source=prepare_arm(key,index,row,model,cam,pred)
    x=pack['locked'];layer=model.layer['neutral'].eval()
    means=layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
    neutral=np.zeros_like(x);neutral[23:]=-means
    _,rest=w.old.mesh(layer,neutral,dict(pack,global0=np.zeros(3)))
    world=[w.old.mat(x[21])@w.old.mat(pack['global0'])]
    for n in range(1,22):world.append(world[int(layer.parents[n])]@w.old.mat(x[n-1]))
    trans=[np.array(row['inputs']['views'][e]['camera']['left_to_view']) for e in SIDES]
    cams=list(map(read,cameras()))
    def project(p):return np.stack([w.old.projection(p@t[:3,:3].T+t[:3,3],c) for t,c in zip(trans,cams)])
    order=np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
    final=x.copy();hands={}
    w.prior.RUN=RUN;w.prior.OUT=OUT;w.prior.KEYS=(key,)
    for i,s in enumerate(SIDES):
        obs=np.stack([np.array(pred[key+'_'+eye+'.png']['xy'])[order+(21 if i==0 else 0)] for eye in SIDES])
        distance=np.linalg.norm(obs-obs[:,:1],axis=2)
        inside=((obs>=0)&(obs<[1280,720])).all(-1)
        used=(inside&(distance<3.2*np.median(distance[:,w.ROOTS],axis=1)[:,None]))[:,w.FINGER_JOINTS]
        alt,choice=w.prior.choose_hand(key,i,layer,pack,means,rest,world,project,obs,used)
        if alt is not None:
            sl=slice(23+15*i,38+15*i);final[sl]=alt[sl];final[19+i]=alt[19+i]
        hands[s]=dict(choice,obs=obs.tolist(),valid=inside.tolist(),fit_mask=used.tolist())
    _,before=w.old.mesh(layer,x,pack)
    v,j=w.old.mesh(layer,final,pack)
    drift=float(1000*np.max(np.linalg.norm(j[:22]-before[:22],axis=1)))
    assert drift<.001
    w.prior.stereo.OUT=OUT
    loaded,obj=w.prior.stereo.export(key,'v5',v,np.asarray(model.face,np.int32))
    assert np.max(abs(loaded-v))<1e-8
    np.savez_compressed(OUT/'poses'/(key+'_v5.npz'),**dict(pack,locked=final))
    for i,s in enumerate(SIDES):
        hands[s]['uv']=project(j[w.prior.IDS[i]]).tolist()
    write(OUT/(key+'_hands.json'),dict(source=source,hands=hands,drift_mm=drift,obj=obj.name,obj_sha256=sha(obj),exit_status=0))
