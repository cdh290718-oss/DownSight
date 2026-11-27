"""Single-pair final v5 comparison with wrist:elbow weights 4:1 and 1:1."""
from pathlib import Path
import sys
import cv2
import torch
import numpy as np
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'2026-11-27_video34_elbow_audit'))
import audit as a
from render_result import load_obj
key='v003_04'
dest=RUN.parents[1]/'outputs/v003_04_equal_wrist_20261127'
w=a.original.w
torch.set_num_threads(1)
cv2.setNumThreads(1)
ref=w.ref
ref.OLD=ref.OUT=ref.compare.OUT=a.OUT
ref.PRIOR=a.PRIOR
ref.cameras=ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=a.cameras
model,cam=ref.prev.ref.common.setup_model()
layer=model.layer['neutral'].eval()
faces=np.asarray(model.face,np.int32)
src=a.read(a.PRIOR/'evidence'/(key+'_render.json'))
prepared=a.OUT/'fit/0/prepared.json'
detect=a.OUT/'predictions.json'
assert a.sha(prepared)==a.read(a.PRIOR/'evidence/inputs.json')['shards']['0']
assert a.sha(detect)==a.read(a.PRIOR/'evidence/detection.json')['predictions_sha256']
row=next(r for r in a.read(prepared)['items'] if r['pair']==key)
path=a.OUT/('poses' if src['source']['palm_fallback'] else 'palm/poses')/(key+'.npz')
assert a.sha(path)==src['source']['palm_sha256']
pack=dict(np.load(path))
pack['locked'][22]+=np.asarray(src['source']['purple_translation_m'])
pred=a.read(detect)
target=a.elbow_observations(row,['L','R'],pred,key)
mask=((target>=0)&(target<[1280,720])).all(2)
tfs=[np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in a.original.SIDES]
cams=list(map(a.read,a.cameras()))
def project(points):
    return np.stack([a.projection(points@t[:3,:3].T+t[:3,3],c) for t,c in zip(tfs,cams)])
obs=a.read(a.OUT/(key+'_hands.json'))['hands']
means=layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
w.prior.RUN,w.prior.OUT,w.prior.KEYS=a.PRIOR,a.OUT,(key,)
w.prior.stereo.OUT=dest
metrics={}
meshes=[]
for weight in (4,1):
    arm=a.replay(pack,target,mask,tfs,model,cam,row,key,weight)
    handpack=dict(pack,locked=arm)
    neutral=np.zeros_like(arm)
    neutral[23:]=-means
    _,rest=w.old.mesh(layer,neutral,dict(pack,global0=np.zeros(3)))
    world=[w.old.mat(arm[21])@w.old.mat(pack['global0'])]
    for n in range(1,22):
        world.append(world[int(layer.parents[n])]@w.old.mat(arm[n-1]))
    final=arm.copy()
    hands={}
    for i,s in enumerate(a.original.SIDES):
        h=obs[s]
        alt,choice=w.prior.choose_hand(key,i,layer,handpack,means,rest,world,project,np.array(h['obs']),np.array(h['fit_mask']))
        hands[s]=dict(source=choice['source'])
        if alt is not None:
            finger=slice(23+15*i,38+15*i)
            final[19+i]=alt[19+i];final[finger]=alt[finger]
    v,j=w.old.mesh(layer,final,pack)
    meshes.append((v,j))
    before=w.old.mesh(layer,arm,pack)[1]
    drift=np.linalg.norm(before[:22]-j[:22],axis=1).max()*1000
    assert drift<.001
    if weight==4:
        assert np.array_equal(final,np.load(a.OUT/'poses'/(key+'_v5.npz'))['locked'])
        obj=a.OUT/'export'/src['obj']
        assert a.sha(obj)==src['obj_sha256']
    else:
        _,obj=w.prior.stereo.export(key,'equal',v,faces)
        np.savez_compressed(dest/'final.npz',**dict(pack,locked=final))
    verify,topology=load_obj(obj)
    assert np.max(abs(verify-v))<1e-8 and np.array_equal(topology,faces)
    for i,s in enumerate(a.original.SIDES):
        uv=project(j[[18+i,20+i]])
        err=np.linalg.norm(uv-target[:,[i,i+2]],axis=-1)
        tips=np.asarray(w.prior.TIPS)
        tipdiff=(project(j[w.prior.IDS[i]])-np.array(obs[s]['obs']))[:,tips]
        gaps=np.linalg.norm(tipdiff,axis=-1)[np.array(obs[s]['valid'])[:,tips]]
        hands[s].update(angle=a.angle(j[18+i]-j[16+i],j[20+i]-j[18+i]),elbow=err[mask[:,i],0].tolist(),wrist=err[mask[:,2+i],1].tolist(),tips=gaps.tolist())
    metrics[str(weight)]=dict(hands=hands,obj_sha256=a.sha(obj),hand_stage_body_drift_mm=float(drift))
basis=world[0]@w.old.mat([0,np.pi/2,0])
q=np.vstack([v@basis for v,j in meshes]);center=(q.max(0)+q.min(0))/2
scale=min(316/np.ptp(q[:,0]),440/np.ptp(q[:,1]))
images=[]
for (v,j),weight in zip(meshes,(4,1)):
    im=a.free_render(v,faces,basis,center,scale)
    pts=np.rint(((j[[16,18,20]]@basis-center)[:,:2])*[scale,-scale]+[175,240]).astype(int)
    cv2.polylines(im,[pts],False,(80,140,220),2,cv2.LINE_AA)
    cv2.circle(im,tuple(pts[1]),5,(40,70,220),-1)
    images.append(a.title(im,f"wrist x{weight} | left elbow {metrics[str(weight)]['hands']['left']['angle']:.1f} deg"))
assert cv2.imwrite(str(RUN/'assets/side.jpg'),np.hstack(images))
metrics['pair']=key
a.write(RUN/'evidence/results.json',metrics)
for weight in ('4','1'):
    for side,h in metrics[weight]['hands'].items():
        print(weight,side,h['source'],round(h['angle'],2),*[round(np.sqrt(np.mean(np.square(h[k]))),2) for k in ['elbow','wrist','tips']])
