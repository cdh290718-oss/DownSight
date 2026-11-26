"""Stereo MANO refit at the current body wrist with fingertip holdout."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import cv2, torch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from wilor.models import load_wilor
from smplx.lbs import batch_rodrigues
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/wilor_stereo_fingers_20261126'
DATA=ROOT/'outputs/realvideo4_dv3_20261125'
CURRENT=RUN.parent/'2026-11-26_purple_arm_hand_a'
sys.path.insert(0,str(RUN.parent/'2026-11-25_realvideo4_dv3'))
from calibrate import projection
read=lambda p:json.loads(p.read_text())
tensor=lambda a:torch.as_tensor(a,dtype=torch.float32)
array=lambda a:a.detach().cpu().numpy()
def project(x,cam):
    rho=x[:,:2].norm(dim=-1).clamp_min(1e-9)
    theta=torch.atan2(-x[:,2],rho)
    radius=theta*0
    for c in reversed(cam['polynomialW2C']):
        radius=radius*theta+c
    c,d,e=cam['affine']
    return (x[:,:2]*(radius/rho)[:,None])@tensor([[c,e],[d,1]])+tensor(cam['image_center'])
def raster(uv,faces):
    mask=np.zeros((180,320),np.uint8)
    for tri in np.rint((uv[faces]-2)/4*16).astype(np.int32):
        cv2.fillConvexPoly(mask,tri,1,shift=4)
    return mask
def boundary(m):
    return np.column_stack(np.where(m-cv2.erode(m,np.ones((3,3),np.uint8))))[:,::-1]*4+2
def rms(d):
    return float(np.sqrt(np.mean((d*d).sum(-1))))
def outline(a,b):
    return float(np.mean([np.minimum(cKDTree(x).query(y)[0],64).mean() for x,y in [(a,b),(b,a)]]))
tips=[4,8,12,16,20]
train=[i for i in range(1,21) if i not in tips]
eyes=['left','right']
cams=[read(DATA/(eye+'_cal.json')) for eye in eyes]
cfg=read(ROOT/'configs/pipeline.head01.developer.json')['paths']
model,_=load_wilor(cfg['wilor_checkpoint'],cfg['wilor_config'])
mano=model.mano.cpu().eval().requires_grad_(False)
faces=np.asarray(mano.faces)
torch.set_num_threads(1)
dets=read(DATA/'predictions.json')
rows=read(DATA/'fit/0/prepared.json')['items']
result={}
def experiment(key,side,panels):
    label=side[0].upper()
    pack=[dict(np.load(OUT/(key+'_'+eye+'_'+label+'.npz'))) for eye in eyes]
    cur=read(CURRENT/'evidence'/(key+'.json'))['results']['control']['hands'][side]
    current=np.array(cur['uv'])
    target=np.array([np.asarray(dets[key+'_'+eye+'.png']['xy'])[np.array([41]+[v for base in range(21,41,4) for v in range(base+3,base-1,-1)])+(21 if side=='left' else 0)] for eye in eyes])
    assert np.allclose(target[:,[0,5,9,13,17]+tips],cur['target'],atol=.001)
    row=next(r for r in rows if r['pair']==key)
    transforms=[np.array(row['inputs']['views'][eye]['camera']['left_to_view']) for eye in eyes]
    def stereo_np(x):
        return np.stack([projection(x@t[:3,:3].T+t[:3,3],c) for t,c in zip(transforms,cams)])
    wrist=least_squares(lambda x:(stereo_np(x[None])[:,0]-current[:,0]).ravel(),[0,0,.5]).x
    assert np.max(abs(stereo_np(wrist[None])[:,0]-current[:,0]))<.1
    inside=(target[:,:,0]>=0)&(target[:,:,0]<1280)&(target[:,:,1]>=0)&(target[:,:,1]<720)
    # Use only observed training joints to set a loose hand-size envelope.
    reach=np.median(np.linalg.norm(target[:,[5,9,13,17]]-target[:,:1],axis=-1),axis=1)
    usable=inside & (np.linalg.norm(target-target[:,:1],axis=-1)<3.2*reach[:,None])
    edges=[]
    for eye in eyes:
        seg=np.load(ROOT/'outputs/realvideo4_dv4_20261125/seg'/(key+'_'+eye+'.npy')).argmax(0)
        edges.append(boundary((seg==(6 if side=='left' else 15)).astype(np.uint8)))
    p=pack[0]
    base=tensor(np.concatenate([p['global_orient'].reshape(1,3,3),p['hand_pose'].reshape(15,3,3)]))
    delta=torch.zeros((16,3),requires_grad=True)
    def forward():
        rot=(batch_rodrigues(delta)@base)[None]
        hand=mano(global_orient=rot[:,:1],hand_pose=rot[:,1:],betas=tensor(p['betas']).reshape(1,-1),pose2rot=False)
        cloud=torch.cat([hand.joints[0,:21],hand.vertices[0]])-hand.joints[0,0]
        cloud=cloud*tensor([-1 if side=='left' else 1,1,1])+tensor(wrist)
        return torch.stack([project(cloud@tensor(t[:3,:3].T)+tensor(t[:3,3]),c) for t,c in zip(transforms,cams)])
    def score(joints,verts):
        return dict(wrist=rms(joints[:,0]-target[:,0]),tips=rms(joints[:,tips]-target[:,tips]),joints=rms(joints[:,train]-target[:,train]),edge=float(np.mean([outline(boundary(raster(v,faces)),e) for v,e in zip(verts,edges)])),per_eye_tips=(np.linalg.norm(joints[:,tips]-target[:,tips],axis=-1)).tolist())
    nativej=np.array([q['uv'] for q in pack]); nativev=np.array([q['verts_uv'] for q in pack])
    start=array(forward()); opt=torch.optim.Adam([delta],lr=.025)
    best=float('inf')
    for i in range(101):
        uv=forward(); joint,vert=uv[:,:21],uv[:,21:]
        err=(joint[:,train]-tensor(target[:,train])).norm(dim=-1)
        robust=lambda a:(torch.sqrt(a.square()+36)-6).mean()
        silhouette=[]
        for n,(v,edge) in enumerate(zip(vert,edges)):
            vn=array(v); contour=boundary(raster(vn,faces))
            # Associate raster boundary samples with visible outline vertices in image space.
            ids=np.unique(cKDTree(vn).query(contour)[1])
            closest=edge[cKDTree(edge).query(vn[ids])[1]]
            back=ids[cKDTree(vn[ids]).query(edge)[1]]
            silhouette.extend([robust((v[ids]-tensor(closest)).norm(dim=1)),robust((v[back]-tensor(edge)).norm(dim=1))])
        loss=robust(err[torch.from_numpy(usable[:,train])])+.6*torch.stack(silhouette).mean()+5*delta.square().sum(-1).mean()
        if loss.item()<best:
            best=loss.item(); chosen=array(uv).copy(); selected=array(delta).copy()
        opt.zero_grad(); loss.backward(); opt.step()
    np.savez_compressed(OUT/(key+'_'+label+'_final.npz'),base_rot=array(base),rot_delta=selected,betas=p['betas'],wrist=wrist)
    stages=dict(native=score(nativej,nativev),shared_seed=score(start[:,:21],start[:,21:]),constrained=score(chosen[:,:21],chosen[:,21:]),smplx=dict(tips=rms(current[:,5:]-target[:,tips]),wrist=rms(current[:,0]-target[:,0])))
    stages['point_check']=dict(rejected_train=[[i for i in train if not flag[i]] for flag in usable],out_of_view_tips=[[i for i in tips if not flag[i]] for flag in inside],target=target.tolist(),usable=usable.tolist())
    for name,pred in [('native',nativej[:,tips]),('shared_seed',start[:,tips]),('constrained',chosen[:,tips]),('smplx',current[:,5:])]:
        diff=pred-target[:,tips]
        stages[name]['visible_frame_tip_rmse']=rms(diff[inside[:,tips]])
        stages[name]['in_frame_count']=int(inside[:,tips].sum())
    for n,eye in enumerate(eyes):
        im=cv2.imread(str(DATA/'images'/(key+'_'+eye+'.png')))
        orig=im.copy(); pts=target[n]
        xy=np.vstack([pts[usable[n]],nativev[n],chosen[n,21:]])
        span=min(550,int(np.ptp(xy,axis=0).max()+32)); a,b=np.clip(np.rint((xy.max(0)+xy.min(0)-span)/2).astype(int),[0,0],[1280-span,720-span])
        tiles=[]
        prev=cv2.imread(str(CURRENT/'assets'/(key+'_cam'+str(n+3)+'_combined.jpg')))[32:]
        for name,v,j in [('Image',None,None),('WiLoR native',nativev[n],nativej[n]),('Shared wrist',start[n,21:],start[n,:21]),('Stereo fit',chosen[n,21:],chosen[n,:21]),('Current SMPL-X',None,None)]:
            im=orig.copy() if name!='Current SMPL-X' else prev.copy()
            if v is not None:
                mask=cv2.resize(raster(v,faces), (1280,720),interpolation=cv2.INTER_NEAREST)>0
                im[mask]=np.rint(.72*im[mask]+.28*np.array([95,180,230])).astype(np.uint8)
                contours,_=cv2.findContours(raster(v,faces),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im,[c*4+2 for c in contours],-1,(0,180,255),1)
                for g in [1,5,9,13,17]:
                    cv2.polylines(im,[np.rint(j[[0]+list(range(g,g+4))]).astype(np.int32)],False,(0,160,255),1)
                for q in pts:
                    cv2.drawMarker(im,tuple(np.rint(q).astype(int)),(255,220,0),cv2.MARKER_CROSS,6,1)
            bar=np.zeros((23,200,3),np.uint8)
            cv2.putText(bar,label+' c'+str(n+3)+' '+name,(3,16),0,.44,(255,255,255),1)
            tiles.append(np.vstack([bar,cv2.resize(im[b:b+span,a:a+span],(200,200))]))
        panels.append(np.hstack(tiles))
    return stages
for key in ['clip003','clip004']:
    panels=[]
    result[key]={side:experiment(key,side,panels) for side in eyes}
    assert cv2.imwrite(str(RUN/'assets'/(key+'.jpg')),np.vstack(panels))
(RUN/'evidence/metrics.json').write_text(json.dumps(result,indent=2))
