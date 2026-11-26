"""Rigid MCP reprojection, bounded forearm twist, and selection with the frozen v4 outline."""
from pathlib import Path
import sys
from itertools import product
import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo4_palm_20261126'
V4=RUN.parent/'2026-11-25_realvideo4_dv4'
sys.path.insert(0,str(V4))
import refit as ref
sys.path.insert(0,str(ref.PRIOR))
from calibrate import rays,projection
read,write,sha=ref.read,ref.write,ref.sha
rot=lambda x:Rotation.from_rotvec(x).as_matrix()
rv=lambda x:Rotation.from_matrix(x).as_rotvec()
unit=lambda x:x/np.linalg.norm(x)
rms=lambda x:float(np.sqrt(np.mean(np.square(x))))
SIDES=['left','right']
DIGITS=['index','middle','ring','pinky']

def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(unit(a)@unit(b),-1,1))))

def normal(points,side):
    return unit(np.cross(points[1]-points[0],points[-1]-points[0]))*(1 if side=='left' else -1)

def triangulate(uv,cams,T):
    R,t=T[:3,:3],T[:3,3]
    o=-R.T@t
    result=[]
    for a,b in zip(rays(uv[0],cams[0]),rays(uv[1],cams[1])@R):
        depth=np.linalg.lstsq(np.stack((a,-b),axis=1),o,rcond=None)[0]
        assert min(depth)>0
        result.append((a*depth[0]+o+b*depth[1])/2)
    return np.array(result)

def world_joints(frame,x):
    z=frame.beta
    j=frame.layer(global_orient=frame.g,body_pose=x[:21].reshape(1,63),betas=z,left_hand_pose=x[23:38].reshape(1,45),
                  right_hand_pose=x[38:].reshape(1,45),jaw_pose=z[:,:3],leye_pose=z[:,:3],reye_pose=z[:,:3],expression=z).joints[0].detach().numpy()
    return (j-j[:1])@rot(x[21].numpy()).T+j[:1]+x[22].numpy()

class Hand:
    def __init__(self,side,x,j,world,uv,wilor,cams,T):
        self.side=side
        self.e,self.w=[JOINT_NAMES.index(side+'_'+k) for k in ['elbow','wrist']]
        self.ids=[self.w]+[JOINT_NAMES.index(side+'_'+k+'1') for k in DIGITS]
        self.uv,self.cams,self.T=uv,cams,T
        self.pivot=j[self.w]
        self.G=world[self.e]
        self.E=rot(x[self.e-1])
        self.W=rot(x[self.w-1])
        self.axis=unit((self.pivot-j[self.e])@self.G)
        self.local=(j[self.ids]-self.pivot)@world[self.w]
        self.tri=triangulate(uv,cams,T)
        self.stereo_err=rms(np.linalg.norm(self.project(self.tri)-uv,axis=2))
        self.width_ratio=np.linalg.norm(self.tri[1]-self.tri[-1])/np.linalg.norm(self.local[1]-self.local[-1])
        self.quality=float(np.clip(np.exp(-.5*(np.log(self.width_ratio)/.30)**2)/(1+(self.stereo_err/3)**2),.10,1))
        sources={'base':j[self.ids],'stereo':self.tri,'wilor':wilor}
        self.options={'keep':np.r_[x[self.w-1],0.]}
        for tag,src in sources.items():
            orientation=Rotation.align_vectors(src[1:]-src[0],self.local[1:])[0].as_matrix()
            seed=np.r_[rv(self.G.T@orientation),0.]
            def residual(p):
                err=(self.project(self.pose(p))-self.uv)[:,1:,:]
                reg=np.r_[rv(self.W.T@self.local_wrist(p))/.95,p[3]/.45]
                return np.r_[np.sqrt(self.quality)*err.ravel()/6,reg]
            sol=least_squares(residual,seed,bounds=([-np.pi]*3+[-np.pi/6],[np.pi]*3+[np.pi/6]),loss='soft_l1',f_scale=1.,max_nfev=150)
            self.options[tag]=sol.x
        self.wilor_normal=normal(wilor,side)

    def local_wrist(self,p):
        return rot(p[:3]*min(1.,np.deg2rad(100)/max(np.linalg.norm(p[:3]),1e-12)))

    def pose(self,p):
        return self.local@(self.G@rot(self.axis*p[3])@self.local_wrist(p)).T+self.pivot

    def project(self,q):
        return np.stack([projection(q,self.cams[0]),projection(q@self.T[:3,:3].T+self.T[:3,3],self.cams[1])])

    def update(self,x,p):
        x[self.e-1]=torch.as_tensor(rv(self.E@rot(self.axis*p[3])))
        x[self.w-1]=torch.as_tensor(rv(self.local_wrist(p)))

    def cost(self,p):
        gap=np.linalg.norm(self.project(self.pose(p))[:,1:]-self.uv[:,1:],axis=2)/6
        reg=(np.linalg.norm(rv(self.W.T@self.local_wrist(p)))/.95)**2+(p[3]/.45)**2
        return self.quality*np.mean(2*(np.sqrt(1+gap*gap)-1))+reg/16

def main(idx):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    key=f'clip{idx+1:03}'
    ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=ref.cameras
    ref.compare.OUT=ref.OUT
    row=read(ref.OLD/'fit/0/prepared.json')['items'][idx]
    detpath=ref.OLD/'predictions.json'
    assert sha(detpath)==read(ref.PRIOR/'evidence/detection.json')['predictions_sha256']
    det=read(detpath);guide=read(OUT/'guided.json')['selected']
    path=ref.OUT/'poses'/(key+'.npz')
    assert sha(path)==read(V4/'evidence'/(key+'.json'))['pose_hash']
    pack=np.load(path);x=torch.tensor(pack['locked'])
    cfg=read(ref.OLD/'fit/0/protocol.json')
    assert sha(ref.OLD/'fit/0/protocol.json')==read(ref.PRIOR/'evidence/inputs.json')['protocol']
    for side in SIDES:
        eye=key+'_'+side
        assert sha(ref.OUT/'seg'/(eye+'.npy'))==read(V4/'evidence/segmentation.json')['eyes'][eye]['sampled_sha256']
    model,Camera=ref.prev.ref.common.setup_model()
    start=dict(body_pose=x[:21].numpy(),delta=x[21].numpy(),translation=x[22].numpy(),betas=pack['betas'],global0=pack['global0'])
    frame=ref.FixedFeet(key,row,start,model,Camera,cfg)
    T=np.asarray(row['inputs']['views']['right']['camera']['left_to_view'])
    cams=list(map(read,ref.cameras()))
    j=world_joints(frame,x)
    world=[rot(x[21].numpy())@rot(pack['global0'])]
    for n,parent in enumerate(frame.layer.parents.tolist()[1:22],1):world.append(world[parent]@rot(x[n-1].numpy()))
    hands=[]
    for s,ids in zip(SIDES,[[62,49,53,57,61],[41,28,32,36,40]]):
        uv=np.stack([np.asarray(det[key+'_'+e+'.png']['xy'])[ids] for e in SIDES])
        assert ((uv>=0)&(uv<[1280,720])).all()
        lr=[np.asarray(guide[key+'_'+e][s[0].upper()]['local'])[[0,5,9,13,17]] for e in SIDES]
        wi=((lr[0]-lr[0][:1])+(lr[1]-lr[1][:1])@T[:3,:3])*.5
        hands.append(Hand(s,x.numpy(),j,world,uv,wi,cams,T))
    scores={};poses={}
    def evaluate(parameters):
        trial=x.clone()
        for h,p in zip(hands,parameters):
            if abs(p[3])>np.pi/6+1e-8:return None,None
            h.update(trial,p)
        if any(trial[h.e-1].norm().item()>np.deg2rad(165)+1e-6 for h in hands):return None,None
        with torch.no_grad():
            verts,_,body,ok=frame.state(trial)
            if not ok:return None,None
            edge=frame.observed(verts)[0].item()
        palm=np.mean([h.cost(p) for h,p in zip(hands,parameters)])
        return dict(body=float(body),edge=edge,total=float(body)+ref.WEIGHT*edge+.01*palm),trial

    for labels in product(*(h.options for h in hands)):
        tag='/'.join(labels)
        score,trial=evaluate([h.options[l] for h,l in zip(hands,labels)])
        if score is not None:scores[tag],poses[tag]=score,trial
    best=min(scores,key=lambda k:scores[k]['total'])
    no_wilor=min((k for k in scores if 'wilor' not in k),key=lambda k:scores[k]['total'])
    parameters={tag:[h.options[l].copy() for h,l in zip(hands,tag.split('/'))] for tag in scores}
    # Local descent using the actual full stereo contour and point objective together.
    for tag in dict.fromkeys((best,no_wilor)):
        par=np.array(parameters[tag])
        chosen=scores[tag]
        fitted=poses[tag]
        for step in np.deg2rad([6,3,1.5]):
            for side in range(2):
                for axis in range(4):
                    origin=par.copy()
                    for sign in (-1,1):
                        other=origin.copy();other[side,axis]+=step*sign
                        score,trial=evaluate(other)
                        if score is not None and score['total']<chosen['total']:
                            chosen,fitted,par=score,trial,other
        scores[tag+'_local'],poses[tag+'_local'],parameters[tag+'_local']=chosen,fitted,par
    no_wilor=no_wilor+'_local'
    best=min((best+'_local',no_wilor),key=lambda tag:scores[tag]['total'])
    wrist_ids=[h.w for h in hands]; details={}
    for label,tag in [('before','keep/keep'),('sapiens_only',no_wilor),('after',best)]:
        current=poses[tag];jj=world_joints(frame,current)
        details[label]={}
        assert np.max(abs(jj[:22]-j[:22]))<2e-5
        for h,p in zip(hands,parameters[tag]):
            q=jj[h.ids];assert np.max(abs(q-h.pose(p)))<1e-5
            pp=h.project(q);errors=np.linalg.norm(pp-h.uv,axis=2)
            normal_fit=normal(q,h.side)
            tip_ids=[JOINT_NAMES.index(h.side+'_'+k) for k in ['thumb']+DIGITS]
            tip_channels=np.array([21,25,29,33,37])+(21 if h.side=='left' else 0)
            tip_det=np.stack([np.asarray(det[key+'_'+e+'.png']['xy'])[tip_channels] for e in SIDES])
            tip_errors=np.linalg.norm(h.project(jj[tip_ids])-tip_det,axis=2)
            details[label][h.side]=dict(mcp_rmse=rms(errors[:,1:]),by_eye=np.sqrt(np.mean(errors[:,1:]**2,axis=1)).tolist(),
                wrist_rmse=rms(errors[:,0]),wrist_shift_mm=float(np.linalg.norm(q[0]-h.pivot)*1000),
                normal_vs_stereo=angle(normal_fit,normal(h.tri,h.side)),normal_vs_wilor=angle(normal_fit,h.wilor_normal),
                normal_change=angle(normal_fit,normal(j[h.ids],h.side)),twist_deg=float(np.rad2deg(p[3])),
                heldout_tips_rmse=rms(tip_errors),heldout_thumb_rmse=rms(tip_errors[:,0]),quality=h.quality,width_ratio=h.width_ratio,stereo_rmse=h.stereo_err,
                uv=pp.tolist(),det=h.uv.tolist())
    dst=OUT/'poses'/(key+'.npz')
    np.savez_compressed(dst,locked=poses[best].numpy(),betas=pack['betas'],global0=pack['global0'])
    write(RUN/'evidence'/(key+'.json'),dict(pair=key,best=best,sapiens_only=no_wilor,scores=scores,metrics=details,
          pose_hash=sha(dst),guided_hash=sha(OUT/'guided.json'),base_hash=sha(path),exit_status=0))
if __name__=='__main__':main(int(sys.argv[1]))
