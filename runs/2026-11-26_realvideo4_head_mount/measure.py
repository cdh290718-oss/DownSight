"""Estimate one shared cam3 translation on 003, then evaluate the sealed candidate."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo4_head_mount_20261126'
PRIOR=RUN.parent/'2026-11-25_realvideo4_dv3'
V4=RUN.parent/'2026-11-25_realvideo4_dv4'
sys.path.insert(0,str(PRIOR))
from calibrate import projection,rays
from fileset import read,write,sha,cameras,OUT as OLD

cal=read(PRIOR/'evidence/calibration.json')
assert [sha(p) for p in cameras()]==cal['converted']
T=np.array(cal['T'])
cams=[read(p) for p in cameras()]
eyes=['left','right']
# With the v4 template facing forward, cam3 sees the model's anatomical left oral
# commissure clearly, while cam4 sees the right one. These are distinct landmarks.
points={'clip003':[[850,664],[485,655]]}
windows={'clip003':[(855,1005,550),(340,490,550)]}

def image(k,e):
    path=OLD/'images'/(k+'_'+eyes[e]+'.png')
    assert sha(path)==read(V4/'evidence/segmentation.json')['eyes'][k+'_'+eyes[e]]['image_sha256']
    return cv2.imread(str(path))

def observed_edge(im,window):
    x0,x1,y0=window
    x=np.arange(x0,x1+1,10)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Only use manually identified jaw-against-dark-shirt strips. Require a sustained
    # brightness transition to reject isolated cable/glare pixels.
    ys=[]
    for u in x:
        on=(np.median(gray[y0:,u-1:u+2],axis=1)>90).astype(int)
        hits=np.where(np.convolve(on,np.ones(4,int),'valid')==4)[0]
        if not len(hits):raise ValueError((int(u),window))
        ys.append(y0+int(hits[0]))
    return np.column_stack((x,ys))

class Case:
    def __init__(self,key,mesh_folder=OUT):
        z=np.load(mesh_folder/(key+'.npz'))
        self.head=z['head']
        self.landmarks=z['lm'][[list(z['names']).index(name) for name in ('left_mouth_1','right_mouth_1')]]
        self.labels=z['head_faces']
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(z['v'],np.float32)),
                                 o3d.core.Tensor(np.ascontiguousarray(z['f'],np.uint32)))
        self.images=[image(key,i) for i in range(2)]
        self.corners=np.array(points[key]) if key in points else None
        self.edges=[observed_edge(im,win) for im,win in zip(self.images,windows[key])] if key in windows else None
        if self.edges is not None:
            self.columns=[self.beams(np.stack(np.meshgrid(uv[:,0],np.arange(350,720)),axis=-1).reshape(-1,2),i)
                          for i,uv in enumerate(self.edges)]

    def beams(self,uv,i):
        directions=rays(uv,cams[i])
        center=np.zeros(3)
        if i:
            center=-T[:3,:3].T@T[:3,3]
            directions=directions@T[:3,:3]
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        return center,directions

    def hit(self,beam,d):
        center,dirs=beam
        q=np.empty((len(dirs),6),np.float32)
        q[:,:3]=center-d
        q[:,3:]=dirs
        ids=self.scene.cast_rays(o3d.core.Tensor(q),nthreads=1)['primitive_ids'].numpy()
        visible=ids<len(self.labels)
        visible[visible]=self.labels[ids[visible]]
        return visible

    def mouth(self,d,offset=None):
        pts=self.landmarks+d if offset is None else offset+d
        return np.vstack((projection(pts[:1],cams[0]),projection(pts[1:]@T[:3,:3].T+T[:3,3],cams[1])))

    def edge_y(self,d):
        pred=[]
        for beam in self.columns:
            mask=self.hit(beam,d).reshape(370,-1)
            pred.append(np.where(mask.any(0),350+mask.argmax(0),720))
        return pred

    def errors(self,d):
        mouth=self.mouth(d)-self.corners
        detected=self.edge_y(d)
        edges=np.concatenate([y-uv[:,1] for y,uv in zip(detected,self.edges)])
        return dict(mouth_reprojected=self.mouth(d).tolist(),mouth_observed=self.corners.tolist(),
                    mouth_rmse_px=float(np.sqrt(np.mean(np.sum(mouth**2,axis=1)))),
                    edge_mae_px=float(np.abs(edges).mean()),edge_observed=[uv.tolist() for uv in self.edges],
                    edge_pred_y=[a.tolist() for a in detected])

def fit():
    case=Case('clip003')
    def mouthfit(uv,lm=None):
        return least_squares(lambda d:(case.mouth(d,lm)-uv).ravel(),[0.,0.,0.],bounds=(-.18,.18)).x
    seed=mouthfit(case.corners)
    # Mean pixel errors of the corner and the visible cropped face boundary have equal weight.
    def loss(d):
        err=case.errors(d)
        return err['mouth_rmse_px']+err['edge_mae_px']
    best=seed.copy()
    for mm in [4,2,1,.5,.25]:
        for _ in range(25):
            values=[(loss(best),best)]
            for delta in np.r_[np.eye(3),-np.eye(3)]*mm/1000:
                values.append((loss(best+delta),best+delta))
            value,next_d=min(values,key=lambda v:v[0])
            if np.array_equal(next_d,best):break
            best=next_d
    poses={}
    for deg in [-10,10]:
        for axis,v in zip('xyz',np.eye(3)):
            q=(case.landmarks-case.head)@Rotation.from_rotvec(np.deg2rad(deg)*v).as_matrix().T+case.head
            poses[axis+str(deg)]=mouthfit(case.corners,q).tolist()
    perturb=[mouthfit(case.corners+e.reshape(2,2)*3) for e in np.r_[np.eye(4),-np.eye(4)]]
    old=np.array(read(PRIOR/'evidence/inputs.json')['head_cam3_m'])
    exact0=np.zeros(3)
    out=dict(fit_key='clip003',fit_source_pose_sha256=sha(ROOT/'outputs/realvideo4_dv4_20261125/poses/clip003.npz'),
             anchor_old_cam3_m=old.tolist(),anchor_candidate_cam3_m=(old+best).tolist(),
             delta_cam3_m=best.tolist(),actual_v4_head_cam3_m=case.head.tolist(),mouth_only_delta_m=seed.tolist(),
             before=case.errors(exact0),mouth_only=case.errors(seed),after=case.errors(best),
             sensitivity=dict(orientation_mouth_only_delta_m=poses,corner_3px_max_delta_change_mm=float(1000*np.max(np.linalg.norm(perturb-seed,axis=1)))),
             angular_axes='cam3 x,y,z',exit_status=0)
    write(RUN/'evidence/fit003.json',out)
    print('fit',best,'corner',out['before']['mouth_rmse_px'],out['after']['mouth_rmse_px'],'edge',out['before']['edge_mae_px'],out['after']['edge_mae_px'],flush=True)

if __name__=='__main__':fit()
