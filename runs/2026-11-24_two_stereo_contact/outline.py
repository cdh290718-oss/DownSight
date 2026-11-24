"""Visible semantic boundaries in the two original fisheye views; hard z-buffer ray hits."""
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from common_run import OUT,prev,project

STRIDE=4
NAMES={'foreground':None,'L_hand':6,'R_hand':15,'L_forearm':7,'R_forearm':16,
       'L_upperarm':11,'R_upperarm':20,'L_thigh':12,'R_thigh':21}
MAX_PX=64.
SIGMA_PX=12.


def boundary(mask):
    image=mask.astype(np.uint8)
    edge=mask & (cv2.erode(image,np.ones((3,3),np.uint8))==0)
    edge[[0,-1],:]=False
    edge[:,[0,-1]]=False
    return edge


class Eye:
    def __init__(self,item,side,cam,adapter,labels,faces):
        self.cam,self.adapter,self.labels,self.faces=cam,adapter,labels,faces
        self.transform=np.array(item['inputs']['views'][side]['camera']['left_to_view'])
        path=OUT/'seg/frames'/(item['pair']+'_'+side)
        with np.load(path/'probabilities.npz') as pack:
            prob=pack['probabilities'][:,STRIDE//2::STRIDE,STRIDE//2::STRIDE].astype(float)
        lab=prob.argmax(0)
        self.shape=lab.shape
        yy,xx=np.indices(self.shape)
        self.pix=np.stack((xx,yy),axis=-1)*STRIDE+STRIDE//2
        directions=cam.camera2world_ray(self.pix.reshape(-1,2)*adapter['scale']+adapter['offset'])
        rays=np.zeros((len(directions),6),np.float32)
        rays[:,3:]=directions
        self.rays=o3d.core.Tensor(rays)
        self.targets={}
        self.support={}
        confidence=prob.max(0)
        for name,ident in NAMES.items():
            mask=lab!=0 if ident is None else lab==ident
            sure=(1-prob[0] if ident is None else confidence)>=.55
            edge=boundary(mask)&sure
            self.support[name]=dict(area=int((mask&sure).sum()),edge=int(edge.sum()))
            if (mask&sure).sum()>=12 and edge.sum()>=6:
                self.targets[name]=self.pix[edge].astype(float)
        self.transform_t=torch.tensor(self.transform,dtype=torch.float32)

    def compare(self,vertices,with_gradient=False):
        vv=vertices@self.transform_t[:3,:3].T+self.transform_t[:3,3]
        v=vv.detach().numpy()
        keep=np.all(v[self.faces,2]>1e-5,axis=1)
        indices=np.flatnonzero(keep)
        scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(v,np.float32)),
                            o3d.core.Tensor(self.faces[keep].astype(np.uint32)))
        hit=scene.cast_rays(self.rays,nthreads=1)
        ids=hit['primitive_ids'].numpy()
        valid=ids<len(indices)
        which=np.zeros(len(ids),int)
        which[valid]=indices[ids[valid]]
        uv=hit['primitive_uvs'].numpy()
        bary=np.column_stack((1-uv.sum(1),uv))
        distances={}
        losses=[]
        for name,target in self.targets.items():
            mask=valid if name=='foreground' else valid&(self.labels[which]==name)
            rr=np.flatnonzero(boundary(mask.reshape(self.shape)))
            if not len(rr):
                distances[name]=MAX_PX
                losses.append(vertices.new_tensor(MAX_PX/SIGMA_PX))
                continue
            src=self.pix.reshape(-1,2)[rr]
            ds,match=cKDTree(target).query(src)
            dt,reverse=cKDTree(src).query(target)
            distances[name]=float((np.minimum(ds,MAX_PX).mean()+np.minimum(dt,MAX_PX).mean())/2)
            if with_gradient:
                def direction_loss(si,goal):
                    pixels=rr[si]
                    face_ids=torch.from_numpy(self.faces[which[pixels]].astype(np.int64))
                    points=(vv[face_ids]*torch.from_numpy(bary[pixels])[:,:,None]).sum(1)
                    pp=(project(points,self.cam)-points.new_tensor(self.adapter['offset']))/self.adapter['scale']
                    # Ray display coordinates are the actual raster samples. Use the calibrated forward Jacobian.
                    residual=pp-pp.detach()+points.new_tensor(src[si]-goal)
                    return residual.norm(dim=1).clamp_max(MAX_PX).mean()/SIGMA_PX
                losses.append((direction_loss(np.arange(len(src)),target[match])+direction_loss(reverse,target))/2)
            else:
                losses.append(vertices.new_tensor(distances[name]/SIGMA_PX))
        return torch.stack(losses).mean(),distances
