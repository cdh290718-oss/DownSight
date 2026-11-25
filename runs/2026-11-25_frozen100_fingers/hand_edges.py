"""Remove edges after boundary extraction so the exclusion never introduces a cut edge."""
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from geometry_inputs import NAMES,STRIDE,boundary,MAX_PX,SIGMA_PX
from common_run import project

HANDS=('L_hand','R_hand')
def expand(mask,radius):
    kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
    return cv2.dilate(mask.astype(np.uint8),kernel).astype(bool)


class MaskedEye:
    def __init__(self,eye,mode,lab,digit_faces,start):
        self.eye=eye
        self.mode=mode
        self.hand_faces=np.isin(eye.labels,HANDS)
        self.excluded=self.hand_faces if mode=='no_hand' else digit_faces
        visible,faces,_=self.raster(start@eye.transform_t[:3,:3].T+eye.transform_t[:3,3])
        observed_near=expand(np.isin(lab,[NAMES[n] for n in HANDS]),1)
        allowed=np.zeros(eye.shape,bool)
        if mode=='palm':
            digit_near=expand((visible & digit_faces[faces]).reshape(eye.shape),2)
            for side in HANDS:
                side_palm=(visible & (eye.labels[faces]==side) & ~digit_faces[faces]).reshape(eye.shape)
                allowed |= expand(lab==NAMES[side],1) & expand(side_palm,2) & ~digit_near
        self.obs_ok=~observed_near | allowed
        self.targets={}
        self.counts={}
        for name,target in eye.targets.items():
            col,row=((target.astype(int)-STRIDE//2)//STRIDE).T
            keep=self.obs_ok[row,col]
            if name in HANDS and mode=='no_hand':keep[:]=False
            if name in HANDS and mode=='palm':
                own_palm=(visible & (eye.labels[faces]==name) & ~digit_faces[faces]).reshape(eye.shape)
                keep &= expand(own_palm,2)[row,col]
            self.counts[name]=[len(target),int(keep.sum())]
            if keep.sum()>=6:self.targets[name]=target[keep]
    def raster(self,vv):
        eye=self.eye
        verts=vv.detach().numpy()
        front=np.all(verts[eye.faces,2]>1e-5,axis=1)
        ids=np.flatnonzero(front)
        scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(verts,np.float32)),o3d.core.Tensor(eye.faces[front].astype(np.uint32)))
        hit=scene.cast_rays(eye.rays,nthreads=1)
        local=hit['primitive_ids'].numpy()
        valid=local<len(ids)
        which=np.zeros(len(local),int)
        which[valid]=ids[local[valid]]
        uv=hit['primitive_uvs'].numpy()
        return valid,which,np.column_stack((1-uv.sum(1),uv))

    def compare(self,vertices,with_gradient=False):
        eye=self.eye
        vv=vertices@eye.transform_t[:3,:3].T+eye.transform_t[:3,3]
        valid,which,bary=self.raster(vv)
        blocked=expand((valid & self.excluded[which]).reshape(eye.shape),1)
        losses=[];metric={}
        for name,target in self.targets.items():
            region=valid if name=='foreground' else valid & (eye.labels[which]==name)
            rr=np.flatnonzero(boundary(region.reshape(eye.shape)) & ~blocked & self.obs_ok)
            if not len(rr):
                losses.append(vertices.new_tensor(MAX_PX/SIGMA_PX));metric[name]=MAX_PX
                continue
            src=eye.pix.reshape(-1,2)[rr]
            ds,forward=cKDTree(target).query(src)
            dt,reverse=cKDTree(src).query(target)
            metric[name]=float((np.minimum(ds,MAX_PX).mean()+np.minimum(dt,MAX_PX).mean())/2)
            if with_gradient:
                def side(si,goal):
                    pixels=rr[si]
                    ids=torch.from_numpy(eye.faces[which[pixels]].astype(np.int64))
                    points=(vv[ids]*torch.from_numpy(bary[pixels])[:,:,None]).sum(1)
                    pp=(project(points,eye.cam)-points.new_tensor(eye.adapter['offset']))/eye.adapter['scale']
                    return (pp-pp.detach()+points.new_tensor(src[si]-goal)).norm(dim=1).clamp_max(MAX_PX).mean()/SIGMA_PX
                losses.append((side(np.arange(len(src)),target[forward])+side(reverse,target))/2)
            else:losses.append(vertices.new_tensor(metric[name]/SIGMA_PX))
        return (torch.stack(losses).mean() if losses else vertices.sum()*0),metric
