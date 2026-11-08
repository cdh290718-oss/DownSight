"""Differentiable Brown projection and Head frame, all geometry in cam3 CV."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from egosmplx_pipeline.body_common import axis_angle_rotation
ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_shape_prior_20261108'
OLD=ROOT/'outputs/synthetic_original_20261108'
read=lambda p:json.loads(Path(p).read_text())

def head_rotation(global_orient,body_pose):
    r=axis_angle_rotation(global_orient.reshape(3))
    for j in [3,6,9,12,15]:r=r@axis_angle_rotation(body_pose.reshape(21,3)[j-1])
    return r

def head_rotation_numpy(p):
    r=cv2.Rodrigues(p['global_orient'].reshape(3).astype(float))[0]
    for j in [3,6,9,12,15]:r=r@cv2.Rodrigues(p['body_pose'].reshape(21,3)[j-1].astype(float))[0]
    return r

def angle_deg(a,b):
    # Polar projection removes float32 roundoff; atan2 stays stable near zero.
    def so3(r):
        u,_,v=np.linalg.svd(np.asarray(r,float));return u@np.diag([1,1,np.linalg.det(u@v)])@v
    r=so3(a)@so3(b).T
    sine=np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return float(np.rad2deg(np.arctan2(sine,(np.trace(r)-1)/2)))

def project_numpy(points,camera):
    t=np.array(camera['left_to_view']);p=np.asarray(points)@t[:3,:3].T+t[:3,3]
    result=np.full((len(p),2),np.nan);valid=p[:,2]>1e-6
    if valid.any():result[valid]=cv2.projectPoints(p[valid].astype(float),np.zeros(3),np.zeros(3),np.array(camera['K']),np.array(camera['distortion']))[0].reshape(-1,2)
    return result

def transform_torch(points,camera):
    t=points.new_tensor(camera['left_to_view']);return points@t[:3,:3].T+t[:3,3]

def project_torch(points,camera,epsilon=1e-4):
    p=transform_torch(points,camera);x,y=p[:,0]/p[:,2].clamp_min(epsilon),p[:,1]/p[:,2].clamp_min(epsilon)
    k1,k2,p1,p2,k3=camera['distortion'];r2=x*x+y*y;rad=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*rad+2*p1*x*y+p2*(r2+2*x*x);yd=y*rad+p1*(r2+2*y*y)+2*p2*x*y
    k=camera['K'];return torch.stack([k[0][0]*xd+k[0][2],k[1][1]*yd+k[1][2]],1)

def shoulder_excess(residual,tolerance):
    return residual*(1-tolerance/residual.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp_min(0)
