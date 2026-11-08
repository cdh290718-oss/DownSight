"""Independent FK, rotation gradient, visibility and support checks."""
import cv2
import numpy as np
import torch
from common import head_rotation, rotation_penalty, shoulder_excess
from contour import raster_faces, make_targets

parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12])
global_r=torch.tensor([.1,-.2,.05],requires_grad=True)
body=torch.zeros((21,3),requires_grad=True)
with torch.no_grad():
    body[2]=torch.tensor([.03,.02,-.01]);body[11]=torch.tensor([-.1,.02,.04])
actual=head_rotation(global_r,body,parents)
expected=cv2.Rodrigues(global_r.detach().numpy().astype(float))[0]
for j in [3,6,9,12,15]:expected=expected@cv2.Rodrigues(body[j-1].detach().numpy().astype(float))[0]
np.testing.assert_allclose(actual.detach().numpy(),expected,atol=2e-7)
rotation_penalty(actual,torch.eye(3),.05).backward()
assert torch.isfinite(global_r.grad).all() and torch.isfinite(body.grad).all()
assert body.grad[0].abs().max()==0
ident=torch.eye(3,requires_grad=True);loss=rotation_penalty(ident,torch.eye(3),.05);loss.backward();assert loss==0 and ident.grad.abs().max()==0
x=torch.tensor([[0.,0.],[30.,0.],[60.,0.]],requires_grad=True);z=shoulder_excess(x,30.);z.square().sum().backward();assert torch.isfinite(x.grad).all() and z[-1,0]==30
triangle=np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]],np.float32)
vertices=np.concatenate([triangle,triangle*2]);faces=np.array([[3,4,5],[0,1,2]],np.int32);uv=np.tile(np.array([[0,0],[2,0],[1,2]],np.float32),(2,1))
yy,xx=np.indices((3,3));rays=np.stack([xx-1,yy-1,np.ones((3,3))],-1).astype(np.float32)
f,b=raster_faces(vertices,faces,uv,rays,10.)
assert f[1,1]==1 and abs(b[1,1].sum()-1)<1e-6
labels=np.zeros((40,50),np.uint8);labels[10:35,15:35]=22;prob=np.eye(29,dtype=np.float32)[labels].transpose(2,0,1)
cfg=dict(garment_arm_exclusion_radius_px=3,contour_border_px=2,torso_classes=[22,23],leg_classes=[8,12,17,21],contour_min_probability=.55,contour_min_pixels=256,contour_min_boundary_points=32,contour_max_points=128)
targets=make_targets(labels,prob,{},cfg);assert targets['torso']['active'] and not targets['legs']['active']
print('PASS: independent Head FK, rotation/shoulder gradients, nearest-surface visibility, missing-leg gating')
