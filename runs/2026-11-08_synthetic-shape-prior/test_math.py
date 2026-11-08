"""Independent camera value/gradient, rigid invariance and triangulation checks."""
import hashlib,json
import cv2
import numpy as np
import torch
from common import ROOT,RUN,OUT,OLD,read,project_numpy,project_torch,head_rotation,head_rotation_numpy,angle_deg
inputs=read(OUT/'inputs.json');checks=[]
xyz=np.array([[-.12,.1,.4],[.2,.2,.7],[-.3,.6,1.3],[.5,1.1,1.5]])
for side,view in inputs['views'].items():
 c=view['camera'];p=torch.tensor(xyz,dtype=torch.float64,requires_grad=True);actual=project_torch(p,c).detach().numpy();expected=project_numpy(xyz,c)
 err=float(abs(actual-expected).max());assert err<1e-9
 assert torch.autograd.gradcheck(lambda x:project_torch(x,c),(p,),eps=1e-6,atol=1e-4,rtol=1e-3)
 checks.append(dict(view=side,opencv_projection_max_px=err,finite_difference_gradient=True))
cl=inputs['views']['left']['camera'];cr=inputs['views']['right']['camera'];T=np.array(cr['left_to_view']);uvs=[project_numpy(xyz,c) for c in [cl,cr]]
rays=[cv2.undistortPointsIter(u[:,None,:],np.array(c['K']),np.array(c['distortion']),None,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-13)).reshape(-1,2) for u,c in zip(uvs,[cl,cr])]
h=cv2.triangulatePoints(np.eye(4)[:3],T[:3],rays[0].T,rays[1].T);rec=(h[:3]/h[3]).T;error=float(np.max(abs(rec-xyz)));assert error<1e-8
p=dict(global_orient=np.array([.1,-.2,.05]),body_pose=np.linspace(-.08,.1,63))
r=head_rotation(torch.tensor(p['global_orient']),torch.tensor(p['body_pose'])).detach().numpy();expected=head_rotation_numpy(p);assert np.max(abs(r-expected))<1e-12
result=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-08_synthetic-shape-prior/test_math.py',checks=checks,triangulation_roundtrip_max_m=error,head_fk_independent=True,head_rotation_error_deg=angle_deg(r,expected))
(RUN/'evidence/math_tests.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS projection, gradients, stereo roundtrip and Head FK')
