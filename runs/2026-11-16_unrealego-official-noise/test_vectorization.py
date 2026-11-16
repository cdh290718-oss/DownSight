import json
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from egosmplx_pipeline.camera import FishEyeCameraCalibrated
root=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent;arch=root.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';cams=[FishEyeCameraCalibrated(arch/'body_calibration_fisheye'/('camera_'+s+'.json')) for s in ['left','right']];rng=np.random.default_rng(1357);x=rng.normal(0,.12,(15,3)).astype('float32');x[:,2]+=.9;y=rng.normal(600,90,(2,15,2)).astype('float32');target=rng.normal(0,.1,(15,3)).astype('float32'); white=np.tile(np.eye(3,dtype='float32')[None],(15,1,1));
def loss(p,batch):
 terms=[]
 for v in range(2):
  if batch:
   z=cams[v].world2camera_pytorch(p);d=(z-torch.tensor(y[v])).norm(dim=1)/(8*np.sqrt(2));terms.append(F.huber_loss(d,torch.zeros_like(d),delta=2.5,reduction='mean'))
  else:
   vals=[]
   for j in range(15):
    z=cams[v].world2camera_pytorch(p[j:j+1])[0];d=(z-torch.tensor(y[v,j])).norm()/(8*np.sqrt(2));vals.append(F.huber_loss(d,torch.zeros_like(d),delta=2.5))
   terms.append(torch.stack(vals).mean())
 if batch:
  rr=torch.einsum('nij,nj->ni',torch.tensor(white),p-torch.tensor(target));d=rr.norm(dim=1)/np.sqrt(3);terms.append(F.huber_loss(d,torch.zeros_like(d),delta=2.5,reduction='mean'))
 else:
  vals=[]
  for j in range(15):
   rr=torch.tensor(white[j])@(p[j]-torch.tensor(target[j]));d=rr.norm()/np.sqrt(3);vals.append(F.huber_loss(d,torch.zeros_like(d),delta=2.5))
  terms.append(torch.stack(vals).mean())
 return sum(terms)
a=torch.tensor(x,requires_grad=True);b=torch.tensor(x,requires_grad=True);la=loss(a,False);lb=loss(b,True);la.backward();lb.backward();ld=float(abs(la.item()-lb.item()));gd=float((a.grad-b.grad).abs().max());assert ld<1e-4 and gd<1e-4,(ld,gd);res={'exit_status':0,'test':'15 non-axis points, two cameras, 2D and 3D whitened Huber residual','loss_difference':ld,'max_gradient_difference':gd,'axis_limit':'test excludes exact camera optical axis; production handles output according to fisheye backend'};(run/'evidence/vectorization_gradient.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
