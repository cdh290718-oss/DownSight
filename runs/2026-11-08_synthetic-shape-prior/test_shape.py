"""Validate beta differentiation through root-pivot transform and prior gradient."""
import json,os
import numpy as np
import torch
from common import ROOT,RUN,OUT,OLD,read,project_torch
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import axis_angle_rotation
cfg=read(OLD/'configs/left.json');paths=cfg['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo']);layer=model.layer['neutral'].cpu().double().eval();layer.requires_grad_(False)
with np.load(OLD/'fits/left/frames/left/raw/params.npz') as z:p={k:torch.tensor(z[k].copy(),dtype=torch.float64).reshape(1,-1) for k in ['betas','global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']}
beta=p.pop('betas').clone().requires_grad_();R=axis_angle_rotation(torch.tensor([.1,-.05,.07],dtype=torch.float64));tr=torch.tensor([.01,.02,.6],dtype=torch.float64)
def geometry(beta):
 o=layer(betas=beta,**p);pivot=o.joints[0,0];points=o.joints[0,[0,15,18,19,20,21]]
 return (points-pivot)@R.T+pivot+tr
assert torch.autograd.gradcheck(geometry,(beta,),eps=1e-6,atol=1e-5,rtol=1e-3)
w=.001;loss=w*beta.square().mean();g=torch.autograd.grad(loss,beta)[0];assert torch.allclose(g,2*w*beta/beta.numel(),atol=1e-12)
# Shared root/shape reconstruction should respond to beta changes, not detached caches.
change=float((geometry(beta+.1)-geometry(beta)).abs().max());assert change>1e-5
result=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-08_synthetic-shape-prior/test_shape.py',beta_root_pivot_finite_difference_pass=True,prior_gradient_matches_analytic=True,beta_perturbation_max_joint_change_m=change)
(RUN/'evidence/shape_tests.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS beta geometry finite differences and shape-prior gradient')
