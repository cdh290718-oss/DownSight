"""Verify actual SMPL-X/fisheye hand-target gradients on all six fixed views."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import read,load

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/hand_stage_ablation_20261104'
RUN=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['A','C'],default='A');args=ap.parse_args()
records=read(OUT/args.group/'inputs.json')['records']
cfg=read(records[0]['config'])
os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo'],EGOSMPLX_V2='1',EGOSMPLX_V2_HAND_STAGE='arm_routed')
model,Camera=setup(cfg['paths']['egosmplx_repo'])
from egosmplx_pipeline import fusion,v2_policy as v2
from egosmplx_pipeline.body_common import BODY_POSE_NAMES
fusion.initialize(model)
layer=fusion.LAYER
arm=np.array([n.endswith(('_Collar','_Shoulder','_Elbow','_Wrist')) for n in BODY_POSE_NAMES])
rows=[]
for rec in records:
    cfg=read(rec['config']);base=Path(cfg['output'])/'frames'/rec['id']
    p=load(base/'body/params.npz');camera=Camera(rec['row']['calibration'])
    lookup={q['name']:q for q in read(rec['row']['predictions']['sapiens'])['keypoints']}
    ctx=v2.Context(lookup,camera,model)
    pp={k:torch.as_tensor(p[k],dtype=torch.float32).reshape(1,-1) for k in fusion.PARAMS}
    body=pp['body_pose'].clone().requires_grad_();orient=pp['global_orient'].clone().requires_grad_()
    tr=torch.as_tensor(p['transl'].reshape(3)).clone().requires_grad_()
    pp.update(body_pose=body,global_orient=orient)
    out=layer(**pp);v=out.vertices[0]+tr;j=out.joints[0,model.joint_idx]+tr;uv=camera.world2camera_pytorch(j)
    hv,hj,hu=v2.routed_hand_geometry(layer,pp,tr,model.joint_idx,camera.world2camera_pytorch,(v,j,uv))
    assert torch.equal(v,hv) and torch.equal(j,hj) and torch.equal(uv,hu)
    bodyloss=ctx.observation(v,j,uv,include_hands=False)
    full=ctx.observation(v,j,uv,hand_geometry=(hv,hj,hu))
    # The two identical body expressions cancel, leaving only hand gradients.
    handloss=full-bodyloss
    norm=lambda g:float(g.norm()) if g is not None else 0.
    def audit(loss):
        grads=torch.autograd.grad(loss,(body,orient,tr),retain_graph=True,allow_unused=True)
        bg=grads[0].reshape(21,3)
        result=dict(arm_gradient_norm=norm(bg[arm]),nonarm_gradient_norm=norm(bg[~arm]),global_orient_gradient_norm=norm(grads[1]),translation_gradient_norm=norm(grads[2]))
        assert result['nonarm_gradient_norm']==0 and result['global_orient_gradient_norm']==0 and result['translation_gradient_norm']==0
        assert result['arm_gradient_norm']>0
        return result
    g=audit(handloss)
    # Direct WiLoR wrist constraints and a vertex projection objective use the
    # same routed branch as reference feasibility and seam/forearm terms.
    wrist=audit((hu[[12,13]]-ctx.hand_xy[:,0]).square().mean())
    ids=np.concatenate(list(model.hand_vertex_idx.values()))
    vertex=audit(camera.world2camera_pytorch(hv[ids]).square().mean())
    bg=torch.autograd.grad(bodyloss,(body,orient,tr),retain_graph=True)
    assert norm(bg[1])>0 and norm(bg[2])>0
    ordinary=ctx.observation(v,j,uv)
    assert torch.equal(ordinary,full)
    rows.append(dict(frame=rec['id'],forward_bitwise_equal=True,objective_value_equal=True,hand_observation=g,wrist_constraint=wrist,hand_vertex_projection=vertex,body_translation_gradient_norm=norm(bg[2]),body_global_orient_gradient_norm=norm(bg[1])))
(RUN/'evidence'/('gradient_audit.json' if args.group=='A' else 'gradient_audit_C.json')).write_text(json.dumps(dict(passed=True,parameter_group=args.group,views=rows,scope='Actual SMPL-X and fisheye direct gradients; does not eliminate indirect candidate selection or rejection coupling'),indent=2)+'\n')
print('PASS: six real-model views, routed hand gradients zero for global pose and non-arm joints')
