from types import SimpleNamespace
import torch
from egosmplx_pipeline import v2_policy as v2
from egosmplx_pipeline.body_common import BODY_POSE_NAMES


def test_hand_branch_preserves_values_but_blocks_nonarm_gradients(monkeypatch):
    monkeypatch.setenv('EGOSMPLX_V2', '1')
    monkeypatch.setenv('EGOSMPLX_V2_HAND_STAGE', 'arm_routed')
    body = torch.linspace(.1, 1., 63).reshape(1, 63).requires_grad_()
    orient = torch.tensor([[.1, .2, .3]], requires_grad=True)
    translation = torch.tensor([.2, .3, 1.], requires_grad=True)
    def layer(body_pose, global_orient):
        points = torch.sin(body_pose.reshape(21, 3)) + global_orient
        return SimpleNamespace(vertices=points[None], joints=points[None])
    project = lambda points: points[:, :2] / points[:, 2:]
    parameters = dict(body_pose=body, global_orient=orient)
    points = layer(**parameters).vertices[0]+translation
    fallback = (points, points, project(points))
    routed = v2.routed_hand_geometry(layer, parameters, translation, list(range(21)), project, fallback)
    assert all(torch.equal(a,b) for a,b in zip(routed, fallback))
    grads = torch.autograd.grad(routed[2].square().sum(), (body, orient, translation), allow_unused=True)
    assert grads[1] is None and grads[2] is None
    for n,g in zip(BODY_POSE_NAMES,grads[0].reshape(21,3)):
        if n.endswith(('_Collar','_Shoulder','_Elbow','_Wrist')):
            assert g.abs().sum() > 0
        else:
            assert torch.equal(g,torch.zeros_like(g))
    # Ordinary body branch still reaches overall pose.
    ordinary = torch.autograd.grad(fallback[2].square().sum(), (orient,translation))
    assert all(g.abs().sum() > 0 for g in ordinary)


def test_body_only_observations_keep_sapiens_wrists_without_hand_dependencies():
    ctx = object.__new__(v2.Context)
    ctx.active = True
    ctx.body_ids = torch.tensor([0,1])
    ctx.body_xy = torch.zeros(2,2)
    ctx.body_weights = torch.ones(2)
    ctx.body_groups = [torch.tensor([0]),torch.tensor([1])]
    pixels = torch.tensor([[20.,0.],[0.,20.]],requires_grad=True)
    loss = ctx.observation(None,None,pixels,include_hands=False)
    assert torch.allclose(loss, .02*torch.log(torch.tensor(2.)))
    loss.backward()
    assert pixels.grad[0,0] > 0 and pixels.grad[1,1] > 0
    assert set(ctx.last_terms) == {'body_2d','sapiens_wrist_2d'}
