import numpy as np
import torch
from egosmplx_pipeline import v2_policy as v2


def test_edge_exemptions_and_legacy(monkeypatch):
    q = dict(x=20, y=100, score=.9)
    monkeypatch.setenv('EGOSMPLX_V2', '1')
    assert v2.body_allowed('nose', q)
    assert v2.body_allowed('left_hip', q)
    assert v2.body_allowed('right_knee', q)
    assert v2.body_allowed('left_ankle', q)
    assert not v2.body_allowed('left_wrist', q)
    assert not v2.body_allowed('right_elbow', dict(q, x=40))
    assert v2.body_allowed('right_elbow', dict(q, x=41))
    assert v2.wrist_fraction() == 0
    monkeypatch.delenv('EGOSMPLX_V2')
    assert v2.body_allowed('left_wrist', q)
    assert v2.wrist_fraction() == .85


def test_depth_mask_excludes_only_head_and_neck(monkeypatch):
    monkeypatch.setenv('EGOSMPLX_V2', '1')
    monkeypatch.setattr(v2, 'vertex_indices', lambda: (np.array([0, 3]), np.array([1]), 4))
    points = torch.tensor([[0., 0., .2], [0., 0., -.2], [0., 0., -.1], [0., 0., .1]], requires_grad=True)
    depth = v2.min_z(points)
    assert np.isclose(float(depth), .1)
    depth.backward()
    assert torch.equal(points.grad[:, 2], torch.tensor([0., 0., 0., 1.]))
    monkeypatch.delenv('EGOSMPLX_V2')
    assert np.isclose(float(v2.min_z(points)), -.2)


def test_group_does_not_dilute_other_groups_and_sparse_is_downweighted():
    wrist = v2.weighted_group(torch.tensor([4.]), torch.tensor([1.]))
    hand = v2.weighted_group(torch.full((20,), 2.), torch.ones(20), 4.)
    dense = v2.weighted_group(torch.full((40,), 2.), torch.ones(40), 4.)
    assert wrist+hand == wrist+dense
    sparse = v2.weighted_group(torch.tensor([2.]), torch.tensor([.1]), 4.)
    assert sparse < hand


def test_v2_requires_explicit_stereo_inputs(monkeypatch):
    monkeypatch.setenv('EGOSMPLX_V2', '1')
    import pytest
    with pytest.raises(ValueError, match='frozen stereo'):
        v2.Context({}, None, None)
