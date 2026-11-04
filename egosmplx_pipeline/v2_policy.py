"""Opt-in stereo observations for the original per-view fitting stages.

No MANO placement, finger fitting or new wrist-alignment stage is implemented.
The legacy path is unchanged when EGOSMPLX_V2 is absent.
"""
import json
import os
from functools import lru_cache
import numpy as np
import torch


def enabled():
    return os.environ.get('EGOSMPLX_V2') == '1'


EXEMPT = {'nose', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
          'left_ankle', 'right_ankle'}


def interior(xy, size=(1280, 720), margin=40):
    """Historical v2 report mask only; current fitting uses in_image."""
    xy = np.asarray(xy)
    return (np.isfinite(xy).all(axis=-1) & (xy[..., 0] > margin)
            & (xy[..., 0] < size[0]-margin) & (xy[..., 1] > margin)
            & (xy[..., 1] < size[1]-margin))


def in_image(xy, size=(1280, 720)):
    xy = np.asarray(xy)
    return (np.isfinite(xy).all(axis=-1) & (xy[..., 0] >= 0)
            & (xy[..., 0] < size[0]) & (xy[..., 1] >= 0)
            & (xy[..., 1] < size[1]))


def body_allowed(name, point, size=(1280, 720)):
    # Confidence is checked by each caller; no extra border exclusion.
    return not enabled() or bool(in_image([point['x'], point['y']], size))


@lru_cache(maxsize=1)
def vertex_indices():
    from utils.human_models import smpl_x
    dominant = smpl_x.layer['neutral'].lbs_weights.detach().cpu().numpy().argmax(1)
    head = dominant == smpl_x.orig_joints_name.index('Head')
    neck = dominant == smpl_x.orig_joints_name.index('Neck')
    if not head.any() or not neck.any():
        raise ValueError('Missing head/neck assignment')
    return np.flatnonzero(~(head | neck)), np.flatnonzero(head), len(dominant)


def min_z(vertices):
    if not enabled():
        return vertices[:, 2].min()
    ids, _, count = vertex_indices()
    if tuple(vertices.shape) != (count, 3):
        raise ValueError('v2 depth mask requires preserved SMPL-X vertex indexing')
    if isinstance(vertices, torch.Tensor):
        ids = torch.as_tensor(ids, device=vertices.device)
    return vertices[ids, 2].min()


def wrist_fraction():
    return 0. if enabled() else .85


def hand_rmse(prediction, target):
    mask = in_image(target) if enabled() else np.ones(len(target), bool)
    if not mask.any():
        return 0.  # No optimization contribution; evaluator records missing as null.
    return float(np.sqrt(np.mean(np.sum((prediction[mask]-target[mask])**2, axis=-1))))


def wrist_mask(target):
    if not enabled():
        return torch.ones(target.shape[0], device=target.device, dtype=target.dtype)
    return (torch.isfinite(target).all(dim=-1)
            & (target[:, 0] >= 0) & (target[:, 0] < 1280)
            & (target[:, 1] >= 0) & (target[:, 1] < 720)).to(target.dtype)


def weighted_group(errors, weights, minimum_support=1.):
    """Normalize within a group; sparse/low confidence groups are not amplified."""
    return (errors*weights).sum()/weights.sum().clamp_min(minimum_support)


def hand_stage():
    mode = os.environ.get('EGOSMPLX_V2_HAND_STAGE', 'full')
    if mode not in ('full', 'body_rigid', 'arm_routed'):
        raise ValueError('Unknown hand-stage ablation mode: ' + mode)
    return mode if enabled() else 'full'


def routed_hand_geometry(layer, parameters, translation, joint_indices, project, fallback):
    """Same geometry values; hand-target gradients reach arm rotations only.

    Body/head objectives use the ordinary forward pass. Candidate ranking and
    feasibility rejection can still couple objectives indirectly; global pose
    is not frozen. All non-arm parameters are detached on this branch.
    """
    if hand_stage() != 'arm_routed' or not torch.is_grad_enabled():
        return fallback
    from egosmplx_pipeline.body_common import BODY_POSE_NAMES
    body = parameters['body_pose'].reshape(21, 3)
    mask = body.new_tensor([n.endswith(('_Collar', '_Shoulder', '_Elbow', '_Wrist'))
                            for n in BODY_POSE_NAMES]).reshape(21, 1)
    routed = {k: v.detach() for k, v in parameters.items()}
    routed['body_pose'] = (body.detach()+(body-body.detach())*mask).reshape(1, -1)
    output = layer(**routed)
    vertices = output.vertices[0]+translation.detach()
    joints = output.joints[0, joint_indices]+translation.detach()
    return vertices, joints, project(joints)


class Context:
    def __init__(self, lookup, camera, model, device='cpu'):
        self.active = enabled()
        if not self.active:
            return
        from egosmplx_pipeline.body_common import JOINT_MAPPING
        from egosmplx_pipeline.camera import FishEyeCameraCalibrated
        metadata = lookup.get('_v2_context')
        if metadata is None:
            raise ValueError('v2 requires frozen stereo observations; refusing silent fallback')
        with open(metadata['path']) as stream:
            self.data = json.load(stream)
        if self.data.get('point_filter') != 'in_image_no_margin':
            raise ValueError('v2 border policy changed: regenerate stereo observations with in_image_no_margin')
        self.camera = camera
        self.tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
        self.ji = {n: i for i, n in enumerate(model.joints_name)}
        self.body_names = [n for n in JOINT_MAPPING if n in lookup and
                           lookup[n]['score'] >= .3 and
                           np.isfinite([lookup[n]['x'], lookup[n]['y'], lookup[n]['score']]).all()
                           and 0 <= lookup[n]['x'] < camera.img_size[0]
                           and 0 <= lookup[n]['y'] < camera.img_size[1]
                           and body_allowed(n, lookup[n], camera.img_size)]
        self.body_ids = torch.tensor([self.ji[JOINT_MAPPING[n]] for n in self.body_names], device=device)
        self.body_xy = self.tensor([[lookup[n]['x'], lookup[n]['y']] for n in self.body_names])
        self.body_weights = self.tensor([min(1., lookup[n]['score'])**2 for n in self.body_names])
        self.body_groups = [torch.tensor([i for i, n in enumerate(self.body_names)
                                        if n.endswith('_wrist') == is_wrist], device=device, dtype=torch.long)
                            for is_wrist in (False, True)]
        self.hand_regressor = self.tensor(np.stack([model.orig_hand_regressor[s] for s in ('left', 'right')]))
        self.hand_xy = self.tensor(self.data['hand_xy'])
        self.hand_weights = self.tensor(self.data['hand_weights'])
        self.tri = self.tensor(self.data['tri'])
        self.whiten = self.tensor(self.data['whiten'])
        self.tri_weights = self.tensor(self.data['tri_weights'])
        self.head_ids = torch.tensor(vertex_indices()[1], device=device)
        self.peer_camera = FishEyeCameraCalibrated(self.data['peer_calibration'])
        self.peer_R = self.tensor(self.data['peer_R'])
        self.peer_t = self.tensor(self.data['peer_t'])

    def observation(self, vertices, joints, pixels, include_hands=True, hand_geometry=None):
        if not self.active:
            raise RuntimeError('v2 objective called in legacy mode')
        # Fixed group weights; every side and modality has its own denominator.
        err = torch.log1p(((pixels[self.body_ids]-self.body_xy)/20.).square().sum(-1))
        body, body_wrist = [weighted_group(err[ix], self.body_weights[ix]) for ix in self.body_groups]
        if not include_hands:
            self.last_terms = {'body_2d': body, 'sapiens_wrist_2d': body_wrist}
            return .01*(body+body_wrist)
        if hand_geometry is not None:
            vertices, joints, _ = hand_geometry
        hand_joints = self.hand_regressor @ vertices
        # Anatomical wrist centers are used for wrist position constraints.
        hand_joints = hand_joints.clone()
        hand_joints[0, 0] = joints[self.ji['L_Wrist']]
        hand_joints[1, 0] = joints[self.ji['R_Wrist']]
        hand_uv = self.camera.world2camera_pytorch(hand_joints.reshape(-1, 3)).reshape(2, 21, 2)
        e2 = torch.log1p(((hand_uv-self.hand_xy)/20.).square().sum(-1))
        residual = hand_joints-self.tri
        e3 = torch.log1p((self.whiten @ residual[..., None])[..., 0].square().sum(-1))
        terms = {'body_2d': body, 'sapiens_wrist_2d': body_wrist}
        total = body+body_wrist
        for side in range(2):
            prefix = ('left', 'right')[side]
            for suffix, error, weights, factor in (
                    ('wilor_wrist_2d', e2, self.hand_weights, .5),
                    ('wrist_3d', e3, self.tri_weights, .5)):
                value = weighted_group(error[side, :1], weights[side, :1])
                terms[prefix+'_'+suffix] = value
                total = total+factor*value
            for suffix, error, weights, factor in (
                    ('hand_2d', e2, self.hand_weights, .5),
                    ('hand_3d', e3, self.tri_weights, .25)):
                value = weighted_group(error[side, 1:], weights[side, 1:], 4.)
                terms[prefix+'_'+suffix] = value
                total = total+factor*value
        self.last_terms = terms
        return .01*total

    def visibility(self, vertices):
        if not self.active:
            return vertices.new_zeros(())
        h = vertices[self.head_ids]
        values = []
        for camera, pts in ((self.camera, h), (self.peer_camera, h@self.peer_R.T+self.peer_t)):
            uv = camera.world2camera_pytorch(pts)
            width, height = camera.img_size
            inside = (torch.sigmoid(uv[:, 0]/10.)*torch.sigmoid((width-uv[:, 0])/10.)
                      *torch.sigmoid(uv[:, 1]/10.)*torch.sigmoid((height-uv[:, 1])/10.))
            values.append((inside*torch.sigmoid(pts[:, 2]/.01)).mean())
        return .03*torch.stack(values).mean()

    def diagnostics(self, vertices, joints, pixels):
        if not self.active:
            return {}
        self.observation(vertices, joints, pixels)
        return {'groups': {k: float(v.detach()) for k, v in self.last_terms.items()},
                'head_visibility_loss': float(self.visibility(vertices).detach()),
                'body_minimum_z_m': float(min_z(vertices).detach()),
                'triangulated_points': int((self.tri_weights > 0).sum()),
                'source': 'WiLoR two-view 2D triangulation; no monocular 3D targets'}
