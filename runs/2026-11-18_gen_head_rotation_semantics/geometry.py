"""Exact body-joint forward kinematics; skinning is unnecessary for these losses.

The original full model remains the reference for both forward/gradient validation
and final evaluation, including the two mesh-vertex heel proxies.
"""
import numpy as np
import torch


def rotation(aa):
    x, y, z = aa.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    theta = aa.norm(dim=-1)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return eye + torch.sinc(theta / np.pi)[..., None, None] * skew + (
        0.5 * torch.sinc(theta / (2 * np.pi)).square())[..., None, None] * (skew @ skew)


class BodyKinematics:
    def __init__(self, layer):
        # Linearity makes regressing the shape offsets once exactly equivalent
        # to regressing all shaped vertices on every optimization step.
        self.rest = (layer.J_regressor @ layer.v_template)[:22].detach().clone()
        self.shape = torch.einsum('jv,vck->jck', layer.J_regressor[:22], layer.shapedirs[:, :, :10]).detach()
        self.parents = layer.parents[:22].tolist()
        self.levels = []
        depth = [0]
        for j in range(1, 22):
            depth.append(depth[self.parents[j]] + 1)
        for d in range(1, max(depth) + 1):
            ids = [j for j in range(22) if depth[j] == d]
            self.levels.append((ids, [self.parents[j] for j in ids]))

    def __call__(self, body, beta, global0, delta, translation):
        b = body.shape[0]
        joints = self.rest + torch.einsum('bk,jck->bjc', beta, self.shape)
        local_rot = rotation(torch.cat([global0[:, None], body], 1))
        positions = joints[:, :1]
        rotations = local_rot[:, :1]
        order = [0]
        for ids, parents in self.levels:
            parent_slots = [order.index(p) for p in parents]
            pr = rotations[:, parent_slots]
            rel = joints[:, ids] - joints[:, parents]
            positions = torch.cat([positions, positions[:, parent_slots] + (pr @ rel[..., None])[..., 0]], 1)
            rotations = torch.cat([rotations, pr @ local_rot[:, ids]], 1)
            order.extend(ids)
        inverse = [order.index(j) for j in range(22)]
        positions = positions[:, inverse]
        rotations = rotations[:, inverse]
        rd = rotation(delta)
        pivot = positions[:, :1]
        positions = ((positions - pivot) @ rd.transpose(-1, -2)) + pivot + translation[:, None]
        return positions, rd @ rotations[:, 15]


def project(points, camera):
    """Same calibrated polynomial, FP64 evaluation and axis convention as pipeline."""
    dtype = points.dtype
    axes = points.new_tensor(camera.camera_axis_transform, dtype=torch.float64)
    p = points.double() @ axes.T
    radius = torch.linalg.vector_norm(p[..., :2], dim=-1).clamp_min(1e-12)
    theta = torch.atan(-p[..., 2] / radius)
    rho = torch.full_like(theta, float(camera.fisheye_inverse_polynomial[-1]))
    for coefficient in reversed(camera.fisheye_inverse_polynomial[:-1]):
        rho = rho * theta + float(coefficient)
    x, y = p[..., 0] * rho / radius, p[..., 1] * rho / radius
    return torch.stack([float(camera.c)*x + float(camera.d)*y + float(camera.img_center[0]),
                        float(camera.e)*x + y + float(camera.img_center[1])], -1).to(dtype)
