"""Head frame math and reusable point losses for this experiment."""
import numpy as np
import torch
from egosmplx_pipeline.body_common import axis_angle_rotation


def head_rotation(global_orient, body_pose, parents, head=15):
    chain = []
    i = head
    while i > 0:
        chain.append(i)
        i = int(parents[i])
    result = axis_angle_rotation(global_orient.reshape(3))
    for i in reversed(chain):
        result = result @ axis_angle_rotation(body_pose.reshape(21, 3)[i - 1])
    return result


def rotation_penalty(predicted, target, sigma_rad):
    # Equals theta^2 / sigma^2 to second order; avoids acos gradient at identity.
    return (predicted - target).square().sum() / (2. * sigma_rad ** 2)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.rad2deg(np.arccos(np.clip((np.trace(a @ b.T) - 1.) / 2., -1., 1.))))


def shoulder_excess(residual, tolerance):
    radius = residual.norm(dim=-1, keepdim=True)
    return residual * (1. - tolerance / radius.clamp_min(1e-6)).clamp_min(0.)


def positive_points(points, epsilon):
    return torch.stack([points[:, 0], points[:, 1], points[:, 2].clamp_min(epsilon)], 1)


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)
