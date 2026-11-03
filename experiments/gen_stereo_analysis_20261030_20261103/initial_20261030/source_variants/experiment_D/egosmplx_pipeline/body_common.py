"""Joint mappings, bounded pose increments, and rigid rotation math. No annotation reader."""
from functools import lru_cache
import numpy as np
import torch

JOINT_MAPPING = {
    "nose": "Nose",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
}


@lru_cache(maxsize=1)
def _body_vertex_indices():
    # The topology is shared by SMPL-X and the pipeline's fused meshes.
    from utils.human_models import smpl_x
    weights = smpl_x.layer['neutral'].lbs_weights.detach().cpu()
    dominant = weights.argmax(dim=1).numpy()
    head = dominant == smpl_x.orig_joints_name.index('Head')
    neck = dominant == smpl_x.orig_joints_name.index('Neck')
    if int(head.sum()) != 3345 or not neck.any():
        raise ValueError('Unexpected SMPL-X head/neck vertex assignment')
    return np.flatnonzero(~(head | neck)), len(dominant)


def body_min_z(v):
    """Minimum camera-space Z outside dominant-LBS Head and Neck vertices."""
    ids, vertex_count = _body_vertex_indices()
    if v.ndim != 2 or v.shape != (vertex_count, 3):
        raise ValueError('Expected the unchanged SMPL-X vertex topology')
    if isinstance(v, torch.Tensor):
        ids = torch.as_tensor(ids, dtype=torch.long, device=v.device)
    return v[ids, 2].min()


def wrist_floor():
    """Fixed minimum wrist distance from camera, in meters."""
    return 0.15


BODY_POSE_NAMES = (
    "L_Hip", "R_Hip", "Spine_1", "L_Knee", "R_Knee", "Spine_2",
    "L_Ankle", "R_Ankle", "Spine_3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)


MAX_DELTA_DEG = {
    "L_Hip": 18.0, "R_Hip": 18.0,
    "Spine_1": 10.0, "Spine_2": 10.0, "Spine_3": 12.0,
    "L_Knee": 22.0, "R_Knee": 22.0,
    "L_Ankle": 14.0, "R_Ankle": 14.0,
    "Neck": 12.0, "Head": 12.0,
    "L_Collar": 15.0, "R_Collar": 15.0,
    "L_Shoulder": 28.0, "R_Shoulder": 28.0,
    "L_Elbow": 25.0, "R_Elbow": 25.0,
    "L_Wrist": 15.0, "R_Wrist": 15.0,
}


def save_obj(vertices, faces, path):
    with path.open("w", encoding="utf-8") as output:
        for vertex in vertices:
            output.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            output.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def axis_angle_rotation(axis_angle):
    x, y, z = axis_angle.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    theta = torch.linalg.norm(axis_angle)
    first = torch.sinc(theta / np.pi)
    second = 0.5 * torch.sinc(theta / (2.0 * np.pi)) ** 2
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + first * skew + second * (skew @ skew)
