"""Estimate a free rigid head transform from visible bilateral segmentation edges."""
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from body_only import OUT, ROOT, RUN, REFERENCE, camera_paths, read, sha, v3, write

sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_landmarks_head_views'))
from export import geometry

SIDES = ('left', 'right')
FACE_CLASSES = (3, 24, 25, 26, 27, 28)
HEAD_JOINTS = (15, 22, 23, 24)
STRIDE = 4
POSITION_LIMIT_MM = 180
CLIP_PX = 128
KERNEL = np.ones((3, 3), np.uint8)


def boundary(mask):
    edge = mask & (cv2.erode(mask.astype(np.uint8), KERNEL) == 0)
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    return edge


class Eye:
    def __init__(self, row, side, camera, camera_file, receipt):
        key = f"{row['pair']}_{side}"
        source = ROOT / 'outputs/frozen100_outline_20261125/seg' / (key + '.npy')
        assert sha(source) == receipt['probability_sha256']
        prob = np.load(source).astype(np.float32)
        label = prob.argmax(0)
        face = np.isin(label, FACE_CLASSES)
        _, components, stats, _ = cv2.connectedComponentsWithStats(face.astype(np.uint8))
        assert len(stats) > 1, key
        region = components == (1 + stats[1:, cv2.CC_STAT_AREA].argmax())

        photo = ROOT / 'outputs/rw12_headonly_20261128/images' / (key + '.png')
        assert sha(photo) == receipt['image_sha256']
        image = cv2.imread(str(photo))
        assert image.shape[:2] == (872, 872)
        nonblack = (image.max(axis=2) > 8).astype(np.uint8)
        curves, _ = cv2.findContours(nonblack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aperture = cv2.drawContours(np.zeros_like(nonblack), [max(curves, key=cv2.contourArea)],
                                    -1, 1, cv2.FILLED)[2::STRIDE, 2::STRIDE]
        inside = cv2.erode(aperture, KERNEL, iterations=2).astype(bool)
        glasses = cv2.dilate((label == 2).astype(np.uint8), KERNEL, iterations=2).astype(bool)
        context = np.isin(label, [0, 13, 22, 23])
        obs_edge = boundary(region) & inside & ~glasses & (prob.max(0) >= .55)
        obs_edge &= cv2.dilate(context.astype(np.uint8), KERNEL).astype(bool)
        y, x = np.nonzero(obs_edge)
        self.count = len(x)
        if self.count < 8:
            return
        self.observed = np.column_stack((x, y)) * STRIDE + 2
        self.target_tree = cKDTree(self.observed)
        pad = CLIP_PX // STRIDE
        ys = slice(max(0, y.min() - pad), min(region.shape[0], y.max() + pad + 1))
        xs = slice(max(0, x.min() - pad), min(region.shape[1], x.max() + pad + 1))
        yy, xx = np.mgrid[ys, xs]
        self.pixels = np.stack((xx, yy), axis=-1) * STRIDE + 2
        self.usable = (inside & ~glasses)[ys, xs]
        transform = np.asarray(row['inputs']['views'][side]['camera']['left_to_view'])
        rotation, translation = transform[:3, :3], transform[:3, 3]
        adapter = read(camera_file)['input_adapter']
        calibrated_pixels = self.pixels.reshape(-1, 2) * adapter['scale'] + adapter['offset']
        rays = camera.camera2world_ray(calibrated_pixels) @ rotation
        origin = np.broadcast_to(-rotation.T @ translation, rays.shape)
        self.rays = o3d.core.Tensor(np.column_stack((origin, rays)).astype(np.float32))

    def compare(self, scene, face_selection):
        result = scene.cast_rays(self.rays, nthreads=1)['primitive_ids'].numpy()
        hit = result < len(face_selection)
        hit[hit] = face_selection[result[hit]]
        predicted = self.pixels[boundary(hit.reshape(self.usable.shape)) & self.usable]
        if not len(predicted):
            return float(CLIP_PX), predicted
        forward = self.target_tree.query(predicted)[0]
        reverse = cKDTree(predicted).query(self.observed)[0]
        return float((np.minimum(forward, CLIP_PX).mean() + np.minimum(reverse, CLIP_PX).mean()) / 2), predicted


def search(loss):
    zero = np.zeros(6)
    axes = np.r_[np.eye(6), -np.eye(6)]
    # Test depth and vertical translations and both signs of pitch/yaw before
    # local search. The final objective contains only contour distance.
    starts = [zero]
    for axis in (0, 1, 3, 4, 5):
        distance = 45 if axis < 3 else 60
        starts.extend((distance * axes[axis], -distance * axes[axis]))
    starts = sorted(starts, key=loss)[:3]
    finals = []
    for seed in starts:
        for rot, move in ((18, 40), (9, 20), (4, 10), (2, 5), (1, 2.5)):
            offsets = axes * np.array([rot] * 3 + [move] * 3)
            for _ in range(14):
                best = min([seed] + [seed + delta for delta in offsets], key=loss)
                if np.array_equal(seed, best):
                    break
                seed = best
        finals.append(seed)
    return min(finals, key=loss), finals


def main(index):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    rows = read(OUT / 'prepared.json')['items']
    row = rows[index]
    pair = row['pair']
    source = read(REFERENCE / 'evidence/inputs.json')['rows'][pair]
    model, camera = v3.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    with np.load(OUT / 'budget_20000.npz') as archive:
        assert archive['selected_step'][index] >= 0
        params = dict(body_pose=archive['body_pose'][index], betas=archive['betas'][index],
                      global_orient=archive['global0'][index], transl=archive['translation'][index])
        delta = archive['delta'][index]
    vertices, joints, start_rotation, _ = geometry(layer, params, delta, False, None)
    head = joints[15]
    faces = np.asarray(model.face, np.int32)
    weights = layer.lbs_weights.detach().numpy()
    dominant = weights.argmax(axis=1)
    face_selection = np.isin(dominant[faces], (12,) + HEAD_JOINTS).all(axis=1)
    head_weight = weights[:, HEAD_JOINTS].sum(axis=1)[:, None]
    topology = o3d.core.Tensor(faces.astype(np.uint32))
    files = camera_paths()
    observations = [Eye(row, side, camera(path), path, source['eyes'][side])
                    for side, path in zip(SIDES, files)]
    eyes = [eye for eye in observations if eye.count >= 8]
    assert eyes, pair

    def evaluate(x):
        rotation = Rotation.from_rotvec(np.deg2rad(x[:3])).as_matrix()
        updated = (vertices - head) @ rotation.T + head + x[3:] / 1000
        points = vertices + head_weight * (updated - vertices)
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(points, np.float32)), topology)
        return [eye.compare(scene, face_selection) for eye in eyes]

    cache = {}

    def loss(x):
        if np.max(np.abs(x[3:])) > POSITION_LIMIT_MM or np.linalg.norm(x[:3]) > 180:
            return np.inf
        key = tuple(x)
        if key not in cache:
            cache[key] = float(np.mean([distance for distance, _ in evaluate(x)]))
        return cache[key]

    start = loss(np.zeros(6))
    best, finals = search(loss)
    final = loss(best)
    orientation = Rotation.from_rotvec(np.deg2rad(best[:3])).as_matrix() @ start_rotation
    write(OUT / 'poses' / f'{pair}.json', dict(pair=pair, subject=source['subject'],
          initial_head_m=head.tolist(), final_head_m=(head + best[3:] / 1000).tolist(),
          orientation_cam3_rotvec_deg=np.rad2deg(Rotation.from_matrix(orientation).as_rotvec()).tolist(),
          transform_delta=best.tolist(), silhouette_points=[eye.count for eye in observations],
          contour_before_px=start, contour_after_px=final, body_only_pose_step=int(read(OUT/'fit_complete.json')['final_selected_steps'][index]),
          search_limit_hit=bool(np.isclose(np.max(np.abs(best[3:])), POSITION_LIMIT_MM, atol=2.5)),
          search_finals=[dict(transform=x.tolist(), loss=loss(x)) for x in finals],
          evaluations=len(cache), exit_status=0))
    print(pair, start, final, len(cache), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
