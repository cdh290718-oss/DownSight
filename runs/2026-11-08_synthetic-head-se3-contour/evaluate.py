"""Held-out body/hand evaluation and common Brown-camera projection overlays."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import read, load
from common import head_rotation, angle_deg
from contour import Contours

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/synthetic_head_se3_contour_20261108'
OLD = ROOT / 'outputs/synthetic_original_20261108'
DATA = OLD / 'dataset/samples/000001'
cfg = read(OLD / 'configs/left.json')
os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],
                  EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo'])
model, Camera = setup(cfg['paths']['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model)
names = read(OLD / 'dataset/joint_names_egosmplx65.json')['names']
assert list(model.joints_name[:65]) == names
head_ids = model.layer['neutral'].lbs_weights.detach().numpy().argmax(1) == model.orig_joints_name.index('Head')
assert model.orig_joints_name.index('Head') == 15
spec = importlib.util.spec_from_file_location('raster', ROOT / 'runs/2026-11-04_mesh-projection-review/code/rasterizer.py')
raster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raster)
assets = RUN / 'assets'
assets.mkdir(exist_ok=True)


def project(xyz, K, dist):
    result = np.full((len(xyz), 2), np.nan)
    valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 1e-6)
    if valid.any():
        result[valid] = cv2.projectPoints(xyz[valid].astype(float), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    return result


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                max=float(x.max()) if len(x) else None)


def draw(image, mesh, uv, rays, title, truth_uv, prediction_uv=None, truth_mesh=False):
    depth = np.full((720, 1280), np.inf, np.float32)
    colors = np.zeros_like(image)
    hp = np.zeros((720, 1280), np.float32)
    drawn, rejected = raster.raster(mesh.astype(np.float32), np.asarray(model.face, np.int32),
                                   head_ids.astype(np.float32), uv.astype(np.float32), rays,
                                   depth, colors, hp, 180.)
    valid = np.isfinite(depth)
    if truth_mesh:
        light = colors[valid].astype(float).mean(1) / 146.7
        colors[valid] = np.clip(light[:, None] * np.array([55, 210, 70]), 0, 255).astype(np.uint8)
    result = image.copy()
    result[valid] = (.45 * result[valid] + .55 * colors[valid]).astype(np.uint8)
    for points, color, radius in [(truth_uv, (0, 230, 0), 5), (prediction_uv, (0, 40, 255), 3)]:
        if points is None:
            continue
        for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
            p = points[i]
            if np.isfinite(p).all() and 0 <= p[0] < 1280 and 0 <= p[1] < 720:
                cv2.circle(result, tuple(np.rint(p).astype(int)), radius, color, -1, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (1280, 40), (25, 25, 25), -1)
    cv2.putText(result, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    return result, dict(drawn_faces=int(drawn), rejected_near_or_long_faces=int(rejected),
                        visible_pixels=int(valid.sum()))


views = []
for side, dirname in [('left', 'left'), ('right', 'right_cpu')]:
    frame = OLD / 'fits' / dirname / 'frames' / side
    stages = {'original_raw': frame / 'raw/params.npz',
              'original_rigid': frame / 'rigid/optimized_params.npz',
              'original_body': frame / 'body/params.npz',
              'original_final': frame / 'fused/body_params.npz'}
    stages['previous_position_alternating'] = ROOT / 'outputs/synthetic_head_anchor_20261108' / side / 'cycle4_rigid/params.npz'
    stages['head_se3_initial'] = OUT / side / 'se3_joint/initial.npz'
    for label in ['se3_joint', 'se3_contour_joint']:
        path = OUT / side / label / 'final.npz'
        if path.exists(): stages[label] = path
    head_data = read(OUT / (side + '_targets.json'))
    target_R = np.asarray(head_data['head_rotation_matrix'])
    targets = head_data['targets']
    exclusion = np.load(OUT / (side + '_contour_masks.npz'))['torso_arm_exclusion']
    for name, q in targets.items():
        q['target'] = np.asarray(q['target'], np.float32).reshape(-1, 2)
        q['arm_exclusion'] = exclusion if name == 'torso' else np.zeros((720, 1280), np.uint8)
    job = read(OLD / 'fits' / dirname / 'jobs' / (side + '.json'))
    original_camera = Camera(job['row']['calibration'])
    contour = Contours(model, original_camera, targets, read(RUN / 'settings.json'))
    with np.load(DATA / (side + '_vertices.npz'), allow_pickle=False) as z:
        gv = z['vertices_camera_cv_m'].copy()
        K, dist = z['K_zero_based'].copy(), z['distortion'].copy()
        head_truth = z['joints_camera_cv_m'][15].copy()
    with np.load(DATA / (side + '_egosmplx65.npz'), allow_pickle=False) as z:
        truth = z['joints_3d_camera_m'].copy()
        truth_uv = z['joints_2d_px'].copy()
        inside = z['in_frame_no_occlusion_test'].copy()
    image = cv2.imread(str(DATA / (side + '.png')))
    yy, xx = np.indices((720, 1280), dtype=float)
    pixels = np.stack([xx.ravel(), yy.ravel()], 1)
    normal = cv2.undistortPointsIter(pixels[:, None, :], K, dist, None, None,
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-10)).reshape(-1, 2)
    rays = np.column_stack([normal, np.ones(len(normal))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays.reshape(720, 1280, 3).astype(np.float32)
    im, renderstats = draw(image, gv, project(gv, K, dist), rays, side + ' / ground truth', truth_uv, truth_mesh=True)
    cv2.imwrite(str(assets / (side + '_truth.jpg')), im, [cv2.IMWRITE_JPEG_QUALITY, 94])
    row = dict(side=side, head_truth_position_m=head_truth.tolist(), truth_pelvis_z_m=float(truth[0, 2]),
               truth_mesh_minimum_z_m=float(gv[:, 2].min()),
               truth_image='assets/' + side + '_truth.jpg', truth_render=renderstats, stages={})
    for label, path in stages.items():
        p = load(path)
        mesh, joints = fusion.geometry(p)
        prediction = joints[:65].copy()
        with torch.no_grad():
            native = model.layer['neutral'](**{k: torch.as_tensor(p[k], dtype=torch.float32).reshape(1, -1)
                                              for k in fusion.PARAMS}).joints[0].numpy() + p['transl'].reshape(3)
        head_error = native[15] - head_truth
        # Float64 independent FK avoids trace/acos rounding tiny angles to zero.
        pred_R = cv2.Rodrigues(p['global_orient'].reshape(3).astype(float))[0]
        for joint in [3, 6, 9, 12, 15]:
            pred_R = pred_R @ cv2.Rodrigues(p['body_pose'].reshape(21, 3)[joint - 1].astype(float))[0]
        if label == 'original_final':
            mesh = load(frame / 'fused/mesh.npz')['vertices_cam']
            for i, hand in enumerate(['left', 'right']):
                prediction[25 + i * 20:45 + i * 20] = (model.orig_hand_regressor[hand] @ mesh)[1:]
        contour_diagnostics = contour.refresh(mesh)
        delta = (prediction - truth) * 1000
        uv = project(prediction, K, dist)
        valid = inside & np.isfinite(uv).all(1)
        body = np.zeros(65, bool)
        body[:25] = True
        new_vertex_uv = project(mesh, K, dist)
        hv = new_vertex_uv[head_ids]
        head_visible = np.isfinite(hv).all(1) & (hv[:, 0] >= 0) & (hv[:, 0] < 1280) & (hv[:, 1] >= 0) & (hv[:, 1] < 720)
        result = dict(body25_absolute_mean_mm=float(np.linalg.norm(delta[:25], axis=1).mean()),
                      body25_pelvis_aligned_mean_mm=float(np.linalg.norm(delta[:25] - delta[:1], axis=1).mean()),
                      body25_inframe_projection_px=stats(np.linalg.norm(uv[valid & body] - truth_uv[valid & body], axis=1)),
                      body_truth_inframe_count=int((inside & body).sum()),
                      body_invalid_prediction_count=int((inside & body & ~np.isfinite(uv).all(1)).sum()),
                      head_orientation_error_deg=angle_deg(pred_R, target_R),
                      head_rotation_matrix=pred_R.tolist(),
                      contour_diagnostics=contour_diagnostics,
                      head_anchor_error_mm=float(np.linalg.norm(head_error) * 1000),
                      head_anchor_error_xyz_mm=(head_error * 1000).tolist(),
                      head_position_m=native[15].tolist(),
                      pelvis_absolute_error_mm=float(np.linalg.norm(delta[0])),
                      head_to_pelvis_vector_error_mm=((prediction[0] - native[15] - (truth[0] - head_truth)) * 1000).tolist(),
                      head_to_pelvis_distance_m=float(np.linalg.norm(prediction[0] - native[15])),
                      truth_head_to_pelvis_distance_m=float(np.linalg.norm(truth[0] - head_truth)),
                      minimum_mesh_z_m=float(mesh[:, 2].min()),
                      behind_camera_vertex_count=int((mesh[:, 2] < 0).sum()),
                      head_vertices_inframe_no_occlusion_count=int(head_visible.sum()),
                      shoulder_projected_width_px=float(np.linalg.norm(uv[8] - uv[9])) if np.isfinite(uv[[8, 9]]).all() else None,
                      wrists_absolute_mean_mm=float(np.linalg.norm(delta[[12, 13]], axis=1).mean()),
                      elbows_absolute_mean_mm=float(np.linalg.norm(delta[[10, 11]], axis=1).mean()),
                      hands40_absolute_mean_mm=float(np.linalg.norm(delta[25:65], axis=1).mean()),
                      hand_model='fused MANO' if label == 'original_final' else 'fixed network SMPL-X hand parameters',
                      per_joint=[dict(name=names[i], absolute_mm=float(np.linalg.norm(delta[i])),
                                      error_xyz_mm=delta[i].tolist(), pred_z_m=float(prediction[i, 2]), truth_z_m=float(truth[i, 2]),
                                      truth_inframe=bool(inside[i]),
                                      projection_brown_px=float(np.linalg.norm(uv[i] - truth_uv[i])) if valid[i] else None)
                                 for i in range(65)],
                      source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if label in ['original_body', 'original_final', 'previous_position_alternating', 'head_se3_initial', 'se3_joint', 'se3_contour_joint']:
            im, rd = draw(image, mesh, new_vertex_uv, rays, side + ' / ' + label + ' / green: GT, red: prediction', truth_uv, uv)
            name = side + '_' + label + '.jpg'
            cv2.imwrite(str(assets / name), im, [cv2.IMWRITE_JPEG_QUALITY, 94])
            result.update(image='assets/' + name, render=rd)
        row['stages'][label] = result
    views.append(row)

result = dict(scope='One synthetic posture, two independently fitted camera views; Head position and orientation GT are optimizer inputs; all other GT evaluation is held out',
              head_anchor_definition='Native SMPL-X Head joint index 15 position and camera-space FK rotation',
              projection='Brown-Conrady used for evaluation and every displayed mesh; original polynomial calibration used during fitting',
              occlusion='GT in-frame flags and head vertex counts have no scene/self-occlusion test',
              renderer='Per-pixel ray triangle depth; triangles crossing near plane or exceeding 180 px edges are omitted; not exact curved-edge rasterization',
              views=views)
(RUN / 'evidence/metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print('evaluated', len(views), 'views')
