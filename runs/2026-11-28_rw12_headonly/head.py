"""Apply the Video 003 head and neck search in the original RW fisheye pixels."""
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.optimize import least_squares

from body_start import ROOT, RUN, OUT, SIDES, camera_paths, read, sha, write, v3

sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_landmarks_head_views'))
from export import geometry

ROT = [11, 14]
STRIDE = 4
LIMIT = 35
FACE_LABELS = [3, 24, 25, 26, 27, 28]


def observed_outline(labels, nose):
    face = np.isin(labels, FACE_LABELS).astype(np.uint8)
    _, components = cv2.connectedComponents(face)
    nx, ny = np.rint((nose - 2) / STRIDE).astype(int)
    selected = components[ny, nx]
    if not selected:
        return np.empty((0, 2), dtype=int)
    region = components == selected
    glasses = cv2.dilate((labels == 2).astype(np.uint8), np.ones((5, 5), np.uint8))
    points = []
    for x in range(2, region.shape[1] - 2, 2):
        ys = np.flatnonzero(region[:, x])
        if len(ys) < 3:
            continue
        y = ys[0]
        if y >= 2 and not glasses[y, x] and labels[y - 1, x] in (0, 13, 22, 23):
            points.append((STRIDE * x + 2, STRIDE * y))
    return np.asarray(points, dtype=int).reshape(-1, 2)


def rmse(residual):
    return float(np.sqrt((residual ** 2).sum(axis=1).mean()))


def fit(pair):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    inputs = read(RUN / 'evidence/inputs.json')
    prepared = OUT / 'fit/0/prepared.json'
    assert sha(prepared) == inputs['shards']['0']
    row = next(x for x in read(prepared)['items'] if x['pair'] == pair)
    model, camera = v3.ref.common.setup_model()
    from smplx.vertex_ids import vertex_ids
    nose_id = vertex_ids['smplx']['nose']
    layer = model.layer['neutral'].eval()
    faces = np.asarray(model.face, np.int32)
    topology = o3d.core.Tensor(faces.astype(np.uint32))
    head_faces = np.isin(layer.lbs_weights.argmax(1).numpy(), [15, 22, 23, 24])[faces].all(1)
    files = camera_paths()
    cams = [camera(p) for p in files]
    adapters = [read(p)['input_adapter'] for p in files]
    tfs = [np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in SIDES]
    observations = inputs['rows'][pair]['eyes']
    detected = np.array([observations[s]['nose_xy'] for s in SIDES])
    path = OUT / 'poses' / f'{pair}_body.npz'
    assert sha(path) == read(path.with_suffix('.json'))['sha256']
    with np.load(path) as archive:
        base, global0, betas = (archive[k] for k in ('locked', 'global0', 'betas'))
    assert not base[ROT].any()
    contours = []
    for side, nose in zip(SIDES, detected):
        path = ROOT / 'outputs/frozen100_outline_20261125/seg' / f'{pair}_{side}.npy'
        assert sha(path) == observations[side]['probability_sha256']
        mask = np.load(path).argmax(axis=0)
        contours.append(observed_outline(mask, nose))

    # Video 003 samples the highest visible face hit at each observed column.
    # Use the nose term alone where the required bilateral contour is absent.
    active = all(len(points) >= 8 for points in contours)
    rows = np.arange(0, 872, STRIDE)
    beams = []
    if active:
        for cam, adapter, transform, points in zip(cams, adapters, tfs, contours):
            x, y = np.meshgrid(points[:, 0], rows)
            pix = np.column_stack((x.ravel(), y.ravel()))
            r, t = transform[:3, :3], transform[:3, 3]
            directions = cam.camera2world_ray(pix * adapter['scale'] + adapter['offset']) @ r
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            origins = np.broadcast_to(-r.T @ t, directions.shape)
            beams.append(o3d.core.Tensor(np.column_stack((origins, directions)).astype(np.float32)))

    def mesh(angles):
        body = base[:21].copy()
        body[ROT] = np.deg2rad(angles).reshape(2, 3)
        params = dict(body_pose=body, global_orient=global0, transl=base[22], betas=betas,
                      left_hand_pose=base[23:38], right_hand_pose=base[38:53])
        return geometry(layer, params, base[21], False, None)[:2]

    def projected_nose(vertices):
        point = vertices[[nose_id]]
        uv = []
        for cam, adapter, transform in zip(cams, adapters, tfs):
            in_eye = point @ transform[:3, :3].T + transform[:3, 3]
            uv.append(((cam.world2camera(in_eye) - adapter['offset']) / adapter['scale'])[0])
        return np.asarray(uv)

    def metrics(vertices):
        uv = projected_nose(vertices)
        edge = None
        if active:
            scene = o3d.t.geometry.RaycastingScene(nthreads=1)
            scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32)), topology)
            gaps = []
            for beam, points in zip(beams, contours):
                ids = scene.cast_rays(beam, nthreads=1)['primitive_ids'].numpy()
                visible = ids < len(faces)
                visible[visible] = head_faces[ids[visible]]
                by_col = visible.reshape(len(rows), len(points))
                y = np.where(by_col.any(axis=0), rows[by_col.argmax(axis=0)], 872)
                gaps.append(np.abs(y - points[:, 1]).mean())
            edge = float(np.mean(gaps))
        return rmse(uv - detected), edge, uv.tolist()

    zero = np.zeros(6)
    axes = np.eye(6)
    cache = {}

    def loss(angles):
        if np.linalg.norm(angles.reshape(2, 3), axis=1).max() > LIMIT:
            return np.inf
        key = tuple(angles)
        if key not in cache:
            nose, edge, _ = metrics(mesh(angles)[0])
            cache[key] = nose + (edge or 0.) + .003 * (angles @ angles)
        return cache[key]

    def nose_error(angles):
        return np.r_[(projected_nose(mesh(angles)[0]) - detected).ravel(), .45 * angles]

    def jac(angles):
        return np.column_stack([(nose_error(angles + d) - nose_error(angles - d)) / .2
                                for d in .1 * axes])

    def refine(angles):
        for step in (8, 4, 2, 1, .5):
            for _ in range(20):
                best = min([angles] + [angles + d for d in step * np.vstack((axes, -axes))], key=loss)
                if np.array_equal(best, angles):
                    break
                angles = best
        return angles

    initial = least_squares(nose_error, zero, jac=jac, bounds=(-LIMIT, LIMIT)).x.reshape(2, 3)
    initial *= np.minimum(1, 34.5 / np.maximum(np.linalg.norm(initial, axis=1, keepdims=True), .01))
    seeds = [zero, initial.ravel(), np.array([12., 0, 0, -8, 0, 0])]
    best = min((refine(seed) for seed in seeds), key=loss)
    before_v, before_j = mesh(zero)
    after_v, after_j = mesh(best)
    body = [i for i in range(22) if i not in (12, 15)]
    drift = float(np.linalg.norm(after_j[body] - before_j[body], axis=1).max() * 1000)
    assert drift < .001
    before, after = metrics(before_v), metrics(after_v)
    write(OUT / 'poses' / f'{pair}_head.json', dict(pair=pair, body_pose_sha256=sha(OUT / 'poses' / f'{pair}_body.npz'),
          method='nose_and_outline' if active else 'nose_only', observed_nose_px=detected.tolist(),
          edge_counts=list(map(len, contours)), edge_points=[x.tolist() for x in contours],
          mount_cam3_m=row['head_position'], initial_head_m=before_j[15].tolist(), final_head_m=after_j[15].tolist(),
          nose_rmse_before_px=before[0], nose_rmse_after_px=after[0],
          edge_mae_before_px=before[1], edge_mae_after_px=after[1],
          projected_nose_before_px=before[2], projected_nose_after_px=after[2],
          neck_head_deg=best.reshape(2, 3).tolist(), body_drift_mm=drift, evaluations=len(cache), exit_status=0))
    print(pair, 'nose and outline' if active else 'nose only', before[0], after[0], flush=True)


if __name__ == '__main__':
    fit(sys.argv[1])
