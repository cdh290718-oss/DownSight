"""Rotate the archived OBJ into a display frame determined by the stereo rig."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-20_rw20000_frozen_mesh12'))
import gallery as g

OUT = ROOT / 'outputs/rw20000_stereo_axes_20261120'
ARCHIVE = 'RW20000_stereo_axes_12_OBJ.zip'
TAG = 'rw20000-stereo-axes-20261120'


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('Degenerate stereo axes')
    return vector / length


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def rig_axes(transform):
    rotation, translation = transform[:3, :3], transform[:3, 3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    right_center = -rotation.T @ translation
    x = unit(right_center)
    optical = np.array([0., 0., 1.])
    down = unit(optical + rotation.T @ optical)
    up = -unit(down - (down @ x) * x)
    basis = np.column_stack([x, up, np.cross(x, up)])
    assert np.allclose(basis.T @ basis, np.eye(3)) and np.isclose(np.linalg.det(basis), 1)
    return basis, right_center


def comparison(path, pair, original, exported, faces, old_basis, pelvis, basis):
    # Both cameras look at the same pelvis, at identical scale. Flip X and Z
    # for a front camera on the rig's -Z side while preserving handedness.
    front = np.diag([-1., 1., -1.])
    side = g.rot([0., np.pi / 2, 0.])
    views = [(original, old_basis, pelvis), (exported, front, pelvis @ basis),
             (original, old_basis @ side, pelvis), (exported, front @ side, pelvis @ basis)]
    coords = np.concatenate([(v - p) @ b for v, b, p in views])
    offset = (coords.max(0) + coords.min(0)) / 2
    scale = min(320 / np.ptp(coords[:, 0]), 442 / np.ptp(coords[:, 1]))
    panels = [g.free_render(v, faces, b, p @ b + offset, scale) for v, b, p in views]
    headings = ['Original | front', 'Stereo | front', 'Original | side', 'Stereo | side']
    bar = np.full((42, 1400, 3), 245, np.uint8)
    for i, title in enumerate(headings):
        cv2.putText(bar, title, (12 + i * 350, 29), 0, .74, (44, 44, 44), 1, cv2.LINE_AA)
    cv2.putText(bar, pair, (1290, 29), 0, .56, (85, 85, 85), 1, cv2.LINE_AA)
    g.save(path, np.vstack([bar, np.hstack(panels)]))


def main():
    originals = g.read(g.RUN / 'evidence/renders_rw20000.json')['rows']
    execution = g.read(g.REF / 'execution.json')
    inputs = g.read(g.REF / 'inputs.json')
    prepared = g.SOURCE / 'prepared.json'
    checkpoint = g.SOURCE / 'budget_20000.npz'
    assert g.sha(prepared) == inputs['solver_input_sha256']
    assert g.sha(checkpoint) == execution['params_sha256']
    assert g.sha(g.DATA / 'reference.json') == execution['reference_sha256']
    items = g.read(prepared)['items']
    refs = g.read(g.DATA / 'reference.json')
    planes = {r['pair']: r for r in g.read(ROOT / 'runs/2026-11-20_rw20000_floor3/evidence/planes.json')['rows']}
    model, camera = g.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    cameras = []
    scales = []
    for side in ['left', 'right']:
        f = g.DATA / ('camera_' + side + '.json')
        assert g.sha(f) == inputs['cameras'][f.name]
        cam = camera(f)
        calibration = g.read(f)
        ray = cam.camera2world_ray(np.array([calibration['image_center']]))[0]
        assert np.allclose(ray, [0., 0., 1.], atol=1e-5)
        cameras.append(cam)
        scales.append(calibration['input_adapter']['scale'])
    done = g.read(g.SOURCE / 'fit_complete.json')['ids']
    rows, members = [], {}
    with np.load(checkpoint) as params, zipfile.ZipFile(OUT / ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as archive:
        for row in originals:
            p, subject = row['pair'], row['subject']
            i = done.index([p, 'rw20000', '2d'])
            item = items[i]
            assert item['pair'] == p and item['condition'] == 'rw20000'
            transforms = [np.array(item['inputs']['views'][s]['camera']['left_to_view']) for s in ['left', 'right']]
            assert np.allclose(transforms[0], np.eye(4))
            basis, right_center = rig_axes(transforms[1])
            source = g.OUT / 'objects' / subject / row['obj']
            assert g.sha(source) == row['source_obj_sha256']
            original, faces = g.load_obj(source)
            fit = {k: params[v][i] for k, v in [('global_orient','global0'), ('body_pose','body_pose'), ('transl','translation'), ('betas','betas')]}
            vertices, j, _, _ = g.geometry(layer, fit, params['delta'][i], False, None)
            assert np.max(np.abs(vertices - original)) < 3e-6
            filename = subject + '_' + p + '_stereo_Y_up.obj'
            obj = OUT / 'objects' / filename
            obj.parent.mkdir(parents=True, exist_ok=True)
            with obj.open('w') as f:
                f.write('# meters; stereo rig Y up; origin at left camera center\n')
                np.savetxt(f, original @ basis, fmt='v %.9f %.9f %.9f')
                np.savetxt(f, faces + 1, fmt='f %d %d %d')
            v, tri = g.load_obj(obj)
            assert np.array_equal(faces, tri)
            recovered = v @ basis.T
            drift = float(np.max(np.linalg.norm(recovered - original, axis=1)))
            assert drift < 1e-8
            pixel_drift = []
            for cam, t, scale in zip(cameras, transforms, scales):
                def project(points):
                    return cam.world2camera(points @ t[:3,:3].T + t[:3,3]) / scale
                displacement = np.linalg.norm(project(recovered) - project(original), axis=1)
                pixel_drift.append(float(np.max(displacement)))
            assert max(pixel_drift) < 1e-4
            pred = j[g.common.NATIVE_IDS][g.common.OBS]
            gt = np.array(refs[p]['gt'])[g.common.OBS]
            def metrics(a, b):
                return [float(np.linalg.norm(pts - b, axis=1).mean() * 1000) for pts in (a, g.align(a, b))]
            before, after = metrics(pred, gt), metrics(pred @ basis, gt @ basis)
            assert np.allclose(before, [row['nonpa12_mm'], row['pa12_mm']], atol=.001)
            assert np.allclose(before, after, atol=.0001, rtol=0)
            trunk = (j[16] + j[17] - j[1] - j[2]) / 2
            floor = None
            if p in planes:
                ref = planes[p]
                assert ref['obj_sha256'] == row['source_obj_sha256']
                carpet = np.asarray(ref['up_left_cv'])
                floor = dict(up_difference_deg=angle(basis[:,1], carpet), trunk_from_floor_up_deg=angle(trunk, carpet),
                             carpet_up_in_rig=(carpet @ basis).tolist())
            old_basis = g.rot(params['delta'][i]) @ g.rot(params['global0'][i])
            path = RUN / 'assets' / p / 'comparison.jpg'
            comparison(path, p, original, v, faces, old_basis, j[0], basis)
            name = subject + '/' + filename
            archive.write(obj, name)
            members[name] = g.sha(obj)
            rows.append(dict(pair=p, subject=subject, sequence=row['sequence'], right_center_m=right_center.tolist(),
                             basis_columns_in_left_camera=basis.tolist(), determinant=float(np.linalg.det(basis)),
                             trunk_from_rig_up_deg=angle(trunk, basis[:,1]), floor_check=floor, source_obj_sha256=g.sha(source),
                             export_obj_sha256=members[name], max_roundtrip_vertex_m=drift, vertex_count=len(v),
                             max_projection_delta_px=pixel_drift, before_nonpa_pa12_mm=before, after_nonpa_pa12_mm=after,
                             images={path.name:g.sha(path)}))
            print(p, pixel_drift, flush=True)
        archive.writestr('manifest.json', json.dumps(members, indent=2))
        archive.writestr('README.txt', 'Meters, right-handed. Origin: left optical center. +X from left camera to right camera. +Y opposite the mean camera look direction projected orthogonal to X. +Z = X cross Y. No measured gravity or floor height.\n')
    with zipfile.ZipFile(OUT / ARCHIVE) as archive:
        assert len(members) == 12 and set(archive.namelist()) == set(members) | {'README.txt', 'manifest.json'}
        assert archive.testzip() is None
        for name, sha in members.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    summary = dict(pairs=len(rows), max_vertex_delta_m=max(r['max_roundtrip_vertex_m'] for r in rows),
                   max_projection_delta_px=np.max([r['max_projection_delta_px'] for r in rows], axis=0).tolist(),
                   mean_before_nonpa_pa12_mm=np.mean([r['before_nonpa_pa12_mm'] for r in rows], axis=0).tolist(),
                   mean_after_nonpa_pa12_mm=np.mean([r['after_nonpa_pa12_mm'] for r in rows], axis=0).tolist())
    g.write(RUN / 'evidence/renders_stereo.json', dict(rows=rows, summary=summary, checkpoint_sha256=g.sha(checkpoint),
            prepared_sha256=g.sha(prepared), cameras_sha256=inputs['cameras'], command='<BODY_PYTHON> -Bu export_rig.py', exit_status=0))
    g.write(RUN / 'evidence/attachment_audit.json',dict(archive=ARCHIVE,tag=TAG,objects=len(members),members=members,
            bytes=(OUT / ARCHIVE).stat().st_size,files_sha256={ARCHIVE:g.sha(OUT / ARCHIVE)},exit_status=0))


if __name__ == '__main__':
    main()
