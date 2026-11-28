"""Fit the Golden D body from a mount-based Head anchor using RW stereo cameras."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/rw12_headonly_20261128'
GOLD = ROOT / 'runs/2026-11-18_rw_frozen200_golden_d'
SOURCE = ROOT / 'outputs/rw_frozen200_golden_d_20261118'
SIDES = ('left', 'right')
PRED = ROOT / 'outputs/rw20000_frozen200_20261118/rw20000_test_frozen_predictions.json'
NOSE = ROOT / 'outputs/rw_test_completion_20261117/original_test_frozen_predictions.json'
MANIFEST = ROOT / 'outputs/rw_test_completion_20261117/test_manifest.json'
PREP = GOLD / 'evidence/preparation.json'
sys.path.insert(0, str(ROOT / 'runs/2026-11-23_test500_joint'))
import body as v3

read, write, sha = v3.read, v3.write, v3.sha


def camera_paths():
    return [SOURCE / ('camera_' + side + '.json') for side in SIDES]


def unit(vector):
    return vector / np.linalg.norm(vector)


def mount_head(transform, brow_to_head):
    rotation, translation = transform[:3, :3], transform[:3, 3]
    center = -rotation.T @ translation
    right = unit(center)
    down = unit(np.array([0., 0., 1.]) + rotation.T @ [0., 0., 1.])
    up = -unit(down - right * (down @ right))
    forward = unit(np.cross(-right, up))
    basis = np.column_stack((-right, up, forward))
    brow = center / 2 + basis @ [0., .05, -.05]
    return brow + basis @ brow_to_head


def prepare():
    archive = read(PREP)
    assert sha(PRED) == archive['prediction_sha256']['rw20000']
    assert sha(NOSE) == archive['prediction_sha256']['original']
    assert sha(MANIFEST) == archive['manifest_sha256']
    assert [sha(p) for p in camera_paths()] == [c['local_sha256'] for c in archive['cameras']]
    manifest = read(ROOT / 'runs/2026-11-28_sapiens_contour12/evidence/results.json')['rows']
    rig = {x['session']: x for x in archive['rigs']}
    samples = {x['pair']: x for x in archive['pairs']}
    images = {x['id']: x for x in read(MANIFEST)['records']}
    body = {x['id']: x for x in read(PRED)}
    nose = {x['id']: x for x in read(NOSE)}
    landmark_map = v3.prev.mapping()
    template = read(v3.base.OUT / 'prepared.json')['items'][0]['points']
    offsets = read(ROOT / 'runs/2026-11-25_real001_head_mount/evidence/inputs.json')['brow_to_head_template_m']
    check_transform = np.asarray(read(ROOT / 'runs/2026-11-25_realvideo4_dv3/evidence/calibration.json')['T'])
    video_anchor = read(ROOT / 'runs/2026-11-27_video34_uniform_v5/evidence/inputs.json')['head_cam3_m']
    assert np.allclose(mount_head(check_transform, offsets), video_anchor, atol=1e-10)
    model, camera = v3.ref.common.setup_model()
    rest = (model.layer['neutral'].J_regressor @ model.layer['neutral'].v_template).detach().numpy()
    cameras = [camera(path) for path in camera_paths()]
    adapters = [read(path)['input_adapter'] for path in camera_paths()]
    rows, observations = [], {}
    for entry in manifest:
        pair = entry['pair']
        sample = samples[pair]
        source = rig[sample['sequence'].rsplit('/', 1)[0]]
        assert (source['calibration_sequence'], source['calibration_frame']) != (sample['sequence'], sample['frame'])
        stereo = np.asarray(source['left_to_right'])
        views = {side: {'camera': {'left_to_view': transform.tolist()}}
                 for side, transform in zip(SIDES, (np.eye(4), stereo))}
        points = [dict({k: t[k] for k in ('index', 'name', 'native_index', 'group', 'accepted')},
                       xy=[], valid=[]) for t in template]
        obs = {}
        for side, image_id, adapter in zip(SIDES, sample['image_ids'], adapters):
            image = images[image_id]
            photo = MANIFEST.parent / image['image']
            digest = entry['eyes'][side]['image_sha256']
            assert sha(photo) == digest == body[image_id]['image_sha256'] == nose[image_id]['image_sha256']
            target = OUT / 'images' / f'{pair}_{side}.png'
            if not target.exists():
                target.symlink_to(photo)
            assert sha(target) == digest
            obs[side] = dict(nose_xy=nose[image_id]['xy'][178], score=nose[image_id]['score'][178],
                             rw20000_score=body[image_id]['score'][178], image_sha256=digest,
                             probability_sha256=entry['eyes'][side]['probability_sha256'])
            for point, index in zip(points, landmark_map):
                xy = np.asarray(body[image_id]['xy'][index])
                point['xy'].append((xy * adapter['scale'] + adapter['offset']).tolist())
                point['valid'].append(bool(((xy >= 0) & (xy < 872)).all()))
        anchor = mount_head(stereo, offsets)
        row = dict(pair=pair, condition='baseline', inputs={'views': views}, points=points,
                   head_position=anchor.tolist())
        seed, reason = v3.base.trial.body_seed(row, cameras)
        if reason:
            seed = v3.base.shoulder_hip_recovery(row, cameras, rest)
        row['initial_body_rotation'] = seed.tolist()
        rows.append(row)
        observations[pair] = dict(subject=entry['subject'], eyes=obs, mount_cam3_m=anchor.tolist(),
                                  body_seed_fallback=reason)
    folder = v3.shard(0)
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'prepared.json', dict(items=rows))
    protocol = (v3.base.OUT / 'protocol.json').read_bytes()
    (folder / 'protocol.json').write_bytes(protocol)
    write(RUN / 'evidence/inputs.json', dict(shards={'0': sha(folder / 'prepared.json')},
          protocol=sha(folder / 'protocol.json'), calibration=sha(PREP), cameras=list(map(sha,camera_paths())),
          body_prediction=sha(PRED), nose_prediction=sha(NOSE), video003_mount_cam3_m=video_anchor,
          rows=observations, exit_status=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    v3.RUN, v3.OUT, v3.FIT = RUN, OUT, OUT / 'fit'
    v3.ref.camera_files = camera_paths
    sys.addaudithook(v3.base.fit.forbid_gt)
    task = sys.argv[1]
    if task == 'prepare':
        prepare()
    else:
        getattr(v3, task)(0)
