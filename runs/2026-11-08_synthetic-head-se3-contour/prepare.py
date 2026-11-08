"""Extract only Head SE(3) from truth; build contours solely from Sapiens."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from contour import make_targets

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT/'outputs/synthetic_head_se3_contour_20261108'
OLD = ROOT/'outputs/synthetic_original_20261108'
DATA = OLD/'dataset/samples/000001'
cfg = json.loads((RUN/'settings.json').read_text())
(RUN/'assets').mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(DATA/'smplx_reconstruction.npz', allow_pickle=False) as p:
    head_local_R = cv2.Rodrigues(p['global_orient'].astype(float).reshape(3))[0]
    for joint in [3, 6, 9, 12, 15]:
        head_local_R = head_local_R@cv2.Rodrigues(p['body_pose'].reshape(21, 3)[joint-1].astype(float))[0]
with np.load(DATA/'geometry.npz', allow_pickle=False) as p:
    object_R = p['object_matrix_world'][:3, :3].astype(float)
assert np.allclose(object_R.T@object_R, np.eye(3), atol=1e-6) and np.linalg.det(object_R) > .999
records = []
for side, dirname in [('left', 'left'), ('right', 'right_cpu')]:
    job = json.loads((OLD/'fits'/dirname/'jobs'/(side+'.json')).read_text())
    seg = Path(job['row']['predictions']['segmentation'])
    labels = np.load(seg/'labels.npy')
    with np.load(seg/'probabilities.npz', allow_pickle=False) as z:
        probabilities = z['probabilities'].astype(np.float32)
    keypoints = json.loads(Path(job['row']['predictions']['sapiens']).read_text())
    lookup = {p['name']: p for p in keypoints['keypoints']}
    image_path = DATA/(side+'.png')
    stats = json.loads((seg/'stats.json').read_text())
    assert stats['image_sha256'] == sha(image_path) == keypoints['image_sha256']
    with np.load(DATA/(side+'_vertices.npz'), allow_pickle=False) as p:
        position = p['joints_camera_cv_m'][15].copy()
        camera_R = p['world_to_camera_cv'][:3, :3].astype(float)
    rotation = camera_R@object_R@head_local_R
    assert np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) and np.linalg.det(rotation) > .999
    targets = make_targets(labels, probabilities, lookup, cfg)
    metadata = dict(side=side, head_position_m=position.tolist(), head_rotation_matrix=rotation.tolist(),
                    head_frame_definition='native SMPL-X joint 15 frame; camera_R @ object_R @ FK(global,spine1,spine2,spine3,neck,head)',
                    head_chain=[0, 3, 6, 9, 12, 15], no_other_gt_in_optimizer=True,
                    image_sha256=sha(image_path), segmentation_labels_sha256=sha(seg/'labels.npy'),
                    segmentation_probabilities_sha256=sha(seg/'probabilities.npz'),
                    class_pixels={str(int(k)): int(n) for k, n in zip(*np.unique(labels, return_counts=True))},
                    targets={name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items() if k!='arm_exclusion'} for name, d in targets.items()},
                    gt_resources_sha256={n:sha(DATA/n) for n in ['smplx_reconstruction.npz', 'geometry.npz', side+'_vertices.npz']})
    (OUT/(side+'_targets.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    np.savez_compressed(OUT/(side+'_contour_masks.npz'), torso_arm_exclusion=targets['torso']['arm_exclusion'])
    (RUN/'evidence'/(side+'_targets.json')).write_text(json.dumps(metadata, indent=2)+'\n')
    overlay = cv2.imread(str(image_path))
    for name, color in [('torso', (0, 230, 230)), ('legs', (230, 0, 230))]:
        d = targets[name]
        for p in d['target']:
            cv2.circle(overlay, tuple(np.rint(p).astype(int)), 3, color, -1)
    cv2.rectangle(overlay, (0, 0), (1280, 75), (25, 25, 25), -1)
    for i, name in enumerate(['torso', 'legs']):
        d = targets[name]
        title = '%s: active=%s, class pixels=%d, exterior points=%d' % (name, d['active'], d['class_pixels'], d['boundary_points'])
        cv2.putText(overlay, title, (12, 26+i*30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(RUN/'assets'/(side+'_contour_targets.jpg')), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
    records.append(dict(side=side, targets={n:{k:v for k,v in d.items() if k not in ['target','arm_exclusion']} for n,d in targets.items()}))
(RUN/'evidence/preparation.json').write_text(json.dumps(dict(exit_status=0, records=records), indent=2)+'\n')
print(json.dumps(records, indent=2))
