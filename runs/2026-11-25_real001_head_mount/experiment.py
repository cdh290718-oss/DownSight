"""Prepare position-only hypotheses and run the unchanged v3 fitting and candidate search."""
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'runs/2026-11-25_real001_golden_d_v3'
sys.path.insert(0, str(PREVIOUS))
import run_pair as original
from render_result import geometry, rot

OUT = ROOT / 'outputs/real001_head_mount_20261125'
SOURCE = original.OUT / 'fit/0'
v3, read, write, sha = original.v3, original.read, original.write, original.sha
CASES = ['historical_left', 'historical_right', 'historical_mean', 'mount', 'mount_minus10', 'mount_plus10']


def unit(v):
    return v / np.linalg.norm(v)


def reconstructed_joint(layer, path):
    with np.load(path, allow_pickle=False) as parameters:
        keys = ['global_orient', 'transl', 'betas', 'body_pose', 'jaw_pose', 'leye_pose',
                'reye_pose', 'expression', 'left_hand_pose', 'right_hand_pose']
        tensors = {k: torch.tensor(parameters[k], dtype=torch.float32).reshape(1, -1) for k in keys}
    with torch.no_grad():
        return layer(**tensors).joints[0, 15].numpy()


def prepare():
    for record in read(ROOT / 'configs/golden_d.2d-v3.json')['locked_files']:
        assert sha(ROOT / record['path']) == record['sha256']
    prior = read(PREVIOUS / 'evidence/inputs.json')
    assert sha(SOURCE / 'prepared.json') == prior['shards']['0']
    assert sha(SOURCE / 'protocol.json') == prior['protocol_sha256']
    row = read(SOURCE / 'prepared.json')['items'][0]
    transform = np.array(row['inputs']['views']['right']['camera']['left_to_view'])
    rotation, translation = transform[:3, :3], transform[:3, 3]
    right_center = -rotation.T @ translation
    right = unit(right_center)
    midpoint = right_center / 2
    down = unit(np.array([0., 0., 1.]) + rotation.T @ [0., 0., 1.])
    up = -unit(down - right * np.dot(down, right))
    forward = unit(np.cross(-right, up))
    assert right[0] > 0 and up[2] < 0 and forward[1] < 0
    model, _ = v3.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    zeros = dict(global_orient=np.zeros(3), transl=np.zeros(3), betas=np.zeros(10), body_pose=np.zeros((21,3)))
    _, joints, _, _ = geometry(layer, zeros, np.zeros(3), False, None)
    brow_ids = [JOINT_NAMES.index(n) for n in ['right_eye_brow5','left_eye_brow5']]
    assert np.isclose(np.min(abs(joints[76:81,0])), abs(joints[brow_ids[0],0]))
    assert np.isclose(np.min(abs(joints[81:86,0])), abs(joints[brow_ids[1],0]))
    brow_to_head = joints[15] - joints[brow_ids].mean(axis=0)
    archive = original.OLD / 'download_bundle/02_legacy_three_methods'
    files = [next(archive.glob('*legacy_original__'+eye+'_cv.npz')) for eye in ('cam3','cam4')]
    left = reconstructed_joint(layer, files[0])
    other = rotation.T @ (reconstructed_joint(layer, files[1]) - translation)
    assert np.allclose(left, row['head_position'], atol=1e-6)
    anchors = [np.array(row['head_position']), other, (left+other)/2]
    brow_positions = {}
    for name, angle in zip(CASES[3:], [0., -10., 10.]):
        pitch = Rotation.from_rotvec(right * np.deg2rad(angle)).as_matrix()
        basis = pitch @ np.stack([-right,up,forward],axis=1)
        assert np.allclose(np.linalg.det(basis),1.)
        brow = midpoint + basis @ [0., .05, -.05]
        brow_positions[name] = brow.tolist()
        anchors.append(brow + basis @ brow_to_head)
    dest = v3.shard(0)
    dest.mkdir(parents=True,exist_ok=True)
    images = OUT / 'images'
    images.mkdir(exist_ok=True)
    for label in CASES[1:]:
        for side in original.sides:
            link = images / (label+'_'+side+'.png')
            if not link.exists():
                link.symlink_to(original.OUT / 'images' / ('real001_'+side+'.png'))
        assert np.isfinite(anchors[CASES.index(label)]).all()
    write(dest / 'prepared.json', dict(items=[dict(row, pair=k, head_position=a.tolist()) for k,a in zip(CASES[1:],anchors[1:])]))
    shutil.copyfile(SOURCE / 'protocol.json',dest / 'protocol.json')
    write(RUN/'evidence/inputs.json',dict(shards={'0':sha(dest/'prepared.json')},source_detector=prior['detector_sha256'],
        images_calibrations=prior['inputs'],source_prepared=sha(SOURCE/'prepared.json'),protocol=sha(SOURCE/'protocol.json'),
        history_params={f.name:sha(f) for f in files},right_center_m=right_center.tolist(),midpoint_m=midpoint.tolist(),
        unit_up=up.tolist(),unit_forward=forward.tolist(),unit_right=right.tolist(),camera_separation_m=float(np.linalg.norm(right_center)),
        landmarks=brow_ids,brow_to_head_template_m=brow_to_head.tolist(),brows_cam3_m=brow_positions,
        anchors_cam3_m={k:a.tolist() for k,a in zip(CASES,anchors)},history_disagreement_m=float(np.linalg.norm(left-other)),
        historical_to_mount_m=float(np.linalg.norm(left-anchors[3])),exit_status=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    v3.RUN, v3.OUT, v3.FIT = RUN, OUT, OUT/'fit'
    v3.ref.camera_files = original.cameras
    mode = sys.argv[1]
    if mode == 'prepare':
        prepare()
    else:
        sys.addaudithook(v3.base.fit.forbid_gt)
        {'fit':v3.fit, 'candidates':v3.candidates}[mode](0)
