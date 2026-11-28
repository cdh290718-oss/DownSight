"""One-pair neck/head ablation on the original calibrated stereo observations."""
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
from smplx.joint_names import JOINT_NAMES

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / '2026-11-27_video34_elbow_audit'))
import audit as a
from calibrate import rays
sys.path.insert(0, str(RUN.parent / '2026-11-26_realvideo4_head_mount'))
from runpy import run_path
face_measure = run_path(str(RUN.parent / '2026-11-26_realvideo4_head_mount' / 'measure.py'))
observed_edge, windows = face_measure['observed_edge'], face_measure['windows']

KEY = 'v003_04'
NECK_HEAD = [11, 14]
SIDES = ('left', 'right')
# Approximate near-side oral commissure in each original image, held out in contour-only fitting and included in the joint fit.
CORNERS = np.array([[850, 663], [485, 655]])
torch.set_num_threads(1)
cv2.setNumThreads(1)
ref = a.original.w.ref
ref.OLD = ref.OUT = ref.compare.OUT = a.OUT
ref.PRIOR = a.PRIOR
ref.cameras = ref.prev.ref.camera_files = ref.prev.base.fit.camera_files = a.cameras
model, cam = ref.prev.ref.common.setup_model()
layer = model.layer['neutral'].eval()
faces = np.asarray(model.face, np.int32)
head_faces = np.isin(layer.lbs_weights.argmax(1).numpy(), [15, 22, 23, 24])[faces].all(1)
topology = o3d.core.Tensor(faces.astype(np.uint32))
receipt = a.read(a.PRIOR / 'evidence' / (KEY + '_render.json'))['source']
pose_file = a.OUT / 'palm/poses' / (KEY + '.npz')
assert a.sha(pose_file) == receipt['palm_sha256']
pack = dict(np.load(pose_file))
seed = pack['locked'].copy()
assert not seed[NECK_HEAD].any()
prep = a.OUT / 'fit/0/prepared.json'
pred = a.OUT / 'predictions.json'
assert a.sha(prep) == a.read(a.PRIOR / 'evidence/inputs.json')['shards']['0']
assert a.sha(pred) == a.read(a.PRIOR / 'evidence/detection.json')['predictions_sha256']
row = next(r for r in a.read(prep)['items'] if r['pair'] == KEY)
target = a.elbow_observations(row, ['L', 'R'], a.read(pred), KEY)
mask = ((target >= 0) & (target < [1280, 720])).all(2)
tfs = [np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in SIDES]
cams = list(map(a.read, a.cameras()))
seg = a.read(a.PRIOR / 'evidence/segmentation.json')['eyes']
edges, hashes, beams = [], [], []
rows = np.arange(350, 720)
for side, window, tf, camera in zip(SIDES, windows['clip003'], tfs, cams):
    name = KEY + '_' + side
    path = a.OUT / 'images' / (name + '.png')
    hashes.append(a.sha(path))
    assert hashes[-1] == seg[name]['image_sha256']
    edge = observed_edge(cv2.imread(str(path)), window)
    edges.append(edge)
    xx, yy = np.meshgrid(edge[:, 0], rows)
    dirs = rays(np.stack([xx.ravel(), yy.ravel()], axis=1), camera) @ tf[:3, :3]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origin = -tf[:3, :3].T @ tf[:3, 3]
    beams.append(np.column_stack([np.broadcast_to(origin, dirs.shape), dirs]))
beam = o3d.core.Tensor(np.vstack(beams).astype(np.float32))
observed = np.array(edges)[:, :, 1]

def mesh(pose):
    return a.pack_pose(layer, pack, pose)

def contour(vertices):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32)), topology)
    ids = scene.cast_rays(beam, nthreads=1)['primitive_ids'].numpy()
    visible = ids < len(faces)
    visible[visible] = head_faces[ids[visible]]
    hits = visible.reshape(2, len(rows), -1)
    return np.where(hits.any(1), rows[hits.argmax(1)], 720)

def project(joints):
    return np.stack([a.projection(joints @ t[:3, :3].T + t[:3, 3], c) for c,t in zip(cams, tfs)])

def correct_arm(start):
    return a.replay(dict(pack, locked=start), target, mask, tfs, model, cam, row, KEY, 1)

base = correct_arm(seed)
purple_seed = seed.copy()
purple_seed[22] += np.array(receipt['purple_translation_m'])
purple = correct_arm(purple_seed)
assert np.array_equal(base[NECK_HEAD], purple[NECK_HEAD])
_, base_j = mesh(base)

def unlocked(degrees):
    pose = base.copy()
    pose[NECK_HEAD] = np.deg2rad(degrees).reshape(2, 3)
    return pose

# Six bounded local rotation components. Fit alternating columns only.
# Pose regularizer and two starts reduce ambiguity in sharing the bend.
use_mouth = False
cache = {}
mouth_ids = [JOINT_NAMES.index(s + "_mouth_1") for s in SIDES]
def loss(degrees):
    key = tuple(degrees)
    if key not in cache:
        if max(np.linalg.norm(degrees.reshape(2, 3), axis=1)) > 35:
            return np.inf
        v, j = mesh(unlocked(degrees))
        predicted = contour(v)
        cache[key] = np.abs(predicted[:, ::2] - observed[:, ::2]).mean() + .003 * (degrees**2).sum()
        if use_mouth:
            m = project(j[mouth_ids])[[0, 1], [0, 1]]
            cache[key] += float(np.sqrt(np.sum((m-CORNERS)**2, axis=1).mean()))
    return cache[key]

def search(initial):
    x = np.array(initial, float)
    for step in (8, 4, 2, 1, .5):
        for _ in range(16):
            candidates = [x, *(x + step * d for d in np.vstack([np.eye(6), -np.eye(6)]))]
            best = min(candidates, key=loss)
            if np.array_equal(x, best):
                break
            x = best
    return x

starts = [np.zeros(6), np.array([12, 0, 0, -8, 0, 0])]
best = min([search(s) for s in starts], key=loss)
local = unlocked(best)
contour_count = len(cache)
use_mouth = True
cache.clear()
from scipy.optimize import least_squares

def mouth_residual(degrees):
    _, j = mesh(unlocked(degrees))
    points = project(j[mouth_ids])[[0, 1], [0, 1]]
    return np.r_[(points-CORNERS).ravel(), .45*degrees]

def jacobian(degrees):
    return np.column_stack([(mouth_residual(degrees+d)-mouth_residual(degrees-d))/.2 for d in .1*np.eye(6)])

mouth_start = least_squares(mouth_residual, np.zeros(6), jac=jacobian, bounds=(-35,35)).x.reshape(2,3)
mouth_start *= np.minimum(1, 34.5/np.maximum(np.linalg.norm(mouth_start, axis=1, keepdims=True), .01))
joint_best = min([search(s) for s in [best, np.zeros(6), mouth_start.ravel()]], key=loss)
joint = unlocked(joint_best)
entries = {}
meshes = {}
for name, pose in [('original', base), ('purple', purple), ('local', local), ('local_joint', joint)]:
    v, j = mesh(pose)
    meshes[name] = (v, j)
    y = contour(v)
    e = np.abs(y - observed)
    mouth_ids = [JOINT_NAMES.index(s + '_mouth_1') for s in SIDES]
    mouth = project(j[mouth_ids])[[0, 1], [0, 1]]
    uv = project(j[[18, 19, 20, 21]])
    diff = np.linalg.norm(uv - target, axis=2)
    arm_shift = np.linalg.norm(j[16:22] - base_j[16:22], axis=1) * 1000
    entries[name] = dict(contour_y=y.tolist(), train_mae=float(e[:, ::2].mean()),
        held_mae=float(e[:, 1::2].mean()), held_mae_eyes=e[:, 1::2].mean(1).tolist(),
        mouth_uv=mouth.tolist(), mouth_rmse=float(np.sqrt(((mouth-CORNERS)**2).sum(1).mean())),
        local_rotvec_deg=np.rad2deg(pose[NECK_HEAD]).tolist(),
        shoulder_elbow_wrist_uv=project(j[16:22]).tolist(), shift_mm=arm_shift.tolist(),
        elbow_wrist_rmse=[float(np.sqrt(np.mean(diff[:, sl][mask[:, sl]]**2))) for sl in (slice(2), slice(2,4))],
        flexion_deg=[a.angle(j[18+i]-j[16+i], j[20+i]-j[18+i]) for i in range(2)])
assert all(max(entries[n]['shift_mm']) < .001 for n in ['local','local_joint'])
a.write(RUN/'evidence/results.json', dict(pair=KEY, source_commit='965d1cf4b1aab97a97f5233f628d9b20eca4cd87',
    image_sha256=hashes, palm_sha256=a.sha(pose_file), prepared_sha256=a.sha(prep),
    predictions_sha256=a.sha(pred), cameras_sha256=list(map(a.sha,a.cameras())),
    observed_xy=np.array(edges).tolist(), near_mouth_xy=CORNERS.tolist(), elbow_wrist_targets=target.tolist(),
    translation_m=receipt['purple_translation_m'], edge_candidates=contour_count, joint_candidates=len(cache), results=entries))

# Publish only model geometry and connected image measurements on blank backgrounds.
colors = {'original':(155,155,155), 'purple':(190,90,165), 'local':(40,155,55), 'local_joint':(200,100,30)}
panels = []
for eye, side in enumerate(SIDES):
    panel = np.full((330,530,3), 246, np.uint8)
    x0 = edges[eye][0,0]
    def field(xy):
        return np.rint((xy-[x0,350])*[3,.8]+[35,24]).astype(np.int32)
    for name, result in entries.items():
        pts = field(np.column_stack([edges[eye][:,0],result['contour_y'][eye]]))
        cv2.polylines(panel,[pts],False,colors[name],2,cv2.LINE_AA)
    for i,p in enumerate(field(edges[eye])):
        cv2.circle(panel,tuple(p),3,(40,40,40),1 if i%2==0 else -1)
    cv2.putText(panel,side+'  gray base / purple / green edge / blue joint',(14,20),0,.62,(35,35,35),1,cv2.LINE_AA)
    panels.append(panel)
assert cv2.imwrite(str(RUN/'assets/contour.png'),np.hstack(panels))
basis = a.original.w.old.mat(base[21]) @ a.original.w.old.mat(pack['global0']) @ a.original.w.old.mat([0,np.pi/2,0])
coords = np.vstack([v @ basis for v,j in meshes.values()])
center = (coords.min(0)+coords.max(0))/2
scale = min(316/np.ptp(coords[:,0]),440/np.ptp(coords[:,1]))
renders = [a.title(a.free_render(v,faces,basis,center,scale),name) for name,(v,j) in meshes.items()]
assert cv2.imwrite(str(RUN/'assets/body.jpg'),np.hstack(renders))
for name,r in entries.items():
    print(name,r['held_mae'],r['mouth_rmse'],r['elbow_wrist_rmse'],r['flexion_deg'],flush=True)
