"""Measure nearest training upper/lower segment direction pair in a torso frame."""
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial import cKDTree

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/conditional_limbs_20261124'
NAMES = [f'{side}_{joint}' for side in ('L','R') for joint in ('Shoulder','Elbow','Wrist','Hip','Knee','Ankle')]
CHAINS = [(0,1,2),(6,7,8),(3,4,5),(9,10,11)]
NATIVE = [16,18,20,1,4,7,17,19,21,2,5,8]


def unit(x):
    norm = np.linalg.norm(x,axis=-1,keepdims=True)
    if np.min(norm) < 1e-8:
        raise ValueError('degenerate torso or bone')
    return x/norm


def directions(x):
    lateral = unit(x[:,0]-x[:,6]+x[:,3]-x[:,9])
    up = x[:,0]+x[:,6]-x[:,3]-x[:,9]
    up = unit(up-(up*lateral).sum(-1,keepdims=True)*lateral)
    axes = np.stack([lateral,up,np.cross(lateral,up)],axis=-1)
    chains = []
    for a,b,c in CHAINS:
        bones = np.stack([unit(x[:,b]-x[:,a]),unit(x[:,c]-x[:,b])],axis=1)
        chains.append((bones@axes).reshape(-1,6))
    return np.stack(chains,axis=1)


def distances(trees, values):
    # Repeated chain features in a vectorized batch arise from changing other limbs.
    result = []
    for k,tree in enumerate(trees):
        arr = np.ascontiguousarray(values[:,k])
        _, first, back = np.unique(arr.view(np.dtype((np.void,arr.itemsize*6))).ravel(),
                                   return_index=True,return_inverse=True)
        result.append(tree.query(arr[first],k=1)[0][back])
    return np.stack(result,axis=1)


def train():
    folder = ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Train'
    with zipfile.ZipFile(folder/'official_splits/UnrealEgoData_rw.zip') as z:
        seqs = z.read('UnrealEgoData_rw/train.txt').decode().splitlines()
    names = json.loads((folder/'prepared/metadata.json').read_text())['train_joint_names']
    with np.load(folder/'prepared/samples.npz') as z:
        seq, side = z['source_sequence'],z['camera_side']
        keep = np.isin(seq,seqs) & (side=='left')
        assert len(set(zip(seq[keep],z['frame_index'][keep])))==sum(keep)==79705
        feat = directions(z['keypoints3d'][keep][:,[names.index(n) for n in NAMES]].astype(float))
    subject = np.array([s.split('/')[2] for s in seq[keep]])
    heldout_dist = np.empty(feat.shape[:2])
    for person in np.unique(subject):
        held = subject==person
        trees = [cKDTree(feat[~held,k]) for k in range(4)]
        heldout_dist[held] = distances(trees,feat[held])
    threshold = np.percentile(heldout_dist,95,axis=0)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'training.npz',features=feat,threshold=threshold)
    (RUN/'evidence/training.json').write_text(json.dumps(dict(frames=len(feat),people=np.unique(subject).tolist(),
            chain_order=['L_arm','R_arm','L_leg','R_leg'],threshold=threshold.tolist(),
            statistic='95th percentile of leave-one-subject-out nearest 6D distance',weight=.01),indent=2)+'\n')


if __name__=='__main__':
    train()
