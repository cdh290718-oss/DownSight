"""Strict edge-through-triangle intersections of the named exported surfaces."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_frozen200_free12'))
import measure as ref


def cuts(a, b):
    # Moller-Trumbore with a finite segment and strict barycentric interior.
    e1, e2 = b[:, 1] - b[:, 0], b[:, 2] - b[:, 0]
    found = np.zeros(len(a), dtype=bool)
    for edge in range(3):
        start = a[:, edge]
        direction = a[:, (edge + 1) % 3] - start
        off = start - b[:, 0]
        p = np.cross(direction, e2)
        q = np.cross(off, e1)
        det = np.einsum('ij,ij->i', e1, p)
        inv = np.divide(1., det, out=np.zeros_like(det), where=abs(det)>1e-16)
        u = np.einsum('ij,ij->i', off, p) * inv
        v = np.einsum('ij,ij->i', direction, q) * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
        eps = 1e-7
        found |= (u>eps) & (v>eps) & (u+v<1-eps) & (t>eps) & (t<1-eps)
    return found


def self_intersections(verts, faces):
    triangles = verts[faces]
    lo, hi = triangles.min(1), triangles.max(1)
    center = (lo+hi)/2
    radius = np.linalg.norm(hi-lo, axis=1)/2
    tree = cKDTree(center)
    hits = []
    for first in range(0, len(faces), 128):
        idx = np.arange(first, min(first+128,len(faces)))
        nearby = tree.query_ball_point(center[idx], radius[idx]+radius.max())
        left = np.repeat(idx, [len(lst) for lst in nearby])
        right = np.concatenate(nearby)
        keep = (left<right) & np.all((lo[left]<=hi[right]) & (lo[right]<=hi[left]),axis=1)
        left, right = left[keep], right[keep]
        shared = (faces[left,:,None] == faces[right,None,:]).any(axis=(1,2))
        left, right = left[~shared], right[~shared]
        a,b = triangles[left],triangles[right]
        ok = cuts(a,b) | cuts(b,a)
        hits.extend(zip(left[ok],right[ok]))
    return hits


def groups(layer, faces):
    labels = ['trunk','L_thigh','R_thigh','trunk','L_shin','R_shin','trunk','L_foot','R_foot',
              'trunk','L_foot','R_foot','trunk','trunk','trunk','head','L_upperarm','R_upperarm',
              'L_forearm','R_forearm','L_hand','R_hand','head','head','head'] + ['L_hand']*15 + ['R_hand']*15
    weights = layer.lbs_weights.detach().numpy()
    assert len(labels) == weights.shape[1]
    names = sorted(set(labels))
    w = np.stack([weights[:,np.array(labels)==n].sum(1) for n in names],axis=1)
    return np.array(names)[w[faces].sum(1).argmax(1)]


def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


def main():
    torch.set_num_threads(1)
    model,_ = ref.common.setup_model()
    layer = model.layer['neutral']
    faces = np.asarray(model.face)
    labels = groups(layer, faces)
    archive = ROOT/'outputs/fixed_floor24_20261123/gallery'
    prior = ROOT/'runs/2026-11-23_fixed_floor24/evidence/gallery.json'
    records = {r['pair']:r for r in ref.read(prior)['rows']}
    bank = ROOT/'outputs/fullbody_joint_candidates_20261123'
    results = {}
    for pair in ['val055','val031','pair068','pair055']:
        split = 'test' if pair.startswith('pair') else 'validation'
        obj = archive/pair/(pair+'.obj')
        assert ref.sha(obj) == records[pair]['obj_sha256']
        verts, objfaces = [], []
        for line in obj.read_text().splitlines():
            fields = line.split()
            if fields[0]=='v':
                verts.append([float(x) for x in fields[1:]])
            elif fields[0]=='f':
                objfaces.append([int(x)-1 for x in fields[1:]])
        assert np.array_equal(faces,objfaces)
        hits = self_intersections(np.asarray(verts),faces)
        pairs = Counter(' / '.join(sorted((labels[a],labels[b]))) for a,b in hits)
        sel = ref.read(bank/(split+'_choices.json'))[pair]
        idx = sel['offset']+sel['fixed']
        params = dict(np.load(bank/(split+'_bank.npz')))
        pred = ref.full_joints(params,layer,idx)[ref.common.NATIVE_IDS]
        truth_file = ref.SOURCE/'reference.json' if split=='test' else ROOT/'outputs/stereo_wrist_candidates_20261123/validation_truth.json'
        gt = np.asarray(ref.read(truth_file)[pair]['gt'])
        scores = {}
        for side in ['L','R']:
            ids = [ref.common.NAMES.index(side+'_'+k) for k in ['Shoulder','Elbow','Wrist']]
            a,b = np.diff(pred[ids],axis=0)
            c,d = np.diff(gt[ids],axis=0)
            scores[side] = dict(fit_bend_deg=angle(a,b),gt_bend_deg=angle(c,d),
                                bend_normal_offset_deg=angle(np.cross(a,b),np.cross(c,d)),
                                upperarm_dir_error_deg=angle(a,c),forearm_dir_error_deg=angle(b,d))
        obj_sha=ref.sha(obj)
        results[pair] = dict(obj_sha256=obj_sha, choice=sel['names'][sel['fixed']], elbows=scores,
                             surface_crossing_face_pairs=dict(pairs))
        print(pair,scores,pairs,flush=True)
    ref.write(RUN/'evidence/results.json',dict(rows=results,
              command='<BODY_PYTHON> -Bu runs/2026-11-23_floor24_intersections/inspect_obj.py'))


if __name__=='__main__':
    main()
