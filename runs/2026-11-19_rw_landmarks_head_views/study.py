"""Calibrate landmarks on official training sequences and run full-2D controlled fits."""
import copy
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/rw_landmarks_head_views_20261119'
REF = ROOT/'runs/2026-11-17_newdetector_d10'
OLD = ROOT/'outputs/unrealego_gt2d_ideal_20261115'
sys.path.insert(0, str(REF))
import common
read, write, sha = common.read, common.write, common.sha
MODES = [('baseline',12,'historical'), ('15history',15,'historical'),
         ('12camera',12,'camera'), ('15camera',15,'camera'),
         ('12torso',12,'torso'), ('15torso',15,'torso')]


def unit(v):
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    assert np.min(lengths) > 1e-5
    return v / lengths


def closest_rotation(m):
    u, _, vt = np.linalg.svd(m)
    return (u * [1, 1, np.linalg.det(u@vt)]) @ vt


def prepare():
    model, _ = common.setup_model()
    layer = model.layer['neutral']
    rest = (layer.J_regressor @ layer.v_template).detach().numpy()
    train = ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Train'
    split = train/'official_splits/UnrealEgoData_rw.zip'
    with zipfile.ZipFile(split) as z:
        seqs = z.read('UnrealEgoData_rw/train.txt').decode().splitlines()
    meta = read(train/'prepared/metadata.json')
    names = meta['fit_joint_names']
    with np.load(train/'prepared/samples.npz') as z:
        seq, eye = z['source_sequence'], z['camera_side']
        take = np.isin(seq, seqs) & (eye=='left')
        pts = z['keypoints3d_fit'][take].astype(float)
        subjects = np.array([s.split('/')[2] for s in seq[take]])
    assert 'S9' not in subjects
    head, neck, ls, rs = [pts[:,names.index(k)] for k in ['Head','Neck','L_Shoulder','R_Shoulder']]
    up = unit(head-neck)
    # The shoulder line supplies roll/twist information as a proxy, not a head sensor.
    lateral = ls-rs
    lateral = unit(lateral-(lateral*up).sum(1)[:,None]*up)
    frames = np.stack([lateral,up,np.cross(lateral,up)],axis=2)
    assert (rest[16]-rest[17])[0] > 0 and (rest[15]-rest[12])[1] > 0
    lengths = [np.linalg.norm(head-neck,axis=1)]
    bone_lengths = [np.linalg.norm(rest[15]-rest[12])]
    for side, ankle, foot in [('R',8,11),('L',7,10)]:
        lengths.append(np.linalg.norm(pts[:,names.index(side+'_Toe')]-pts[:,names.index(side+'_Ankle')],axis=1))
        bone_lengths.append(np.linalg.norm(rest[foot]-rest[ankle]))
    medians, rotations = [], []
    for s in np.unique(subjects):
        mask = subjects==s
        medians.append([np.median(a[mask]) for a in lengths])
        rotations.append(closest_rotation(frames[mask].mean(0)))
    ratios = np.median(medians,axis=0)/bone_lengths
    cam = closest_rotation(np.mean(rotations,axis=0))
    deviations = np.rad2deg(np.arccos(np.clip(((frames*cam).sum((1,2))-1)/2,-1,1)))
    calibration = dict(training_subjects=np.unique(subjects).tolist(),training_left_views=len(pts),
        training_split_sha256=sha(split),prepared_train_sha256=sha(train/'prepared/samples.npz'),
        training_sequences=len(set(seq[take])),order=['Neck','R_ToeBase','L_ToeBase'],
        train_length_medians_per_person_mm=(np.array(medians)*1000).tolist(),
        template_lengths_mm=(np.array(bone_lengths)*1000).tolist(),ratios=ratios.tolist(),
        camera_R_head_proxy=cam.tolist(),train_proxy_deviation_deg_p50_p90=np.percentile(deviations,[50,90]).tolist(),
        camera_frame='SMPL-X +X to anatomical left, +Y up, +Z cross; training left camera CV; equal weight per subject',
        limits='Fixed camera-relative orientation proxy, no measured anatomical head-to-camera mounting rotation.')
    write(RUN/'evidence/calibration.json',calibration)
    prev=ROOT/'outputs/rw_head_neck_oracle_20261119'
    prior=read(prev/'prepared.json')['items']
    items=[]
    for base in prior:
        obs=read(OLD/base['pair']/'observations_local.json')
        for label,count,mode in MODES:
            item=copy.deepcopy(base)
            item.update(condition=label,head_mode=mode)
            for name,virtual in [('Neck',22),('R_Toe',23),('L_Toe',24)]:
                j=obs['names'].index(name)
                xy=np.array(obs['views'])[:,j]
                assert (xy[:,2]>0).all() and np.isfinite(xy).all()
                item['points'].append(dict(name=name,index=obs['indices'][j],native_index=virtual,
                    group='body',accepted=False,xy=xy[:,:2].tolist(),valid=[count==15]*2))
            items.append(item)
    cfg=read(prev/'protocol.json')
    cfg.update(conditions=[k for k,_,_ in MODES], landmark_ratios=ratios.tolist(),
               camera_R_head_proxy=cam.tolist(),scope='Paired change in landmark mask and orientation handling',
               baseline_commit='736ee9d7ca77b54571a91f0a07648c5ad189bd1d')
    write(OUT/'protocol.json',cfg)
    write(OUT/'prepared.json',dict(items=items))


def optimize():
    import torch
    import fit
    class LandmarkProblem(fit.Problem):
        def __init__(self, prepared, cfg, model, camera_cls):
            super().__init__(prepared,cfg,model,camera_cls)
            assert all(x['rank']==0 for x in self.initialization.values())
            self.ratios=cfg['landmark_ratios']
            self.torso=torch.tensor([x['head_mode']=='torso' for x in self.items])
            for i,item in enumerate(self.items):
                if item['head_mode']=='camera':
                    self.head_rot[i]=self.t(cfg['camera_R_head_proxy'])
            # super already constructed identical starting root and translation for every mode.
            self.movable=torch.ones((len(self.items),21,1))
            for j in [12,15]:
                self.movable[self.torso,j-1]=0

        def parameters(self):
            body,beta=super().parameters()
            return body*self.movable,beta

        def forward(self):
            native, rh, beta=super().forward()
            proxies=[native[:,a]+k*(native[:,b]-native[:,a])
                     for a,b,k in zip([15,8,7],[12,11,10],self.ratios)]
            return torch.cat([native,torch.stack(proxies,dim=1)],dim=1),rh,beta

        def objective(self,values):
            native,rh,beta=values
            # Torso condition has identity local Neck and Head rotations and no absolute rotation target.
            target_rh=torch.where(self.torso[:,None,None],self.head_rot,rh)
            return super().objective((native,target_rh,beta))
    fit.Problem=LandmarkProblem
    fit.RUN=fit.OUT=OUT
    sys.argv=[__file__,'--threads','1']
    fit.main()


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        prepare()
    elif sys.argv[1]=='fit':
        optimize()
