"""Common anatomically named spans; parameters and raw observations remain local."""
import hashlib
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/rw_subject_shape_20261124'
EV = RUN/'evidence'
sys.path.insert(0, str(ROOT/'runs/2026-11-23_stereo_wrist_candidates'))
import shared
ref, read, write, sha = shared.ref, shared.read, shared.write, shared.sha
SPANS = [(s+'_'+a, s+'_'+b) for s in ('L','R') for a,b in
         [('Shoulder','Elbow'),('Elbow','Wrist'),('Hip','Knee'),('Knee','Ankle')]]
SPANS += [('L_Shoulder','R_Shoulder'),('L_Hip','R_Hip')]
BANK = ROOT/'outputs/fullbody_joint_candidates_20261123'
FOUR = {'pair068':['L_hand / R_hand'], 'pair055':['L_hand / L_thigh'],
        'val031':['L_forearm / L_upperarm','R_forearm / R_upperarm'],
        'val055':['L_forearm / L_upperarm','R_forearm / R_upperarm']}


def sizes(points, index):
    a,b = [[index[n] for n in col] for col in zip(*SPANS)]
    return np.linalg.norm(points[...,a,:]-points[...,b,:],axis=-1)


def ratios(lengths):
    leg = lengths[..., [2,3,6,7]].sum(axis=-1)/2
    return 100 * lengths / leg[...,None]


class ShapeModel:
    def __init__(self, layer):
        import torch
        self.rest = (layer.J_regressor @ layer.v_template).detach().numpy()[:22].astype(float)
        self.dirs = torch.einsum('jv,vck->jck',layer.J_regressor[:22],layer.shapedirs[:,:,:10]).detach().numpy().astype(float)
        self.index = dict(zip(ref.common.NAMES, ref.common.NATIVE_IDS))

    def predict(self, beta):
        return sizes(self.rest + self.dirs@beta, self.index)

    def solve(self, target, scale):
        mask = np.isfinite(target)
        def residual(beta):
            return np.r_[(self.predict(beta)[mask]-target[mask])/scale, beta]
        fit = least_squares(residual, np.zeros(10), bounds=(-3.,3.), loss='linear')
        assert fit.success and np.isfinite(fit.x).all()
        return fit.x


def inputs(split):
    path = shared.TM if split=='test' else shared.VM
    source = (shared.base.OUT if split=='test' else shared.VAL)/'prepared.json'
    expected = (read(shared.base.RUN/'evidence/inputs.json')['solver_input_sha256'] if split=='test' else
                read(shared.RUN/'evidence/splits.json')['hashes']['validation_prepared'])
    assert sha(source)==expected
    assert sha(path)==read(shared.RUN/'evidence/splits.json')['hashes'][split+'_manifest']
    images = shared.pair_images(path)
    choices = read(BANK/(split+'_choices.json'))
    rows = read(source)['items']
    for pair,r in choices.items():
        item = rows[r['index']]
        im = images[int(pair[-3:])-1]
        assert item['pair']==pair
        assert item['condition']==('rw20000' if split=='test' else 'baseline')
        yield pair, item, im, r


def rank_sequence(seq):
    return hashlib.sha256(('DownSight-shape-v1/'+seq).encode()).hexdigest()
