"""Reconstruct metric wrist-relative MANO vertices from the selected old predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from wilor.models import load_wilor

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/native_hand_wrist_views_20261127'

def read(path):
    return json.loads(path.read_text())

def main():
    cfg = read(ROOT/'configs/pipeline.head01.developer.json')
    paths = cfg['paths']
    model, _ = load_wilor(paths['wilor_checkpoint'], paths['wilor_config'])
    mano = model.mano.cpu().eval()
    receipt = read(RUN.parent/'2026-11-27_wilor_native_hands/evidence/results.json')['cases']
    selected = read(RUN.parent/'2026-11-27_stereo_hand_selection/evidence/selection.json')
    guide = read(ROOT/'outputs/realvideo4_palm_20261126/guided.json')['selected']
    errors = {}
    for clip, data in selected.items():
        for side, hand in data['hands'].items():
            eye = {'cam3':'left','cam4':'right'}[hand['source']]
            letter = side[0].upper()
            tag = f'{clip}_{eye}_{letter}'
            path = ROOT/'outputs/wilor_stereo_fingers_20261126'/(tag+'.npz')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt[tag]['input_pack_sha256']
            with np.load(path) as p, torch.no_grad():
                params = {k:torch.from_numpy(p[k])[None] for k in ('global_orient','hand_pose','betas')}
                out = mano(**params,pose2rot=False)
                mirror = np.array([-1 if side=='left' else 1,1,1])
                j = out.joints[0,:21].numpy()*mirror
                v = out.vertices[0].numpy()*mirror
                native = np.asarray(guide[f'{clip}_{eye}'][letter]['local'])
                errors[tag] = float(np.linalg.norm(j-native,axis=1).max()*1000)
                assert errors[tag] < .5
                np.savez(OUT/(tag+'.npz'),j=j-j[0],v=v-j[0],faces=p['faces'])
    (RUN/'evidence/mano_replay_mm.json').write_text(json.dumps(errors,indent=2)+'\n')

if __name__ == '__main__':
    main()
