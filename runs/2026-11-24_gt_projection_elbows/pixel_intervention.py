"""Paired pixel interventions on two prespecified frames, with identical body seeds."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/gt_projection_elbows_20261124'
FIT = OUT / 'fit'
sys.path.insert(0, str(ROOT/'runs/2026-11-24_elbow_limits'))
import select_pose as elbow
from select_pose import prev, search
import candidates as wrist

CASES = ['val031','val055']
MODES = ['detector','projected_wrists','projected_all','annotated_all']
def key(frame, mode):
    return frame+'_'+mode

def prepare():
    items = {v['pair']:v for v in prev.read(prev.VAL/'prepared.json')['items']}
    truth = prev.read(prev.OUT/'validation_truth.json')
    images = prev.pair_images(prev.VM)
    model,camera = prev.ref.common.setup_model()
    cams = [camera(p) for p in prev.ref.camera_files()]
    ad = [prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    rows,errors,image_inputs = [],{},{}
    for frame in CASES:
        item = items[frame]
        pair = images[int(frame[-3:])-1]
        image_inputs[frame]=[dict(image=v['image'],sha256=v['image_sha256']) for v in pair]
        target = np.array(truth[frame]['gt'])[prev.ref.common.OBS]
        assert pair[0]['subject']==truth[frame]['subject']
        assert [p['name'] for p in item['points']]==[prev.ref.common.NAMES[i] for i in prev.ref.common.OBS]
        errors[frame]={}
        ks = prev.mapping()
        per_eye = []
        for eye,(side,im) in enumerate(zip(['left','right'],pair)):
            T = np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
            pixel = cams[eye].world2camera(target@T[:3,:3].T+T[:3,3])
            scale,offset=ad[eye]['scale'],ad[eye]['offset']
            annotated = np.array(im['keypoints308_xy'])[ks]*scale+offset
            mask = np.asarray(im['keypoints308_supervision_mask'])[ks].astype(bool)
            assert np.array_equal(mask,[p['valid'][eye] for p in item['points']])
            detected = np.array([p['xy'][eye] for p in item['points']])
            def distances(a,b):
                return {p['name']:float(np.linalg.norm(x-y)/scale) if ok else None
                        for p,x,y,ok in zip(item['points'],a,b,mask)}
            errors[frame][side]=dict(det_vs_official2d=distances(detected,annotated),
                det_vs_projection=distances(detected,pixel),official2d_vs_projection=distances(annotated,pixel))
            per_eye.append((pixel,annotated))
        for mode in MODES:
            row=deepcopy(item)
            row['pair']=key(frame,mode)
            for j,point in enumerate(row['points']):
                if mode=='detector' or (mode=='projected_wrists' and 'Wrist' not in point['name']):
                    continue
                for eye,values in enumerate(per_eye):
                    if point['valid'][eye]:
                        point['xy'][eye]=values[mode=='annotated_all'][j].tolist()
            rows.append(row)
    prev.write(OUT/'image_inputs.json',image_inputs)
    prev.write(FIT/'prepared.json',dict(items=rows))
    (FIT/'protocol.json').write_bytes((prev.VAL/'protocol.json').read_bytes())
    prev.write(RUN/'evidence/input_errors.json',dict(frames=errors,unit='original image px',
        prepared_sha256=prev.sha(FIT/'prepared.json'),protocol_sha256=prev.sha(FIT/'protocol.json'),
        archived_input_sha256=prev.sha(prev.VAL/'prepared.json'),targets_sha256=prev.sha(prev.OUT/'validation_truth.json'),
        manifest_sha256=prev.sha(prev.VM),camera_sha256=[prev.sha(p) for p in prev.ref.camera_files()]))

def fit():
    impl=prev.base.fit
    impl.RUN=impl.OUT=FIT
    impl.camera_files=prev.ref.camera_files
    impl.Problem=prev.base.trial.TorsoOnly
    sys.argv=[__file__,'--threads','1']
    impl.main()

if __name__=='__main__':
    {'prepare':prepare,'fit':fit}[sys.argv[1]]()
