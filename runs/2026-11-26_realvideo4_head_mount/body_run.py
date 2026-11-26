"""Rerun the original 20k/v3/v4 pipeline with only the frozen Head target changed."""
import sys
import shutil
from pathlib import Path
import torch
import cv2
import numpy as np
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo4_head_mount_20261126'
OLD=ROOT/'outputs/realvideo4_dv3_20261125'
D3=RUN.parent/'2026-11-25_realvideo4_dv3'
D4=RUN.parent/'2026-11-25_realvideo4_dv4'
sys.path.insert(0,str(D3))
from solve import v3,cameras,read,write,sha
sys.path.insert(0,str(D4))
import refit

def prepare():
    assert sha(RUN/'evidence/fit003.json')==read(RUN/'evidence/check.json')['frozen_fit_sha256']
    conf=read(RUN/'evidence/fit003.json')
    folder=OUT/'fit/0';folder.mkdir(parents=True,exist_ok=True)
    (OUT/'evidence').mkdir(exist_ok=True)
    inputs=read(D3/'evidence/inputs.json')
    assert sha(OLD/'fit/0/prepared.json')==inputs['shards']['0']
    orig=read(OLD/'fit/0/prepared.json')
    for row in orig['items']:
        assert np.allclose(row['head_position'],conf['anchor_old_cam3_m'])
        row['head_position']=conf['anchor_candidate_cam3_m']
    write(folder/'prepared.json',orig)
    src=OLD/'fit/0/protocol.json'
    assert sha(src)==inputs['protocol']
    shutil.copyfile(src,folder/'protocol.json')
    write(OUT/'evidence/inputs.json',dict(protocol=inputs['protocol'],head_cam3_m=conf['anchor_candidate_cam3_m'],
             previous_head_cam3_m=inputs['head_cam3_m'],shards={'0':sha(folder/'prepared.json')}))
    sources=[(OUT/'images',OLD/'images')]+[(OUT/p.name,p) for p in cameras()]
    for path in (ROOT/'outputs/realvideo4_dv4_20261125/seg').glob('clip*.npy'):
        sources.append((OUT/'seg'/path.name,path))
    for dest,source in sources:
        if dest.is_symlink():assert dest.resolve()==source.resolve()
        else:dest.symlink_to(source.resolve())
    shutil.copyfile(D4/'evidence/segmentation.json',OUT/'evidence/segmentation.json')
    shutil.copyfile(D3/'evidence/calibration.json',OUT/'evidence/calibration.json')

def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    mode=sys.argv[1]
    v3.RUN=v3.OUT=OUT;v3.FIT=OUT/'fit';v3.ref.camera_files=cameras
    if mode=='prepare':prepare();return
    if mode=='v4':
        refit.OLD=refit.OUT=refit.PRIOR=refit.RUN=OUT
        refit.run(int(sys.argv[2]));return
    sys.addaudithook(v3.base.fit.forbid_gt)
    if mode=='fit':v3.fit(0);return
    assert mode=='candidates'
    v3.candidates(0)
    write(OUT/'evidence/results.json',dict(selected_hash=sha(OUT/'fit/0/selected.npz')))
if __name__=='__main__':main()
