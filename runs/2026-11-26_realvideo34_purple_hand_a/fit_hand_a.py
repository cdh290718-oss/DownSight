"""Run the frozen A optimizer on each purple body's translated palm initialization."""
from pathlib import Path
import sys
import numpy as np

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo34_purple_hand_a_20261126'
PURPLE=RUN.parent/'2026-11-26_realvideo4_pitch_obj'
METHOD=RUN.parent/'2026-11-26_realvideo4_wilor_outline'
sys.path.insert(0,str(METHOD))
import paired
from paired import read, write, sha

def main(index):
    key=f'clip{index+1:03}'
    assert key in ('clip003','clip004')
    purple=read(PURPLE/'evidence/render.json')['rows'][key]['Purple']
    assert purple['degree']==0
    parent=paired.PREV
    start=paired.palm.OUT/'poses'/(key+'.npz')
    metadata=read(parent/'evidence'/(key+'.json'))
    assert sha(start)==metadata['pose_hash']
    # The published A runner looks up the palm seed and its receipt in a shared
    # folder. Build that input locally, retaining the original quality values.
    seed_dir=OUT/'palm'
    for folder in ('poses','evidence'):
        (seed_dir/folder).mkdir(parents=True,exist_ok=True)
    data=dict(np.load(start))
    v4=paired.ref.OUT/'poses'/(key+'.npz')
    assert sha(v4)==purple['pose_sha256']
    with np.load(v4) as p:
        diff=np.where(np.any(p['locked']!=data['locked'],axis=1))[0]
        assert set(diff)<=set([17,18,19,20]) and np.array_equal(p['global0'],data['global0']) and np.array_equal(p['betas'],data['betas'])
    data['locked'][22]+=np.array(purple['translation_m'])
    dest=seed_dir/'poses'/(key+'.npz')
    np.savez_compressed(dest,**data)
    write(seed_dir/'evidence'/(key+'.json'),dict(metadata,pose_hash=sha(dest)))
    write(RUN/'evidence'/(key+'_inputs.json'),dict(source_palm_sha256=sha(start),translated_palm_sha256=sha(dest),
           purple=purple,optimizer_sha256=sha(METHOD/'paired.py'),steps=paired.STEPS))
    paired.palm.OUT=paired.PREV=seed_dir
    paired.MODES=['control']
    paired.RUN=RUN
    paired.OUT=OUT
    # A's objective uses the original semantic masks via frame.observed. The
    # independent WiLoR-only diagnostic is unnecessary in this A-only run.
    paired.visible_targets=lambda frame,key:([],{})
    for side in paired.SIDES:
        k=key+'_'+side
        assert sha(paired.ref.OUT/'seg'/(k+'.npy'))==read(paired.palm.V4/'evidence/segmentation.json')['eyes'][k]['sampled_sha256']
    paired.run(index)
if __name__=='__main__':main(int(sys.argv[1]))
