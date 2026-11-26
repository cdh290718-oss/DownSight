"""Adjust shoulder and elbow rotations to the stereo elbow and hand wrist targets."""
import sys
from pathlib import Path
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/purple_arm_hand_a_20261126'
PREVIOUS = RUN.parent / '2026-11-26_realvideo34_purple_hand_a'
sys.path.insert(0, str(PREVIOUS))
import fit_hand_a as baseline
from fit_hand_a import paired, read, write, sha

def fit(index):
    torch.set_num_threads(1)
    key = f'clip{index+1:03}'
    ref = paired.ref
    ref.prev.ref.camera_files = ref.prev.base.fit.camera_files = ref.cameras
    ref.compare.OUT = ref.OUT
    receipt = read(PREVIOUS/'evidence'/(key+'_inputs.json'))
    source = baseline.OUT/'palm/poses'/(key+'.npz')
    assert sha(source) == receipt['translated_palm_sha256']
    pack = dict(np.load(source))
    seed = torch.from_numpy(pack['locked'])
    start = dict(body_pose=pack['locked'][:21], delta=pack['locked'][21],
                 translation=pack['locked'][22], global0=pack['global0'], betas=pack['betas'])
    row = read(ref.OLD/'fit/0/prepared.json')['items'][index]
    assert row['pair'] == key
    model, cameras = ref.prev.ref.common.setup_model()
    frame = ref.FixedFeet(key,row,start,model,cameras,read(ref.OLD/'fit/0/protocol.json'))
    before = read(PREVIOUS/'evidence'/(key+'.json'))['results']['control']['hands']
    targets = []
    for n in range(2):
        elbow = [next(p['xy'][n] for p in row['points'] if p['name']==side+'_Elbow' and p['valid'][n]) for side in ['L','R']]
        wrist = [before[side]['target'][n][0] for side in paired.SIDES]
        targets.append(elbow+wrist)
    target = seed.new_tensor(targets)
    movable = torch.tensor([15,16,17,18])
    def projections(pose):
        joints = paired.body(frame,pose)[1]
        limb = joints[[18,19,20,21]]
        uv = torch.stack([paired.project(limb@eye.transform_t[:3,:3].T+eye.transform_t[:3,3],eye.cam) for eye in frame.eyes])
        return joints, uv

    delta = torch.zeros_like(seed[movable],requires_grad=True)
    solver = torch.optim.LBFGS([delta],max_iter=100,line_search_fn='strong_wolfe')
    def pose():
        return seed.index_copy(0,movable,seed[movable]+delta)
    def closure():
        solver.zero_grad()
        _,uv = projections(pose())
        residual = (uv-target).square().sum(-1)
        loss = residual[:,:2].mean() + 4*residual[:,2:].mean() + 25*delta.square().sum(1).mean()
        loss.backward()
        return loss
    solver.step(closure)
    with torch.no_grad():
        arm = pose()
        old, initial = projections(seed)
        new, fitted = projections(arm)
        assert torch.isfinite(fitted).all()
        lengths = lambda j: (j[[18,19,20,21]]-j[[16,17,18,19]]).norm(dim=1)
        drift = (lengths(old)-lengths(new)).abs().max().item()*1000
        head = (old[15]-new[15]).norm().item()*1000
        assert drift < .01 and head < .001
    dest = OUT/'palm/poses'/(key+'.npz')
    np.savez_compressed(dest,**dict(pack,locked=arm.detach().numpy()))
    meta = read(baseline.OUT/'palm/evidence'/(key+'.json'))
    write(OUT/'palm/evidence'/(key+'.json'),dict(meta,pose_hash=sha(dest)))
    write(RUN/'evidence'/(key+'_arm.json'),dict(source_sha256=sha(source),seed_sha256=sha(dest),target=targets,
          initial=initial.tolist(),fitted=fitted.tolist(),head_drift_mm=head,max_bone_drift_mm=drift,
          rotation_delta=delta.detach().tolist(), ids=['L_elbow','R_elbow','L_wrist','R_wrist']))
    paired.palm.OUT = paired.PREV = OUT/'palm'
    paired.OUT, paired.RUN = OUT,RUN
    paired.MODES = ['control']
    paired.visible_targets = lambda frame,key:([],{})
    for side in paired.SIDES:
        k=key+'_'+side
        assert sha(ref.OUT/'seg'/(k+'.npy')) == read(paired.palm.V4/'evidence/segmentation.json')['eyes'][k]['sampled_sha256']
    paired.run(index)

if __name__ == '__main__':
    fit(int(sys.argv[1]))
