"""Isolate replacing B's wrist solve while keeping the observed hand and its WiLoR rotations fixed."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.optimize import least_squares

RUN = Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'2026-11-27_stereo_hand_selection'))
from choose import old, CASES, SIDES, DIGITS, TIPS, JOINT_NAMES, sha, read, write
from compare_fit import ARM_OUT, ROOTS

def main():
    torch.set_num_threads(1)
    model,_ = old.ref.prev.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    cams = list(map(read,old.ref.cameras()))
    selected = RUN.parent/'2026-11-27_stereo_hand_selection/evidence/metrics.json'
    fixed = read(selected)
    rows = {r['pair']:r for r in read(old.ref.OLD/'fit/0/prepared.json')['items']}
    names = ['wrist']+[d+j for d in DIGITS for j in ('1','2','3','')]
    record,pool = {},{k:[] for k in ('B','pre_B','four_MCP')}
    for key in CASES:
        src = ARM_OUT/'palm/poses'/(key+'.npz')
        assert sha(src)==read(old.BASE/'evidence'/(key+'.json'))['seed_sha256']
        before = dict(np.load(src))
        refpath = old.ROOT/'outputs/stereo_hand_selection_20261127/poses'/(key+'.npz')
        assert sha(refpath)==fixed[key]['pose_sha256']
        refpack = dict(np.load(refpath))
        seed = before['locked'].copy()
        reference = refpack['locked']
        assert np.array_equal(seed[:19],reference[:19]) and np.array_equal(seed[21:23],reference[21:23])
        assert np.array_equal(before['betas'],refpack['betas']) and np.array_equal(before['global0'],refpack['global0'])
        seed[23:] = reference[23:]  # prior choice held constant across all three conditions
        _,jp = old.mesh(layer,seed,before)
        world = [old.mat(seed[21])@old.mat(before['global0'].reshape(3))]
        for n in range(1,22):
            world.append(world[int(layer.parents[n])]@old.mat(seed[n-1]))
        transforms = [np.array(rows[key]['inputs']['views'][eye]['camera']['left_to_view']) for eye in SIDES]
        def project(points):
            return np.stack([old.projection(points@t[:3,:3].T+t[:3,3],cam) for t,cam in zip(transforms,cams)])
        trial=seed.copy()
        record[key]={}
        for i,side in enumerate(SIDES):
            ids=[JOINT_NAMES.index(side+'_'+n) for n in names]
            h=fixed[key]['hands'][side]
            obs=np.asarray(h['observation'])
            mask=np.asarray(h['inside'])[:,TIPS]
            # Use exactly the original B eligibility mask for these knuckles.
            meta=read(old.BASE/'evidence'/(key+'.json'))['results']['B']['hands'][side]
            valid=np.asarray(meta['used'])[:,[meta['point_indices'].index(j) for j in ROOTS]]
            assert valid.any(1).all()
            w=20+i
            pivot=jp[w]
            local=(jp[np.asarray(ids)[ROOTS]]-pivot)@world[w]
            parent=world[18+i]
            initial=old.mat(seed[w-1])
            def loss(r):
                orient=parent@old.mat(r)
                uv=project(local@orient.T+pivot)
                return np.r_[((uv-obs[:,ROOTS])/6)[valid].ravel(),old.rv(initial.T@old.mat(r))/.65]
            t=time.perf_counter()
            fit=least_squares(loss,seed[w-1],loss='soft_l1',max_nfev=180)
            assert fit.success
            trial[w-1]=old.rv(old.mat(fit.x))
            info=dict(source=h['source'],mcp_observations=int(valid.sum()),optimizer_calls=fit.nfev,solve_elapsed_s=time.perf_counter()-t,variants={})
            for tag,pose in [('B',reference),('pre_B',seed),('four_MCP',trial)]:
                _,j=old.mesh(layer,pose,before)
                uv=project(j[ids])
                err=np.linalg.norm(uv-obs,axis=-1)[:,TIPS][mask]
                pool[tag].extend(err.tolist())
                deg=np.rad2deg(np.linalg.norm(old.rv(old.mat(reference[w-1]).T@old.mat(pose[w-1]))))
                info['variants'][tag]=dict(wrist_diff_to_B_deg=float(deg),tip_rmse_px=old.rms(err),tip_errors_px=err.tolist(),
                                           wrist_drift_mm=float(1000*np.linalg.norm(j[w]-pivot)))
                assert info['variants'][tag]['wrist_drift_mm']<.001
                if tag=='B': assert np.isclose(old.rms(err),h['tips_rmse_px']['selected'])
                if tag=='four_MCP': assert np.max(abs(uv[:,ROOTS]-project(local@(parent@old.mat(fit.x)).T+pivot)))<.01
            record[key][side]=info
    summary={k:dict(count=len(v),tip_rmse_px=old.rms(v)) for k,v in pool.items()}
    write(RUN/'evidence/wrist.json',dict(cases=record,pooled=summary,exit_status=0))
    print(record,summary)
if __name__=='__main__':
    main()
