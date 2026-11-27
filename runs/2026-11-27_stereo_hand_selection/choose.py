"""Choose a native finger pose per hand, then evaluate the shared SMPL-X result."""
from pathlib import Path
import sys
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
PRIOR = RUN.parent/'2026-11-27_hand_pose_transfer'
OUT = RUN.parents[1]/'outputs/stereo_hand_selection_20261127'
sys.path.insert(0, str(PRIOR))
import transfer as old
from compare_fit import FINGER_JOINTS
from transfer import SIDES, TIPS, DIGITS, JOINT_NAMES, sha, read, write
SOURCES = ('cam3','cam4')
CASES = ('clip003','clip004')

def choose(key):
    source_file = PRIOR/'evidence'/(key+'.json')
    data = read(source_file)['results']
    hands = {}
    for side in SIDES:
        baseline = data['B']['hands'][side]
        ids = baseline['point_indices']
        assert ids == FINGER_JOINTS and not set(ids).intersection(TIPS)
        valid = np.asarray(baseline['used'], dtype=bool)
        target = np.asarray(baseline['observation'])[:, ids]
        assert valid.shape == (2,15) and valid.any(1).all()
        scores = {}
        for source in SOURCES:
            hand = data[source+'_B_wrist']['hands'][side]
            assert np.array_equal(target, np.asarray(hand['observation'])[:,ids])
            uv = np.asarray(hand['uv'])[:,ids]
            distance = np.linalg.norm(uv-target, axis=2)
            losses = 2*(np.sqrt(1+(distance/6)**2)-1)
            per_eye = [float(loss[mask].mean()) for loss,mask in zip(losses,valid)]
            scores[source] = dict(score=float(np.mean(per_eye)), per_eye=per_eye,
                                  residual_px=[d[m].tolist() for d,m in zip(distance,valid)])
        winner = min(SOURCES, key=lambda s: scores[s]['score'])
        hands[side] = dict(source=winner, scores=scores, point_ids=ids, used=valid.tolist())
    return dict(input_sha256=sha(source_file), hands=hands)

def main():
    choices = {key:choose(key) for key in CASES}
    write(RUN/'evidence/selection.json',choices)
    torch.set_num_threads(1)
    model,_ = old.ref.prev.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    views = list(map(read,old.ref.cameras()))
    assert [sha(p) for p in old.ref.cameras()] == read(old.ref.PRIOR/'evidence/calibration.json')['converted']
    prepared = old.ref.OLD/'fit/0/prepared.json'
    assert sha(prepared) == read(old.ref.PRIOR/'evidence/inputs.json')['shards']['0']
    rows = {row['pair']:row for row in read(prepared)['items']}
    metrics = {}
    pools = {mode:[] for mode in ('selected','B',*SOURCES)}
    names = ['wrist'] + [d+s for d in DIGITS for s in ('1','2','3','')]
    for key in CASES:
        data = read(PRIOR/'evidence'/(key+'.json'))['results']
        seed_file = old.BASE_OUT/'poses'/(key+'_B.npz')
        assert sha(seed_file) == data['B']['pose_sha256']
        pack = dict(np.load(seed_file))
        x = pack['locked'].copy()
        _, seed_joints = old.mesh(layer,x,pack)
        for i,side in enumerate(SIDES):
            mode = choices[key]['hands'][side]['source']+'_B_wrist'
            file = old.OUT/'poses'/(key+'_'+mode+'.npz')
            assert sha(file) == data[mode]['pose_sha256']
            mapped = dict(np.load(file))
            assert np.array_equal(x[:23],mapped['locked'][:23])
            assert all(np.array_equal(pack[k],mapped[k]) for k in pack if k!='locked')
            finger_slice = slice(23+i*15,38+i*15)
            x[finger_slice] = mapped['locked'][finger_slice]
        pose_file = OUT/'poses'/(key+'.npz')
        np.savez_compressed(pose_file, **dict(pack,locked=x))
        _,joints = old.mesh(layer,x,pack)
        drift = float(1000*np.linalg.norm(seed_joints[:22]-joints[:22],axis=-1).max())
        assert drift < .001
        hands = {}
        for side in SIDES:
            winner = choices[key]['hands'][side]['source']
            j = joints[[JOINT_NAMES.index(side+'_'+n) for n in names]]
            uv = []
            for view,cam in zip(SIDES,views):
                t = np.asarray(rows[key]['inputs']['views'][view]['camera']['left_to_view'])
                uv.append(old.projection(j@t[:3,:3].T+t[:3,3],cam))
            h = data['B']['hands'][side]
            obs = np.asarray(h['observation'])
            mask = np.asarray(h['inside'])[:,TIPS]
            replay_max = float(np.linalg.norm(np.asarray(uv)-data[winner+'_B_wrist']['hands'][side]['uv'],axis=-1).max())
            assert replay_max < .05
            stats = {}
            for mode in pools:
                if mode == 'selected':
                    pred = uv
                else:
                    prior_mode = 'B' if mode=='B' else mode+'_B_wrist'
                    pred = data[prior_mode]['hands'][side]['uv']
                errors = np.linalg.norm(np.asarray(pred)-obs,axis=-1)[:,TIPS][mask]
                stats[mode] = old.rms(errors)
                pools[mode].extend(errors.tolist())
            hands[side] = dict(uv=np.asarray(uv).tolist(), observation=obs.tolist(), inside=h['inside'],
                               source=winner, tips_rmse_px=stats, count=int(mask.sum()), replay_max_px=replay_max)
        metrics[key] = dict(body_drift_mm=drift, wrist_rotations_exact=bool(np.array_equal(x[19:21],pack['locked'][19:21])),
                            seed_sha256=sha(seed_file), pose_sha256=sha(pose_file), hands=hands)
    metrics['pooled'] = {m:dict(count=len(v),rmse_px=old.rms(v)) for m,v in pools.items()}
    assert all(len(v)==33 for v in pools.values())
    write(RUN/'evidence/metrics.json',metrics)
    print({key:[v['source'] for v in c['hands'].values()] for key,c in choices.items()},metrics['pooled'])

if __name__ == '__main__':
    main()
