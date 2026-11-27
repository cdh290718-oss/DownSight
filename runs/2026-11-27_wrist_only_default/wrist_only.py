"""Fit WiLoR finger candidates to calibrated stereo observations using wrist rotation only."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/wrist_only_default_20261127'
GALLERY = RUN.parent/'2026-11-27_perspective_wrist_gallery'
spec = importlib.util.spec_from_file_location("perspective_gallery", GALLERY/"experiment.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
from perspective_gallery import SIDES, KEYS, ROOTS, FINGER_JOINTS, paired, ref, old, read, write, sha

def seed(key):
    if key in KEYS[:2]:
        path = paired.palm.OUT/'poses'/(key+'.npz')
        expected = read(paired.PREV/'evidence'/(key+'.json'))['pose_hash']
    else:
        path = prior.ARM_OUT/'palm/poses'/(key+'.npz')
        expected = read(RUN.parent/'2026-11-26_purple_arm_hand_a/evidence'/(key+'_arm.json'))['seed_sha256']
    assert sha(path)==expected
    return dict(np.load(path)),expected

def run():
    # Enforce the standalone entry's independence from comparison poses and metrics.
    def observe(event,args):
        if event=='open' and isinstance(args[0],str):
            path = Path(args[0])
            if path.name.endswith(('_B.npz','_B.json')) or path==GALLERY/'evidence/results.json':
                raise RuntimeError('Unexpected comparison input: '+path.name)
    sys.addaudithook(observe)
    model,_ = prior.setup()
    layer = model.layer['neutral'].eval()
    means = layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
    rows = {r['pair']:r for r in read(ref.OLD/'fit/0/prepared.json')['items']}
    cams = list(map(read,ref.cameras()))
    det = ref.OLD/'predictions.json'
    assert sha(det)==read(ref.PRIOR/'evidence/detection.json')['predictions_sha256']
    predictions = read(det)
    order = np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
    rests,records = {},{}
    prior.stereo.OUT = OUT
    for key in KEYS:
        pack,digest = seed(key)
        start = pack['locked']
        shape = (pack['betas'].dtype.str,pack['betas'].tobytes(),start.dtype.str)
        if shape not in rests:
            neutral = np.zeros_like(start)
            neutral[23:] = -means
            _,rests[shape] = old.mesh(layer,neutral,dict(pack,global0=np.zeros(3)))
        world = [old.mat(start[21])@old.mat(pack['global0'])]
        for n in range(1,22):
            world.append(world[int(layer.parents[n])]@old.mat(start[n-1]))
        transforms = [np.asarray(rows[key]['inputs']['views'][eye]['camera']['left_to_view']) for eye in SIDES]
        def project(points):
            return np.stack([old.projection(points@t[:3,:3].T+t[:3,3],c) for t,c in zip(transforms,cams)])
        final = start.copy()
        hands = {}
        for i,side in enumerate(SIDES):
            obs = np.stack([np.array(predictions[key+'_'+eye+'.png']['xy'])[order+(21 if i==0 else 0)] for eye in SIDES])
            distance = np.linalg.norm(obs-obs[:,:1],axis=2)
            inside = ((obs>=0)&(obs<[1280,720])).all(-1)
            used = (inside&(distance<3.2*np.median(distance[:,ROOTS],axis=1)[:,None]))[:,FINGER_JOINTS]
            x,hands[side] = prior.choose_hand(key,i,layer,pack,means,rests[shape],world,project,obs,used)
            if x is not None:
                sl = slice(23+15*i,38+15*i)
                final[sl] = x[sl]
                final[19+i] = x[19+i]
        np.savez_compressed(OUT/(key+'.npz'),**dict(pack,locked=final))
        v,_ = old.mesh(layer,final,pack)
        _,obj = prior.stereo.export(key,'wrist',v,np.asarray(model.face,dtype=np.int32))
        records[key] = dict(seed_sha256=digest,obj_sha256=sha(obj),hands=hands)
    write(OUT/'fit.json',dict(frames=records,neutral_forwards=len(rests)))

def verify():
    fit = read(OUT/'fit.json')
    previous = read(GALLERY/'evidence/results.json')['metrics']
    for key,r in fit['frames'].items():
        with np.load(OUT/(key+'.npz')) as actual, np.load(prior.OUT/'poses'/(key+'_wrist.npz')) as expected:
            assert set(actual.files)==set(expected.files)
            assert all(np.array_equal(actual[k],expected[k]) for k in actual.files)
        refcase = previous[key]
        assert r['obj_sha256']==refcase['obj_sha256']['wrist']
        assert all(h['source']==refcase['hands'][s]['source'] for s,h in r['hands'].items())
        r['all_parameters_equal'] = True
        r['previous_obj_sha256'] = refcase['obj_sha256']['wrist']
    write(RUN/'evidence/results.json',dict(**fit,images={},exit_status=0))

if __name__=='__main__':
    if sys.argv[1:]==['fit']:
        run()
    elif sys.argv[1:]==['verify']:
        verify()
    else:
        raise SystemExit('choose fit or verify')
