"""Search complete SMPL-X poses using only the archived Golden D v2 objective."""
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-23_stereo_wrist_candidates'))
import shared as prev

OUT = ROOT / 'outputs/fullbody_joint_candidates_20261123'
CHAINS = [(16,18,20), (17,19,21), (1,4,7), (2,5,8)]
FIELDS = ['body_pose', 'delta', 'translation', 'global0', 'betas']
DEPTH_MM = [-240,-120,-60,-20,0,20,60,120,240]
SWIVEL_DEG = [-90,-30,0,30,90,180]
rot = lambda v: Rotation.from_rotvec(v).as_matrix()
rv = lambda m: Rotation.from_matrix(m).as_rotvec()


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def turns(a, b):
    a,b = np.broadcast_arrays(unit(a), unit(b))
    axis = np.cross(a,b)
    sine = np.linalg.norm(axis,axis=-1)
    dot = (a*b).sum(-1)
    opposite = (sine < 1e-10) & (dot < 0)
    axis[opposite] = np.cross(a[opposite],np.eye(3)[np.argmin(abs(a[opposite]),axis=1)])
    return rot(unit(axis)*np.arctan2(sine,dot)[:,None])


def chain_options(xyz, world, parents, chain, rays, radius, caps):
    """Reach both end-point rays, then rotate the bend plane about the chain axis."""
    s,e,w = chain
    S,E,W = xyz[[s,e,w]]
    a,b = np.linalg.norm(E-S),np.linalg.norm(W-E)
    targets = (np.asarray(rays)[:,None,:] *
               (radius + np.asarray(DEPTH_MM)/1000)[None,:,None]).reshape(-1,3)
    targets = np.unique(np.vstack([W,targets]),axis=0)
    dist = np.linalg.norm(targets-S,axis=1)
    good = (dist > abs(a-b)+1e-6) & (dist < a+b-1e-6) & (targets[:,2]>0)
    end = np.repeat(targets[good],len(SWIVEL_DEG),axis=0)
    if not len(end):
        return [],[],[]
    dist = np.linalg.norm(end-S,axis=1,keepdims=True)
    axis = (end-S)/dist
    along = (a*a-b*b+dist**2)/(2*dist)
    plane = (E-S) - axis * ((E-S)*axis).sum(1,keepdims=True)
    good_plane = np.linalg.norm(plane,axis=1)>1e-7
    angles = np.tile(np.deg2rad(SWIVEL_DEG),sum(good))
    bend = Rotation.from_rotvec(axis*angles[:,None]).apply(unit(plane))
    mid = S+axis*along+bend*np.sqrt(np.maximum(0,a*a-along**2))
    r1 = turns(E-S,mid-S)
    new_s = r1@world[s]
    new_e = turns((r1@(W-E)),end-mid)@r1@world[e]
    local = np.stack([rv(world[parents[s]].T@new_s),rv(new_s.transpose(0,2,1)@new_e)],axis=1)
    keep = good_plane & (np.linalg.norm(local,axis=2)<=caps).all(1)
    return local[keep],mid[keep],end[keep]


def frame(item, wrist, baseline, model, camera, cfg):
    p = prev.base.trial.TorsoOnly([dict(item,condition='baseline')],cfg,model,camera)
    seed = torch.from_numpy(wrist['global0']).float().reshape(1,3)
    assert not wrist['betas'].any() and not any(pt['accepted'] for pt in item['points'])
    caps = dict(zip(p.body_ids.tolist(),p.caps.numpy()))
    def evaluate(body, delta, tr):
        b = torch.as_tensor(np.asarray(body),dtype=torch.float32)
        n = len(b)
        d,t = [torch.as_tensor(np.asarray(v),dtype=torch.float32).expand(n,3) for v in [delta,tr]]
        beta = torch.zeros(n,10)
        xyz,rh = p.fk(b,beta,seed.expand(n,3),d,t)
        p.bodyvar = b[:,p.body_ids]
        loss,ok,_ = p.objective((xyz,rh,beta))
        assert torch.isfinite(loss).all()
        return xyz.numpy(),loss.numpy(),ok.numpy()

    body,delta,tr = [wrist[k] for k in FIELDS[:3]]
    xyz,old_loss,ok = evaluate([body],delta,tr)
    assert ok[0]
    start = xyz[0].astype(float)
    pts = {pt['native_index']:pt for pt in item['points']}
    rays = {}
    for _,e,w in CHAINS:
        if all(pts[e]['valid']) and all(pts[w]['valid']):
            pix = np.asarray(pts[w]['xy'][:1])
            rays[w] = [unit(start[w]),unit(p.cameras[0].camera2world_ray(pix)[0])]
    seeds = [('fixed',body,delta)]
    for axis in np.eye(3):
        for angle in [-4,4]:
            step = rot(axis*np.deg2rad(angle))
            tag = str(int(axis.argmax()))+'_'+str(angle)
            seeds.append(('root_'+tag,body,rv(step@rot(delta))))
            spine = body.copy()
            spine[8] = rv(rot(body[8])@step)
            if np.linalg.norm(spine[8]) <= caps[8]:
                seeds.append(('spine3_'+tag,spine,delta))

    output = []
    trials = 0
    base_loss = float(evaluate([baseline['body_pose']],baseline['delta'],baseline['translation'])[1][0])
    def keep(name,pose,d,t,loss):
        return dict(name=name,body_pose=pose,delta=d,translation=t,loss=float(loss))
    output.extend([keep('baseline',baseline['body_pose'],baseline['delta'],baseline['translation'],base_loss),
                   keep('wrist',body,delta,tr,old_loss[0])])
    for tag,pose,d in seeds:
        xyz = evaluate([pose],d,tr)[0][0].astype(float)
        t = tr+(start[15]-xyz[15])
        xyz,loss,ok = evaluate([pose],d,t)
        if not ok[0]:
            continue
        joint = xyz[0].astype(float)
        assert np.linalg.norm(joint[15]-start[15])<2e-6
        parents = p.fk.parents
        world = [rot(d)@rot(wrist['global0'])]
        for k in range(1,22):
            world.append(world[parents[k]]@rot(pose[k-1]))
        banks, mids, ends = [],[],[]
        for s,e,w in CHAINS:
            local,mid,end = [],[],[]
            if w in rays:
                local,mid,end = chain_options(joint,world,parents,(s,e,w),rays[w],np.linalg.norm(start[w]),
                                              [caps[s-1],caps[e-1]])
            bank = np.repeat(pose[None],1+len(local),axis=0)
            if len(local):
                bank[1:,[s-1,e-1]] = local
            banks.append(bank)
            mids.append(np.vstack([joint[e],np.asarray(mid).reshape(-1,3)]))
            ends.append(np.vstack([joint[w],np.asarray(end).reshape(-1,3)]))
        xyz,loss,valid = evaluate(np.concatenate(banks),d,t)
        trials += len(loss)
        merged = pose.copy()
        offset = 0
        for (s,e,w),bank,mid,end in zip(CHAINS,banks,mids,ends):
            sl = slice(offset,offset+len(bank))
            err = max(np.linalg.norm(xyz[sl,e]-mid,axis=1).max(),np.linalg.norm(xyz[sl,w]-end,axis=1).max())
            assert err < 5e-5, (item['pair'],tag,w,err)
            assert valid[offset]
            j = np.where(valid[sl],loss[sl],np.inf).argmin()
            merged[[s-1,e-1]] = bank[j,[s-1,e-1]]
            offset += len(bank)
        _,combined,good = evaluate([merged],d,t)
        assert good[0]
        output.append(keep(tag,merged,d,t,combined[0]))
    assert output[2]['name']=='fixed'
    return output,trials


def search(split):
    torch.set_num_threads(1)
    prev.base.fit.camera_files = prev.ref.camera_files
    folder = prev.VAL if split=='validation' else prev.base.OUT
    wristfile = prev.OUT/(split+'_objective.npz')
    baselinefile = folder/'budget_20000.npz'
    assert prev.sha(wristfile)==prev.read(prev.RUN/'evidence/verification.json')['diagnostics'][split]['saved_parameter_hashes']['objective']
    assert prev.sha(baselinefile)==prev.read(prev.RUN/('evidence/'+split+'_coverage.json'))['baseline_sha256']
    wrist,base = [dict(np.load(path)) for path in (wristfile,baselinefile)]
    cfg = prev.read(folder/'protocol.json')
    items = prev.read(folder/'prepared.json')['items']
    target = 'baseline' if split=='validation' else 'rw20000'
    ids = {r['pair']:i for i,r in enumerate(items) if r['condition']==target}
    assert len(ids)==len(wrist['pair'])==len(set(wrist['pair']))
    model,camera = prev.ref.common.setup_model()
    poses = []
    records = {}
    started = time.monotonic()
    for j,key in enumerate(wrist['pair']):
        i = ids[key]
        w,b = [{f:pack[f][index] for f in FIELDS} for pack,index in [(wrist,j),(base,i)]]
        with torch.no_grad():
            options,trials = frame(items[i],w,b,model,camera,cfg)
        assert abs(options[0]['loss']-base['loss'][i]) < 2e-6
        losses = np.array([c['loss'] for c in options])
        assert losses[1] <= losses[0]+1e-6
        fixed = int(losses[:3].argmin())
        best = int(losses.argmin())
        records[key] = dict(index=i,offset=len(poses),names=[c['name'] for c in options],loss=losses.tolist(),
                            fixed=fixed,best=best,chain_probes=trials)
        for c in options:
            poses.append(dict(w,**{f:c[f] for f in FIELDS[:3]}))
        if (j+1)%25==0:
            print(split,j+1,'pairs',round(time.monotonic()-started,1),'s',flush=True)
    path = OUT/(split+'_bank.npz')
    np.savez_compressed(path,**{f:np.stack([c[f] for c in poses]) for f in FIELDS})
    prev.write(OUT/(split+'_choices.json'),records)
    prev.write(RUN/('evidence/'+split+'_search.json'),dict(pairs=len(records),seconds=time.monotonic()-started,
               source_sha256=prev.sha(Path(__file__)),parameter_sha256=prev.sha(path),
               choices_sha256=prev.sha(OUT/(split+'_choices.json')),wrist_sha256=prev.sha(wristfile),
               baseline_sha256=prev.sha(baselinefile),prepared_sha256=prev.sha(folder/'prepared.json'),
               protocol_sha256=prev.sha(folder/'protocol.json'),chain_probes=sum(r['chain_probes'] for r in records.values()),
               composed_poses=sum(len(r['names'])-2 for r in records.values()),exit_status=0))


if __name__=='__main__':
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],str):
            name = Path(args[0]).name
            if name in {'reference.json','validation_truth.json','samples.npz','validation.json','test_manifest.json'} or name.endswith('_results.json'):
                raise PermissionError('Scoring inputs excluded from search')
    sys.addaudithook(guard)
    search(sys.argv[1])
