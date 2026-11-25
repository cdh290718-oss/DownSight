"""Compare official 2D fitting with free and hard-constrained stereo forearm directions."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/official_forearm_direction_20261125'
SOURCE=ROOT/'outputs/gt_projection_elbows_20261124'
PREV=ROOT/'runs/2026-11-24_gt_projection_elbows'
sys.path.insert(0,str(ROOT/'runs/2026-11-24_stereo_forearm_seed'))
import forearm_seed as seed
j=seed.joint
p=seed.prev
FRAMES=['val055','val031']
METHODS=['official2d','direction_init','direction_fixed']
STEPS=1000
ELBOWS=[18,19]
WRISTS=[20,21]


def direction_error(a,b):
    u,v=j.unit(np.asarray(a,dtype=float)),j.unit(np.asarray(b,dtype=float))
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),(u*v).sum(-1)))


def lock(problem,indices,targets):
    """Postmultiply local elbow rotations so each distal bone has the requested camera direction."""
    body=problem.parameters()[0].numpy()[indices]
    xyz=problem.forward()[0].numpy()[indices]
    matrices=Rotation.from_rotvec(body.reshape(-1,3)).as_matrix().reshape(len(indices),21,3,3)
    globals_at=[j.rot(problem.delta.numpy()[indices])@j.rot(problem.global0.numpy()[indices])]
    for k in range(1,20):
        globals_at.append(globals_at[problem.fk.parents[k]]@matrices[:,k-1])
    for side,(e,w) in enumerate(zip(ELBOWS,WRISTS)):
        swing=j.turns(xyz[:,w]-xyz[:,e],targets[:,side])
        parent=globals_at[problem.fk.parents[e]]
        correction=j.rv(parent.transpose(0,2,1)@swing@globals_at[e])
        col=problem.body_ids.tolist().index(e-1)
        problem.bodyvar[indices,col]=torch.tensor(correction,dtype=torch.float32)
        assert (np.linalg.norm(correction,axis=1)<=float(problem.caps[col])+1e-6).all()


def fit():
    def protect(event,args):
        if event=='open' and args and isinstance(args[0],str):
            name=Path(args[0]).name
            if name in ['reference.json','validation_truth.json','samples.npz','scores.json','validation.json']:
                raise PermissionError(name)
    sys.addaudithook(protect)
    verified=p.read(PREV/'evidence/validation_search.json')
    files={'prepared_sha256':SOURCE/'fit/prepared.json','protocol_sha256':SOURCE/'fit/protocol.json',
           'choices_sha256':SOURCE/'validation_choices.json','parameter_sha256':SOURCE/'validation_bank.npz'}
    assert all(p.sha(path)==verified[k] for k,path in files.items())
    items={r['pair']:r for r in p.read(files['prepared_sha256'])['items']}
    choices=p.read(files['choices_sha256'])
    bank=dict(np.load(files['parameter_sha256']))
    rows=[]; ids=[]
    for frame in FRAMES:
        k=frame+'_annotated_all'
        rows.extend([items[k]]*len(METHODS))
        ids.extend([choices[k]['offset']+choices[k]['best']]*len(METHODS))
    cfg=p.read(files['protocol_sha256'])
    model,cam=p.ref.common.setup_model()
    p.base.fit.camera_files=p.ref.camera_files
    prob=p.base.trial.TorsoOnly(rows,cfg,model,cam)
    adapter=[p.read(path)['input_adapter'] for path in p.ref.camera_files()]
    caps=dict(zip(prob.body_ids.tolist(),prob.caps.tolist()))
    original={k:bank[k][ids] for k in j.FIELDS}
    idx=[i*len(METHODS)+2 for i in range(len(FRAMES))]
    with torch.no_grad():
        prob.delta.copy_(torch.tensor(original['delta']))
        prob.tr.copy_(torch.tensor(original['translation']))
        prob.bodyvar.copy_(torch.tensor(original['body_pose'][:,prob.body_ids]))
        assert np.allclose(prob.global0.numpy(),original['global0']) and not original['betas'].any()
        xyz=prob.forward()[0].numpy().copy()
        targets=[]
        for n,frame in enumerate(FRAMES):
            i=n*len(METHODS)
            opts,diag=seed.make_seeds(rows[i],{k:v[i] for k,v in original.items()},xyz[i],prob.fk.parents,caps,prob.cameras,adapter)
            targets.append([diag[s]['direction'] for s in ['left','right']])
            for h in [1,2]:
                prob.bodyvar[i+h].copy_(torch.tensor(opts['both'][prob.body_ids]))
        targets=np.array(targets)
        lock(prob,idx,targets)
    variables=[prob.bodyvar,prob.delta,prob.tr]
    adam=torch.optim.Adam([dict(params=[v],lr=lr) for v,lr in zip(variables,prob.lr)])
    losses=torch.full((len(rows),),float('inf'))
    saved=[v.detach().clone() for v in variables]
    steps=np.zeros(len(rows),int)
    maximum=0.
    for step in range(STEPS+1):
        adam.zero_grad()
        value=prob.forward()
        loss,ok,_=prob.objective(value)
        with torch.no_grad():
            direction=(value[0][idx][:,WRISTS]-value[0][idx][:,ELBOWS]).numpy()
            maximum=max(maximum,float(direction_error(direction,targets).max()))
            assert maximum<.001 and torch.isfinite(loss).all()
            better=ok & (loss<losses)
            losses[better]=loss[better]
            steps[better.numpy()]=step
            for v,s in zip(variables,saved):
                s[better]=v[better]
        if step<STEPS:
            loss.sum().backward()
            adam.step()
            with torch.no_grad():
                prob.bodyvar.mul_(torch.clamp(prob.caps/prob.bodyvar.norm(dim=2).clamp_min(1e-8),max=1.)[...,None])
                lock(prob,idx,targets)
    with torch.no_grad():
        for v,s in zip(variables,saved):
            v.copy_(s)
        loss,ok,_=prob.objective(prob.forward())
        assert ok.all() and torch.allclose(loss,losses,atol=2e-6)
        body,beta=prob.parameters()
        path=OUT/'selected.npz'
        np.savez_compressed(path,body_pose=body.numpy(),global0=prob.global0.numpy(),delta=prob.delta.numpy(),translation=prob.tr.numpy(),betas=beta.numpy())
    p.write(RUN/'evidence/fit.json',dict(frames=FRAMES,methods=METHODS,steps=STEPS,source_hashes={k:p.sha(v) for k,v in files.items()},
                targets=targets.tolist(),max_direction_constraint_error_deg=maximum,steps_chosen=steps.tolist(),original_loss=losses.tolist(),selected_sha256=p.sha(path),exit_status=0))


def score():
    log=p.read(RUN/'evidence/fit.json')
    assert p.sha(OUT/'selected.npz')==log['selected_sha256']
    params=dict(np.load(OUT/'selected.npz'))
    items={r['pair']:r for r in p.read(SOURCE/'fit/prepared.json')['items']}
    truth=p.read(p.OUT/'validation_truth.json')
    ims=p.pair_images(p.VM)
    model,cam=p.ref.common.setup_model()
    cams=[cam(path) for path in p.ref.camera_files()]
    adapters=[p.read(path)['input_adapter'] for path in p.ref.camera_files()]
    obs=p.ref.common.OBS
    native=np.array(p.ref.common.NATIVE_IDS)[obs]
    result={}
    for n,f in enumerate(FRAMES):
        gt=np.array(truth[f]['gt'])[obs]
        gdir=gt[[list(native).index(w) for w in WRISTS]]-gt[[list(native).index(e) for e in ELBOWS]]
        target=log['targets'][n]
        result[f]=dict(triangulated_vs_gt_deg=direction_error(target,gdir).tolist(),methods={})
        item=items[f+'_annotated_all']; im=ims[int(f[-3:])-1]
        for b,method in enumerate(METHODS):
            joints=p.ref.full_joints(params,model.layer['neutral'],n*len(METHODS)+b)
            d=joints[WRISTS]-joints[ELBOWS]
            x=joints[native]
            v=p.ref.pixel_errors(item,x,im,p.mapping(),cams,adapters)
            err=1000*np.linalg.norm(x-gt,axis=1)
            result[f]['methods'][method]=dict(angle_deg=direction_error(d,gdir).tolist(),
                wrist_mm=[float(err[list(native).index(w)]) for w in WRISTS],nonpa=float(err.mean()),
                pa=float(1000*np.linalg.norm(p.align(x,gt)-gt,axis=1).mean()),
                rmse_px=float(np.sqrt(sum(a['sum_sq_px2']['fit_vs_gt'] for a in v)/sum(a['n'] for a in v))))
    p.write(RUN/'evidence/scores.json',result)

if __name__=='__main__':
    torch.set_num_threads(1)
    {'fit':fit,'score':score}[sys.argv[1]]()
