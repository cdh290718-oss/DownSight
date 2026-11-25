"""Fit on detector inputs only, then evaluate against official wrists in a separate process."""
from pathlib import Path
import copy
import sys

import numpy as np
import torch

from detect import EYES, FRAMES, OUT, ROOT, RUN, digest, read, write

sys.path.insert(0, str(ROOT/'runs/2026-11-24_stereo_forearm_seed'))
import forearm_seed as seed

p = seed.prev
SOURCES = ('rw','wilor','guided')
STARTS = ('v3','stereo')
STEPS = 1000


def prepare():
    detections = read(RUN/'evidence/detections.json')
    guided = read(RUN/'evidence/guided.json')
    photos = read(OUT/'images.json')['frames']
    inputs = {r['pair']:r for r in read(p.VAL/'prepared.json')['items']}
    adapters = [read(path)['input_adapter'] for path in p.ref.camera_files()]
    items = {}
    for n,frame in enumerate(FRAMES):
        rw = inputs[frame]
        items[frame] = {'rw':rw}
        for src,observations in [('wilor',detections),('guided',guided)]:
            mixed = copy.deepcopy(rw)
            for pt in mixed['points']:
                if pt['name'] not in ('L_Wrist','R_Wrist'):
                    continue
                for v in range(2):
                    name = Path(photos[2*n+v]['image'])
                    assert name.stem.endswith(EYES[v]) and digest(name) == detections['input_sha256'][name.stem]
                    uv = observations['selected'][name.stem][pt['name'][0]]['uv']
                    if pt['valid'][v] and uv is not None:
                        assert np.isfinite(uv).all()
                        pt['xy'][v] = (np.asarray(uv)*adapters[v]['scale']+adapters[v]['offset']).tolist()
            items[frame][src] = mixed
    return items,adapters


def fit():
    sys.addaudithook(seed.block_labels)
    inputs,adapters = prepare()
    bank_path = seed.joint.OUT/'validation_bank.npz'
    choices_path = seed.joint.OUT/'validation_choices.json'
    bank,choices = dict(np.load(bank_path)),read(choices_path)
    order = [(f,s,t) for f in FRAMES for s in SOURCES for t in STARTS]
    rows = [inputs[f][s] for f,s,t in order]
    assert not any(pt['accepted'] for row in rows for pt in row['points'])
    ids = [choices[f]['offset']+choices[f]['best'] for f,s,t in order]
    original = {k:bank[k][ids] for k in seed.joint.FIELDS}
    p.base.fit.camera_files=p.ref.camera_files
    model,cls=p.ref.common.setup_model()
    prob=p.base.trial.TorsoOnly(rows,read(p.VAL/'protocol.json'),model,cls)
    caps=dict(zip(prob.body_ids.tolist(),prob.caps.tolist()))
    with torch.no_grad():
        for var,value in [(prob.bodyvar,original['body_pose'][:,prob.body_ids]),(prob.tr,original['translation']),
                          (prob.delta,original['delta'])]:
            var.copy_(torch.tensor(value))
        assert not original['betas'].any() and np.allclose(prob.global0.numpy(),original['global0'])
        xyz=prob.forward()[0].numpy().copy()
        for i,(frame,source,start) in enumerate(order):
            if start=='stereo':
                opts,_=seed.make_seeds(rows[i],{k:v[i] for k,v in original.items()},xyz[i],prob.fk.parents,
                                       caps,prob.cameras,adapters)
                prob.bodyvar[i].copy_(torch.tensor(opts['both'][prob.body_ids]))
    variables=[prob.bodyvar,prob.delta,prob.tr]
    adam=torch.optim.Adam([dict(params=[v],lr=lr) for v,lr in zip(variables,prob.lr)])
    saved=[v.detach().clone() for v in variables]
    best=torch.full((len(rows),),float('inf'))
    step_ids=torch.full((len(rows),),-1)
    for step in range(STEPS+1):
        adam.zero_grad()
        loss,ok,_=prob.objective(prob.forward())
        assert torch.isfinite(loss).all()
        with torch.no_grad():
            if step==0:
                for i,(f,s,t) in enumerate(order):
                    if s=='rw' and t=='v3':
                        assert abs(float(loss[i])-choices[f]['loss'][choices[f]['best']])<2e-6
            improved=ok & (loss<best)
            best[improved],step_ids[improved]=loss[improved],step
            for var,checkpoint in zip(variables,saved):
                checkpoint[improved]=var[improved]
        if step==STEPS:
            break
        loss.sum().backward()
        assert all(torch.isfinite(v.grad).all() for v in variables)
        adam.step()
        with torch.no_grad():
            prob.bodyvar.mul_(torch.clamp(prob.caps/prob.bodyvar.norm(dim=2).clamp_min(1e-8),max=1.)[...,None])
    with torch.no_grad():
        for v,x in zip(variables,saved):
            v.copy_(x)
        loss,ok,_=prob.objective(prob.forward())
        assert ok.all() and torch.allclose(loss,best,atol=2e-6)
        body,beta=prob.parameters()
        np.savez_compressed(OUT/'best.npz',body_pose=body.numpy(),delta=prob.delta.numpy(),betas=beta.numpy(),
                            translation=prob.tr.numpy(),global0=prob.global0.numpy())
    selected={f:{s:min(STARTS,key=lambda t:float(best[order.index((f,s,t))])) for s in SOURCES} for f in FRAMES}
    paths=[p.VAL/'protocol.json',p.VAL/'prepared.json',bank_path,choices_path,RUN/'evidence/detections.json',RUN/'evidence/guided.json',
           *p.ref.camera_files(),OUT/'best.npz']
    write(RUN/'evidence/fit.json',dict(order=order,loss=best.tolist(),step=step_ids.tolist(),budget=STEPS,
          selected=selected,sha256={q.name:digest(q) for q in paths},exit_status=0))


def score():
    inputs,adapters=prepare()
    log=read(RUN/'evidence/fit.json')
    assert digest(OUT/'best.npz')==log['sha256']['best.npz']
    pack=dict(np.load(OUT/'best.npz'))
    truth=read(p.OUT/'validation_truth.json')
    official={r['pair']:r for r in read(ROOT/'outputs/gt_projection_elbows_20261124/fit/prepared.json')['items']}
    model,cls=p.ref.common.setup_model()
    cameras=[cls(path) for path in p.ref.camera_files()]
    obs=p.ref.common.OBS
    names=[p.ref.common.NAMES[i] for i in obs]
    native=np.asarray(p.ref.common.NATIVE_IDS)[obs]
    frames={}
    for f in FRAMES:
        frames[f]={}
        gt=np.asarray(truth[f]['gt'])[obs]
        anchors={pt['name']:pt for pt in official[f+'_annotated_all']['points']}
        for src in SOURCES:
            row=inputs[f][src]
            pts={pt['name']:pt for pt in row['points']}
            transform=np.asarray(row['inputs']['views']['right']['camera']['left_to_view'])
            dst=frames[f][src]=dict(wrist={},fits={})
            for side in ('L','R'):
                w,e=[side+'_'+part for part in ('Wrist','Elbow')]
                a,b=[seed.triangulate(pts[name],transform,cameras,adapters) for name in (w,e)]
                assert b is not None
                rays=[seed.joint.unit(cam.camera2world_ray(np.asarray(xy)[None])[0])
                      for cam,xy in zip(cameras,pts[w]['xy'])]
                rot,tr=transform[:3,:3],transform[:3,3]
                ray_depths=np.linalg.lstsq(np.column_stack([rays[0],-rot.T@rays[1]]),-rot.T@tr,rcond=None)[0]
                wg,eg=gt[[names.index(w),names.index(e)]]
                uv=[(np.asarray(pt['xy'])-np.array([d['offset'] for d in adapters]))/
                    np.array([d['scale'] for d in adapters])[:,None] for pt in (pts[w],anchors[w])]
                dst['wrist'][side]=dict(ray_depths_m=ray_depths.tolist(),uv_px=uv[0].tolist(),official_px=uv[1].tolist(),
                        error_px=np.linalg.norm(uv[0]-uv[1],axis=1).tolist(),
                        triangulated_mm=float(1000*np.linalg.norm(a-wg)) if a is not None else None,
                        forearm_deg=seed.angle(a-b,wg-eg) if a is not None else None)
            for t in STARTS:
                i=log['order'].index([f,src,t])
                x=p.ref.full_joints(pack,model.layer['neutral'],i)[native]
                err=1000*np.linalg.norm(x-gt,axis=1)
                dst['fits'][t]=dict(nonpa_mm=float(err.mean()),pa_mm=float(1000*np.linalg.norm(p.align(x,gt)-gt,axis=1).mean()),
                    wrist_mm={},forearm_deg={})
                for side in ('L','R'):
                    w,e=[names.index(side+'_'+part) for part in ('Wrist','Elbow')]
                    dst['fits'][t]['wrist_mm'][side]=float(err[w])
                    dst['fits'][t]['forearm_deg'][side]=seed.angle(x[w]-x[e],gt[w]-gt[e])
            dst['selected']=log['selected'][f][src]
    write(RUN/'evidence/scores.json',dict(frames=frames,eye_order=EYES,exit_status=0))


if __name__=='__main__':
    torch.set_num_threads(1)
    {'fit':fit,'score':score}[sys.argv[1]]()
