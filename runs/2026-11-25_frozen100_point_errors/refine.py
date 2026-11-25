"""Repeat the frozen four-start refinement on the archived test split."""
from pathlib import Path
import sys
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/frozen100_point_errors_20261125'
sys.path.insert(0,str(ROOT/'runs/2026-11-24_stereo_forearm_seed'))
from forearm_seed import prev,joint,block_labels,make_seeds,BRANCHES,STEPS

def fit():
    sys.addaudithook(block_labels)
    bank_path = joint.OUT/'test_bank.npz'
    choice_path = joint.OUT/'test_choices.json'
    prepared_path = prev.base.OUT/'prepared.json'
    protocol_path = prev.base.OUT/'protocol.json'
    spec=prev.read(joint.RUN/'evidence/test_search.json')
    for path,key in [(bank_path,'parameter_sha256'),(choice_path,'choices_sha256'),(prepared_path,'prepared_sha256'),(protocol_path,'protocol_sha256')]:
        assert prev.sha(path)==spec[key]
    bank, choices = dict(np.load(bank_path)), prev.read(choice_path)
    items = {r['pair']: r for r in prev.read(prepared_path)['items'] if r['condition']=='rw20000'}
    assert len(items)==len(choices)==100
    frames=sorted(items)
    rows = [dict(items[f],condition='baseline') for f in frames for _ in BRANCHES]
    assert all(not pt['accepted'] for row in rows for pt in row['points'])
    prev.base.fit.camera_files = prev.ref.camera_files
    model, camera = prev.ref.common.setup_model()
    cfg = prev.read(protocol_path)
    p = prev.base.trial.TorsoOnly(rows, cfg, model, camera)
    adapters = [prev.read(path)['input_adapter'] for path in prev.ref.camera_files()]
    ids = [choices[f]['offset']+choices[f]['best'] for f in frames for _ in BRANCHES]
    original = {k: bank[k][ids] for k in joint.FIELDS}
    caps = dict(zip(p.body_ids.tolist(), p.caps.tolist()))
    meta = {}
    with torch.no_grad():
        assert not original['betas'].any()
        assert np.allclose(original['global0'], p.global0.numpy(), atol=1e-6)
        for var, value in [(p.delta, original['delta']), (p.tr, original['translation']),
                           (p.bodyvar, original['body_pose'][:, p.body_ids.tolist()])]:
            var.copy_(torch.tensor(value))
        xyz = p.forward()[0].numpy().astype(float)
        for j, f in enumerate(frames):
            i = j*len(BRANCHES)
            poses, diag = make_seeds(items[f], {k: v[i] for k, v in original.items()}, xyz[i],
                                     p.fk.parents, caps, p.cameras, adapters)
            for k, branch in enumerate(BRANCHES):
                p.bodyvar[i+k].copy_(torch.tensor(poses[branch][p.body_ids.tolist()]))
            meta[f] = dict(triangulation=diag, archive_name=choices[f]['names'][choices[f]['best']])

    variables = [p.bodyvar, p.delta, p.tr]
    optimizer = torch.optim.Adam([dict(params=[var], lr=lr) for var, lr in zip(variables, p.lr)])
    best = [v.detach().clone() for v in variables]
    best_loss = torch.full((len(rows),), float('inf'))
    best_step = torch.full((len(rows),), -1)
    for step in range(STEPS+1):
        optimizer.zero_grad()
        loss, ok, _ = p.objective(p.forward())
        assert torch.isfinite(loss).all()
        with torch.no_grad():
            if step == 0:
                init_loss, init_ok = loss.tolist(), ok.tolist()
                for j, f in enumerate(frames):
                    i = j*len(BRANCHES)
                    assert init_ok[i] and abs(init_loss[i]-choices[f]['loss'][choices[f]['best']]) < 2e-6
            better = ok & (loss < best_loss)
            best_loss[better] = loss[better]
            best_step[better] = step
            for v, saved in zip(variables, best):
                saved[better] = v[better]
        if step == STEPS:
            break
        loss.sum().backward()
        assert all(torch.isfinite(v.grad).all() for v in variables)
        optimizer.step()
        with torch.no_grad():
            p.bodyvar.mul_(torch.clamp(p.caps/p.bodyvar.norm(dim=2).clamp_min(1e-8), max=1.)[...,None])
    with torch.no_grad():
        for v, saved in zip(variables, best):
            v.copy_(saved)
        loss, ok, _ = p.objective(p.forward())
        found=best_step>=0
        assert ok[found].all() and torch.allclose(loss[found], best_loss[found], atol=2e-6)
        body, beta = p.parameters()
        pack = dict(body_pose=body.numpy(), betas=beta.numpy(), delta=p.delta.numpy(),
                    translation=p.tr.numpy(), global0=p.global0.numpy())
        for j, f in enumerate(frames):
            i = j*len(BRANCHES)
            scores = [init_loss[i]] + best_loss[i:i+len(BRANCHES)].tolist()
            labels = ['archive'] + list(BRANCHES)
            meta[f].update(winner=labels[int(np.argmin(scores))], losses={t:float(v) if np.isfinite(v) else None for t,v in zip(labels,scores)},
                           initial_loss=dict(zip(BRANCHES, init_loss[i:i+len(BRANCHES)])),
                           initial_feasible=dict(zip(BRANCHES, init_ok[i:i+len(BRANCHES)])),
                           steps=dict(zip(BRANCHES, best_step[i:i+len(BRANCHES)].tolist())))
        np.savez_compressed(OUT/'checkpoints.npz', **pack)
    inputs = [bank_path, choice_path, prepared_path, protocol_path, *prev.ref.camera_files()]
    prev.write(RUN/'evidence/selection.json', dict(frames=meta, steps_per_start=STEPS,
               fields=BRANCHES, hashes={path.name:prev.sha(path) for path in inputs},
               checkpoints_sha256=prev.sha(OUT/'checkpoints.npz'), exit_status=0))


if __name__=='__main__':
    torch.set_num_threads(1)
    fit()
