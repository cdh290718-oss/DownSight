"""Two-frame diagnostic. Select checkpoints in fit, unlock body labels only in score."""
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import least_squares

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-23_fullbody_joint_candidates'))
import search as joint

prev = joint.prev
OUT = ROOT / 'outputs/stereo_forearm_seed_20261124'
FRAMES = ('val031', 'val055')
BRANCHES = ('old', 'left', 'right', 'both')
STEPS = 1000


def block_labels(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = str(args[0]).lower()
        if any(k in name for k in ('truth', 'samples.npz', 'reference.json', 'scores.json',
                                   'validation_results', 'gt_projection_elbows', 'test_manifest')):
            raise PermissionError('Body evaluation resource requested while fitting')


def triangulate(point, transform, cameras, adapters):
    if not all(point['valid']):
        return None
    uv = np.asarray(point['xy'])
    rays = [joint.unit(cam.camera2world_ray(pixel[None])[0]) for cam, pixel in zip(cameras, uv)]
    rotation, translation = transform[:3, :3], transform[:3, 3]
    origin = -rotation.T @ translation
    direction = rotation.T @ rays[1]
    matrix = np.column_stack([rays[0], -direction])
    depths, _, rank, _ = np.linalg.lstsq(matrix, origin, rcond=None)
    if rank < 2 or min(depths) <= 0:
        return None
    start = (depths[0]*rays[0] + origin + depths[1]*direction)/2

    def residual(x):
        viewed = (x, rotation @ x + translation)
        return np.concatenate([(cam.world2camera(y[None])[0] - pix)/adapter['scale']
                               for cam, y, pix, adapter in zip(cameras, viewed, uv, adapters)])

    sol = least_squares(residual, start)
    if sol.success and np.isfinite(sol.x).all() and min(sol.x[2], (rotation @ sol.x + translation)[2]) > 0:
        return sol.x
    return None


def make_seeds(item, original, xyz, parents, caps, cameras, adapters):
    body = original['body_pose']
    worlds = [joint.rot(original['delta']) @ joint.rot(original['global0'])]
    for k in range(1, 22):
        worlds.append(worlds[parents[k]] @ joint.rot(body[k-1]))
    points = {p['native_index']: p for p in item['points']}
    transform = np.asarray(item['inputs']['views']['right']['camera']['left_to_view'])
    poses = {branch: body.copy() for branch in BRANCHES}
    diagnostics = {}
    for side, (s, e, w) in zip(('left', 'right'), joint.CHAINS[:2]):
        elbow, wrist = [triangulate(points[k], transform, cameras, adapters) for k in (e, w)]
        info = diagnostics[side] = {'available': False}
        if elbow is None or wrist is None:
            continue
        upper = elbow - xyz[s]
        forearm = wrist - elbow
        if min(np.linalg.norm(upper), np.linalg.norm(forearm)) < 1e-6:
            continue
        swing = joint.turns((xyz[e]-xyz[s])[None], upper[None])[0]
        shoulder_world = swing @ worlds[s]
        bend = joint.turns((swing @ (xyz[w]-xyz[e]))[None], forearm[None])[0]
        elbow_world = bend @ swing @ worlds[e]
        local = [joint.rv(worlds[parents[s]].T @ shoulder_world), joint.rv(shoulder_world.T @ elbow_world)]
        info.update(available=True, clipped=[], direction=joint.unit(forearm).tolist())
        for k, angle in zip((s-1, e-1), local):
            scale = min(1., caps[k]/max(np.linalg.norm(angle), 1e-12))
            info['clipped'].append(bool(scale < 1))
            poses[side][k] = poses['both'][k] = scale * angle
    return poses, diagnostics


def fit():
    sys.addaudithook(block_labels)
    bank_path = joint.OUT/'validation_bank.npz'
    choice_path = joint.OUT/'validation_choices.json'
    prepared_path = prev.VAL/'prepared.json'
    protocol_path = prev.VAL/'protocol.json'
    bank, choices = dict(np.load(bank_path)), prev.read(choice_path)
    items = {r['pair']: r for r in prev.read(prepared_path)['items']}
    rows = [items[f] for f in FRAMES for _ in BRANCHES]
    assert all(not pt['accepted'] for row in rows for pt in row['points'])
    prev.base.fit.camera_files = prev.ref.camera_files
    model, camera = prev.ref.common.setup_model()
    cfg = prev.read(protocol_path)
    p = prev.base.trial.TorsoOnly(rows, cfg, model, camera)
    adapters = [prev.read(path)['input_adapter'] for path in prev.ref.camera_files()]
    ids = [choices[f]['offset']+choices[f]['best'] for f in FRAMES for _ in BRANCHES]
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
        for j, f in enumerate(FRAMES):
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
                for j, f in enumerate(FRAMES):
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
        assert ok.all() and torch.allclose(loss, best_loss, atol=2e-6)
        body, beta = p.parameters()
        pack = dict(body_pose=body.numpy(), betas=beta.numpy(), delta=p.delta.numpy(),
                    translation=p.tr.numpy(), global0=p.global0.numpy())
        for j, f in enumerate(FRAMES):
            i = j*len(BRANCHES)
            scores = [init_loss[i]] + best_loss[i:i+len(BRANCHES)].tolist()
            labels = ['archive'] + list(BRANCHES)
            meta[f].update(winner=labels[int(np.argmin(scores))], losses=dict(zip(labels, scores)),
                           initial_loss=dict(zip(BRANCHES, init_loss[i:i+len(BRANCHES)])),
                           initial_feasible=dict(zip(BRANCHES, init_ok[i:i+len(BRANCHES)])),
                           steps=dict(zip(BRANCHES, best_step[i:i+len(BRANCHES)].tolist())))
        np.savez_compressed(OUT/'checkpoints.npz', **pack)
    inputs = [bank_path, choice_path, prepared_path, protocol_path, *prev.ref.camera_files()]
    prev.write(RUN/'evidence/selection.json', dict(frames=meta, steps_per_start=STEPS,
               fields=BRANCHES, hashes={path.name:prev.sha(path) for path in inputs},
               checkpoints_sha256=prev.sha(OUT/'checkpoints.npz'), exit_status=0))


def angle(a, b):
    return float(np.rad2deg(np.arccos(np.clip(joint.unit(a) @ joint.unit(b), -1, 1))))


def score():
    ev = prev.read(RUN/'evidence/selection.json')
    assert ev['checkpoints_sha256'] == prev.sha(OUT/'checkpoints.npz')
    pack = dict(np.load(OUT/'checkpoints.npz'))
    bank = dict(np.load(joint.OUT/'validation_bank.npz'))
    choices = prev.read(joint.OUT/'validation_choices.json')
    truth = prev.read(prev.OUT/'validation_truth.json')
    layer = prev.ref.common.setup_model()[0].layer['neutral']
    obs = prev.ref.common.OBS
    names = [prev.ref.common.NAMES[i] for i in obs]
    native = np.asarray(prev.ref.common.NATIVE_IDS)[obs]
    result = {}
    for j, f in enumerate(FRAMES):
        gt = np.asarray(truth[f]['gt'])[obs]
        choice = choices[f]
        start = choice['offset']
        positions = [prev.ref.full_joints(bank, layer, start+k)[native] for k in range(len(choice['names']))]
        old_errors = (1000*np.linalg.norm(np.asarray(positions)-gt, axis=2).mean(1))
        oracle = int(old_errors.argmin())
        row = result[f] = dict(old_bank_size=len(positions), old_bank_best_non_pa_mm=float(old_errors[oracle]),
                               old_bank_best_non_pa_name=choice['names'][oracle], methods={},
                               selected=ev['frames'][f]['winner'], triangulated_direction_deg={})
        assert choice['best'] == int(np.argmin(choice['loss']))
        row['old_bank'] = []
        for index, x in enumerate(positions):
            angles = {}
            for side in ('L', 'R'):
                e, w = [names.index(side+'_'+name) for name in ('Elbow', 'Wrist')]
                angles[side] = angle(x[w]-x[e], gt[w]-gt[e])
            row['old_bank'].append(dict(name=choice['names'][index], archived_loss=choice['loss'][index],
                                        non_pa_mm=float(old_errors[index]), forearm_deg=angles))
        for tag, x in [('archive', positions[choice['best']])] + [
                (branch, prev.ref.full_joints(pack, layer, j*len(BRANCHES)+k)[native])
                for k, branch in enumerate(BRANCHES)]:
            errs = 1000*np.linalg.norm(x-gt, axis=1)
            metric = dict(non_pa_mm=float(errs.mean()), pa_mm=float(1000*np.linalg.norm(prev.align(x,gt)-gt,axis=1).mean()),
                          forearm_deg={}, wrist_mm={})
            for side in ('left', 'right'):
                prefix = side[0].upper()+'_'
                e, w = [names.index(prefix+n) for n in ('Elbow', 'Wrist')]
                metric['forearm_deg'][side] = angle(x[w]-x[e], gt[w]-gt[e])
                metric['wrist_mm'][side] = float(errs[w])
                diag = ev['frames'][f]['triangulation'][side]
                if diag['available']:
                    row['triangulated_direction_deg'][side] = angle(diag['direction'], gt[w]-gt[e])
            row['methods'][tag] = metric
    prev.write(RUN/'evidence/evaluation.json', dict(frames=result, truth_sha256=prev.sha(prev.OUT/'validation_truth.json'),
                                                   exit_status=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    {'fit':fit, 'score':score}[sys.argv[1]]()
