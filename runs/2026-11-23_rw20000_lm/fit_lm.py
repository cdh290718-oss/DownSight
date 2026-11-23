"""Projected, diagonally damped Levenberg–Marquardt for the frozen v2 objective."""
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.linalg import solve

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_frozen200_free12'))
import run_fit as base
from geometry import project

OUT = ROOT / 'outputs/rw20000_lm_20261123'
MAX_STEPS = 400
FD_STEP = 1e-6


class Frame:
    def __init__(self, problem, i):
        self.problem = problem
        self.i = i
        self.cfg = problem.cfg
        self.ids = problem.body_ids
        self.count = len(self.ids)
        self.active = (problem.mask[0, self.ids, 0] != 0)
        self.caps = problem.caps[self.active].double()
        self.pose_dims = int(self.active.sum()) * 3
        self.seed = problem.global0[i].double()
        self.start = np.r_[np.zeros(self.pose_dims + 3), problem.tr[i].detach().numpy()]
        valid = problem.valid[i].double()
        weights = torch.zeros(valid.shape[1], dtype=torch.double)
        for group, masks in problem.group_ids.items():
            weight = self.cfg['shoulder_weight'] if group == 'shoulder' else 1.
            mask = masks[i]
            weights += weight * mask.double() / mask.sum().clamp_min(1)
        self.weights = (self.cfg['metric_loss_weight'] * weights * valid /
                        valid.sum(0).clamp_min(1)).sqrt()
        self.selected = problem.selected[i]
        self.transforms = problem.transforms[i].double()

    def unpack(self, x):
        pose = x[:, :self.pose_dims].reshape(len(x), -1, 3)
        body = x.new_zeros((len(x), 21, 3))
        body[:, self.ids[self.active]] = pose
        return body, pose, x[:, -6:-3], x[:, -3:]

    def residual(self, array):
        x = torch.from_numpy(np.atleast_2d(array))
        body, pose, delta, translation = self.unpack(x)
        p, cfg = self.problem, self.cfg
        joints, _ = p.fk(body, x.new_zeros((len(x), 10)), self.seed.expand(len(x), 3), delta, translation)
        viewed = (torch.einsum('vij,bnj->bvni', self.transforms[:, :3, :3], joints[:, p.native_ids])
                  + self.transforms[None, :, None, :3, 3])
        uv = torch.stack([project(viewed[:, v], cam) for v, cam in enumerate(p.cameras)], 1)
        diff = (uv - p.xy[self.i]) / (cfg['sigma_px_fixed'] * np.sqrt(2.))
        d = diff.norm(dim=-1)
        threshold = cfg['metric_huber_delta']
        # Each 2-vector contributes exactly the weighted radial Huber to 0.5*r.T*r.
        tail = (2 * threshold * d - threshold**2).clamp_min(0).sqrt() / d.clamp_min(1e-15)
        factor = torch.where(d <= threshold, 1., tail) * self.weights
        pixel = diff * factor[..., None]
        head = (joints[:, 15] - p.head_pos[self.i]) * np.sqrt(2 * cfg['head_position_weight']) / cfg['head_position_sigma_m']
        prior = pose / self.caps[None, :, None] * np.sqrt(2 * cfg['pose_prior_weight'] / (3 * self.count))
        z = viewed[:, :, self.selected, 2]
        guard = ((cfg['minimum_observed_joint_z_m'] - z) / .01).clamp_min(0) * np.sqrt(.01 / int(self.selected.sum()))
        residual = torch.cat([term.flatten(1) for term in (pixel, head, prior, guard)], 1)
        feasible = (z >= cfg['minimum_observed_joint_z_m']).all(2).all(1)
        return residual.numpy(), feasible.numpy()

    def project_caps(self, x):
        pose = x[:self.pose_dims].reshape(-1, 3)
        pose *= np.minimum(1., self.caps.numpy() / np.linalg.norm(pose, axis=1).clip(1e-15))[:, None]
        return x


def optimize(frame, checkpoint):
    x = frame.start.copy()
    r, feasible = frame.residual(x)
    r, ok = r[0], bool(feasible[0])
    loss = r @ r / 2
    initial = loss
    best, bestloss, beststep = x.copy(), (loss if ok else np.inf), 0
    damping = 1e-3
    accepted, stale, linearize = 0, 0, True
    for step in range(1, MAX_STEPS + 1):
        if linearize:
            h = FD_STEP * np.maximum(1., abs(x))
            perturbed, _ = frame.residual(x + np.diag(h))
            jac = ((perturbed - r) / h[:, None]).T
            normal, gradient = jac.T @ jac, jac.T @ r
            diagonal = np.maximum(normal.diagonal(), 1e-8)
        update = solve(normal + np.diag(damping * diagonal), -gradient, assume_a='pos')
        trial = frame.project_caps(x + update)
        update = trial - x
        predicted = -(gradient @ update + .5 * update @ normal @ update)
        candidate, candidate_ok = frame.residual(trial)
        candidate = candidate[0]
        new_loss = candidate @ candidate / 2
        gain = (loss - new_loss) / predicted if predicted > 0 else -1.
        linearize = gain > 0 and np.isfinite(new_loss)
        improvement = 0.
        if linearize:
            improvement = (loss - new_loss) / max(loss, 1e-15)
            x, r, loss, ok = trial, candidate, new_loss, bool(candidate_ok[0])
            accepted += 1
        if gain > .75:
            damping = max(1e-12, damping / 3)
        elif gain < .25:
            damping = min(1e15, damping * 3)
        stale = stale + 1 if improvement < 1e-8 else 0
        if step % checkpoint == 0:
            if ok and loss < bestloss:
                best, bestloss, beststep = x.copy(), loss, step
            if stale >= checkpoint and np.isfinite(bestloss):
                break
    assert np.isfinite(bestloss), 'no feasible checkpoint'
    return best, dict(initial_loss=initial, selected_loss=bestloss, selected_step=beststep,
                      iterations=step, accepted=accepted, stop='budget' if step==MAX_STEPS else 'stagnation')


def load_archived_start(problem, items):
    init = base.read(base.OUT/'initialization.json')
    indices = [i for i,(_,c,_) in enumerate(init['ids']) if c=='rw20000']
    assert [init['ids'][i][0] for i in indices] == [r['pair'] for r in items]
    assert init['protocol_sha256']==base.sha(base.OUT/'protocol.json')
    diffs = {}
    for key, current, saved in zip(('body','delta','translation','shape','global0'),
                                   problem.variables+[problem.global0],
                                   init['initial_parameters']+[init['global0']]):
        target = torch.from_numpy(np.asarray(saved, dtype=np.float32)[indices])
        diffs[key] = float((target-current).abs().max())
        assert diffs[key] <= (2e-7 if key=='translation' else 0.)
        current.copy_(target)
        assert torch.equal(current,target)
    return diffs


def main():
    def protect(event, args):
        if event == 'open' and args and isinstance(args[0], str):
            if Path(args[0]).name in {'reference.json','results.json','fit_results.json','budget_20000.npz','samples.npz','body_gt_official.npz'}:
                raise PermissionError('Scoring data and Adam fits are excluded from LM fitting.')
    sys.addaudithook(protect)
    torch.set_num_threads(1)
    frozen = base.OUT
    audit = base.read(base.RUN/'evidence/inputs.json')
    assert base.sha(frozen/'prepared.json') == audit['solver_input_sha256']
    assert base.sha(frozen/'protocol.json') == audit['protocol_sha256']
    for f in base.camera_files():
        assert base.sha(f)==audit['cameras'][f.name]
    items = [r for r in base.read(frozen/'prepared.json')['items'] if r['condition']=='rw20000']
    assert len({r['pair'] for r in items})==len(items)==100
    cfg = base.read(frozen/'protocol.json')
    model, cam = base.common.setup_model()
    base.fit.camera_files = base.camera_files
    # The legacy constructor looks up its fixed shape basis under 'baseline'.
    p = base.trial.TorsoOnly([dict(r,condition='baseline') for r in items],cfg,model,cam)
    assert not p.basis.any() and not p.route.any() and p.native_ids.max()<22
    with torch.no_grad():
        seed_roundoff = load_archived_start(p, items)
        init_loss, init_ok, _ = p.objective(p.forward())
    # Preserve its FP32 rest skeleton and starting parameters, then evaluate LM in FP64.
    p.fk.rest = p.fk.rest.double()
    p.fk.shape = p.fk.shape.double()
    chosen, diagnostics = [], []
    start = time.monotonic()
    with torch.no_grad():
        for i,item in enumerate(items):
            frame = Frame(p,i)
            x, row = optimize(frame, cfg['checkpoint_steps'])
            assert np.isclose(row['initial_loss'], init_loss[i].item(),rtol=1e-4,atol=2e-6)
            body, _, delta, tr = frame.unpack(torch.from_numpy(x[None]))
            chosen.append((body[0].numpy(), delta[0].numpy(), tr[0].numpy()))
            row.update(pair=item['pair'], seed_feasible=bool(init_ok[i]), seed_reference_loss=init_loss[i].item())
            diagnostics.append(row)
            if (i+1)%10==0: print('completed',i+1,'/ 100',round(time.monotonic()-start,1),'s',flush=True)
        body, delta, tr = (np.stack(values) for values in zip(*chosen))
        p.bodyvar.copy_(torch.tensor(body[:,p.body_ids]))
        p.delta.copy_(torch.tensor(delta))
        p.tr.copy_(torch.tensor(tr))
        p.fk.rest, p.fk.shape = p.fk.rest.float(), p.fk.shape.float()
        values, feasible, _ = p.objective(p.forward())
        assert feasible.all()
        for row,val in zip(diagnostics,values.tolist()):
            row['selected_reference_loss']=val
            assert np.isclose(row['selected_loss'],val,rtol=1e-4,atol=2e-6)
    OUT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(OUT/'lm.npz',body_pose=body,delta=delta,translation=tr,
                        global0=p.global0.numpy(),betas=np.zeros((len(items),10)))
    base.write(RUN/'evidence/fit.json',dict(rows=diagnostics,elapsed_s=time.monotonic()-start,
               max_steps=MAX_STEPS,checkpoint_steps=cfg['checkpoint_steps'],fd_step=FD_STEP,
               protocol_sha256=audit['protocol_sha256'],input_sha256=audit['solver_input_sha256'],
               camera_sha256=audit['cameras'],params_sha256=base.sha(OUT/'lm.npz'),feasible=int(feasible.sum()),
               seed_preload_differences=seed_roundoff,initialization_sha256=base.sha(frozen/'initialization.json'),
               torch_version=str(torch.__version__),exit_status=0))


if __name__=='__main__':
    main()
