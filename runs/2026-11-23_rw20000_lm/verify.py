"""Audit LM initial states, selected states and pooled scores against the archived runs."""
from collections import Counter
import platform
import subprocess

import numpy as np
import scipy
import torch
from fit_lm import RUN, OUT, base, Frame, load_archived_start

def directional_check(problem, items):
    """Test the LM linearization against the native gradient off the initial pose."""
    rng=np.random.default_rng(116)
    cases={}
    for pair in ['pair001','pair073']:
        i=next(i for i,row in enumerate(items) if row['pair']==pair)
        frame=Frame(problem,i)
        x=frame.start.copy()
        x[:-3] += rng.normal(0,.025,len(x)-3)
        x[-3:] += rng.normal(0,.003,3)
        direction=rng.normal(size=len(x))
        direction/=np.linalg.norm(direction)
        body,_,delta,tr=frame.unpack(torch.from_numpy(x[None]))
        with torch.no_grad():
            problem.bodyvar[i].copy_(body[0,problem.body_ids])
            problem.delta[i].copy_(delta[0])
            problem.tr[i].copy_(tr[0])
        loss=problem.objective(problem.forward())[0][i]
        grad=torch.autograd.grad(loss,(problem.bodyvar,problem.delta,problem.tr))
        native=np.r_[grad[0][i,frame.active].flatten().numpy(),grad[1][i].numpy(),grad[2][i].numpy()] @ direction
        problem.fk.rest,problem.fk.shape=problem.fk.rest.double(),problem.fk.shape.double()
        with torch.no_grad():
            residuals,_=frame.residual(np.stack([x-1e-6*direction,x,x+1e-6*direction]))
        problem.fk.rest,problem.fk.shape=problem.fk.rest.float(),problem.fk.shape.float()
        deriv=residuals[1] @ ((residuals[2]-residuals[0]) / 2e-6)
        assert np.isclose(deriv,native,rtol=1e-4,atol=1e-5)
        assert np.isclose(.5*(residuals[1] @ residuals[1]),loss.item(),rtol=1e-4,atol=2e-6)
        cases[pair]=dict(lm_directional_derivative=float(deriv),reference_directional_derivative=float(native))
    return cases


def main():
    torch.set_num_threads(1)
    fit = base.read(RUN/'evidence/fit.json')
    scores = base.read(RUN/'evidence/scores.json')
    cfg = base.read(base.OUT/'protocol.json')
    items = [dict(r,condition='baseline') for r in base.read(base.OUT/'prepared.json')['items'] if r['condition']=='rw20000']
    model,camera = base.common.setup_model()
    base.fit.camera_files = base.camera_files
    problem = base.trial.TorsoOnly(items,cfg,model,camera)
    with torch.no_grad():
        assert load_archived_start(problem, items)==fit['seed_preload_differences']
    init = base.read(base.OUT/'initialization.json')
    indices = [i for i,(_,c,_) in enumerate(init['ids']) if c=='rw20000']
    assert [init['ids'][i][0] for i in indices]==[r['pair'] for r in items]
    archived = [np.asarray(v,dtype=np.float32)[indices] for v in init['initial_parameters']]
    assert all(np.array_equal(a,b.detach().numpy()) for a,b in zip(archived,problem.variables))
    assert np.array_equal(np.asarray(init['global0'],dtype=np.float32)[indices],problem.global0.numpy())
    with np.load(OUT/'lm.npz') as z:
        poses=z['body_pose'][:,problem.body_ids]
        cap_roundoff=float((np.linalg.norm(poses,axis=2)-problem.caps.double().numpy()).max())
        assert cap_roundoff <= 1e-12
        assert not z['betas'].any() and not z['body_pose'][:,[11,14]].any()
        assert np.array_equal(z['global0'],problem.global0.numpy())
    # Per-pair RMSEs require square pooling; the 2D counts can differ across frames.
    for method in ('adam','lm'):
        rows=[r[method] for r in scores['rows']]
        s=scores['summary'][method]
        for field in ('mpjpe_mm','pa_mpjpe_mm','nonwrist_mpjpe_mm'):
            assert np.isclose(np.mean([r[field] for r in rows]),s[field])
        for field in ('rmse_mm','wrist_rmse_mm'):
            assert np.isclose(np.sqrt(np.mean([r[field]**2 for r in rows])),s[field])
        assert np.isclose(np.sqrt(sum(r['gt_sq_px'] for r in rows)/sum(r['n2'] for r in rows)),s['gt_rmse_px'])
    with np.load(base.OUT/'budget_20000.npz') as params:
        adam_loss=params['loss'][indices].copy()
    # The published Adam control refit nine baseline frames.
    depth=base.ROOT/'outputs/rw20000_wrist_depth_20261123'
    pair_index={r['pair']:i for i,r in enumerate(items)}
    with np.load(depth/'budget_20000.npz') as params:
        for i,(pair,condition,_) in enumerate(base.read(depth/'fit_complete.json')['ids']):
            if condition=='baseline':
                adam_loss[pair_index[pair]]=params['loss'][i]
    lm_loss=np.array([r['selected_reference_loss'] for r in fit['rows']])
    diff=np.array([r['lm']['mpjpe_mm']-r['adam']['mpjpe_mm'] for r in scores['rows']])
    steps=[r['iterations'] for r in fit['rows']]
    gradients=directional_check(problem,items)
    base.write(RUN/'evidence/checks.json',dict(exit_status=0,base=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),seeds_identical=True,gradient_checks=gradients,pose_caps_pass=True,max_angle_cap_roundoff_rad=cap_roundoff,score_pooling_pass=True,
        pairs=len(steps),valid_view_points=sum(r['lm']['n2'] for r in scores['rows']),
        lm_feasible=fit['feasible'],stops=dict(Counter(r['stop'] for r in fit['rows'])),
        iterations_min=int(min(steps)),iterations_median=float(np.median(steps)),iterations_max=int(max(steps)),
        seed_max_abs_objective_diff=max(abs(r['initial_loss']-r['seed_reference_loss']) for r in fit['rows']),
        selection_max_abs_objective_diff=max(abs(r['selected_loss']-r['selected_reference_loss']) for r in fit['rows']),
        lower_selected_objective=int((lm_loss<adam_loss).sum()),lower_mpjpe=int((diff<0).sum()),
        median_mpjpe_difference_mm=float(np.median(diff)),python=platform.python_version(),
        scipy=scipy.__version__,numpy=np.__version__,torch=str(torch.__version__),
        commands=['<BODY_PYTHON> -Bu '+str(f.relative_to(base.ROOT)) for f in
                  [RUN/'fit_lm.py',RUN/'measure_lm.py',RUN/'verify.py']],device='cpu',threads=1))
    print('Verified initial parameters, caps, feasibility and six pooled metrics.',flush=True)


if __name__=='__main__':
    main()
