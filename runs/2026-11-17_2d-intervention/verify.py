"""Validate fast kinematics and loss against the original full mesh model."""
import copy
import sys
import time

import numpy as np
import torch

from common import EVIDENCE, OUT, RUN, read, setup_model, sha, write
from fit import Problem, forbid_gt
from geometry import project, rotation


def reference_forward(problem):
    body,beta = problem.parameters()
    b = len(problem.items)
    fixed = {k:torch.zeros(b,n) for k,n in [('left_hand_pose',45),('right_hand_pose',45),
                                          ('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
    full = problem.layer(body_pose=body.reshape(b,-1),betas=beta,
                         global_orient=problem.global0,**fixed)
    delta_r = rotation(problem.delta)
    pivot = full.joints[:,:1]
    native = (full.joints[:,:22]-pivot)@delta_r.transpose(1,2)+pivot+problem.tr[:,None]
    head_r = rotation(problem.global0)
    for j in [3,6,9,12,15]:head_r = head_r@rotation(body[:,j-1])
    return native,delta_r@head_r,beta


def advance(problem,steps,checkpoint_at=None):
    opt = torch.optim.Adam([dict(params=[v],lr=lr) for v,lr in zip(problem.variables,problem.lr)])
    saved=[v.detach().clone() for v in problem.variables]
    for step in range(steps):
        opt.zero_grad();loss=problem.objective(problem.forward())[0]
        if step==checkpoint_at:
            current=[v.detach().clone() for v in problem.variables]
            with torch.no_grad():
                for v,s in zip(problem.variables,saved):v.copy_(s)
                for v,s in zip(problem.variables,current):v.copy_(s)
            loss=problem.objective(problem.forward())[0]
        loss.sum().backward();opt.step()
        with torch.no_grad():
            norm = problem.bodyvar.norm(dim=2).clamp_min(1e-8)
            problem.bodyvar.mul_(torch.minimum(torch.ones_like(norm),problem.caps[None]/norm)[...,None])


def scalar_reference_objective(problem,values):
    """Original per-item/per-view reduction, independent of batched loss layout."""
    native,head,beta=values;cfg=problem.cfg;totals=[]
    def huber(r):
        d=cfg['metric_huber_delta']
        return torch.where(r<d,.5*r*r,d*(r-.5*d))
    for i,item in enumerate(problem.items):
        total=native.new_zeros(())
        for group in ['body','shoulder','wrist']:
            costs=[]
            for k,p in enumerate(item['points']):
                if p['group']!=group or not any(p['valid']):continue
                j=native[i,p['native_index']]
                if p['accepted'] and item['method']=='fixed_mixed':
                    rr=problem.white[i,k]@(j-problem.xyz[i,k])
                    cost=huber(rr.norm()/np.sqrt(3))
                else:
                    views=[]
                    for v in range(2):
                        if not p['valid'][v]:continue
                        tr=problem.transforms[i,v]
                        pixel=problem.cameras[v].world2camera_pytorch((tr[:3,:3]@j+tr[:3,3])[None])[0]
                        views.append(huber((pixel-problem.t(p['xy'][v])).norm()/(cfg['sigma_px_fixed']*np.sqrt(2))))
                    cost=torch.stack(views).mean()
                costs.append(cost)
            if costs:total=total+cfg['metric_loss_weight']*(cfg['shoulder_weight'] if group=='shoulder' else 1.)*torch.stack(costs).mean()
        joints=native[i,problem.native_ids]
        for v in range(2):
            transform=problem.transforms[i,v]
            z=(joints@transform[:3,:3].T+transform[:3,3])[:,2]
            total=total+.01*torch.relu((cfg['minimum_observed_joint_z_m']-z)/.01).square().mean()/2
        total=total+cfg['head_position_weight']*((native[i,15]-problem.head_pos[i])/cfg['head_position_sigma_m']).square().sum()
        total=total+cfg['head_rotation_weight']*(head[i]-problem.head_rot[i]).square().sum()/(2*np.deg2rad(cfg['head_rotation_sigma_deg'])**2)
        total=total+cfg['pose_prior_weight']*(problem.bodyvar[i]/problem.caps[:,None]).square().mean()
        total=total+cfg['shape_prior_weight']*beta[i].square().mean()
        totals.append(total)
    return torch.stack(totals)


def main():
    sys.addaudithook(forbid_gt)
    torch.set_num_threads(1);torch.manual_seed(20261117)
    model,cam = setup_model();torch.set_num_threads(1)
    cfg = read(RUN/'protocol.json'); prepared = read(OUT/'prepared.json')['items']
    subset = [p for p in prepared if p['pair'] in ['body01','body04','body10'] and p['condition']=='baseline']
    problem = Problem(subset,cfg,model,cam)
    comparisons = []
    for state in ['zero','random']:
        if state == 'random':
            with torch.no_grad():
                problem.bodyvar.normal_(0,.15);problem.delta.normal_(0,.04)
                problem.shapevar.normal_(0,.2);problem.tr.add_(torch.randn_like(problem.tr)*.01)
        fast = problem.forward(); ref = reference_forward(problem)
        forward_error = float((fast[0]-ref[0]).abs().max())
        orientation_error = float((fast[1]-ref[1]).abs().max())
        loss,_,_ = problem.objective(fast); reference_loss,_,_ = problem.objective(ref)
        gradients = torch.autograd.grad(loss.sum(),problem.variables,retain_graph=True)
        reference_gradients = torch.autograd.grad(reference_loss.sum(),problem.variables)
        with torch.no_grad():
            scalar_error=float((problem.objective(problem.forward())[0]-scalar_reference_objective(problem,problem.forward())).abs().max())
        errors = [float((a-b).abs().max()) for a,b in zip(gradients,reference_gradients)]
        relative = [float((a-b).norm()/b.norm().clamp_min(1e-8)) for a,b in zip(gradients,reference_gradients)]
        row = dict(state=state,body_joint_max_difference_m=forward_error,
                   head_rotation_max_difference=orientation_error,
                   objective_max_difference=float((loss-reference_loss).abs().max()),
                   scalar_vs_batched_objective_max_difference=scalar_error,
                   gradient_max_absolute_difference=errors,gradient_relative_l2_difference=relative)
        comparisons.append(row)
        assert forward_error<2e-6 and orientation_error<2e-6,row
        assert scalar_error<2e-5,row
        assert max(relative)<5e-4,row
    # Compare the vectorized projection with the existing pipeline projection.
    with torch.no_grad():
        native = problem.forward()[0]
        projection_error = max(float((project(native,c)-c.world2camera_pytorch(native.reshape(-1,3)).reshape(native.shape[0],22,2)).abs().max())
                               for c in problem.cameras)
    assert projection_error < .02,projection_error
    # Batch independence: each row's optimization agrees with the same row alone.
    one_cfg = dict(cfg,methods=['2d'])
    multi = Problem(subset,one_cfg,model,cam)
    single = Problem(subset[:1],one_cfg,model,cam)
    advance(multi,50);advance(single,50)
    independence = [float((a[0]-b[0]).abs().max()) for a,b in zip(multi.variables,single.variables)]
    assert max(independence)<2e-4,independence
    checkpointed=Problem(subset[:1],one_cfg,model,cam)
    advance(checkpointed,50,checkpoint_at=25)
    snapshot_errors=[float((a-b).abs().max()) for a,b in zip(checkpointed.variables,single.variables)]
    assert max(snapshot_errors)<1e-6,snapshot_errors
    # Fixed-variable fingerprints, independent of intervention and method.
    all_problem = Problem(prepared,cfg,model,cam)
    initialization_equal = True
    for name in cfg['pairs']:
        ix = [k for k,item in enumerate(all_problem.items) if item['pair']==name]
        for tensor in [all_problem.global0,all_problem.basis,all_problem.valid,all_problem.white,
                       all_problem.head_pos,all_problem.head_rot,all_problem.tr]:
            assert all(torch.equal(tensor[k],tensor[ix[0]]) for k in ix)
        for method in cfg['methods']:
            mi = [k for k in ix if all_problem.items[k]['method']==method]
            assert all(torch.equal(all_problem.route[k],all_problem.route[mi[0]]) for k in mi)
    # Benchmark representative full-size batches before choosing CPU thread count.
    benchmarks = []
    for threads in [1,2,4]:
        torch.set_num_threads(threads)
        start=time.time()
        for _ in range(30):
            for v in all_problem.variables:v.grad=None
            all_problem.objective(all_problem.forward())[0].sum().backward()
        benchmarks.append(dict(threads=threads,seconds_per_forward_backward=(time.time()-start)/30))
    result = dict(command='<BODY_PYTHON> runs/2026-11-17_2d-intervention/verify.py',exit_status=0,
                  comparisons=comparisons,projection_max_difference_px=projection_error,
                  batch_vs_single_50step_parameter_max_differences=independence,
                  checkpoint_save_restore_vs_uninterrupted_parameter_differences=snapshot_errors,
                  fixed_controls_verified=initialization_equal,
                  fits_in_batch=len(all_problem.items),cpu_benchmarks=benchmarks,
                  recommended_threads=min(benchmarks,key=lambda r:r['seconds_per_forward_backward'])['threads'],
                  protocol_sha256=sha(RUN/'protocol.json'),
                  geometry_source_sha256=sha(RUN/'geometry.py'),fit_source_sha256=sha(RUN/'fit.py'))
    write(EVIDENCE/'verification.json',result)
    print(result,flush=True)


if __name__ == '__main__':
    main()
