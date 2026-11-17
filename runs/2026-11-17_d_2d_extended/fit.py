"""Independent, vectorized fixed-condition fits; no body-3D GT reader.

Each batch row has its own Adam moments and parameters. Summing independent row
objectives gives the same per-row gradients as fitting the rows individually.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import EVIDENCE, OUT, ROOT, RUN, camera_files, read, setup_model, sha, write
from geometry import BodyKinematics, project
from egosmplx_pipeline.body_common import BODY_POSE_NAMES


def forbid_gt(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.path.basename(os.fsdecode(args[0]))
        if name in ['samples.npz', 'body_gt_official.npz', 'params.npz', 'oracle_shape.json']:
            raise PermissionError('Body-3D/network ground-truth access prohibited during fitting: ' + name)


def baseline_basis(layer, model, points, threshold):
    lookup = {p['index']:p for p in points if p['accepted']}
    rest = (layer.J_regressor @ layer.v_template).detach().numpy()
    derivative = torch.einsum('jv,vck->jck', layer.J_regressor, layer.shapedirs[:,:,:10]).detach().numpy()
    info = np.zeros((10,10))
    for a,b in [(8,9),(1,2),(8,10),(9,11),(10,12),(11,13),(1,3),(2,4),(3,5),(4,6)]:
        if a not in lookup or b not in lookup:
            continue
        pa,pb = lookup[a],lookup[b]
        d = np.array(pa['xyz'])-np.array(pb['xyz']); u = d/np.linalg.norm(d)
        variance = u@(np.array(pa['covariance'])+np.array(pb['covariance']))@u
        ja,jb = model.joint_idx[a],model.joint_idx[b]
        rest_d = rest[ja]-rest[jb]
        grad = rest_d/np.linalg.norm(rest_d)@(derivative[ja]-derivative[jb])
        info += np.outer(grad,grad)/variance
    eigenvalues, vec = np.linalg.eigh(info)
    retained = vec[:,eigenvalues >= threshold]
    basis = np.zeros((10,10),np.float32); basis[:,:retained.shape[1]] = retained
    return basis, dict(eigenvalues=eigenvalues.tolist(), rank=retained.shape[1], basis=basis.tolist())


class Problem:
    def __init__(self, prepared, cfg, model, camera_cls, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.layer = model.layer['neutral'].cpu().eval()
        self.layer.requires_grad_(False)
        self.fk = BodyKinematics(self.layer)
        self.cameras = [camera_cls(p) for p in camera_files()]
        self.t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        # CPU keeps model read-only and leaves the occupied GPU untouched.
        assert device == 'cpu'
        base = {i['pair']: i for i in prepared if i['condition'] == 'baseline'}
        bases = {name: (np.zeros((10,10),np.float32),dict(rank=0,reason='fixed mean shape for matched comparison')) for name in base}
        self.initialization = {name: value[1] for name,value in bases.items()}
        self.items = [dict(item, method=method) for method in cfg['methods'] for item in prepared]
        self.ids = [(i['pair'],i['condition'],i['method']) for i in self.items]
        b = len(self.items)
        self.basis = self.t(np.array([bases[i['pair']][0] for i in self.items]))
        self.native_ids = torch.tensor([p['native_index'] for p in self.items[0]['points']])
        self.xy = self.t(np.array([[p['xy'] for p in i['points']] for i in self.items])).transpose(1,2)
        self.valid = self.t(np.array([[p['valid'] for p in i['points']] for i in self.items])).transpose(1,2)
        self.route = self.t([[p['accepted'] and i['method'] in ('fixed_mixed','3d') for p in i['points']]
                             for i in self.items])
        self.xyz = self.t([[p.get('xyz',[0.,0.,0.]) for p in i['points']] for i in self.items])
        self.white = self.t(np.array([[np.linalg.inv(np.linalg.cholesky(np.array(p['covariance'])))
                                      if p['accepted'] else np.eye(3) for p in i['points']] for i in self.items]))
        self.transforms = self.t(np.array([[i['inputs']['views'][s]['camera']['left_to_view']
                                            for s in ['left','right']] for i in self.items]))
        self.head_pos = self.t([i['anchor']['position'] for i in self.items])
        self.head_rot = self.t([i['anchor']['rotation'] for i in self.items])
        self.global0 = self.t(np.array([cv2.Rodrigues(np.array(i['anchor']['rotation'],float))[0].reshape(3)
                                        for i in self.items]))
        self.active = list(cfg['absolute_pose_caps_deg'])
        self.body_ids = torch.tensor([BODY_POSE_NAMES.index(n) for n in self.active])
        self.caps = self.t(np.deg2rad([cfg['absolute_pose_caps_deg'][n] for n in self.active]))
        self.bodyvar = torch.zeros(b,len(self.active),3,requires_grad=True)
        self.delta = torch.zeros(b,3,requires_grad=True)
        self.shapevar = torch.zeros(b,10,requires_grad=True)
        body = torch.zeros(b,21,3)
        with torch.no_grad():
            native,_ = self.fk(body,torch.zeros(b,10),self.global0,torch.zeros(b,3),torch.zeros(b,3))
        self.tr = (self.head_pos-native[:,15]).detach().clone().requires_grad_(True)
        self.variables = [self.bodyvar,self.delta,self.tr,self.shapevar]
        self.lr = [cfg['pose_lr'],cfg['rotation_lr'],cfg['translation_lr'],cfg['shape_lr']]
        self.group_ids = {}
        selected = self.valid.sum(1).gt(0)
        for row,item in enumerate(self.items):
            if item['method']=='3d':
                selected[row]=torch.tensor([p['accepted'] for p in item['points']],device=selected.device)
        for name in ['body','shoulder','wrist']:
            mask = self.t([p['group'] == name for p in self.items[0]['points']])
            self.group_ids[name] = (mask*selected).float()
        self.selected = selected

    def parameters(self):
        body = torch.zeros(len(self.items),21,3)
        body[:,self.body_ids] = self.bodyvar
        beta = (self.basis @ self.shapevar[...,None])[...,0]
        return body,beta

    def forward(self):
        body,beta = self.parameters()
        native,head_rotation = self.fk(body,beta,self.global0,self.delta,self.tr)
        return native,head_rotation,beta

    def objective(self, values):
        native,head_rotation,beta = values
        cfg = self.cfg
        joints = native[:,self.native_ids]
        viewed = torch.einsum('bvij,bnj->bvni', self.transforms[:,:,:3,:3], joints)+self.transforms[:,:,:3,3][:,:,None]
        projections = torch.stack([project(viewed[:,v],cam) for v,cam in enumerate(self.cameras)],1)
        d2 = (projections-self.xy).norm(dim=-1)/(cfg['sigma_px_fixed']*np.sqrt(2))
        costs2 = (F.huber_loss(d2,torch.zeros_like(d2),delta=cfg['metric_huber_delta'],reduction='none')
                  * self.valid).sum(1)/self.valid.sum(1).clamp_min(1)
        residual = torch.einsum('bnij,bnj->bni',self.white,joints-self.xyz)
        d3 = residual.norm(dim=-1)/np.sqrt(3)
        costs3 = F.huber_loss(d3,torch.zeros_like(d3),delta=cfg['metric_huber_delta'],reduction='none')
        costs = self.route*costs3+(1-self.route)*costs2
        terms = {}
        for name,mask in self.group_ids.items():
            multiplier = cfg['shoulder_weight'] if name == 'shoulder' else 1.
            terms['metric/'+name] = cfg['metric_loss_weight']*multiplier*(costs*mask).sum(1)/mask.sum(1).clamp_min(1)
        z = viewed[:,:,:,2]
        guard = F.relu((cfg['minimum_observed_joint_z_m']-z)/.01).square()
        terms['positive_guard'] = .01*(guard*self.selected[:,None]).sum((1,2))/(2*self.selected.sum(1).clamp_min(1))
        feasible = ((z >= cfg['minimum_observed_joint_z_m']) | ~self.selected[:,None]).all(2).all(1)
        terms['head_position'] = cfg['head_position_weight']*((native[:,15]-self.head_pos)/cfg['head_position_sigma_m']).square().sum(1)
        terms['head_orientation'] = cfg['head_rotation_weight']*(head_rotation-self.head_rot).square().sum((1,2))/(2*np.deg2rad(cfg['head_rotation_sigma_deg'])**2)
        terms['pose_prior'] = cfg['pose_prior_weight']*(self.bodyvar/self.caps[None,:,None]).square().mean((1,2))
        terms['shape_prior'] = cfg['shape_prior_weight']*beta.square().mean(1)
        return sum(terms.values()),feasible,terms


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--threads',type=int,default=1)
    a = ap.parse_args()
    sys.addaudithook(forbid_gt)
    cfg = read(RUN/'protocol.json')
    prepared = read(OUT/'prepared.json')['items']
    assert not (OUT/'fit_complete.json').exists(), 'Do not overwrite completed fit.'
    assert not (OUT/'trace.jsonl').exists(), 'Do not overwrite a prior trace.'
    model,cam = setup_model(); torch.set_num_threads(a.threads)
    torch.manual_seed(20261117)
    problem = Problem(prepared,cfg,model,cam)
    variables = problem.variables
    optimizer = torch.optim.Adam([dict(params=[v],lr=lr) for v,lr in zip(variables,problem.lr)])
    b = len(problem.items)
    best = [v.detach().clone() for v in variables]
    bestloss = torch.full((b,),float('inf')); beststep = torch.full((b,),-1,dtype=torch.long)
    history = []
    write(OUT/'initialization.json',dict(initialization=problem.initialization,ids=problem.ids,
                                        global0=problem.global0.tolist(),
                                        initial_parameters=[v.detach().tolist() for v in variables],
                                        protocol_sha256=sha(RUN/'protocol.json')))
    start = time.time()
    with (OUT/'trace.jsonl').open('x') as log:
        for step in range(max(cfg['budgets'])+1):
            optimizer.zero_grad()
            values = problem.forward()
            loss,feasible,terms = problem.objective(values)
            if not bool(torch.isfinite(loss).all()):
                bad = torch.where(~torch.isfinite(loss))[0].tolist()
                write(OUT/'fit_failure.json',dict(step=step,bad=[problem.ids[i] for i in bad],exit_status=1))
                raise RuntimeError('Nonfinite training objective')
            if step % cfg['checkpoint_steps'] == 0:
                with torch.no_grad():
                    improved = feasible & (loss < bestloss)
                    bestloss[improved] = loss[improved]
                    beststep[improved] = step
                    for current,saved in zip(variables,best):
                        saved[improved] = current[improved]
            if step % cfg['trace_steps'] == 0:
                row = dict(step=step,loss=loss.detach().tolist(),
                           joints=values[0][:,problem.native_ids].detach().tolist())
                history.append(row)
                log.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');log.flush()
                if step % 1000 == 0:
                    print('step',step,'/',max(cfg['budgets']),'feasible',int(feasible.sum()),'/',b,
                          'mean loss',float(loss.mean()),'elapsed_s',round(time.time()-start,1),flush=True)
            if step in cfg['budgets']:
                current = [v.detach().clone() for v in variables]
                with torch.no_grad():
                    for v,saved in zip(variables,best):v.copy_(saved)
                    body,beta = problem.parameters()
                    np.savez_compressed(OUT/('budget_'+str(step)+'.npz'),
                                        body_pose=body.detach().numpy(),betas=beta.detach().numpy(),
                                        delta=problem.delta.detach().numpy(),translation=problem.tr.detach().numpy(),
                                        global0=problem.global0.detach().numpy(),loss=bestloss.numpy(),selected_step=beststep.numpy())
                    selected_loss,selected_feasible,_ = problem.objective(problem.forward())
                    assert torch.allclose(selected_loss[beststep>=0],bestloss[beststep>=0],atol=1e-6,rtol=1e-5)
                    before = history[-1-cfg['convergence']['window_steps']//cfg['trace_steps']]
                    relative = np.abs(np.array(history[-1]['loss'])-np.array(before['loss']))/np.maximum(np.abs(before['loss']),1e-12)
                    motion = np.linalg.norm(np.array(history[-1]['joints'])-np.array(before['joints']),axis=2).max(1)*1000
                    write(OUT/('diagnostic_'+str(step)+'.json'),dict(
                        budget=step,ids=problem.ids,selected_step=beststep.tolist(),
                        selected_feasible=selected_feasible.tolist(),
                        relative_loss_change=relative.tolist(),max_joint_motion_mm=motion.tolist(),
                        convergence_pass=((relative<cfg['convergence']['relative_loss_change_max']) &
                                          (motion<cfg['convergence']['max_joint_motion_mm'])).tolist()))
                    for v,saved in zip(variables,current):v.copy_(saved)
                # Saving touches parameter versions even when restored exactly.
                # Rebuild the graph before continuing the same Adam trajectory.
                values = problem.forward()
                loss,feasible,terms = problem.objective(values)
            if step == max(cfg['budgets']):break
            loss.sum().backward()
            assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in variables)
            factor = next(f for s,f in reversed(cfg['lr_schedule']) if step >= s)
            for group,lr in zip(optimizer.param_groups,problem.lr):group['lr'] = lr*factor
            optimizer.step()
            with torch.no_grad():
                norm = problem.bodyvar.norm(dim=2).clamp_min(1e-8)
                problem.bodyvar.mul_(torch.minimum(torch.ones_like(norm),problem.caps[None]/norm)[...,None])
    write(OUT/'fit_complete.json',dict(exit_status=0,ids=problem.ids,steps=max(cfg['budgets']),
                                     elapsed_seconds=time.time()-start,protocol_sha256=sha(RUN/'protocol.json'),
                                     final_selected_steps=beststep.tolist(),device='cpu',threads=a.threads,
                                     body_gt_read=False))
    print('DONE',b,'fits',flush=True)


if __name__ == '__main__':
    main()
