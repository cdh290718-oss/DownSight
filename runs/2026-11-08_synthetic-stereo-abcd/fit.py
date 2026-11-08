"""Controlled A-D joint fitting. No GT geometry access beyond extracted inputs."""
import argparse,hashlib,json,os,time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,RUN,OUT,OLD,read,head_rotation,angle_deg,project_torch,project_numpy,transform_torch,shoulder_excess
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import load
from egosmplx_pipeline.body_common import BODY_POSE_NAMES,MAX_DELTA_DEG,axis_angle_rotation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['A_left','A_right','B','C','D'],required=True);ap.add_argument('--seed',choices=['left','right'],default='left');a=ap.parse_args()
    label=a.case+'_'+a.seed;dest=OUT/label;dest.mkdir(exist_ok=False)
    cfg=read(RUN/'settings.json');inputs=read(OUT/'inputs.json');job=read(OLD/'fits'/('left' if a.seed=='left' else 'right_cpu')/'jobs'/(a.seed+'.json'));paths=job['config']['paths']
    os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo']);torch.set_num_threads(4)
    from egosmplx_pipeline import fusion
    fusion.initialize(model);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
    t=lambda x:torch.as_tensor(x,dtype=torch.float32)
    raw=load(Path(job['output'])/'raw/params.npz');head_target=t(inputs['head_position_m']);headR=t(inputs['head_rotation_matrix'])
    views=[a.case.split('_')[1]] if a.case.startswith('A_') else ['left','right']
    targets={}
    for side in views:
        view=inputs['views'][side];groups={}
        for g in ['body','wrist','shoulder']:
            ps=[p for p in view['body_points'] if p['group']==g];groups[g]=dict(ids=torch.tensor([p['index'] for p in ps]),xy=t([p['xy'] for p in ps]),weights=t([p['weight'] for p in ps]),multiplier=cfg['shoulder_weight'] if g=='shoulder' else 1.)
        for i,hand in enumerate(['left','right']):
            if hand not in view['hands']:continue
            h=view['hands'][hand];keep=h['included_indices']
            if not keep:continue
            ids=np.array([12+i]+list(range(25+20*i,45+20*i)))
            groups['hand_'+hand]=dict(ids=torch.tensor(ids[keep]),xy=t(np.array(h['xy'])[keep]),weights=t(np.ones(len(keep))*h['confidence']**2),multiplier=cfg['wilor_weight_per_hand'])
        targets[side]=groups
    fixed_keys=['betas','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression'];fixed={k:t(raw[k]).reshape(1,-1) for k in fixed_keys}
    body0=t(raw['body_pose']).reshape(21,3)
    # Solve global orientation from Head frame and body FK, not GT body orientation.
    local_chain=head_rotation(torch.zeros(3),body0).detach().numpy();R0=np.array(inputs['head_rotation_matrix'])@local_chain.T;global0=t(cv2.Rodrigues(R0)[0].reshape(3))
    with torch.no_grad():
        base=layer(body_pose=body0.reshape(1,-1),global_orient=global0.reshape(1,3),**fixed);translation0=head_target-base.joints[0,15];base_joints=base.joints[0,list(model.joint_idx)]+translation0
        floors={s:transform_torch(base_joints[[12,13]],inputs['views'][s]['camera']).norm(dim=1)*cfg['wrist_floor_fraction'] for s in views}
    if a.case=='D':fixed['betas']=t(read(OUT/'oracle_shape.json')['betas']).reshape(fixed['betas'].shape)
    active=list(MAX_DELTA_DEG);ids=torch.tensor([BODY_POSE_NAMES.index(n) for n in active]);prior_caps=t(np.deg2rad([MAX_DELTA_DEG[n] for n in active]));hard_caps=prior_caps*(cfg['relaxed_cap_multiplier'] if a.case=='C' else 1.)
    delta=torch.zeros(len(active),3,requires_grad=True);gd=torch.zeros(3,requires_grad=True);tr=translation0.detach().clone().requires_grad_(True)
    def forward():
        body=body0.clone();body[ids]+=delta
        local=layer(body_pose=body.reshape(1,-1),global_orient=global0.reshape(1,3),**fixed);pivot=local.joints[0,0];R=axis_angle_rotation(gd)
        mesh=(local.vertices[0]-pivot)@R.T+pivot+tr;native=(local.joints[0]-pivot)@R.T+pivot+tr
        return mesh,native,native[list(model.joint_idx)],R@head_rotation(global0,body)
    def objective(values):
        mesh,native,joints,R=values;terms={};feasible=True
        for side,groups in targets.items():
            camera=inputs['views'][side]['camera'];valid_ids=[]
            for name,g in groups.items():
                residual=project_torch(joints[g['ids']],camera,cfg['minimum_observed_joint_z_m'])-g['xy']
                if name=='shoulder':residual=shoulder_excess(residual,cfg['shoulder_tolerance_px'])
                e=F.smooth_l1_loss(residual/t([1280,720]),torch.zeros_like(residual),beta=.005,reduction='none').mean(1)
                terms[side+'/'+name]=g['multiplier']*(e*g['weights']).sum()/g['weights'].sum().clamp_min(1e-8)/len(views);valid_ids.extend(g['ids'].tolist())
            z=transform_torch(joints[sorted(set(valid_ids))],camera)[:,2]
            terms[side+'/positive_joint_guard']=.01*F.relu((cfg['minimum_observed_joint_z_m']-z)/.01).square().mean()/len(views)
            radius=transform_torch(joints[[12,13]],camera).norm(dim=1)
            terms[side+'/wrist_guard']=cfg['wrist_guard_weight']*F.relu((floors[side]-radius)/.05).square().mean()/len(views)
            feasible=feasible and bool((z>=cfg['minimum_observed_joint_z_m']).all() and (radius>=floors[side]-1e-6).all())
        terms['head_position']=cfg['head_position_weight']*((native[15]-head_target)/cfg['head_position_sigma_m']).square().sum()
        terms['head_orientation']=cfg['head_rotation_weight']*(R-headR).square().sum()/(2*np.deg2rad(cfg['head_rotation_sigma_deg'])**2)
        terms['pose_prior']=cfg['pose_prior_weight']*(delta/prior_caps[:,None]).square().mean()
        terms['translation_prior']=cfg['translation_prior_weight']*((tr-translation0)/t([.10,.10,.18])).square().mean()
        terms['rotation_prior']=cfg['rotation_prior_weight']*(gd/np.deg2rad(cfg['root_increment_limit_deg'])).square().mean()
        return sum(terms.values()),feasible,{k:float(v.detach()) for k,v in terms.items()}
    def save(label,step,values):
        mesh,native,joints,Rh=values;body=body0.clone();body[ids]+=delta.detach();R=axis_angle_rotation(gd).detach().numpy()@cv2.Rodrigues(global0.numpy().astype(float))[0]
        p={k:v.copy() for k,v in raw.items()};p.update({k:v.detach().numpy().reshape(raw[k].shape) for k,v in fixed.items()});p.update(body_pose=body.detach().numpy().reshape(raw['body_pose'].shape),global_orient=cv2.Rodrigues(R.astype(float))[0].astype(np.float32).reshape(1,3),transl=tr.detach().numpy().reshape(1,3))
        mv,jj=fusion.geometry(p);err=float(np.max(np.abs(mv-mesh.detach().numpy())));pixel_errors={}
        for side in views:
            ix=sorted(set(i for g in targets[side].values() for i in g['ids'].tolist()));camera=inputs['views'][side]['camera'];uv=project_numpy(jj[ix],camera);uv2=project_torch(joints[ix],camera).detach().numpy();pixel_errors[side]=float(np.max(np.abs(uv-uv2)))
        assert err<1e-5 and max(pixel_errors.values())<.05,(err,pixel_errors)
        p.update(smplx_mesh_cam=mv,smplx_body_joints_3d_render_cam=jj[:25],smplx_body_joints_3d_local=jj[:25]-p['transl'].reshape(3),smplx_joint_cam_root_relative=jj-jj[:1])
        np.savez_compressed(dest/(label+'.npz'),**p)
        return dict(step=step,head_position_error_mm=float((native[15]-head_target).norm()*1000),head_orientation_error_deg=angle_deg(Rh.detach().numpy(),headR.numpy()),head_rotation=Rh.detach().tolist(),pelvis_left_m=joints[0].detach().tolist(),minimum_mesh_z_m=float(mesh[:,2].min()),reload_mesh_max_m=err,reload_projection_max_px=pixel_errors,sha256=hashlib.sha256((dest/(label+'.npz')).read_bytes()).hexdigest())
    with torch.no_grad():initial=save('initial',0,forward())
    opt=torch.optim.Adam([{'params':[delta],'lr':cfg['pose_lr']},{'params':[gd],'lr':cfg['rotation_lr']},{'params':[tr],'lr':cfg['translation_lr']}]);best=None;best_loss=float('inf');best_step=None;trace=[];start=time.time()
    for step in range(cfg['steps']+1):
        opt.zero_grad();values=forward();loss,feasible,terms=objective(values)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
        if step%cfg['checkpoint_steps']==0 and feasible and float(loss.detach())<best_loss:best=[x.detach().clone() for x in [delta,gd,tr]];best_step=step;best_loss=float(loss.detach())
        if step%100==0 or step==cfg['steps']:
            q=dict(step=step,loss=float(loss.detach()),feasible=feasible,terms=terms,pelvis_left_m=values[2][0].detach().tolist(),head_position_error_mm=float((values[1][15]-head_target).norm().detach()*1000));trace.append(q);print(label,step,round(q['loss'],6),flush=True)
        if step==cfg['steps']:break
        loss.backward();assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [delta,gd,tr]);opt.step()
        with torch.no_grad():
            delta.mul_(torch.minimum(torch.ones_like(hard_caps),hard_caps/delta.norm(dim=1).clamp_min(1e-8))[:,None]);gd.mul_(torch.clamp(t(np.deg2rad(cfg['root_increment_limit_deg']))/gd.norm().clamp_min(1e-8),max=1.));tr[2].clamp_(.05,3.)
    if best is None:raise RuntimeError('No feasible checkpoint')
    with torch.no_grad():
        for x,b in zip([delta,gd,tr],best):x.copy_(b)
        values=forward();loss,feasible,terms=objective(values);assert feasible;final=save('final',best_step,values)
    q=dict(case=a.case,seed=a.seed,fit_views=views,coordinate_frame='left/cam3 CV',settings=cfg,initial=initial,final=final,trace=trace,best_step=best_step,recorded_best_loss=best_loss,selected_loss=float(loss),terms=terms,elapsed_seconds=time.time()-start,pose_increments_deg={n:float(x) for n,x in zip(active,np.rad2deg(delta.detach().norm(dim=1).numpy()))},hard_caps_deg={n:float(x) for n,x in zip(active,np.rad2deg(hard_caps.numpy()))},prior_caps_deg=MAX_DELTA_DEG,fixed_parameters=fixed_keys,shape_source='oracle' if a.case=='D' else 'network',wrist_floor_m={s:x.tolist() for s,x in floors.items()},wrist_distance_m={s:transform_torch(values[2][[12,13]],inputs['views'][s]['camera']).norm(dim=1).tolist() for s in views},joint_optimization=True,gt_inputs='Head pose and cameras; D additionally shape; no GT body pose/keypoints/silhouette',input_sha256=hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest())
    (dest/'optimization.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print('DONE',label,best_step,flush=True)
if __name__=='__main__':main()
