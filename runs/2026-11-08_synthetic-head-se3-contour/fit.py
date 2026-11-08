"""Joint body/root optimization with Oracle Head SE(3), optional Sapiens contour."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import read, load
from egosmplx_pipeline.body_common import BODY_POSE_NAMES, MAX_DELTA_DEG, JOINT_MAPPING, axis_angle_rotation
from common import head_rotation, rotation_penalty, angle_deg, shoulder_excess, positive_points, weighted_mean
from contour import Contours

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT/'outputs/synthetic_head_se3_contour_20261108'
OLD = ROOT/'outputs/synthetic_original_20261108'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--side', choices=['left','right'],required=True);ap.add_argument('--contour',type=int,choices=[0,1],required=True);a=ap.parse_args()
    case='se3_contour_joint' if a.contour else 'se3_joint'
    dest=OUT/a.side/case;dest.mkdir(parents=True,exist_ok=False)
    cfg=read(RUN/'settings.json');job=read(OLD/'fits'/('left' if a.side=='left' else 'right_cpu')/'jobs'/(a.side+'.json'))
    paths=job['config']['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo'])
    model,Camera=setup(paths['egosmplx_repo'])
    from egosmplx_pipeline import fusion
    fusion.initialize(model);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
    parents=layer.parents.detach().numpy();assert list(parents[[3,6,9,12,15]])==[0,3,6,9,12]
    t=lambda x:torch.as_tensor(x,dtype=torch.float32)
    source=Path(job['output'])/'raw/params.npz';raw=load(source)
    obs=read(job['row']['predictions']['sapiens']);lookup={q['name']:q for q in obs['keypoints']}
    target_data=read(OUT/(a.side+'_targets.json'));head_target=t(target_data['head_position_m']);rotation_target=t(target_data['head_rotation_matrix'])
    assert obs['image_sha256']==target_data['image_sha256']
    cam=Camera(job['row']['calibration'])
    names=[n for n in JOINT_MAPPING if n!='nose' and n in lookup and np.isfinite([lookup[n]['x'],lookup[n]['y'],lookup[n]['score']]).all() and lookup[n]['score']>=.3 and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
    indices=torch.tensor([model.joints_name.index(JOINT_MAPPING[n]) for n in names]);target=t([[lookup[n]['x'],lookup[n]['y']] for n in names]);weights=t([min(1.,lookup[n]['score'])**2 for n in names]);scale=t([1280,720])
    groups={n:torch.tensor([i for i,k in enumerate(names) if (k.endswith('_shoulder') if n=='shoulder' else k.endswith('_wrist') if n=='wrist' else not k.endswith(('_shoulder','_wrist')))],dtype=torch.long) for n in ['body','wrist','shoulder']}
    fixed_keys=['betas','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression'];fixed={k:t(raw[k]).reshape(1,-1) for k in fixed_keys}
    body0=t(raw['body_pose']).reshape(21,3);raw_global=t(raw['global_orient']).reshape(3)
    raw_head_rotation=head_rotation(raw_global,body0,parents).detach().numpy()
    alignment=np.asarray(target_data['head_rotation_matrix'])@raw_head_rotation.T
    initial_R=alignment@cv2.Rodrigues(raw['global_orient'].astype(float).reshape(3))[0]
    global0=t(cv2.Rodrigues(initial_R)[0].reshape(3))
    with torch.no_grad():
        init=layer(body_pose=body0.reshape(1,-1),global_orient=global0.reshape(1,3),**fixed)
        translation0=head_target-init.joints[0,15]
        initial_head_rotation=head_rotation(global0,body0,parents).numpy()
        initial_anchor_error=float((init.joints[0,15]+translation0-head_target).norm())
        assert initial_anchor_error<1e-6 and angle_deg(initial_head_rotation,rotation_target.numpy())<.1
    active=list(MAX_DELTA_DEG);active_ids=torch.tensor([BODY_POSE_NAMES.index(n) for n in active]);caps=t(np.deg2rad([MAX_DELTA_DEG[n] for n in active]))
    delta=torch.zeros(len(active),3,requires_grad=True);gd=torch.zeros(3,requires_grad=True);tr=translation0.detach().clone().requires_grad_(True)
    # Target file has only the Head frame and observed segmentation, not GT body geometry.
    masks=np.load(OUT/(a.side+'_contour_masks.npz'))
    contour_targets={n:dict(q,target=np.asarray(q['target'],np.float32).reshape(-1,2),arm_exclusion=masks['torso_arm_exclusion'] if n=='torso' else np.zeros((720,1280),np.uint8)) for n,q in target_data['targets'].items()}
    contour=Contours(model,cam,contour_targets,cfg)

    def forward():
        body=body0.clone();body[active_ids]+=delta
        local=layer(body_pose=body.reshape(1,-1),global_orient=global0.reshape(1,3),**fixed)
        pivot=local.joints[0,0];rotation=axis_angle_rotation(gd)
        mesh=(local.vertices[0]-pivot)@rotation.T+pivot+tr
        native=(local.joints[0]-pivot)@rotation.T+pivot+tr
        joints=native[list(model.joint_idx)]
        pixels=cam.world2camera_pytorch(positive_points(joints[indices],cfg['minimum_observed_joint_z_m']))
        hR=rotation@head_rotation(global0,body,parents)
        return mesh,native,joints,pixels,hR

    with torch.no_grad():initial=forward();wrist_floor=initial[2][[12,13]].norm(dim=1)*cfg['wrist_floor_fraction']

    def objective(values):
        mesh,native,joints,pixels,hR=values;terms={}
        for n,ix in groups.items():
            residual=pixels[ix]-target[ix]
            if n=='shoulder':residual=shoulder_excess(residual,cfg['shoulder_tolerance_px'])
            e=F.smooth_l1_loss(residual/scale,torch.zeros_like(residual),beta=.005,reduction='none').mean(1)
            terms[n]=weighted_mean(e,weights[ix])*(cfg['shoulder_weight'] if n=='shoulder' else 1.)
        terms['head_position']=cfg['head_position_weight']*((native[15]-head_target)/cfg['head_position_sigma_m']).square().sum()
        terms['head_orientation']=cfg['head_rotation_weight']*rotation_penalty(hR,rotation_target,np.deg2rad(cfg['head_rotation_sigma_deg']))
        terms['pose_prior']=cfg['pose_prior_weight']*(delta/caps[:,None]).square().mean()
        terms['translation_prior']=cfg['translation_prior_weight']*((tr-translation0)/t([.10,.10,.18])).square().mean()
        terms['rotation_prior']=cfg['rotation_prior_weight']*(gd/np.deg2rad(cfg['root_increment_limit_deg'])).square().mean()
        terms['positive_joint_guard']=.01*F.relu((cfg['minimum_observed_joint_z_m']-joints[indices,2])/.01).square().mean()
        terms['wrist_guard']=cfg['wrist_guard_weight']*F.relu((wrist_floor-joints[[12,13]].norm(dim=1))/.05).square().mean()
        if a.contour:
            terms['contour'],contour_terms=contour.loss(mesh)
        else:contour_terms={}
        loss=sum(terms.values())
        feasible=bool((joints[indices,2]>=cfg['minimum_observed_joint_z_m']).all() and (joints[[12,13]].norm(dim=1)>=wrist_floor-1e-6).all())
        feasible=feasible and (not a.contour or contour.valid())
        return loss,feasible,{k:float(v.detach()) for k,v in terms.items()},contour_terms

    def save_state(label,step,values):
        mesh,native,joints,pixels,hR=values
        body=body0.detach().clone();body[active_ids]+=delta.detach()
        R=axis_angle_rotation(gd).detach().numpy()@cv2.Rodrigues(global0.numpy().astype(float))[0]
        p={k:v.copy() for k,v in raw.items()};p.update(body_pose=body.numpy().reshape(raw['body_pose'].shape),global_orient=cv2.Rodrigues(R.astype(float))[0].astype(np.float32).reshape(1,3),transl=tr.detach().numpy().reshape(1,3))
        mv,jj=fusion.geometry(p);err=float(np.max(np.abs(mv-mesh.detach().numpy())))
        uv=cam.world2camera(jj[indices.numpy()].astype(float));perr=float(np.max(np.abs(uv-pixels.detach().numpy())))
        if err>1e-5 or perr>.05:raise RuntimeError('Reload mismatch %g m %g selected px'%(err,perr))
        p.update(smplx_mesh_cam=mv,smplx_body_joints_3d_render_cam=jj[:25],smplx_body_joints_3d_local=jj[:25]-p['transl'].reshape(3),smplx_body_joints_2d_render_cam=cam.world2camera(jj[:25].astype(float)),smplx_joint_cam_root_relative=jj-jj[:1])
        assert all(np.array_equal(p[k],raw[k]) for k in fixed_keys)
        np.savez_compressed(dest/(label+'.npz'),**p)
        return dict(label=label,step=step,head_position_error_mm=float((native[15]-head_target).norm().detach()*1000),head_orientation_error_deg=angle_deg(hR.detach().numpy(),rotation_target.numpy()),head_rotation=hR.detach().tolist(),head_position=native[15].detach().tolist(),pelvis_position=joints[0].detach().tolist(),minimum_vertex_z_m=float(mesh[:,2].min().detach()),reload_mesh_max_m=err,reload_selected_px_max=perr,source_sha256=hashlib.sha256((dest/(label+'.npz')).read_bytes()).hexdigest())

    with torch.no_grad():
        contour.refresh(initial[0]);initial_record=save_state('initial',0,initial)
    opt=torch.optim.Adam([{'params':[delta],'lr':cfg['pose_lr']},{'params':[gd],'lr':cfg['rotation_lr']},{'params':[tr],'lr':cfg['translation_lr']}])
    best=None;best_loss=float('inf');best_step=None;trace=[];start=time.time()
    for step in range(cfg['steps']+1):
        opt.zero_grad();values=forward()
        checkpoint=(step%cfg['contour_refresh_steps']==0 or step==cfg['steps'])
        if checkpoint and a.contour:contour.refresh(values[0])
        loss,feasible,terms,contour_terms=objective(values)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint objective')
        if checkpoint and feasible and float(loss.detach())<best_loss:
            best_loss=float(loss.detach());best=[x.detach().clone() for x in [delta,gd,tr]];best_step=step
        if step%100==0 or step==cfg['steps']:
            entry=dict(step=step,loss=float(loss.detach()),feasible=feasible,terms=terms,contour_terms=contour_terms,contour_diagnostics=contour.last_diagnostics if a.contour else {},head_position_error_mm=float((values[1][15]-head_target).norm().detach()*1000),head_orientation_error_deg=angle_deg(values[4].detach().numpy(),rotation_target.numpy()),pelvis_position_m=values[2][0].detach().tolist())
            trace.append(entry);print(a.side,case,step,round(entry['loss'],6),round(entry['head_orientation_error_deg'],3),flush=True)
        if step==cfg['steps']:break
        loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [delta,gd,tr])
        opt.step()
        with torch.no_grad():
            delta.mul_(torch.minimum(torch.ones_like(caps),caps/delta.norm(dim=1).clamp_min(1e-8))[:,None]);gd.mul_(torch.clamp(t(np.deg2rad(cfg['root_increment_limit_deg']))/gd.norm().clamp_min(1e-8),max=1.));tr[2].clamp_(.05,3.)
    if best is None:raise RuntimeError('No feasible joint optimization state')
    with torch.no_grad():
        for x,b in zip([delta,gd,tr],best):x.copy_(b)
        values=forward();contour.refresh(values[0]);loss,feasible,terms,ct=objective(values);assert feasible
        final_record=save_state('final',best_step,values)
    report=dict(side=a.side,case=case,settings=cfg,initial=initial_record,final=final_record,trace=trace,selected_best_loss=float(loss),recorded_best_loss=best_loss,best_step=best_step,steps=cfg['steps'],elapsed_seconds=time.time()-start,selected_points=names,contour_enabled=bool(a.contour),contour_diagnostics=contour.last_diagnostics,terms=terms,contour_terms=ct,pose_increments_deg={n:float(x) for n,x in zip(active,np.rad2deg(delta.detach().norm(dim=1).numpy()))},fixed_parameters=fixed_keys,all_fixed_parameters_equal=True,wrist_floor_m=wrist_floor.tolist(),wrist_distances_m=values[2][[12,13]].norm(dim=1).tolist(),initialization='single rigid alignment of predicted Head frame to oracle Head frame; then all parameters optimized together, no alternating stages',source_raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),observations_sha256=hashlib.sha256(Path(job['row']['predictions']['sapiens']).read_bytes()).hexdigest())
    (dest/'optimization.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DONE',a.side,case,best_step,flush=True)

if __name__=='__main__':main()
