"""Four initialization factors; one seed-independent objective. No GT body reader."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,RUN,OUT,OLD,read,head_rotation,angle_deg,project_numpy,project_torch,transform_torch
from initialization import shape_basis,root_from_head
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import BODY_POSE_NAMES,axis_angle_rotation

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();case=next(c for c in read(RUN/'cases.json') if c['case']==a.case);cfg=read(RUN/'settings.json')
 allowed={n+'.json' for n in ['shape','pose'] if case[n+'_seed']=='network'};reads=[]
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).resolve()
  if ((ROOT/'outputs') in path.parents and OUT not in path.parents) or (path.parent==OUT/'seeds' and path.name not in allowed) or path.name in ['oracle_shape.json','params.npz']:
   raise PermissionError('Forbidden network/GT data read in this fitting case')
  if path.suffix in ['.json','.npz'] and ROOT in path.parents:reads.append(str(path.relative_to(ROOT)))
 sys.addaudithook(audit)
 dest=OUT/(a.case+'_left');dest.mkdir(exist_ok=False);paths=read(OUT/'model_paths.json');os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo']);torch.set_num_threads(4)
 from egosmplx_pipeline import fusion
 fusion.initialize(model);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
 t=lambda x:torch.as_tensor(x,dtype=torch.float32)
 inputs=read(OUT/'inputs.json');anchor=read(OUT/'anchors/oracle.json');points=read(OUT/'triangulated.json')['points'];accepted=[p for p in points if p['accepted']];target_ids=sorted(set(p['index'] for p in accepted));head_target=t(anchor['position']);headR=t(anchor['rotation'])
 fixed={k:torch.zeros(1,n) for k,n in [('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
 body_seed=np.zeros((21,3),np.float32) if case['pose_seed']=='template' else np.array(read(OUT/'seeds/pose.json')['values'],np.float32)
 shape_seed=np.zeros(10,np.float32) if case['shape_seed']=='mean' else np.array(read(OUT/'seeds/shape.json')['values'],np.float32)
 basis_np,initialization=shape_basis(layer,model,points,cfg['shape_information_threshold']);beta0=basis_np@(basis_np.T@shape_seed);basis=t(basis_np);shape=t(basis_np.T@beta0).requires_grad_(True)
 g0,tr0=root_from_head(layer,body_seed,beta0,fixed,anchor);global0=t(g0);tr=t(tr0).requires_grad_(True);gd=torch.zeros(3,requires_grad=True)
 active=list(cfg['absolute_pose_caps_deg']);ids=torch.tensor([BODY_POSE_NAMES.index(n) for n in active]);caps=t(np.deg2rad([cfg['absolute_pose_caps_deg'][n] for n in active]));bodyvar=t(body_seed[ids.numpy()]).clone().requires_grad_(True)
 assert np.all(body_seed[[9,10]]==0) and bool((bodyvar.norm(dim=1)<=caps).all())
 metric={}
 for name in ['body','wrist','shoulder','hand_left','hand_right']:
  ps=[p for p in accepted if p['group']==name]
  if ps:metric[name]=dict(ids=torch.tensor([p['index'] for p in ps]),xyz=t([p['xyz'] for p in ps]),whitener=t([np.linalg.inv(np.linalg.cholesky(np.array(p['covariance']))) for p in ps]),multiplier=cfg['shoulder_weight'] if name=='shoulder' else cfg['wilor_weight_per_hand'] if name.startswith('hand_') else 1.)
 def get_beta():return (basis@shape).reshape(1,-1)
 def get_body():
  b=torch.zeros(21,3);b[ids]=bodyvar;return b
 def forward():
  b=get_body();local=layer(body_pose=b.reshape(1,-1),global_orient=global0.reshape(1,3),betas=get_beta(),**fixed);pivot=local.joints[0,0];R=axis_angle_rotation(gd);mesh=(local.vertices[0]-pivot)@R.T+pivot+tr;native=(local.joints[0]-pivot)@R.T+pivot+tr
  return mesh,native,native[list(model.joint_idx)],R@head_rotation(global0,b)
 def objective(values):
  mesh,native,joints,Rh=values;terms={};feasible=True
  for name,g in metric.items():
   residual=torch.einsum('nij,nj->ni',g['whitener'],joints[g['ids']]-g['xyz']);distance=residual.norm(dim=1)/np.sqrt(3)
   terms['metric/'+name]=cfg['metric_loss_weight']*g['multiplier']*F.huber_loss(distance,torch.zeros_like(distance),delta=cfg['metric_huber_delta'],reduction='mean')
  for side in ['left','right']:
   z=transform_torch(joints[target_ids],inputs['views'][side]['camera'])[:,2];terms[side+'/positive_joint_guard']=.01*F.relu((cfg['minimum_observed_joint_z_m']-z)/.01).square().mean()/2;feasible=feasible and bool((z>=cfg['minimum_observed_joint_z_m']).all())
  terms['head_position']=cfg['head_position_weight']*((native[15]-head_target)/cfg['head_position_sigma_m']).square().sum()
  terms['head_orientation']=cfg['head_rotation_weight']*(Rh-headR).square().sum()/(2*np.deg2rad(cfg['head_rotation_sigma_deg'])**2)
  terms['pose_prior']=cfg['pose_prior_weight']*(bodyvar/caps[:,None]).square().mean();terms['shape_prior']=cfg['shape_prior_weight']*get_beta().square().mean()
  return sum(terms.values()),feasible,{k:float(v.detach()) for k,v in terms.items()}
 def save(name,step):
  mesh,native,joints,Rh=forward();R=axis_angle_rotation(gd).detach().numpy()@cv2.Rodrigues(global0.numpy().astype(float))[0]
  p={k:v.detach().numpy().copy() for k,v in fixed.items()};p.update(betas=get_beta().detach().numpy(),body_pose=get_body().detach().numpy(),global_orient=cv2.Rodrigues(R.astype(float))[0].astype(np.float32).reshape(1,3),transl=tr.detach().numpy().reshape(1,3))
  mv,jj=fusion.geometry(p);err=float(np.max(np.abs(mv-mesh.detach().numpy())));assert err<1e-5,err
  projection={}
  for side in ['left','right']:
   camera=inputs['views'][side]['camera'];valid=transform_torch(joints[target_ids],camera)[:,2].detach().numpy()>cfg['minimum_observed_joint_z_m'];ii=np.array(target_ids)[valid]
   if len(ii):projection[side]=float(np.max(np.abs(project_numpy(jj[ii],camera)-project_torch(joints[ii],camera).detach().numpy())))
  assert all(x<.05 for x in projection.values()),projection
  p.update(coordinate_frame=np.array('cam3 CV'),smplx_joint_cam=jj,smplx_mesh_cam=mv,smplx_body_joints_3d_render_cam=jj[:25],smplx_body_joints_3d_local=jj[:25]-p['transl'],smplx_joint_cam_root_relative=jj-jj[:1]);np.savez_compressed(dest/(name+'.npz'),**p)
  loss,feasible,terms=objective((mesh,native,joints,Rh))
  return dict(step=step,selected_loss=float(loss),feasible=feasible,terms=terms,betas_final=get_beta().detach().reshape(-1).tolist(),pose_absolute_deg={n:float(x) for n,x in zip(active,np.rad2deg(bodyvar.detach().norm(dim=1).numpy()))},head_position_error_mm=float((native[15]-head_target).norm()*1000),head_orientation_error_deg=angle_deg(Rh.detach().numpy(),np.array(anchor['rotation'])),pelvis_left_m=joints[0].detach().tolist(),reload_mesh_max_m=err,reload_projection_max_px=projection,sha256=hashlib.sha256((dest/(name+'.npz')).read_bytes()).hexdigest())
 initial=save('initial',0);variables=[bodyvar,gd,tr,shape];lrs=[cfg['pose_lr'],cfg['rotation_lr'],cfg['translation_lr'],cfg['shape_lr']];opt=torch.optim.Adam([{'params':[v],'lr':lr} for v,lr in zip(variables,lrs)]);trace=[];budgets={};best=None;best_step=None;bestloss=float('inf');start=time.time()
 for step in range(cfg['steps']+1):
  opt.zero_grad();values=forward();loss,feasible,terms=objective(values)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
  if step%cfg['checkpoint_steps']==0 and feasible and float(loss.detach())<bestloss:best=[x.detach().clone() for x in variables];best_step=step;bestloss=float(loss.detach())
  if step%100==0 or step==cfg['steps']:
   trace.append(dict(step=step,loss=float(loss.detach()),feasible=feasible,terms=terms,betas=get_beta().detach().reshape(-1).tolist(),target_joints_m=values[2][target_ids].detach().tolist(),pelvis_left_m=values[2][0].detach().tolist()));print(a.case,step,float(loss.detach()),flush=True)
  if step in cfg['budget_steps']:
   if best is None:raise RuntimeError('No feasible checkpoint at budget '+str(step))
   current=[x.detach().clone() for x in variables]
   with torch.no_grad():
    for x,b in zip(variables,best):x.copy_(b)
    budgets[str(step)]=save('budget_'+str(step),best_step)
    for x,b in zip(variables,current):x.copy_(b)
   values=forward();loss,feasible,terms=objective(values)
  if step==cfg['steps']:break
  loss.backward();assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in variables);factor=next(q['factor'] for q in reversed(cfg['lr_schedule']) if step>=q['step'])
  for group,lr in zip(opt.param_groups,lrs):group['lr']=lr*factor
  opt.step()
  with torch.no_grad():bodyvar.mul_(torch.minimum(torch.ones_like(caps),caps/bodyvar.norm(dim=1).clamp_min(1e-8))[:,None])
 with torch.no_grad():
  for x,b in zip(variables,best):x.copy_(b)
  final=save('final',best_step)
 first,last=trace[-11],trace[-1];relative=abs(last['loss']-first['loss'])/max(abs(first['loss']),1e-12);motion=float(np.linalg.norm(np.array(last['target_joints_m'])-np.array(first['target_joints_m']),axis=1).max()*1000)
 q=dict(case=a.case,seed='left',case_definition=case,initial=initial,initialization=initialization,budgets=budgets,final=final,trace=trace,best_step=best_step,selected_loss=final['selected_loss'],terms=final['terms'],betas_final=final['betas_final'],betas_initial=beta0.tolist(),shape_seed_before_projection=shape_seed.tolist(),body_seed=body_seed.tolist(),pose_absolute_deg=final['pose_absolute_deg'],hard_caps_deg=cfg['absolute_pose_caps_deg'],prior_caps_deg=cfg['absolute_pose_caps_deg'],fixed_parameters={k:v.detach().reshape(-1).tolist() for k,v in fixed.items()},foot_local_pose='zero in all cases',layer_hand_convention=dict(use_pca=bool(getattr(layer,'use_pca',False)),flat_hand_mean=bool(getattr(layer,'flat_hand_mean',False))),read_files=sorted(set(reads)),allowed_seed_files=sorted(allowed),gt_inputs='Generating cameras and native Head position/orientation only; all fitting body/hand coordinates automatic.',elapsed_seconds=time.time()-start,convergence=dict(window_start_step=first['step'],end_step=last['step'],relative_loss_change=relative,max_target_joint_displacement_mm=motion,diagnostic_pass=relative<.001 and motion<.1),initial_global=g0.tolist(),initial_translation=tr0.tolist(),settings_sha256=hashlib.sha256((RUN/'settings.json').read_bytes()).hexdigest(),objective_scope='No network-centered pose/root/translation prior. Common absolute-pose caps and mean shape prior; no network hand/face/foot parameter.')
 (dest/'optimization.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print('DONE',a.case,best_step,flush=True)
if __name__=='__main__':main()
