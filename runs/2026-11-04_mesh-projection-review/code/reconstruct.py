"""Rebuild final meshes from saved v0 and stage-two parameters; no fitting."""
import argparse,copy,json,os,sys
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--v0',type=Path,required=True);p.add_argument('--stage',type=Path,required=True);p.add_argument('--network',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(os.environ['EGOSMPLX_BACKEND']);sys.path.insert(0,os.environ['DOUBLE_SPHERE_SOURCE']);os.environ['EGOSMPLX_REPO_ROOT']=str(repo);os.environ['EGOSMPLX_CONFIG']=str(repo/'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py')
from egosmplx_pipeline.bootstrap import setup
smpl,_=setup(repo);layer=copy.deepcopy(smpl.layer['neutral']).cuda().eval();layer.requires_grad_(False);T=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');a.output.mkdir(parents=True,exist_ok=False);faces=np.asarray(smpl.face,dtype=np.int32);head=(smpl.layer['neutral'].lbs_weights.detach().cpu().numpy().argmax(1)==smpl.orig_joints_name.index('Head')).astype(np.float32);np.savez_compressed(a.output/'topology.npz',faces=faces,head_weight=head);rows=[]
entries=json.loads(a.selection.read_text())['selected'];batches=sorted(set(x['frame']//300*10 for x in entries))
for batch in batches:
 name=f'batch_{batch:03d}.npz'
 with np.load(a.v0/name,allow_pickle=False) as base,np.load(a.stage/name,allow_pickle=False) as new:
  assert np.array_equal(base['frame_ids'],new['frame_ids']);frames=base['frame_ids'];n=len(frames);raw=[]
  for frame in frames:
   with np.load(a.network/'smplx'/f'gen_{int(frame):05d}_00_smplx.npz',allow_pickle=False) as init:raw.append({k:init[k].copy() for k in ('betas','jaw_pose','leye_pose','reye_pose','expression')})
  fixed={k:T(np.stack([x[k].reshape(-1) for x in raw])) for k in raw[0]};kw=dict(global_orient=T(base['global_orient']),transl=T(base['transl']),**fixed)
  with torch.no_grad():
   out0=layer(body_pose=T(base['body_pose']).reshape(n,-1),left_hand_pose=T(base['left_hand_pose']),right_hand_pose=T(base['right_hand_pose']),**kw);out1=layer(body_pose=T(new['body_pose']).reshape(n,-1),left_hand_pose=T(new['left_hand_pose']),right_hand_pose=T(new['right_hand_pose']),**kw)
  vs0=out0.vertices.cpu().numpy();vs1=out1.vertices.cpu().numpy();js0=out0.joints[:,smpl.joint_idx].cpu().numpy();js1=out1.joints[:,smpl.joint_idx].cpu().numpy()
  for entry in entries:
   frame=entry['frame'];hits=np.flatnonzero(frames==frame)
   if not len(hits):continue
   i=int(hits[0]);d0=float(np.max(np.abs(vs0[i]-base['vertices_cam2'][i])));dj0=float(np.max(np.abs(js0[i]-base['joints_cam2'][i])));dj1=float(np.max(np.abs(js1[i]-new['joints_cam2'][i])));assert max(d0,dj0,dj1)<1e-5,(frame,d0,dj0,dj1)
   np.savez_compressed(a.output/f'{frame:05d}.npz',v0_vertices=vs0[i],stage_vertices=vs1[i]);rows.append({'frame':frame,'v0_vertex_max_abs_m':d0,'v0_joint_max_abs_m':dj0,'two_stage_joint_max_abs_m':dj1});print('frame',frame,'verified',d0,dj0,dj1,flush=True)
assert len(rows)==len(entries)
(a.output/'verification.json').write_text(json.dumps({'frames':rows,'head_vertex_count':int(head.sum()),'vertex_count':len(head),'triangle_count':len(faces),'model_parameters_updated':False},indent=2)+'\n')
