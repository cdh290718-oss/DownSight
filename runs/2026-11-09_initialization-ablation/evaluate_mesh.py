"""Held-out posed-mesh and bone-frame evaluation to expose unobserved twist."""
import hashlib,json,os
import cv2
import numpy as np
from common import ROOT,RUN,OUT,OLD,read,angle_deg
from stages import stages
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.body_common import BODY_POSE_NAMES
paths=read(OLD/'configs/left.json')['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo']);layer=model.layer['neutral'];parents=layer.parents.detach().cpu().numpy();owner=layer.lbs_weights.detach().cpu().numpy().argmax(1);bodymask=(owner<20)&~np.isin(owner,[12,15]);D=OLD/'dataset/samples/000001'
with np.load(D/'left_vertices.npz') as z:gtmesh=z['vertices_camera_cv_m'];cameraR=z['world_to_camera_cv'][:3,:3]
with np.load(D/'geometry.npz') as z:objectR=z['object_matrix_world'][:3,:3]
with np.load(D/'smplx_reconstruction.npz') as z:gtbody=z['body_pose'].reshape(21,3);gtroot=cameraR@objectR@cv2.Rodrigues(z['global_orient'].reshape(3).astype(float))[0]
def frames(root,body):
 rotations=[root]
 for i in range(1,22):rotations.append(rotations[parents[i]]@cv2.Rodrigues(body[i-1].astype(float))[0])
 return rotations
truthframes=frames(gtroot,gtbody);anchor=read(OUT/'anchors/oracle.json');assert angle_deg(truthframes[15],np.array(anchor['rotation']))<.001
rows=[]
for stage in stages():
 path=stage['folder']/(stage['file']+'.npz')
 with np.load(path) as p:
  mesh=p['smplx_mesh_cam'];body=p['body_pose'].reshape(21,3);rotations=frames(cv2.Rodrigues(p['global_orient'].reshape(3).astype(float))[0],body)
  errors=np.linalg.norm(mesh-gtmesh,axis=1)*1000;local=[angle_deg(cv2.Rodrigues(x.astype(float))[0],cv2.Rodrigues(y.astype(float))[0]) for x,y in zip(body,gtbody)];glob=[angle_deg(x,y) for x,y in zip(rotations,truthframes)]
  rows.append(dict(stage=stage['label'],posed_mesh_absolute_mean_mm=float(errors.mean()),posed_mesh_absolute_p95_mm=float(np.percentile(errors,95)),body_surface_without_head_neck_hands_mean_mm=float(errors[bodymask].mean()),body_surface_vertex_count=int(bodymask.sum()),body21_local_rotation_mean_deg=float(np.mean(local)),native22_camera_rotation_mean_deg=float(np.mean(glob)),per_joint=[dict(name='Pelvis' if i==0 else BODY_POSE_NAMES[i-1],camera_rotation_error_deg=glob[i],local_rotation_error_deg=None if i==0 else local[i-1]) for i in range(22)],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
q=dict(exit_status=0,scope='Post-fit only. Absolute corresponding posed vertices, no rigid/scale alignment. Body-surface mask: dominant skinning joint index <20 excluding Neck12 and Head15. Camera-frame bone rotations include root and forward kinematics; errors can reveal twist invisible to sparse positions. No checkpoint selection by these scores.',truth_source_sha256={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['left_vertices.npz','geometry.npz','smplx_reconstruction.npz']},stages=rows)
(RUN/'evidence/mesh_metrics.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print('mesh and bone frames',len(rows))
