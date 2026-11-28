"""Audit head observations and arm changes for the published ten-frame result."""
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
from smplx.vertex_ids import vertex_ids

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
N=RUN.parent/'2026-11-28_video003_v5_nose'
OUT=ROOT/'outputs/video003_v5_nose_20261128'
STRICT=ROOT/'outputs/video003_strict_v5_20261127'
DEST=ROOT/'outputs/v003_elbow_nose_audit_20261128'
sys.path.insert(0,str(RUN.parent/'2026-11-27_video34_elbow_audit'))
import audit as a
from calibrate import rays
from render_result import load_obj
torch.set_num_threads(1)
cv2.setNumThreads(1)
ref=a.original.w.ref
ref.OLD=ref.OUT=ref.compare.OUT=a.OUT
ref.PRIOR=a.PRIOR
ref.cameras=ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=a.cameras
model,cam=ref.prev.ref.common.setup_model()
layer=model.layer['neutral'].eval()
faces=np.asarray(model.face,np.int32)
faceids=np.isin(layer.lbs_weights.argmax(1).numpy(),[15,22,23,24])[faces].all(1)
cams=list(map(a.read,a.cameras()))
rows={r['pair']:r for r in a.read(a.OUT/'fit/0/prepared.json')['items']}
pred=a.read(a.OUT/'predictions.json')
records={}
for index in range(1,11):
 key=f'v003_{index:02d}'
 row=rows[key]
 rec=a.read(N/'evidence'/(key+'.json'))
 new=dict(np.load(OUT/'poses'/(key+'.npz')))
 base=dict(np.load(STRICT/'poses'/(key+'_v5.npz')))
 assert a.sha(OUT/'export'/(key+'.obj'))==rec['obj_sha256']
 v,j=a.pack_pose(layer,new,new['locked'])
 obj,f=load_obj(OUT/'export'/(key+'.obj'))
 assert np.array_equal(f,faces) and np.max(abs(v-obj))<1e-8
 tf=[np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in ('left','right')]
 def proj(points):
  return np.stack([a.projection(points@t[:3,:3].T+t[:3,3],c) for t,c in zip(tf,cams)])
 scene=o3d.t.geometry.RaycastingScene(nthreads=1)
 scene.add_triangles(o3d.core.Tensor(v.astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
 pics=[]
 for e,side in enumerate(('left','right')):
  img=cv2.imread(str(a.OUT/'images'/(key+'_'+side+'.png')))
  x0=[710,0][e]
  xx,yy=np.meshgrid(np.arange(x0,x0+570),np.arange(400,720))
  directions=rays(np.column_stack((xx.ravel(),yy.ravel())),cams[e])@tf[e][:3,:3]
  directions/=np.linalg.norm(directions,axis=1,keepdims=True)
  origin=-tf[e][:3,:3].T@tf[e][:3,3]
  beam=np.column_stack((np.broadcast_to(origin,directions.shape),directions)).astype('float32')
  ids=scene.cast_rays(o3d.core.Tensor(beam),nthreads=1)['primitive_ids'].numpy()
  visible=ids<len(faces); visible[visible]=faceids[ids[visible]]
  edge=cv2.morphologyEx(visible.reshape(320,570).astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
  pic=img[400:720,x0:x0+570].copy()
  pic[edge.astype(bool)]=(60,175,255)
  for p in rec['edge_points'][e]:cv2.circle(pic,tuple(np.array(p)-[x0,400]),2,(220,235,0),-1)
  for p,color in [(rec['nose_uv'][e],(220,235,0)),(rec['projected_nose'][e],(60,175,255))]:
   cv2.drawMarker(pic,tuple(np.rint(np.array(p)-[x0,400]).astype(int)),color,cv2.MARKER_CROSS,16,2)
  pics.append(a.title(pic,key+' cam'+str(e+3)+'  cyan: detection / orange: projection'))
 assert cv2.imwrite(str(DEST/(key+'_face.jpg')),np.hstack(pics))
 if index not in (3,8):continue
 translated=base['locked'].copy()
 translated[22]+=np.array(rec['translation_mm'])/1000
 translated[[11,14]]=new['locked'][[11,14]]
 assert np.allclose(translated[:15],new['locked'][:15]) and np.allclose(translated[19:],new['locked'][19:])
 target=a.elbow_observations(row,['L','R'],pred,key)
 mask=((target>=0)&(target<[1280,720])).all(2)
 replay=a.replay(dict(base,locked=translated),target,mask,tf,model,cam,row,key,1)
 assert np.max(abs(replay-new['locked']))<1e-5
 no_wrist=a.replay(dict(base,locked=translated),target,mask,tf,model,cam,row,key,0)
 stages={'v5':base['locked'],'head_translation':translated,'elbow_only':no_wrist,'wrist_elbow':new['locked']}
 root=a.original.w.old.mat(base['locked'][21])@a.original.w.old.mat(base['global0'])
 basis=root@a.original.w.old.mat([0,np.pi/2,0])
 geometry={name:a.pack_pose(layer,base,p) for name,p in stages.items()}
 union=np.vstack([v@basis for v,j in geometry.values()])
 center=(union.min(0)+union.max(0))/2
 scale=min(316/np.ptp(union[:,0]),440/np.ptp(union[:,1]))
 images=[]; metrics={}
 for name,(vv,jj) in geometry.items():
  tile=a.free_render(vv,faces,basis,center,scale)
  for i in range(2):
   pts=np.rint(((jj[[16+i,18+i,20+i]]@basis-center)[:,:2])*[scale,-scale]+[175,240]).astype(int)
   cv2.polylines(tile,[pts],False,[(220,150,30),(40,120,220)][i],2,cv2.LINE_AA)
   cv2.circle(tile,tuple(pts[1]),5,[(220,150,30),(40,120,220)][i],-1)
  images.append(a.title(tile,name))
  uv=proj(jj[[18,19,20,21]])
  error=np.linalg.norm(uv-target,axis=2)
  info={}
  for i,side in enumerate(('left','right')):
   errors={}
   for idx,label in [(i,'elbow'),(i+2,'wrist')]:
    errors[label]=[float(error[e,idx]) if mask[e,idx] else None for e in range(2)]
   info[side]=dict(flexion_deg=a.angle(jj[18+i]-jj[16+i],jj[20+i]-jj[18+i]),errors_px=errors,
       joint_uv=proj(jj[[16+i,18+i,20+i]]).tolist())
  metrics[name]=info
 assert cv2.imwrite(str(RUN/'assets'/(key+'_stages.jpg')),np.hstack(images))
 records[key]=dict(stages=metrics,target=target.tolist(),target_valid=mask.tolist(),translation_mm=rec['translation_mm'],
                   source_obj_sha256=rec['obj_sha256'],replay_equal=True)
a.write(RUN/'evidence/results.json',dict(cases=records,exit_status=0))
